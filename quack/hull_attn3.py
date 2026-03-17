import math
from typing import Literal

import torch

from quack.flash_hull_attn import flash_hull_attn
from quack.hull_attn import (
    _hull_attn_full_fwd,
    _hull_attn_full_fwd_keymask,
    _hull_attn_full_fwd_masked,
    _hull_attn_full_bwd_dkdv,
    _hull_attn_full_bwd_dkdv_keymask,
    _hull_attn_full_bwd_dkdv_masked,
    _hull_attn_full_bwd_dq,
    _hull_attn_full_bwd_dq_keymask,
    _hull_attn_full_bwd_dq_masked,
)
from quack.hull_attn_codex import (
    DEFAULT_K_BLOCK,
    DEFAULT_Q_BLOCK,
    hull_attn_codex,
)


HullAttn3Backend = Literal[
    "flash",
    "cute_masked",
    "codex_unmasked",
    "codex_seq_lens",
    "codex_key_mask",
    "reference",
]


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not torch.is_floating_point(q):
        raise ValueError("q, k, and v must be floating-point tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shapes [batch, heads, seq, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q, k, and v must have the same number of heads")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same key sequence length")
    if q.shape[-1] != 2 or k.shape[-1] != 2 or v.shape[-1] != 2:
        raise ValueError("hull_attn3 is specialized to head_dim=2")


def _normalize_mask_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    batch = q.shape[0]
    seqlen_k = k.shape[-2]
    dense_mask = attention_mask

    if key_padding_mask is not None and seq_lens is not None:
        raise ValueError("Provide only one of key_padding_mask or seq_lens")

    if dense_mask is not None and dense_mask.ndim == 1 and not torch.is_floating_point(dense_mask):
        if dense_mask.shape[0] != batch:
            raise ValueError("attention_mask as seq_lens must have shape [batch]")
        if seq_lens is not None or key_padding_mask is not None:
            raise ValueError("attention_mask as seq_lens conflicts with explicit mask inputs")
        seq_lens = dense_mask
        dense_mask = None
    elif (
        dense_mask is not None
        and dense_mask.ndim == 2
        and dense_mask.dtype == torch.bool
        and dense_mask.shape == (batch, seqlen_k)
    ):
        if seq_lens is not None or key_padding_mask is not None:
            raise ValueError("attention_mask as key_padding_mask conflicts with explicit mask inputs")
        key_padding_mask = dense_mask
        dense_mask = None

    if dense_mask is not None and key_padding_mask is not None:
        raise ValueError("Dense attention_mask and key_padding_mask are mutually exclusive")
    if dense_mask is not None and seq_lens is not None:
        raise ValueError("Dense attention_mask and seq_lens are mutually exclusive")

    if seq_lens is not None:
        if seq_lens.ndim != 1 or seq_lens.shape[0] != batch:
            raise ValueError("seq_lens must have shape [batch]")
        seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()

    if key_padding_mask is not None:
        if key_padding_mask.ndim != 2 or key_padding_mask.shape != (batch, seqlen_k):
            raise ValueError("key_padding_mask must have shape [batch, seqlen_k]")
        key_padding_mask = key_padding_mask.to(device=q.device, dtype=torch.bool).contiguous()

    return dense_mask, key_padding_mask, seq_lens


def _apply_attention_mask(scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return scores
    mask = attention_mask.to(device=scores.device)
    if mask.dtype == torch.bool:
        return scores.masked_fill(~mask, -torch.inf)
    return scores + mask.to(dtype=scores.dtype)


def _apply_key_padding_mask(
    scores: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    if key_padding_mask is None:
        return scores
    return scores.masked_fill(~key_padding_mask[:, None, None, :], -torch.inf)


def _canonicalize_dense_mask(
    attention_mask: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]
    mask = attention_mask.to(device=q.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        return None
    if mask.shape[0] not in (1, batch):
        return None
    if mask.shape[1] not in (1, heads):
        return None
    if mask.shape[2] not in (1, seqlen_q):
        return None
    if mask.shape[3] not in (1, seqlen_k):
        return None
    if mask.dtype == torch.bool:
        mask = torch.where(
            mask,
            torch.zeros((), device=q.device, dtype=torch.float32),
            torch.full((), -torch.inf, device=q.device, dtype=torch.float32),
        )
    else:
        mask = mask.to(dtype=torch.float32)

    if all(stride == 0 or stride % 4 == 0 for stride in mask.stride()[:-1]):
        return mask

    padded_k = ((mask.shape[-1] + 3) // 4) * 4
    if padded_k != mask.shape[-1]:
        mask_padded = torch.empty((*mask.shape[:-1], padded_k), device=q.device, dtype=torch.float32)
        mask_padded[..., : mask.shape[-1]] = mask
        mask_padded[..., mask.shape[-1] :] = 0.0
        return mask_padded[..., : mask.shape[-1]]
    return mask.contiguous()


def _mask_chunk(
    attention_mask: torch.Tensor | None,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    return attention_mask[..., q_start:q_end, k_start:k_end]


def _key_mask_chunk(
    key_padding_mask: torch.Tensor | None,
    k_start: int,
    k_end: int,
) -> torch.Tensor | None:
    if key_padding_mask is None:
        return None
    return key_padding_mask[:, k_start:k_end]


def _safe_reference_full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = _apply_attention_mask(scores, attention_mask)
    scores = _apply_key_padding_mask(scores, key_padding_mask)

    valid_rows = torch.isfinite(scores).any(dim=-1)
    safe_scores = torch.where(valid_rows.unsqueeze(-1), scores, torch.zeros_like(scores))
    probs = torch.softmax(safe_scores, dim=-1)
    probs = torch.where(valid_rows.unsqueeze(-1), probs, torch.zeros_like(probs))
    out = torch.matmul(probs, v.float()).to(dtype=q.dtype)

    lse = torch.logsumexp(safe_scores, dim=-1)
    lse = torch.where(valid_rows, lse, torch.full_like(lse, -torch.inf))
    return out, lse


def _prefix_key_padding_to_seq_lens(key_padding_mask: torch.Tensor | None) -> torch.Tensor | None:
    if key_padding_mask is None:
        return None
    if key_padding_mask.shape[-1] <= 1:
        return key_padding_mask.to(dtype=torch.int32).sum(dim=-1)
    mask_i32 = key_padding_mask.to(dtype=torch.int32)
    if torch.all(mask_i32[:, 1:] <= mask_i32[:, :-1]):
        return mask_i32.sum(dim=-1).to(dtype=torch.int32)
    return None


def _key_padding_from_seq_lens(seq_lens: torch.Tensor, seqlen_k: int) -> torch.Tensor:
    positions = torch.arange(seqlen_k, device=seq_lens.device, dtype=seq_lens.dtype)
    return positions.unsqueeze(0) < seq_lens.unsqueeze(1)


def _row_valid_mask(
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    canonical_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, heads, seqlen_q, _ = q.shape
    row_valid = torch.ones((batch, heads, seqlen_q), device=q.device, dtype=torch.bool)

    if canonical_mask is None and attention_mask is not None:
        canonical_mask = _canonicalize_dense_mask(attention_mask, q, k)
    if canonical_mask is not None:
        dense_row_valid = torch.isfinite(canonical_mask).any(dim=-1)
        if dense_row_valid.shape[0] == 1:
            dense_row_valid = dense_row_valid.expand(batch, dense_row_valid.shape[1], dense_row_valid.shape[2])
        if dense_row_valid.shape[1] == 1:
            dense_row_valid = dense_row_valid.expand(dense_row_valid.shape[0], heads, dense_row_valid.shape[2])
        if dense_row_valid.shape[2] == 1:
            dense_row_valid = dense_row_valid.expand(dense_row_valid.shape[0], dense_row_valid.shape[1], seqlen_q)
        row_valid &= dense_row_valid

    if key_padding_mask is not None:
        key_valid = key_padding_mask.any(dim=-1)[:, None, None].expand(batch, heads, seqlen_q)
        row_valid &= key_valid

    return row_valid


def _dispatch_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
    q_block: int,
    k_block: int,
    prefer_flash: bool,
    need_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, HullAttn3Backend]:
    can_use_flash = (
        prefer_flash
        and not need_lse
        and attention_mask is None
        and key_padding_mask is None
        and seq_lens is None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-2] == k.shape[-2]
    )
    if can_use_flash:
        return flash_hull_attn(q, k, v, scale=scale), None, "flash"

    can_use_codex = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and attention_mask is None
    if can_use_codex:
        if seq_lens is not None:
            result = hull_attn_codex(
                q,
                k,
                v,
                scale=scale,
                seq_lens=seq_lens,
                q_block=q_block,
                k_block=k_block,
                return_lse=need_lse,
            )
            out, lse = result if need_lse else (result, None)
            return out, lse, "codex_seq_lens"

        prefix_seq_lens = _prefix_key_padding_to_seq_lens(key_padding_mask)
        if prefix_seq_lens is not None:
            result = hull_attn_codex(
                q,
                k,
                v,
                scale=scale,
                seq_lens=prefix_seq_lens,
                q_block=q_block,
                k_block=k_block,
                return_lse=need_lse,
            )
            out, lse = result if need_lse else (result, None)
            return out, lse, "codex_seq_lens"

        if key_padding_mask is not None:
            result = hull_attn_codex(
                q,
                k,
                v,
                scale=scale,
                key_padding_mask=key_padding_mask,
                q_block=q_block,
                k_block=k_block,
                return_lse=need_lse,
            )
            out, lse = result if need_lse else (result, None)
            return out, lse, "codex_key_mask"

        result = hull_attn_codex(
            q,
            k,
            v,
            scale=scale,
            q_block=q_block,
            k_block=k_block,
            return_lse=need_lse,
        )
        out, lse = result if need_lse else (result, None)
        return out, lse, "codex_unmasked"

    fast_forward = _fast_custom_op_forward(
        q,
        k,
        v,
        scale=scale,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
        need_lse=need_lse,
    )
    if fast_forward is not None:
        out, lse = fast_forward
        return out, lse, "cute_masked"

    out, lse = _safe_reference_full_attention(
        q,
        k,
        v,
        scale=scale,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
    )
    return out, lse, "reference"


def _fast_custom_op_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    need_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return None

    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]
    canonical_mask = _canonicalize_dense_mask(attention_mask, q, k)
    key_mask = None if key_padding_mask is None else key_padding_mask.to(device=q.device, dtype=torch.int32).contiguous()
    if canonical_mask is None and key_mask is None and attention_mask is not None:
        return None
    if canonical_mask is not None and key_mask is not None:
        return None

    q_contig = q.contiguous()
    k_contig = k.contiguous()
    v_contig = v.contiguous()
    q_flat = q_contig.view(batch * heads, seqlen_q, 2)
    k_flat = k_contig.view(batch * heads, seqlen_k, 2)
    v_flat = v_contig.view(batch * heads, seqlen_k, 2)
    out_flat = torch.empty_like(q_flat)
    lse_flat = torch.empty((batch * heads, seqlen_q), device=q.device, dtype=torch.float32)

    if key_mask is not None:
        _hull_attn_full_fwd_keymask(q_flat, k_flat, v_flat, key_mask, heads, scale, out_flat, lse_flat)
    elif canonical_mask is not None:
        _hull_attn_full_fwd_masked(q_flat, k_flat, v_flat, canonical_mask, heads, scale, out_flat, lse_flat)
    else:
        _hull_attn_full_fwd(q_flat, k_flat, v_flat, scale, out_flat, lse_flat)

    out = out_flat.view_as(q_contig)
    lse = lse_flat.view(batch, heads, seqlen_q)
    if attention_mask is not None or key_padding_mask is not None:
        row_valid = _row_valid_mask(
            attention_mask,
            key_padding_mask,
            q_contig,
            k_contig,
            canonical_mask=canonical_mask,
        )
        if not row_valid.all():
            out = torch.where(row_valid.unsqueeze(-1), out, torch.zeros_like(out))
            lse = torch.where(row_valid, lse, torch.full_like(lse, -torch.inf))
    return out, lse if need_lse else None


def _fast_custom_op_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return None

    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]

    canonical_mask = _canonicalize_dense_mask(attention_mask, q, k)
    key_mask = None if key_padding_mask is None else key_padding_mask.to(device=q.device, dtype=torch.int32).contiguous()
    if canonical_mask is not None and key_mask is not None:
        return None

    q_contig = q.contiguous()
    k_contig = k.contiguous()
    v_contig = v.contiguous()
    out_contig = out.contiguous()
    grad_out_contig = grad_out.contiguous()
    lse_contig = lse.contiguous()

    if attention_mask is not None or key_padding_mask is not None:
        row_valid = _row_valid_mask(
            attention_mask,
            key_padding_mask,
            q_contig,
            k_contig,
            canonical_mask=canonical_mask,
        )
        if not row_valid.all():
            out_contig = torch.where(row_valid.unsqueeze(-1), out_contig, torch.zeros_like(out_contig))
            lse_contig = torch.where(row_valid, lse_contig, torch.zeros_like(lse_contig))

    flat_shape_q = (batch * heads, seqlen_q, 2)
    flat_shape_k = (batch * heads, seqlen_k, 2)
    q_flat = q_contig.view(flat_shape_q)
    k_flat = k_contig.view(flat_shape_k)
    v_flat = v_contig.view(flat_shape_k)
    out_flat = out_contig.view(flat_shape_q)
    lse_flat = lse_contig.view(batch * heads, seqlen_q)
    grad_out_flat = grad_out_contig.view(flat_shape_q)
    dq_flat = torch.empty_like(q_flat)
    dk_flat = torch.empty_like(k_flat)
    dv_flat = torch.empty_like(v_flat)

    if key_mask is not None:
        _hull_attn_full_bwd_dq_keymask(
            q_flat,
            k_flat,
            v_flat,
            key_mask,
            heads,
            out_flat,
            lse_flat,
            grad_out_flat,
            scale,
            dq_flat,
        )
        _hull_attn_full_bwd_dkdv_keymask(
            q_flat,
            k_flat,
            v_flat,
            key_mask,
            heads,
            out_flat,
            lse_flat,
            grad_out_flat,
            scale,
            dk_flat,
            dv_flat,
        )
        return dq_flat.view_as(q_contig), dk_flat.view_as(k_contig), dv_flat.view_as(v_contig)

    if canonical_mask is not None:
        _hull_attn_full_bwd_dq_masked(
            q_flat,
            k_flat,
            v_flat,
            canonical_mask,
            heads,
            out_flat,
            lse_flat,
            grad_out_flat,
            scale,
            dq_flat,
        )
        _hull_attn_full_bwd_dkdv_masked(
            q_flat,
            k_flat,
            v_flat,
            canonical_mask,
            heads,
            out_flat,
            lse_flat,
            grad_out_flat,
            scale,
            dk_flat,
            dv_flat,
        )
        return dq_flat.view_as(q_contig), dk_flat.view_as(k_contig), dv_flat.view_as(v_contig)

    _hull_attn_full_bwd_dq(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        lse_flat,
        grad_out_flat,
        scale,
        dq_flat,
    )
    _hull_attn_full_bwd_dkdv(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        lse_flat,
        grad_out_flat,
        scale,
        dk_flat,
        dv_flat,
    )
    return dq_flat.view_as(q_contig), dk_flat.view_as(k_contig), dv_flat.view_as(v_contig)


def _blockwise_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    q_block: int,
    k_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]
    q_block = min(q_block, seqlen_q)
    k_block = min(k_block, seqlen_k)

    grad_out_f32 = grad_out.float()
    v_f32 = v.float()
    q_f32 = q.float()
    k_f32 = k.float()
    out_f32 = out.float()

    dq = torch.zeros_like(q_f32)
    dk = torch.zeros_like(k_f32)
    dv = torch.zeros_like(v_f32)

    for q_start in range(0, seqlen_q, q_block):
        q_end = min(q_start + q_block, seqlen_q)
        q_chunk = q_f32[:, :, q_start:q_end, :]
        grad_chunk = grad_out_f32[:, :, q_start:q_end, :]
        out_chunk = out_f32[:, :, q_start:q_end, :]
        lse_chunk = lse[:, :, q_start:q_end]
        valid_rows = torch.isfinite(lse_chunk)
        safe_lse = torch.where(valid_rows, lse_chunk, torch.zeros_like(lse_chunk))
        delta = (grad_chunk * out_chunk).sum(dim=-1, keepdim=True)
        dq_chunk = torch.zeros_like(q_chunk)

        for k_start in range(0, seqlen_k, k_block):
            k_end = min(k_start + k_block, seqlen_k)
            k_chunk = k_f32[:, :, k_start:k_end, :]
            v_chunk = v_f32[:, :, k_start:k_end, :]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scale
            scores = _apply_attention_mask(scores, _mask_chunk(attention_mask, q_start, q_end, k_start, k_end))
            scores = _apply_key_padding_mask(scores, _key_mask_chunk(key_padding_mask, k_start, k_end))

            probs = torch.exp(scores - safe_lse.unsqueeze(-1))
            probs = torch.where(valid_rows.unsqueeze(-1), probs, torch.zeros_like(probs))

            dv[:, :, k_start:k_end, :] += torch.matmul(probs.transpose(-1, -2), grad_chunk)
            dp = torch.matmul(grad_chunk, v_chunk.transpose(-1, -2))
            ds = probs * (dp - delta)
            dq_chunk += torch.matmul(ds, k_chunk)
            dk[:, :, k_start:k_end, :] += torch.matmul(ds.transpose(-1, -2), q_chunk)

        dq[:, :, q_start:q_end, :] = dq_chunk * scale

    dk *= scale
    return dq.to(dtype=q.dtype), dk.to(dtype=k.dtype), dv.to(dtype=v.dtype)


def _pack_output(
    out: torch.Tensor,
    lse: torch.Tensor | None,
    backend: HullAttn3Backend,
    return_lse: bool,
    return_backend: bool,
):
    if return_lse and return_backend:
        return out, lse, backend
    if return_lse:
        return out, lse
    if return_backend:
        return out, backend
    return out


class _HullAttn3Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        scale,
        attention_mask,
        key_padding_mask,
        seq_lens,
        q_block,
        k_block,
    ):
        out, lse, _ = _dispatch_forward(
            q,
            k,
            v,
            scale=scale,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            seq_lens=seq_lens,
            q_block=q_block,
            k_block=k_block,
            prefer_flash=False,
            need_lse=True,
        )
        key_padding_mask_for_bwd = (
            _key_padding_from_seq_lens(seq_lens, k.shape[-2]) if seq_lens is not None else key_padding_mask
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.attention_mask = attention_mask
        ctx.key_padding_mask = key_padding_mask_for_bwd
        ctx.q_block = q_block
        ctx.k_block = k_block
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        grads = _fast_custom_op_backward(
            q,
            k,
            v,
            out,
            lse,
            grad_out,
            scale=ctx.scale,
            attention_mask=ctx.attention_mask,
            key_padding_mask=ctx.key_padding_mask,
        )
        if grads is None:
            grads = _blockwise_backward(
                q,
                k,
                v,
                out,
                lse,
                grad_out,
                scale=ctx.scale,
                attention_mask=ctx.attention_mask,
                key_padding_mask=ctx.key_padding_mask,
                q_block=ctx.q_block,
                k_block=ctx.k_block,
            )
        dq, dk, dv = grads
        return dq, dk, dv, None, None, None, None, None, None, None


def hull_attn3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attention_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    q_block: int = DEFAULT_Q_BLOCK,
    k_block: int = DEFAULT_K_BLOCK,
    prefer_flash: bool = True,
    return_lse: bool = False,
    return_backend: bool = False,
):
    """Experimental hull attention path with exact forward/backward.

    Dispatch order:
    1. FA4 padded path for inference-only unmasked fp16/bf16 inputs with matching q/k lengths.
    2. Codex multi-row kernel for trainable unmasked and length-masked fp16/bf16 inputs.
    3. Exact PyTorch fallback for dense masks and other unsupported combinations.
    """
    _check_inputs(q, k, v)
    if q_block <= 0 or k_block <= 0:
        raise ValueError("q_block and k_block must be positive")
    if attention_mask is not None and attention_mask.requires_grad:
        raise ValueError("attention_mask gradients are not supported")
    if key_padding_mask is not None and key_padding_mask.requires_grad:
        raise ValueError("key_padding_mask gradients are not supported")
    if seq_lens is not None and seq_lens.requires_grad:
        raise ValueError("seq_lens gradients are not supported")

    attention_mask, key_padding_mask, seq_lens = _normalize_mask_inputs(
        q, k, attention_mask, key_padding_mask, seq_lens
    )
    scale = (1.0 / math.sqrt(2.0)) if scale is None else float(scale)

    needs_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    if needs_grad:
        if return_lse or return_backend:
            raise NotImplementedError("return_lse and return_backend are only supported in no-grad mode")
        return _HullAttn3Function.apply(
            q, k, v, scale, attention_mask, key_padding_mask, seq_lens, q_block, k_block
        )

    out, lse, backend = _dispatch_forward(
        q,
        k,
        v,
        scale=scale,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
        seq_lens=seq_lens,
        q_block=q_block,
        k_block=k_block,
        prefer_flash=prefer_flash,
        need_lse=return_lse,
    )
    return _pack_output(out, lse, backend, return_lse, return_backend)


__all__ = ["hull_attn3"]
