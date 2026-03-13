import math
import operator
from functools import lru_cache
from typing import Literal, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import torch

from quack.cache_utils import compile_and_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map


HullAttnMode = Literal["full", "topk1", "topk4"]
DEFAULT_Q_BLOCK = 128
DEFAULT_K_BLOCK = 256
CUTE_K_BLOCK = 128


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mode: HullAttnMode,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
):
    if mode not in ("full", "topk1", "topk4"):
        raise ValueError(f"Unsupported hull_attn mode: {mode}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if not q.is_cuda:
        raise ValueError("hull_attn currently expects CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, seq, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same shape")
    if q.shape[-1] != 2:
        raise ValueError("hull_attn is specialized to head_dim=2")
    if attention_mask is not None and attention_mask.requires_grad:
        raise ValueError("attention_mask gradients are not supported")
    if key_padding_mask is not None and key_padding_mask.requires_grad:
        raise ValueError("key_padding_mask gradients are not supported")
    if seq_lens is not None and seq_lens.requires_grad:
        raise ValueError("seq_lens gradients are not supported")


def _normalize_mask_inputs(
    q: torch.Tensor,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if key_padding_mask is not None and seq_lens is not None:
        raise ValueError("Provide only one of key_padding_mask or seq_lens")

    dense_mask = attention_mask
    if dense_mask is not None and dense_mask.ndim == 1 and torch.is_floating_point(dense_mask) is False:
        if seq_lens is not None or key_padding_mask is not None:
            raise ValueError("attention_mask as seq_lens conflicts with explicit mask inputs")
        seq_lens = dense_mask
        dense_mask = None
    elif dense_mask is not None and dense_mask.ndim == 2 and dense_mask.dtype == torch.bool:
        if seq_lens is not None or key_padding_mask is not None:
            raise ValueError("attention_mask as key_padding_mask conflicts with explicit mask inputs")
        key_padding_mask = dense_mask
        dense_mask = None

    if dense_mask is not None and key_padding_mask is not None:
        raise ValueError("Dense attention_mask and key_padding_mask are mutually exclusive")
    if dense_mask is not None and seq_lens is not None:
        raise ValueError("Dense attention_mask and seq_lens are mutually exclusive")

    if seq_lens is not None:
        if seq_lens.ndim != 1 or seq_lens.shape[0] != q.shape[0]:
            raise ValueError("seq_lens must have shape [batch]")
        seq_lens = seq_lens.to(device=q.device, dtype=torch.int64)
        positions = torch.arange(q.shape[-2], device=q.device, dtype=torch.int64)
        key_padding_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)

    if key_padding_mask is not None:
        if key_padding_mask.ndim != 2 or key_padding_mask.shape != (q.shape[0], q.shape[-2]):
            raise ValueError("key_padding_mask must have shape [batch, seq]")
        if key_padding_mask.dtype != torch.bool:
            key_padding_mask = key_padding_mask.to(dtype=torch.bool)
        key_padding_mask = key_padding_mask.to(device=q.device)

    return dense_mask, key_padding_mask


def _apply_attention_mask(scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return scores
    if attention_mask.dtype == torch.bool:
        return scores.masked_fill(~attention_mask.to(device=scores.device), -torch.inf)
    return scores + attention_mask.to(device=scores.device, dtype=scores.dtype)


def _apply_key_padding_mask(
    scores: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    if key_padding_mask is None:
        return scores
    return scores.masked_fill(~key_padding_mask.to(device=scores.device), -torch.inf)


def _gather_selected_v(v: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    q_len = indices.shape[-2]
    expanded_v = v.unsqueeze(-3).expand(*v.shape[:-2], q_len, v.shape[-2], v.shape[-1])
    gather_idx = indices.unsqueeze(-1).expand(*indices.shape, v.shape[-1])
    return torch.gather(expanded_v, dim=-2, index=gather_idx)


def _mask_chunk(
    attention_mask: torch.Tensor | None,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
):
    if attention_mask is None:
        return None
    return attention_mask[..., q_start:q_end, k_start:k_end]


def _key_mask_chunk(
    key_padding_mask: torch.Tensor | None,
    k_start: int,
    k_end: int,
):
    if key_padding_mask is None:
        return None
    return key_padding_mask[:, None, None, k_start:k_end]


def _streaming_full_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
    out = torch.empty_like(q_f32)
    lse = torch.empty((batch, heads, seqlen_q), device=q.device, dtype=torch.float32)

    q_block = min(DEFAULT_Q_BLOCK, seqlen_q)
    k_block = min(DEFAULT_K_BLOCK, seqlen_k)

    for q_start in range(0, seqlen_q, q_block):
        q_end = min(q_start + q_block, seqlen_q)
        q_chunk = q_f32[:, :, q_start:q_end, :]
        m = torch.full(
            q_chunk.shape[:-1],
            -torch.inf,
            device=q.device,
            dtype=torch.float32,
        )
        l = torch.zeros_like(m)
        acc = torch.zeros_like(q_chunk)

        for k_start in range(0, seqlen_k, k_block):
            k_end = min(k_start + k_block, seqlen_k)
            k_chunk = k_f32[:, :, k_start:k_end, :]
            v_chunk = v_f32[:, :, k_start:k_end, :]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scale
            mask_chunk = _mask_chunk(attention_mask, q_start, q_end, k_start, k_end)
            scores = _apply_attention_mask(scores, mask_chunk)
            scores = _apply_key_padding_mask(scores, _key_mask_chunk(key_padding_mask, k_start, k_end))
            block_max = scores.max(dim=-1).values
            m_new = torch.maximum(m, block_max)
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new.unsqueeze(-1))
            l = l * alpha + p.sum(dim=-1)
            acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_chunk)
            m = m_new

        out[:, :, q_start:q_end, :] = acc / l.unsqueeze(-1)
        lse[:, :, q_start:q_end] = m + torch.log(l)

    return out.to(dtype=q.dtype), lse


def _reference_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk: int,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    scores = _apply_attention_mask(scores, attention_mask)
    scores = _apply_key_padding_mask(scores, key_padding_mask[:, None, None, :] if key_padding_mask is not None else None)
    k_select = min(topk, scores.shape[-1])
    topk_scores, topk_indices = torch.topk(scores, k=k_select, dim=-1, largest=True, sorted=True)
    probs = torch.softmax(topk_scores, dim=-1)
    selected_v = _gather_selected_v(v.float(), topk_indices)
    return (probs.unsqueeze(-1) * selected_v).sum(dim=-2).to(dtype=q.dtype)


@cute.jit
def _better_score(score0: Float32, idx0: Int32, score1: Float32, idx1: Int32) -> bool:
    return (score1 > score0) or ((score1 == score0) and (idx1 >= 0) and ((idx0 < 0) or (idx1 < idx0)))


class HullAttnSparseForwardCute:
    def __init__(
        self,
        dtype,
        topk: int,
        seqlen_q: int,
        seqlen_k: int,
        mask_batch_broadcast: bool = True,
        mask_head_broadcast: bool = True,
        mask_q_broadcast: bool = False,
        mask_k_broadcast: bool = False,
        k_block: int = CUTE_K_BLOCK,
    ):
        self.dtype = dtype
        self.topk = topk
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.mask_batch_broadcast = mask_batch_broadcast
        self.mask_head_broadcast = mask_head_broadcast
        self.mask_q_broadcast = mask_q_broadcast
        self.mask_k_broadcast = mask_k_broadcast
        self.k_block = k_block

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMask: Optional[cute.Tensor],
        mKeyMask: Optional[cute.Tensor],
        num_heads: Int32,
        mO: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(mQ, mK, mV, mMask, mKeyMask, num_heads, mO, scale).launch(
            grid=[self.seqlen_q, mQ.shape[0], 1],
            block=[cute.arch.WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMask: Optional[cute.Tensor],
        mKeyMask: Optional[cute.Tensor],
        num_heads: Int32,
        mO: cute.Tensor,
        scale: Float32,
    ):
        lane_idx = cute.arch.lane_idx()
        q_idx, bh_idx, _ = cute.arch.block_idx()

        if q_idx < self.seqlen_q and bh_idx < mQ.shape[0]:
            batch_idx = bh_idx // num_heads
            head_idx = bh_idx - batch_idx * num_heads
            q0 = Float32(mQ[bh_idx, q_idx, 0])
            q1 = Float32(mQ[bh_idx, q_idx, 1])
            top_scores = cute.make_rmem_tensor(self.topk, Float32)
            top_indices = cute.make_rmem_tensor(self.topk, Int32)
            for i in cutlass.range_constexpr(self.topk):
                top_scores[i] = -Float32.inf
                top_indices[i] = Int32(-1)

            num_k_tiles = cute.ceil_div(self.seqlen_k, self.k_block)
            elems_per_lane = const_expr(cute.ceil_div(self.k_block, cute.arch.WARP_SIZE))
            for select_idx in cutlass.range_constexpr(self.topk):
                best_score = -Float32.inf
                best_idx = Int32(-1)
                for k_tile in cutlass.range(num_k_tiles, unroll=1):
                    for i in cutlass.range_constexpr(elems_per_lane):
                        k_idx = k_tile * self.k_block + lane_idx + i * cute.arch.WARP_SIZE
                        if k_idx < self.seqlen_k:
                            is_selected = False
                            for prev_idx in cutlass.range_constexpr(self.topk):
                                if const_expr(prev_idx < select_idx):
                                    is_selected = is_selected or (top_indices[prev_idx] == k_idx)
                            if not is_selected:
                                score = (q0 * Float32(mK[bh_idx, k_idx, 0]) + q1 * Float32(mK[bh_idx, k_idx, 1])) * scale
                                if const_expr(mMask is not None):
                                    mask_batch_idx = Int32.zero if const_expr(self.mask_batch_broadcast) else batch_idx
                                    mask_head_idx = Int32.zero if const_expr(self.mask_head_broadcast) else head_idx
                                    mask_q_idx = Int32.zero if const_expr(self.mask_q_broadcast) else q_idx
                                    mask_k_idx = Int32.zero if const_expr(self.mask_k_broadcast) else k_idx
                                    score += Float32(mMask[mask_batch_idx, mask_head_idx, mask_q_idx, mask_k_idx])
                                if const_expr(mKeyMask is not None):
                                    score = score if mKeyMask[batch_idx, k_idx] != 0 else -Float32.inf
                                if _better_score(best_score, best_idx, score, k_idx):
                                    best_score = score
                                    best_idx = k_idx

                for xor_mask in cutlass.range_constexpr(5):
                    other_score = cute.arch.shuffle_sync_bfly(best_score, offset=1 << xor_mask)
                    other_idx = cute.arch.shuffle_sync_bfly(best_idx, offset=1 << xor_mask)
                    if _better_score(best_score, best_idx, other_score, other_idx):
                        best_score = other_score
                        best_idx = other_idx

                top_scores[select_idx] = best_score
                top_indices[select_idx] = best_idx

            if lane_idx == 0:
                if const_expr(self.topk == 1):
                    if top_indices[0] >= 0:
                        mO[bh_idx, q_idx, 0] = mO.element_type(mV[bh_idx, top_indices[0], 0])
                        mO[bh_idx, q_idx, 1] = mO.element_type(mV[bh_idx, top_indices[0], 1])
                    else:
                        mO[bh_idx, q_idx, 0] = mO.element_type(Float32.nan)
                        mO[bh_idx, q_idx, 1] = mO.element_type(Float32.nan)
                else:
                    max_score = top_scores[0]
                    denom = Float32.zero
                    acc0 = Float32.zero
                    acc1 = Float32.zero
                    for i in cutlass.range_constexpr(self.topk):
                        idx = top_indices[i]
                        if idx >= 0:
                            prob = cute.math.exp(top_scores[i] - max_score, fastmath=True)
                            denom += prob
                            acc0 += prob * Float32(mV[bh_idx, idx, 0])
                            acc1 += prob * Float32(mV[bh_idx, idx, 1])
                    inv_denom = cute.arch.rcp_approx(denom)
                    mO[bh_idx, q_idx, 0] = mO.element_type(acc0 * inv_denom)
                    mO[bh_idx, q_idx, 1] = mO.element_type(acc1 * inv_denom)


class HullAttnFullForwardCute:
    def __init__(
        self,
        dtype,
        seqlen_q: int,
        seqlen_k: int,
        mask_batch_broadcast: bool = True,
        mask_head_broadcast: bool = True,
        mask_q_broadcast: bool = False,
        mask_k_broadcast: bool = False,
        k_block: int = CUTE_K_BLOCK,
    ):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.mask_batch_broadcast = mask_batch_broadcast
        self.mask_head_broadcast = mask_head_broadcast
        self.mask_q_broadcast = mask_q_broadcast
        self.mask_k_broadcast = mask_k_broadcast
        self.k_block = k_block

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMask: Optional[cute.Tensor],
        mKeyMask: Optional[cute.Tensor],
        num_heads: Int32,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(mQ, mK, mV, mMask, mKeyMask, num_heads, mO, mLSE, scale).launch(
            grid=[self.seqlen_q, mQ.shape[0], 1],
            block=[cute.arch.WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMask: Optional[cute.Tensor],
        mKeyMask: Optional[cute.Tensor],
        num_heads: Int32,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        scale: Float32,
    ):
        lane_idx = cute.arch.lane_idx()
        q_idx, bh_idx, _ = cute.arch.block_idx()

        if q_idx < self.seqlen_q and bh_idx < mQ.shape[0]:
            batch_idx = bh_idx // num_heads
            head_idx = bh_idx - batch_idx * num_heads
            q0 = Float32(mQ[bh_idx, q_idx, 0])
            q1 = Float32(mQ[bh_idx, q_idx, 1])
            max_score = -Float32.inf
            num_k_tiles = cute.ceil_div(self.seqlen_k, self.k_block)
            elems_per_lane = const_expr(cute.ceil_div(self.k_block, cute.arch.WARP_SIZE))

            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                local_max = -Float32.inf
                for i in cutlass.range_constexpr(elems_per_lane):
                    k_idx = k_tile * self.k_block + lane_idx + i * cute.arch.WARP_SIZE
                    if k_idx < self.seqlen_k:
                        score = (q0 * Float32(mK[bh_idx, k_idx, 0]) + q1 * Float32(mK[bh_idx, k_idx, 1])) * scale
                        if const_expr(mMask is not None):
                            mask_batch_idx = Int32.zero if const_expr(self.mask_batch_broadcast) else batch_idx
                            mask_head_idx = Int32.zero if const_expr(self.mask_head_broadcast) else head_idx
                            mask_q_idx = Int32.zero if const_expr(self.mask_q_broadcast) else q_idx
                            mask_k_idx = Int32.zero if const_expr(self.mask_k_broadcast) else k_idx
                            score += Float32(mMask[mask_batch_idx, mask_head_idx, mask_q_idx, mask_k_idx])
                        if const_expr(mKeyMask is not None):
                            score = score if mKeyMask[batch_idx, k_idx] != 0 else -Float32.inf
                        local_max = cute.arch.fmax(local_max, score)
                block_max = cute.arch.warp_reduction(local_max, cute.arch.fmax)
                max_score = cute.arch.fmax(max_score, block_max)

            denom = Float32.zero
            acc0 = Float32.zero
            acc1 = Float32.zero
            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                local_sum = Float32.zero
                local_acc0 = Float32.zero
                local_acc1 = Float32.zero
                for i in cutlass.range_constexpr(elems_per_lane):
                    k_idx = k_tile * self.k_block + lane_idx + i * cute.arch.WARP_SIZE
                    if k_idx < self.seqlen_k:
                        score = (q0 * Float32(mK[bh_idx, k_idx, 0]) + q1 * Float32(mK[bh_idx, k_idx, 1])) * scale
                        if const_expr(mMask is not None):
                            mask_batch_idx = Int32.zero if const_expr(self.mask_batch_broadcast) else batch_idx
                            mask_head_idx = Int32.zero if const_expr(self.mask_head_broadcast) else head_idx
                            mask_q_idx = Int32.zero if const_expr(self.mask_q_broadcast) else q_idx
                            mask_k_idx = Int32.zero if const_expr(self.mask_k_broadcast) else k_idx
                            score += Float32(mMask[mask_batch_idx, mask_head_idx, mask_q_idx, mask_k_idx])
                        if const_expr(mKeyMask is not None):
                            score = score if mKeyMask[batch_idx, k_idx] != 0 else -Float32.inf
                        prob = cute.math.exp(score - max_score, fastmath=True)
                        local_sum += prob
                        local_acc0 += prob * Float32(mV[bh_idx, k_idx, 0])
                        local_acc1 += prob * Float32(mV[bh_idx, k_idx, 1])
                denom += cute.arch.warp_reduction(local_sum, operator.add)
                acc0 += cute.arch.warp_reduction(local_acc0, operator.add)
                acc1 += cute.arch.warp_reduction(local_acc1, operator.add)

            if lane_idx == 0:
                inv_denom = cute.arch.rcp_approx(denom)
                mO[bh_idx, q_idx, 0] = mO.element_type(acc0 * inv_denom)
                mO[bh_idx, q_idx, 1] = mO.element_type(acc1 * inv_denom)
                mLSE[bh_idx, q_idx] = max_score + cute.math.log(denom, fastmath=True)


class HullAttnFullBackwardDQCute:
    def __init__(
        self,
        dtype,
        seqlen_q: int,
        seqlen_k: int,
        mask_batch_broadcast: bool = True,
        mask_head_broadcast: bool = True,
        mask_q_broadcast: bool = False,
        mask_k_broadcast: bool = False,
        k_block: int = CUTE_K_BLOCK,
    ):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.mask_batch_broadcast = mask_batch_broadcast
        self.mask_head_broadcast = mask_head_broadcast
        self.mask_q_broadcast = mask_q_broadcast
        self.mask_k_broadcast = mask_k_broadcast
        self.k_block = k_block

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMask: Optional[cute.Tensor],
        mKeyMask: Optional[cute.Tensor],
        num_heads: Int32,
        mOut: cute.Tensor,
        mLSE: cute.Tensor,
        mdOut: cute.Tensor,
        mdQ: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(mQ, mK, mV, mMask, mKeyMask, num_heads, mOut, mLSE, mdOut, mdQ, scale).launch(
            grid=[self.seqlen_q, mQ.shape[0], 1],
            block=[cute.arch.WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMask: Optional[cute.Tensor],
        mKeyMask: Optional[cute.Tensor],
        num_heads: Int32,
        mOut: cute.Tensor,
        mLSE: cute.Tensor,
        mdOut: cute.Tensor,
        mdQ: cute.Tensor,
        scale: Float32,
    ):
        lane_idx = cute.arch.lane_idx()
        q_idx, bh_idx, _ = cute.arch.block_idx()

        if q_idx < self.seqlen_q and bh_idx < mQ.shape[0]:
            batch_idx = bh_idx // num_heads
            head_idx = bh_idx - batch_idx * num_heads
            q0 = Float32(mQ[bh_idx, q_idx, 0])
            q1 = Float32(mQ[bh_idx, q_idx, 1])
            do0 = Float32(mdOut[bh_idx, q_idx, 0])
            do1 = Float32(mdOut[bh_idx, q_idx, 1])
            out0 = Float32(mOut[bh_idx, q_idx, 0])
            out1 = Float32(mOut[bh_idx, q_idx, 1])
            lse = Float32(mLSE[bh_idx, q_idx])
            delta = do0 * out0 + do1 * out1
            dq0 = Float32.zero
            dq1 = Float32.zero
            num_k_tiles = cute.ceil_div(self.seqlen_k, self.k_block)
            elems_per_lane = const_expr(cute.ceil_div(self.k_block, cute.arch.WARP_SIZE))

            for k_tile in cutlass.range(num_k_tiles, unroll=1):
                local_dq0 = Float32.zero
                local_dq1 = Float32.zero
                for i in cutlass.range_constexpr(elems_per_lane):
                    k_idx = k_tile * self.k_block + lane_idx + i * cute.arch.WARP_SIZE
                    if k_idx < self.seqlen_k:
                        k0 = Float32(mK[bh_idx, k_idx, 0])
                        k1 = Float32(mK[bh_idx, k_idx, 1])
                        v0 = Float32(mV[bh_idx, k_idx, 0])
                        v1 = Float32(mV[bh_idx, k_idx, 1])
                        score = (q0 * k0 + q1 * k1) * scale
                        if const_expr(mMask is not None):
                            mask_batch_idx = Int32.zero if const_expr(self.mask_batch_broadcast) else batch_idx
                            mask_head_idx = Int32.zero if const_expr(self.mask_head_broadcast) else head_idx
                            mask_q_idx = Int32.zero if const_expr(self.mask_q_broadcast) else q_idx
                            mask_k_idx = Int32.zero if const_expr(self.mask_k_broadcast) else k_idx
                            score += Float32(mMask[mask_batch_idx, mask_head_idx, mask_q_idx, mask_k_idx])
                        if const_expr(mKeyMask is not None):
                            score = score if mKeyMask[batch_idx, k_idx] != 0 else -Float32.inf
                        prob = cute.math.exp(score - lse, fastmath=True)
                        dp = do0 * v0 + do1 * v1
                        ds = prob * (dp - delta)
                        local_dq0 += ds * k0
                        local_dq1 += ds * k1
                dq0 += cute.arch.warp_reduction(local_dq0, operator.add)
                dq1 += cute.arch.warp_reduction(local_dq1, operator.add)

            if lane_idx == 0:
                mdQ[bh_idx, q_idx, 0] = mdQ.element_type(dq0 * scale)
                mdQ[bh_idx, q_idx, 1] = mdQ.element_type(dq1 * scale)


class HullAttnFullBackwardDKDVCute:
    def __init__(
        self,
        dtype,
        seqlen_q: int,
        seqlen_k: int,
        mask_batch_broadcast: bool = True,
        mask_head_broadcast: bool = True,
        mask_q_broadcast: bool = False,
        mask_k_broadcast: bool = False,
        q_block: int = DEFAULT_Q_BLOCK,
    ):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.mask_batch_broadcast = mask_batch_broadcast
        self.mask_head_broadcast = mask_head_broadcast
        self.mask_q_broadcast = mask_q_broadcast
        self.mask_k_broadcast = mask_k_broadcast
        self.q_block = q_block

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMask: Optional[cute.Tensor],
        mKeyMask: Optional[cute.Tensor],
        num_heads: Int32,
        mOut: cute.Tensor,
        mLSE: cute.Tensor,
        mdOut: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(mQ, mK, mV, mMask, mKeyMask, num_heads, mOut, mLSE, mdOut, mdK, mdV, scale).launch(
            grid=[self.seqlen_k, mK.shape[0], 1],
            block=[cute.arch.WARP_SIZE, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mMask: Optional[cute.Tensor],
        mKeyMask: Optional[cute.Tensor],
        num_heads: Int32,
        mOut: cute.Tensor,
        mLSE: cute.Tensor,
        mdOut: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        scale: Float32,
    ):
        lane_idx = cute.arch.lane_idx()
        k_idx, bh_idx, _ = cute.arch.block_idx()

        if k_idx < self.seqlen_k and bh_idx < mK.shape[0]:
            batch_idx = bh_idx // num_heads
            head_idx = bh_idx - batch_idx * num_heads
            k0 = Float32(mK[bh_idx, k_idx, 0])
            k1 = Float32(mK[bh_idx, k_idx, 1])
            v0 = Float32(mV[bh_idx, k_idx, 0])
            v1 = Float32(mV[bh_idx, k_idx, 1])
            dk0 = Float32.zero
            dk1 = Float32.zero
            dv0 = Float32.zero
            dv1 = Float32.zero
            num_q_tiles = cute.ceil_div(self.seqlen_q, self.q_block)
            elems_per_lane = const_expr(cute.ceil_div(self.q_block, cute.arch.WARP_SIZE))

            for q_tile in cutlass.range(num_q_tiles, unroll=1):
                local_dk0 = Float32.zero
                local_dk1 = Float32.zero
                local_dv0 = Float32.zero
                local_dv1 = Float32.zero
                for i in cutlass.range_constexpr(elems_per_lane):
                    q_idx = q_tile * self.q_block + lane_idx + i * cute.arch.WARP_SIZE
                    if q_idx < self.seqlen_q:
                        q0 = Float32(mQ[bh_idx, q_idx, 0])
                        q1 = Float32(mQ[bh_idx, q_idx, 1])
                        do0 = Float32(mdOut[bh_idx, q_idx, 0])
                        do1 = Float32(mdOut[bh_idx, q_idx, 1])
                        out0 = Float32(mOut[bh_idx, q_idx, 0])
                        out1 = Float32(mOut[bh_idx, q_idx, 1])
                        lse = Float32(mLSE[bh_idx, q_idx])
                        score = (q0 * k0 + q1 * k1) * scale
                        if const_expr(mMask is not None):
                            mask_batch_idx = Int32.zero if const_expr(self.mask_batch_broadcast) else batch_idx
                            mask_head_idx = Int32.zero if const_expr(self.mask_head_broadcast) else head_idx
                            mask_q_idx = Int32.zero if const_expr(self.mask_q_broadcast) else q_idx
                            mask_k_idx = Int32.zero if const_expr(self.mask_k_broadcast) else k_idx
                            score += Float32(mMask[mask_batch_idx, mask_head_idx, mask_q_idx, mask_k_idx])
                        if const_expr(mKeyMask is not None):
                            score = score if mKeyMask[batch_idx, k_idx] != 0 else -Float32.inf
                        prob = cute.math.exp(score - lse, fastmath=True)
                        delta = do0 * out0 + do1 * out1
                        dp = do0 * v0 + do1 * v1
                        ds = prob * (dp - delta)
                        local_dk0 += ds * q0
                        local_dk1 += ds * q1
                        local_dv0 += prob * do0
                        local_dv1 += prob * do1
                dk0 += cute.arch.warp_reduction(local_dk0, operator.add)
                dk1 += cute.arch.warp_reduction(local_dk1, operator.add)
                dv0 += cute.arch.warp_reduction(local_dv0, operator.add)
                dv1 += cute.arch.warp_reduction(local_dv1, operator.add)

            if lane_idx == 0:
                mdK[bh_idx, k_idx, 0] = mdK.element_type(dk0 * scale)
                mdK[bh_idx, k_idx, 1] = mdK.element_type(dk1 * scale)
                mdV[bh_idx, k_idx, 0] = mdV.element_type(dv0)
                mdV[bh_idx, k_idx, 1] = mdV.element_type(dv1)


@lru_cache(maxsize=None)
def _compile_hull_attn_full_fwd(dtype, seqlen_q, seqlen_k, mask_shape, key_mask_shape):
    key = ("hull_attn_full_fwd", dtype, seqlen_q, seqlen_k, mask_shape, key_mask_shape)
    has_mask = mask_shape is not None
    has_key_mask = key_mask_shape is not None

    def _compile():
        bh_sym = cute.sym_int()
        batch_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, 2)
        q_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        k_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        v_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        mask_cute = (
            fake_tensor(Float32, mask_shape, divisibility=4)
            if has_mask
            else None
        )
        key_mask_cute = fake_tensor(Int32, key_mask_shape, divisibility=1) if has_key_mask else None
        out_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        lse_cute = fake_tensor(Float32, (bh_sym, seqlen_q), divisibility=1)
        op = HullAttnFullForwardCute(
            dtype,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            mask_batch_broadcast=has_mask and mask_shape[0] == 1,
            mask_head_broadcast=has_mask and mask_shape[1] == 1,
            mask_q_broadcast=has_mask and mask_shape[2] == 1,
            mask_k_broadcast=has_mask and mask_shape[3] == 1,
        )
        return cute.compile(
            op,
            q_cute,
            k_cute,
            v_cute,
            mask_cute,
            key_mask_cute,
            Int32(1),
            out_cute,
            lse_cute,
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


@lru_cache(maxsize=None)
def _compile_hull_attn_full_bwd_dq(dtype, seqlen_q, seqlen_k, mask_shape, key_mask_shape):
    key = ("hull_attn_full_bwd_dq", dtype, seqlen_q, seqlen_k, mask_shape, key_mask_shape)
    has_mask = mask_shape is not None
    has_key_mask = key_mask_shape is not None

    def _compile():
        bh_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, 2)
        q_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        k_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        v_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        mask_cute = (
            fake_tensor(Float32, mask_shape, divisibility=4)
            if has_mask
            else None
        )
        key_mask_cute = fake_tensor(Int32, key_mask_shape, divisibility=1) if has_key_mask else None
        out_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        lse_cute = fake_tensor(Float32, (bh_sym, seqlen_q), divisibility=1)
        dout_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        dq_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        op = HullAttnFullBackwardDQCute(
            dtype,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            mask_batch_broadcast=has_mask and mask_shape[0] == 1,
            mask_head_broadcast=has_mask and mask_shape[1] == 1,
            mask_q_broadcast=has_mask and mask_shape[2] == 1,
            mask_k_broadcast=has_mask and mask_shape[3] == 1,
        )
        return cute.compile(
            op,
            q_cute,
            k_cute,
            v_cute,
            mask_cute,
            key_mask_cute,
            Int32(1),
            out_cute,
            lse_cute,
            dout_cute,
            dq_cute,
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


@lru_cache(maxsize=None)
def _compile_hull_attn_full_bwd_dkdv(dtype, seqlen_q, seqlen_k, mask_shape, key_mask_shape):
    key = ("hull_attn_full_bwd_dkdv", dtype, seqlen_q, seqlen_k, mask_shape, key_mask_shape)
    has_mask = mask_shape is not None
    has_key_mask = key_mask_shape is not None

    def _compile():
        bh_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, 2)
        q_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        k_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        v_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        mask_cute = (
            fake_tensor(Float32, mask_shape, divisibility=4)
            if has_mask
            else None
        )
        key_mask_cute = fake_tensor(Int32, key_mask_shape, divisibility=1) if has_key_mask else None
        out_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        lse_cute = fake_tensor(Float32, (bh_sym, seqlen_q), divisibility=1)
        dout_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        dk_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        dv_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        op = HullAttnFullBackwardDKDVCute(
            dtype,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            mask_batch_broadcast=has_mask and mask_shape[0] == 1,
            mask_head_broadcast=has_mask and mask_shape[1] == 1,
            mask_q_broadcast=has_mask and mask_shape[2] == 1,
            mask_k_broadcast=has_mask and mask_shape[3] == 1,
        )
        return cute.compile(
            op,
            q_cute,
            k_cute,
            v_cute,
            mask_cute,
            key_mask_cute,
            Int32(1),
            out_cute,
            lse_cute,
            dout_cute,
            dk_cute,
            dv_cute,
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


@lru_cache(maxsize=None)
def _compile_hull_attn_sparse_fwd(dtype, topk, seqlen_q, seqlen_k, mask_shape, key_mask_shape):
    key = ("hull_attn_sparse_fwd", dtype, topk, seqlen_q, seqlen_k, mask_shape, key_mask_shape)
    has_mask = mask_shape is not None
    has_key_mask = key_mask_shape is not None

    def _compile():
        bh_sym = cute.sym_int()
        div = math.gcd(128 // dtype.width, 2)
        q_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        k_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        v_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=div)
        mask_cute = fake_tensor(Float32, mask_shape, divisibility=1) if has_mask else None
        key_mask_cute = fake_tensor(Int32, key_mask_shape, divisibility=1) if has_key_mask else None
        out_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=div)
        op = HullAttnSparseForwardCute(
            dtype,
            topk=topk,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            mask_batch_broadcast=has_mask and mask_shape[0] == 1,
            mask_head_broadcast=has_mask and mask_shape[1] == 1,
            mask_q_broadcast=has_mask and mask_shape[2] == 1,
            mask_k_broadcast=has_mask and mask_shape[3] == 1,
        )
        return cute.compile(
            op,
            q_cute,
            k_cute,
            v_cute,
            mask_cute,
            key_mask_cute,
            Int32(1),
            out_cute,
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


@torch.library.custom_op("quack::_hull_attn_full_fwd", mutates_args={"out", "lse"}, device_types="cuda")
def _hull_attn_full_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn forward expects head_dim=2"
    assert k.shape == (bh, k.shape[1], 2)
    assert v.shape == (bh, k.shape[1], 2)
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_fwd(dtype, seqlen_q, k.shape[1], None, None)(
        q, k, v, None, None, 1, out, lse, scale
    )


@torch.library.custom_op(
    "quack::_hull_attn_full_fwd_masked",
    mutates_args={"out", "lse"},
    device_types="cuda",
)
def _hull_attn_full_fwd_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    scale: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn forward expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_fwd(dtype, seqlen_q, k.shape[1], tuple(mask.shape), None)(
        q, k, v, mask, None, num_heads, out, lse, scale
    )


@torch.library.custom_op(
    "quack::_hull_attn_full_fwd_keymask",
    mutates_args={"out", "lse"},
    device_types="cuda",
)
def _hull_attn_full_fwd_keymask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    num_heads: int,
    scale: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn forward expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_fwd(dtype, seqlen_q, k.shape[1], None, tuple(key_mask.shape))(
        q, k, v, None, key_mask, num_heads, out, lse, scale
    )


@torch.library.custom_op("quack::_hull_attn_sparse_fwd", mutates_args={"out"}, device_types="cuda")
def _hull_attn_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk: int,
    scale: float,
    out: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert topk in (1, 4), "CuTe sparse hull_attn supports topk=1 or 4"
    assert head_dim == 2, "CuTe sparse hull_attn expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_sparse_fwd(dtype, topk, seqlen_q, k.shape[1], None, None)(
        q, k, v, None, None, 1, out, scale
    )


@torch.library.custom_op("quack::_hull_attn_sparse_fwd_masked", mutates_args={"out"}, device_types="cuda")
def _hull_attn_sparse_fwd_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    topk: int,
    scale: float,
    out: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert topk in (1, 4), "CuTe sparse hull_attn supports topk=1 or 4"
    assert head_dim == 2, "CuTe sparse hull_attn expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_sparse_fwd(dtype, topk, seqlen_q, k.shape[1], tuple(mask.shape), None)(
        q, k, v, mask, None, num_heads, out, scale
    )


@torch.library.custom_op("quack::_hull_attn_sparse_fwd_keymask", mutates_args={"out"}, device_types="cuda")
def _hull_attn_sparse_fwd_keymask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    num_heads: int,
    topk: int,
    scale: float,
    out: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert topk in (1, 4), "CuTe sparse hull_attn supports topk=1 or 4"
    assert head_dim == 2, "CuTe sparse hull_attn expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_sparse_fwd(dtype, topk, seqlen_q, k.shape[1], None, tuple(key_mask.shape))(
        q, k, v, None, key_mask, num_heads, out, scale
    )


@torch.library.custom_op("quack::_hull_attn_full_bwd_dq", mutates_args={"dq"}, device_types="cuda")
def _hull_attn_full_bwd_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dq: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn backward expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_bwd_dq(dtype, seqlen_q, k.shape[1], None, None)(
        q, k, v, None, None, 1, out, lse, grad_out, dq, scale
    )


@torch.library.custom_op("quack::_hull_attn_full_bwd_dq_masked", mutates_args={"dq"}, device_types="cuda")
def _hull_attn_full_bwd_dq_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dq: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn backward expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_bwd_dq(dtype, seqlen_q, k.shape[1], tuple(mask.shape), None)(
        q, k, v, mask, None, num_heads, out, lse, grad_out, dq, scale
    )


@torch.library.custom_op("quack::_hull_attn_full_bwd_dq_keymask", mutates_args={"dq"}, device_types="cuda")
def _hull_attn_full_bwd_dq_keymask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    num_heads: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dq: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn backward expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_bwd_dq(dtype, seqlen_q, k.shape[1], None, tuple(key_mask.shape))(
        q, k, v, None, key_mask, num_heads, out, lse, grad_out, dq, scale
    )


@torch.library.custom_op(
    "quack::_hull_attn_full_bwd_dkdv",
    mutates_args={"dk", "dv"},
    device_types="cuda",
)
def _hull_attn_full_bwd_dkdv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn backward expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_bwd_dkdv(dtype, seqlen_q, k.shape[1], None, None)(
        q, k, v, None, None, 1, out, lse, grad_out, dk, dv, scale
    )


@torch.library.custom_op(
    "quack::_hull_attn_full_bwd_dkdv_masked",
    mutates_args={"dk", "dv"},
    device_types="cuda",
)
def _hull_attn_full_bwd_dkdv_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn backward expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_bwd_dkdv(dtype, seqlen_q, k.shape[1], tuple(mask.shape), None)(
        q, k, v, mask, None, num_heads, out, lse, grad_out, dk, dv, scale
    )


@torch.library.custom_op(
    "quack::_hull_attn_full_bwd_dkdv_keymask",
    mutates_args={"dk", "dv"},
    device_types="cuda",
)
def _hull_attn_full_bwd_dkdv_keymask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    num_heads: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> None:
    bh, seqlen_q, head_dim = q.shape
    assert head_dim == 2, "CuTe hull_attn backward expects head_dim=2"
    dtype = torch2cute_dtype_map[q.dtype]
    _compile_hull_attn_full_bwd_dkdv(dtype, seqlen_q, k.shape[1], None, tuple(key_mask.shape))(
        q, k, v, None, key_mask, num_heads, out, lse, grad_out, dk, dv, scale
    )


@_hull_attn_full_fwd.register_fake
def _hull_attn_full_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_fwd(dtype, q.size(1), k.size(1), None, None)
        _compile_hull_attn_full_bwd_dq(dtype, q.size(1), k.size(1), None, None)
        _compile_hull_attn_full_bwd_dkdv(dtype, q.size(1), k.size(1), None, None)


@_hull_attn_full_fwd_masked.register_fake
def _hull_attn_full_fwd_masked_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    scale: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_fwd(dtype, q.size(1), k.size(1), tuple(mask.shape), None)
        _compile_hull_attn_full_bwd_dq(dtype, q.size(1), k.size(1), tuple(mask.shape), None)
        _compile_hull_attn_full_bwd_dkdv(dtype, q.size(1), k.size(1), tuple(mask.shape), None)


@_hull_attn_full_fwd_keymask.register_fake
def _hull_attn_full_fwd_keymask_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    num_heads: int,
    scale: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_fwd(dtype, q.size(1), k.size(1), None, tuple(key_mask.shape))
        _compile_hull_attn_full_bwd_dq(dtype, q.size(1), k.size(1), None, tuple(key_mask.shape))
        _compile_hull_attn_full_bwd_dkdv(dtype, q.size(1), k.size(1), None, tuple(key_mask.shape))


@_hull_attn_sparse_fwd.register_fake
def _hull_attn_sparse_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk: int,
    scale: float,
    out: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_sparse_fwd(dtype, topk, q.size(1), k.size(1), None, None)


@_hull_attn_sparse_fwd_masked.register_fake
def _hull_attn_sparse_fwd_masked_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    topk: int,
    scale: float,
    out: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_sparse_fwd(dtype, topk, q.size(1), k.size(1), tuple(mask.shape), None)


@_hull_attn_sparse_fwd_keymask.register_fake
def _hull_attn_sparse_fwd_keymask_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    num_heads: int,
    topk: int,
    scale: float,
    out: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_sparse_fwd(dtype, topk, q.size(1), k.size(1), None, tuple(key_mask.shape))


@_hull_attn_full_bwd_dq.register_fake
def _hull_attn_full_bwd_dq_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dq: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_bwd_dq(dtype, q.size(1), k.size(1), None, None)


@_hull_attn_full_bwd_dq_masked.register_fake
def _hull_attn_full_bwd_dq_masked_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dq: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_bwd_dq(dtype, q.size(1), k.size(1), tuple(mask.shape), None)


@_hull_attn_full_bwd_dq_keymask.register_fake
def _hull_attn_full_bwd_dq_keymask_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    num_heads: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dq: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_bwd_dq(dtype, q.size(1), k.size(1), None, tuple(key_mask.shape))


@_hull_attn_full_bwd_dkdv.register_fake
def _hull_attn_full_bwd_dkdv_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_bwd_dkdv(dtype, q.size(1), k.size(1), None, None)


@_hull_attn_full_bwd_dkdv_masked.register_fake
def _hull_attn_full_bwd_dkdv_masked_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    num_heads: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_bwd_dkdv(dtype, q.size(1), k.size(1), tuple(mask.shape), None)


@_hull_attn_full_bwd_dkdv_keymask.register_fake
def _hull_attn_full_bwd_dkdv_keymask_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
    num_heads: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not any(isinstance(x, torch.SymInt) for x in (q.size(1), k.size(1))):
        dtype = torch2cute_dtype_map[q.dtype]
        _compile_hull_attn_full_bwd_dkdv(dtype, q.size(1), k.size(1), None, tuple(key_mask.shape))


def _can_use_cute_full_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> bool:
    return (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] == 2
        and q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and (key_padding_mask is None or key_padding_mask.is_contiguous())
    )


def _can_use_cute_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> bool:
    return _can_use_cute_full_fwd(q, k, v, attention_mask, key_padding_mask)


def _prepare_cute_mask(
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
    return mask.contiguous()


def _prepare_cute_key_mask(
    key_padding_mask: torch.Tensor | None,
    q: torch.Tensor,
) -> torch.Tensor | None:
    if key_padding_mask is None:
        return None
    if key_padding_mask.shape != (q.shape[0], q.shape[-2]):
        return None
    return key_padding_mask.to(device=q.device, dtype=torch.int32).contiguous()


def _can_use_cute_full_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> bool:
    return (
        _can_use_cute_full_fwd(q, k, v, attention_mask, key_padding_mask)
        and out.is_contiguous()
        and grad_out.is_contiguous()
        and grad_out.dtype == q.dtype
    )


def _full_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _can_use_cute_full_fwd(q, k, v, attention_mask, key_padding_mask):
        batch, heads, seqlen_q, _ = q.shape
        seqlen_k = k.shape[-2]
        q_flat = q.view(batch * heads, seqlen_q, 2)
        k_flat = k.view(batch * heads, seqlen_k, 2)
        v_flat = v.view(batch * heads, seqlen_k, 2)
        mask_flat = _prepare_cute_mask(attention_mask, q, k)
        key_mask = _prepare_cute_key_mask(key_padding_mask, q)
        out_flat = torch.empty_like(q_flat)
        lse_flat = torch.empty((batch * heads, seqlen_q), device=q.device, dtype=torch.float32)
        if key_mask is not None:
            _hull_attn_full_fwd_keymask(
                q_flat, k_flat, v_flat, key_mask, heads, scale, out_flat, lse_flat
            )
        elif mask_flat is None:
            _hull_attn_full_fwd(q_flat, k_flat, v_flat, scale, out_flat, lse_flat)
        else:
            _hull_attn_full_fwd_masked(
                q_flat, k_flat, v_flat, mask_flat, heads, scale, out_flat, lse_flat
            )
        return out_flat.view_as(q), lse_flat.view(batch, heads, seqlen_q)
    return _streaming_full_attention_forward(
        q, k, v, scale=scale, attention_mask=attention_mask, key_padding_mask=key_padding_mask
    )


def _sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk: int,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    if _can_use_cute_sparse_fwd(q, k, v, attention_mask, key_padding_mask):
        batch, heads, seqlen_q, _ = q.shape
        seqlen_k = k.shape[-2]
        q_flat = q.view(batch * heads, seqlen_q, 2)
        k_flat = k.view(batch * heads, seqlen_k, 2)
        v_flat = v.view(batch * heads, seqlen_k, 2)
        mask_flat = _prepare_cute_mask(attention_mask, q, k)
        key_mask = _prepare_cute_key_mask(key_padding_mask, q)
        out_flat = torch.empty_like(q_flat)
        if key_mask is not None:
            _hull_attn_sparse_fwd_keymask(
                q_flat, k_flat, v_flat, key_mask, heads, topk, scale, out_flat
            )
        elif mask_flat is None:
            _hull_attn_sparse_fwd(q_flat, k_flat, v_flat, topk, scale, out_flat)
        else:
            _hull_attn_sparse_fwd_masked(
                q_flat, k_flat, v_flat, mask_flat, heads, topk, scale, out_flat
            )
        return out_flat.view_as(q)
    return _reference_topk_attention(
        q, k, v, topk=topk, scale=scale, attention_mask=attention_mask, key_padding_mask=key_padding_mask
    )


def _full_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _can_use_cute_full_bwd(q, k, v, out, grad_out, attention_mask, key_padding_mask):
        batch, heads, seqlen_q, _ = q.shape
        seqlen_k = k.shape[-2]
        flat_shape_q = (batch * heads, seqlen_q, 2)
        flat_shape_k = (batch * heads, seqlen_k, 2)
        q_flat = q.view(flat_shape_q)
        k_flat = k.view(flat_shape_k)
        v_flat = v.view(flat_shape_k)
        out_flat = out.view(flat_shape_q)
        lse_flat = lse.view(batch * heads, seqlen_q)
        grad_out_flat = grad_out.contiguous().view(flat_shape_q)
        mask_flat = _prepare_cute_mask(attention_mask, q, k)
        key_mask = _prepare_cute_key_mask(key_padding_mask, q)
        dq_flat = torch.empty_like(q_flat)
        dk_flat = torch.empty_like(k_flat)
        dv_flat = torch.empty_like(v_flat)
        if key_mask is not None:
            _hull_attn_full_bwd_dq_keymask(
                q_flat, k_flat, v_flat, key_mask, heads, out_flat, lse_flat, grad_out_flat, scale, dq_flat
            )
            _hull_attn_full_bwd_dkdv_keymask(
                q_flat, k_flat, v_flat, key_mask, heads, out_flat, lse_flat, grad_out_flat, scale, dk_flat, dv_flat
            )
        elif mask_flat is None:
            _hull_attn_full_bwd_dq(
                q_flat, k_flat, v_flat, out_flat, lse_flat, grad_out_flat, scale, dq_flat
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
        else:
            _hull_attn_full_bwd_dq_masked(
                q_flat,
                k_flat,
                v_flat,
                mask_flat,
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
                mask_flat,
                heads,
                out_flat,
                lse_flat,
                grad_out_flat,
                scale,
                dk_flat,
                dv_flat,
            )
        return dq_flat.view_as(q), dk_flat.view_as(k), dv_flat.view_as(v)

    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]
    q_block = min(DEFAULT_Q_BLOCK, seqlen_q)
    k_block = min(DEFAULT_K_BLOCK, seqlen_k)

    grad_out_f32 = grad_out.float()
    v_f32 = v.float()
    q_f32 = q.float()
    k_f32 = k.float()
    out_f32 = out.float()

    dq = torch.zeros_like(q_f32)
    dk = torch.zeros_like(k_f32)
    dv = torch.zeros_like(v_f32)

    for q_start in range(0, seqlen_q, q_block):
        q_end = min(q_start + q_block, q_block + q_start)
        q_chunk = q_f32[:, :, q_start:q_end, :]
        grad_chunk = grad_out_f32[:, :, q_start:q_end, :]
        out_chunk = out_f32[:, :, q_start:q_end, :]
        lse_chunk = lse[:, :, q_start:q_end]
        delta = (grad_chunk * out_chunk).sum(dim=-1, keepdim=True)
        dq_chunk = torch.zeros_like(q_chunk)

        for k_start in range(0, seqlen_k, k_block):
            k_end = min(k_start + k_block, seqlen_k)
            k_chunk = k_f32[:, :, k_start:k_end, :]
            v_chunk = v_f32[:, :, k_start:k_end, :]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scale
            mask_chunk = _mask_chunk(attention_mask, q_start, q_end, k_start, k_end)
            scores = _apply_attention_mask(scores, mask_chunk)
            scores = _apply_key_padding_mask(scores, _key_mask_chunk(key_padding_mask, k_start, k_end))
            probs = torch.exp(scores - lse_chunk.unsqueeze(-1))
            dv[:, :, k_start:k_end, :] += torch.matmul(probs.transpose(-1, -2), grad_chunk)
            dp = torch.matmul(grad_chunk, v_chunk.transpose(-1, -2))
            ds = probs * (dp - delta)
            dq_chunk += torch.matmul(ds, k_chunk)
            dk[:, :, k_start:k_end, :] += torch.matmul(ds.transpose(-1, -2), q_chunk)

        dq[:, :, q_start:q_end, :] = dq_chunk * scale

    dk *= scale
    return dq.to(dtype=q.dtype), dk.to(dtype=k.dtype), dv.to(dtype=v.dtype)


class HullAttnFullFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, attention_mask, key_padding_mask):
        out, lse = _full_attention_forward(
            q,
            k,
            v,
            scale=scale,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.attention_mask = attention_mask
        ctx.key_padding_mask = key_padding_mask
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _full_attention_backward(
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
        return dq, dk, dv, None, None, None


def hull_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mode: HullAttnMode = "full",
    scale: float | None = None,
    attention_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference hull attention API specialized to head_dim=2.

    Args:
        q: Query tensor with shape [batch, heads, seq, 2].
        k: Key tensor with shape [batch, heads, seq, 2].
        v: Value tensor with shape [batch, heads, seq, 2].
        mode: One of "full", "topk1", or "topk4".
        scale: Optional score scale. Defaults to 1 / sqrt(2).
        attention_mask: Optional dense mask broadcastable to [batch, heads, seq, seq].
            Boolean masks use True for valid positions. Floating-point masks are additive.
        key_padding_mask: Optional boolean mask with shape [batch, seq]. True means valid key.
        seq_lens: Optional valid lengths with shape [batch].

    Returns:
        Tensor with shape [batch, heads, seq, 2].
    """
    attention_mask, key_padding_mask = _normalize_mask_inputs(
        q, attention_mask, key_padding_mask, seq_lens
    )
    _check_inputs(q, k, v, mode, attention_mask, key_padding_mask, None)
    scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else float(scale)

    if mode == "full":
        return HullAttnFullFunction.apply(q, k, v, scale, attention_mask, key_padding_mask)

    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise NotImplementedError(f"hull_attn mode={mode!r} is forward-only in v1")

    topk = 1 if mode == "topk1" else 4
    return _sparse_attention_forward(
        q, k, v, topk=topk, scale=scale, attention_mask=attention_mask, key_padding_mask=key_padding_mask
    )


__all__ = ["hull_attn"]
