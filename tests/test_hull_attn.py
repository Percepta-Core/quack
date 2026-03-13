# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

import math
from unittest import mock

import pytest
import torch

from quack.hull_attn import (
    _hull_attn_full_bwd_dkdv,
    _hull_attn_full_bwd_dkdv_masked,
    _hull_attn_full_bwd_dq,
    _hull_attn_full_bwd_dq_masked,
    _hull_attn_full_fwd,
    _hull_attn_full_fwd_masked,
    hull_attn,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

TOLERANCES = {
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16: (2e-3, 2e-3),
    torch.float32: (1e-4, 1e-4),
}

CUTE_TOLERANCES = {
    torch.bfloat16: (8e-3, 8e-3),
    torch.float16: (2e-3, 2e-3),
}


def _reference_hull_attn(q, k, v, mode="full", scale=None, attention_mask=None):
    scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else float(scale)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, -torch.inf)
        else:
            scores = scores + attention_mask.float()
    if mode == "full":
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v.float()).to(dtype=q.dtype)

    k_select = 1 if mode == "topk1" else 4
    topk_scores, topk_indices = torch.topk(
        scores, k=min(k_select, scores.shape[-1]), dim=-1, largest=True, sorted=True
    )
    probs = torch.softmax(topk_scores, dim=-1)
    selected_v = torch.gather(
        v.float().unsqueeze(-3).expand(*v.shape[:-2], q.shape[-2], v.shape[-2], v.shape[-1]),
        dim=-2,
        index=topk_indices.unsqueeze(-1).expand(*topk_indices.shape, v.shape[-1]),
    )
    return (probs.unsqueeze(-1) * selected_v).sum(dim=-2).to(dtype=q.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hull_attn_full_matches_reference_and_gradients(dtype):
    atol, rtol = TOLERANCES[dtype]
    device = "cuda"
    torch.random.manual_seed(0)
    shape = (2, 4, 8, 2)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = hull_attn(q, k, v, mode="full")
    out_ref = _reference_hull_attn(q_ref, k_ref, v_ref, mode="full")

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("mask_kind", ["bool", "float"])
def test_hull_attn_full_mask_matches_reference(mask_kind):
    device = "cuda"
    dtype = torch.float32
    torch.random.manual_seed(1)
    shape = (1, 2, 6, 2)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    if mask_kind == "bool":
        mask = torch.ones((1, 1, 6, 6), device=device, dtype=torch.bool)
        mask[..., 0, -1] = False
        mask[..., 3, :2] = False
    else:
        mask = torch.zeros((1, 1, 6, 6), device=device, dtype=torch.float32)
        mask[..., 1, 4] = -1e4
        mask[..., 2, 0] = -1e4

    out = hull_attn(q, k, v, mode="full", attention_mask=mask)
    out_ref = _reference_hull_attn(q, k, v, mode="full", attention_mask=mask)
    torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hull_attn_full_key_padding_mask_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    device = "cuda"
    torch.manual_seed(3)
    shape = (2, 3, 9, 2)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    key_padding_mask = torch.tensor(
        [[True, True, True, True, True, False, False, False, False],
         [True, True, True, True, False, False, False, False, False]],
        device=device,
        dtype=torch.bool,
    )
    dense_mask = key_padding_mask[:, None, None, :]

    out = hull_attn(q, k, v, mode="full", key_padding_mask=key_padding_mask)
    out_ref = _reference_hull_attn(q_ref, k_ref, v_ref, mode="full", attention_mask=dense_mask)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hull_attn_full_seq_lens_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    device = "cuda"
    torch.manual_seed(4)
    shape = (2, 2, 10, 2)
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    seq_lens = torch.tensor([7, 4], device=device, dtype=torch.int32)
    key_padding_mask = (
        torch.arange(shape[2], device=device)[None, :] < seq_lens[:, None]
    )
    dense_mask = key_padding_mask[:, None, None, :]

    out = hull_attn(q, k, v, mode="full", seq_lens=seq_lens)
    out_ref = _reference_hull_attn(q_ref, k_ref, v_ref, mode="full", attention_mask=dense_mask)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("mode", ["topk1", "topk4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hull_attn_topk_matches_reference(mode, dtype):
    atol, rtol = TOLERANCES[dtype]
    device = "cuda"
    torch.random.manual_seed(2)
    shape = (2, 3, 7, 2)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    mask = torch.ones((1, 1, 7, 7), device=device, dtype=torch.bool)
    mask[..., 5, 1] = False

    out = hull_attn(q, k, v, mode=mode, attention_mask=mask)
    out_ref = _reference_hull_attn(q, k, v, mode=mode, attention_mask=mask)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("mode", ["topk1", "topk4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_topk_forces_cute_forward(mode, dtype):
    atol, rtol = CUTE_TOLERANCES[dtype]
    q = torch.randn((1, 2, 16, 2), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out_ref = _reference_hull_attn(q, k, v, mode=mode)

    with mock.patch("quack.hull_attn._reference_topk_attention", side_effect=AssertionError):
        out = hull_attn(q, k, v, mode=mode)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("mode", ["topk1", "topk4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_topk_forces_cute_key_padding_forward(mode, dtype):
    atol, rtol = CUTE_TOLERANCES[dtype]
    q = torch.randn((2, 2, 16, 2), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    key_padding_mask = torch.tensor(
        [[True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False],
         [True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False]],
        device="cuda",
        dtype=torch.bool,
    )
    out_ref = _reference_hull_attn(q, k, v, mode=mode, attention_mask=key_padding_mask[:, None, None, :])

    with mock.patch("quack.hull_attn._reference_topk_attention", side_effect=AssertionError):
        out = hull_attn(q, k, v, mode=mode, key_padding_mask=key_padding_mask)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


def test_hull_attn_topk_rejects_autograd():
    device = "cuda"
    q = torch.randn((1, 2, 4, 2), device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    with pytest.raises(NotImplementedError, match="forward-only"):
        hull_attn(q, k, v, mode="topk4")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_width_matched_benchmark_shapes_smoke(dtype):
    sdpa_shape = (64, 8, 2048, 16)
    hull_shape = (64, 64, 2048, 2)
    device = "cuda"

    q_sdpa = torch.randn(sdpa_shape, device=device, dtype=dtype).contiguous()
    k_sdpa = torch.randn_like(q_sdpa)
    v_sdpa = torch.randn_like(q_sdpa)
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=None, dropout_p=0.0, is_causal=False
    )

    q_hull = torch.randn(hull_shape, device=device, dtype=dtype).contiguous()
    k_hull = torch.randn_like(q_hull)
    v_hull = torch.randn_like(q_hull)
    seq_lens = torch.full((hull_shape[0],), hull_shape[2], device=device, dtype=torch.int32)
    out_hull = hull_attn(q_hull, k_hull, v_hull, mode="full", seq_lens=seq_lens)

    assert out_sdpa.shape == sdpa_shape
    assert out_hull.shape == hull_shape
    assert torch.isfinite(out_sdpa).all()
    assert torch.isfinite(out_hull).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_full_forces_cute_forward(dtype):
    atol, rtol = CUTE_TOLERANCES[dtype]
    q = torch.randn((1, 2, 32, 2), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out_ref = _reference_hull_attn(q, k, v, mode="full")

    with mock.patch("quack.hull_attn._streaming_full_attention_forward", side_effect=AssertionError):
        out = hull_attn(q, k, v, mode="full")

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mask_kind", ["bool", "float"])
def test_hull_attn_full_forces_cute_masked_forward(dtype, mask_kind):
    atol, rtol = CUTE_TOLERANCES[dtype]
    q = torch.randn((1, 2, 16, 2), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    if mask_kind == "bool":
        mask = torch.ones((1, 1, 16, 16), device="cuda", dtype=torch.bool)
        mask[..., 3, 7] = False
        mask[..., 4, :3] = False
    else:
        mask = torch.zeros((1, 1, 16, 16), device="cuda", dtype=torch.float32)
        mask[..., 3, 7] = -1e4
        mask[..., 4, :3] = -1e4
    out_ref = _reference_hull_attn(q, k, v, mode="full", attention_mask=mask)

    with mock.patch("quack.hull_attn._streaming_full_attention_forward", side_effect=AssertionError):
        out = hull_attn(q, k, v, mode="full", attention_mask=mask)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_full_forces_cute_key_padding_forward(dtype):
    atol, rtol = CUTE_TOLERANCES[dtype]
    q = torch.randn((2, 2, 16, 2), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    key_padding_mask = torch.tensor(
        [[True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False],
         [True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False]],
        device="cuda",
        dtype=torch.bool,
    )
    out_ref = _reference_hull_attn(q, k, v, mode="full", attention_mask=key_padding_mask[:, None, None, :])

    with mock.patch("quack.hull_attn._streaming_full_attention_forward", side_effect=AssertionError):
        out = hull_attn(q, k, v, mode="full", key_padding_mask=key_padding_mask)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_cute_custom_ops_match_reference(dtype):
    atol, rtol = CUTE_TOLERANCES[dtype]
    torch.manual_seed(0)
    q = torch.randn((2, 24, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 24, 2), device="cuda", dtype=dtype)
    v = torch.randn((2, 24, 2), device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(2.0)
    out = torch.empty_like(q)
    lse = torch.empty((2, 24), device="cuda", dtype=torch.float32)

    _hull_attn_full_fwd(q, k, v, scale, out, lse)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1)
    out_ref = torch.matmul(probs, v.float()).to(dtype=dtype)
    lse_ref = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=1e-4, rtol=1e-4)

    grad_out = torch.randn_like(out)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _hull_attn_full_bwd_dq(q, k, v, out, lse, grad_out, scale, dq)
    _hull_attn_full_bwd_dkdv(q, k, v, out, lse, grad_out, scale, dk, dv)

    dp = torch.matmul(grad_out.float(), v.float().transpose(-1, -2))
    delta = (grad_out.float() * out_ref.float()).sum(dim=-1, keepdim=True)
    ds = probs * (dp - delta)
    dq_ref = torch.matmul(ds, k.float()) * scale
    dk_ref = torch.matmul(ds.transpose(-1, -2), q.float()) * scale
    dv_ref = torch.matmul(probs.transpose(-1, -2), grad_out.float())
    torch.testing.assert_close(dq, dq_ref.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(dk, dk_ref.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(dv, dv_ref.to(dtype), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mask_kind", ["bool", "float"])
def test_hull_attn_cute_masked_custom_ops_match_reference(dtype, mask_kind):
    atol, rtol = CUTE_TOLERANCES[dtype]
    torch.manual_seed(1)
    q = torch.randn((2, 16, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 16, 2), device="cuda", dtype=dtype)
    v = torch.randn((2, 16, 2), device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(2.0)
    if mask_kind == "bool":
        mask_in = torch.ones((2, 16, 16), device="cuda", dtype=torch.bool)
        mask_in[:, 2, 5] = False
        mask_in[:, 6, :4] = False
        mask = torch.where(mask_in, 0.0, -torch.inf).to(torch.float32).unsqueeze(1)
    else:
        mask = torch.zeros((2, 1, 16, 16), device="cuda", dtype=torch.float32)
        mask[:, 0, 2, 5] = -1e4
        mask[:, 0, 6, :4] = -1e4
    out = torch.empty_like(q)
    lse = torch.empty((2, 16), device="cuda", dtype=torch.float32)

    _hull_attn_full_fwd_masked(q, k, v, mask, 1, scale, out, lse)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale + mask[:, 0]
    probs = torch.softmax(scores, dim=-1)
    out_ref = torch.matmul(probs, v.float()).to(dtype=dtype)
    lse_ref = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=1e-4, rtol=1e-4)

    grad_out = torch.randn_like(out)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _hull_attn_full_bwd_dq_masked(q, k, v, mask, 1, out, lse, grad_out, scale, dq)
    _hull_attn_full_bwd_dkdv_masked(q, k, v, mask, 1, out, lse, grad_out, scale, dk, dv)

    dp = torch.matmul(grad_out.float(), v.float().transpose(-1, -2))
    delta = (grad_out.float() * out_ref.float()).sum(dim=-1, keepdim=True)
    ds = probs * (dp - delta)
    dq_ref = torch.matmul(ds, k.float()) * scale
    dk_ref = torch.matmul(ds.transpose(-1, -2), q.float()) * scale
    dv_ref = torch.matmul(probs.transpose(-1, -2), grad_out.float())
    torch.testing.assert_close(dq, dq_ref.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(dk, dk_ref.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(dv, dv_ref.to(dtype), atol=atol, rtol=rtol)
