import math

import pytest
import torch

from quack.hull_attn3 import hull_attn3


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

TOLERANCES = {
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16: (5e-3, 5e-3),
    torch.float32: (1e-4, 1e-4),
}


def _reference_hull_attn(
    q,
    k,
    v,
    scale=None,
    attention_mask=None,
    key_padding_mask=None,
):
    scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else float(scale)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask.to(device=scores.device), -torch.inf)
        else:
            scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)
    if key_padding_mask is not None:
        scores = scores.masked_fill(~key_padding_mask[:, None, None, :].to(device=scores.device), -torch.inf)

    valid_rows = torch.isfinite(scores).any(dim=-1)
    safe_scores = torch.where(valid_rows.unsqueeze(-1), scores, torch.zeros_like(scores))
    probs = torch.softmax(safe_scores, dim=-1)
    probs = torch.where(valid_rows.unsqueeze(-1), probs, torch.zeros_like(probs))
    out = torch.matmul(probs, v.float()).to(dtype=q.dtype)

    lse = torch.logsumexp(safe_scores, dim=-1)
    lse = torch.where(valid_rows, lse, torch.full_like(lse, -torch.inf))
    return out, lse


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn3_unmasked_flash_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(0)
    q = torch.randn((2, 3, 64, 2), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out, backend = hull_attn3(q, k, v, return_backend=True)
    out_ref, _ = _reference_hull_attn(q, k, v)

    assert backend == "flash"
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn3_seq_lens_forward_backward_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(1)
    q = torch.randn((2, 2, 17, 2), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((2, 2, 23, 2), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    seq_lens = torch.tensor([23, 11], device="cuda", dtype=torch.int32)
    key_padding_mask = torch.arange(k.shape[-2], device="cuda")[None, :] < seq_lens[:, None]

    out = hull_attn3(q, k, v, seq_lens=seq_lens)
    out_ref, _ = _reference_hull_attn(q_ref, k_ref, v_ref, key_padding_mask=key_padding_mask)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


def test_hull_attn3_dense_bool_mask_backward_matches_reference():
    atol, rtol = TOLERANCES[torch.float32]
    torch.manual_seed(2)
    q = torch.randn((2, 2, 13, 2), device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    mask = torch.ones((13, 13), device="cuda", dtype=torch.bool)
    mask[0, -1] = False
    mask[5, :3] = False

    out = hull_attn3(q, k, v, attention_mask=mask)
    out_ref, _ = _reference_hull_attn(q_ref, k_ref, v_ref, attention_mask=mask)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn3_dense_float_mask_fast_path_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(4)
    q = torch.randn((2, 2, 11, 2), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((2, 2, 13, 2), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    mask = torch.zeros((2, 11, 13), device="cuda", dtype=torch.float32)
    mask[:, 0, -1] = -torch.inf
    mask[:, 3, :2] = -torch.inf
    mask[:, 7, 5] = -1e4

    with torch.no_grad():
        _, backend = hull_attn3(
            q.detach(),
            k.detach(),
            v.detach(),
            attention_mask=mask,
            return_backend=True,
        )
    assert backend == "cute_masked"

    out = hull_attn3(q, k, v, attention_mask=mask)
    out_ref, _ = _reference_hull_attn(q_ref, k_ref, v_ref, attention_mask=mask[:, None, :, :])
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn3_all_masked_seq_lens_are_finite(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(3)
    q = torch.randn((2, 2, 9, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 2, 11, 2), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    seq_lens = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    out, lse, backend = hull_attn3(q, k, v, seq_lens=seq_lens, return_lse=True, return_backend=True)
    key_padding_mask = torch.arange(k.shape[-2], device="cuda")[None, :] < seq_lens[:, None]
    out_ref, lse_ref = _reference_hull_attn(q, k, v, key_padding_mask=key_padding_mask)

    assert backend == "codex_seq_lens"
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.isneginf(lse[0]).all()
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=1e-4, rtol=1e-4)
