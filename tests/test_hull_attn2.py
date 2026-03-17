import math

import pytest
import torch

from quack.hull_attn2 import hull_attn2


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TOLERANCES = {
    torch.bfloat16: (1.5e-2, 1.5e-2),
    torch.float16: (2e-3, 2e-3),
}
GRAD_TOLERANCES = {
    torch.bfloat16: (5e-2, 5e-2),
    torch.float16: (1e-2, 1e-2),
}


def _reference(q, k, v, scale=None, attention_mask=None):
    scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else float(scale)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, -torch.inf)
        else:
            scores = scores + attention_mask.float()
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(dtype=q.dtype)


# --- forward correctness ---

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(0)
    q = torch.randn((2, 3, 37, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 3, 53, 2), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    out = hull_attn2(q, k, v)
    ref = _reference(q, k, v, scale=1.0 / math.sqrt(2.0))
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_seq_lens(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(1)
    q = torch.randn((2, 2, 41, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 2, 61, 2), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    seq_lens = torch.tensor([61, 37], device="cuda", dtype=torch.int32)
    dense_mask = (
        torch.arange(k.shape[-2], device="cuda")[None, :] < seq_lens[:, None]
    )[:, None, None, :]
    out = hull_attn2(q, k, v, seq_lens=seq_lens)
    ref = _reference(q, k, v, scale=1.0 / math.sqrt(2.0), attention_mask=dense_mask)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_key_padding_mask(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(2)
    q = torch.randn((2, 2, 29, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 2, 47, 2), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    kpm = torch.ones((2, 47), device="cuda", dtype=torch.bool)
    kpm[0, 5] = False
    kpm[1, 11:17] = False
    kpm[1, 39:] = False
    out = hull_attn2(q, k, v, key_padding_mask=kpm)
    ref = _reference(q, k, v, scale=1.0 / math.sqrt(2.0),
                     attention_mask=kpm[:, None, None, :])
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# --- backward correctness ---

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_backward_matches_reference(dtype):
    atol, rtol = GRAD_TOLERANCES[dtype]
    torch.manual_seed(10)
    q = torch.randn((2, 3, 37, 2), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((2, 3, 53, 2), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    scale = 1.0 / math.sqrt(2.0)

    out = hull_attn2(q, k, v, scale=scale)
    loss = out.float().square().mean()
    loss.backward()
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    ref = _reference(q2, k2, v2, scale=scale)
    ref.float().square().mean().backward()

    torch.testing.assert_close(dq, q2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk, k2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv, v2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_backward_seq_lens(dtype):
    atol, rtol = GRAD_TOLERANCES[dtype]
    torch.manual_seed(11)
    q = torch.randn((2, 2, 41, 2), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((2, 2, 61, 2), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    seq_lens = torch.tensor([61, 37], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(2.0)

    out = hull_attn2(q, k, v, scale=scale, seq_lens=seq_lens)
    out.float().square().mean().backward()
    dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()

    dense_mask = (
        torch.arange(k.shape[-2], device="cuda")[None, :] < seq_lens[:, None]
    )[:, None, None, :]
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    _reference(q2, k2, v2, scale=scale, attention_mask=dense_mask).float().square().mean().backward()

    torch.testing.assert_close(dq, q2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dk, k2.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(dv, v2.grad, atol=atol, rtol=rtol)
