import math

import pytest
import torch

from quack.hull_attn_codex import hull_attn_codex


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

TOLERANCES = {
    torch.bfloat16: (1.5e-2, 1.5e-2),
    torch.float16: (2e-3, 2e-3),
}


def _reference_hull_attn(q, k, v, scale=None, attention_mask=None):
    scale = (1.0 / math.sqrt(q.shape[-1])) if scale is None else float(scale)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, -torch.inf)
        else:
            scores = scores + attention_mask.float()
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(dtype=q.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_codex_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(0)
    q = torch.randn((2, 3, 37, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 3, 53, 2), device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    out = hull_attn_codex(q, k, v)
    out_ref = _reference_hull_attn(q, k, v, scale=1.0 / math.sqrt(2.0))
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_codex_seq_lens_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(1)
    q = torch.randn((2, 2, 41, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 2, 61, 2), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    seq_lens = torch.tensor([61, 37], device="cuda", dtype=torch.int32)
    dense_mask = (torch.arange(k.shape[-2], device="cuda")[None, :] < seq_lens[:, None])[
        :, None, None, :
    ]

    out = hull_attn_codex(q, k, v, seq_lens=seq_lens)
    out_ref = _reference_hull_attn(q, k, v, scale=1.0 / math.sqrt(2.0), attention_mask=dense_mask)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_codex_key_padding_mask_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(2)
    q = torch.randn((2, 2, 29, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 2, 47, 2), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    key_padding_mask = torch.ones((2, 47), device="cuda", dtype=torch.bool)
    key_padding_mask[0, 5] = False
    key_padding_mask[1, 11:17] = False
    key_padding_mask[1, 39:] = False

    out = hull_attn_codex(q, k, v, key_padding_mask=key_padding_mask)
    out_ref = _reference_hull_attn(
        q, k, v, scale=1.0 / math.sqrt(2.0), attention_mask=key_padding_mask[:, None, None, :]
    )
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_codex_prefix_key_padding_mask_routes_to_seq_lens(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(3)
    q = torch.randn((2, 2, 31, 2), device="cuda", dtype=dtype)
    k = torch.randn((2, 2, 59, 2), device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    seq_lens = torch.tensor([59, 43], device="cuda", dtype=torch.int32)
    key_padding_mask = torch.arange(k.shape[-2], device="cuda")[None, :] < seq_lens[:, None]

    out = hull_attn_codex(q, k, v, key_padding_mask=key_padding_mask)
    out_ref = _reference_hull_attn(
        q, k, v, scale=1.0 / math.sqrt(2.0), attention_mask=key_padding_mask[:, None, None, :]
    )
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_codex_backward_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(4)
    q = torch.randn((2, 2, 13, 2), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((2, 2, 17, 2), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = hull_attn_codex(q, k, v)
    out_ref = _reference_hull_attn(q_ref, k_ref, v_ref, scale=1.0 / math.sqrt(2.0))
    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_codex_backward_seq_lens_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(5)
    q = torch.randn((2, 2, 11, 2), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((2, 2, 19, 2), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    seq_lens = torch.tensor([19, 13], device="cuda", dtype=torch.int32)
    attention_mask = (torch.arange(k.shape[-2], device="cuda")[None, :] < seq_lens[:, None])[
        :, None, None, :
    ]

    out = hull_attn_codex(q, k, v, seq_lens=seq_lens)
    out_ref = _reference_hull_attn(
        q_ref, k_ref, v_ref, scale=1.0 / math.sqrt(2.0), attention_mask=attention_mask
    )
    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hull_attn_codex_backward_key_padding_mask_matches_reference(dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.manual_seed(6)
    q = torch.randn((2, 2, 11, 2), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((2, 2, 19, 2), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    key_padding_mask = torch.ones((2, 19), device="cuda", dtype=torch.bool)
    key_padding_mask[0, 3] = False
    key_padding_mask[1, 8:11] = False
    key_padding_mask[1, 17:] = False

    out = hull_attn_codex(q, k, v, key_padding_mask=key_padding_mask)
    out_ref = _reference_hull_attn(
        q_ref, k_ref, v_ref, scale=1.0 / math.sqrt(2.0), attention_mask=key_padding_mask[:, None, None, :]
    )
    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad, q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=atol, rtol=rtol)
