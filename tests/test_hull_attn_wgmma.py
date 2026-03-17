"""Tests for WGMMA hull attention (grouped-head trick)."""

import math
import pytest
import torch
import torch.nn.functional as F

from quack.hull_attn_wgmma import hull_attn_wgmma, TILE_N, HEADS_PER_GROUP


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("batch,heads,seqlen", [
    (1, 8, 64),
    (2, 16, 128),
])
def test_hull_attn_wgmma_forward(batch, heads, seqlen, dtype):
    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(2)
    q = torch.randn(batch, heads, seqlen, 2, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = hull_attn_wgmma(q, k, v, scale=scale)
    ref = F.scaled_dot_product_attention(q, k, v, scale=scale)

    diff = (out.float() - ref.float()).abs()
    print(f"shape={list(q.shape)} max_diff={diff.max():.6f} mean_diff={diff.mean():.6f}")
    assert diff.max() < 0.05, f"Max diff too large: {diff.max()}"


if __name__ == "__main__":
    test_hull_attn_wgmma_forward(1, 8, 64, torch.bfloat16)
    print("PASSED")
