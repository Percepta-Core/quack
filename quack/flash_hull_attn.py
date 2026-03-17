# Flash Hull Attention — custom dim=2 forward/backward kernels, no padding.
#
# Uses scalar FMAs instead of tensor cores, operating directly on [B*H,S,2]
# tensors without padding to dim=8. This saves 4x memory vs the padded FA4 path.

import math

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from quack.fa4.cute_dsl_utils import to_cute_tensor
from quack.fa4.flash_hull_fwd_sm90 import (
    FlashHullForwardSm90,
    FlashHullBackwardDQSm90,
    FlashHullBackwardDKDVSm90,
)

# ---- dtype mapping -----------------------------------------------------------
_torch2cute = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}

# ---- compile cache -----------------------------------------------------------
_compile_cache: dict = {}

HEAD_DIM = 2


def _get_compiled(name, dtype, builder):
    key = (name, dtype)
    if key not in _compile_cache:
        _compile_cache[key] = builder()
    return _compile_cache[key]


def _to_3d(t):
    """Flatten [B, H, S, D] → [B*H, S, D] contiguously."""
    B, H, S, D = t.shape
    return t.reshape(B * H, S, D)


def _ct3(t, assumed_align=16):
    """Create CuTe tensor from a 3D [BH, S, D] tensor."""
    return to_cute_tensor(t, assumed_align=assumed_align)


def _compile_fwd(dtype, q3, k3, v3, out3, lse2, scale):
    op = FlashHullForwardSm90(dtype)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        op, _ct3(q3), _ct3(k3), _ct3(v3), _ct3(out3),
        _ct3(lse2, assumed_align=4), scale, stream,
        options="--enable-tvm-ffi",
    )


def _compile_bwd_dq(dtype, q3, k3, v3, out3, lse2, dout3, dq3, scale):
    op = FlashHullBackwardDQSm90(dtype)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        op, _ct3(q3), _ct3(k3), _ct3(v3), _ct3(out3),
        _ct3(lse2, assumed_align=4), _ct3(dout3), _ct3(dq3),
        scale, stream, options="--enable-tvm-ffi",
    )


def _compile_bwd_dkdv(dtype, q3, k3, v3, out3, lse2, dout3, dk3, dv3, scale):
    op = FlashHullBackwardDKDVSm90(dtype)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        op, _ct3(q3), _ct3(k3), _ct3(v3), _ct3(out3),
        _ct3(lse2, assumed_align=4), _ct3(dout3), _ct3(dk3), _ct3(dv3),
        scale, stream, options="--enable-tvm-ffi",
    )


class FlashHullAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        B, H, S, _ = q.shape
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = torch.empty_like(q)
        lse = torch.empty(B, H, S, dtype=torch.float32, device=q.device)
        dtype = _torch2cute[q.dtype]

        q3, k3, v3, out3 = _to_3d(q), _to_3d(k), _to_3d(v), _to_3d(out)
        lse2 = lse.reshape(B * H, S)

        fn = _get_compiled(
            "fwd", dtype, lambda: _compile_fwd(dtype, q3, k3, v3, out3, lse2, scale)
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn(q3, k3, v3, out3, lse2, scale, stream)

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        scale = ctx.scale
        grad_out = grad_out.contiguous()
        dtype = _torch2cute[q.dtype]
        B, H, S, _ = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        q3, k3, v3, out3 = _to_3d(q), _to_3d(k), _to_3d(v), _to_3d(out)
        dout3, dq3, dk3, dv3 = _to_3d(grad_out), _to_3d(dq), _to_3d(dk), _to_3d(dv)
        lse2 = lse.reshape(B * H, S)

        fn_dq = _get_compiled(
            "bwd_dq", dtype,
            lambda: _compile_bwd_dq(dtype, q3, k3, v3, out3, lse2, dout3, dq3, scale),
        )
        fn_dkdv = _get_compiled(
            "bwd_dkdv", dtype,
            lambda: _compile_bwd_dkdv(
                dtype, q3, k3, v3, out3, lse2, dout3, dk3, dv3, scale
            ),
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        fn_dq(q3, k3, v3, out3, lse2, dout3, dq3, scale, stream)
        fn_dkdv(q3, k3, v3, out3, lse2, dout3, dk3, dv3, scale, stream)
        return dq, dk, dv, None


def flash_hull_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Drop-in replacement for ``hull_attn(mode="full")`` using a custom dim=2 kernel.

    Supports autograd (backward pass).

    Args:
        q, k, v: ``[batch, heads, seqlen, 2]``  (hull_attn layout)
        scale: softmax scale, defaults to ``1/sqrt(2)``

    Returns:
        ``[batch, heads, seqlen, 2]``
    """
    if q.shape[-1] != HEAD_DIM:
        raise ValueError("flash_hull_attn is specialized to head_dim=2")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("flash_hull_attn requires float16 or bfloat16")
    scale = scale if scale is not None else 1.0 / math.sqrt(HEAD_DIM)
    return FlashHullAttnFunction.apply(q, k, v, scale)
