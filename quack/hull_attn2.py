"""Hull Attention v2 — warp-shuffle kernel for head_dim=2.

Forward and backward kernels where K/V (or Q/dO) values are broadcast across
warp lanes via ``__shfl_sync`` instead of shared memory.  Every warp operates
independently — no ``__syncthreads()`` required for the core computation.
"""

import math
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
import torch

from quack.cache_utils import compile_and_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map


WARP_SIZE = 32
DEFAULT_Q_BLOCK = 256
DEFAULT_K_BLOCK = 256


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

class HullAttn2ForwardCute:
    """Warp-shuffle hull attention forward (head_dim=2).

    Each thread owns one query row.  K/V elements are loaded one-per-lane and
    broadcast via warp shuffle — zero shared memory, zero barriers.
    """

    def __init__(self, dtype, seqlen_q: int, seqlen_k: int, q_block: int):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.q_block = q_block

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
        mSeqLens: cute.Tensor | None, mKeyMask: cute.Tensor | None,
        num_heads: Int32, mO: cute.Tensor, mLSE: cute.Tensor,
        scale: Float32, stream: cuda.CUstream,
    ):
        self.kernel(
            mQ, mK, mV, mSeqLens, mKeyMask, num_heads, mO, mLSE, scale,
        ).launch(
            grid=[cute.ceil_div(self.seqlen_q, self.q_block), mQ.shape[0], 1],
            block=[self.q_block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
        mSeqLens: cute.Tensor | None, mKeyMask: cute.Tensor | None,
        num_heads: Int32, mO: cute.Tensor, mLSE: cute.Tensor,
        scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile_idx, bh_idx, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        row = q_tile_idx * self.q_block + tidx
        active_q = row < self.seqlen_q and bh_idx < mQ.shape[0]
        batch_idx = bh_idx // num_heads

        q0 = Float32.zero
        q1 = Float32.zero
        if active_q:
            q0 = Float32(mQ[bh_idx, row, 0])
            q1 = Float32(mQ[bh_idx, row, 1])

        running_max = -Float32.inf
        running_sum = Float32.zero
        acc0 = Float32.zero
        acc1 = Float32.zero

        if const_expr(mSeqLens is not None):
            k_end = cutlass.min(Int32(mSeqLens[batch_idx]), Int32(self.seqlen_k))
        else:
            k_end = Int32(self.seqlen_k)
        num_k_tiles = cute.ceil_div(k_end, WARP_SIZE)

        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            k_base = k_tile * WARP_SIZE
            k_idx = k_base + lane

            k0_reg = Float32.zero
            k1_reg = Float32.zero
            v0_reg = Float32.zero
            v1_reg = Float32.zero
            valid_reg = Int32(0)

            if k_idx < k_end:
                k0_reg = Float32(mK[bh_idx, k_idx, 0])
                k1_reg = Float32(mK[bh_idx, k_idx, 1])
                v0_reg = Float32(mV[bh_idx, k_idx, 0])
                v1_reg = Float32(mV[bh_idx, k_idx, 1])
                valid_reg = Int32(1)
                if const_expr(mKeyMask is not None):
                    if mKeyMask[batch_idx, k_idx] == 0:
                        valid_reg = Int32(0)

            for i in cutlass.range_constexpr(WARP_SIZE):
                ki0 = cute.arch.shuffle_sync(k0_reg, i)
                ki1 = cute.arch.shuffle_sync(k1_reg, i)
                vi0 = cute.arch.shuffle_sync(v0_reg, i)
                vi1 = cute.arch.shuffle_sync(v1_reg, i)
                vi_valid = cute.arch.shuffle_sync(valid_reg, i)

                if active_q:
                    if vi_valid != Int32(0):
                        score = (q0 * ki0 + q1 * ki1) * scale
                        new_max = cute.arch.fmax(running_max, score)
                        alpha = cute.math.exp(running_max - new_max, fastmath=True)
                        p = cute.math.exp(score - new_max, fastmath=True)
                        running_sum = running_sum * alpha + p
                        acc0 = acc0 * alpha + p * vi0
                        acc1 = acc1 * alpha + p * vi1
                        running_max = new_max

        if active_q:
            if running_sum > Float32.zero:
                inv_sum = Float32(1.0) / running_sum
                mO[bh_idx, row, 0] = mO.element_type(acc0 * inv_sum)
                mO[bh_idx, row, 1] = mO.element_type(acc1 * inv_sum)
                mLSE[bh_idx, row] = running_max + cute.math.log(
                    running_sum, fastmath=True
                )
            else:
                mO[bh_idx, row, 0] = mO.element_type(Float32.zero)
                mO[bh_idx, row, 1] = mO.element_type(Float32.zero)
                mLSE[bh_idx, row] = -Float32.inf


# ---------------------------------------------------------------------------
# Backward dQ kernel
# ---------------------------------------------------------------------------

class HullAttn2BackwardDQCute:
    """Compute dQ via warp-shuffle over K tiles (head_dim=2)."""

    def __init__(self, dtype, seqlen_q: int, seqlen_k: int, q_block: int):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.q_block = q_block

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
        mOut: cute.Tensor, mLSE: cute.Tensor, mdOut: cute.Tensor,
        mSeqLens: cute.Tensor | None, mKeyMask: cute.Tensor | None,
        num_heads: Int32, mdQ: cute.Tensor,
        scale: Float32, stream: cuda.CUstream,
    ):
        self.kernel(
            mQ, mK, mV, mOut, mLSE, mdOut, mSeqLens, mKeyMask,
            num_heads, mdQ, scale,
        ).launch(
            grid=[cute.ceil_div(self.seqlen_q, self.q_block), mQ.shape[0], 1],
            block=[self.q_block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
        mOut: cute.Tensor, mLSE: cute.Tensor, mdOut: cute.Tensor,
        mSeqLens: cute.Tensor | None, mKeyMask: cute.Tensor | None,
        num_heads: Int32, mdQ: cute.Tensor,
        scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile_idx, bh_idx, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        row = q_tile_idx * self.q_block + tidx
        active_q = row < self.seqlen_q and bh_idx < mQ.shape[0]
        batch_idx = bh_idx // num_heads

        q0 = Float32.zero
        q1 = Float32.zero
        do0 = Float32.zero
        do1 = Float32.zero
        lse = -Float32.inf
        delta = Float32.zero
        dq0 = Float32.zero
        dq1 = Float32.zero

        if active_q:
            q0 = Float32(mQ[bh_idx, row, 0])
            q1 = Float32(mQ[bh_idx, row, 1])
            do0 = Float32(mdOut[bh_idx, row, 0])
            do1 = Float32(mdOut[bh_idx, row, 1])
            lse = mLSE[bh_idx, row]
            o0 = Float32(mOut[bh_idx, row, 0])
            o1 = Float32(mOut[bh_idx, row, 1])
            delta = do0 * o0 + do1 * o1

        if const_expr(mSeqLens is not None):
            k_end = cutlass.min(Int32(mSeqLens[batch_idx]), Int32(self.seqlen_k))
        else:
            k_end = Int32(self.seqlen_k)
        num_k_tiles = cute.ceil_div(k_end, WARP_SIZE)

        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            k_base = k_tile * WARP_SIZE
            k_idx = k_base + lane

            k0_reg = Float32.zero
            k1_reg = Float32.zero
            v0_reg = Float32.zero
            v1_reg = Float32.zero
            valid_reg = Int32(0)

            if k_idx < k_end:
                k0_reg = Float32(mK[bh_idx, k_idx, 0])
                k1_reg = Float32(mK[bh_idx, k_idx, 1])
                v0_reg = Float32(mV[bh_idx, k_idx, 0])
                v1_reg = Float32(mV[bh_idx, k_idx, 1])
                valid_reg = Int32(1)
                if const_expr(mKeyMask is not None):
                    if mKeyMask[batch_idx, k_idx] == 0:
                        valid_reg = Int32(0)

            for i in cutlass.range_constexpr(WARP_SIZE):
                ki0 = cute.arch.shuffle_sync(k0_reg, i)
                ki1 = cute.arch.shuffle_sync(k1_reg, i)
                vi0 = cute.arch.shuffle_sync(v0_reg, i)
                vi1 = cute.arch.shuffle_sync(v1_reg, i)
                vi_valid = cute.arch.shuffle_sync(valid_reg, i)

                if active_q:
                    if vi_valid != Int32(0):
                        score = (q0 * ki0 + q1 * ki1) * scale
                        p = cute.math.exp(score - lse, fastmath=True)
                        dp = do0 * vi0 + do1 * vi1
                        ds = p * (dp - delta)
                        dq0 = dq0 + ds * ki0
                        dq1 = dq1 + ds * ki1

        if active_q:
            mdQ[bh_idx, row, 0] = mdQ.element_type(dq0 * scale)
            mdQ[bh_idx, row, 1] = mdQ.element_type(dq1 * scale)


# ---------------------------------------------------------------------------
# Backward dK/dV kernel
# ---------------------------------------------------------------------------

class HullAttn2BackwardDKDVCute:
    """Compute dK and dV via warp-shuffle over Q tiles (head_dim=2)."""

    def __init__(self, dtype, seqlen_q: int, seqlen_k: int, k_block: int):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.k_block = k_block

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
        mOut: cute.Tensor, mLSE: cute.Tensor, mdOut: cute.Tensor,
        mSeqLens: cute.Tensor | None, mKeyMask: cute.Tensor | None,
        num_heads: Int32, mdK: cute.Tensor, mdV: cute.Tensor,
        scale: Float32, stream: cuda.CUstream,
    ):
        self.kernel(
            mQ, mK, mV, mOut, mLSE, mdOut, mSeqLens, mKeyMask,
            num_heads, mdK, mdV, scale,
        ).launch(
            grid=[cute.ceil_div(self.seqlen_k, self.k_block), mK.shape[0], 1],
            block=[self.k_block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
        mOut: cute.Tensor, mLSE: cute.Tensor, mdOut: cute.Tensor,
        mSeqLens: cute.Tensor | None, mKeyMask: cute.Tensor | None,
        num_heads: Int32, mdK: cute.Tensor, mdV: cute.Tensor,
        scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        k_tile_idx, bh_idx, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        col = k_tile_idx * self.k_block + tidx
        active_k = col < self.seqlen_k and bh_idx < mK.shape[0]
        batch_idx = bh_idx // num_heads

        k_valid = active_k
        if const_expr(mSeqLens is not None):
            if active_k:
                if col >= Int32(mSeqLens[batch_idx]):
                    k_valid = False
        if const_expr(mKeyMask is not None):
            if k_valid:
                if mKeyMask[batch_idx, col] == 0:
                    k_valid = False

        k0 = Float32.zero
        k1 = Float32.zero
        v0 = Float32.zero
        v1 = Float32.zero
        if k_valid:
            k0 = Float32(mK[bh_idx, col, 0])
            k1 = Float32(mK[bh_idx, col, 1])
            v0 = Float32(mV[bh_idx, col, 0])
            v1 = Float32(mV[bh_idx, col, 1])

        dk0 = Float32.zero
        dk1 = Float32.zero
        dv0 = Float32.zero
        dv1 = Float32.zero

        num_q_tiles = cute.ceil_div(Int32(self.seqlen_q), WARP_SIZE)

        for q_tile in cutlass.range(num_q_tiles, unroll=1):
            q_base = q_tile * WARP_SIZE
            q_idx = q_base + lane

            q0_reg = Float32.zero
            q1_reg = Float32.zero
            do0_reg = Float32.zero
            do1_reg = Float32.zero
            lse_reg = Float32.zero
            delta_reg = Float32.zero
            q_valid_reg = Int32(0)

            if q_idx < self.seqlen_q:
                q0_reg = Float32(mQ[bh_idx, q_idx, 0])
                q1_reg = Float32(mQ[bh_idx, q_idx, 1])
                do0_reg = Float32(mdOut[bh_idx, q_idx, 0])
                do1_reg = Float32(mdOut[bh_idx, q_idx, 1])
                lse_reg = mLSE[bh_idx, q_idx]
                o0_tmp = Float32(mOut[bh_idx, q_idx, 0])
                o1_tmp = Float32(mOut[bh_idx, q_idx, 1])
                delta_reg = do0_reg * o0_tmp + do1_reg * o1_tmp
                q_valid_reg = Int32(1)

            for i in cutlass.range_constexpr(WARP_SIZE):
                qi0 = cute.arch.shuffle_sync(q0_reg, i)
                qi1 = cute.arch.shuffle_sync(q1_reg, i)
                doi0 = cute.arch.shuffle_sync(do0_reg, i)
                doi1 = cute.arch.shuffle_sync(do1_reg, i)
                li = cute.arch.shuffle_sync(lse_reg, i)
                di = cute.arch.shuffle_sync(delta_reg, i)
                q_valid_i = cute.arch.shuffle_sync(q_valid_reg, i)

                if k_valid:
                    if q_valid_i != Int32(0):
                        score = (qi0 * k0 + qi1 * k1) * scale
                        p = cute.math.exp(score - li, fastmath=True)
                        dp = doi0 * v0 + doi1 * v1
                        ds = p * (dp - di)
                        dk0 = dk0 + ds * qi0
                        dk1 = dk1 + ds * qi1
                        dv0 = dv0 + p * doi0
                        dv1 = dv1 + p * doi1

        if k_valid:
            mdK[bh_idx, col, 0] = mdK.element_type(dk0 * scale)
            mdK[bh_idx, col, 1] = mdK.element_type(dk1 * scale)
            mdV[bh_idx, col, 0] = mdV.element_type(dv0)
            mdV[bh_idx, col, 1] = mdV.element_type(dv1)
        elif active_k:
            mdK[bh_idx, col, 0] = mdK.element_type(Float32.zero)
            mdK[bh_idx, col, 1] = mdK.element_type(Float32.zero)
            mdV[bh_idx, col, 0] = mdV.element_type(Float32.zero)
            mdV[bh_idx, col, 1] = mdV.element_type(Float32.zero)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _compile_fwd(dtype, seqlen_q, seqlen_k, use_seq_lens, use_key_mask, q_block):
    key = ("hull_attn2_fwd", dtype, seqlen_q, seqlen_k, use_seq_lens, use_key_mask,
           q_block)

    def _compile():
        bh = cute.sym_int()
        bs = cute.sym_int()
        div = math.gcd(128 // dtype.width, 2)
        q_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        k_ct = fake_tensor(dtype, (bh, seqlen_k, 2), divisibility=div)
        v_ct = fake_tensor(dtype, (bh, seqlen_k, 2), divisibility=div)
        sl = fake_tensor(Int32, (bs,), divisibility=1) if use_seq_lens else None
        km = (fake_tensor(Int32, (bs, seqlen_k), divisibility=1)
              if use_key_mask else None)
        o_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        lse = fake_tensor(Float32, (bh, seqlen_q), divisibility=1)
        op = HullAttn2ForwardCute(dtype, seqlen_q, seqlen_k, q_block)
        return cute.compile(
            op, q_ct, k_ct, v_ct, sl, km, Int32(1), o_ct, lse,
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


@lru_cache(maxsize=None)
def _compile_bwd_dq(dtype, seqlen_q, seqlen_k, use_seq_lens, use_key_mask, q_block):
    key = ("hull_attn2_bwd_dq", dtype, seqlen_q, seqlen_k, use_seq_lens,
           use_key_mask, q_block)

    def _compile():
        bh = cute.sym_int()
        bs = cute.sym_int()
        div = math.gcd(128 // dtype.width, 2)
        q_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        k_ct = fake_tensor(dtype, (bh, seqlen_k, 2), divisibility=div)
        v_ct = fake_tensor(dtype, (bh, seqlen_k, 2), divisibility=div)
        o_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        lse = fake_tensor(Float32, (bh, seqlen_q), divisibility=1)
        do_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        sl = fake_tensor(Int32, (bs,), divisibility=1) if use_seq_lens else None
        km = (fake_tensor(Int32, (bs, seqlen_k), divisibility=1)
              if use_key_mask else None)
        dq_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        op = HullAttn2BackwardDQCute(dtype, seqlen_q, seqlen_k, q_block)
        return cute.compile(
            op, q_ct, k_ct, v_ct, o_ct, lse, do_ct, sl, km,
            Int32(1), dq_ct, Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


@lru_cache(maxsize=None)
def _compile_bwd_dkdv(
    dtype, seqlen_q, seqlen_k, use_seq_lens, use_key_mask, k_block
):
    key = ("hull_attn2_bwd_dkdv", dtype, seqlen_q, seqlen_k, use_seq_lens,
           use_key_mask, k_block)

    def _compile():
        bh = cute.sym_int()
        bs = cute.sym_int()
        div = math.gcd(128 // dtype.width, 2)
        q_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        k_ct = fake_tensor(dtype, (bh, seqlen_k, 2), divisibility=div)
        v_ct = fake_tensor(dtype, (bh, seqlen_k, 2), divisibility=div)
        o_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        lse = fake_tensor(Float32, (bh, seqlen_q), divisibility=1)
        do_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        sl = fake_tensor(Int32, (bs,), divisibility=1) if use_seq_lens else None
        km = (fake_tensor(Int32, (bs, seqlen_k), divisibility=1)
              if use_key_mask else None)
        dk_ct = fake_tensor(dtype, (bh, seqlen_k, 2), divisibility=div)
        dv_ct = fake_tensor(dtype, (bh, seqlen_k, 2), divisibility=div)
        op = HullAttn2BackwardDKDVCute(dtype, seqlen_q, seqlen_k, k_block)
        return cute.compile(
            op, q_ct, k_ct, v_ct, o_ct, lse, do_ct, sl, km,
            Int32(1), dk_ct, dv_ct, Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_inputs(q, k, v):
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if not q.is_cuda:
        raise ValueError("hull_attn2 expects CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, seq, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q, k, and v must have the same number of heads")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same key sequence length")
    if q.shape[-1] != 2 or k.shape[-1] != 2 or v.shape[-1] != 2:
        raise ValueError("hull_attn2 is specialized to head_dim=2")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("hull_attn2 supports float16 and bfloat16 only")


# ---------------------------------------------------------------------------
# Internal forward / backward
# ---------------------------------------------------------------------------

def _forward(q_flat, k_flat, v_flat, out_flat, lse_flat, heads, scale,
             seq_lens_i32, key_mask_i32, q_block):
    dtype = torch2cute_dtype_map[q_flat.dtype]
    seqlen_q, seqlen_k = q_flat.shape[1], k_flat.shape[1]
    compiled = _compile_fwd(
        dtype, seqlen_q, seqlen_k,
        seq_lens_i32 is not None, key_mask_i32 is not None, q_block,
    )
    compiled(q_flat, k_flat, v_flat, seq_lens_i32, key_mask_i32,
             heads, out_flat, lse_flat, scale)


def _backward(q_flat, k_flat, v_flat, out_flat, lse_flat, dout_flat,
              dq_flat, dk_flat, dv_flat, heads, scale,
              seq_lens_i32, key_mask_i32, q_block, k_block):
    dtype = torch2cute_dtype_map[q_flat.dtype]
    seqlen_q, seqlen_k = q_flat.shape[1], k_flat.shape[1]
    use_sl = seq_lens_i32 is not None
    use_km = key_mask_i32 is not None

    compiled_dq = _compile_bwd_dq(
        dtype, seqlen_q, seqlen_k, use_sl, use_km, q_block)
    compiled_dkdv = _compile_bwd_dkdv(
        dtype, seqlen_q, seqlen_k, use_sl, use_km, k_block)

    compiled_dq(q_flat, k_flat, v_flat, out_flat, lse_flat, dout_flat,
                seq_lens_i32, key_mask_i32, heads, dq_flat, scale)
    compiled_dkdv(q_flat, k_flat, v_flat, out_flat, lse_flat, dout_flat,
                  seq_lens_i32, key_mask_i32, heads, dk_flat, dv_flat,
                  scale)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------

class HullAttn2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, seq_lens_i32, key_mask_i32, q_block, k_block):
        batch, heads, seqlen_q, _ = q.shape
        seqlen_k = k.shape[-2]

        q_flat = q.contiguous().view(batch * heads, seqlen_q, 2)
        k_flat = k.contiguous().view(batch * heads, seqlen_k, 2)
        v_flat = v.contiguous().view(batch * heads, seqlen_k, 2)
        out_flat = torch.empty_like(q_flat)
        lse_flat = torch.empty(
            batch * heads, seqlen_q, device=q.device, dtype=torch.float32)

        _forward(q_flat, k_flat, v_flat, out_flat, lse_flat, heads, scale,
                 seq_lens_i32, key_mask_i32, q_block)

        ctx.save_for_backward(q, k, v,
                              out_flat.view_as(q), lse_flat.view(batch, heads, seqlen_q))
        ctx.scale = scale
        ctx.seq_lens_i32 = seq_lens_i32
        ctx.key_mask_i32 = key_mask_i32
        ctx.q_block = q_block
        ctx.k_block = k_block
        return out_flat.view_as(q), lse_flat.view(batch, heads, seqlen_q)

    @staticmethod
    def backward(ctx, grad_out, _grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        batch, heads, seqlen_q, _ = q.shape
        seqlen_k = k.shape[-2]

        q_flat = q.contiguous().view(batch * heads, seqlen_q, 2)
        k_flat = k.contiguous().view(batch * heads, seqlen_k, 2)
        v_flat = v.contiguous().view(batch * heads, seqlen_k, 2)
        out_flat = out.contiguous().view(batch * heads, seqlen_q, 2)
        lse_flat = lse.contiguous().view(batch * heads, seqlen_q)
        dout_flat = grad_out.contiguous().view(batch * heads, seqlen_q, 2)

        dq_flat = torch.zeros_like(q_flat)
        dk_flat = torch.zeros_like(k_flat)
        dv_flat = torch.zeros_like(v_flat)

        _backward(q_flat, k_flat, v_flat, out_flat, lse_flat, dout_flat,
                  dq_flat, dk_flat, dv_flat, heads, ctx.scale,
                  ctx.seq_lens_i32, ctx.key_mask_i32,
                  ctx.q_block, ctx.k_block)

        return (dq_flat.view_as(q), dk_flat.view_as(k), dv_flat.view_as(v),
                None, None, None, None, None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hull_attn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    seq_lens: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    q_block: int = DEFAULT_Q_BLOCK,
    k_block: int = DEFAULT_K_BLOCK,
    return_lse: bool = False,
):
    """Warp-shuffle hull attention (head_dim=2) with forward + backward.

    Args:
        q, k, v: ``[batch, heads, seqlen, 2]``
        scale: softmax scale, defaults to ``1/sqrt(2)``
        seq_lens: ``[batch]`` int32 — valid key lengths per batch element
        key_padding_mask: ``[batch, seqlen_k]`` bool
        q_block: query rows per CTA (must be multiple of 32)
        k_block: key rows per CTA for backward dK/dV (must be multiple of 32)
        return_lse: also return log-sum-exp
    """
    _check_inputs(q, k, v)
    if q_block % WARP_SIZE or k_block % WARP_SIZE:
        raise ValueError("q_block and k_block must be multiples of 32")
    if key_padding_mask is not None and seq_lens is not None:
        raise ValueError("Provide only one of key_padding_mask or seq_lens")

    scale = (1.0 / math.sqrt(2.0)) if scale is None else float(scale)

    seq_lens_i32 = None
    if seq_lens is not None:
        seq_lens_i32 = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()

    key_mask_i32 = None
    if key_padding_mask is not None:
        key_mask_i32 = key_padding_mask.to(
            device=q.device, dtype=torch.int32).contiguous()

    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        out, lse = HullAttn2Function.apply(
            q, k, v, scale, seq_lens_i32, key_mask_i32, q_block, k_block)
    else:
        batch, heads, seqlen_q, _ = q.shape
        seqlen_k = k.shape[-2]
        q_flat = q.contiguous().view(batch * heads, seqlen_q, 2)
        k_flat = k.contiguous().view(batch * heads, seqlen_k, 2)
        v_flat = v.contiguous().view(batch * heads, seqlen_k, 2)
        out_flat = torch.empty_like(q_flat)
        lse_flat = torch.empty(
            batch * heads, seqlen_q, device=q.device, dtype=torch.float32)
        _forward(q_flat, k_flat, v_flat, out_flat, lse_flat, heads, scale,
                 seq_lens_i32, key_mask_i32, q_block)
        out = out_flat.view_as(q)
        lse = lse_flat.view(batch, heads, seqlen_q)

    return (out, lse) if return_lse else out


__all__ = ["hull_attn2"]
