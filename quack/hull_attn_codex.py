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


DEFAULT_Q_BLOCK = 64
DEFAULT_K_BLOCK = 64


class HullAttnCodexFullForwardCute:
    def __init__(self, dtype, seqlen_q: int, seqlen_k: int, q_block: int, k_block: int):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.q_block = q_block
        self.k_block = k_block
        self.num_load_iters = math.ceil(k_block / q_block)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mSeqLens: cute.Tensor | None,
        mKeyMask: cute.Tensor | None,
        num_heads: Int32,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mQ,
            mK,
            mV,
            mSeqLens,
            mKeyMask,
            num_heads,
            mO,
            mLSE,
            scale,
        ).launch(
            grid=[cute.ceil_div(self.seqlen_q, self.q_block), mQ.shape[0], 1],
            block=[self.q_block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mSeqLens: cute.Tensor | None,
        mKeyMask: cute.Tensor | None,
        num_heads: Int32,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile_idx, bh_idx, _ = cute.arch.block_idx()
        row = q_tile_idx * self.q_block + tidx
        active_q = row < self.seqlen_q and bh_idx < mQ.shape[0]
        batch_idx = bh_idx // num_heads

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            mK.element_type,
            cute.make_ordered_layout((self.k_block, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(
            mV.element_type,
            cute.make_ordered_layout((self.k_block, 2), order=(1, 0)),
            byte_alignment=16,
        )

        q0 = Float32.zero
        q1 = Float32.zero
        running_max = -Float32.inf
        running_sum = Float32.zero
        acc0 = Float32.zero
        acc1 = Float32.zero

        if active_q:
            q0 = Float32(mQ[bh_idx, row, 0])
            q1 = Float32(mQ[bh_idx, row, 1])

        if const_expr(mSeqLens is not None):
            k_end = cutlass.min(Int32(mSeqLens[batch_idx]), Int32(self.seqlen_k))
        else:
            k_end = Int32(self.seqlen_k)
        num_k_tiles = cute.ceil_div(k_end, self.k_block)

        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            k_start = k_tile * self.k_block
            tile_k = cutlass.min(k_end - k_start, self.k_block)

            for load_iter in cutlass.range_constexpr(self.num_load_iters):
                load_row = tidx + load_iter * self.q_block
                if load_row < tile_k:
                    k_idx = k_start + load_row
                    sK[load_row, 0] = mK[bh_idx, k_idx, 0]
                    sK[load_row, 1] = mK[bh_idx, k_idx, 1]
                    sV[load_row, 0] = mV[bh_idx, k_idx, 0]
                    sV[load_row, 1] = mV[bh_idx, k_idx, 1]
            cute.arch.barrier()

            if active_q:
                for i in cutlass.range(self.k_block, unroll_full=True):
                    if i < tile_k:
                        key_is_valid = True
                        if const_expr(mKeyMask is not None):
                            key_is_valid = mKeyMask[batch_idx, k_start + i] != 0
                        if key_is_valid:
                            k0 = Float32(sK[i, 0])
                            k1 = Float32(sK[i, 1])
                            v0 = Float32(sV[i, 0])
                            v1 = Float32(sV[i, 1])
                            score = (q0 * k0 + q1 * k1) * scale
                            new_max = cute.arch.fmax(running_max, score)
                            alpha = cute.math.exp(running_max - new_max, fastmath=True)
                            p = cute.math.exp(score - new_max, fastmath=True)
                            running_sum = running_sum * alpha + p
                            acc0 = acc0 * alpha + p * v0
                            acc1 = acc1 * alpha + p * v1
                            running_max = new_max

            cute.arch.barrier()

        if active_q:
            if running_sum > Float32.zero:
                inv_sum = Float32(1.0) / running_sum
                mO[bh_idx, row, 0] = mO.element_type(acc0 * inv_sum)
                mO[bh_idx, row, 1] = mO.element_type(acc1 * inv_sum)
                mLSE[bh_idx, row] = running_max + cute.math.log(running_sum, fastmath=True)
            else:
                mO[bh_idx, row, 0] = mO.element_type(Float32.zero)
                mO[bh_idx, row, 1] = mO.element_type(Float32.zero)
                mLSE[bh_idx, row] = -Float32.inf


class HullAttnCodexFullBackwardDQCute:
    def __init__(self, dtype, seqlen_q: int, seqlen_k: int, q_block: int, k_block: int):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.q_block = q_block
        self.k_block = k_block
        self.num_load_iters = math.ceil(k_block / q_block)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdOut: cute.Tensor,
        mDelta: cute.Tensor,
        mSeqLens: cute.Tensor | None,
        mKeyMask: cute.Tensor | None,
        num_heads: Int32,
        mdQ: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mQ,
            mK,
            mV,
            mLSE,
            mdOut,
            mDelta,
            mSeqLens,
            mKeyMask,
            num_heads,
            mdQ,
            scale,
        ).launch(
            grid=[cute.ceil_div(self.seqlen_q, self.q_block), mQ.shape[0], 1],
            block=[self.q_block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdOut: cute.Tensor,
        mDelta: cute.Tensor,
        mSeqLens: cute.Tensor | None,
        mKeyMask: cute.Tensor | None,
        num_heads: Int32,
        mdQ: cute.Tensor,
        scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q_tile_idx, bh_idx, _ = cute.arch.block_idx()
        row = q_tile_idx * self.q_block + tidx
        active_q = row < self.seqlen_q and bh_idx < mQ.shape[0]
        batch_idx = bh_idx // num_heads

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            mK.element_type,
            cute.make_ordered_layout((self.k_block, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(
            mV.element_type,
            cute.make_ordered_layout((self.k_block, 2), order=(1, 0)),
            byte_alignment=16,
        )

        q0 = Float32.zero
        q1 = Float32.zero
        do0 = Float32.zero
        do1 = Float32.zero
        delta = Float32.zero
        lse = -Float32.inf
        dq0 = Float32.zero
        dq1 = Float32.zero

        if active_q:
            q0 = Float32(mQ[bh_idx, row, 0])
            q1 = Float32(mQ[bh_idx, row, 1])
            do0 = Float32(mdOut[bh_idx, row, 0])
            do1 = Float32(mdOut[bh_idx, row, 1])
            delta = Float32(mDelta[bh_idx, row])
            lse = Float32(mLSE[bh_idx, row])

        if const_expr(mSeqLens is not None):
            k_end = cutlass.min(Int32(mSeqLens[batch_idx]), Int32(self.seqlen_k))
        else:
            k_end = Int32(self.seqlen_k)
        num_k_tiles = cute.ceil_div(k_end, self.k_block)

        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            k_start = k_tile * self.k_block
            tile_k = cutlass.min(k_end - k_start, self.k_block)

            for load_iter in cutlass.range_constexpr(self.num_load_iters):
                load_row = tidx + load_iter * self.q_block
                if load_row < tile_k:
                    k_idx = k_start + load_row
                    sK[load_row, 0] = mK[bh_idx, k_idx, 0]
                    sK[load_row, 1] = mK[bh_idx, k_idx, 1]
                    sV[load_row, 0] = mV[bh_idx, k_idx, 0]
                    sV[load_row, 1] = mV[bh_idx, k_idx, 1]
            cute.arch.barrier()

            if active_q:
                if lse == lse and lse != -Float32.inf:
                    for i in cutlass.range(self.k_block, unroll_full=True):
                        if i < tile_k:
                            key_is_valid = True
                            if const_expr(mKeyMask is not None):
                                key_is_valid = mKeyMask[batch_idx, k_start + i] != 0
                            if key_is_valid:
                                k0 = Float32(sK[i, 0])
                                k1 = Float32(sK[i, 1])
                                v0 = Float32(sV[i, 0])
                                v1 = Float32(sV[i, 1])
                                score = (q0 * k0 + q1 * k1) * scale
                                p = cute.math.exp(score - lse, fastmath=True)
                                dp = do0 * v0 + do1 * v1
                                ds = p * (dp - delta)
                                dq0 += ds * k0
                                dq1 += ds * k1

            cute.arch.barrier()

        if active_q:
            if lse == lse and lse != -Float32.inf:
                mdQ[bh_idx, row, 0] = mdQ.element_type(dq0 * scale)
                mdQ[bh_idx, row, 1] = mdQ.element_type(dq1 * scale)
            else:
                mdQ[bh_idx, row, 0] = mdQ.element_type(Float32.zero)
                mdQ[bh_idx, row, 1] = mdQ.element_type(Float32.zero)


class HullAttnCodexFullBackwardDKDVCute:
    def __init__(self, dtype, seqlen_q: int, seqlen_k: int, q_block: int, k_block: int):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.q_block = q_block
        self.k_block = k_block
        self.num_q_load_iters = math.ceil(q_block / k_block)
        self.num_q_tiles = math.ceil(seqlen_q / q_block)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdOut: cute.Tensor,
        mDelta: cute.Tensor,
        mSeqLens: cute.Tensor | None,
        mKeyMask: cute.Tensor | None,
        num_heads: Int32,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mQ,
            mK,
            mV,
            mLSE,
            mdOut,
            mDelta,
            mSeqLens,
            mKeyMask,
            num_heads,
            mdK,
            mdV,
            scale,
        ).launch(
            grid=[cute.ceil_div(self.seqlen_k, self.k_block), mK.shape[0], 1],
            block=[self.k_block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdOut: cute.Tensor,
        mDelta: cute.Tensor,
        mSeqLens: cute.Tensor | None,
        mKeyMask: cute.Tensor | None,
        num_heads: Int32,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        k_tile_idx, bh_idx, _ = cute.arch.block_idx()
        col = k_tile_idx * self.k_block + tidx
        active_k = col < self.seqlen_k and bh_idx < mK.shape[0]
        batch_idx = bh_idx // num_heads

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            mQ.element_type,
            cute.make_ordered_layout((self.q_block, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sdO = smem.allocate_tensor(
            mdOut.element_type,
            cute.make_ordered_layout((self.q_block, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sLSE = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout((self.q_block, 1), order=(1, 0)),
            byte_alignment=16,
        )
        sDelta = smem.allocate_tensor(
            Float32,
            cute.make_ordered_layout((self.q_block, 1), order=(1, 0)),
            byte_alignment=16,
        )

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

        for q_tile in cutlass.range(self.num_q_tiles, unroll=1):
            q_start = q_tile * self.q_block
            tile_q = cutlass.min(Int32(self.seqlen_q) - q_start, Int32(self.q_block))

            for load_iter in cutlass.range_constexpr(self.num_q_load_iters):
                load_row = tidx + load_iter * self.k_block
                if load_row < tile_q:
                    q_idx = q_start + load_row
                    sQ[load_row, 0] = mQ[bh_idx, q_idx, 0]
                    sQ[load_row, 1] = mQ[bh_idx, q_idx, 1]
                    sdO[load_row, 0] = mdOut[bh_idx, q_idx, 0]
                    sdO[load_row, 1] = mdOut[bh_idx, q_idx, 1]
                    sLSE[load_row, 0] = Float32(mLSE[bh_idx, q_idx])
                    sDelta[load_row, 0] = Float32(mDelta[bh_idx, q_idx])
            cute.arch.barrier()

            if k_valid:
                for i in cutlass.range(self.q_block, unroll_full=True):
                    if i < tile_q:
                        row_lse = sLSE[i, 0]
                        if row_lse == row_lse and row_lse != -Float32.inf:
                            q0 = Float32(sQ[i, 0])
                            q1 = Float32(sQ[i, 1])
                            do0 = Float32(sdO[i, 0])
                            do1 = Float32(sdO[i, 1])
                            delta = sDelta[i, 0]
                            score = (q0 * k0 + q1 * k1) * scale
                            p = cute.math.exp(score - row_lse, fastmath=True)
                            dp = do0 * v0 + do1 * v1
                            ds = p * (dp - delta)
                            dk0 += ds * q0
                            dk1 += ds * q1
                            dv0 += p * do0
                            dv1 += p * do1

            cute.arch.barrier()

        if active_k:
            if k_valid:
                mdK[bh_idx, col, 0] = mdK.element_type(dk0 * scale)
                mdK[bh_idx, col, 1] = mdK.element_type(dk1 * scale)
                mdV[bh_idx, col, 0] = mdV.element_type(dv0)
                mdV[bh_idx, col, 1] = mdV.element_type(dv1)
            else:
                mdK[bh_idx, col, 0] = mdK.element_type(Float32.zero)
                mdK[bh_idx, col, 1] = mdK.element_type(Float32.zero)
                mdV[bh_idx, col, 0] = mdV.element_type(Float32.zero)
                mdV[bh_idx, col, 1] = mdV.element_type(Float32.zero)


@lru_cache(maxsize=None)
def _compile_hull_attn_codex_fwd(
    dtype,
    seqlen_q: int,
    seqlen_k: int,
    use_seq_lens: bool,
    use_key_mask: bool,
    q_block: int,
    k_block: int,
):
    key = (
        "hull_attn_codex_full_fwd",
        dtype,
        seqlen_q,
        seqlen_k,
        use_seq_lens,
        use_key_mask,
        q_block,
        k_block,
    )

    def _compile():
        bh_sym = cute.sym_int()
        batch_sym = cute.sym_int()
        divisibility = math.gcd(128 // dtype.width, 2)
        q_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=divisibility)
        k_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=divisibility)
        v_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=divisibility)
        seq_lens_cute = fake_tensor(Int32, (batch_sym,), divisibility=1) if use_seq_lens else None
        key_mask_cute = (
            fake_tensor(Int32, (batch_sym, seqlen_k), divisibility=1) if use_key_mask else None
        )
        out_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=divisibility)
        lse_cute = fake_tensor(Float32, (bh_sym, seqlen_q), divisibility=1)
        op = HullAttnCodexFullForwardCute(dtype, seqlen_q, seqlen_k, q_block, k_block)
        return cute.compile(
            op,
            q_cute,
            k_cute,
            v_cute,
            seq_lens_cute,
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
def _compile_hull_attn_codex_bwd_dq(
    dtype,
    seqlen_q: int,
    seqlen_k: int,
    use_seq_lens: bool,
    use_key_mask: bool,
    q_block: int,
    k_block: int,
):
    key = (
        "hull_attn_codex_full_bwd_dq",
        dtype,
        seqlen_q,
        seqlen_k,
        use_seq_lens,
        use_key_mask,
        q_block,
        k_block,
    )

    def _compile():
        bh_sym = cute.sym_int()
        batch_sym = cute.sym_int()
        divisibility = math.gcd(128 // dtype.width, 2)
        q_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=divisibility)
        k_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=divisibility)
        v_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=divisibility)
        lse_cute = fake_tensor(Float32, (bh_sym, seqlen_q), divisibility=1)
        dout_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=divisibility)
        delta_cute = fake_tensor(Float32, (bh_sym, seqlen_q), divisibility=1)
        seq_lens_cute = fake_tensor(Int32, (batch_sym,), divisibility=1) if use_seq_lens else None
        key_mask_cute = (
            fake_tensor(Int32, (batch_sym, seqlen_k), divisibility=1) if use_key_mask else None
        )
        dq_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=divisibility)
        op = HullAttnCodexFullBackwardDQCute(dtype, seqlen_q, seqlen_k, q_block, k_block)
        return cute.compile(
            op,
            q_cute,
            k_cute,
            v_cute,
            lse_cute,
            dout_cute,
            delta_cute,
            seq_lens_cute,
            key_mask_cute,
            Int32(1),
            dq_cute,
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


@lru_cache(maxsize=None)
def _compile_hull_attn_codex_bwd_dkdv(
    dtype,
    seqlen_q: int,
    seqlen_k: int,
    use_seq_lens: bool,
    use_key_mask: bool,
    q_block: int,
    k_block: int,
):
    key = (
        "hull_attn_codex_full_bwd_dkdv",
        dtype,
        seqlen_q,
        seqlen_k,
        use_seq_lens,
        use_key_mask,
        q_block,
        k_block,
    )

    def _compile():
        bh_sym = cute.sym_int()
        batch_sym = cute.sym_int()
        divisibility = math.gcd(128 // dtype.width, 2)
        q_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=divisibility)
        k_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=divisibility)
        v_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=divisibility)
        lse_cute = fake_tensor(Float32, (bh_sym, seqlen_q), divisibility=1)
        dout_cute = fake_tensor(dtype, (bh_sym, seqlen_q, 2), divisibility=divisibility)
        delta_cute = fake_tensor(Float32, (bh_sym, seqlen_q), divisibility=1)
        seq_lens_cute = fake_tensor(Int32, (batch_sym,), divisibility=1) if use_seq_lens else None
        key_mask_cute = (
            fake_tensor(Int32, (batch_sym, seqlen_k), divisibility=1) if use_key_mask else None
        )
        dk_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=divisibility)
        dv_cute = fake_tensor(dtype, (bh_sym, seqlen_k, 2), divisibility=divisibility)
        op = HullAttnCodexFullBackwardDKDVCute(dtype, seqlen_q, seqlen_k, q_block, k_block)
        return cute.compile(
            op,
            q_cute,
            k_cute,
            v_cute,
            lse_cute,
            dout_cute,
            delta_cute,
            seq_lens_cute,
            key_mask_cute,
            Int32(1),
            dk_cute,
            dv_cute,
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
):
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if not q.is_cuda:
        raise ValueError("hull_attn_codex expects CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shapes [batch, heads, seq, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("q, k, and v must have the same number of heads")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same key sequence length")
    if q.shape[-1] != 2 or k.shape[-1] != 2 or v.shape[-1] != 2:
        raise ValueError("hull_attn_codex is specialized to head_dim=2")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("hull_attn_codex currently supports float16 and bfloat16 only")
    if attention_mask is not None:
        raise NotImplementedError("dense attention_mask is not implemented in hull_attn_codex")
    if key_padding_mask is not None and seq_lens is not None:
        raise ValueError("Provide only one of key_padding_mask or seq_lens")
    if key_padding_mask is not None and key_padding_mask.shape != (q.shape[0], k.shape[-2]):
        raise ValueError("key_padding_mask must have shape [batch, seqlen_k]")
    if seq_lens is not None and (seq_lens.ndim != 1 or seq_lens.shape[0] != q.shape[0]):
        raise ValueError("seq_lens must have shape [batch]")


def _prefix_seq_lens_from_key_padding_mask(
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if key_padding_mask is None:
        return None
    key_padding_mask = key_padding_mask.to(dtype=torch.bool)
    seq_lens = key_padding_mask.to(dtype=torch.int32).sum(dim=-1).to(dtype=torch.int32)
    positions = torch.arange(key_padding_mask.shape[1], device=key_padding_mask.device)
    prefix_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    return seq_lens if torch.equal(prefix_mask, key_padding_mask) else None


def _prepare_runtime_masks(
    key_padding_mask: torch.Tensor | None,
    seq_lens: torch.Tensor | None,
    seqlen_k: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    seq_lens_i32 = None
    key_mask_i32 = None
    key_padding_mask_bool = None

    if seq_lens is not None:
        seq_lens_i32 = seq_lens.to(device=device, dtype=torch.int32).contiguous()
        positions = torch.arange(seqlen_k, device=device)
        key_padding_mask_bool = (positions.unsqueeze(0) < seq_lens_i32.unsqueeze(1)).contiguous()

    if key_padding_mask is not None:
        key_padding_mask_bool = key_padding_mask.to(device=device, dtype=torch.bool).contiguous()
        prefix_seq_lens = _prefix_seq_lens_from_key_padding_mask(key_padding_mask_bool)
        if prefix_seq_lens is not None:
            seq_lens_i32 = prefix_seq_lens.to(device=device, dtype=torch.int32).contiguous()
        else:
            key_mask_i32 = key_padding_mask_bool.to(dtype=torch.int32).contiguous()

    return seq_lens_i32, key_mask_i32, key_padding_mask_bool


def _full_attention_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    q_block: int = DEFAULT_Q_BLOCK,
    k_block: int = DEFAULT_K_BLOCK,
    seq_lens_i32: torch.Tensor | None = None,
    key_mask_i32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]
    q_flat = q.contiguous().view(batch * heads, seqlen_q, 2)
    k_flat = k.contiguous().view(batch * heads, seqlen_k, 2)
    v_flat = v.contiguous().view(batch * heads, seqlen_k, 2)
    out_flat = torch.empty_like(q_flat)
    lse_flat = torch.empty((batch * heads, seqlen_q), device=q.device, dtype=torch.float32)

    compiled = _compile_hull_attn_codex_fwd(
        torch2cute_dtype_map[q.dtype],
        seqlen_q,
        seqlen_k,
        seq_lens_i32 is not None,
        key_mask_i32 is not None,
        q_block,
        k_block,
    )
    compiled(
        q_flat,
        k_flat,
        v_flat,
        seq_lens_i32,
        key_mask_i32,
        heads,
        out_flat,
        lse_flat,
        scale,
    )

    out = out_flat.view(batch, heads, seqlen_q, 2)
    lse = lse_flat.view(batch, heads, seqlen_q)
    return out, lse


def _full_attention_backward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    delta: torch.Tensor,
    scale: float,
    q_block: int = DEFAULT_Q_BLOCK,
    k_block: int = DEFAULT_K_BLOCK,
    seq_lens_i32: torch.Tensor | None = None,
    key_mask_i32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]
    q_flat = q.contiguous().view(batch * heads, seqlen_q, 2)
    k_flat = k.contiguous().view(batch * heads, seqlen_k, 2)
    v_flat = v.contiguous().view(batch * heads, seqlen_k, 2)
    lse_flat = lse.contiguous().view(batch * heads, seqlen_q)
    dout_flat = grad_out.contiguous().view(batch * heads, seqlen_q, 2)
    delta_flat = delta.contiguous().view(batch * heads, seqlen_q)
    dq_flat = torch.empty_like(q_flat)
    dk_flat = torch.empty_like(k_flat)
    dv_flat = torch.empty_like(v_flat)

    compiled_dq = _compile_hull_attn_codex_bwd_dq(
        torch2cute_dtype_map[q.dtype],
        seqlen_q,
        seqlen_k,
        seq_lens_i32 is not None,
        key_mask_i32 is not None,
        q_block,
        k_block,
    )
    compiled_dkdv = _compile_hull_attn_codex_bwd_dkdv(
        torch2cute_dtype_map[q.dtype],
        seqlen_q,
        seqlen_k,
        seq_lens_i32 is not None,
        key_mask_i32 is not None,
        q_block,
        k_block,
    )

    compiled_dq(
        q_flat,
        k_flat,
        v_flat,
        lse_flat,
        dout_flat,
        delta_flat,
        seq_lens_i32,
        key_mask_i32,
        heads,
        dq_flat,
        scale,
    )
    compiled_dkdv(
        q_flat,
        k_flat,
        v_flat,
        lse_flat,
        dout_flat,
        delta_flat,
        seq_lens_i32,
        key_mask_i32,
        heads,
        dk_flat,
        dv_flat,
        scale,
    )

    dq = dq_flat.view(batch, heads, seqlen_q, 2)
    dk = dk_flat.view(batch, heads, seqlen_k, 2)
    dv = dv_flat.view(batch, heads, seqlen_k, 2)
    return dq, dk, dv


def _full_attention_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    grad_out: torch.Tensor,
    scale: float,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[-2]
    q_block = min(DEFAULT_Q_BLOCK, seqlen_q)
    k_block = min(DEFAULT_K_BLOCK, seqlen_k)

    grad_out_f32 = grad_out.float()
    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()
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
        delta = (grad_chunk * out_chunk).sum(dim=-1, keepdim=True)
        dq_chunk = torch.zeros_like(q_chunk)

        for k_start in range(0, seqlen_k, k_block):
            k_end = min(k_start + k_block, seqlen_k)
            k_chunk = k_f32[:, :, k_start:k_end, :]
            v_chunk = v_f32[:, :, k_start:k_end, :]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scale
            if key_padding_mask is not None:
                scores = scores.masked_fill(~key_padding_mask[:, None, None, k_start:k_end], -torch.inf)

            probs = torch.exp(scores - lse_chunk.unsqueeze(-1))
            if key_padding_mask is not None:
                probs = probs.masked_fill(~key_padding_mask[:, None, None, k_start:k_end], 0.0)
            probs = torch.where(
                torch.isfinite(lse_chunk).unsqueeze(-1),
                probs,
                torch.zeros_like(probs),
            )

            dv[:, :, k_start:k_end, :] += torch.matmul(probs.transpose(-1, -2), grad_chunk)
            dp = torch.matmul(grad_chunk, v_chunk.transpose(-1, -2))
            ds = probs * (dp - delta)
            dq_chunk += torch.matmul(ds, k_chunk)
            dk[:, :, k_start:k_end, :] += torch.matmul(ds.transpose(-1, -2), q_chunk)

        dq[:, :, q_start:q_end, :] = dq_chunk * scale

    dk *= scale
    return dq.to(dtype=q.dtype), dk.to(dtype=k.dtype), dv.to(dtype=v.dtype)


class HullAttnCodexFullFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, seq_lens, key_padding_mask, q_block, k_block):
        seq_lens_i32, key_mask_i32, key_padding_mask_bool = _prepare_runtime_masks(
            key_padding_mask,
            seq_lens,
            k.shape[-2],
            q.device,
        )
        out, lse = _full_attention_forward_impl(
            q,
            k,
            v,
            scale=scale,
            q_block=q_block,
            k_block=k_block,
            seq_lens_i32=seq_lens_i32,
            key_mask_i32=key_mask_i32,
        )
        saved_tensors = [q, k, v, out, lse]
        if seq_lens_i32 is not None:
            saved_tensors.append(seq_lens_i32)
        if key_mask_i32 is not None:
            saved_tensors.append(key_mask_i32)
        ctx.save_for_backward(*saved_tensors)
        ctx.has_seq_lens = seq_lens_i32 is not None
        ctx.has_key_mask = key_mask_i32 is not None
        ctx.scale = scale
        ctx.q_block = q_block
        ctx.k_block = k_block
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        q, k, v, out, lse = saved[:5]
        offset = 5
        seq_lens_i32 = saved[offset] if ctx.has_seq_lens else None
        offset += 1 if ctx.has_seq_lens else 0
        key_mask_i32 = saved[offset] if ctx.has_key_mask else None
        delta = (grad_out.float() * out.float()).sum(dim=-1)
        dq, dk, dv = _full_attention_backward_impl(
            q,
            k,
            v,
            lse,
            grad_out,
            delta,
            scale=ctx.scale,
            q_block=ctx.q_block,
            k_block=ctx.k_block,
            seq_lens_i32=seq_lens_i32,
            key_mask_i32=key_mask_i32,
        )
        return dq, dk, dv, None, None, None, None, None


def hull_attn_codex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    attention_mask: torch.Tensor | None = None,
    key_padding_mask: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    q_block: int = DEFAULT_Q_BLOCK,
    k_block: int = DEFAULT_K_BLOCK,
    return_lse: bool = False,
):
    _check_inputs(q, k, v, attention_mask, key_padding_mask, seq_lens)
    if q_block <= 0 or k_block <= 0:
        raise ValueError("q_block and k_block must be positive")

    scale = (1.0 / math.sqrt(2.0)) if scale is None else float(scale)
    requires_grad = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    if requires_grad and return_lse:
        raise NotImplementedError("return_lse=True is not supported when gradients are required")

    if requires_grad:
        return HullAttnCodexFullFunction.apply(
            q,
            k,
            v,
            scale,
            seq_lens,
            key_padding_mask,
            q_block,
            k_block,
        )

    seq_lens_i32, key_mask_i32, _ = _prepare_runtime_masks(
        key_padding_mask,
        seq_lens,
        k.shape[-2],
        q.device,
    )
    out, lse = _full_attention_forward_impl(
        q,
        k,
        v,
        scale=scale,
        q_block=q_block,
        k_block=k_block,
        seq_lens_i32=seq_lens_i32,
        key_mask_i32=key_mask_i32,
    )
    return (out, lse) if return_lse else out


__all__ = ["hull_attn_codex"]
