# Custom dim=2 attention kernel — scalar FMAs, no tensor cores, no padding.
#
# Uses flattened [B*H, S, 2] tensors for simpler address computation.
# Forward: 128 threads/CTA, each handles 1 Q row, cooperative K/V smem load.
# Backward dQ: same structure — iterate K tiles, accumulate dQ per thread.
# Backward dK/dV: 128 threads/CTA per K tile, iterate Q tiles.

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

TILE_M = 128  # rows per CTA (= num threads)
TILE_N = 64   # K/V tile size for forward and backward


class FlashHullForwardSm90:
    """Scalar online-softmax attention kernel for head_dim=2 on SM90."""

    def __init__(self, dtype):
        self.dtype = dtype

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,    # (B*H, S, 2)
        mK: cute.Tensor,    # (B*H, S, 2)
        mV: cute.Tensor,    # (B*H, S, 2)
        mO: cute.Tensor,    # (B*H, S, 2)
        mLSE: cute.Tensor,  # (B*H, S)
        scale: Float32,
        stream: cuda.CUstream,
    ):
        num_bh = cute.size(mQ.shape[0])
        seqlen_q = cute.size(mQ.shape[1])
        num_m_blocks = cute.ceil_div(seqlen_q, TILE_M)
        self.kernel(mQ, mK, mV, mO, mLSE, scale).launch(
            grid=[num_m_blocks, num_bh, 1],
            block=[TILE_M, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
               mO: cute.Tensor, mLSE: cute.Tensor, scale: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, bh_idx, _ = cute.arch.block_idx()
        q_row = m_block * TILE_M + tidx
        seqlen_q = cute.size(mQ.shape[1])
        seqlen_k = cute.size(mK.shape[1])
        active_q = q_row < seqlen_q

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            mK.element_type, cute.make_ordered_layout((TILE_N, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(
            mV.element_type, cute.make_ordered_layout((TILE_N, 2), order=(1, 0)),
            byte_alignment=16,
        )

        q0 = Float32.zero
        q1 = Float32.zero
        if active_q:
            q0 = Float32(mQ[bh_idx, q_row, 0])
            q1 = Float32(mQ[bh_idx, q_row, 1])

        running_max = -Float32.inf
        running_sum = Float32.zero
        acc0 = Float32.zero
        acc1 = Float32.zero

        num_k_tiles = cute.ceil_div(seqlen_k, TILE_N)
        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            k_start = k_tile * TILE_N
            tile_k = cutlass.min(seqlen_k - k_start, TILE_N)
            if tidx < tile_k:
                k_idx = k_start + tidx
                sK[tidx, 0] = mK[bh_idx, k_idx, 0]
                sK[tidx, 1] = mK[bh_idx, k_idx, 1]
                sV[tidx, 0] = mV[bh_idx, k_idx, 0]
                sV[tidx, 1] = mV[bh_idx, k_idx, 1]
            cute.arch.barrier()
            if active_q:
                for i in cutlass.range_constexpr(TILE_N):
                    if i < tile_k:
                        k0 = Float32(sK[i, 0])
                        k1 = Float32(sK[i, 1])
                        v0 = Float32(sV[i, 0])
                        v1 = Float32(sV[i, 1])
                        s = (q0 * k0 + q1 * k1) * scale
                        new_max = cute.arch.fmax(running_max, s)
                        alpha = cute.math.exp(running_max - new_max, fastmath=True)
                        p = cute.math.exp(s - new_max, fastmath=True)
                        running_sum = running_sum * alpha + p
                        acc0 = acc0 * alpha + p * v0
                        acc1 = acc1 * alpha + p * v1
                        running_max = new_max
            cute.arch.barrier()

        if active_q:
            if running_sum > Float32.zero:
                inv_sum = Float32(1.0) / running_sum
                mO[bh_idx, q_row, 0] = mO.element_type(acc0 * inv_sum)
                mO[bh_idx, q_row, 1] = mO.element_type(acc1 * inv_sum)
                mLSE[bh_idx, q_row] = (
                    running_max + cute.math.log(running_sum, fastmath=True)
                )
            else:
                mO[bh_idx, q_row, 0] = mO.element_type(Float32.zero)
                mO[bh_idx, q_row, 1] = mO.element_type(Float32.zero)
                mLSE[bh_idx, q_row] = -Float32.inf


class FlashHullBackwardDQSm90:
    """Backward dQ: 128 threads/CTA, each thread computes dQ for one Q row."""

    def __init__(self, dtype):
        self.dtype = dtype

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
                 mO: cute.Tensor, mLSE: cute.Tensor, mdOut: cute.Tensor,
                 mdQ: cute.Tensor, scale: Float32, stream: cuda.CUstream):
        num_bh = cute.size(mQ.shape[0])
        seqlen_q = cute.size(mQ.shape[1])
        self.kernel(mQ, mK, mV, mO, mLSE, mdOut, mdQ, scale).launch(
            grid=[cute.ceil_div(seqlen_q, TILE_M), num_bh, 1],
            block=[TILE_M, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
               mO: cute.Tensor, mLSE: cute.Tensor, mdOut: cute.Tensor,
               mdQ: cute.Tensor, scale: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, bh_idx, _ = cute.arch.block_idx()
        q_row = m_block * TILE_M + tidx
        seqlen_q = cute.size(mQ.shape[1])
        seqlen_k = cute.size(mK.shape[1])
        active_q = q_row < seqlen_q

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            mK.element_type, cute.make_ordered_layout((TILE_N, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sV = smem.allocate_tensor(
            mV.element_type, cute.make_ordered_layout((TILE_N, 2), order=(1, 0)),
            byte_alignment=16,
        )

        q0 = Float32.zero
        q1 = Float32.zero
        do0 = Float32.zero
        do1 = Float32.zero
        lse = Float32.zero
        delta = Float32.zero
        if active_q:
            q0 = Float32(mQ[bh_idx, q_row, 0])
            q1 = Float32(mQ[bh_idx, q_row, 1])
            do0 = Float32(mdOut[bh_idx, q_row, 0])
            do1 = Float32(mdOut[bh_idx, q_row, 1])
            out0 = Float32(mO[bh_idx, q_row, 0])
            out1 = Float32(mO[bh_idx, q_row, 1])
            lse = Float32(mLSE[bh_idx, q_row])
            delta = do0 * out0 + do1 * out1

        dq0 = Float32.zero
        dq1 = Float32.zero

        num_k_tiles = cute.ceil_div(seqlen_k, TILE_N)
        for k_tile in cutlass.range(num_k_tiles, unroll=1):
            k_start = k_tile * TILE_N
            tile_k = cutlass.min(seqlen_k - k_start, TILE_N)
            if tidx < tile_k:
                k_idx = k_start + tidx
                sK[tidx, 0] = mK[bh_idx, k_idx, 0]
                sK[tidx, 1] = mK[bh_idx, k_idx, 1]
                sV[tidx, 0] = mV[bh_idx, k_idx, 0]
                sV[tidx, 1] = mV[bh_idx, k_idx, 1]
            cute.arch.barrier()
            if active_q:
                for i in cutlass.range_constexpr(TILE_N):
                    if i < tile_k:
                        k0 = Float32(sK[i, 0])
                        k1 = Float32(sK[i, 1])
                        v0 = Float32(sV[i, 0])
                        v1 = Float32(sV[i, 1])
                        prob = cute.math.exp(
                            (q0 * k0 + q1 * k1) * scale - lse, fastmath=True
                        )
                        dp = do0 * v0 + do1 * v1
                        ds = prob * (dp - delta)
                        dq0 += ds * k0
                        dq1 += ds * k1
            cute.arch.barrier()

        if active_q:
            mdQ[bh_idx, q_row, 0] = mdQ.element_type(dq0 * scale)
            mdQ[bh_idx, q_row, 1] = mdQ.element_type(dq1 * scale)


class FlashHullBackwardDKDVSm90:
    """Backward dK/dV: 128 threads/CTA, each thread computes dK/dV for one K row."""

    def __init__(self, dtype):
        self.dtype = dtype

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
                 mO: cute.Tensor, mLSE: cute.Tensor, mdOut: cute.Tensor,
                 mdK: cute.Tensor, mdV: cute.Tensor, scale: Float32,
                 stream: cuda.CUstream):
        num_bh = cute.size(mK.shape[0])
        seqlen_k = cute.size(mK.shape[1])
        self.kernel(mQ, mK, mV, mO, mLSE, mdOut, mdK, mdV, scale).launch(
            grid=[cute.ceil_div(seqlen_k, TILE_M), num_bh, 1],
            block=[TILE_M, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
               mO: cute.Tensor, mLSE: cute.Tensor, mdOut: cute.Tensor,
               mdK: cute.Tensor, mdV: cute.Tensor, scale: Float32):
        tidx, _, _ = cute.arch.thread_idx()
        k_block, bh_idx, _ = cute.arch.block_idx()
        k_row = k_block * TILE_M + tidx
        seqlen_q = cute.size(mQ.shape[1])
        seqlen_k = cute.size(mK.shape[1])
        active_k = k_row < seqlen_k

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            mQ.element_type, cute.make_ordered_layout((TILE_N, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sdO = smem.allocate_tensor(
            mQ.element_type, cute.make_ordered_layout((TILE_N, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sOut = smem.allocate_tensor(
            mQ.element_type, cute.make_ordered_layout((TILE_N, 2), order=(1, 0)),
            byte_alignment=16,
        )
        sLSE = smem.allocate_tensor(
            Float32, cute.make_ordered_layout((TILE_N,), order=(0,)),
            byte_alignment=16,
        )

        k0 = Float32.zero
        k1 = Float32.zero
        v0 = Float32.zero
        v1 = Float32.zero
        if active_k:
            k0 = Float32(mK[bh_idx, k_row, 0])
            k1 = Float32(mK[bh_idx, k_row, 1])
            v0 = Float32(mV[bh_idx, k_row, 0])
            v1 = Float32(mV[bh_idx, k_row, 1])

        dk0 = Float32.zero
        dk1 = Float32.zero
        dv0 = Float32.zero
        dv1 = Float32.zero

        num_q_tiles = cute.ceil_div(seqlen_q, TILE_N)
        for q_tile in cutlass.range(num_q_tiles, unroll=1):
            q_start = q_tile * TILE_N
            tile_q = cutlass.min(seqlen_q - q_start, TILE_N)
            if tidx < tile_q:
                q_idx = q_start + tidx
                sQ[tidx, 0] = mQ[bh_idx, q_idx, 0]
                sQ[tidx, 1] = mQ[bh_idx, q_idx, 1]
                sdO[tidx, 0] = mdOut[bh_idx, q_idx, 0]
                sdO[tidx, 1] = mdOut[bh_idx, q_idx, 1]
                sOut[tidx, 0] = mO[bh_idx, q_idx, 0]
                sOut[tidx, 1] = mO[bh_idx, q_idx, 1]
                sLSE[tidx] = mLSE[bh_idx, q_idx]
            cute.arch.barrier()
            if active_k:
                for i in cutlass.range_constexpr(TILE_N):
                    if i < tile_q:
                        sq0 = Float32(sQ[i, 0])
                        sq1 = Float32(sQ[i, 1])
                        sdo0 = Float32(sdO[i, 0])
                        sdo1 = Float32(sdO[i, 1])
                        sout0 = Float32(sOut[i, 0])
                        sout1 = Float32(sOut[i, 1])
                        slse = Float32(sLSE[i])
                        prob = cute.math.exp(
                            (sq0 * k0 + sq1 * k1) * scale - slse, fastmath=True
                        )
                        delta = sdo0 * sout0 + sdo1 * sout1
                        dp = sdo0 * v0 + sdo1 * v1
                        ds = prob * (dp - delta)
                        dk0 += ds * sq0
                        dk1 += ds * sq1
                        dv0 += prob * sdo0
                        dv1 += prob * sdo1
            cute.arch.barrier()

        if active_k:
            mdK[bh_idx, k_row, 0] = mdK.element_type(dk0 * scale)
            mdK[bh_idx, k_row, 1] = mdK.element_type(dk1 * scale)
            mdV[bh_idx, k_row, 0] = mdV.element_type(dv0)
            mdV[bh_idx, k_row, 1] = mdV.element_type(dv1)
