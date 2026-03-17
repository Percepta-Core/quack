"""WGMMA Hull Attention — grouped-head tensor-core kernel for head_dim=2.

Uses the grouped-head trick to pack 8 independent dim-2 heads into synthetic
width-16 operands for SM90 WGMMA.  Score generation (QK) uses tensor cores;
softmax and P@V accumulation remain scalar.

Forward-only, dense full attention, bf16/fp16, num_heads % 8 == 0.
"""

import math
import operator
from functools import lru_cache, partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch

from quack import copy_utils, layout_utils, sm90_utils
from quack.cache_utils import compile_and_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.fa4 import pipeline
from quack.fa4.softmax import Softmax
from quack.fa4 import utils as fa4_utils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADS_PER_GROUP = 8
HEAD_DIM = 2
D_PACKED = HEADS_PER_GROUP * HEAD_DIM  # 16
Q_ROWS_PER_CTA = 8
M_WGMMA = Q_ROWS_PER_CTA * HEADS_PER_GROUP  # 64
TILE_N = 64
NUM_STAGES = 2
WARP_SIZE = 32
NUM_THREADS = 256  # 2 warpgroups


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
class WGMMAHullForwardSm90:
    """WGMMA hull attention forward kernel (head_dim=2, SM90)."""

    def __init__(self, dtype, seqlen_q: int, seqlen_k: int, debug_n_index: bool = False):
        self.dtype = dtype
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.num_k_tiles = math.ceil(seqlen_k / TILE_N)
        self.debug_n_index = debug_n_index

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,        # [B*H, S_q, 2]
        mK_pack: cute.Tensor,   # [B*H_groups, S_k, 16]
        mV_pack: cute.Tensor,   # [B*H_groups, S_k, 16]  (raw gmem for scalar P@V)
        mO: cute.Tensor,        # [B*H, S_q, 2]
        mLSE: cute.Tensor,      # [B*H, S_q]
        scale: Float32,
        stream: cuda.CUstream,
    ):
        # Smem layouts: swizzled for WGMMA
        sQ_sel_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (M_WGMMA, D_PACKED)
        )
        sK_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (TILE_N, D_PACKED), NUM_STAGES
        )

        # Shared storage
        sQ_sel_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sQ_sel_layout)], 1024
        ]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)], 1024
        ]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, NUM_STAGES * 2]

        @cute.struct
        class SharedStorage:
            mbar_ptr_K: mbar_ptr_K_struct
            sQ_sel: sQ_sel_struct
            sK: sK_struct

        # TMA for K
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        self.tma_copy_bytes_K = cute.size_in_bytes(
            mK_pack.element_type, cute.select(sK_layout, mode=[0, 1])
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK_pack,
            cute.select(sK_layout, mode=[0, 1]),
            (TILE_N, D_PACKED),
            1,
        )

        # Tiled MMA: both Q_sel and K from smem
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, TILE_N),
        )

        self.SharedStorage = SharedStorage

        num_q_tiles = cute.ceil_div(self.seqlen_q, Q_ROWS_PER_CTA)
        self.kernel(
            mQ, tma_tensor_K, mV_pack, mO, mLSE,
            tma_atom_K,
            scale,
            sQ_sel_layout, sK_layout,
            tiled_mma_qk,
        ).launch(
            grid=[num_q_tiles, mK_pack.shape[0], 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,        # TMA tensor
        mV: cute.Tensor,        # raw gmem [B*H_groups, S_k, 16]
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        scale: Float32,
        sQ_sel_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_K)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.SharedStorage)

        sQ_sel = storage.sQ_sel.get_tensor(
            sQ_sel_layout.outer, swizzle=sQ_sel_layout.inner
        )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)

        # Pipeline for K
        pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, 4
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=NUM_STAGES,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes_K,
            defer_sync=False,
        )

        q_tile_idx, bh_group_idx, _ = cute.arch.block_idx()
        bh_base = bh_group_idx * HEADS_PER_GROUP

        # ---- Construct Q_sel in smem (all threads) ----
        self.construct_q_sel(mQ, sQ_sel, tidx, bh_base, q_tile_idx)
        cute.arch.sync_threads()

        # ---- Branch ----
        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(56)
            self.load(mK, sK, tma_atom_K, pipeline_k, bh_group_idx)
        else:
            cute.arch.setmaxregister_increase(256)
            self.mma(
                mV, mO, mLSE, sQ_sel, sK,
                pipeline_k, tiled_mma_qk, scale,
                tidx - 128, bh_group_idx, bh_base, q_tile_idx,
            )

    @cute.jit
    def construct_q_sel(
        self,
        mQ: cute.Tensor,
        sQ_sel: cute.Tensor,
        tidx: Int32,
        bh_base: Int32,
        q_tile_idx: Int32,
    ):
        """Fill Q_sel [64, 16] in smem with selector-packed Q values.

        Row r → (q_idx=r//8, h_idx=r%8).  Only cols 2*h_idx, 2*h_idx+1 nonzero.
        256 threads × 4 elements each = 1024 = 64×16.
        """
        q_start = q_tile_idx * Q_ROWS_PER_CTA
        for i in cutlass.range_constexpr(4):
            flat = tidx * 4 + i
            row = flat // D_PACKED
            col = flat % D_PACKED
            q_idx = row // HEADS_PER_GROUP
            h_idx = row % HEADS_PER_GROUP
            global_q = q_start + q_idx
            global_bh = bh_base + h_idx
            # Default zero; overwrite in-place if this is a nonzero position.
            # Direct smem writes are side effects and DO take effect inside branches.
            sQ_sel[row, col] = self.dtype(Float32.zero)
            if col == 2 * h_idx:
                if global_q < self.seqlen_q:
                    sQ_sel[row, col] = mQ[global_bh, global_q, 0]
            if col == 2 * h_idx + 1:
                if global_q < self.seqlen_q:
                    sQ_sel[row, col] = mQ[global_bh, global_q, 1]

    @cute.jit
    def load(
        self,
        mK: cute.Tensor,
        sK: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        bh_group_idx: Int32,
    ):
        """Producer: TMA load K tiles."""
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            mK_cur = mK[bh_group_idx, None, None]
            gK = cute.local_tile(mK_cur, (TILE_N, D_PACKED), (None, 0))
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), gK, sK
            )
            load_K = copy_utils.tma_producer_copy_fn(load_K, pipeline_k)

            kv_producer_state = pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Producer, NUM_STAGES
            )
            for n_tile in cutlass.range(self.num_k_tiles, unroll=1):
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_tile, producer_state=kv_producer_state)
                kv_producer_state.advance()

    @cute.jit
    def mma(
        self,
        mV: cute.Tensor,       # raw gmem [B*H_groups, S_k, 16]
        mO: cute.Tensor,       # [B*H, S_q, 2]
        mLSE: cute.Tensor,     # [B*H, S_q]
        sQ_sel: cute.Tensor,   # [64, 16] smem
        sK: cute.Tensor,       # [TILE_N, 16, stages] smem
        pipeline_k: cutlass.pipeline.PipelineAsync,
        tiled_mma_qk: cute.TiledMma,
        scale: Float32,
        tidx: Int32,
        bh_group_idx: Int32,
        bh_base: Int32,
        q_tile_idx: Int32,
    ):
        # Partition WGMMA fragments
        wg_mma_qk = tiled_mma_qk.get_slice(tidx)
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (M_WGMMA, TILE_N, D_PACKED), sQ_sel, sK
        )

        # Identity tensor for logical index mapping
        cS = cute.make_identity_tensor((M_WGMMA, TILE_N))
        tScS = wg_mma_qk.partition_C(cS)
        tScS_mn = layout_utils.reshape_acc_to_mn(tScS)

        # Determine num_rows from accumulator shape
        acc_shape = tiled_mma_qk.partition_shape_C((M_WGMMA, TILE_N))
        num_rows = acc_shape[0][0] * acc_shape[1]

        # Softmax
        LOG2E = Float32(math.log2(math.e))
        softmax_scale_log2 = scale * LOG2E
        softmax = Softmax.create(softmax_scale_log2, num_rows=num_rows, softmax_scale=scale)
        softmax.reset()

        # Output accumulators: 2 fp32 per logical row
        acc_o0 = cute.make_rmem_tensor(num_rows, Float32)
        acc_o1 = cute.make_rmem_tensor(num_rows, Float32)
        acc_o0.fill(0.0)
        acc_o1.fill(0.0)

        kv_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, NUM_STAGES
        )
        q_start = q_tile_idx * Q_ROWS_PER_CTA

        # ---- First K-tile (is_first=True) ----
        pipeline_k.consumer_wait(kv_consumer_state)
        acc_S = sm90_utils.gemm_zero_init(
            tiled_mma_qk, (M_WGMMA, TILE_N), tSrQ, tSrK,
            B_idx=kv_consumer_state.index, wg_wait=0,
        )
        pipeline_k.consumer_release(kv_consumer_state)

        row_scale = softmax.online_softmax(acc_S, is_first=True)
        self.scalar_pv(
            acc_S, tScS, acc_o0, acc_o1, mV, bh_group_idx,
            0 * TILE_N, num_rows,
        )
        kv_consumer_state.advance()

        # ---- Remaining K-tiles (is_first=False) ----
        for n_tile_minus1 in cutlass.range(self.num_k_tiles - 1, unroll=1):
            n_tile = n_tile_minus1 + 1
            pipeline_k.consumer_wait(kv_consumer_state)
            acc_S = sm90_utils.gemm_zero_init(
                tiled_mma_qk, (M_WGMMA, TILE_N), tSrQ, tSrK,
                B_idx=kv_consumer_state.index, wg_wait=0,
            )
            pipeline_k.consumer_release(kv_consumer_state)

            row_scale = softmax.online_softmax(acc_S, is_first=False)
            # Rescale existing output accumulators
            for r in cutlass.range(num_rows, unroll_full=True):
                acc_o0[r] = acc_o0[r] * row_scale[r]
                acc_o1[r] = acc_o1[r] * row_scale[r]

            self.scalar_pv(
                acc_S, tScS_mn, acc_o0, acc_o1, mV, bh_group_idx,
                n_tile * TILE_N, num_rows,
            )
            kv_consumer_state.advance()

        # ---- Finalize ----
        final_scale = softmax.finalize()
        for r in cutlass.range(num_rows, unroll_full=True):
            acc_o0[r] = acc_o0[r] * final_scale[r]
            acc_o1[r] = acc_o1[r] * final_scale[r]

        # Quad reduction (4 threads share each logical row)
        acc_o0.store(fa4_utils.warp_reduce(acc_o0.load(), operator.add, width=4))
        acc_o1.store(fa4_utils.warp_reduce(acc_o1.load(), operator.add, width=4))

        # Write output (only first lane in each quad)
        n_elems_total = cutlass.const_expr(cute.size(tScS.shape))
        id_m_out = cute.make_rmem_tensor(acc_S.shape, Float32)
        for ii in cutlass.range(n_elems_total, unroll_full=True):
            id_m_out[ii] = Float32(tScS[ii][0])
        id_m_out_mn = layout_utils.reshape_acc_to_mn(id_m_out)

        lane = cute.arch.lane_idx()
        if lane % 4 == 0:
            for r in cutlass.range(num_rows, unroll_full=True):
                logical_m = Int32(id_m_out_mn[r, 0])
                q_idx = logical_m // HEADS_PER_GROUP
                h_idx = logical_m % HEADS_PER_GROUP
                global_q = q_start + q_idx
                global_bh = bh_base + h_idx
                if global_q < self.seqlen_q:
                    mO[global_bh, global_q, 0] = mO.element_type(acc_o0[r])
                    mO[global_bh, global_q, 1] = mO.element_type(acc_o1[r])
                    mLSE[global_bh, global_q] = softmax.row_sum[r]

    @cute.jit
    def scalar_pv(
        self,
        acc_S: cute.Tensor,
        tScS: cute.Tensor,
        acc_o0: cute.Tensor,
        acc_o1: cute.Tensor,
        mV: cute.Tensor,
        bh_group_idx: Int32,
        k_tile_start: Int32,
        num_rows: cutlass.Constexpr[int],
    ):
        """Scalar P@V: accumulate attention_probs × V into output.

        Key insight: softmax writes exp values through reshape_acc_to_mn(acc_S),
        so we must read probabilities through the same M,N view. To get matching
        identity coordinates, we build rmem coordinate tensors with the SAME
        default layout as acc_S (both from make_rmem_tensor), then reshape
        identically.
        """
        # Build coordinate rmem tensors with SAME dtype as acc_S (Float32) to ensure
        # reshape_acc_to_mn produces identical layout.
        n_elems = cutlass.const_expr(cute.size(acc_S.shape))
        id_m = cute.make_rmem_tensor(acc_S.shape, Float32)
        id_n = cute.make_rmem_tensor(acc_S.shape, Float32)
        for i in cutlass.range(n_elems, unroll_full=True):
            id_m[i] = Float32(tScS[i][0])
            id_n[i] = Float32(tScS[i][1])

        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        id_m_mn = layout_utils.reshape_acc_to_mn(id_m)
        id_n_mn = layout_utils.reshape_acc_to_mn(id_n)
        n_cols = cute.size(acc_S_mn, mode=[1])

        for r in cutlass.range(num_rows, unroll_full=True):
            for c in cutlass.range(n_cols, unroll_full=True):
                p_val = acc_S_mn[r, c]
                logical_m = Int32(id_m_mn[r, c])
                logical_n = Int32(id_n_mn[r, c])
                h_idx = logical_m % Int32(HEADS_PER_GROUP)
                k_pos = k_tile_start + logical_n
                v0 = Float32(mV[bh_group_idx, k_pos, Int32(2) * h_idx])
                v1 = Float32(mV[bh_group_idx, k_pos, Int32(2) * h_idx + Int32(1)])
                acc_o0[r] = acc_o0[r] + p_val * v0
                acc_o1[r] = acc_o1[r] + p_val * v1


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _compile_wgmma_fwd(dtype, seqlen_q, seqlen_k):
    key = ("hull_attn_wgmma_fwd", dtype, seqlen_q, seqlen_k)

    def _compile():
        bh = cute.sym_int()
        bh_groups = cute.sym_int()
        div = math.gcd(128 // dtype.width, 2)
        q_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        k_ct = fake_tensor(dtype, (bh_groups, seqlen_k, D_PACKED), divisibility=D_PACKED)
        v_ct = fake_tensor(dtype, (bh_groups, seqlen_k, D_PACKED), divisibility=D_PACKED)
        o_ct = fake_tensor(dtype, (bh, seqlen_q, 2), divisibility=div)
        lse_ct = fake_tensor(Float32, (bh, seqlen_q), divisibility=1)
        op = WGMMAHullForwardSm90(dtype, seqlen_q, seqlen_k)
        return cute.compile(
            op, q_ct, k_ct, v_ct, o_ct, lse_ct,
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    return compile_and_cache(key, _compile)


# ---------------------------------------------------------------------------
# Forward dispatch
# ---------------------------------------------------------------------------
def _forward(q, k, v, scale):
    """Internal forward: pack inputs and run kernel."""
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[2]
    assert heads % HEADS_PER_GROUP == 0
    num_groups = heads // HEADS_PER_GROUP

    # Pack K and V: [B, H, S, 2] → [B, H//8, S, 16]
    k_pack = k.reshape(batch, num_groups, HEADS_PER_GROUP, seqlen_k, HEAD_DIM)
    k_pack = k_pack.permute(0, 1, 3, 2, 4).reshape(batch * num_groups, seqlen_k, D_PACKED)
    k_pack = k_pack.contiguous()
    v_pack = v.reshape(batch, num_groups, HEADS_PER_GROUP, seqlen_k, HEAD_DIM)
    v_pack = v_pack.permute(0, 1, 3, 2, 4).reshape(batch * num_groups, seqlen_k, D_PACKED)
    v_pack = v_pack.contiguous()

    q_flat = q.contiguous().reshape(batch * heads, seqlen_q, 2)
    out_flat = torch.empty_like(q_flat)
    lse_flat = torch.empty(batch * heads, seqlen_q, device=q.device, dtype=torch.float32)

    dtype = torch2cute_dtype_map[q.dtype]
    compiled = _compile_wgmma_fwd(dtype, seqlen_q, seqlen_k)
    compiled(q_flat, k_pack, v_pack, out_flat, lse_flat, scale)
    return out_flat.view_as(q), lse_flat.view(batch, heads, seqlen_q)


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------
class HullAttnWGMMAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        out, lse = _forward(q, k, v, scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("WGMMA hull attention backward not yet implemented")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def hull_attn_wgmma(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """WGMMA hull attention for head_dim=2, num_heads divisible by 8.

    Args:
        q, k, v: ``[batch, heads, seq, 2]``
        scale: softmax scale, defaults to ``1/sqrt(2)``

    Returns:
        ``[batch, heads, seq, 2]``
    """
    if q.shape[-1] != HEAD_DIM:
        raise ValueError("hull_attn_wgmma is specialized to head_dim=2")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("hull_attn_wgmma requires float16 or bfloat16")
    if q.shape[1] % HEADS_PER_GROUP != 0:
        raise ValueError(f"num_heads must be divisible by {HEADS_PER_GROUP}")
    if q.shape[2] % TILE_N != 0:
        raise ValueError(f"seqlen must be divisible by {TILE_N} (TILE_N)")
    scale = scale if scale is not None else 1.0 / math.sqrt(HEAD_DIM)
    return HullAttnWGMMAFunction.apply(q, k, v, scale)
