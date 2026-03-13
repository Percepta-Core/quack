# Hull Attention Speedup Plan

## Goal

Improve `hull_attn` for the real workload, with separate success criteria for:

- dense/full attention used in training
- sparse forward-only inference modes `topk1` and `topk4`
- `seq_lens` / key-padding input, not just dense masks
- end-to-end transformer throughput, not just isolated kernel microbenchmarks

## Current Baseline (2026-03-13, H200, bf16)

Width-matched comparison at `[64, 64, 2048, 2]` vs SDPA at `[64, 8, 2048, 16]`:

| Backend | No seq_lens | With seq_lens |
|---------|-------------|---------------|
| PyTorch SDPA | 1.49 ms | 5.77 ms |
| QuACK hull full | 28.2 ms | 34.6 ms |
| **Ratio** | **18.9x slower** | **6.0x slower** |

Sparse forward at `[64, 64, 2048, 2]` (no SDPA equivalent):

| Mode | CuTe kernel | Reference (Python) |
|------|-------------|-------------------|
| topk1 | 12.2 ms | 395.9 ms |
| topk4 | 50.3 ms | 402.7 ms |

Key observations:

- hull full without seq_lens is 19x slower than SDPA — the primary gap to close
- seq_lens actually makes hull *slower* (28.2 → 34.6 ms), confirming it scans all keys and adds mask
  overhead without shortening the K loop
- SDPA with seq_lens is 3.9x slower than without (mask materialization cost)
- topk1 CuTe is already 32x faster than the Python reference
- topk4 CuTe is 8x faster than reference but 4x slower than topk1

## Root Cause Analysis

The 19x gap comes from two structural problems in the full-attention kernel, not from missing
micro-optimizations:

1. **One warp per query row per head.** The grid is `[seqlen_q, batch*heads, 1]` with 32 threads per
   CTA. At `[64, 64, 2048, 2]` that launches 64×64×2048 = 8.4M CTAs, each doing trivial work (two
   FMAs per key element). Launch overhead and scheduling inefficiency dominate.

2. **Two-pass softmax.** The kernel scans all K positions twice — once to find the max score, then
   again to compute exp and accumulate. For `head_dim=2` where compute is negligible, this doubles
   memory traffic. A single-pass online softmax recurrence (already described in `hull_attn.md`)
   would halve the traffic.

The topk4 slowness relative to topk1 has a simpler cause: topk4 rescans all keys 4 times (once per
selection round).

## Phase 1: Multi-Row CTA Mapping

This is the dominant bottleneck. Fix the kernel mapping before anything else.

Replace the one-warp-per-query-row grid with a CTA design that handles multiple query rows and/or
multiple heads per CTA.

Target:

- each CTA processes a tile of query rows (e.g. 16–64 rows) across one or more heads
- load K/V tiles once per CTA, reuse across all query rows in the tile
- shared-memory staging for K/V tiles follows naturally from multi-row design
- reduce CTA count from millions to thousands

This single change addresses both the launch overhead and the K/V reload redundancy. Shared-memory
staging is not a separate decision — it is inherent in multi-row tiling.

Expected impact:

- largest single improvement, likely 5–15x
- enables all subsequent optimizations to operate on a reasonable kernel structure

## Phase 2: Single-Pass Online Softmax

Replace the two-pass max-then-exp scan with the standard online softmax recurrence:

```
for each k:
    score = q · k * scale
    new_max = max(running_max, score)
    correction = exp(running_max - new_max)
    running_sum = running_sum * correction + exp(score - new_max)
    acc = acc * correction + exp(score - new_max) * v
    running_max = new_max
```

This halves the number of K/V loads in the full forward kernel.

Requirements:

- numerically equivalent to the current two-pass implementation
- must produce the same logsumexp state needed by the backward kernel
- the backward kernel already uses recomputation, so it should not need a two-pass structure either

Expected impact:

- ~2x reduction in memory traffic for forward
- meaningful at `head_dim=2` where memory traffic dominates compute

## Phase 3: Sparse Kernel Improvements

### topk1 → argmax specialization

Treat topk1 as a single-pass argmax + value gather rather than a generic top-k selection.

- single pass over K: track best score, best index, and the corresponding V values
- no separate selection rounds
- no softmax (topk1 output is just the value at the argmax position)

Expected impact: modest — topk1 is already 12.2 ms. Mainly simplifies the code.

### topk4 → single-pass blockwise accumulation

Replace the 4-round rescan with a single pass that maintains a sorted top-4 register buffer:

- for each K tile, compare each score against the current 4th-best
- insert into the sorted buffer if better
- after all tiles, softmax over the 4 retained scores and gather V

This changes topk4 from O(4·K) to O(K) key loads.

Expected impact: ~4x improvement (50 ms → ~12 ms), matching topk1 scaling.

### Keep sparse kernels length-aware

After the multi-row CTA change, sparse kernels should accept `k_end` per batch element and loop only
to `k_end` instead of the padded length. This is straightforward once the CTA mapping is fixed.

## Phase 4: seq_lens Early-Exit

Only valuable after Phases 1–2 fix the kernel structure.

Make the K loop bound per-batch-element based on `seq_lens`, so the kernel stops at valid keys
instead of scanning padding and masking it out.

Current behavior: seq_lens adds mask overhead without shortening the loop, making it *slower* than
unmasked (28.2 → 34.6 ms).

Target behavior: seq_lens should be *faster* than full-length when valid lengths are shorter.

Also expose query lengths if the real workload has padded query rows.

## Phase 5: Scheduling and Tuning

Only after the kernel structure from Phases 1–2 is stable.

1. Consider persistent CTA scheduling over `(batch, head, q-block)` if launch overhead is still
   material after multi-row tiling.

2. Tune tile shapes for the target regime `[64, 64, 2048, 2]`:

   - query rows per CTA
   - heads per CTA
   - K tile size
   - number of pipeline stages

3. Run a lightweight `ncu` pass to check occupancy, memory bandwidth utilization, and warp stall
   reasons. This replaces the original Phase 0 full profiling — at `head_dim=2` the bottleneck
   structure is clear enough that a targeted ncu check is more useful than a broad profiling phase.

## Phase 6: Integration Guardrails

1. Add a debug trace mode in `hull_attn`.

For each call, report:

- mode
- training vs forward-only
- dtype
- contiguity
- mask form
- whether `seq_lens` / query lengths are present
- chosen backend path

Transformer regressions are otherwise hard to diagnose.

2. Split kernel variants by dispatch path.

Define separate fast paths for:

- unmasked dense full (no mask branches)
- key-padding / `seq_lens` dense full
- sparse topk1
- sparse topk4
- dense mask fallback

Dispatch should be explicit and traceable. Any unsupported combination should fail or fall back
deliberately, never silently.

3. Keep smoke coverage in CI, but do not add timing assertions there.

CI should verify correctness, dispatch, and fallback behavior. Performance gating should run in a
separate benchmark environment.

## Priority Order

1. multi-row CTA mapping (Phase 1) — 19x gap is dominated by this
2. single-pass online softmax (Phase 2) — halves memory traffic
3. topk4 single-pass redesign (Phase 3) — 4x slower than topk1
4. topk1 argmax specialization (Phase 3) — simplification, modest speedup
5. seq_lens early-exit (Phase 4) — only valuable after kernel restructure
6. persistent scheduling and tuning (Phase 5)
7. query-length support (Phase 4)
8. integration guardrails (Phase 6)

## Success Criteria

- dense full attention materially faster than 28.2 ms at `[64, 64, 2048, 2]` — target under 5 ms
- gap vs SDPA narrows from 19x to under 4x
- seq_lens is faster than full-length when valid lengths are shorter (currently slower)
- topk4 approaches topk1 latency (~12 ms, not 50 ms)
- dense training throughput improves on the real transformer workload
- intended runs do not silently fall back to dense-mask or Python paths
- numerical behavior remains correct for forward and backward on all supported modes
