# Hull Attention Speedup Plan

## Goal

Improve `hull_attn` performance for the real workload:

- full attention on `[B, 64, T, 2]`
- `seq_lens` / key-padding input, not dense masks
- forward-only sparse modes `topk1` / `topk4`
- priority on end-to-end transformer throughput, not just microbenchmarks

## Phase 1: Remove Obvious Wasted Work

1. Make `seq_lens` shorten the K loop inside the CuTe kernels.

Current behavior still scans all `T` keys and applies a mask inside the loop, so varied lengths
cost almost the same as full length.

Expected impact:

- direct speedup for padded LRA-style batches
- no semantic change
- low implementation risk

2. Optionally expose query lengths as well.

Right now padded query rows are still computed and zeroed later by the caller.

Expected impact:

- avoids wasted work on invalid query rows
- useful if nanomoe can provide both valid query and key extents

## Phase 2: Fix Full-Attention Kernel Mapping

1. Replace the current one-warp-per-query-row mapping with a multi-row CTA design.

`head_dim=2` is too small to amortize overhead well with one row per warp.

Target:

- each CTA handles multiple query rows and/or multiple heads
- increase useful work per launch and per K/V tile load

2. Add shared-memory staging for K/V tiles.

Current kernel is simple and correct, but `D=2` means memory traffic dominates quickly.

Target:

- load K/V once per CTA tile
- reuse across multiple query rows

3. Split kernel variants by path.

Have separate fast kernels for:

- unmasked full-length
- key-padding / `seq_lens`
- dense mask fallback

Reason:

- the unmasked path should not pay for mask branches

## Phase 3: Add Better Scheduling

1. Move to a persistent CTA scheduler over `(batch, head, q-block)`.

At `D=2`, launch/index overhead matters more than usual.

Target:

- keep SMs busy across many small-row tasks
- reduce scheduling inefficiency at large `B * H * T`

2. Tune tile shapes for the real regime.

Primary benchmark target:

- `[64, 64, 2048, 2]`

Need to search:

- query rows per CTA
- heads per CTA
- K tile size
- staging strategy

## Phase 4: Sparse Eval Kernels

1. Special-case `topk1`.

Treat it as argmax attention rather than generic top-k.

Expected impact:

- lower overhead
- simpler inner loop
- likely large win for eval

2. Redesign `topk4`.

Current implementation is correct but expensive because it effectively rescans keys for repeated
selection.

Better design:

- blockwise candidate selection
- keep a small in-register top-4 heap/list
- merge candidates across tiles

3. Keep `seq_lens` efficient for sparse modes too.

Sparse kernels should stop at valid `k_end`, not scan padded keys.

## Phase 5: Integration Guardrails

1. Add a debug trace mode in `hull_attn`.

For each call, report:

- mode
- dtype
- contiguity
- mask form
- chosen backend path

Reason:

- transformer regressions are otherwise hard to diagnose

2. Add benchmark gates for the real shapes.

Track at least:

- SDPA `[64, 8, 2048, 16]`
- hull full `[64, 64, 2048, 2]`
- hull `topk1`
- hull `topk4`
- with and without `seq_lens`

3. Keep width-matched smoke coverage in tests, but do not add timing assertions in CI.

## Priority Order

1. `seq_lens` early-exit in full kernels
2. multi-row / multi-head CTA mapping for full kernels
3. shared-memory staged K/V tiles
4. `topk1` specialization
5. `topk4` redesign
6. persistent scheduling and tuning
7. optional query-length support

## Success Criteria

- varied `seq_lens` faster than full-length in standalone benchmarks
- full hull materially faster than current `34.5 ms` at `[64, 64, 2048, 2]`
- `topk1` and `topk4` improve over current CuTe sparse baselines
- transformer-level gap vs SDPA narrows substantially
- no fallback to dense-mask or Python paths in intended runs
