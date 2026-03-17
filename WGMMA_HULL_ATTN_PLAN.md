# WGMMA Hull Attention Plan

## Goal

Make the exact `head_dim=2` hull attention path tensor-core-friendly on SM90 without changing
attention semantics.

Target workload:

- H200 / SM90
- bf16 / fp16
- non-causal full attention
- real benchmark shape: `[64, 64, 2048, 2]`


## Current State

The repo does not currently use a WGMMA-based grouped-head path for hull attention.

Current hull-specific implementations are scalar:

- `quack/flash_hull_attn.py`
- `quack/fa4/flash_hull_fwd_sm90.py`

Those kernels operate directly on `[B * H, S, 2]` and avoid tensor cores entirely. That is a
reasonable baseline, but it leaves score generation on the table when we want to exploit SM90
WGMMA.

The generic FA4 path is also not a direct answer for `head_dim=2`. It rounds head dimensions up to
a multiple of 16, which is fine for regular flash attention but wasteful here:

- `quack/fa4/flash_fwd.py`


## Problem

Plain WGMMA does not fit `head_dim=2` well:

- the dot product is only 2-wide
- generic tensor-core layouts expect a much wider K dimension
- padding every head from 2 to 16 wastes bandwidth and shared memory
- the output/value side is also only 2-wide, so a naive WGMMA-on-everything design can spend more
  effort on packing than on useful math

For hull attention, the real bottleneck is still score generation plus kernel structure, not raw
FLOP throughput. Any WGMMA plan has to preserve that reality.


## Proposed Fix

Use a **grouped-head WGMMA trick** for the **QK score step only**.

Core idea:

- keep model semantics as `64` independent heads of dimension `2`
- group `8` real heads together inside the kernel
- synthesize a width-`16` QK problem for WGMMA
- keep online softmax and the `P @ V` accumulation scalar / register-based

This is not a model change. It is only an internal kernel layout change.


## Non-Goals

- Do not reinterpret the model as `8` heads of dimension `16`
- Do not perform a single softmax across mixed heads
- Do not force the `P @ V` path onto WGMMA in v1
- Do not reuse `PackGQA` as the main mechanism for this


## Relation To PackGQA

This plan is different from FA4 `PackGQA`.

`PackGQA` folds multiple query heads into the sequence dimension for grouped-query attention. It is
about addressing and scheduling for GQA-compatible semantics.

This plan instead packs several independent dim-2 heads into a synthetic width-16 operand so WGMMA
can be used for the score matmul while preserving per-head softmax semantics.

Both ideas "group heads", but they solve different problems.


## Grouped-Head Trick

Let:

- `G = 8` heads per group
- `d = 2` head dim
- `Dg = G * d = 16`

For one group of 8 heads, pack each key token into a 16-wide vector:

```text
K_pack[token] =
  [k_h0_x, k_h0_y, k_h1_x, k_h1_y, ..., k_h7_x, k_h7_y]
```

Now define a **selector-packed** query row for one logical `(query_token, head_in_group)` pair:

```text
Q_sel(query_token, head=r) =
  [0, 0, ..., q_r_x, q_r_y, ..., 0, 0]
```

Only the 2 lanes corresponding to that head are nonzero.

Then:

```text
score(query_token, head=r, key_token) = Q_sel_r · K_pack[key_token]
```

This exactly equals the original dim-2 dot product for that head, because the other 14 lanes are
zero.


## Why This Preserves Semantics

We are not mixing attention probabilities across heads.

Instead:

- each logical row in the score matrix still corresponds to one original `(query_token, head)`
- the packed width-16 representation is only a compute trick
- softmax still runs per logical row
- output accumulation still writes back to the original `[batch, heads, seq, 2]` layout

So the kernel computes the same attention as the scalar implementation.


## Forward Kernel Shape

For one CTA working on one batch element and one 8-head group:

- choose `Q_ROWS_PER_CTA` query tokens, for example `8` or `16`
- logical row count becomes `M = Q_ROWS_PER_CTA * 8`
- key tile width stays `N = 64` or `128`
- packed inner dimension is `K = 16`

So the WGMMA score step becomes:

```text
[M, 16] x [N, 16]^T -> [M, N]
```

Example:

- `Q_ROWS_PER_CTA = 8`
- `M = 64`
- one CTA computes scores for 8 query positions across 8 heads


## Forward Kernel Structure

### Phase A: Load and Pack

1. Load one K/V tile for 8 heads into shared memory
2. Repack K into `sK_pack[tile_n, 16]`
3. Load a small tile of Q for those 8 heads
4. Form selector-packed Q fragments in registers or shared memory

### Phase B: Score Generation

1. Run WGMMA on packed `Q` and packed `K`
2. Produce score fragments shaped as logical rows by key tile
3. Reshape the WGMMA output back to logical rows:
   `(query_row_in_cta, head_in_group, key_idx_in_tile)`

### Phase C: Softmax and Output

1. Maintain online softmax state per logical row:
   `running_max`, `running_sum`
2. Keep output accumulators scalar:
   two fp32 values per logical row
3. Update the output accumulator with the current tile
4. Normalize and store back to `[batch, heads, seq, 2]`


## Why `P @ V` Should Stay Scalar In V1

The output is only 2-wide per head.

That means:

- the useful accumulation state per logical row is only 2 fp32 values
- scalar accumulation is cheap
- WGMMA would require extra packing for `P` and `V` that may cost more than the math itself
- score generation is the more natural place to use tensor cores

So the pragmatic split is:

- WGMMA for `QK`
- scalar online softmax
- scalar `P @ V`


## Backward Plan

Use the same grouped-head WGMMA trick first for **score recompute**, not for every backward
subproblem.

### Backward v1

1. Save only:
   - output `O`
   - row-wise `LSE`
2. In backward, reload grouped Q and grouped K tiles
3. Recompute scores using grouped-head WGMMA
4. Reconstruct probabilities row-by-row
5. Accumulate:
   - `dQ` scalar
   - `dK` scalar
   - `dV` scalar

This keeps the first implementation narrow and lowers packing complexity.

### Backward v2 Optional

If score recompute is clearly dominant, consider adding WGMMA to some backward score-related
products. Do not commit to this until the forward grouped-head path is proven worthwhile.


## CTA Mapping

The grouped-head trick only helps if CTA mapping is also fixed.

Required changes:

- one CTA should handle multiple query rows, not one query row
- K/V tiles must be reused across all rows in the CTA
- grid should look like `(batch, head_group, q_tile)` rather than one warp per query row

Suggested first shape:

- head group size: `8`
- query rows per CTA: `8` or `16`
- K tile: `64` or `128`
- one warpgroup dedicated to QK WGMMA
- remaining threads help with load/store and scalar softmax work as needed


## Data Layout Plan

### Inputs

Keep external API unchanged:

- `Q, K, V`: `[batch, heads, seq, 2]`

### Internal View

Within a head group of 8:

- `K_pack`: `[seq_tile, 16]`
- `V_group`: keep as `[seq_tile, 8, 2]` or equivalent unpacked layout
- `Q_sel`: logical rows of `[16]`, one row per `(query_token, head_in_group)`

### Output

Write back directly to:

- `O`: `[batch, heads, seq, 2]`

Do not expose grouped layouts outside the kernel.


## Implementation Plan

### Step 1: Prototype Forward-Only WGMMA QK

Build a new experimental forward kernel that:

- only supports dense full attention
- only supports `head_dim=2`
- only supports head counts divisible by `8`
- computes `QK` via grouped-head WGMMA
- keeps softmax and output scalar

Success criteria:

- numerically matches current `flash_hull_attn`
- faster than the current scalar forward on `[64, 64, 2048, 2]`

### Step 2: Autotune Tile Shapes

Sweep:

- query rows per CTA
- key tile size
- whether selector-packed Q lives in registers or shared memory
- number of warpgroups per CTA

### Step 3: Add Backward Score Recompute

Reuse the grouped-head forward packing machinery in backward for score recompute.

Keep gradient accumulation scalar first.

### Step 4: Dispatch Integration

Add a narrow fast path in `flash_hull_attn` or a sibling module:

- use grouped-head WGMMA path only when constraints are satisfied
- otherwise fall back to the current scalar dim-2 implementation


## Risks

### 1. Q Packing Overhead Cancels WGMMA Gain

The key risk is that forming selector-packed Q rows costs too much.

Mitigation:

- start with small query tiles
- keep Q in registers if possible
- benchmark score-kernel-only microcases before full integration

### 2. Softmax Still Dominates

Even if QK gets faster, scalar online softmax and bookkeeping may still dominate.

Mitigation:

- treat WGMMA as only one part of the redesign
- keep the multi-row CTA structure as a hard requirement

### 3. Backward Complexity Explodes

A fully WGMMA-ified backward can become much more complex than the forward.

Mitigation:

- limit v1 backward to WGMMA score recompute only
- keep `dQ`, `dK`, and `dV` scalar until profiling proves otherwise

### 4. Head Grouping Assumptions Leak Into API

The grouping factor must stay internal.

Mitigation:

- never change external tensor shapes
- dispatch only when `num_heads % 8 == 0`


## Validation Plan

### Correctness

Compare against current exact hull attention on:

- forward outputs
- `LSE`
- backward gradients for `Q`, `K`, `V`
- masked and unmasked variants once dense full works

### Numerical Checks

Test:

- bf16
- fp16
- short and long sequences
- exact equality targets replaced by tolerance-based checks vs current reference path

### Performance

Measure:

- forward latency on `[64, 64, 2048, 2]`
- backward latency on `[64, 64, 2048, 2]`
- achieved bandwidth
- sensitivity to query tile and key tile sizes

### Profiling

Use `ncu` or equivalent to confirm:

- WGMMA instructions are actually emitted
- CTA count is reduced materially vs the old mapping
- shared-memory traffic is not exploding
- tensor-core work is not hidden under packing overhead


## Open Questions

1. Should selector-packed Q be built in registers or staged in shared memory?
2. Is `G = 8` the best group size, or is a larger synthetic width worthwhile?
3. Should one CTA own one head group, or multiple head groups when sequence tiles are small?
4. Is there a clean way to reuse FA4 load/store helpers without inheriting too much generic logic?


## Recommendation

Pursue this plan only as a **new experimental path**, not as a direct mutation of the current
scalar dim-2 kernel.

The best first milestone is:

- grouped-head WGMMA for forward `QK`
- scalar online softmax
- scalar output accumulation
- dense full attention only

If that does not beat the current scalar kernel meaningfully, stop there. If it does, extend the
same packing idea to backward score recompute.
