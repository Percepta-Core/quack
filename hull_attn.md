# Hull Attention Plan

## Goal

Implement a QuACK-side kernel path for `hullattn` and keep `nanomoe` as the integration layer.

The target is deliberately narrow:

- GPU target: H200
- CUDA path only
- `d_model = 128`
- `num_heads = 64`
- `head_dim = 2`
- non-causal attention
- full-sequence LRA ListOps first

The first real deliverable is not a generic attention library. It is an H200-optimized `head_dim=2` attention path with:

- exact full attention for training
- exact full forward and backward using flash-attention style streaming
- inference-only approximate variants for `top_k=1` and `top_k=4`


## Why QuACK

The kernel should live in QuACK, not in `nanomoe`.

Reasons:

- QuACK already has CuTe-DSL build, compile-cache, testing, and packaging patterns.
- QuACK already has reusable row-wise pieces: reductions, softmax patterns, varlen helpers, and top-k building blocks.
- `nanomoe` should stay responsible for model dispatch, experiment wiring, fallback reference code, and autograd-facing integration.

Planned split:

- `quack`: low-level kernels, Python wrappers, low-level correctness tests, perf microbenchmarks
- `nanomoe`: backend dispatch, reference implementation, experiment plumbing, end-to-end validation


## Scope

### In scope

- exact forward kernel for full attention
- exact backward kernel for full attention
- both specialized to `head_dim = 2`
- inference kernels for `top_k=1` and `top_k=4`
- ListOps-scale validation first

### Out of scope for v1

- generic `head_dim`
- causal masking
- arbitrary sparse training backward
- Path-X-first tuning
- multi-architecture support beyond H200


## High-Level Product Shape

We should treat this as two related kernel families sharing the same frontend API:

1. Training / exact path
   - exact full attention
   - flash-attention style streaming forward
   - flash-attention style recompute backward

2. Inference / approximate path
   - stream scores and keep only top-k state
   - support `k=1` and `k=4`
   - no training backward for this path in v1

That split matches the actual use case:

- training needs exactness and a stable backward
- inference can trade accuracy for lower work and lower memory traffic


## Target API

Keep the Python surface small and explicit.

```python
hull_attn(
    q,
    k,
    v,
    mode="full",          # "full", "topk1", "topk4"
    scale=None,
    attention_mask=None,
)
```

Autograd behavior:

- `mode="full"`: forward + backward implemented in QuACK
- `mode="topk1"`: forward-only in v1
- `mode="topk4"`: forward-only in v1

Input assumptions:

- `q, k, v` are CUDA tensors
- shape `[batch, heads, seq, 2]`
- dtype `bf16` first, then optionally `fp16`
- non-causal

Output:

- attended tensor of shape `[batch, heads, seq, 2]`


## Core Design Decision

### Training path should be full flash attention, not sparse top-k

The previous plan emphasized eval-first sparse kernels. That is no longer the right center of gravity.

For training, implement the standard exact attention semantics:

1. `scores = q @ k^T * scale`
2. row-wise softmax over the full key axis
3. output `o = p @ v`
4. backward from `do` to `dq`, `dk`, `dv`

The implementation should still be specialized aggressively to `head_dim = 2`, but the math is exact full attention.

Reason:

- the full path needs a real backward
- `head_dim = 2` makes score math unusually cheap
- H200 bandwidth and on-chip memory are better spent on a fused streaming exact kernel than on a training-time sparse approximation that still needs bespoke backward semantics


## Kernel Design Notes

### Observation: `head_dim = 2` changes the optimization problem

This is not generic flash attention.

Useful properties:

- each score is only two multiply-add pairs
- query, key, and value state is tiny
- score FLOPs are cheap relative to memory movement
- normalization, bookkeeping, and tile scheduling matter more than dot-product throughput

Implication:

- optimize for streaming and reuse of K/V tiles
- keep Q in registers whenever possible
- keep softmax state per query row in registers
- avoid generic code paths built around larger head dimensions


## Forward Kernel Plan (`mode="full"`)

Implement a full flash-attention style forward kernel specialized to `head_dim=2`.

Per `(batch, head, query_tile)`:

1. load a tile of queries into registers
2. initialize per-row running max `m`, running sum `l`, and running output accumulator `acc`
3. stream over K/V tiles
4. compute score fragments from 2D dot products
5. apply mask if present
6. update online softmax state
7. update output accumulator using the rescaled flash-attention recurrence
8. write final normalized output

Notes:

- no materialized score matrix
- no materialized probability matrix
- use the standard online softmax recurrence
- output accumulator is only 2 floats per row, which should make register pressure manageable


## Backward Kernel Plan (`mode="full"`)

Implement full exact backward using flash-attention style recomputation, still specialized to `head_dim=2`.

Backward objectives:

- `dv = p^T @ do`
- `dp = do @ v^T`
- `ds = p * (dp - delta)` where `delta = sum(dp * p, dim=-1)`
- `dq = ds @ k`
- `dk = ds^T @ q`

Recommended structure:

1. save only the forward output and row-wise logsumexp or equivalent normalization state
2. recompute score tiles during backward instead of storing full probabilities
3. stream over the same K/V tiles used by forward
4. accumulate `dv` and `dk` tilewise
5. accumulate `dq` per query tile

Why this is tractable here:

- `head_dim=2` keeps `dq`, `dk`, `dv`, and `o` accumulators tiny
- recomputation cost is low because score generation is cheap
- exact backward stays aligned with standard flash-attention math instead of inventing new sparse backward semantics

Likely saved tensors / metadata:

- output `o`
- per-row normalization state needed to reconstruct probabilities
- shape / stride / mask metadata


## Inference Kernel Plan (`mode="topk1"` and `mode="topk4"`)

Inference can use a separate kernel family.

Semantics:

1. compute streamed scores from full QK
2. maintain top-k state per query row for `k=1` or `k=4`
3. softmax only over the retained entries
4. reduce the selected V rows

This path is intentionally not training-compatible in v1.

Design notes:

- hard-specialize `k=1` and `k=4` instead of building a generic top-k kernel first
- use fixed-size register state for scores, indices, and value accumulators
- avoid routing through the current standalone `quack.topk` op in the hot path


## Proposed Phases

### Phase 0: environment and target proof

Before writing the real kernel:

1. confirm QuACK builds and imports in the environment used by `nanomoe`
2. verify H200 target assumptions and toolchain support
3. confirm we can compile and launch an H200-targeted custom op in the expected runtime

Exit criterion:

- a trivial QuACK custom op builds and runs in the target environment on H200


### Phase 1: exact full forward kernel

Build the forward kernel for `mode="full"`.

Requirements:

- exact semantics
- `head_dim=2` specialization
- ListOps-scale sequence lengths first
- numerically stable online softmax

Exit criterion:

- Python-callable forward kernel matches a PyTorch reference on small cases


### Phase 2: exact full backward kernel

Build the backward kernel for `mode="full"`.

Requirements:

- exact gradients for `q`, `k`, `v`
- flash-attention style recomputation
- no score/probability materialization

Exit criterion:

- autograd path matches reference gradients on small and medium cases


### Phase 3: inference top-k kernels

Add `mode="topk1"` and `mode="topk4"` forward kernels.

Requirements:

- specialized kernels, not a generic parameterized top-k loop
- same frontend API as the exact path
- numerically matched against a reference top-k implementation

Exit criterion:

- both inference modes callable from Python and validated against reference


### Phase 4: nanomoe integration

After QuACK has a stable API:

1. add optional QuACK import in `nanomoe`
2. route training full-attention hull path through `mode="full"`
3. route inference approximations through `mode="topk1"` and `mode="topk4"`
4. keep the existing reference implementation as fallback


## Validation Plan

### Low-level QuACK tests

Add tests for:

- exact output match vs PyTorch reference on tiny tensors
- gradient match vs PyTorch reference for `mode="full"`
- mask handling
- `mode="topk1"` and `mode="topk4"` correctness
- bf16 numerical tolerance

### Cross-repo validation

Compare QuACK against the `nanomoe` reference helper on:

- small synthetic tensors
- one real ListOps batch for training
- one real ListOps batch for inference

### Performance checks

Measure:

- compile time
- forward latency
- backward latency
- peak memory

Compare against:

- PyTorch dense SDPA / reference attention
- current `nanomoe` hull attention reference path


## Main Risks

1. H200-specific tuning may still need a fallback schedule for bring-up and correctness.
2. `head_dim=2` can make generic flash-attention tiling choices suboptimal.
3. register pressure can still grow if we over-tile queries or over-fuse masking and bookkeeping.
4. inference top-k may look simple but can become latency-dominated by selection logic if not hard-specialized.
5. Path-X requirements could tempt premature generalization before ListOps is stable.


## Recommended Next Steps

1. Prove QuACK can build and run a trivial custom op on H200 in the target environment.
2. Create a new QuACK module for exact `hull_attn` forward with fixed `head_dim=2`.
3. Add the corresponding exact backward with flash-style recomputation.
4. Add unit tests against a PyTorch full-attention reference.
5. Implement separate inference kernels for `topk=1` and `topk=4`.
6. Integrate all three modes into `nanomoe` behind a safe fallback dispatch.
