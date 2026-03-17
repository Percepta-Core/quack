# 🦆 QuACK: A Quirky Assortment of CuTe Kernels 🦆

Kernels are written in the [CuTe-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Installation

``` bash
# For CUDA 12.9:
pip install quack-kernels

# For CUDA 13.1:
pip install 'quack-kernels[cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Or using uv (faster):
uv pip install 'quack-kernels[cu13]'
```

## Requirements

- H100 or B200/B300 GPU
- CUDA toolkit 12.9+
- Python 3.12

## Kernels 🐥

- 🦆 RMSNorm forward + backward
- 🦆 Softmax forward + backward
- 🦆 Cross entropy forward + backward
- 🦆 Layernorm forward
- 🦆 Hull attention forward + backward (head_dim=2)
- 🦆 Hopper gemm + epilogue
- 🦆 Blackwell gemm + epilogue

## Usage

```
from quack import rmsnorm, softmax, cross_entropy
```

## Documentations

[2025-07-10] We have a comprehensive
[blogpost](media/2025-07-10-membound-sol.md) on how to get memory-bound kernels
to speed-of-light, right in the comfort of Python thanks to the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Performance

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

See our [blogpost](media/2025-07-10-membound-sol.md) for the details.

### Hull Attention (head_dim=2)

Width-matched comparison: hull attention at `[64, 64, 2048, 2]` vs SDPA/FA4 at `[64, 8, 2048, 16]` (same total width). H200, bf16.

**Forward**

| Kernel | Median | Peak Memory |
|---|---|---|
| FA4 (8h×16d) | 1.04 ms | 320 MiB |
| SDPA (8h×16d) | 1.80 ms | 228 MiB |
| FA4 pad→8 (64h×2d) | 7.05 ms | 800 MiB |
| hull_attn_codex | 12.99 ms | 736 MiB |
| SDPA (64h×2d) | 12.86 ms | 832 MiB |
| hull_attn2 | 13.63 ms | 736 MiB |
| hull_attn3 | 15.62 ms | 736 MiB |
| flash_hull_attn | 15.60 ms | 736 MiB |
| hull_attn | 28.71 ms | 736 MiB |

**Backward (fwd + bwd)**

| Kernel | Median | Peak Memory |
|---|---|---|
| SDPA (8h×16d) | 6.75 ms | 1128 MiB |
| hull_attn2 | 38.51 ms | 576 MiB |
| hull_attn_codex | 41.10 ms | 576 MiB |
| SDPA (64h×2d) | 48.38 ms | 3136 MiB |
| flash_hull_attn | 48.83 ms | 576 MiB |
| hull_attn3 | 66.81 ms | 576 MiB |
| hull_attn | 82.48 ms | 576 MiB |

The best hull kernels (hull_attn2, hull_attn_codex) are still **12–13x slower** than FA4 on the equivalent 16d shape forward, and **5–6x slower** backward vs SDPA 16d. They do use less memory than SDPA on 2d inputs (576 vs 3136 MiB backward). Work in progress.

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install

# For CUDA 13.1:
pip install 'quack-kernels[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Or using uv:
uv pip install 'quack-kernels[dev,cu13]'
```
