# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuACK (Quirky Assortment of CuTe Kernels) — high-performance CUDA kernels written in [CuTe-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), targeting H100 (SM90) and B200/B300 (SM100) GPUs. Package name: `quack-kernels`.

## Build & Development

```bash
# Install (dev)
uv pip install -e '.[dev]'
pre-commit install

# For CUDA 13.1
uv pip install -e '.[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Lint & format
uv run ruff check --fix quack/ tests/
uv run ruff format quack/ tests/

# Run all tests
uv run pytest tests/

# Run a single test
uv run pytest tests/test_rmsnorm.py -x
uv run pytest tests/test_rmsnorm.py::test_rmsnorm_fwd -x -k "bfloat16"
```

## CuTe DSL Conventions

For code inside `@cute.jit` or `@cute.kernel` decorated functions, only a subset of Python syntax is supported. Follow [Control Flow](docs/dsl_control_flow.rst) and [Limitations](docs/limitations.rst).

Key rules:
- `cutlass.const_expr()` marks compile-time constants; `cutlass.range_constexpr()` unrolls loops at compile time
- `cutlass.range()` is for dynamic runtime loops
- No early `break`/`continue` in loops; no `return` values from jit functions
- Python lists/dicts inside DSL are static (compile-time only)
- Types must be determinable at compile time (no dependent types)
- Variables defined inside control flow bodies are not accessible outside

## Architecture

### Kernel patterns

**Reduction kernels** (`rmsnorm.py`, `softmax.py`, `cross_entropy.py`) inherit from `ReductionBase` in `reduction_base.py`. They share a pattern: configure cluster size, get tiled copies, allocate reduction buffers with mbarriers, then launch a `@cute.kernel`.

**GEMM** has a multi-layer design:
- `gemm.py` — public API, validates inputs, selects SM version, caches compiled kernels
- `gemm_interface.py` — unified interface across SM versions
- `gemm_sm90.py` / `gemm_sm100.py` — SM-specific implementations
- `gemm_default_epi.py` + `gemm_*_epi.py` — epilogue variants (bias, activation, etc.)
- `gemm_config.py` — `GemmConfig` dataclass with tile sizes, cluster dims, swizzle settings

### Core utilities

- `copy_utils.py` — memory copy operations (shared↔register, async copies, tiled copies)
- `layout_utils.py` — layout algebra (transpose, select, expand, permute)
- `cute_dsl_utils.py` — dtype mapping, device capability queries, parameter base classes
- `tile_scheduler.py` — tile scheduling for persistent kernels
- `varlen_utils.py` — variable-length sequence support

### Testing

Tests use pytest with parametrize across dtypes (`float32`, `float16`, `bfloat16`), dimensions, and batch sizes. Each test includes a reference implementation for numerical validation.

## Iteration Speed

When iterating on kernel code, run a small subset of tests (1-3 parametrizations) rather than the full test suite. Use `-k` or pass specific test IDs to pytest. Only run the full suite when finalizing changes.

## Code Style

- Favor concise, self-explanatory code
- Avoid unnecessary line breaks
- Empty lines should not have any space
- Line length: 100 (ruff)
- Ruff allows: lambda assignment (E731), single-char vars I/O/l (E741), unused locals (F841)
