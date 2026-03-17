"""Unified benchmark: all hull attention kernels, forward + backward, speed + memory."""

import statistics
import sys
import traceback

import torch
import torch.nn.functional as F


def _benchmark(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    peak_alloc = []
    for _ in range(iters):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
        peak_alloc.append(torch.cuda.max_memory_allocated())
    return {
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "peak_MiB": max(peak_alloc) / (1024**2),
    }


def _safe_benchmark(fn, warmup=5, iters=20):
    try:
        return _benchmark(fn, warmup, iters)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"
    except Exception as e:
        traceback.print_exc()
        return f"error: {e}"


def _fmt(r):
    if isinstance(r, str):
        return r
    return f"{r['median_ms']:>8.3f}ms  {r['min_ms']:>8.3f}ms  {r['peak_MiB']:>8.1f}MiB"


def main():
    dtype = torch.bfloat16
    device = "cuda"
    batch, heads, seqlen, head_dim = 8, 64, 512, 2
    shape = (batch, heads, seqlen, head_dim)
    warmup, iters = 5, 20

    torch.manual_seed(0)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # --- imports ---
    from quack.hull_attn import hull_attn
    from quack.hull_attn2 import hull_attn2
    from quack.hull_attn3 import hull_attn3
    from quack.hull_attn_codex import hull_attn_codex
    from quack.flash_hull_attn import flash_hull_attn

    # ===== FORWARD =====
    fwd = {}

    fwd["pytorch_sdpa"] = _safe_benchmark(
        lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False),
        warmup, iters,
    )
    fwd["hull_attn"] = _safe_benchmark(
        lambda: hull_attn(q, k, v, mode="full"), warmup, iters,
    )
    fwd["hull_attn2"] = _safe_benchmark(
        lambda: hull_attn2(q, k, v), warmup, iters,
    )
    fwd["hull_attn3"] = _safe_benchmark(
        lambda: hull_attn3(q, k, v), warmup, iters,
    )
    fwd["hull_attn_codex"] = _safe_benchmark(
        lambda: hull_attn_codex(q, k, v), warmup, iters,
    )
    fwd["flash_hull_attn"] = _safe_benchmark(
        lambda: flash_hull_attn(q, k, v), warmup, iters,
    )

    print(f"shape={list(shape)}  dtype={dtype}  warmup={warmup}  iters={iters}")
    print()
    print("=== FORWARD ===")
    print(f"  {'kernel':20s}  {'median':>10s}  {'min':>10s}  {'peak mem':>10s}")
    print(f"  {'-'*54}")
    for name, r in fwd.items():
        print(f"  {name:20s}  {_fmt(r)}")

    # ===== BACKWARD (fwd+bwd) =====
    def _bwd(attn_fn, **kwargs):
        def run():
            qi = q.detach().clone().requires_grad_(True)
            ki = k.detach().clone().requires_grad_(True)
            vi = v.detach().clone().requires_grad_(True)
            out = attn_fn(qi, ki, vi, **kwargs)
            out.float().square().mean().backward()
        return run

    bwd = {}
    bwd["pytorch_sdpa"] = _safe_benchmark(
        _bwd(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)),
        warmup, iters,
    )
    bwd["hull_attn"] = _safe_benchmark(
        _bwd(hull_attn, mode="full"), warmup, iters,
    )
    bwd["hull_attn2"] = _safe_benchmark(
        _bwd(hull_attn2), warmup, iters,
    )
    bwd["hull_attn3"] = _safe_benchmark(
        _bwd(hull_attn3), warmup, iters,
    )
    bwd["hull_attn_codex"] = _safe_benchmark(
        _bwd(hull_attn_codex), warmup, iters,
    )
    bwd["flash_hull_attn"] = _safe_benchmark(
        _bwd(flash_hull_attn), warmup, iters,
    )

    print()
    print("=== BACKWARD (fwd + bwd) ===")
    print(f"  {'kernel':20s}  {'median':>10s}  {'min':>10s}  {'peak mem':>10s}")
    print(f"  {'-'*54}")
    for name, r in bwd.items():
        print(f"  {name:20s}  {_fmt(r)}")


if __name__ == "__main__":
    main()
