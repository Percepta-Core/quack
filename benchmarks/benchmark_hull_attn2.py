"""Benchmark hull_attn2 (warp-shuffle) vs hull_attn_codex (smem) vs hull_attn (original).

Measures both latency and peak GPU memory.
"""

import argparse
import math
import statistics

import torch
import torch.nn.functional as F


def _benchmark(fn, warmup: int, iters: int):
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
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "peak_MiB": max(peak_alloc) / (1024**2),
    }


def _fmt(r):
    if isinstance(r, str):
        return r
    return f"med={r['median_ms']:.3f}ms  min={r['min_ms']:.3f}ms  peak={r['peak_MiB']:.1f}MiB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--seq-lens", action="store_true")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = "cuda"
    bm = lambda fn: _benchmark(fn, args.warmup, args.iters)
    torch.manual_seed(0)
    shape = (args.batch, args.heads, args.seqlen, 2)

    seq_lens = None
    if args.seq_lens:
        seq_lens = torch.linspace(
            args.seqlen, max(args.seqlen // 4, 1), args.batch, device=device
        ).round().to(dtype=torch.int32)

    sdpa_mask = None
    if seq_lens is not None:
        kpm = torch.arange(args.seqlen, device=device)[None, :] < seq_lens[:, None]
        sdpa_mask = kpm[:, None, None, :]

    print(
        f"shape={list(shape)}  dtype={dtype}  backward={args.backward}  "
        f"seq_lens={args.seq_lens}"
    )
    print("-" * 90)

    results = {}

    # --- pytorch sdpa ---
    def _sdpa():
        q = torch.randn(shape, device=device, dtype=dtype)
        k, v = torch.randn_like(q), torch.randn_like(q)
        if args.backward:
            q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False)
            out.float().square().mean().backward()
        else:
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False)
    try:
        results["pytorch_sdpa"] = bm(_sdpa)
    except Exception as e:
        results["pytorch_sdpa"] = str(e)

    # --- hull_attn (original) ---
    try:
        from quack.hull_attn import hull_attn
        def _hull_orig():
            q = torch.randn(shape, device=device, dtype=dtype)
            k, v = torch.randn_like(q), torch.randn_like(q)
            if args.backward:
                q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
                out = hull_attn(q, k, v, mode="full", seq_lens=seq_lens)
                out.float().square().mean().backward()
            else:
                hull_attn(q, k, v, mode="full", seq_lens=seq_lens)
        results["hull_attn_orig"] = bm(_hull_orig)
    except Exception as e:
        results["hull_attn_orig"] = f"error: {e}"

    # --- hull_attn_codex (smem) --- forward only
    if not args.backward:
        try:
            from quack.hull_attn_codex import hull_attn_codex
            def _codex():
                q = torch.randn(shape, device=device, dtype=dtype)
                k, v = torch.randn_like(q), torch.randn_like(q)
                hull_attn_codex(q, k, v, seq_lens=seq_lens)
            results["hull_attn_codex"] = bm(_codex)
        except Exception as e:
            results["hull_attn_codex"] = f"error: {e}"

    # --- hull_attn2 (warp-shuffle) ---
    try:
        from quack.hull_attn2 import hull_attn2
        def _v2():
            q = torch.randn(shape, device=device, dtype=dtype)
            k, v = torch.randn_like(q), torch.randn_like(q)
            if args.backward:
                q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
                out = hull_attn2(q, k, v, seq_lens=seq_lens)
                out.float().square().mean().backward()
            else:
                hull_attn2(q, k, v, seq_lens=seq_lens)
        results["hull_attn2"] = bm(_v2)
    except Exception as e:
        results["hull_attn2"] = f"error: {e}"

    # --- print ---
    for name, r in results.items():
        print(f"  {name:20s}  {_fmt(r)}")


if __name__ == "__main__":
    main()
