import argparse
import statistics

import torch
import torch.nn.functional as F

from quack.hull_attn import hull_attn
from quack.hull_attn3 import hull_attn3


def _benchmark(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = []
    peak_allocated = []
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
        peak_allocated.append(torch.cuda.max_memory_allocated())

    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "peak_MiB": max(peak_allocated) / (1024**2),
    }


def _fmt(result):
    if isinstance(result, str):
        return result
    return f"median={result['median_ms']:.3f} ms  mean={result['mean_ms']:.3f} ms  peak={result['peak_MiB']:.1f} MiB"


def _safe_benchmark(fn, warmup: int, iters: int):
    try:
        return _benchmark(fn, warmup, iters)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--seq-lens", action="store_true")
    args = parser.parse_args()
    if args.mask and args.seq_lens:
        raise ValueError("--mask and --seq-lens are mutually exclusive")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = "cuda"
    shape = (args.batch, args.heads, args.seqlen, 2)

    torch.manual_seed(0)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    seq_lens = None
    sdpa_mask = None
    hull_attn3_mask = None
    hull_mask = None
    if args.mask:
        hull_attn3_mask = torch.zeros((args.batch, args.seqlen, args.seqlen), device=device, dtype=torch.float32)
        hull_attn3_mask[:, 0, -1] = -torch.inf
        hull_attn3_mask[:, 3, :8] = -torch.inf
        hull_attn3_mask[:, 9, 17] = -1e4
        hull_mask = hull_attn3_mask[:, None, :, :]
        sdpa_mask = hull_mask
    if args.seq_lens:
        seq_lens = torch.linspace(args.seqlen, max(args.seqlen // 4, 1), args.batch, device=device)
        seq_lens = seq_lens.round().to(dtype=torch.int32)
        key_padding_mask = torch.arange(args.seqlen, device=device)[None, :] < seq_lens[:, None]
        sdpa_mask = key_padding_mask[:, None, None, :]

    def run_hull_attn3_forward():
        hull_attn3(q, k, v, attention_mask=hull_attn3_mask, seq_lens=seq_lens)

    def run_hull_attn_forward():
        hull_attn(q, k, v, mode="full", attention_mask=hull_mask, seq_lens=seq_lens)

    def run_sdpa_forward():
        F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False)

    def run_hull_attn3_backward():
        q_in = q.detach().clone().requires_grad_(True)
        k_in = k.detach().clone().requires_grad_(True)
        v_in = v.detach().clone().requires_grad_(True)
        hull_attn3(
            q_in, k_in, v_in, attention_mask=hull_attn3_mask, seq_lens=seq_lens
        ).float().square().mean().backward()

    def run_hull_attn_backward():
        q_in = q.detach().clone().requires_grad_(True)
        k_in = k.detach().clone().requires_grad_(True)
        v_in = v.detach().clone().requires_grad_(True)
        hull_attn(
            q_in, k_in, v_in, mode="full", attention_mask=hull_mask, seq_lens=seq_lens
        ).float().square().mean().backward()

    def run_sdpa_backward():
        q_in = q.detach().clone().requires_grad_(True)
        k_in = k.detach().clone().requires_grad_(True)
        v_in = v.detach().clone().requires_grad_(True)
        F.scaled_dot_product_attention(
            q_in, k_in, v_in, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
        ).float().square().mean().backward()

    forward_results = {
        "hull_attn3": _safe_benchmark(run_hull_attn3_forward, args.warmup, args.iters),
        "hull_attn": _safe_benchmark(run_hull_attn_forward, args.warmup, args.iters),
        "pytorch_sdpa": _safe_benchmark(run_sdpa_forward, args.warmup, args.iters),
    }
    backward_results = {
        "hull_attn3": _safe_benchmark(run_hull_attn3_backward, args.warmup, args.iters),
        "hull_attn": _safe_benchmark(run_hull_attn_backward, args.warmup, args.iters),
        "pytorch_sdpa": _safe_benchmark(run_sdpa_backward, args.warmup, args.iters),
    }

    print(
        f"shape={list(shape)}  dtype={dtype}  mask={args.mask}  seq_lens={args.seq_lens}  "
        f"warmup={args.warmup}  iters={args.iters}"
    )
    print("forward")
    for name, result in forward_results.items():
        print(f"  {name:12s} {_fmt(result)}")
    print("backward")
    for name, result in backward_results.items():
        print(f"  {name:12s} {_fmt(result)}")


if __name__ == "__main__":
    main()
