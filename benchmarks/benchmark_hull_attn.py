import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from quack.hull_attn import _reference_topk_attention, _streaming_full_attention_forward, hull_attn


def _maybe_import_fa4():
    try:
        from flash_attn.cute import flash_attn_func

        return flash_attn_func, None
    except Exception as exc:
        repo_root = Path(__file__).resolve().parents[2] / "flash-attention"
        if repo_root.exists():
            sys.path.insert(0, str(repo_root))
            try:
                from flash_attn.cute import flash_attn_func

                return flash_attn_func, None
            except Exception as nested_exc:
                return None, nested_exc
        return None, exc


def _benchmark_cuda(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def _benchmark_cuda_memory(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    peak_allocated = []
    peak_reserved = []
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
        peak_reserved.append(torch.cuda.max_memory_reserved())
    return {
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "peak_allocated_bytes": max(peak_allocated),
        "peak_reserved_bytes": max(peak_reserved),
    }


def _format_result(name: str, result):
    if isinstance(result, str):
        return f"{name:18s} {result}"
    if "peak_allocated_bytes" not in result:
        return (
            f"{name:18s} mean={result['mean_ms']:.3f} ms "
            f"median={result['median_ms']:.3f} ms "
            f"min={result['min_ms']:.3f} ms "
            f"max={result['max_ms']:.3f} ms"
        )
    return (
        f"{name:18s} mean={result['mean_ms']:.3f} ms "
        f"median={result['median_ms']:.3f} ms "
        f"min={result['min_ms']:.3f} ms "
        f"max={result['max_ms']:.3f} ms "
        f"peak_alloc={result['peak_allocated_bytes'] / (1024 ** 2):.2f} MiB "
        f"peak_reserved={result['peak_reserved_bytes'] / (1024 ** 2):.2f} MiB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["width-matched", "single"], default="width-matched")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--mode", choices=["full", "topk1", "topk4"], default="full")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--seq-lens", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = "cuda"
    benchmark_fn = _benchmark_cuda_memory if args.measure_memory else _benchmark_cuda
    torch.manual_seed(0)

    if args.scenario == "width-matched":
        if args.backward:
            raise ValueError("width-matched scenario is forward-only")
        if args.mode != "full":
            raise ValueError("width-matched scenario benchmarks full attention only")
        if args.mask:
            raise ValueError("width-matched scenario expects seq_lens or unmasked inputs, not dense masks")

        batch = args.batch
        seqlen = args.seqlen
        sdpa_shape = (batch, 8, seqlen, 16)
        hull_shape = (batch, 64, seqlen, 2)
        seq_lens = None
        if args.seq_lens:
            seq_lens = torch.linspace(seqlen, max(seqlen // 4, 1), batch, device=device)
            seq_lens = seq_lens.round().to(dtype=torch.int32)

        q_sdpa = torch.randn(sdpa_shape, device=device, dtype=dtype).contiguous()
        k_sdpa = torch.randn_like(q_sdpa)
        v_sdpa = torch.randn_like(q_sdpa)
        q_hull = torch.randn(hull_shape, device=device, dtype=dtype).contiguous()
        k_hull = torch.randn_like(q_hull)
        v_hull = torch.randn_like(q_hull)

        sdpa_mask = None
        if seq_lens is not None:
            key_padding_mask = torch.arange(seqlen, device=device)[None, :] < seq_lens[:, None]
            sdpa_mask = key_padding_mask[:, None, None, :]

        results = {
            "pytorch_sdpa": benchmark_fn(
                lambda: F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False
                ),
                warmup=args.warmup,
                iters=args.iters,
            ),
            "quack_hull": benchmark_fn(
                lambda: hull_attn(q_hull, k_hull, v_hull, mode="full", seq_lens=seq_lens),
                warmup=args.warmup,
                iters=args.iters,
            ),
        }
        if isinstance(results["pytorch_sdpa"], dict) and isinstance(results["quack_hull"], dict):
            results["ratio"] = (
                f"{results['quack_hull']['mean_ms'] / results['pytorch_sdpa']['mean_ms']:.2f}x "
                "slower (hull / sdpa)"
            )

        print(
            f"scenario=width-matched batch={batch} seqlen={seqlen} dtype={dtype} "
            f"sdpa_shape={sdpa_shape} hull_shape={hull_shape} seq_lens={args.seq_lens} "
            f"measure_memory={args.measure_memory}"
        )
        for name, result in results.items():
            print(_format_result(name, result))
        return

    if args.head_dim != 2:
        raise ValueError("single scenario targets hull_attn head_dim=2")

    q = torch.randn((args.batch, args.heads, args.seqlen, 2), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = None
    seq_lens = None
    if args.mask:
        mask = torch.zeros((args.batch, args.heads, args.seqlen, args.seqlen), device=device, dtype=torch.float32)
        mask[..., 0, -1] = -1e4
        mask[..., 1, :5] = -1e4
    elif args.seq_lens:
        seq_lens = torch.linspace(args.seqlen, max(args.seqlen // 4, 1), args.batch, device=device)
        seq_lens = seq_lens.round().to(dtype=torch.int32)

    results = {}

    def quack_cute_run():
        if args.backward:
            if args.mode != "full":
                raise ValueError("Sparse modes are forward-only")
            q_in = q.detach().clone().requires_grad_(True)
            k_in = k.detach().clone().requires_grad_(True)
            v_in = v.detach().clone().requires_grad_(True)
            out = hull_attn(
                q_in,
                k_in,
                v_in,
                mode=args.mode,
                attention_mask=mask,
                seq_lens=seq_lens,
            )
            out.float().square().mean().backward()
        else:
            hull_attn(q, k, v, mode=args.mode, attention_mask=mask, seq_lens=seq_lens)

    def quack_streaming_run():
        if args.backward:
            if args.mode != "full":
                raise ValueError("Sparse modes are forward-only")
            q_in = q.detach().clone().requires_grad_(True)
            k_in = k.detach().clone().requires_grad_(True)
            v_in = v.detach().clone().requires_grad_(True)
            out = _streaming_full_attention_forward(
                q_in,
                k_in,
                v_in,
                scale=1.0 / math.sqrt(2.0),
                attention_mask=mask,
                key_padding_mask=(
                    (torch.arange(args.seqlen, device=device)[None, :] < seq_lens[:, None])
                    if seq_lens is not None
                    else None
                ),
            )[0]
            out.float().square().mean().backward()
        else:
            if args.mode == "full":
                _streaming_full_attention_forward(
                    q,
                    k,
                    v,
                    scale=1.0 / math.sqrt(2.0),
                    attention_mask=mask,
                    key_padding_mask=(
                        (torch.arange(args.seqlen, device=device)[None, :] < seq_lens[:, None])
                        if seq_lens is not None
                        else None
                    ),
                )[0]
            else:
                _reference_topk_attention(
                    q,
                    k,
                    v,
                    topk=1 if args.mode == "topk1" else 4,
                    scale=1.0 / math.sqrt(2.0),
                    attention_mask=mask,
                    key_padding_mask=(
                        (torch.arange(args.seqlen, device=device)[None, :] < seq_lens[:, None])
                        if seq_lens is not None
                        else None
                    ),
                )

    def pytorch_sdpa_run():
        if args.mode != "full":
            raise ValueError("PyTorch SDPA baseline only applies to full attention")
        sdpa_mask = None
        if mask is not None:
            sdpa_mask = mask
        elif seq_lens is not None:
            key_padding_mask = torch.arange(args.seqlen, device=device)[None, :] < seq_lens[:, None]
            sdpa_mask = key_padding_mask[:, None, None, :]
        if args.backward:
            q_in = q.detach().clone().requires_grad_(True)
            k_in = k.detach().clone().requires_grad_(True)
            v_in = v.detach().clone().requires_grad_(True)
            out = F.scaled_dot_product_attention(
                q_in,
                k_in,
                v_in,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            out.float().square().mean().backward()
        else:
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=0.0,
                is_causal=False,
            )

    results["quack_cute"] = benchmark_fn(quack_cute_run, warmup=args.warmup, iters=args.iters)
    results["quack_reference"] = benchmark_fn(
        quack_streaming_run,
        warmup=args.warmup,
        iters=args.iters,
    )
    if args.mode == "full":
        results["pytorch_sdpa"] = benchmark_fn(
            pytorch_sdpa_run,
            warmup=args.warmup,
            iters=args.iters,
        )

    if args.mode != "full":
        results["fa4"] = "not applicable to sparse top-k mode"
    elif args.head_dim < 8 or args.head_dim % 8 != 0:
        results["fa4"] = "unsupported for head_dim=2"
    else:
        flash_attn_func, fa4_error = _maybe_import_fa4()
        if flash_attn_func is None:
            results["fa4"] = f"unavailable ({type(fa4_error).__name__}: {fa4_error})"
        else:
            q_fa4 = q.transpose(1, 2).contiguous()
            k_fa4 = k.transpose(1, 2).contiguous()
            v_fa4 = v.transpose(1, 2).contiguous()
            results["fa4"] = _benchmark_cuda(
                lambda: flash_attn_func(q_fa4, k_fa4, v_fa4, causal=False),
                warmup=args.warmup,
                iters=args.iters,
            )

    print(
        f"scenario=single batch={args.batch} heads={args.heads} seqlen={args.seqlen} "
        f"head_dim={args.head_dim} dtype={dtype} mode={args.mode} mask={args.mask} seq_lens={args.seq_lens} "
        f"backward={args.backward} measure_memory={args.measure_memory}"
    )
    for name, result in results.items():
        print(_format_result(name, result))


if __name__ == "__main__":
    main()
