import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from quack.flash_hull_attn import flash_hull_attn
from quack.hull_attn import _reference_topk_attention, _streaming_full_attention_forward, hull_attn
from quack.hull_attn_codex import hull_attn_codex


def _maybe_import_fa4():
    try:
        from flash_attn.cute import flash_attn_func

        return flash_attn_func, None
    except Exception:
        pass
    # Try loading from sibling flash-attention repo with a stub top-level package
    # to avoid importing the C extension (flash_attn_2_cuda).
    repo_root = Path(__file__).resolve().parents[2] / "flash-attention"
    if not repo_root.exists():
        return None, FileNotFoundError(f"{repo_root} not found")
    import types

    fa_stub = types.ModuleType("flash_attn")
    fa_stub.__path__ = [str(repo_root / "flash_attn")]
    fa_stub.__version__ = "0.0.0"
    sys.modules.setdefault("flash_attn", fa_stub)
    sys.path.insert(0, str(repo_root))
    try:
        from flash_attn.cute import flash_attn_func

        return flash_attn_func, None
    except Exception as exc:
        return None, exc


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
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "peak_MiB": max(peak_allocated) / (1024**2),
    }


def _fmt(result):
    if isinstance(result, str):
        return result
    return (
        f"median={result['median_ms']:.3f} ms  "
        f"peak={result['peak_MiB']:.1f} MiB"
    )


def _fmt_row(name, r2d, r16d):
    col1 = f"{name:20s}"
    col2 = _fmt(r2d) if r2d else "—"
    col3 = _fmt(r16d) if r16d else "—"
    return f"{col1} | {col2:40s} | {col3}"


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
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = "cuda"
    bm = lambda fn: _benchmark(fn, args.warmup, args.iters)
    torch.manual_seed(0)

    if args.scenario == "width-matched":
        if args.backward:
            raise ValueError("width-matched scenario is forward-only")
        if args.mode != "full":
            raise ValueError("width-matched scenario benchmarks full attention only")
        if args.mask:
            raise ValueError("width-matched scenario expects seq_lens or unmasked, not dense masks")

        batch, seqlen = args.batch, args.seqlen
        shape_2d = (batch, 64, seqlen, 2)
        shape_16d = (batch, 8, seqlen, 16)
        seq_lens = None
        if args.seq_lens:
            seq_lens = torch.linspace(seqlen, max(seqlen // 4, 1), batch, device=device)
            seq_lens = seq_lens.round().to(dtype=torch.int32)

        sdpa_mask = None
        if seq_lens is not None:
            kpm = torch.arange(seqlen, device=device)[None, :] < seq_lens[:, None]
            sdpa_mask = kpm[:, None, None, :]

        flash_attn_func, fa4_error = _maybe_import_fa4()
        fa4_ok = flash_attn_func is not None

        def _run_2d(name):
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            q = torch.randn(shape_2d, device=device, dtype=dtype).contiguous()
            k, v = torch.randn_like(q), torch.randn_like(q)
            if name == "pytorch_sdpa":
                return bm(lambda: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False,
                ))
            if name == "fa4":
                if seq_lens is not None:
                    return "not applicable with seq_lens"
                q_f = F.pad(q.transpose(1, 2).contiguous(), (0, 6))
                k_f = F.pad(k.transpose(1, 2).contiguous(), (0, 6))
                v_f = F.pad(v.transpose(1, 2).contiguous(), (0, 6))
                sc = 1.0 / math.sqrt(2)
                return bm(lambda: flash_attn_func(
                    q_f, k_f, v_f, causal=False, softmax_scale=sc,
                )[0][..., :2])
            if name == "flash_hull":
                if seq_lens is not None:
                    return "not applicable with seq_lens"
                return bm(lambda: flash_hull_attn(q, k, v))
            if name == "quack_hull":
                return bm(lambda: hull_attn(q, k, v, mode="full", seq_lens=seq_lens))
            if name == "quack_codex":
                return bm(lambda: hull_attn_codex(q, k, v, seq_lens=seq_lens))
            return None

        def _run_16d(name):
            torch.cuda.empty_cache()
            torch.manual_seed(0)
            q = torch.randn(shape_16d, device=device, dtype=dtype).contiguous()
            k, v = torch.randn_like(q), torch.randn_like(q)
            if name == "pytorch_sdpa":
                return bm(lambda: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False,
                ))
            if name == "fa4":
                if seq_lens is not None:
                    return "not applicable with seq_lens"
                q_f = q.transpose(1, 2).contiguous()
                k_f = k.transpose(1, 2).contiguous()
                v_f = v.transpose(1, 2).contiguous()
                sc = 1.0 / math.sqrt(16)
                return bm(lambda: flash_attn_func(
                    q_f, k_f, v_f, causal=False, softmax_scale=sc,
                )[0])
            return None

        backends = ["pytorch_sdpa", "fa4", "flash_hull", "quack_hull", "quack_codex"]

        def _safe_run(fn):
            try:
                return fn()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return "OOM"

        rows = []
        for name in backends:
            if name == "fa4" and not fa4_ok:
                msg = f"unavailable ({fa4_error})"
                rows.append((name, msg, msg))
                continue
            r2d = _safe_run(lambda: _run_2d(name))
            r16d = _safe_run(lambda: _run_16d(name))
            labels = {"fa4": "fa4 (pad→8)", "flash_hull": "flash_hull_attn", "quack_codex": "hull_attn_codex"}
            rows.append((labels.get(name, name), r2d, r16d))

        hdr_2d = f"64h x 2d {list(shape_2d)}"
        hdr_16d = f"8h x 16d {list(shape_16d)}"
        print(
            f"scenario=width-matched  batch={batch}  seqlen={seqlen}  dtype={dtype}  "
            f"seq_lens={args.seq_lens}"
        )
        print(f"{'':20s} | {hdr_2d:40s} | {hdr_16d}")
        print("-" * 110)
        for name, r2d, r16d in rows:
            print(_fmt_row(name, r2d, r16d))
        return

    # ---- single scenario ----
    if args.head_dim != 2:
        raise ValueError("single scenario targets hull_attn head_dim=2")

    q = torch.randn((args.batch, args.heads, args.seqlen, 2), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = None
    seq_lens = None
    if args.mask:
        mask = torch.zeros(
            (args.batch, args.heads, args.seqlen, args.seqlen), device=device, dtype=torch.float32
        )
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
                q_in, k_in, v_in, mode=args.mode, attention_mask=mask, seq_lens=seq_lens,
            )
            out.float().square().mean().backward()
        else:
            hull_attn(q, k, v, mode=args.mode, attention_mask=mask, seq_lens=seq_lens)

    def quack_codex_run():
        if args.mode != "full":
            raise ValueError("hull_attn_codex only applies to full attention")
        if mask is not None:
            raise ValueError("hull_attn_codex does not support dense masks")
        if args.backward:
            q_in = q.detach().clone().requires_grad_(True)
            k_in = k.detach().clone().requires_grad_(True)
            v_in = v.detach().clone().requires_grad_(True)
            out = hull_attn_codex(q_in, k_in, v_in, seq_lens=seq_lens)
            out.float().square().mean().backward()
        else:
            hull_attn_codex(q, k, v, seq_lens=seq_lens)

    def quack_streaming_run():
        kpm = None
        if seq_lens is not None:
            kpm = torch.arange(args.seqlen, device=device)[None, :] < seq_lens[:, None]
        if args.backward:
            if args.mode != "full":
                raise ValueError("Sparse modes are forward-only")
            q_in = q.detach().clone().requires_grad_(True)
            k_in = k.detach().clone().requires_grad_(True)
            v_in = v.detach().clone().requires_grad_(True)
            out = _streaming_full_attention_forward(
                q_in, k_in, v_in, scale=1.0 / math.sqrt(2.0),
                attention_mask=mask, key_padding_mask=kpm,
            )[0]
            out.float().square().mean().backward()
        elif args.mode == "full":
            _streaming_full_attention_forward(
                q, k, v, scale=1.0 / math.sqrt(2.0),
                attention_mask=mask, key_padding_mask=kpm,
            )[0]
        else:
            _reference_topk_attention(
                q, k, v, topk=1 if args.mode == "topk1" else 4,
                scale=1.0 / math.sqrt(2.0), attention_mask=mask, key_padding_mask=kpm,
            )

    def pytorch_sdpa_run():
        if args.mode != "full":
            raise ValueError("PyTorch SDPA baseline only applies to full attention")
        sdpa_mask = None
        if mask is not None:
            sdpa_mask = mask
        elif seq_lens is not None:
            kpm = torch.arange(args.seqlen, device=device)[None, :] < seq_lens[:, None]
            sdpa_mask = kpm[:, None, None, :]
        if args.backward:
            q_in = q.detach().clone().requires_grad_(True)
            k_in = k.detach().clone().requires_grad_(True)
            v_in = v.detach().clone().requires_grad_(True)
            out = F.scaled_dot_product_attention(
                q_in, k_in, v_in, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False,
            )
            out.float().square().mean().backward()
        else:
            F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=0.0, is_causal=False,
            )

    results["quack_cute"] = bm(quack_cute_run)
    if args.mode == "full" and mask is None:
        results["quack_codex"] = bm(quack_codex_run)
    else:
        results["quack_codex"] = "not applicable"
    results["quack_reference"] = bm(quack_streaming_run)
    if args.mode == "full":
        results["pytorch_sdpa"] = bm(pytorch_sdpa_run)

    if args.mode != "full":
        results["fa4"] = "not applicable to sparse top-k mode"
    elif args.backward:
        results["fa4"] = "not implemented for backward"
    elif seq_lens is not None:
        results["fa4"] = "not applicable with seq_lens"
    else:
        flash_attn_func, fa4_error = _maybe_import_fa4()
        if flash_attn_func is None:
            results["fa4"] = f"unavailable ({type(fa4_error).__name__}: {fa4_error})"
        else:
            pad_dim = max(8, args.head_dim) - args.head_dim
            q_fa4 = q.transpose(1, 2).contiguous()
            k_fa4 = k.transpose(1, 2).contiguous()
            v_fa4 = v.transpose(1, 2).contiguous()
            if pad_dim > 0:
                q_fa4 = F.pad(q_fa4, (0, pad_dim))
                k_fa4 = F.pad(k_fa4, (0, pad_dim))
                v_fa4 = F.pad(v_fa4, (0, pad_dim))
            fa4_scale = 1.0 / math.sqrt(args.head_dim)

            def fa4_run():
                out = flash_attn_func(
                    q_fa4, k_fa4, v_fa4, causal=False, softmax_scale=fa4_scale
                )[0]
                if pad_dim > 0:
                    return out[..., :args.head_dim]
                return out

            results["fa4"] = bm(fa4_run)

    if args.mode == "full" and not args.backward and seq_lens is None:
        results["flash_hull_attn"] = bm(lambda: flash_hull_attn(q, k, v))
    elif args.mode == "full" and seq_lens is not None:
        results["flash_hull_attn"] = "not applicable with seq_lens"
    elif args.mode == "full":
        results["flash_hull_attn"] = "not implemented for backward"

    print(
        f"scenario=single  batch={args.batch}  heads={args.heads}  seqlen={args.seqlen}  "
        f"head_dim={args.head_dim}  dtype={dtype}  mode={args.mode}  mask={args.mask}  "
        f"seq_lens={args.seq_lens}  backward={args.backward}"
    )
    for name, result in results.items():
        print(f"  {name:20s} {_fmt(result)}")


if __name__ == "__main__":
    main()
