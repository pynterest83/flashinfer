"""Cases and a subprocess worker for the SM90 MLA FP8-KV repack tests and benchmark.

The Hopper MLA kernel (include/flashinfer/attention/mla_hopper.cuh) is compiled by
FlashInfer's JIT from ``flashinfer.jit.env.FLASHINFER_INCLUDE_DIR``. To compare two
header trees (e.g. the stock wheel headers and this checkout's ``include/``) each one
runs in its own process with its own ``FLASHINFER_WORKSPACE_BASE``, so neither JIT
cache is mixed with the other or with the user's default ``~/.cache/flashinfer``.

Worker usage (one header tree per process):

    python mla_fp8_repack_cases.py outputs --include DIR --workspace DIR --out FILE.pt
    python mla_fp8_repack_cases.py bench   --include DIR --workspace DIR --out FILE.json

``outputs`` runs every case in CASES once, saves the outputs, and records how many of
REPEATS further calls differ bitwise from the first (determinism). ``bench`` times the
GLM-5.3-Flash sparse-decode shapes (CUPTI kernel time, L2 displaced before each launch).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

KERNEL_NAME = "BatchMLAPageAttentionHopperKernel"
SM_SCALE = 0.0625
REPEATS = 50


def activate(include_dir: str | os.PathLike, workspace: str | os.PathLike) -> None:
    """Point FlashInfer's JIT at ``include_dir`` with a private JIT workspace.

    Must run before anything imports flashinfer (the workspace is read at import).
    """
    if "flashinfer" in sys.modules:
        raise RuntimeError("activate() must run before flashinfer is imported")
    include_dir = Path(include_dir).resolve()
    if not (include_dir / "flashinfer" / "attention" / "mla_hopper.cuh").is_file():
        raise FileNotFoundError(f"{include_dir} is not a FlashInfer include root")
    os.environ["FLASHINFER_WORKSPACE_BASE"] = str(Path(workspace).resolve())
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "9.0a")
    import flashinfer.jit.env as jit_env

    jit_env.FLASHINFER_INCLUDE_DIR = include_dir


def installed_include_dir() -> Path:
    """The installed wheel's headers, located without importing flashinfer."""
    import importlib.util

    spec = importlib.util.find_spec("flashinfer")
    assert spec is not None and spec.submodule_search_locations
    return Path(spec.submodule_search_locations[0]) / "data" / "include"


@dataclass(frozen=True)
class Case:
    """One planned BatchMLAPagedAttentionWrapper(backend="fa3") call."""

    name: str
    num_rows: int  # requests
    q_len: int  # queries per request
    kv_lens: tuple[int, ...]  # per request (cycled)
    num_cache_tokens: int
    page_size: int
    head_dim_kpe: int
    causal: bool
    ckv_scale: float | None  # None: ckv_scale_arr
    kpe_scale: float
    num_heads: int = 16
    seed: int = 0

    @property
    def scale_mode(self) -> str:
        return "array" if self.ckv_scale is None else f"tensor{self.ckv_scale:g}"


def _glm(name: str, rows: int, n: int, ckv_scale: float | None, **kw) -> Case:
    # GLM-5.3-Flash TP4 sparse decode: 16 heads, rope-free 512 latent, one query per
    # row, 2048 + (context mod 4) selected slots scattered over the cache.
    return Case(
        name, rows, 1, (2048, 2049, 2050, 2051), n, 1, 0, False, ckv_scale, 1.0, **kw
    )


def _dsv3(name: str, q_len: int, causal: bool, ckv: float | None, kpe: float) -> Case:
    # DeepSeek-style MLA with a 64-wide rope part (exercises the KPE repack).
    return Case(
        name,
        8,
        q_len,
        (1, 63, 64, 65, 700, 2047, 3001, 4096),
        16384,
        64,
        64,
        causal,
        ckv,
        kpe,
    )


CASES: tuple[Case, ...] = (
    _glm("glm_q6_s1", 6, 32768, 1.0),
    _glm("glm_q6_s037", 6, 32768, 0.37),
    _glm("glm_q96_s1", 96, 98304, 1.0),
    _glm("glm_q96_s037", 96, 98304, 0.37),
    _glm("glm_q192_s1", 192, 8192, 1.0),
    _glm("glm_q192_s037", 192, 8192, 0.37),
    # Not foldable in BF16 (2^120 * 300 overflows): generic per-tensor path.
    _glm("glm_q96_s300", 96, 32768, 300.0),
    # Per-(token, 128-column) scale array: generic path.
    _glm("glm_q96_arr", 96, 32768, None),
    _dsv3("dsv3_q1_s1", 1, False, 1.0, 1.0),
    _dsv3("dsv3_q4c_s037", 4, True, 0.37, 0.5),
    _dsv3("dsv3_q4c_arr", 4, True, None, 0.75),
)


@dataclass
class Operands:
    wrapper: object
    q_nope: object
    q_pe: object
    ckv_fp8: object
    kpe_fp8: object
    ckv_scale_arr: object | None
    ckv_deq: object  # exact BF16 dequantization including the scale
    kpe_deq: object
    extra: dict = field(default_factory=dict)


def build(case: Case, kv_dtype, device="cuda") -> Operands:
    """Plan ``case`` for ``kv_dtype`` and draw its seeded operands."""
    import torch
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    g = torch.Generator(device=device).manual_seed(20261019 + case.seed)
    gc = torch.Generator().manual_seed(20261019 + case.seed)
    kv_lens = [case.kv_lens[i % len(case.kv_lens)] for i in range(case.num_rows)]
    ps = case.page_size
    num_pages = case.num_cache_tokens // ps
    pages_per_row = [-(-n // ps) for n in kv_lens]
    if ps == 1:
        # Sparse decode: each row picks distinct slots scattered over the cache.
        indices = [torch.randperm(num_pages, generator=gc)[:n] for n in kv_lens]
    else:
        perm = torch.randperm(num_pages, generator=gc)
        assert sum(pages_per_row) <= num_pages
        offs = [0]
        for p in pages_per_row:
            offs.append(offs[-1] + p)
        indices = [perm[offs[i] : offs[i + 1]] for i in range(case.num_rows)]
    kv_indices = torch.cat(indices).to(torch.int32).to(device)
    kv_indptr = (
        torch.tensor([0] + pages_per_row, dtype=torch.int32).cumsum(0).int().to(device)
    )
    qo_indptr = (
        torch.arange(case.num_rows + 1, dtype=torch.int32, device=device) * case.q_len
    )
    kv_len_arr = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa3")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            case.num_heads,
            512,
            case.head_dim_kpe,
            ps,
            case.causal,
            SM_SCALE,
            torch.bfloat16,
            kv_dtype,
        )

    nq = case.num_rows * case.q_len
    q_nope = torch.randn(nq, case.num_heads, 512, generator=g, device=device).bfloat16()
    q_pe = torch.randn(
        nq, case.num_heads, case.head_dim_kpe, generator=g, device=device
    ).bfloat16()
    # ~N(0, 0.6) rounded to e4m3: both signs, subnormals, and the full exponent range.
    ckv_fp8 = (torch.randn(num_pages, ps, 512, generator=g, device=device) * 0.6).to(
        torch.float8_e4m3fn
    )
    kpe_fp8 = (
        torch.randn(num_pages, ps, case.head_dim_kpe, generator=g, device=device) * 0.6
    ).to(torch.float8_e4m3fn)
    bf = torch.bfloat16
    if case.ckv_scale is None:
        arr = torch.rand(num_pages, ps, 4, generator=g, device=device) * 1.75 + 0.25
        # The kernel's array path: bf16(float(bf16(x)) * scale), one rounding.
        ckv_deq = (ckv_fp8.float() * arr.repeat_interleave(128, dim=-1)).to(bf)
    else:
        arr = None
        ckv_deq = (ckv_fp8.to(bf) * torch.tensor(case.ckv_scale, dtype=bf)).to(bf)
    kpe_deq = (kpe_fp8.to(bf) * torch.tensor(case.kpe_scale, dtype=bf)).to(bf)
    return Operands(wrapper, q_nope, q_pe, ckv_fp8, kpe_fp8, arr, ckv_deq, kpe_deq)


def run_fp8(case: Case, ops: Operands):
    kw = {"kpe_scale": case.kpe_scale}
    if ops.ckv_scale_arr is None:
        kw["ckv_scale"] = case.ckv_scale
    else:
        kw["ckv_scale_arr"] = ops.ckv_scale_arr
    return ops.wrapper.run(ops.q_nope, ops.q_pe, ops.ckv_fp8, ops.kpe_fp8, **kw)


def run_bf16(ops: Operands):
    return ops.wrapper.run(ops.q_nope, ops.q_pe, ops.ckv_deq, ops.kpe_deq)


def outputs_main(out: Path) -> None:
    """Every case: FP8 output, BF16-kernel output on the dequantized cache, and the
    number of REPEATS repeated FP8 calls that differ bitwise from the first."""
    import torch

    result = {}
    for case in CASES:
        ops = build(case, torch.float8_e4m3fn)
        first = run_fp8(case, ops).clone()
        mismatched = 0
        for _ in range(REPEATS):
            mismatched += int(not torch.equal(run_fp8(case, ops), first))
        bf16_ops = build(case, torch.bfloat16)
        ref = run_bf16(bf16_ops)
        torch.cuda.synchronize()
        result[case.name] = {
            "case": asdict(case),
            "fp8": first.cpu(),
            "bf16_on_dequant": ref.cpu(),
            "repeat_mismatches": mismatched,
            "repeats": REPEATS,
        }
        print(
            f"{case.name}: repeat_mismatches={mismatched}/{REPEATS} "
            f"fp8==bf16(dequant): {torch.equal(first, ref)}",
            flush=True,
        )
        del ops, bf16_ops
        torch.cuda.empty_cache()
    torch.save(result, out)


BENCH_QUERIES = (1, 6, 24, 96, 192)
BENCH_CONTEXTS = (8192, 32768, 98304)


def time_us(fn, *, flush_l2: bool, warmup: int = 20, reps: int = 100) -> list[float]:
    """Per-launch CUPTI kernel times (torch.profiler) of KERNEL_NAME."""
    import torch

    flush = torch.empty(
        64 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda"
    ).normal_()
    for _ in range(warmup):
        if flush_l2:
            flush.sum()
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(reps):
            if flush_l2:
                flush.sum()
            fn()
        torch.cuda.synchronize()
    us = [
        e.device_time_range.elapsed_us()
        if hasattr(e, "device_time_range")
        else e.time_range.elapsed_us()
        for e in prof.events()
        if KERNEL_NAME in e.name
    ]
    if len(us) != reps:
        raise RuntimeError(f"expected {reps} {KERNEL_NAME} records, got {len(us)}")
    return sorted(us)


def bench_main(out: Path, kv: str) -> None:
    import torch

    kv_dtype = {"e4m3": torch.float8_e4m3fn, "bf16": torch.bfloat16}[kv]
    rows = []
    for n in BENCH_CONTEXTS:
        for q in BENCH_QUERIES:
            case = _glm(f"bench_q{q}_n{n}", q, n, 1.0)
            ops = build(case, kv_dtype)
            if kv == "e4m3":
                fn = lambda c=case, o=ops: run_fp8(c, o)  # noqa: E731
            else:
                fn = lambda o=ops: run_bf16(o)  # noqa: E731
            cold = time_us(fn, flush_l2=True)
            warm = time_us(fn, flush_l2=False)
            rows.append(
                {
                    "kv": kv,
                    "queries": q,
                    "ctx": n,
                    "cold_median_us": statistics.median(cold),
                    "cold_p10_us": cold[10],
                    "cold_p90_us": cold[90],
                    "warm_median_us": statistics.median(warm),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            del ops, fn
            torch.cuda.empty_cache()
    out.write_text(json.dumps(rows, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["outputs", "bench", "build"])
    ap.add_argument("--include", required=True)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out")
    ap.add_argument("--kv", default="e4m3", choices=["e4m3", "bf16"])
    a = ap.parse_args()
    activate(a.include, a.workspace)
    if a.mode == "build":  # CPU-only JIT build of every module the cases use
        import torch
        from flashinfer.jit.attention.modules import gen_batch_mla_module

        for kpe in sorted({c.head_dim_kpe for c in CASES}):
            for kv_dtype in (torch.float8_e4m3fn, torch.bfloat16):
                spec = gen_batch_mla_module(
                    "fa3",
                    torch.bfloat16,
                    kv_dtype,
                    torch.bfloat16,
                    torch.int32,
                    512,
                    kpe,
                    False,
                )
                spec.build(verbose=False)
                print("built", spec.jit_library_path, flush=True)
        return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "outputs":
        outputs_main(out)
    else:
        bench_main(out, a.kv)


if __name__ == "__main__":
    main()
