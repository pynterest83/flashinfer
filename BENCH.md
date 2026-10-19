# SM90 MLA FP8-KV repack: before/after

`BatchMLAPageAttentionHopperKernel` (include/flashinfer/attention/mla_hopper.cuh) with an
E4M3 KV cache, stock FlashInfer 0.7.0.post1 headers vs this branch. One H200 (SXM, 141 GB),
CUDA 13.0 JIT for sm_90a, 2026-10-19.

Shape: GLM-5.3-Flash TP4 sparse decode, the production vLLM call
(`BatchMLAPagedAttentionWrapper(backend="fa3")`, page_size 1): 16 heads, rope-free
512 latent (`head_dim_kpe` 0), bf16 query/output, one query per row, 2048-2051 selected
slots per row scattered over the cache, `ckv_scale = kpe_scale = 1.0`. Per-launch CUPTI
kernel time (torch.profiler), 20 warm-up + 100 timed launches, median.

Reproduce (one process per header tree, each with its own JIT workspace):

    python tests/attention/mla_fp8_repack_cases.py bench --include <include root> \
        --workspace <dir> --out bench.json [--kv bf16]
    python benchmarks/mla_fp8_repack_sass.py stock=<stock .so> patched=<patched .so>

Median kernel time, cold L2 (64 MiB displaced before each launch), us:

| rows | cache tokens | BF16 KV | FP8 stock | FP8 patched | FP8 patched (rep) | patched vs stock |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | 16.7 | 41.2 | 21.6 | 21.0 | -48% |
| 6 | 8192 | 17.0 | 52.5 | 26.7 | 26.7 | -49% |
| 24 | 8192 | 23.4 | 73.3 | 41.0 | 41.1 | -44% |
| 96 | 8192 | 70.1 | 254.3 | 139.2 | 140.0 | -45% |
| 192 | 8192 | 122.3 | 444.8 | 259.2 | 259.4 | -42% |
| 1 | 32768 | 17.9 | 40.4 | 21.4 | 21.9 | -47% |
| 6 | 32768 | 17.2 | 52.3 | 26.9 | 27.0 | -48% |
| 24 | 32768 | 24.0 | 76.1 | 41.9 | 41.8 | -45% |
| 96 | 32768 | 71.5 | 250.6 | 140.8 | 141.0 | -44% |
| 192 | 32768 | 122.7 | 449.8 | 260.6 | 261.2 | -42% |
| 1 | 98304 | 17.0 | 40.7 | 21.7 | 21.4 | -47% |
| 6 | 98304 | 17.3 | 52.2 | 26.9 | 26.8 | -48% |
| 24 | 98304 | 24.7 | 76.2 | 42.2 | 42.4 | -45% |
| 96 | 98304 | 74.4 | 257.4 | 144.5 | 144.4 | -44% |
| 192 | 98304 | 127.4 | 459.6 | 266.9 | 267.0 | -42% |

Median kernel time, warm L2 (back-to-back), us:

| rows | cache tokens | BF16 KV | FP8 stock | FP8 patched | FP8 patched (rep) | patched vs stock |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | 15.1 | 24.8 | 16.6 | 16.5 | -33% |
| 6 | 8192 | 15.8 | 39.6 | 22.5 | 22.4 | -43% |
| 24 | 8192 | 21.8 | 60.8 | 36.6 | 36.3 | -40% |
| 96 | 8192 | 66.7 | 228.2 | 128.4 | 128.4 | -44% |
| 192 | 8192 | 118.5 | 441.6 | 242.0 | 242.0 | -45% |
| 1 | 32768 | 15.5 | 24.8 | 16.9 | 16.9 | -32% |
| 6 | 32768 | 15.5 | 39.3 | 22.5 | 22.4 | -43% |
| 24 | 32768 | 21.9 | 60.7 | 36.4 | 36.3 | -40% |
| 96 | 32768 | 70.6 | 220.9 | 128.5 | 127.9 | -42% |
| 192 | 32768 | 121.5 | 441.3 | 242.0 | 242.2 | -45% |
| 1 | 98304 | 15.5 | 24.8 | 16.9 | 17.0 | -32% |
| 6 | 98304 | 15.9 | 40.0 | 22.6 | 22.6 | -43% |
| 24 | 98304 | 22.6 | 60.8 | 36.4 | 36.4 | -40% |
| 96 | 98304 | 74.7 | 243.3 | 139.7 | 139.9 | -43% |
| 192 | 98304 | 127.2 | 450.8 | 262.9 | 264.0 | -42% |

"patched (rep)" is an independent second process. The trial variant that dropped
`ckv_scale_arr` support and had no proxy fence (u2sf) measured 19.6-20.9 / 25.3-25.9 /
136.5-140.9 us at 1 / 6 / 96 rows cold, i.e. the retained fallback path and the fence
cost about 1-1.5 us cold and nothing measurable warm.

SASS of the kernel (causal instantiation; `cuobjdump`, 16 B per instruction):

| module | stock KiB | patched KiB | stock regs | patched regs | spills |
|---|---:|---:|---:|---:|---:|
| FP8 KV, head_dim_kpe 0 | 213.6 | 100.8 | 232 | 216 | 0 / 0 |
| FP8 KV, head_dim_kpe 64 | 228.6 | 124.4 | 232 | 234 | 0 / 0 |
| BF16 KV (unchanged) | 80.2 | 80.2 | 214 | 214 | 0 / 0 |

Why: the stock repack loop is fully unrolled (16 chunks per thread) and inlined at three
call sites, with the per-chunk `ckv_scale_arr` branch and a second multiply per element
pair; the kernel became instruction-fetch bound (cold-L2 cost of one 1-row launch 41 us
vs 17 us for BF16 KV). Most of the fixed cost goes away; the per-tile cost of the
repack itself remains (FP8 is still 1.6-2x BF16 at 6-192 rows).

Correctness (tests/attention/test_mla_hopper_fp8_repack.py, 11 cases: GLM sparse decode
at 6/96/192 rows with scales 1.0, 0.37, 300 (unfoldable, generic path) and a scale array;
DeepSeek-style page_size 64 with a 64-wide rope part, causal and not, scales 1.0,
0.37/0.5 and a scale array):

- bitwise equal to the stock kernel on every case;
- bitwise equal to the BF16-KV kernel on the exactly dequantized cache on every case;
- 50 repeated calls bitwise identical to the first on every case.

Without the `fence.proxy.async` the partially unrolled kernel fails the last check on
the GLM scale-array case (50/50 repeats differ, and the result differs from the stock
kernel); a 4x-unrolled variant failed it as well, on per-tensor scales, in an earlier
trial. With the fence every variant tried (2x and 4x unroll) passes.
