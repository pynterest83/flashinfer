"""Regression tests for the SM90 MLA FP8-KV shared-memory repack (mla_hopper.cuh).

For every case in mla_fp8_repack_cases.CASES (GLM-5.3-Flash-style sparse decode with a
rope-free 512 latent, DeepSeek-style paged MLA with a 64-wide rope part; per-tensor
scales 1.0, 0.37 and 300 (not foldable), and a per-(token, 128-column) scale array):

* the FP8-KV kernel built from this checkout's headers is bitwise equal to the one
  built from the stock headers (the installed wheel's ``data/include``);
* it is bitwise equal to the BF16-KV kernel run on the exactly dequantized cache;
* REPEATS repeated calls are bitwise identical to the first (determinism).

Each header tree is compiled and run in its own subprocess with its own FlashInfer
JIT workspace (FlashInfer reads its include root per process). Set
FLASHINFER_FP8_REPACK_TEST_DIR to keep the workspaces/outputs between runs, and
FLASHINFER_FP8_REPACK_REFERENCE_INCLUDE to compare against another header tree.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO_INCLUDE = HERE.parents[1] / "include"
sys.path.insert(0, str(HERE))
import mla_fp8_repack_cases as cases  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 MLA kernel needs an SM90 GPU",
)


def _run_worker(include: Path, workdir: Path, tag: str) -> dict:
    out = workdir / f"outputs_{tag}.pt"
    if not out.exists():
        subprocess.run(
            [
                sys.executable,
                str(HERE / "mla_fp8_repack_cases.py"),
                "outputs",
                "--include",
                str(include),
                "--workspace",
                str(workdir / f"ws_{tag}"),
                "--out",
                str(out),
            ],
            check=True,
        )
    return torch.load(out)


@pytest.fixture(scope="module")
def outputs() -> tuple[dict, dict]:
    root = os.environ.get("FLASHINFER_FP8_REPACK_TEST_DIR")
    tmp = None if root else tempfile.TemporaryDirectory()
    workdir = Path(root or tmp.name)
    workdir.mkdir(parents=True, exist_ok=True)
    reference = Path(
        os.environ.get("FLASHINFER_FP8_REPACK_REFERENCE_INCLUDE")
        or cases.installed_include_dir()
    )
    try:
        yield (
            _run_worker(reference, workdir, "reference"),
            _run_worker(REPO_INCLUDE, workdir, "patched"),
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


CASE_NAMES = [c.name for c in cases.CASES]


@pytest.mark.parametrize("name", CASE_NAMES)
def test_bitwise_equal_to_reference_headers(outputs, name):
    reference, patched = outputs
    assert torch.equal(patched[name]["fp8"], reference[name]["fp8"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_bitwise_equal_to_bf16_kernel_on_dequantized_cache(outputs, name):
    _, patched = outputs
    assert torch.equal(patched[name]["fp8"], patched[name]["bf16_on_dequant"])


@pytest.mark.parametrize("name", CASE_NAMES)
def test_repeated_calls_are_deterministic(outputs, name):
    _, patched = outputs
    assert patched[name]["repeat_mismatches"] == 0, (
        f"{patched[name]['repeat_mismatches']}/{patched[name]['repeats']} repeated calls differ"
    )
