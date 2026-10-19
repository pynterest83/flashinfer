"""SASS size, registers and spills of BatchMLAPageAttentionHopperKernel in JIT modules.

    python mla_fp8_repack_sass.py label=path/to/module.so [label=...]

Prints one CSV row per (module, mask mode): instruction count, SASS KiB (16 B per
instruction), registers, stack, local spill, and HMUL2/PRMT/LDS/STS counts.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys

KERNEL = "BatchMLAPageAttentionHopperKernel"


def stats(label: str, so: str) -> list[dict]:
    sass = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True,
                          check=True).stdout
    res = subprocess.run(["cuobjdump", "-res-usage", so], capture_output=True, text=True,
                         check=True).stdout
    counts: dict[str, dict[str, int]] = {}
    fn = None
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            fn = m.group(1) if KERNEL in m.group(1) else None
            continue
        if fn:
            m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
            if m:
                ops = counts.setdefault(fn, {})
                ops["_all"] = ops.get("_all", 0) + 1
                op = m.group(1).split(".")[0]
                ops[op] = ops.get(op, 0) + 1
    rows = []
    for fn, ops in counts.items():
        mm = re.search(re.escape(fn) + r":\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:\d+ LOCAL:(\d+)", res)
        rows.append({
            "module": label, "mask": "causal" if "ILb1E" in fn else "noncausal",
            "instrs": ops["_all"], "sass_kib": round(ops["_all"] * 16 / 1024, 1),
            "regs": int(mm.group(1)), "stack": int(mm.group(2)), "local_spill": int(mm.group(3)),
            **{k: ops.get(k, 0) for k in ("HMUL2", "PRMT", "LDS", "STS")},
        })
    return rows


def main() -> None:
    rows = []
    for arg in sys.argv[1:]:
        label, so = arg.split("=", 1)
        rows += stats(label, so)
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


if __name__ == "__main__":
    main()
