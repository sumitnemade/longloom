#!/usr/bin/env python3
"""Run every example and print a status matrix (run / SKIPPED / FAILED)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES = sorted(
    p for p in list(HERE.glob("0*.py")) + list(HERE.glob("1*.py"))
    if p.name[0].isdigit()
)


def main() -> int:
    ran, skipped, failed = [], [], []
    for path in EXAMPLES:
        start = time.time()
        proc = subprocess.run([sys.executable, path.name], cwd=HERE,
                              capture_output=True, text=True, timeout=600)
        dt = time.time() - start
        status = {0: "RUN", 3: "SKIP"}.get(proc.returncode, "FAIL")
        print(f"{path.name:32s} {status:4s} ({dt:5.1f}s)")
        if status == "RUN":
            ran.append(path.name)
        elif status == "SKIP":
            reason = (proc.stdout.strip().splitlines() or ["?"])[-1]
            print(f"{'':32s}      └─ {reason[:90]}")
            skipped.append(path.name)
        else:
            print(proc.stdout[-800:]); print(proc.stderr[-800:])
            failed.append(path.name)

    print(f"\n{len(ran)} ran, {len(skipped)} skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())