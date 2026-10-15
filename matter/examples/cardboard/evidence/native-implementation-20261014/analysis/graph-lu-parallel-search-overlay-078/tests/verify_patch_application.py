#!/usr/bin/env python3
"""Check that both packaged patches reproduce their exact overlay files."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
BASE = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-stable-free-graph-and-general-arithmetic-076")
CHECKER_BASE = Path("/Users/home/cardboard-implementation-20261014/analysis/selector5-iterative-refinement-check-004")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="graph-lu-patch-check-") as temp:
        root = Path(temp)
        for rel in ("matter/src/metal/fgmres.metalinc", "matter/src/runtime.mm"):
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE / rel, dest)
        checker_name = "selector5_refinement_kernels.mm"
        shutil.copyfile(CHECKER_BASE / checker_name, root / checker_name)

        subprocess.run(["git", "apply", "--check", str(PACKAGE / "source-overlay.patch")],
                       cwd=root, check=True)
        subprocess.run(["git", "apply", "--check", str(PACKAGE / "checker/extend-prepared-input-parity.patch")],
                       cwd=root, check=True)
        subprocess.run(["git", "apply", str(PACKAGE / "source-overlay.patch")],
                       cwd=root, check=True)
        subprocess.run(["git", "apply", str(PACKAGE / "checker/extend-prepared-input-parity.patch")],
                       cwd=root, check=True)

        pairs = (
            (root / "matter/src/metal/fgmres.metalinc", PACKAGE / "overlay/matter/src/metal/fgmres.metalinc"),
            (root / "matter/src/runtime.mm", PACKAGE / "overlay/matter/src/runtime.mm"),
            (root / checker_name, PACKAGE / "checker" / checker_name),
        )
        for applied, expected in pairs:
            if applied.read_bytes() != expected.read_bytes():
                raise SystemExit(f"applied patch differs from packaged overlay: {expected}")
            print(f"PASS {applied.relative_to(root)} sha256={sha(applied)}")


if __name__ == "__main__":
    main()
