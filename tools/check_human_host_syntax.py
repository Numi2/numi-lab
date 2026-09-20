"""Syntax-check Human host owners using the owning CMake target definitions.

This compiles translation units without linking or executing a Human trajectory.
It does not substitute fabricated material paths or relax compiler diagnostics.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    blocks = re.findall(
        r"target_compile_definitions\(\s*metalrobo_numilab_human_myosim_visual_probe\s+PRIVATE\s+(.*?)\n\)",
        cmake, re.DOTALL,
    )
    if len(blocks) != 1:
        raise SystemExit("Human target compile-definition block is missing or ambiguous")
    definitions = re.findall(r'\b(NUMI_[A-Z_]+)="([^"\n]+)"', blocks[0])
    names = [name for name, _ in definitions]
    required = {
        "NUMI_HUMAN_PASSIVE_TISSUE_MATERIAL", "NUMI_HUMAN_PECTORALIS_FASCIA_MATERIAL",
        "NUMI_HUMAN_THORACOLUMBAR_FASCIA_MATERIAL", "NUMI_HUMAN_ANTERIOR_THORAX_MATERIAL",
        "NUMI_HUMAN_REGIONAL_MYOFASCIA_MATERIAL", "NUMI_HUMAN_PLANTAR_FASCIA_MATERIAL",
        "NUMI_HUMAN_OPEN_KNEE_LIGAMENT_MATERIAL", "NUMI_MATTER_METALLIB",
    }
    if set(names) != required or len(names) != len(required):
        raise SystemExit("Human CMake material definitions changed; inspect the owning target")
    compiler = os.environ.get("CXX", "clang++")
    common = [compiler, "-std=c++20", "-fobjc-arc", "-fno-fast-math", "-fsyntax-only",
              "-I" + str(root / "include"), "-I" + str(root / "matter/include")]
    with tempfile.TemporaryDirectory(prefix="human-syntax-") as build:
        flags = []
        for name, value in definitions:
            value = value.replace("${CMAKE_CURRENT_SOURCE_DIR}", str(root))
            value = value.replace("${CMAKE_CURRENT_BINARY_DIR}", build)
            if "${" in value:
                raise SystemExit("Unresolved CMake expression in " + name)
            if name != "NUMI_MATTER_METALLIB" and not Path(value).is_file():
                raise SystemExit("Source material is missing: " + value)
            flags.append("-D" + name + "=" + json.dumps(value))
        subprocess.run([*common, str(root / "src/metal/MetalArticulatedOperator.mm")], check=True)
        subprocess.run([*common, *flags, str(root / "apps/numilab_human_myosim_visual_probe.mm")], check=True)
    print("Human host syntax checks passed with eight CMake-owned definitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
