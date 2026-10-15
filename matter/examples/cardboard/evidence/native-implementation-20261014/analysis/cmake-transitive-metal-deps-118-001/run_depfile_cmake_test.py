#!/usr/bin/env python3
"""Verify CMake consumes xcrun metal -MMD depfiles with both native generators."""
from __future__ import annotations

import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent
FIXTURE = ROOT / "cmake-fixture"


def run(args: list[str], cwd: pathlib.Path) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=90)
    return result.stdout


def main() -> None:
    (FIXTURE / "inc").mkdir(parents=True, exist_ok=True)
    (FIXTURE / "main.metal").write_text(
        '#include <metal_stdlib>\nusing namespace metal;\n'
        '#include "inc/value.metalinc"\n'
        'kernel void depfile_probe(device float* x [[buffer(0)]], '
        'uint i [[thread_position_in_grid]]) { x[i] = probe_value(); }\n',
        encoding="utf-8")
    leaf = FIXTURE / "inc/value.metalinc"
    leaf.write_text("inline float probe_value() { return 1.0f; }\n", encoding="utf-8")
    (FIXTURE / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\n"
        "project(metal_depfile_probe NONE)\n"
        "set(AIR ${CMAKE_CURRENT_BINARY_DIR}/probe.air)\n"
        "set(LIB ${CMAKE_CURRENT_BINARY_DIR}/probe.metallib)\n"
        "set(DEP ${CMAKE_CURRENT_BINARY_DIR}/probe.d)\n"
        "add_custom_command(OUTPUT ${LIB}\n"
        " COMMAND xcrun -sdk macosx metal -std=metal4.0 -MMD -MF ${DEP} -MT ${LIB}\n"
        "   -c ${CMAKE_CURRENT_SOURCE_DIR}/main.metal -o ${AIR}\n"
        " COMMAND xcrun -sdk macosx metallib ${AIR} -o ${LIB}\n"
        " DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main.metal\n"
        " DEPFILE ${DEP} VERBATIM)\n"
        "add_custom_target(probe ALL DEPENDS ${LIB})\n", encoding="utf-8")

    results: dict[str, object] = {}
    for generator, build_name in (("Ninja", "build-ninja"),
                                  ("Unix Makefiles", "build-make")):
        build = FIXTURE / build_name
        build.mkdir(exist_ok=True)
        cfg = run(["cmake", "-G", generator, "-S", str(FIXTURE), "-B", str(build)], FIXTURE)
        first = run(["cmake", "--build", str(build), "--target", "probe", "-v"], FIXTURE)
        (ROOT / f"{build_name}-configure.log").write_text(cfg, encoding="utf-8")
        (ROOT / f"{build_name}-initial-build.log").write_text(first, encoding="utf-8")
        depfile = build / "probe.d"
        if not depfile.is_file() or str(leaf) not in depfile.read_text(encoding="utf-8"):
            raise RuntimeError(f"{generator}: compiler depfile omitted included leaf")
        time.sleep(1.1)  # Ensure the filesystem timestamp advances on coarse filesystems.
        leaf.write_text("inline float probe_value() { return 2.0f; }\n", encoding="utf-8")
        second = run(["cmake", "--build", str(build), "--target", "probe", "-v"], FIXTURE)
        (ROOT / f"{build_name}-incremental-build.log").write_text(second, encoding="utf-8")
        compiler_runs = second.count("metal -std=metal4.0")
        if compiler_runs != 1:
            raise RuntimeError(f"{generator}: touched transitive include caused {compiler_runs} Metal compilations")
        results[generator] = {
            "configured": True,
            "initial_build_completed": True,
            "depfile_contains_transitive_include": True,
            "touched_include_rebuilt_metal": True,
            "metal_compiler_reinvocations_after_touch": compiler_runs,
            "cmake_configure_stdout_sha256": __import__("hashlib").sha256(cfg.encode()).hexdigest(),
            "initial_build_stdout_sha256": __import__("hashlib").sha256(first.encode()).hexdigest(),
            "incremental_build_stdout_sha256": __import__("hashlib").sha256(second.encode()).hexdigest(),
        }
    (ROOT / "cmake-test-result.json").write_text(
        json.dumps({"schema": "numi.cardboard.metal-cmake-depfile-test.v1",
                    "results": results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
