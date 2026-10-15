#!/usr/bin/env python3
"""Audit host wiring and execute the exact selector-kernel body on CPU.

The Metal function body is extracted verbatim from the selected source file;
only Metal qualifiers/attributes and the surrounding ABI types are adapted so
the same predicate can be compiled as ordinary C++. The host parser and
transaction ordering are source audits, not behavioral runtime tests.
"""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get(
    "SOURCE142",
    "/Users/home/cardboard-implementation-20261014/snapshots/source-bounded-native-cgls-mask-repair-142",
))
RUNTIME = (ROOT / "matter/src/runtime.mm").read_text()
FEM = (BASE / "matter/src/metal/fem.metalinc").read_text()
CGLS_RUNTIME = (ROOT / "matter/src/metal/base_cgls_runtime.metalinc").read_text()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def source_function(text, name, next_marker):
    start = text.index(name)
    end = text.index(next_marker, start)
    return text[start:end]


def extract_kernel(text, name):
    marker = f"kernel void {name}("
    start = text.index(marker)
    opening = text.index("{", start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise AssertionError(f"unterminated kernel {name}")


def cpu_compile_and_execute_kernel():
    compiler = shutil.which("c++") or shutil.which("clang++")
    require(compiler is not None, "no C++ compiler available for selector-body check")
    kernel = extract_kernel(
        CGLS_RUNTIME, "nm_fgmres_select_base_cgls_primary_policy")
    # These are syntax adaptations only; the predicate/body remains extracted
    # verbatim from the actual Metal source.
    kernel = kernel.replace("kernel void ", "void ", 1)
    kernel = re.sub(r"\[\[[^\]]*\]\]", "", kernel)
    kernel = re.sub(r"\b(constant|device)\s+", "", kernel)
    kernel = re.sub(r"\buint\b", "unsigned", kernel)

    harness = r'''#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
using std::isfinite;
struct float4 { float x, y, z, w; };
struct NMMatterDispatchGPU { unsigned environmentCount, objectCount; };
struct NMMixedSolverGPU { float4 residualTolerances; };
struct NMFGMRESStateGPU { float4 diagnostics; };
struct NMMatterStatusGPU { bool healthy; };
bool environmentHealthy(const NMMatterStatusGPU* s, unsigned e) {
    return s[e].healthy;
}
'''
    harness += kernel
    harness += r'''
static int run(unsigned idx, unsigned start, bool healthy, float certified,
               float baseNorm, float bound, unsigned objects=1u,
               unsigned count=1u, float priorMarker=0.0f,
               unsigned environment=0u) {
    NMMatterDispatchGPU dispatch{count, objects};
    NMMixedSolverGPU solver{{bound, 0.0f, 0.0f, 0.0f}};
    NMFGMRESStateGPU states[1]{{{0.0f, 0.0f, certified, 0.0f}}};
    NMMatterStatusGPU statuses[1]{{healthy}};
    float4 metrics[1]{{baseNorm, 0.0f, 0.0f, priorMarker}};
    nm_fgmres_select_base_cgls_primary_policy(
        dispatch, solver, start, idx, metrics, states, statuses, environment);
    if (environment >= count) return metrics[0].w == priorMarker ? 0 : -1;
    return metrics[0].w == 3.0f ? 1 : (metrics[0].w == priorMarker ? 0 : -1);
}
int main() {
    if (run(47u, 48u, true, 1.0f, 2.0f, 1.0f) != 0) return 1;
    if (run(48u, 48u, true, 1.0f, 2.0f, 1.0f) != 1) return 2;
    if (run(49u, 48u, true, 1.0f, 2.0f, 1.0f) != 1) return 3;
    if (run(48u, 48u, false, 1.0f, 2.0f, 1.0f) != 0) return 4;
    if (run(48u, 48u, true, 0.0f, 2.0f, 1.0f) != 0) return 5;
    if (run(48u, 48u, true, 1.0f, 1.0f, 1.0f) != 0) return 6;
    if (run(48u, 48u, true, 1.0f, 0.5f, 1.0f) != 0) return 7;
    if (run(48u, 48u, true, 1.0f,
            std::numeric_limits<float>::quiet_NaN(), 1.0f) != 0) return 8;
    if (run(48u, 48u, true, 1.0f, 2.0f, 1.0f, 2u) != 0) return 9;
    if (run(48u, 48u, true, 1.0f, 2.0f, 1.0f, 1u, 1u, 2.0f) != 0) return 10;
    if (run(48u, 48u, true, 1.0f, 2.0f, 1.0f, 1u, 0u) != 0) return 11;
    if (run(48u, 48u, true, 1.0f, 2.0f, 1.0f, 1u, 1u, 0.0f, 1u) != 0) return 12;
    if (run(48u, 48u, true, 1.0f, 2.0f,
            std::numeric_limits<float>::infinity()) != 0) return 13;
    std::puts("PASS extracted production selector kernel CPU cases");
    return 0;
}
'''
    with tempfile.TemporaryDirectory(prefix="base-cgls-selector-") as temp:
        source = Path(temp) / "selector.cpp"
        executable = Path(temp) / "selector-check"
        source.write_text(harness)
        built = subprocess.run(
            [compiler, "-std=c++17", "-O0", str(source), "-o", str(executable)],
            text=True, capture_output=True, check=False)
        require(built.returncode == 0,
                "extracted selector kernel did not compile:\n" + built.stderr)
        ran = subprocess.run([str(executable)], text=True, capture_output=True,
                             check=False)
        require(ran.returncode == 0,
                f"extracted selector kernel CPU cases failed ({ran.returncode}): {ran.stderr}")
        print(ran.stdout.strip())


def source_contract_audit():
    # Configuration is default-off, canonical, and restricted to the existing
    # bounded CGLS root:all scope. These checks inspect production host source;
    # they do not emulate its parser or claim runtime configuration coverage.
    require('"NM_FEM_BASE_CGLS_PRIMARY_AFTER"' in RUNTIME,
            "primary selector is not wired into runtime configuration")
    require("std::to_string(parsed) != primaryAfter" in RUNTIME and
            "parsed > 256ul" in RUNTIME,
            "primary selector canonical/range checks are absent")
    require("requires_cgls_iterations_and_control_step_all_newton_selector" in RUNTIME and
            "femBaseCGLSIterations == 0u" in RUNTIME and
            "!femBaseCauchyFallbackAllNewtonIterations" in RUNTIME,
            "primary selector does not fail closed outside CGLS root:all mode")
    require('"nm_fgmres_select_base_cgls_primary_policy"' in RUNTIME,
            "new selector kernel is not registered with the runtime pipeline set")
    require("healthy_base_above_noop_bound_with_original_linear_certificate" in RUNTIME,
            "enabled log does not state the exact selection predicate")

    armijo = source_function(
        RUNTIME,
        "encodeFEMResidualArmijoBaseStatus();\n                if (baseCGLSPrimaryPolicySelectedNewton)",
        "} else {\n                dispatchThreads(\"nm_fem_select_backtracking\"",
    )
    ordered = (
        "encodeFEMResidualArmijoBaseStatus();",
        '"nm_fgmres_select_base_cgls_primary_policy"',
        "encodeBaseCauchyDirection();",
        "encodeBaseCauchyDirectionTrace(true)",
        "!baseCGLSPrimaryPolicySelectedNewton ? 2u : 1u",
        "!baseCGLSPrimaryPolicySelectedNewton &&",
        '"base_cgls_primary_policy"',
    )
    positions = [armijo.index(item) for item in ordered]
    require(positions == sorted(positions),
            "selection/installation/ordinary Armijo order changed")
    require("baseCauchyNewtonSelected &&\n                        !baseCGLSPrimaryPolicySelectedNewton &&\n                        directionPass == 0u ? 1u : 0u" in armijo,
            "primary policy permits a synthetic second fallback pass")
    require("NM_FEM_BASE_CGLS_PRIMARY_AFTER" not in RUNTIME.split(
        "const bool baseCauchyNewtonSelected =", 1)[1].split(
        "if (baseCauchyNewtonSelected) {", 1)[0],
        "legacy exact-index selection was modified")

    decide = source_function(
        FEM,
        "kernel void nm_fem_armijo_decide(",
        "kernel void nm_fem_synchronize_environment_line_search(",
    )
    require("const bool certifiedNoop" in decide and
            "state.x <= certificateResidualBound" in decide and
            "state.y <= certificateResidualBound" in decide,
            "certified no-op branch is absent from the unchanged selected FEM source")
    install = source_function(
        CGLS_RUNTIME,
        "kernel void nm_fgmres_install_base_cgls_direction(",
        "} // namespace numi_matter_metal",
    )
    require("originalCertified = states[environment].diagnostics.z > 0.5f" in install and
            "states[environment].diagnostics.z = 0.0f;" in install,
            "CGLS still must be gated by and invalidate the original linear certificate")
    require("newBufferWithLength" not in extract_kernel(
        CGLS_RUNTIME, "nm_fgmres_select_base_cgls_primary_policy"),
        "selector unexpectedly allocates a new per-environment vector")


def main():
    cpu_compile_and_execute_kernel()
    source_contract_audit()
    print("PASS production host source-contract audit (not runtime parser coverage)")


if __name__ == "__main__":
    main()
