#!/usr/bin/env python3
"""Generate a source-bound diagnostic clone of production edgeEdgeCCD.

The frozen production source remains untouched. The generated pair consists
of exact edgeEdgeCCD function-body clones with only (1) the 16/17 literal loop
bound and function name changed, and (2) observational writes of the already-
computed iteration, closest points and segment parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

EXPECTED_CONTACT_SHA256 = "3cef6ae3dbd384d7b26ff6b0eb4ecda6da7a32be74cdea0badc5975eef692132"
EXPECTED_FROZEN_MANIFEST_SHA256 = "5fb22c61c280f576f15677c8cf6dbbc174beceaa225973242f6d53f9c7957446"
EXPECTED_PARENT_REVISION = "35bfb518c542a306f563cc364ef8496837068b9a"
SCHEMA = "numi.matter.private_ee_ccd_metal_diagnostic_build.v2"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_edge_ee(source: str) -> str:
    marker = "inline bool edgeEdgeCCD("
    start = source.rfind(marker)
    if start < 0:
        raise ValueError("source lacks the production edgeEdgeCCD definition")
    open_brace = source.find("{", start)
    if open_brace < 0:
        raise ValueError("edgeEdgeCCD signature has no body")
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise ValueError("unterminated edgeEdgeCCD body")


def make_instrumented_clone(function: str, budget: int) -> str:
    if budget not in (16, 17):
        raise ValueError("only the captured 16-versus-17 comparison is supported")
    clone = function.replace(
        "inline bool edgeEdgeCCD(", f"inline bool edgeEdgeCCD{budget}Diagnostic(", 1)
    old_tail = """    thread float& separation,
    thread bool& certified
) {
    certified = false;"""
    new_tail = """    thread float& separation,
    thread bool& certified,
    thread uint& iterationsExecuted,
    thread float3& witnessP,
    thread float3& witnessQ,
    thread float2& witnessParameters,
    device float4* iterationTrace,
    const uint traceOffset
) {
    certified = false;
    iterationsExecuted = 0u;
    normal = float3(0.0f);
    parameters = float2(0.0f);
    point = float3(0.0f);
    witnessP = float3(0.0f);
    witnessQ = float3(0.0f);
    witnessParameters = float2(0.0f);"""
    if clone.count(old_tail) != 1:
        raise ValueError("unexpected edgeEdgeCCD parameter/body boundary")
    clone = clone.replace(old_tail, new_tail, 1)
    old_loop = "for (uint iteration = 0u; iteration < 16u; ++iteration) {"
    if clone.count(old_loop) != 1:
        raise ValueError("expected exactly one 16-step EE conservative-advancement loop")
    if budget == 17:
        clone = clone.replace(
            old_loop, "for (uint iteration = 0u; iteration < 17u; ++iteration) {", 1)
    old_closest = """        closestPointsOnSegments(p0, p1, q0, q1, s, t, pointP, pointQ);
        const float3 delta = pointP - pointQ;"""
    new_closest = """        closestPointsOnSegments(p0, p1, q0, q1, s, t, pointP, pointQ);
        iterationsExecuted = iteration + 1u;
        witnessP = pointP;
        witnessQ = pointQ;
        witnessParameters = float2(s, t);
        const uint trace = traceOffset + 3u * iteration;
        const float3 delta = pointP - pointQ;"""
    if clone.count(old_closest) != 1:
        raise ValueError("expected exactly one EE closest-segment evaluation site")
    clone = clone.replace(old_closest, new_closest, 1)
    old_distance = """        const float distance = length(delta);
        toi = time;"""
    new_distance = """        const float distance = length(delta);
        iterationTrace[trace] = float4(time, distance, effectiveThickness, speedBound);
        iterationTrace[trace + 1u] = float4(pointP, s);
        iterationTrace[trace + 2u] = float4(pointQ, t);
        toi = time;"""
    if clone.count(old_distance) != 1:
        raise ValueError("expected exactly one EE distance calculation site")
    clone = clone.replace(old_distance, new_distance, 1)
    return clone


KERNEL = r"""
kernel void nm_contact_ee_ccd_budget_diagnostic(
    device const float4* cases [[buffer(0)]],
    device float4* results [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    device float4* iterationTrace [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    const uint base = 9u * index;
    const float budgetWord = cases[base].z;
    const bool valid = cases[base].x == 1.0f &&
        (budgetWord == 16.0f || budgetWord == 17.0f) &&
        isfinite(cases[base].y) && cases[base].y > 0.0f;
    const uint out = 5u * index;
    if (!valid) {
        results[out] = float4(0.0f);
        results[out + 1u] = float4(0.0f);
        results[out + 2u] = float4(0.0f);
        results[out + 3u] = float4(0.0f);
        results[out + 4u] = float4(0.0f);
        return;
    }
    float toi = 0.0f, separation = INFINITY;
    float3 normal = float3(0.0f), point = float3(0.0f);
    float2 parameters = float2(0.0f), witnessParameters = float2(0.0f);
    float3 witnessP = float3(0.0f), witnessQ = float3(0.0f);
    uint iterationsExecuted = 0u;
    bool certified = false;
    const uint traceBase = index * 51u;
    bool hit = false;
    if (budgetWord == 16.0f) {
        hit = edgeEdgeCCD16Diagnostic(
            cases[base+1u].xyz, cases[base+5u].xyz,
            cases[base+2u].xyz, cases[base+6u].xyz,
            cases[base+3u].xyz, cases[base+7u].xyz,
            cases[base+4u].xyz, cases[base+8u].xyz,
            cases[base].y, toi, normal, parameters, point, separation, certified,
            iterationsExecuted, witnessP, witnessQ, witnessParameters,
            iterationTrace, traceBase);
    } else {
        hit = edgeEdgeCCD17Diagnostic(
            cases[base+1u].xyz, cases[base+5u].xyz,
            cases[base+2u].xyz, cases[base+6u].xyz,
            cases[base+3u].xyz, cases[base+7u].xyz,
            cases[base+4u].xyz, cases[base+8u].xyz,
            cases[base].y, toi, normal, parameters, point, separation, certified,
            iterationsExecuted, witnessP, witnessQ, witnessParameters,
            iterationTrace, traceBase);
    }
    results[out] = float4(hit ? 1.0f : 0.0f,
        certified ? 1.0f : 0.0f, toi, separation);
    results[out + 1u] = float4(normal, witnessParameters.x);
    results[out + 2u] = float4(point, witnessParameters.y);
    results[out + 3u] = float4(witnessP, float(iterationsExecuted));
    results[out + 4u] = float4(witnessQ, budgetWord);
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    frozen_manifest_path = source_root / "frozen-source.json"
    frozen_manifest_bytes = frozen_manifest_path.read_bytes()
    frozen_manifest_sha = sha(frozen_manifest_bytes)
    if frozen_manifest_sha != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise SystemExit(f"wrong frozen source manifest: {frozen_manifest_sha}")
    frozen_manifest = json.loads(frozen_manifest_bytes)
    if frozen_manifest.get("parent_revision") != EXPECTED_PARENT_REVISION:
        raise SystemExit("frozen source parent revision changed")
    file_hashes = frozen_manifest.get("source_sha256")
    if not isinstance(file_hashes, dict) or len(file_hashes) != 775:
        raise SystemExit("unexpected frozen source file inventory")
    for relative, expected in file_hashes.items():
        path = source_root / relative
        if not path.is_file() or sha(path.read_bytes()) != expected:
            raise SystemExit(f"frozen source file does not match manifest: {relative}")
    contact_path = source_root / "matter/src/metal/contact.metalinc"
    contact_bytes = contact_path.read_bytes()
    contact_sha = sha(contact_bytes)
    if contact_sha != EXPECTED_CONTACT_SHA256:
        raise SystemExit(f"wrong frozen contact source: {contact_sha}")
    function = extract_edge_ee(contact_bytes.decode("utf-8"))
    clone16 = make_instrumented_clone(function, 16)
    clone17 = make_instrumented_clone(function, 17)
    clone = clone16 + "\n" + clone17
    include_path = (source_root / "matter/src/matter_kernels.metal").resolve()
    generated = (
        f'#include "{include_path}"\n'
        "namespace numi_matter_metal {\n"
        + clone + "\n" + KERNEL + "\n} // namespace numi_matter_metal\n"
    ).encode("utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = args.output_dir / "private_ee_ccd_diagnostic.metal"
    generated_path.write_bytes(generated)
    manifest = {
        "schema": SCHEMA,
        "frozen_source_root": str(source_root),
        "frozen_source_manifest_sha256": frozen_manifest_sha,
        "frozen_source_parent_revision": EXPECTED_PARENT_REVISION,
        "frozen_source_file_count": len(file_hashes),
        "contact_source": "matter/src/metal/contact.metalinc",
        "contact_source_sha256": contact_sha,
        "extracted_production_function_sha256": sha(function.encode("utf-8")),
        "instrumented_function_sha256": sha(clone.encode("utf-8")),
        "instrumented_16_function_sha256": sha(clone16.encode("utf-8")),
        "instrumented_17_function_sha256": sha(clone17.encode("utf-8")),
        "generated_metal_sha256": sha(generated),
        "generated_metal": generated_path.name,
        "production_function": "numi_matter_metal::edgeEdgeCCD",
        "diagnostic_functions": [
            "numi_matter_metal::edgeEdgeCCD16Diagnostic",
            "numi_matter_metal::edgeEdgeCCD17Diagnostic",
        ],
        "diagnostic_kernel": "numi_matter_metal::nm_contact_ee_ccd_budget_diagnostic",
        "transformation": [
            "compile separate 16- and 17-bound clones; the diagnostic input selects one",
            "rename only the extracted clone symbols; preserve production edgeEdgeCCD",
            "store existing loop iteration, closest points, and segment parameters",
            "initialize output-only witness values before the loop",
        ],
        "production_function_modified": False,
        "flags": ["-std=metal4.0", "-O3", "-fno-fast-math"],
        "qualification": "source-derived diagnostic only; native results require external execution",
    }
    (args.output_dir / "build-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
