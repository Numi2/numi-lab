#!/usr/bin/env python3
from pathlib import Path

import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_SOURCE = PACKAGE_ROOT / "matter/src/metal/contact.metalinc"
STAGING_SOURCE = PACKAGE_ROOT / "source/matter/src/metal/contact.metalinc"
SOURCE = REPO_SOURCE if REPO_SOURCE.exists() else STAGING_SOURCE
OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PACKAGE_ROOT / "build/material_column_pair_test_v2.metal"

def extract_function(text: str, signature: str) -> str:
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 0
    for cursor in range(brace, len(text)):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                return text[start:cursor + 1]
    raise ValueError(f"unclosed function: {signature}")

source = SOURCE.read_text()
functions = "\n\n".join(extract_function(source, signature) for signature in (
    "inline bool surfacePrimitivesShareNode(",
    "inline bool exactMixedMortarParentPair(",
    "inline bool surfacePairEligible(",
))
header = r"""#include <metal_stdlib>
using namespace metal;
#define NM_INVALID_INDEX 0xffffffffu
#define NM_FEM_SURFACE_DISABLE_DEFORMABLE_CONTACT (1u << 31u)
#define NM_OBJECT_DISABLE_SELF_CONTACT (1u << 7u)
#define NM_OBJECT_DISABLE_DEFORMABLE_CONTACT (1u << 8u)
#define NM_CONTINUUM_SURFACE_POINT 1u
#define NM_TOPOLOGY_PAIRED_COHESIVE (1u << 5u)
#define NM_TOPOLOGY_COHESIVE_MIXED_MODE (1u << 6u)
#define NM_TOPOLOGY_COHESIVE_MORTAR (1u << 8u)
#define NM_COHESIVE_LAW_MIXED_MODE_BILINEAR 1u
struct NMMatterDispatchGPU { uint objectCount; uint gridNodeCount; uint femNodeCount; uint tetrahedronCount; uint cohesiveFaceCount; };
struct NMContinuumSurfacePrimitiveGPU { uint4 nodesAndObject; uint4 identity; float4 boundsMinimum; float4 boundsMaximum; };
struct NMFEMTopologyNodeGPU { uint4 identity; };
struct NMContinuumObjectGPU { uint representation; uint materialIndex; uint flags; uint schedulerIndex; };
struct NMIncidenceRangeGPU { uint first; uint count; uint objectIndex; uint reserved; };
typedef uint4 nm_uint4;
struct NMCohesiveFaceGPU { uint4 nodesAndFirst; uint4 adjacency; uint4 lawIdentity; };
"""
kernel = r"""
kernel void testMaterialColumnPairEligibility(
    const device NMMatterDispatchGPU* dispatch [[buffer(0)]],
    const device NMContinuumSurfacePrimitiveGPU* left [[buffer(1)]],
    const device NMContinuumSurfacePrimitiveGPU* right [[buffer(2)]],
    const device NMContinuumObjectGPU* objects [[buffer(3)]],
    const device NMFEMTopologyNodeGPU* nodes [[buffer(4)]],
    const device NMIncidenceRangeGPU* faceRanges [[buffer(5)]],
    const device nm_uint4* faceRefs [[buffer(6)]],
    const device NMCohesiveFaceGPU* cohesiveFaces [[buffer(7)]],
    device const uint* environmentIndices [[buffer(8)]],
    device uint* result [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= 8u) return;
    result[tid] = surfacePairEligible(
        dispatch[0], left[tid], right[tid], objects, nodes, environmentIndices[tid],
        faceRanges, faceRefs, cohesiveFaces) ? 1u : 0u;
}
"""
OUTPUT.write_text(header + "\n" + functions + "\n" + kernel)
print(OUTPUT)
