#!/usr/bin/env python3
"""Static contract checks for the opt-in, diagnostic-only status-5 observer."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1] / "overlay/matter"
runtime = (root / "src/runtime.mm").read_text()
metal = (root / "src/metal/fem.metalinc").read_text()
shared = (root / "include/numi/matter/shared.h").read_text()

# The only domains are distinct first-event slots; producer coverage is explicit.
assert "environment * 2u + domain" in metal
assert "captures[slot].capture.x != 0u" in metal
assert "statuses[environment].code != NM_STATUS_INVALID_DEFORMATION" in metal
assert "captureStatus5(10u, NM_STATUS5_CAPTURE_DOMAIN_GLOBAL" in runtime
assert "captureStatus5(20u, NM_STATUS5_CAPTURE_DOMAIN_GLOBAL" in runtime
assert "captureStatus5(31u, NM_STATUS5_CAPTURE_DOMAIN_PRIVATE" in runtime
assert "captureStatus5(41u, NM_STATUS5_CAPTURE_DOMAIN_PRIVATE" in runtime
assert "captureStatus5(50u, NM_STATUS5_CAPTURE_DOMAIN_GLOBAL" in runtime
assert "captureStatus5(60u, NM_STATUS5_CAPTURE_DOMAIN_GLOBAL" in runtime
assert "captureStatus5(30u, NM_STATUS5_CAPTURE_DOMAIN_PRIVATE" in runtime
assert "captureStatus5(40u, NM_STATUS5_CAPTURE_DOMAIN_PRIVATE" in runtime

# The FD-only status producer is intentionally not hooked; physical global/private
# producer counts and stage IDs are checked separately from that path.
force_calls = list(re.finditer(r'dispatchThreads\s*\(\s*"nm_fem_internal_forces"', runtime))
assert len(force_calls) == 5, len(force_calls)
for index, call in enumerate(force_calls):
    following = runtime[call.end():runtime.find("});", call.end()) + 3]
    if index == 0:
        assert "captureStatus5" not in following
    else:
        assert "captureStatus5" in runtime[call.end():call.end() + 3500]
geometry_calls = list(re.finditer(r'dispatchThreads\s*\(\s*"nm_fem_armijo_validate_geometry"', runtime))
assert len(geometry_calls) == 3, len(geometry_calls)
assert "captureStatus5(30u" not in runtime[geometry_calls[0].end():geometry_calls[0].end()+1800]
assert "captureStatus5(30u" in runtime[geometry_calls[1].end():geometry_calls[1].end()+1800]
assert "captureStatus5(40u" in runtime[geometry_calls[2].end():geometry_calls[2].end()+1800]

# The observer reads existing inputs and writes only its own diagnostic record.
start = metal.index("kernel void nm_fem_status5_capture(")
end = metal.index("\n}\n", start) + 3
body = metal[start:end]
assert "recordFailure(" not in body
assert "evaluateAlgorithmicStress" not in body
assert "projectMaterialState" not in body
assert "captures[slot] = record;" in body
assert "acceptedMaterialState[historyBase + state]" in body
assert "environmentParameters[parameterBase + parameter]" in body
assert "material.parameterCount <=\n                    NM_STATUS5_CAPTURE_MAX_PARAMETER_WORDS" in body
assert "NM_STATUS5_INDEX_SOLVER_ITERATION" in runtime
assert "NM_STATUS5_CAPTURE_MAX_PARAMETER_WORDS 512u" in shared

# Opt-in parser is strict and rejects unsupported source layouts before encoding.
assert "NM_FEM_STATUS5_CAPTURE_PATH" in runtime
assert "NM_FEM_STATUS5_CAPTURE_ROOT" in runtime
assert "status5Requested" in runtime and "status5Selected" in runtime
for guard in [
    "state.dispatch.particleCount != 0u",
    "state.dispatch.cohesiveFaceCount != 0u",
    "state.dispatch.materialStateStride > NM_MAX_MATERIAL_STATE",
    "status5CaptureBytes > 1024u * 1024u",
]:
    assert guard in runtime
assert "O_EXCL" in runtime or "O_CREAT | O_EXCL" in runtime
assert "numi.matter.fem_status5_capture.v1" in runtime

print("status5 capture source contract: PASS")
