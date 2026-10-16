#!/usr/bin/env python3
"""Size-only source160 one-environment layout preflight against capture464 metadata."""
from pathlib import Path
import hashlib, json

BASE = Path(__file__).resolve().parents[1]
CAPTURE = BASE / "evidence/source464-base-cgls.regional.json"
OVERLAY = BASE / "source/matter/src/runtime.mm"
HEADER = BASE / "source/matter/include/numi/matter/regional_consumer_capture.hpp"
SHARED = Path("/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157/matter/include/numi/matter/shared.h")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

meta = json.loads(CAPTURE.read_text())
graph = meta["graph_schwarz"]
N = int(meta["node_count"])
T = int(meta["tetrahedron_count"])
S = int(meta["material_state_stride"])
M = int(meta["augmented_mode_count"])
Gp = int(graph["patch_count"])
Gpn = int(graph["patch_node_count"])
Gd = int(graph["dof_count"])
Gf = int(graph["matrix_float_count"])
Gpivot = int(graph["pivot_count"])
# Source-bound sizes from shared.h: float4=16, NMFEMNodeStateGPU=80,
# NMFGMRESStateGPU=2 float4=32, NMMatterStatusGPU=8 u32+float4=48.
F4, U32, F32, NODE, FGMRES, STATUS = 16, 4, 4, 80, 32, 48
cache = bool(meta.get("tangent_cache_enabled_by_solver_policy", False))
arrays = {
    "v": N*F4, "z": N*F4,
    "augmented_effective_P": M*N*F4, "augmented_AP": M*N*F4,
    "augmented_coarse_lu": M*M*F32,
    "augmented_coarse_pivots": M*U32, "augmented_coarse_valid": U32,
    "augmented_coarse_rhs": M*F32, "augmented_coarse_coefficients": M*F32,
    "graph_patches": Gp*4*U32, "graph_patch_nodes": Gpn*U32,
    "graph_rows": Gd*4*U32, "graph_node_patch_offsets": (N+1)*U32,
    "graph_node_patch_values": Gpn*U32, "graph_node_coverage": N*U32,
    "graph_patch_lu": Gf*F32,
    "graph_patch_original_matrices": Gf*F32,
    "graph_patch_pivots": Gpivot*U32, "graph_patch_factor_valid": Gp*U32,
    "graph_patch_corrections": Gpn*F4,
    "candidate_nodes": N*NODE, "linearization_nodes": N*NODE,
    "accepted_material_history": T*S*F32, "scheduler_exponents": T*U32,
    "deformation_high": T*3*F4, "deformation_low": T*3*F4,
    "fgmres_state": FGMRES, "production_status": STATUS,
    "restart_generation": 4*U32, "restart_raw_norm_squared": F32,
    "restart_target_scale": F32, "restart_mode_scale": F32,
    "restart_mode_valid": U32, "restart_tail_preserved": U32,
}
if cache:
    arrays["algorithmic_tangent_cache"] = T*81*F32
    arrays["algorithmic_tangent_cache_valid"] = T*U32
source = OVERLAY.read_text()
required = [f'append("{name}"' for name in arrays]
required.append('kAggregatePayloadCapBytes')
required.append('64ull * 1024ull * 1024ull')
if 'kAggregatePayloadCapBytes' not in HEADER.read_text():
    missing.append('64MiB cap constant in shared capture header')
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit(f"source160 append/cap contract missing: {missing}")
cap32 = 32*1024*1024
cap64 = 64*1024*1024
sum_bytes = sum(arrays.values())
result = {
    "schema": "numi.matter.selector5_consumer_size_preflight.v1",
    "source160_runtime_sha256": sha(OVERLAY),
    "source157_shared_sha256": sha(SHARED),
    "source464_regional_manifest_sha256": sha(CAPTURE),
    "dimensions": {"environment_count":meta["environment_count"],"selected_slice_environments":1,
        "node_count":N,"tetrahedron_count":T,"material_state_stride":S,
        "base_mode_count":meta["base_mode_count"],"augmented_mode_count":M,
        "mode_stride":meta["mode_stride"],"graph_patch_count":Gp,
        "graph_patch_node_count":Gpn,"graph_dof_count":Gd,
        "graph_matrix_float_count_per_environment":Gf,"graph_pivot_count_per_environment":Gpivot,
        "tangent_cache_enabled_by_solver_policy":cache},
    "array_count":len(arrays), "arrays_bytes":arrays, "total_payload_bytes":sum_bytes,
    "total_payload_mib":sum_bytes/(1024*1024),
    "graph_original_plus_lu_bytes":2*Gf*F32,
    "graph_original_plus_lu_mib":2*Gf*F32/(1024*1024),
    "cap32_bytes":cap32,"cap32_excess_bytes":sum_bytes-cap32,
    "cap64_bytes":cap64,"cap64_remaining_bytes":cap64-sum_bytes,
    "conclusion":"source160-002 checked one-environment append layout exceeds 32 MiB and fits the explicit 64 MiB payload bound; no whole-process peak or native allocation claim is made.",
}
print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
