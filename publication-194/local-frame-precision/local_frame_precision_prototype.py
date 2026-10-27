from __future__ import annotations
import hashlib, importlib.util, json, math, struct, sys
from fractions import Fraction
from pathlib import Path
import numpy as np

S = Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKDIR = E / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
C = S / "partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz"
AUDIT_TOOL = ROOT / "audit_skin_bone_surfaces_159.py"
HELPER = Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
OUT = S / "per-bank-workers-20261027/common-sub-float32-root-local-frame-001"

PINS = {
    "candidate_sha256": "7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af",
    "exact-emitted.triangles.csv": "0d3418064af1d4f34d19a1ec91992afceacaaa70b6c982d4a2637ded1d829eac",
    "exact-emitted.nodes.csv": "99e58d6afb68e0939670f69d4132bfa730cf3e988a7f70a179a41a89eac3802c",
    "exact-emitted.bodies.csv": "dae52ab190e02586421b56381b03a4453aff1058edca7b6a42afcb9051b50b90",
    "all-bank-containment.csv.banks.csv": "46568d191d84dd68f1fbcfe02b7938b15d4a7b3b786f268f46bcfe916070c6df",
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def loadmod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def exact_tri(tri):
    return tuple(tuple(Fraction.from_float(float(x)) for x in p)
                 for p in np.asarray(tri).reshape(3, 3))

def length(a, b):
    return math.sqrt(sum(float(a[k] - b[k]) ** 2 for k in range(3)))

def f32bits(v):
    a = np.asarray(v, dtype=np.float32)
    return [int(x) for x in a.view(np.uint32).reshape(-1)]

def point_on_triangle(p, tri, exact):
    n = exact._cross(exact._sub(tri[1], tri[0]), exact._sub(tri[2], tri[0]))
    if not any(n) or exact._dot(n, exact._sub(p, tri[0])) != 0:
        return False
    axes = exact._axes(n)
    p2 = tuple(p[k] for k in axes)
    signs = [exact._orient2(tuple(a[k] for k in axes), tuple(b[k] for k in axes), p2)
             for a, b in zip(tri, tri[1:] + tri[:1])]
    return min(signs) >= 0 or max(signs) <= 0

if sha(C) != PINS["candidate_sha256"]:
    raise RuntimeError("candidate pin changed")
audit_tool = loadmod("local_precision_audit_owner", AUDIT_TOOL)
helper = loadmod("local_precision_bank_helper", HELPER)
exact = audit_tool.exact_module()
with np.load(C, allow_pickle=False) as z:
    P32 = np.asarray(z["points_world_m"], dtype=np.float32).copy()
    F = np.asarray(z["source_O_faces"], dtype=np.int64).copy()
T, ids, gids, bodies, M, off, pos, bankkeys, pins, _ = helper.read_bank_inputs(BANKDIR, E)
for name, digest in PINS.items():
    if name == "candidate_sha256":
        continue
    if pins.get(name) != digest:
        raise RuntimeError(f"emitted input pin mismatch for {name}")

cases = [("43/16", 43, 16, 19841, 145), ("65/99", 65, 99, 14173, 174)]
results = []
for label, body, stable_bone, skin_face, bone_stable in cases:
    bank_rows = np.flatnonzero((ids[:, 0] == body) & (ids[:, 1] == stable_bone) & (ids[:, 3] == bone_stable))
    if len(bank_rows) != 1:
        raise RuntimeError(f"expected unique pinned bone triangle for {label}, got {len(bank_rows)}")
    bi = int(bank_rows[0])
    nodes = F[skin_face].astype(np.int64)
    world_skin = P32.astype(np.float64)
    skin_local_all = (world_skin - pos[body]) @ M[body].T - off[body]
    skin_local64 = skin_local_all[nodes]
    bone_local64 = T[bi].astype(np.float64)
    original_s, original_b = exact_tri(skin_local64), exact_tri(bone_local64)
    original_roots = sorted(set(exact.triangle_intersection_points(original_s, original_b)))
    if len(original_roots) != 2:
        raise RuntimeError(f"expected exact segment in original {label}, got {len(original_roots)} points")
    original_length = length(original_roots[0], original_roots[1])

    # Candidate local-frame prototype: round the actual skin and bone triangle
    # vertices to the existing per-body Float32 coordinate domain before
    # reconstructing the exact local intersection. No saved source is changed.
    skin_local32 = skin_local64.astype(np.float32)
    bone_local32 = bone_local64.astype(np.float32)
    skin_q = exact_tri(skin_local32.astype(np.float64))
    bone_q = exact_tri(bone_local32.astype(np.float64))
    local_roots = sorted(set(exact.triangle_intersection_points(skin_q, bone_q)))
    rounded_roots = np.asarray([[float(x) for x in p] for p in local_roots], dtype=np.float32)
    local_root_bits = [f32bits(x) for x in rounded_roots]
    local_unique = len({tuple(row) for row in local_root_bits})
    local_rounded_incidence = [
        {"skin": point_on_triangle(exact_tri([row])[0], skin_q, exact),
         "bone": point_on_triangle(exact_tri([row])[0], bone_q, exact)}
        for row in rounded_roots
    ]

    # Transform local Float32 vertices and roots back through the pinned body
    # pose, then apply the accepted world-XYZ Float32 emission.
    pose_matrix, pose_offset, pose_position = M[body], off[body], pos[body]
    def local_to_world(local):
        return (np.asarray(local, dtype=np.float64) + pose_offset) @ pose_matrix + pose_position
    skin_world_roundtrip64 = local_to_world(skin_local32)
    skin_world_roundtrip32 = skin_world_roundtrip64.astype(np.float32)
    pinned_skin32 = P32[nodes]
    skin_delta = skin_world_roundtrip64 - pinned_skin32.astype(np.float64)
    root_world_roundtrip64 = local_to_world(rounded_roots)
    root_world_roundtrip32 = root_world_roundtrip64.astype(np.float32)
    world_root_bits = [f32bits(x) for x in root_world_roundtrip32]
    world_unique = len({tuple(row) for row in world_root_bits})
    original_root_local32 = np.asarray([[float(x) for x in p] for p in original_roots], dtype=np.float32)
    original_root_world32 = local_to_world(original_root_local32).astype(np.float32)

    results.append({
        "bank": label,
        "body": body,
        "stable_bone_id": stable_bone,
        "skin_face": skin_face,
        "bone_triangle": bone_stable,
        "pinned_skin_vertices": [int(x) for x in nodes],
        "original_exact_local_root_count": len(original_roots),
        "original_exact_local_segment_length_m": original_length,
        "original_local_root_f32_unique_count": len({tuple(f32bits(x)) for x in original_root_local32}),
        "local_f32_triangle_reintersection_root_count": len(local_roots),
        "local_f32_triangle_reintersection_segment_length_m": length(local_roots[0], local_roots[1]) if len(local_roots) == 2 else None,
        "local_f32_intersection_roots_m": [[float(x) for x in p] for p in local_roots],
        "local_f32_rounded_root_bits": local_root_bits,
        "local_f32_rounded_roots_distinct": local_unique == len(local_roots),
        "local_f32_rounded_root_exact_incidence_on_quantized_triangles": local_rounded_incidence,
        "world_f32_roundtrip_root_bits": world_root_bits,
        "world_f32_roundtrip_roots_distinct": world_unique == len(local_roots),
        "original_exact_roots_transformed_to_world_f32_bits": [f32bits(x) for x in original_root_world32],
        "world_f32_roundtrip_skin_source_vertex_max_abs_delta_m": float(np.max(np.abs(skin_delta))),
        "world_f32_roundtrip_skin_source_vertex_max_euclidean_delta_m": float(np.max(np.linalg.norm(skin_delta, axis=1))),
        "world_f32_roundtrip_skin_source_vertex_changed_float32_component_count": int(np.count_nonzero(skin_world_roundtrip32.view(np.uint32) != pinned_skin32.view(np.uint32))),
        "world_f32_roundtrip_skin_source_vertex_total_components": int(pinned_skin32.size),
        "geometry_or_candidate_written": False,
        "physical_admission": False,
    })

report = {
    "schema": "numi.human.common-sub-float32-root-local-frame-prototype.v1",
    "status": "LOCAL_FRAME_DISTINCTION_TESTED_WORLD_FRAME_COLLAPSE_RETAINED",
    "candidate_sha256": PINS["candidate_sha256"],
    "emitted_input_pins": pins,
    "method": "Two retained incident skin/bone triangle pairs were transformed to the pinned body-local frame, quantized to Float32, reintersected with the exact rational predicate, and transformed back through the pinned body pose to world Float32. No source geometry was modified.",
    "results": results,
    "owner_code_review": {
        "double_registered_surface_source": "/Users/numi/Work/unpaired-skin-fat-mass-ownership-070/apps/NumiHumanCoarseTissuePreparation.hpp:21, RegisteredSurface donorLocalPoints/worldPoints are V3=std::array<double,3>",
        "accepted_shell_fem_float32_gate": "/Users/numi/Work/unpaired-skin-fat-mass-ownership-070/apps/NumiHumanCoarseTissuePreparation.hpp:1163-1173, world FEM node coordinates are componentwise cast to float32 before normalized determinant admission",
        "body_relative_resting_support_abi": "/Users/numi/Work/unpaired-skin-fat-mass-ownership-070/include/metalrobo/numi_human_resting_visual_gpu.h:14-18 and src/metal/NumiHumanRestingSupportGeometry.metalinc:79-103; local float4 skin influence coordinates are rotated and added to compensated body pose for support queries",
        "surface_mesh_candidate_world_abi": "/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/candidate.npz stores points_world_m as Float32; audit matched-query parity requires exact world-coordinate Float32 bits",
        "owner_scope_limit": "The body-relative influence ABI is a support/presentation point-query path, not a triangle-mesh collision or FEM topology ABI. The current candidate and matched-query formats have no patch-local origin field."
    },
    "conclusion": "Body-local Float32 can be prototyped for point identity, but a world-space Float32 round trip re-collapses the roots. Existing support-geometry local coordinates do not carry skin triangle topology, while the accepted tissue/FEM and audit paths require world-space Float32 geometry. A safe owner-integrated remesh therefore needs a separately specified local-frame topology/transform interface; this prototype does not establish one.",
    "geometry_changed": False,
    "full_or_all_bank_audit_run": False,
    "native_admission": False,
    "accepted_physical_steps": 0,
}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
