"""Construction-only separating-plane target for one finite triangle pair.

Uses the pinned 466 triangle SAT axis inventory but considers both signs of
all face and edge-cross axes. This is a finite-triangle construction target;
it is not a whole-bank or whole-mesh clearance certificate.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np

SAT_PATH = Path(__file__).resolve().parents[3] / "D/D/triangle-sat-rows-466/triangle_sat_rows_466.py"
SAT_SHA256 = "cf4df8cd8a3f2773f0992a5f9a1d9b899d64db781b0a58cc2ed618285fa11cf3"

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

if _sha(SAT_PATH) != SAT_SHA256:
    raise RuntimeError("pinned triangle SAT 466 source hash mismatch")
_spec=importlib.util.spec_from_file_location("triangle_sat_466_for_551",SAT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load pinned SAT helper: {SAT_PATH}")
_sat=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=_sat
_spec.loader.exec_module(_sat)


def _validated_triangles(skin_triangle_mm,bone_triangle_mm):
    skin=np.asarray(skin_triangle_mm,dtype=np.float64)
    bone=np.asarray(bone_triangle_mm,dtype=np.float64)
    if skin.shape!=(3,3) or bone.shape!=(3,3):
        raise ValueError("skin and bone triangles must each have shape (3,3)")
    if not np.isfinite(skin).all() or not np.isfinite(bone).all():
        raise ValueError("triangle coordinates must be finite")
    for name,tri in (("skin",skin),("bone",bone)):
        e=np.array([tri[1]-tri[0],tri[2]-tri[1],tri[0]-tri[2]])
        scale=max(float(np.max(np.linalg.norm(e,axis=1))),1.0)
        area=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0])))
        if not np.isfinite(area) or area <= 1e-14*scale*scale:
            raise ValueError(f"{name} triangle is degenerate")
    return skin,bone


def _signed_sat_axes(skin,bone):
    """All normalized standard triangle SAT axes with both signs."""
    bases=_sat.triangle_sat_axis_candidates(skin,bone)
    out=[]
    for base in bases:
        n=np.asarray(base["normal_world_unit"],dtype=np.float64)
        for extra_sign in (1,-1):
            row=dict(base)
            row["normal_world_unit"]=n*float(extra_sign)
            row["axis_sign"]=int(base["axis_sign"])*extra_sign
            row["extra_sign"]=extra_sign
            out.append(row)
    if not out:
        raise ValueError("no nondegenerate triangle SAT axes")
    return out


def _float32_world_m_emission_margin_mm(skin_mm,bone_mm,normal):
    """Conservative one-ULP projection bound for coordinates emitted in metres.

    Inputs and return value use millimetres. World coordinates are converted
    to metres before Float32 spacing is measured, then the bound is converted
    back to millimetres. One ULP per coordinate is used for both fixed and
    moving triangles, providing a conservative projection margin.
    """
    n=np.abs(np.asarray(normal,dtype=np.float64))
    def max_projection_rounding(tri_mm):
        metres=np.asarray(tri_mm,dtype=np.float64)*1e-3
        f=np.abs(metres).astype(np.float32)
        if not np.isfinite(f).all():
            raise ValueError("coordinates overflow Float32 metre emission")
        nxt=np.nextafter(f,np.float32(np.inf),dtype=np.float32)
        ulp_m=nxt.astype(np.float64)-f.astype(np.float64)
        if not np.isfinite(ulp_m).all() or np.any(ulp_m<0):
            raise ValueError("invalid Float32 metre-emission ULP")
        return float(np.max(np.sum(n[None,:]*ulp_m*1000.,axis=1)))
    return max_projection_rounding(skin_mm)+max_projection_rounding(bone_mm)


def choose_pair_sat_plane(skin_triangle_mm,bone_triangle_mm):
    """Select the least-motion SAT support halfspace for a triangle pair.

    `corner_lower_mm[i]` is the signed 458-compatible row lower bound;
    negative values preserve existing room on already-clear corners.
    `positive_corner_deficits_mm` clips these only for axis scoring. If all
    three corners meet `support_mm + margin_mm` along the
    returned normal, the finite skin triangle and finite fixed bone triangle
    have disjoint projection intervals on this standard triangle SAT axis.
    No claim is made about other triangles in the bone bank or mesh.
    """
    skin,bone=_validated_triangles(skin_triangle_mm,bone_triangle_mm)
    axes=_signed_sat_axes(skin,bone)
    scored=[]
    for a in axes:
        n=np.asarray(a["normal_world_unit"],dtype=np.float64)
        support=float(np.max(bone@n))
        margin=_float32_world_m_emission_margin_mm(skin,bone,n)
        deficits=np.maximum(0.,support+margin-skin@n)
        # Geometry-first tie key makes selected halfspace stable to face winding
        # and vertex permutation; source fields provide deterministic provenance.
        geom=tuple(float(x) for x in np.round(n,14))
        key=(float(np.max(deficits)),float(np.sum(deficits)),geom,
             int(a["axis_source_order"]),int(a["axis_index"]),int(a["axis_sign_order"]),int(a["extra_sign"]))
        scored.append((key,a,n,support,margin,deficits))
    key,a,n,support,margin,deficits=min(scored,key=lambda x:x[0])
    bone_projection=bone@n
    plane_point=bone[int(np.argmax(bone_projection))].copy()
    signed=skin@n-support
    raw_lower=support+margin-skin@n
    return {
        "normal_world_unit":n.copy(),
        "plane_point_mm":plane_point.copy(),
        "float32_ulp_margin_mm":float(margin),
        "corner_signed_distance_mm":signed.copy(),
        "corner_lower_mm":raw_lower.copy(),
        "positive_corner_deficits_mm":deficits.copy(),
        "axis_strategy":"triangle_sat_minimum_positive_corner_deficit_both_signs",
        "axis_source":str(a["axis_source"]),
        "axis_index":int(a["axis_index"]),
        "axis_sign":int(a["axis_sign"]),
        "axis_sign_order":int(a["axis_sign_order"]),
        "axis_extra_sign":int(a["extra_sign"]),
        "axis_source_order":int(a["axis_source_order"]),
        "support_mm":support,
        "target_offset_mm":float(support+margin),
        "max_positive_corner_deficit_mm":float(np.max(deficits)),
        "sum_positive_corner_deficit_mm":float(np.sum(deficits)),
        "candidate_axis_count":int(len(axes)),
        "separation_claim":"all three skin corners meeting support_mm + float32_ulp_margin_mm imply projection separation of this finite triangle pair on the selected standard SAT axis",
        "whole_bank_clearance_certificate":False,
        "float32_margin_units":"world coordinates emitted as Float32 metres; ULP bound converted to millimetres",
    }
