#!/usr/bin/env python3
"""Extract and fail-closed validate 538/541 face ancestry for the 541 layer path.

This preserves every parent row. The CSR stores topological unions, not mass
fractions; it is metadata for the registered surface only.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

DEFAULT_CANDIDATE = Path("/Users/numi/Work/numi-human-connected-surface-069/D/focused-clearance-541/topology/run-003-filtered/candidate.npz")
DEFAULT_OUT = Path(__file__).resolve().parent / "run-001"

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def validate_csr(name, offsets, rows, nrows, *, allow_empty=False, parent_count=None):
    offsets=np.asarray(offsets); rows=np.asarray(rows)
    if offsets.ndim!=1 or rows.ndim!=1 or offsets.dtype.kind not in "iu" or rows.dtype.kind not in "iu":
        raise ValueError(f"{name}: offsets and rows must be 1D integer arrays")
    if len(offsets)!=nrows+1 or not len(offsets) or int(offsets[0])!=0 or int(offsets[-1])!=len(rows):
        raise ValueError(f"{name}: CSR shape/endpoints do not match {nrows} rows")
    if np.any(np.diff(offsets.astype(np.int64))<0):
        raise ValueError(f"{name}: CSR offsets decrease")
    lengths=np.diff(offsets.astype(np.int64))
    if not allow_empty and np.any(lengths==0):
        raise ValueError(f"{name}: empty ancestry row at face {int(np.flatnonzero(lengths==0)[0])}")
    if np.any(rows<0) or (parent_count is not None and np.any(rows>=parent_count)):
        raise ValueError(f"{name}: parent row is outside its declared domain")
    multi=0
    for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        q=rows[int(a):int(b)]
        if len(q)>1:
            multi+=1
            if np.any(np.diff(q.astype(np.int64))<=0):
                raise ValueError(f"{name}: repeated or unsorted parent IDs at face {i}")
    return {"row_count":int(nrows),"entry_count":int(len(rows)),
            "empty_row_count":int(np.count_nonzero(lengths==0)),
            "multi_parent_row_count":int(multi),
            "maximum_row_width":int(lengths.max(initial=0))}

def validate_arrays(z):
    required=("points_world_m","source_O_faces","face_source_ancestry_indptr",
              "face_source_ancestry_face_rows","refinement_face_parent_ancestry_indptr",
              "refinement_face_parent_ancestry_face_rows","refinement_face_parent_face_row",
              "refinement_face_parent_barycentric")
    missing=[k for k in required if k not in z]
    if missing: raise ValueError(f"missing required candidate arrays: {missing}")
    points=np.asarray(z["points_world_m"])
    faces=np.asarray(z["source_O_faces"])
    if points.ndim!=2 or points.shape[1]!=3 or points.dtype.kind not in "fiu" or not np.isfinite(points).all():
        raise ValueError("candidate outer point array must be finite Nx3")
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in "iu" or faces.size==0:
        raise ValueError("candidate source_O_faces must be a nonempty integer Fx3 array")
    if faces.min()<0 or faces.max()>=len(points):
        raise ValueError("candidate face has an out-of-range vertex")
    f=len(faces)
    src=validate_csr("face_source_ancestry",z["face_source_ancestry_indptr"],
        z["face_source_ancestry_face_rows"],f)
    parent_count=None
    if "prior_input_parent__refinement_face_parent_face_row" in z:
        parent_count=len(z["prior_input_parent__refinement_face_parent_face_row"])
    parent=validate_csr("refinement_face_parent_ancestry",
        z["refinement_face_parent_ancestry_indptr"],
        z["refinement_face_parent_ancestry_face_rows"],f,parent_count=parent_count)
    scalar=np.asarray(z["refinement_face_parent_face_row"])
    bary=np.asarray(z["refinement_face_parent_barycentric"],dtype=np.float64)
    if scalar.shape!=(f,) or scalar.dtype.kind not in "iu" or bary.shape!=(f,3,3):
        raise ValueError("singular immediate-parent face labels/barycentrics have the wrong shape")
    offsets=np.asarray(z["refinement_face_parent_ancestry_indptr"],dtype=np.int64)
    rows=np.asarray(z["refinement_face_parent_ancestry_face_rows"],dtype=np.int64)
    widths=np.diff(offsets)
    multi=widths>1; single=widths==1
    if np.any(scalar[multi]!=-1) or not np.isnan(bary[multi]).all():
        raise ValueError("multi-parent faces must retain -1 singular labels and NaN singular barycentrics")
    for i in np.flatnonzero(single):
        a,b=offsets[i:i+2]
        if int(scalar[i])!=int(rows[a]) or not np.isfinite(bary[i]).all() or np.any(bary[i]<-1e-8) or np.any(bary[i]>1+1e-8) or not np.allclose(bary[i].sum(axis=1),1.0,atol=1e-8,rtol=0):
            raise ValueError(f"single-parent face {int(i)} has inconsistent scalar parent/barycentric metadata")
    if not np.all(single|multi):
        raise ValueError("immediate-parent ancestry contains an empty face row")
    out={"point_count":int(len(points)),"face_count":int(f),
         "face_source_ancestry":src,"immediate_parent_face_ancestry":parent,
         "multi_parent_single_label_sentinel_count":int(np.count_nonzero(multi)),
         "single_parent_barycentric_rows_checked":int(np.count_nonzero(single)),
         "material_or_mass_weights_present":False,
         "layers_or_volume_cells_present":any(k in z for k in ("inner_points_m","deep_points_m","tetrahedra","parent_tetrahedra","cell_mass_kg"))}
    return out

def extract(candidate: Path, outdir: Path):
    with np.load(candidate,allow_pickle=False) as archive:
        z={k:archive[k] for k in archive.files}
    summary=validate_arrays(z)
    outdir.mkdir(parents=True,exist_ok=False)
    keys=("source_O_faces","face_source_ancestry_indptr","face_source_ancestry_face_rows",
          "refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows",
          "refinement_face_parent_face_row","refinement_face_parent_barycentric")
    np.savez_compressed(outdir/"registered_surface_face_ancestry.npz",**{k:z[k] for k in keys})
    report={"schema":"numi.human.registered-surface-face-ancestry.541.v1",
            "status":"ANCESTRY_VALIDATED_LAYER_INPUT_REQUIRED",
            "candidate_path":str(candidate),"candidate_sha256":sha256(candidate),
            "ancestry_npz":"registered_surface_face_ancestry.npz",
            "ancestry_npz_sha256":sha256(outdir/"registered_surface_face_ancestry.npz"),
            "provenance":"Exact CSR rows copied without selecting a single parent or inventing fractions.",
            "limitations":["face-parent CSR is a topological union, not a conservative mass-transfer fraction",
                          "this candidate contains no O/I/D layers or volume-cell/mass ancestry",
                          "no FEM cells or physical transfer are prepared by this extractor"],
            **summary}
    (outdir/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidate",type=Path,default=DEFAULT_CANDIDATE)
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    a=ap.parse_args()
    print(json.dumps(extract(a.candidate,a.output_dir),indent=2,sort_keys=True))
if __name__=="__main__": main()
