#!/usr/bin/env python3
"""Export only the already-audited 075 missing-palm rows for C++ preparation tests."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=E/"arm-actuation-preparation-001/barycentric-embedding-owner-001/current-075-missing-palm-fixture.tsv"
REPORT=E/"arm-actuation-preparation-001/skin-embedding-owner-audit-001.json"
CANDIDATE=E/"replacement-mechanical-075-comp0-local-two-ring-001/mechanical-surface-candidate.npz"
GEOMETRY=E/"finger-profile-069d-full-ancestry-clearance/replacement-geometry.npz"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    report=json.loads(REPORT.read_text())
    rows=report["candidate_map_audit"]["hand_rows_missing_source_vertex"]
    with np.load(CANDIDATE,allow_pickle=False) as c,np.load(GEOMETRY,allow_pickle=False) as g:
        faces=np.asarray(g["faces"],dtype=np.int64); xyz=np.asarray(g["world"],dtype=np.float64)
        weights=np.asarray(g["weights"],dtype=np.float64)
        records=[]
        frame_offsets=[]
        active_rows=np.flatnonzero(np.asarray(c["vertex_kind"])!=0)
        residual_by_row=np.asarray(c["source_embedding_residual_vector_m"],dtype=np.float64)
        if len(active_rows)!=len(residual_by_row):
            raise ValueError("sparse residual rows do not align with vertex_kind != 0 ancestry")
        for item in rows:
            row=int(item["row"]); face=int(c["source_face"][row])
            beta=np.asarray(c["source_barycentric"][row],dtype=np.float64)
            if face<0 or face>=len(faces): raise ValueError(f"row {row} source face out of range")
            if not np.isfinite(beta).all() or np.min(beta)<-1e-8 or abs(float(beta.sum())-1)>1e-8:
                raise ValueError(f"row {row} barycentric input invalid")
            tri=faces[face]
            source_point=beta@xyz[tri]
            active_index=int(np.flatnonzero(active_rows==row)[0])
            residual=residual_by_row[active_index]
            outer=np.asarray(c["outer_world"][row],dtype=np.float64)
            frame_offsets.append(outer-source_point-residual)
            rowsdata=[]
            for vertex in tri:
                w=weights[vertex]
                if not np.isfinite(w).all() or np.min(w)<0 or abs(float(w.sum())-1)>1e-4:
                    raise ValueError(f"row {row} source weight row invalid at vertex {vertex}")
                rowsdata.append((int(vertex),xyz[vertex],w))
            records.append((row,face,beta,source_point,outer,residual,rowsdata))
    if len(records)!=167: raise ValueError(f"expected audited 167 missing hand palm rows; got {len(records)}")
    frame_offsets=np.asarray(frame_offsets,dtype=np.float64)
    fitted_frame_offset=np.median(frame_offsets,axis=0)
    max_frame_offset_deviation=float(np.max(np.linalg.norm(frame_offsets-fitted_frame_offset[None,:],axis=1)))
    lines=[f"{len(records)} {len(faces)} "+" ".join(format(float(x),".17g") for x in fitted_frame_offset)+" "+format(max_frame_offset_deviation,".17g")]
    for row,face,beta,source_point,outer,residual,tri in records:
        values=[str(row),str(face),*(format(float(x),".17g") for x in beta),
                *(format(float(x),".17g") for x in source_point),
                *(format(float(x),".17g") for x in outer),
                *(format(float(x),".17g") for x in residual)]
        for vertex,point,w in tri:
            values.append(str(vertex));values.extend(format(float(x),".17g") for x in point)
            values.extend(format(float(x),".17g") for x in w)
        lines.append(" ".join(values))
    OUT.write_text("\n".join(lines)+"\n")
    meta={"schema":"numi.human.source-embedding-current-candidate-test-fixture.v1",
      "status":"fixture_only_candidate_not_admitted",
      "candidate_path":str(CANDIDATE),"candidate_sha256":sha(CANDIDATE),
      "source_geometry_path":str(GEOMETRY),"source_geometry_sha256":sha(GEOMETRY),
      "owner_audit_path":str(REPORT),"owner_audit_sha256":sha(REPORT),
      "fixture_path":str(OUT),"fixture_sha256":sha(OUT),"rows":len(records),
      "source_face_count":len(faces),"source_vertex_count":len(xyz),
      "robust_global_world_offset_m":[float(x) for x in fitted_frame_offset],
      "max_deviation_from_global_world_offset_m":max_frame_offset_deviation,
      "interpretation":"CPU preparation helper check over the exact already-audited 075 face/bary maps. It neither admits the candidate nor qualifies runtime FEM/contact."}
    outmeta=OUT.with_suffix(".manifest.json");outmeta.write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))
if __name__=="__main__": main()
