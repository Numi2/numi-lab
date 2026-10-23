#!/usr/bin/env python3
"""Read-only exact ankle source-patch ancestry comparison: 069i vs accepted 294."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.skin_source_payload_preflight import decode_payload
P069=E/"finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin"
P294=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
Q=np.load(E/"ankle-coupled-geometry-weight-021/ankle-coupled-021-candidate.npz")
G=np.load(E/"finger-profile-069i-left-ring-web-retreat/replacement-geometry.npz")
FACES=np.asarray(Q["changed_source_faces"],int)
R=np.asarray(Q["source_vertex_ids"],int)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=decode_payload(P069.read_bytes());b=decode_payload(P294.read_bytes())
 def unpack(d):
  V=np.asarray(d["vertices_f"][:,:3],np.float32);W=np.asarray(d["full_weights"],np.float32);F=np.asarray(d["indices"],np.int64).reshape(-1,3)
  U=np.asarray(d["bindings_u"]);B=np.asarray(d["bindings_f"])
  return V,W,F,U,B
 v0,w0,f0,u0,b0=unpack(a);v1,w1,f1,u1,b1=unpack(b)
 if v0.shape!=v1.shape or w0.shape!=w1.shape or f0.shape!=f1.shape:raise ValueError("global payload dimensions differ")
 face_local=(np.array_equal(f0[FACES],f1[FACES]))
 touched=np.unique(f0[FACES].ravel())
 # Include all one-ring neighbor faces/rows to verify the deformation boundary basis.
 one_ring_faces=np.flatnonzero(np.any(np.isin(f0,touched),axis=1));one_ring_rows=np.unique(f0[one_ring_faces].ravel())
 def eq_rows(x,y,ids):return bool(np.array_equal(x[ids],y[ids]))
 donor={}
 for name,x,y in [("binding_indices",u0,u1),("binding_transforms",b0,b1)]:
  donor[name]={"shape_left":list(x.shape),"shape_294":list(y.shape),"global_bitwise_equal":bool(np.array_equal(x,y))}
 result={
  "schema":"numi.human.ankle-source-patch-069i-vs-294-exact-comparison.v1",
  "status":"read_only_correspondence_diagnostic_not_transplant_or_admission",
  "inputs":{"069i_payload":str(P069),"069i_sha256":sha(P069),"294_payload":str(P294),"294_sha256":sha(P294),"candidate021_sha256":sha(E/"ankle-coupled-geometry-weight-021/ankle-coupled-021-candidate.npz")},
  "global_shape":{"vertices":[int(len(v0)),int(len(v1))],"faces":[int(len(f0)),int(len(f1))],"weight_shape":[list(w0.shape),list(w1.shape)]},
  "candidate_patch":{"021_changed_source_rows":int(len(R)),"changed_source_faces":int(len(FACES)),"patch_triangles_bitwise_equal":face_local,"patch_vertex_ids_count":int(len(touched)),"one_ring_face_count":int(len(one_ring_faces)),"one_ring_vertex_count":int(len(one_ring_rows)),
   "changed_vertex_coords_bitwise_equal":eq_rows(v0,v1,R),"changed_vertex_full86_weights_bitwise_equal":eq_rows(w0,w1,R),
   "changed_face_support_coords_bitwise_equal":eq_rows(v0,v1,touched),"changed_face_support_full86_weights_bitwise_equal":eq_rows(w0,w1,touched),
   "one_ring_coords_bitwise_equal":eq_rows(v0,v1,one_ring_rows),"one_ring_weights_bitwise_equal":eq_rows(w0,w1,one_ring_rows),
   "max_changed_row_coordinate_delta_mm":float(np.linalg.norm(v0[R].astype(float)-v1[R].astype(float),axis=1).max()*1000),
   "max_patch_support_coordinate_delta_mm":float(np.linalg.norm(v0[touched].astype(float)-v1[touched].astype(float),axis=1).max()*1000),
   "max_patch_support_weight_delta":float(np.max(np.abs(w0[touched]-w1[touched]))),
   "changed_face_triangle_id_mismatch_count":int(np.count_nonzero(np.any(f0[FACES]!=f1[FACES],axis=1))),
   "changed_row_weight_deltas_to_294":{"count":int(len(R)),"max_half_L1":float(np.max(.5*np.abs(Q["full_weights_after"]-w1[R]).sum(1)))}},
  "donor_binding_correspondence":donor,
  "limits":["Only source topology/coordinates/weights/binding correspondence is compared; no candidate delta is applied to 294.","A bitwise match would justify a bounded transplant experiment, not physical acceptance or updated coarse/FEM maps.","Registration snapshots, C maps, mechanical attachments, tissue layers, and dynamics are not certified by this comparison."]}
 out=E/"ankle-coupled-geometry-weight-021/ankle-patch-069i-vs-294-correspondence.json"
 if out.exists():raise SystemExit("refusing overwrite "+str(out))
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");json.loads(out.read_text())
 print(json.dumps(result,indent=2));print("REPORT="+str(out));print("SHA256="+sha(out))
if __name__=="__main__":main()
