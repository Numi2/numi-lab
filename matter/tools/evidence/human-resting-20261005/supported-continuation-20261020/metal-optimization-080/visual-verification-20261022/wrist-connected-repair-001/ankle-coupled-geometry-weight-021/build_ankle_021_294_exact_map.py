#!/usr/bin/env python3
"""Exact coordinate+86-weight row and oriented-triangle correspondence for 021 ankle patch."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
P069=E/"finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin"
P294=Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
Q=np.load(E/"ankle-coupled-geometry-weight-021/ankle-coupled-021-candidate.npz")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 sys.path.insert(0,"/Users/n/numi-human-anatomy-finish-1378/src")
 from numilab_human.skin_source_payload_preflight import decode_payload
 a=decode_payload(P069.read_bytes());b=decode_payload(P294.read_bytes())
 v0=np.asarray(a["vertices_f"][:,:3],np.float32);w0=np.asarray(a["full_weights"],np.float32);f0=np.asarray(a["indices"],np.int64).reshape(-1,3)
 v1=np.asarray(b["vertices_f"][:,:3],np.float32);w1=np.asarray(b["full_weights"],np.float32);f1=np.asarray(b["indices"],np.int64).reshape(-1,3)
 ids=np.asarray(Q["source_vertex_ids"],int);patch=np.asarray(Q["changed_source_faces"],int);support=np.unique(f0[patch])
 ring_faces=np.flatnonzero(np.any(np.isin(f0,support),axis=1));ring_rows=np.unique(f0[ring_faces])
 if v0.shape!=v1.shape or w0.shape!=w1.shape or f0.shape!=f1.shape:raise ValueError("payload shapes differ")
 index={}
 for i in range(len(v1)):
  key=v1[i].tobytes()+w1[i].tobytes();index.setdefault(key,[]).append(i)
 rowmap={}
 missing=[];amb=[]
 for i in ring_rows:
  hits=index.get(v0[i].tobytes()+w0[i].tobytes(),[])
  if len(hits)==1:rowmap[int(i)]=int(hits[0])
  elif not hits:missing.append(int(i))
  else:amb.append({"source_row":int(i),"target_count":len(hits),"sample":list(map(int,hits[:8]))})
 if missing or amb or len(set(rowmap.values()))!=len(rowmap):raise ValueError(f"not unique exact source row map missing={len(missing)} ambiguous={len(amb)} targets={len(set(rowmap.values()))}/{len(rowmap)}")
 bytri={}
 for j,t in enumerate(f1):bytri.setdefault(tuple(sorted(map(int,t))),[]).append((int(j),tuple(map(int,t))))
 facemap={}
 bad=[]
 for fi in ring_faces:
  mapped=tuple(rowmap[int(x)] for x in f0[fi]);cand=bytri.get(tuple(sorted(mapped)),[])
  if len(cand)!=1:bad.append({"source_face":int(fi),"mapped_vertices":mapped,"target_match_count":len(cand)});continue
  targetid,target=cand[0]
  cyclic=target in (mapped,(mapped[1],mapped[2],mapped[0]),(mapped[2],mapped[0],mapped[1]))
  if not cyclic:bad.append({"source_face":int(fi),"target_face":targetid,"reason":"orientation_reversed"});continue
  facemap[int(fi)]=targetid
 if bad:raise ValueError(f"triangle ancestry not unique/oriented: {bad[:5]}")
 maprows=np.asarray(sorted(ring_rows),np.int64);maptarget=np.asarray([rowmap[int(i)] for i in maprows],np.int64)
 pfaces=np.asarray(sorted(patch),np.int64);ptarget=np.asarray([facemap[int(i)] for i in pfaces],np.int64)
 rfaces=np.asarray(sorted(ring_faces),np.int64);rtarget=np.asarray([facemap[int(i)] for i in rfaces],np.int64)
 if len(set(ptarget.tolist()))!=len(ptarget) or len(set(rtarget.tolist()))!=len(rtarget):raise ValueError("triangle mapping is not injective")
 changed_targets=np.asarray([rowmap[int(i)] for i in ids],np.int64)
 mapnpz=E/"ankle-coupled-geometry-weight-021/ankle-patch-069i-to-294-exact-index-map.npz"
 np.savez_compressed(mapnpz,one_ring_source_rows_069i=maprows,one_ring_rows_294=maptarget,changed_source_rows_069i=ids,changed_source_rows_294=changed_targets,changed_source_faces_069i=pfaces,changed_source_faces_294=ptarget,one_ring_faces_069i=rfaces,one_ring_faces_294=rtarget)
 inv=json.load(open(E/"native-integrated-candidate-294-clean007-shared-owner-release-005/invocation.json"))
 report={"schema":"numi.human.ankle-patch-069i-to-294-exact-correspondence.v1","status":"exact_local_index_map_found_not_applied","source_payloads":{"069i":{"path":str(P069),"sha256":sha(P069)},"294":{"path":str(P294),"sha256":sha(P294)},"294_native_invocation_payload_sha256":inv["skin_payload_sha256"]},
  "021_candidate_sha256":sha(E/"ankle-coupled-geometry-weight-021/ankle-coupled-021-candidate.npz"),
  "exact_correspondence":{"coordinate_and_full86_weight_identity_key":"Float32 source xyz bytes concatenated with all 86 Float32 source weights; exact equality only, no tolerance or nearest-point search.","one_ring_source_rows":int(len(maprows)),"one_ring_rows_unique_exact_matches":int(len(maptarget)),"source_index_equals_target_index":int(np.sum(maprows==maptarget)),"source_index_reassigned":int(np.sum(maprows!=maptarget)),"all_one_ring_coords_and_weights_exact":True,"binding_indices_all86_bitwise_equal":bool(np.array_equal(np.asarray(a["bindings_u"]),np.asarray(b["bindings_u"]))),"binding_transforms_all86_bitwise_equal":bool(np.array_equal(np.asarray(a["bindings_f"]),np.asarray(b["bindings_f"]))),
  "changed_rows_021":int(len(ids)),"changed_rows_unique_target_map":int(len(changed_targets)),"changed_source_faces_021":int(len(pfaces)),"changed_faces_map_unique_same_orientation":int(len(ptarget)),"one_ring_faces":int(len(rfaces)),"one_ring_faces_map_unique_same_orientation":int(len(rtarget)),"raw_face_ids_equal_on_changed_patch":int(np.sum(pfaces==ptarget)),"all_raw_triangle_index_rows_equal":bool(np.array_equal(f0[pfaces],f1[ptarget])),
  "patch_source_row_ids_069i":pfaces.tolist() if False else ids.astype(int).tolist(),"changed_row_target_ids_294":changed_targets.astype(int).tolist(),"changed_face_id_pairs_069i_to_294":[[int(x),int(y)] for x,y in zip(pfaces,ptarget)]},
  "mapping_npz":str(mapnpz),"mapping_npz_sha256":sha(mapnpz),
  "scope":"Read-only coordinate/weight identity remap for the exact 021 ankle affected face patch plus one-ring; does not modify either skin source, construct a 294 payload, recompute a C/FEM map, or establish physical admission.",
  "limits":["The 069i and 294 payload hashes differ, and raw per-face indices do not match at identical face IDs; an explicit unique local row/face reindex is required.","This is source correspondence only. It does not show the ankle deformation candidate passes 294's surrounding-surface, bone, mechanical-layer, mass, or runtime gates."]}
 out=E/"ankle-coupled-geometry-weight-021/ankle-patch-069i-to-294-exact-correspondence.json"
 if out.exists():raise SystemExit("refusing overwrite "+str(out))
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");json.loads(out.read_text())
 print(json.dumps({"report":str(out),"sha256":sha(out),"mapping_npz_sha256":sha(mapnpz),"correspondence":report["exact_correspondence"]},indent=2))
if __name__=="__main__":main()
