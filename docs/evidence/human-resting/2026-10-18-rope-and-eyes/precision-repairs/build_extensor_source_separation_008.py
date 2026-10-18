import sys,json,time,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,"/Users/n/numi-human-closed-outer-skin-20261017/src")
from numilab_human.passive_attachment_composition import _read_nhtiss4,_nhtiss_row_arrays,_row_slices,_rest_body_poses_f32,_child_rest_affine
from numilab_human.passive_reference_receipt_composition import compose_reference_receipt
from numilab_human.resting_anatomy_verify import prepare_surface
p=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-suspension-reference-probe-20261018-001")
src=Path(json.loads((p/"tissue-native-precision-retirement-004/result.json").read_bytes())["payload_path"]);d=_read_nhtiss4(src);out=p/"extensor-source-precision-separation-008";out.mkdir(exist_ok=False);world=np.load(p/"native016-step0-tissue-precision-positions.npz")
rigid=Path(json.loads((p/"attempt-017/native-run/invocation.json").read_bytes())["argv"][1]);_,poses=_rest_body_poses_f32(rigid);replacements=[]
for sid,edge in [(137,(89,90)),(138,(2054,2055))]:
 a=_nhtiss_row_arrays(d,next(r for r in d["records"] if int(r[6])==sid));v=a["positions"].copy();f=a["faces"];w=world[str(sid)].copy();aff,bias=_child_rest_affine(a,poses);i,j=edge
 aliases=np.flatnonzero(np.all(v==v[i],axis=1));star=np.flatnonzero(np.isin(f,aliases).any(axis=1));old=v.copy();startri=f[star]
 def cross(points):
  q=points[startri].astype(float);return np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
 sourcecross=cross(v);worldcross=cross(w);direction=w[i].astype(float)-w[j];direction/=np.linalg.norm(direction);chosen=None;trials=[]
 for amount in (2e-7,3e-7,4e-7,6e-7,8e-7):
  delta=np.linalg.solve(aff[i],direction*amount);nv=v.copy();nv[aliases]=(v[aliases].astype(float)+delta).astype("<f4")
  actual=nv[aliases].astype(float)-v[aliases];nw=w.copy();nw[aliases]=(w[aliases].astype(float)+np.einsum("nij,nj->ni",aff[aliases],actual)).astype("<f4")
  sc=cross(nv);wc=cross(nw);good=bool(np.all(np.einsum("ij,ij->i",sourcecross,sc)>0) and np.all(np.einsum("ij,ij->i",worldcross,wc)>0))
  extent=float(np.max(np.linalg.norm(actual,axis=1)));separation=float(np.linalg.norm(nw[i].astype(float)-nw[j]))
  trials.append({"requested_world_delta_m":amount,"max_source_delta_m":extent,"edge_world_length_m":separation,"all_incident_face_orientations_preserved":good})
  if good and separation>2.5e-7 and extent<1e-6:chosen=(nv,nw,extent,separation);break
 if chosen is None:
  (out/f"row{sid}-rejected.json").write_text(json.dumps(trials,indent=2));raise ValueError(f"no bounded orientation-preserving source separation for{sid}")
 nv,nw,extent,separation=chosen
 # Referenced topology stays closed with every source seam class moved together.
 used=np.unique(f);remap=np.full(len(v),-1,dtype=int);remap[used]=np.arange(len(used))
 before=prepare_surface((51005,sid),v[used],remap[f].astype(np.uint32));after=prepare_surface((51005,sid),nv[used],remap[f].astype(np.uint32));assert before.closed and after.closed
 patch=out/f"row{sid}.npz";report=out/f"row{sid}-report.json"
 np.savez(patch,vertices6=np.column_stack((nv,a["normals"])),binding_indices=a["binding_indices"],weights=a["weights"],faces=f)
 report.write_text(json.dumps({"source_payload_path":str(src),"source_sha256":d["sha256"],"stable_id":sid,"edge":edge,"moved_source_coordinate_alias_slots":aliases.tolist(),"source_positions_before":v[aliases].tolist(),"source_positions_after":nv[aliases].tolist(),"trials":trials,"source_and_world_incident_orientations_preserved":True,"closed_source_topology_preserved":True,"all_face_and_binding_weight_and_normal_bytes_preserved":True,"max_source_displacement_m":extent,"native_motion_admission":False,"anatomical_admission":False},indent=2)+"\n")
 replacements.append((sid,patch,report));print(json.dumps({"row":sid,"delta_m":extent,"edge_m":separation}),flush=True)
result=compose_reference_receipt(p/"tissue-native-precision-retirement-004/composed-receipt/resting-anatomy-receipt.json",out/"composed-receipt",replacements,extensions={})
after=_read_nhtiss4(Path(result["payload_path"]))
for aa,bb in zip(d["records"],after["records"]):
 x,y=_row_slices(d,aa),_row_slices(after,bb);assert x["binding_bytes"]==y["binding_bytes"]
 if int(aa[6]) not in (137,138):assert x["local_faces"]==y["local_faces"] and x["vertex_bytes"]==y["vertex_bytes"]
result["unchanged_rows_all_vertex_and_face_bytes"]=162;(out/"result.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)
