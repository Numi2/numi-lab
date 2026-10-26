"""Complete typed-bank surface crossing audit; geometry preparation only."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
BANKHELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
EXACT=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def exact_module():
 sys.path.insert(0,str(E/"incremental-exact-audit-owner-253/src"))
 return module("numilab_human.connected_skin_bone_exact_159",EXACT)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def records(triangles,face_ids,exact):
 # Exact stored Float64 local coordinates, without a second Float32 cast.
 points=[tuple(Fraction.from_float(float(x)) for x in p) for p in np.asarray(triangles).reshape(-1,3)]
 faces=np.arange(len(points),dtype=np.int64).reshape(-1,3)
 return [(tri,lo,hi,int(face_ids[i]),ids) for i,(tri,lo,hi,row,ids) in enumerate(exact._records(points,faces))]
def intersect_bank(points,faces,bank,bank_ids,exact):
 """AABB exclusion covers every input face. Exact narrowphase covers overlaps."""
 S=np.asarray(points)[np.asarray(faces)]; B=np.asarray(bank)
 if not len(S) or not len(B):return {"skin_faces":len(S),"bank_faces":len(B),"skin_aabb_active":0,"bank_aabb_active":0,"result":None}
 blo,bhi=B.min((0,1)),B.max((0,1))
 slo,shi=S.min(1),S.max(1)
 active=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
 if not len(active):return {"skin_faces":len(S),"bank_faces":len(B),"skin_aabb_active":0,"bank_aabb_active":0,"result":None}
 rlo,rhi=slo[active].min(0),shi[active].max(0)
 bactive=np.flatnonzero(np.all(B.max(1)>=rlo,axis=1)&np.all(B.min(1)<=rhi,axis=1))
 sr=records(S[active],active,exact);br=records(B[bactive],np.asarray(bank_ids)[bactive],exact)
 result=exact._audit_pair(sr,br,same_surface=False)
 return {"skin_faces":len(S),"bank_faces":len(B),"skin_aabb_active":len(active),"bank_aabb_active":len(bactive),"result":result}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("candidate");ap.add_argument("output");ap.add_argument("--max-seconds",type=float,default=180);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 z=np.load(a.candidate,allow_pickle=False);P=np.asarray(z["points_world_m"],dtype=np.float32).astype(np.float64);F=np.asarray(z["source_O_faces"],dtype=np.int64)
 if not np.isfinite(P).all() or F.ndim!=2 or F.shape[1]!=3 or F.min()<0 or F.max()>=len(P):raise ValueError("invalid surface")
 helper=module("connected_bone_bank_159",BANKHELPER);exact=exact_module()
 bankdir=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
 T,ids,gids,bodies,M,off,pos,keys,pins,exact_unused=helper.read_bank_inputs(bankdir,E)
 local={};rows=[];complete=False
 for body0,bone0 in np.unique(ids[:,:2],axis=0):
  body,bone=int(body0),int(bone0)
  if body not in local:local[body]=(P-pos[body])@M[body].T-off[body]
  mask=(ids[:,0]==body)&(ids[:,1]==bone)
  r=intersect_bank(local[body],F,T[mask],ids[mask,3],exact)
  rows.append({"body":body,"stable_bone_id":bone,**r})
  (out/"progress.json").write_text(json.dumps({"completed_banks":len(rows),"elapsed_seconds":time.monotonic()-start,"last_body":body,"last_bone":bone})+"\n")
  if time.monotonic()-start>a.max_seconds:break
 complete=len(rows)==len(np.unique(ids[:,:2],axis=0))
 report={"status":"COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" if complete else "BOUNDED_PARTIAL_AUDIT","candidate_sha256":sha(a.candidate),"script_sha256":sha(__file__),"exact_predicate_sha256":sha(EXACT),"surface_point_count":len(P),"surface_face_count":len(F),"bank_count":len(rows),"expected_bank_count":len(np.unique(ids[:,:2],axis=0)),"complete":complete,"elapsed_seconds":time.monotonic()-start,"geometry_or_native_admission":False,"coordinate_rule":"exact F32 emitted world points transformed by pinned raw F32 bank matrix in Float64; exact rational narrowphase on stored bank-local values, no additional F32 cast","limitations":["Surface crossing check does not certify closed-solid containment, contact dynamics, tissue deformation or native acceptance."],"banks":rows}
 (out/"report.json").write_text(json.dumps(report,indent=2,default=lambda v:v.item() if hasattr(v,"item") else str(v))+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="banks"}))
if __name__=="__main__":main()
