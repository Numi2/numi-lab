#!/usr/bin/env python3
from pathlib import Path
import hashlib,importlib.util,json,sys,numpy as np
S=Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638")
B=S/"partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001"
C=B/"candidate.npz";P=S/"fit-from-663-001/seed-retry-663-001/fraction-small-001/candidate.npz";A=B/"audit-159/report.json";SP=B/"audit-130/surface-self-pairs.npy";SS=B/"audit-130/summary.json"
O=Path("/Users/numi/Work/numi-human-connected-surface-069/D/local-two-parent-retriangulation-402/two_parent_patch_flips_402.py")
G=S/"edge-refinement-661-threebank-001/refresh_guards_multi_parent_generation002.py";AD=G.with_name("connected_refinement_adapter546_generation002.py")
OUT=Path(__file__).with_name("report.json")
EX=[19309,19310,19311,19312,19352,19353,19354,19355,19417,19418,19419,19420,19870,19871,19872,19873]
PIN={"C":"7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af","P":"beaff5ca4deeba288662cbb614b1750516948c9400476618c6a9d45f823c9a35","A":"c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e","SP":"55737cf1229ed3c3f82eb23b50874fb56749277d1c3a190bccfa1f3a991a57de","SS":"0ec99fa47245355d60858b05e1240b0e562de229c13721ea11659f6f4acdf5cf","O":"f68271083bbb97905487f7571ded8264fde1b575759feaf4f5aadf9c42076764","G":"90bafb6a51e9d7a2dd9b49570fcd977423616e18e992c2d4a4306feea5bd5eab","AD":"6603e244496dfbbeb76f7cfbaa56c80f39b1a18e6cccb50727406148ba885fe4"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mod(n,p):
 q=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(q);sys.modules[n]=m;q.loader.exec_module(m);return m
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def area(x,t):
 p=np.asarray(x)[np.asarray(t,dtype=np.int64)].astype(np.float64);return float(.5*np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0])))
def amin(x,t):
 p=np.asarray(x)[np.asarray(t,dtype=np.int64)].astype(np.float64);e=[np.linalg.norm(p[1]-p[0]),np.linalg.norm(p[2]-p[1]),np.linalg.norm(p[0]-p[2])]
 if min(e)<=0:return 0.
 a=np.arccos(np.clip(np.dot(p[1]-p[0],p[2]-p[0])/(e[0]*e[2]),-1,1));b=np.arccos(np.clip(np.dot(p[0]-p[1],p[2]-p[1])/(e[0]*e[1]),-1,1));return float(np.rad2deg(min(a,b,np.pi-a-b)))
def asp(x,t):
 p=np.asarray(x)[np.asarray(t,dtype=np.int64)].astype(np.float64);e=[np.dot(p[1]-p[0],p[1]-p[0]),np.dot(p[2]-p[1],p[2]-p[1]),np.dot(p[0]-p[2],p[0]-p[2])];a=area(x,t);return float(max(e)/(4*np.sqrt(3)*a)) if a>0 else float("inf")
def main():
 got={"C":sha(C),"P":sha(P),"A":sha(A),"SP":sha(SP),"SS":sha(SS),"O":sha(O),"G":sha(G),"AD":sha(AD)}
 if got!=PIN:raise ValueError(got)
 c=load(C);p=load(P);audit=json.loads(A.read_text());sp=np.load(SP,allow_pickle=False);ss=json.loads(SS.read_text());b=mod("owner402_ring43",O);g=mod("refreshg2_ring43",G)
 base_guard=g.validate_ancestry(c,p,audit)
 bank=next(x for x in audit["banks"] if x.get("body")==43 and x.get("stable_bone_id")==16);pairs=np.asarray(bank["result"]["triangle_pairs"],dtype=np.int64).reshape(-1,2)
 f=c["source_O_faces"];d=c["parent_fine_detail_face_row"];s=c["parent_fine_subface_row"];k=c["face_kind"];dr=sorted(set(map(int,d[EX])))
 if dr!=[8265,8269] or not set(EX)<=set(map(int,pairs[:,0])):raise ValueError("target witness/ancestry mismatch")
 em=b.edge_map(f);ring=set(EX)
 for r in EX:
  for a,z in zip(f[r],np.roll(f[r],-1)):
   for incident in em[(min(int(a),int(z)),max(int(a),int(z)))]:
    t=int(incident[0])
    if int(d[t]) in dr:ring.add(t)
 ring=sorted(ring);ring_only=sorted(set(ring)-set(EX))
 out,rep=b.propose_patch_flips(c,dr,ring,max_flips=4,max_rounds=1,min_angle_gain_deg=0.5,min_normal_dot=0.0)
 ops=rep["operations"];metrics=[];quality=True;ident=[]
 for op in ops:
  rows=list(map(int,op["face_rows"]));oldarea=sum(area(c["points_world_m"],f[r]) for r in rows);new=[]
  for r in rows:
   t=out["source_O_faces"][r];a=area(c["points_world_m"],t);new.append({"row":r,"tri":list(map(int,t)),"area":a,"min_angle":amin(c["points_world_m"],t),"aspect":asp(c["points_world_m"],t),"fraction_old_patch_area":a/oldarea if oldarea else 0})
  passed=all(x["area"]>0 and x["min_angle"]>=0.5 and x["aspect"]<=50 and x["fraction_old_patch_area"]>=0.1 for x in new);quality &= passed
  ident.append({"rows":rows,"detail":sorted(set(map(int,d[rows]))),"subface":sorted(set(map(int,s[rows]))),"kind":sorted(set(map(int,k[rows]))),"passes_same_detail_subface_kind":len(set(map(int,d[rows])))==len(set(map(int,s[rows])))==len(set(map(int,k[rows])))==1,"incident_to_excluded":bool(set(rows)&set(EX))})
  metrics.append({"rows":rows,"target":sorted(set(rows)&set(EX)),"new_faces":new,"angle_gain":op["angle_gain_deg"],"shape_pass":passed})
 after=g.validate_ancestry(out,p,audit) if ops and quality else None
 report={"schema":"numi.human.b43-16.one-ring-expansion-prep.v1","status":"NO_ELIGIBLE_SHAPE_SAFE_FLIP" if not ops or not quality else "LOCAL_FLIP_FOUND_REPORT_ONLY","candidate_written":False,"geometry_admitted":False,"physical_steps":0,
 "candidate_sha256":PIN["C"],"target_body":43,"stable_bone_id":16,"exact_target_pair_count":len(pairs),"unique_crossing_faces":len(set(map(int,pairs[:,0]))),"excluded_seed_faces":EX,"detail_rows":dr,
 "expanded_seed_faces":ring,"expanded_seed_count":len(ring),"one_extra_ring_only_count":len(ring_only),"one_extra_ring_only_faces":ring_only,"owner_patch_faces":rep["selected_face_count"],"locked_outer_edges":rep["locked_outer_edge_count"],"locked_detail_seams":rep["locked_inter_parent_seam_edge_count"],
 "limits":{"max_flips":4,"rounds":1,"min_angle_gain":0.5},"accepted_flip_count":len(ops),"rejections":rep["rejection_reason_counts"],"operations":ops,"operation_identity_checks":ident,"shape_metrics":metrics,"shape_gates":{"min_angle":0.5,"max_aspect":50,"min_area_fraction_old_patch":0.1,"pass":bool(ops and quality)},
 "generation002_ancestry_guard_before":base_guard,"generation002_ancestry_guard_after_report_only":after,"stored_self_pairs":len(sp),"updated_self_or_bone_checks":"NOT_RUN",
 "source_material_transfer":"Owner 402 can union source/detail/subface/immediate-parent CSR only for same kind, detail and subface, and preserves boundary/seams. It preserves face_kind and other arrays; it does not recompute continuous physical material, volume or layer transfer.",
 "limitations":["report only; no candidate emitted","no updated exact bone or self predicates","ancestry validation does not establish clearance or physical admission"]}
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"report_path":str(OUT),"report_sha256":sha(OUT),"status":report["status"],"expanded_seed_count":len(ring),"one_extra_ring_faces":len(ring_only),"accepted":len(ops),"rejections":rep["rejection_reason_counts"],"locked_seams":rep["locked_inter_parent_seam_edge_count"],"operations":ops,"shape_metrics":metrics,"ancestry_after_passed":after is not None},indent=2,sort_keys=True))
if __name__=="__main__":main()
