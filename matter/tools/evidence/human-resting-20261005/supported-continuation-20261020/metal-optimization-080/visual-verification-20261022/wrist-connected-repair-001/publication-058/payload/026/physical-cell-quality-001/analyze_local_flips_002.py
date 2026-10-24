#!/usr/bin/env python3
"""Bounded local same-material bistellar flip feasibility around D37 quality witnesses."""
import csv, hashlib, itertools, json
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT=ROOT/"native-candidate020-failure-witness-026"/"physical-cell-quality-001"
D=ROOT/"wholebody-attachment-source-rebind-037-candidate020-identity004-002"/"generated"
COORD=ROOT/"native-candidate020-failure-witness-026"/"run-001"/"initial-skin-fem-nodes.csv"
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))
def detq(points,tets):
    X=points[np.asarray(tets,dtype=np.int64)].astype(np.float64)
    det=np.einsum("ij,ij->i",X[:,1]-X[:,0],np.cross(X[:,2]-X[:,0],X[:,3]-X[:,0]))
    L=np.stack([np.linalg.norm(X[:,i]-X[:,j],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1).max(axis=1)
    return det,np.abs(det)/L**3
def facets(tets):
    c={}
    for t in tets:
        a,b,d,e=map(int,t)
        for f in ((a,b,d),(a,b,e),(a,d,e),(b,d,e)):
            k=tuple(sorted(f)); c[k]=c.get(k,0)+1
    return {k for k,n in c.items() if n==1}
def assess(kind, old_ids, old_tets, new_tets, points):
    od,oq=detq(points,old_tets); nd,nq=detq(points,new_tets)
    oldv=float(np.abs(od).sum()/6); newv=float(np.abs(nd).sum()/6)
    boundary_equal=facets(old_tets)==facets(new_tets)
    # Orientability is recoverable by reordering; report signed signs before that.
    positive=bool(np.all(nd!=0))
    return {"kind":kind,"old_tet_rows":list(map(int,old_ids)),"old_tets":np.asarray(old_tets).tolist(),
      "new_tets":np.asarray(new_tets).tolist(),"old_q":oq.tolist(),"new_q":nq.tolist(),
      "old_min_q":float(oq.min()),"new_min_q":float(nq.min()),
      "signed_det_positive_count":int(np.sum(nd>0)),"signed_det_negative_count":int(np.sum(nd<0)),
      "all_new_nonzero_volume":positive,"canonical_boundary_facets_equal":bool(boundary_equal),
      "old_patch_volume_m3":oldv,"new_patch_volume_m3":newv,"relative_volume_delta":(newv-oldv)/oldv,
      "geometric_flip_gate":bool(boundary_equal and positive and abs(newv-oldv)<=1e-8*oldv)}
trow=read(D/"tetrahedra.csv"); T=np.asarray([[int(r[k]) for k in "abcd"] for r in trow],dtype=np.int64)
materials=np.asarray([int(r["material"]) for r in trow])
xyzrow=read(COORD); P=np.asarray([[float(r[k]) for k in ("x_m","y_m","z_m")] for r in xyzrow],dtype=np.float32)
# Face and edge incidence indexed once over the small 28,632-tet candidate.
face_owner={}; edge_owner={}
for it,t in enumerate(T):
    a,b,c,d=map(int,t)
    for f in ((a,b,c),(a,b,d),(a,c,d),(b,c,d)):
        face_owner.setdefault(tuple(sorted(f)),[]).append(it)
    for e in itertools.combinations(sorted((a,b,c,d)),2):
        edge_owner.setdefault(e,[]).append(it)
targets=[26104,11329,19107]
trials=[]
for target in targets:
    mat=int(materials[target]); target_row=trow[target]
    oldq=float(detq(P,T[[target]])[1][0])
    # 2-to-3 across every same-material interior face of target tet.
    for face in itertools.combinations(map(int,T[target]),3):
        owners=face_owner.get(tuple(sorted(face)),[])
        if len(owners)!=2: continue
        neigh=owners[0] if owners[1]==target else owners[1]
        if int(materials[neigh])!=mat: continue
        face_set=set(face)
        p=next(v for v in T[target] if int(v) not in face_set)
        q=next(v for v in T[neigh] if int(v) not in face_set)
        a,b,c=map(int,face)
        new=[[p,q,a,b],[p,q,b,c],[p,q,c,a]]
        trials.append({"target_row":target,"target_material":mat,"target_parent_tet":int(target_row["parent_tet"]),
          "target_stable_source_face":int(target_row["stable_source_face"]),"target_q":oldq,
          "neighbor_row":int(neigh),"shared_face":sorted(face),**assess("2-to-3",sorted((target,neigh)),T[[target,neigh]],new,P)})
    # 3-to-2 around each internal edge in target tet.
    for e in itertools.combinations(sorted(map(int,T[target])),2):
        star=edge_owner.get(e,[])
        if len(star)!=3 or any(int(materials[i])!=mat for i in star): continue
        # Edge must not appear in any material-boundary face.
        boundary_edge=any(len(owners)==1 and set(e).issubset(f) and int(materials[owners[0]])==mat
                          for f,owners in face_owner.items())
        if boundary_edge: continue
        union=set(map(int,T[star].reshape(-1))); opp=sorted(union-set(e))
        if len(opp)!=3: continue
        p,q=map(int,e); a,b,c=opp
        new=[[a,b,c,p],[a,b,c,q]]
        trials.append({"target_row":target,"target_material":mat,"target_parent_tet":int(target_row["parent_tet"]),
          "target_stable_source_face":int(target_row["stable_source_face"]),"target_q":oldq,
          "edge":list(e),"star_rows":sorted(map(int,star)),**assess("3-to-2",sorted(star),T[star],new,P)})
best_by_target={}
for target in targets:
    relevant=[x for x in trials if x["target_row"]==target and x["geometric_flip_gate"]]
    target_old=float(detq(P,T[[target]])[1][0])
    best_by_target[str(target)]={"target_q":target_old,"gate_passing_flip_count":len(relevant),
      "best_patch_q_gain":max((x["new_min_q"]-x["old_min_q"] for x in relevant),default=None),
      "best_patch_min_q":max((x["new_min_q"] for x in relevant),default=None),
      "best_trials":[x for x in relevant if x["new_min_q"]==max((y["new_min_q"] for y in relevant),default=-1)][:3]}
report={"schema":"numi.human.local-layer-tet-flip-diagnostic.026.v1","status":"READ_ONLY_LOCAL_ALGEBRA",
 "input_sha256":{"D37_tetrahedra.csv":sha(D/"tetrahedra.csv"),"compiled_initial_skin_fem_nodes.csv":sha(COORD),
 "D37_nodes.csv":sha(D/"nodes.csv"),"D37_source_faces.csv":sha(D/"faces.csv")},
 "targets":{"26104":"native failure object-local tet 26104 / global 29419; fat; stable face 89542",
 "11329":"minimum-quality skin tet; donor group 92","19107":"minimum-quality fat tet; donor group 20"},
 "tests_performed":"Enumerated 2-to-3 flips across every same-material internal face of each target and 3-to-2 flips across every same-material interior edge star of each target. Gate checks canonical cavity boundary equality, nonzero new F32 volumes, and patch absolute-volume parity <=1e-8 relative. No candidate is emitted or promoted.",
 "target_results":best_by_target,"all_trials":trials,
 "limitation":"This local algebra does not certify non-overlap by exact tet intersection, ancestry/mass transfer, or global material interfaces; it is a candidate selector only."}
(OUT/"local-flip-diagnostic.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"output":str(OUT/"local-flip-diagnostic.json"),"target_results":best_by_target,
 "trial_count":len(trials),"geometric_flip_gate_pass_count":sum(t["geometric_flip_gate"] for t in trials)},indent=2))
