#!/usr/bin/env python3
"""Read-only, saved-report geometry analysis for the exact 594 candidate."""
import hashlib, json, pathlib, collections, numpy as np
D = pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546")
BASE = D / "scalar-retention-590/raw003-paired-v2-001/regional-552"
STATE = BASE / "final/state.json"
OUT = D / "crossing-diagnosis-594"
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def pct(x, ps=(0,10,50,90,100)):
    return [float(z) for z in np.percentile(np.asarray(x),ps)]
def normals(T):
    return np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0])
def angle_min_deg(T):
    e=[np.linalg.norm(T[:,1]-T[:,0],axis=1),np.linalg.norm(T[:,2]-T[:,1],axis=1),np.linalg.norm(T[:,0]-T[:,2],axis=1)]
    angles=[]
    for i in range(3):
        a,b,c=e[i],e[(i+1)%3],e[(i+2)%3]
        co=np.clip((b*b+c*c-a*a)/(2*np.maximum(b*c,1e-300)),-1,1)
        angles.append(np.arccos(co))
    return np.min(np.stack(angles,axis=1),axis=1)*180/np.pi
s=json.load(open(STATE)); cand=pathlib.Path(s["candidate"]); surf=pathlib.Path(s["surface_report"]); psum=pathlib.Path(s["point_summary"])
r=json.load(open(surf)); p=json.load(open(psum)); a=np.load(cand,allow_pickle=False)
F=np.asarray(a["source_O_faces"],dtype=np.int64); RF=np.asarray(a["area_reference_037_source_O_faces"],dtype=np.int64)
X=np.asarray(a["points_world_m"],dtype=np.float64); R37=np.asarray(a["area_reference_037_points_world_m_f64"],dtype=np.float64)
R0=np.asarray(a["local_witness_cluster_222_source_reference_R0_m"],dtype=np.float64)
assert np.array_equal(F,RF) and len(F)==r["surface_face_count"]
bank_rows=[]; allpairs=[]
for b in r["banks"]:
    res=b.get("result") or {}; pp=res.get("triangle_pairs",[])
    if not pp: continue
    key=(int(b["body"]),int(b["stable_bone_id"]))
    faces=sorted({int(q[0]) for q in pp}); nodes=sorted({int(v) for fi in faces for v in F[fi]})
    C=X[F[faces]].mean(axis=1); disp=np.linalg.norm(X-R0,axis=1)*1000
    bank_rows.append({"body":key[0],"stable_bone_id":key[1],"pairs":len(pp),
        "skin_faces":len(faces),"bone_faces":len({int(q[1]) for q in pp}),
        "skin_centroid_world_m_mean":[float(v) for v in C.mean(axis=0)],
        "skin_centroid_world_m_std":[float(v) for v in C.std(axis=0)],
        "cross_face_vertex_source_displacement_mm_p50_max":[float(np.median(disp[nodes])),float(np.max(disp[nodes]))]})
    allpairs.extend((int(q[0]),int(q[1]),key[0],key[1]) for q in pp)
assert len(allpairs)==1593 and sum(x["pairs"] for x in bank_rows)==1593
crossfaces=sorted({q[0] for q in allpairs})
ef=collections.defaultdict(list)
for fi,t in enumerate(F):
    for u,v in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):
        ef[(min(int(u),int(v)),max(int(u),int(v)))].append(fi)
adj=[set() for _ in F]
for fs in ef.values():
    if len(fs)>1:
        for i in fs: adj[i].update(j for j in fs if j!=i)
from collections import deque
boundary={fi for fs in ef.values() if len(fs)==1 for fi in fs}
dist=np.full(len(F),-1,dtype=np.int32); q=deque()
for fi in boundary: dist[fi]=0; q.append(fi)
while q:
    i=q.popleft()
    for j in adj[i]:
        if dist[j]<0: dist[j]=dist[i]+1; q.append(j)
remaining=set(crossfaces); comps=[]
while remaining:
    seed=remaining.pop(); stack=[seed]; comp=[seed]
    while stack:
        i=stack.pop(); ns=adj[i]&remaining; remaining-=ns; stack.extend(ns); comp.extend(ns)
    comps.append(comp)
comps.sort(key=len,reverse=True)
T=X[F]; Q=R37[F]; n=normals(T); nr=normals(Q)
cos=np.sum(n*nr,axis=1)/(np.linalg.norm(n,axis=1)*np.linalg.norm(nr,axis=1)+1e-300)
area_ratio=np.linalg.norm(n,axis=1)/np.maximum(np.linalg.norm(nr,axis=1),1e-300)
angles=angle_min_deg(T)
cd=cos[crossfaces]; cr=area_ratio[crossfaces]; ca=angles[crossfaces]
n0=normals(R0[F[crossfaces]])
c0=np.sum(n[crossfaces]*n0,axis=1)/(np.linalg.norm(n[crossfaces],axis=1)*np.linalg.norm(n0,axis=1)+1e-300)
disp=np.linalg.norm(X-R0,axis=1)*1000; cnodes=sorted({int(v) for fi in crossfaces for v in F[fi]})
poses_path=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/registered-skin-poses.json")
export_report=poses_path.parent/"export-report-final.json"; er=json.load(open(export_report))
topo=[]; root=D/"connected-topology-580"
for i in (1,2,3):
    rd=root/f"run-{i:03d}"; sr=json.load(open(rd/"search/report.json")); fc=json.load(open(rd/"full-check/report.json")); fr=json.load(open(rd/"full-check/audit-159/report.json"))
    topo.append({"run":i,"before_pairs":sr["before_pairs"],"search_after_pairs":sr["after_pairs_composed"],
        "search_reduction":sr["reduction"],"accepted_operations":len(sr["accepted_operations"]),
        "exact_triangle_calls":sr["exact_triangle_calls"],"search_seconds":sr["seconds"],
        "full_exact_pairs":sum((b.get("result") or {}).get("count",0) for b in fr["banks"]),
        "full_inside_pairs":fc["contained_pairs"],"full_self_pairs":fc["self_pairs"],
        "candidate_selected":json.load(open(rd/"final/state.json"))["candidate_selected"],
        "search_report_sha256":sha(rd/"search/report.json"),"full_report_sha256":sha(rd/"full-check/report.json")})
c538=D.parent/"connected-quality-patch-538/run-001/report.json"; a538=D.parent/"connected-quality-patch-538/root-exact-run001/audit-159/report.json"; r538=json.load(open(c538))
report={
 "schema":"numi.human.surface-crossing-distribution-diagnostic.594.v1","status":"READ_ONLY_DIAGNOSTIC_NO_NEW_AUDIT",
 "candidate":str(cand),"candidate_sha256":sha(cand),"state_path":str(STATE),"state_sha256":sha(STATE),
 "surface_report":str(surf),"surface_report_sha256":sha(surf),"point_summary":str(psum),"point_summary_sha256":sha(psum),
 "surface_pairs":len(allpairs),"unique_crossed_skin_faces":len(crossfaces),
 "unique_crossed_bank_faces":len({(q[2],q[3],q[1]) for q in allpairs}),
 "active_banks":sorted(bank_rows,key=lambda z:-z["pairs"]),
 "bank_name_availability":{"export_report":str(export_report),"export_report_sha256":sha(export_report),
     "source_record_mapping":er.get("source_record_mapping"),"anatomical_member_names_available":False,
     "reporting_rule":"Use typed body/stable_bone_id and registered world centroids; do not infer bone names from IDs."},
 "spatial_crossing_patch":{"face_components":len(comps),"largest_component_face_counts":[len(x) for x in comps[:20]],
     "component_bank_pair_counts":[{"component":i,"skin_face_count":len(comp),"bank_pair_counts":[
       {"body":k[0],"stable_bone_id":k[1],"pair_count":v}
       for k,v in collections.Counter((q[2],q[3]) for q in allpairs if q[0] in set(comp)).most_common(8)]}
      for i,comp in enumerate(comps[:12])],
     "boundary_edge_count":sum(len(x)==1 for x in ef.values()),"boundary_face_count":len(boundary),
     "cross_faces_touching_boundary":sum(int(dist[f]==0) for f in crossfaces),
     "cross_face_boundary_graph_distance_histogram":{str(int(k)):int(v) for k,v in sorted(collections.Counter(int(dist[f]) for f in crossfaces).items())},
     "exact_pair_feature_classification":"Not stored by audit159 report; counts establish topological interior location, not edge-edge versus face-interior narrowphase feature."},
 "crossed_vertex_source_displacement_to_R0_mm":{"percentiles_0_50_90_99_100":pct(disp[cnodes],[0,50,90,99,100]),
     "nodes_over_12mm":int((disp[cnodes]>12.0).sum()),"nodes":len(cnodes),
     "interpretation":"12 mm Euclidean source-cap proximity diagnostic from preserved R0 coordinates; no cap exceedance on these vertices."},
 "cross_face_quality":{"current_to_fixed037_area_ratio_percentiles_0_10_50_90_100":pct(cr),
     "current_to_fixed037_oriented_normal_cosine_percentiles_0_10_50_90_100":pct(cd),
     "cross_faces_negative_current_vs_037_normal":int((cd<0).sum()),
     "current_triangle_min_angle_deg_percentiles_0_10_50_90_100":pct(ca),
     "current_vs_R0_normal_cosine_percentiles_0_10_50_90_100":pct(c0),
     "cross_faces_negative_current_vs_R0_normal":int((c0<0).sum()),
     "normal_note":"Same-index geometry diagnostic only; does not certify registered outward sign."},
 "proven_fixed_coordinate_topology":{"method":"Connected regional patch retriangulation with preserved coordinates/point IDs, 237 typed anchors, 185-bank exact incremental screening and unchanged full545 gates.",
     "owner_source":str(root/"search_connected580.py"),"candidate_generator":str(root/"connected_candidates580.py"),
     "runs":topo,"interpretation":"Verified reductions per completed run. Run 003 starts from 1705, not run 002's 1717; do not add the reductions as a single uninterrupted chain.",
     "538_composer":{"report":str(c538),"report_sha256":sha(c538),"retained_flips":r538["retained_flip_count"],
        "removed_flips":r538["removed_flip_count"],
        "full_exact_pairs":sum((b.get("result") or {}).get("count",0) for b in json.load(open(a538))["banks"]),
        "interpretation":"Self-safe composition evidence, but no measured surface-crossing reduction versus its base is recorded here."}},
 "next_mechanism":"Reuse connected_candidates580.patch_candidates on the exact 1593 state, with centers limited to vertices of the six dominant bank-pair connected patches and candidate patch rows drawn from their edge-connected crossing components plus one-ring neighbors. Apply nonoverlapping same-bank patch batches; retain exact 185-bank strict bankwise pair reduction, no introduced self pairs, fixed037 area/orientation, unchanged boundary/anchors/CSR. Existing search_connected580.py supports a bounded local search and recorded exact counters; do not claim compatibility with refined node domain until its lineage check is run.",
 "limitations":["Read retained state and audit reports; no new exact geometry queries, solver, or audit executed.",
   "No anatomical labels in pinned bank export; spatial grouping is reported without guessed names.",
   "Surface pair list omits exact narrowphase feature classification.",
   "Diagnostic is not geometry admission or physical/native qualification."],
 "topology_580_helper_sha256":sha(root/"connected_candidates580.py"),"topology_580_search_sha256":sha(root/"search_connected580.py"),
 "538_composer_source_sha256":sha(D.parent/"connected-quality-patch-538/compose_surviving_flips_538.py"),
 "point_owner_status":p.get("status"),"point_contained_nodes":p.get("inside_nodes"),"point_exact_self_pairs":p.get("exact_PLC_surface_pairs"),
 "geometry_admitted":False,"native_admission":False,"accepted_physical_steps":0}
OUT.mkdir(parents=True,exist_ok=True); out=OUT/"report.json"; out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps({"path":str(out),"sha256":sha(out),"pairs":report["surface_pairs"],"banks":len(bank_rows),"components":len(comps)}))

