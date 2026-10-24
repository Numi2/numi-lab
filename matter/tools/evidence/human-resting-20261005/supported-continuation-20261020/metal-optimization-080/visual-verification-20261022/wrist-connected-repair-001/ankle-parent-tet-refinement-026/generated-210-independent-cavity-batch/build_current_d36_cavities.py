#!/usr/bin/env python3
import csv,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict,deque
import numpy as np
E=Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
G=E/"wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001/generated"
H=E/"ankle-parent-tet-refinement-026/generated-157-d36-actual007-O-affected-face-rerun/actual007-O-affected-faces.json"
ND=E/"native-app-main-header-integration-151/source-preflight-007/initial-skin-fem-nodes.csv"
OUT=E/"ankle-parent-tet-refinement-026/generated-210-independent-cavity-batch"
OUT.mkdir(exist_ok=True)
N=2411
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
F=list(csv.DictReader(open(G/"faces.csv"))); T=list(csv.DictReader(open(G/"tetrahedra.csv")))
assert len(F)==4777 and len(T)==28632
tri=np.array([[int(f[k]) for k in ("a","b","c")] for f in F],np.int64)
frow=np.array([int(f["face"]) for f in F],np.int64)
fstable=np.array([int(f["stable_source_face"]) for f in F],np.int64)
fparent=np.array([int(f["parent_surface_face"]) for f in F],np.int64)
tn0=np.array([[int(t[k]) for k in ("a","b","c","d")] for t in T],np.int64)
tmat=np.array([int(t["material"]) for t in T],np.int8)
tparent=np.array([int(t["parent_tet"]) for t in T],np.int64)
tface=np.array([int(t["parent_surface_face"]) for t in T],np.int64)
tstable=np.array([int(t["stable_source_face"]) for t in T],np.int64)
assert np.array_equal(np.arange(len(T)),np.array([int(t["tet"]) for t in T]))
# Actual007 emitted geometry indexed by the native layer/row binding; local D36 node ID = layer*N+row.
actual=np.zeros((3*N,3),np.float64); mass=np.zeros(3*N,np.float64); found=np.zeros(3*N,bool)
for r in csv.DictReader(open(ND)):
 idx=int(r["layer"])*N+int(r["row"])
 assert int(r["global_node"])==1666+idx
 actual[idx]=[float(r["x_m"]),float(r["y_m"]),float(r["z_m"])]
 mass[idx]=float(r["mass_kg"]); found[idx]=True
assert found.all()
# Reference coordinates from D36 nodes table.
rows=list(csv.DictReader(open(G/"nodes.csv")))
xyz=np.zeros((3*N,3),np.float64)
for r in rows:
 i=int(r["row"])
 for layer,pre in enumerate(("o","i","d")): xyz[layer*N+i]=[float(r[f"{pre}x"]),float(r[f"{pre}y"]),float(r[f"{pre}z"])]
# Exact current child face -> stable parent face identity is used for tet join.
hit=json.load(open(H)); hitrows=sorted({int(x["d36_face_row"]) for x in hit["intersections"]})
hit_seed_stable=set(fstable[hitrows].tolist())
# A repair may move any original O/I/D row touched by a witnessed crossing.
# Seed every parent surface face in the complete O-row vertex stars, not just
# the hit triangles; this leaves those columns inside the cavity, away from the
# fixed stitch.
hit_node_rows=sorted(set(int(v % N) for v in tri[hitrows].ravel()))
star_mask=np.any(np.isin(tri, np.asarray(hit_node_rows,dtype=np.int64)),axis=1)
star_stable=set(fstable[star_mask].tolist())
seed=hit_seed_stable|star_stable
handstable=89542; assert handstable in seed
# Parent-stable adjacency from shared edges across current child faces.
edge_stables=defaultdict(set)
for s,t in zip(fstable,tri):
 for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edge_stables[tuple(sorted((int(a),int(b))))].add(int(s))
adj=defaultdict(set)
for ids in edge_stables.values():
 for s in ids:adj[s].update(ids-{s})
for s in set(fstable.tolist()):adj.setdefault(int(s),set())
def components(seedset):
 rem=set(seedset);out=[]
 while rem:
  r=rem.pop();c={r};q=[r]
  while q:
   u=q.pop()
   for v in adj[u]&rem:rem.remove(v);c.add(v);q.append(v)
  out.append(c)
 return sorted(out,key=lambda c:(-len(c),min(c)))
def expand(C,r):
 S=set(C)
 for _ in range(r):S|=set().union(*(adj[s] for s in tuple(S)))
 return S
# Preorient each tet positively in actual007; this is only for boundary orientation derivation.
def det(t,pts):
 p=pts[t];return float(np.linalg.det((p[1:]-p[0]).T))
tnd=tn0.copy()
dets=np.array([det(t,actual) for t in tnd])
neg=dets<0
tnd[neg,:2]=tnd[neg,1::-1]
assert np.all(np.abs(dets)>0)
stable_tets=defaultdict(list)
for i,s in enumerate(tstable):stable_tets[int(s)].append(i)
def top(S):
 tids=sorted(i for s in S for i in stable_tets[int(s)])
 c=Counter();ori=defaultdict(Counter);mats=defaultdict(set);owners=defaultdict(list)
 for ti in tids:
  t=tnd[ti]
  for f in ((t[1],t[2],t[3]),(t[0],t[3],t[2]),(t[0],t[1],t[3]),(t[0],t[2],t[1])):
   f=tuple(map(int,f));k=tuple(sorted(f));c[k]+=1;mats[k].add(int(tmat[ti]));owners[k].append((ti,f))
   pm=[k.index(x) for x in f];inv=sum(pm[i]>pm[j] for i in range(3) for j in range(i+1,3));ori[k][1 if inv%2==0 else -1]+=1
 b=[k for k,n in c.items() if n==1];ec=Counter()
 for f in b:
  for a,z in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):ec[tuple(sorted((a,z)))]+=1
 # Boundary vertex link must form one or more cycles, never a pinched disconnected set.
 link=defaultdict(lambda:defaultdict(set))
 for f in b:
  for v in f:
   others=[u for u in f if u!=v];link[v][others[0]].add(others[1]);link[v][others[1]].add(others[0])
 badlinks=[]
 for v,g in link.items():
  if any(len(nbs)!=2 for nbs in g.values()):badlinks.append(v);continue
  remain=set(g);cc=0
  while remain:
   cc+=1;q=[remain.pop()]
   while q:
    u=q.pop()
    for w in g[u]&remain:remain.remove(w);q.append(w)
  if cc!=1:badlinks.append(v)
 badori=sum(1 for k,n in c.items() if n==2 and not(ori[k].get(1)==1 and ori[k].get(-1)==1))
 assert not any(n>2 for n in c.values())
 boundary=[owners[k][0][1] for k in c if c[k]==1]
 interface=[(owners[k][0][1],owners[k][1][1],owners[k][0][0],owners[k][1][0]) for k in c if c[k]==2 and mats[k]=={1,2}]
 # More than 2 selected tets per face is an invalid whole-cavity union.
 return tids,c,b,ec,badlinks,badori,boundary,interface
# Determine minimal cavity per connected affected component. Each hit O vertex row
# must be interior to its component's selected cavity so all three layer nodes can move.
seedcomps=components(seed)
def protected_for(C):
 rows=set()
 for fi in hitrows:
  if int(fstable[fi]) in C:rows.update(int(v%N) for v in tri[fi])
 return sorted(rows)
def good(S, protected_rows):
 z=top(S);tids,c,b,ec,bl,bo,bnd,inter=z
 if not (all(v==2 for v in ec.values()) and not bl and not bo and not any(v>2 for v in c.values())):return False,z
 stitch_nodes={v for f in bnd if len({int(v//N) for v in f})>1 for v in f}
 protected={int(layer*N+row) for row in protected_rows for layer in range(3)}
 return not bool(stitch_nodes & protected),z
cand=[]
for ci,C in enumerate(seedcomps):
 prows=protected_for(C)
 for r in range(12):
  S=expand(C,r);ok,z=good(S,prows)
  if ok:
   cand.append({"component":ci,"rings":r,"seed":C,"stable":S,"top":z,"protected_rows":prows});break
 else:raise RuntimeError(f"no closed cavity containing hit columns away from stitch for component {ci}")
def boundary_nodes(z):return {v for f in z[6] for v in f}
groups=[{"items":[x],"stable":set(x["stable"]),"expanded":0,
         "protected_rows":set(x["protected_rows"])} for x in cand]
final=[]
for iteration in range(100):
 tops=[top(g["stable"]) for g in groups]
 merged=False
 for i in range(len(groups)):
  for j in range(i+1,len(groups)):
   if set(tops[i][0])&set(tops[j][0]):
    groups[i]={"items":groups[i]["items"]+groups[j]["items"],
               "stable":groups[i]["stable"]|groups[j]["stable"],
               "expanded":max(groups[i]["expanded"],groups[j]["expanded"]),
               "protected_rows":groups[i]["protected_rows"]|groups[j]["protected_rows"]}
    groups.pop(j);merged=True;break
  if merged:break
 if merged:continue
 expanded=False
 for g in groups:
  ok,z=good(g["stable"],g["protected_rows"])
  if ok:continue
  for r in range(1,9):
   S=expand(g["stable"],r);ok2,z2=good(S,g["protected_rows"])
   if ok2:
    g["stable"]=S;g["expanded"]+=r;expanded=True;break
  else:raise RuntimeError("merged cavity cannot close with moved columns inside")
 if expanded:continue
 final=[{"stable":set(g["stable"]),"top":top(g["stable"]),
         "expanded_after_merge":g["expanded"],
         "seed_components":[x["component"] for x in g["items"]],
         "protected_rows":sorted(g["protected_rows"])} for g in groups]
 break
else:raise RuntimeError("overlap merge did not converge")
for i,a in enumerate(final):
 for b in final[i+1:]:
  assert not(set(a["top"][0])&set(b["top"][0]))
  shared=boundary_nodes(a["top"])&boundary_nodes(b["top"])
  movedA={layer*N+row for row in a["protected_rows"] for layer in range(3)}
  movedB={layer*N+row for row in b["protected_rows"] for layer in range(3)}
  assert not(shared&(movedA|movedB)), "shared stitch contains a moved source row"
# Helper for exact layer classes and signed physical volumes
def layer_class(f):
 layers=sorted({int(v//N) for v in f})
 return "outer_O" if layers==[0] else "skin_fat_interface_I" if layers==[1] else "deep_D" if layers==[2] else "cut_sidewall"
def vol(x):
 return abs(float(np.linalg.det((x[1:]-x[0]).T)))/6.
manifest={"schema":"numi.human.current-d36-cavity-batch.v1",
 "cavity_overlap_policy":"merge tetrahedral overlap only; shared fixed boundary node/edge IDs are allowed, with byte-identical fixed coordinates",
 "source_method_status":"unchanged topology extraction; not geometry repair",
 "shared_fixed_stitch_pairs":[{"cavity_a":i,"cavity_b":j,"shared_global_local_node_ids":sorted(boundary_nodes(final[i]["top"])&boundary_nodes(final[j]["top"]))} for i in range(len(final)) for j in range(i+1,len(final)) if boundary_nodes(final[i]["top"])&boundary_nodes(final[j]["top"])],
 "inputs":{"d36_manifest_sha256":"467ab80f8973975713bbe33df20aa618ee3af280514d4f7af70e57648aab1fad",
 "faces_csv_sha256":sha(G/"faces.csv"),"tetrahedra_csv_sha256":sha(G/"tetrahedra.csv"),
 "nodes_csv_sha256":sha(G/"nodes.csv"),"actual007_node_dump_sha256":sha(ND),
 "actual007_launch_sha256":"5a139be96b4f4b518b6cb56e987eeebd749e530206fa3db429e4bf0de6459ade",
 "actual007_pose_sha256":"e26eddfa625d29d6136e164726b83b70394d2d936466d5d558ef046ddb7447c6",
 "hit_report_sha256":sha(H)},
 "identity":{ "seed_dense_face_rows":len(hitrows),"seed_stable_parent_faces":len(seed),
 "hit_face_stable_parent_ids":len(hit_seed_stable),"full_vertex_star_stable_parent_ids":len(star_stable),
 "hit_vertex_row_count":len(hit_node_rows),
 "stable_identity_join":"faces.csv stable_source_face to tetrahedra.csv stable_source_face; parent_surface_face retained separately",
 "hand_surface_face_row":4643,"hand_stable_source_face":89542,"hand_parent_surface_face":3776,
 "hand_tet_row":26104,"hand_parent_tet":25186,"hand_failure_included":False},
  "seed_components":len(seedcomps),"hit_face_seed_parent_ids":len(hit_seed_stable),
 "full_hit_vertex_star_parent_ids":len(star_stable),"hit_surface_rows":len(hitrows),
 "hit_vertex_rows":len(hit_node_rows),"exported_cavity_count":len(final),"cavities":[]}
for ci,g in enumerate(final):
 S=g["stable"]; tids,c,b,ec,bl,bo,bnd,inter=g["top"]
 Tsel=tn0[tids].copy(); Tpos=tnd[tids].copy()
 nodes=np.unique(Tsel); assert np.all(found[nodes])
 sfrows=np.flatnonzero(np.isin(fstable,list(S)))
 selected_parent_surface=sorted(set(tface[tids].tolist()))
 # Required hand parent and row checked by exact IDs, not mistaken dense face number.
 hand=bool(np.any(tparent[tids]==25186) and np.any(tstable[tids]==89542))
 if hand:
  assert 26104 in tids and 25186 in tparent[tids] and 89542 in S
  manifest["identity"]["hand_failure_included"]=True
 # raw boundary facets are outward-oriented from positive actual007 tets.
 bf=np.asarray(bnd,np.int64)
 bowner=np.asarray([int(next(ti for ti,ff in owners) if False else -1) for owners in []],np.int64) if False else None
 # recompute exact facet owner data using positive tets
 owners=defaultdict(list)
 for ti in tids:
  t=Tpos[tids.index(ti)] if False else tnd[ti]
  for f in ((t[1],t[2],t[3]),(t[0],t[3],t[2]),(t[0],t[1],t[3]),(t[0],t[2],t[1])):
   f=tuple(map(int,f));owners[tuple(sorted(f))].append((ti,f))
 bf_owner=np.asarray([owners[tuple(sorted(map(int,f)))][0][0] for f in bnd],np.int64)
 bstable=np.asarray([tstable[ti] for ti in bf_owner],np.int64)
 bparent=np.asarray([tface[ti] for ti in bf_owner],np.int64)
 bclass=np.asarray([layer_class(f) for f in bf],dtype="U24")
 iface=np.asarray([x[0] for x in inter],np.int64).reshape((-1,3))
 iface_owner=np.asarray([[x[2],x[3]] for x in inter],np.int64).reshape((-1,2))
 iface_stable=np.column_stack([tstable[iface_owner[:,0]],tstable[iface_owner[:,1]]]) if len(iface_owner) else np.empty((0,2),np.int64)
 # Volumes computed in actual007 Double and baseline Float32 for selected tets.
 dv=np.array([np.linalg.det((actual[t[1:]]-actual[t[0]]).T) for t in Tpos],np.float64)
 f32=actual.astype(np.float32)
 d32=np.array([np.linalg.det((f32[t[1:]].astype(np.float64)-f32[t[0]].astype(np.float64)).T) for t in Tpos],np.float64)
 assert np.all(dv>0) and np.all(d32>0)
 matsums={}
 for code in (1,2):
  ix=np.flatnonzero(tmat[tids]==code)
  matsums[str(code)]={"tet_count":int(len(ix)),
    "actual007_volume_m3":float(sum(vol(actual[Tpos[i]]) for i in ix)),
    "reference_volume_m3":float(sum(vol(xyz[Tpos[i]]) for i in ix)),
    "mass_node_sum_kg_diagnostic":float(mass[nodes].sum())}
 fname=OUT/f"cavity-{ci:02d}-{'hand' if hand else 'region'}.npz"
 np.savez_compressed(fname,
   cavity_component_id=np.asarray([ci],np.int64),
   d36_manifest_sha256=np.asarray(["467ab80f8973975713bbe33df20aa618ee3af280514d4f7af70e57648aab1fad"]),
   selected_stable_source_face_ids=np.asarray(sorted(S),np.int64),
   selected_parent_surface_face_ids=np.asarray(selected_parent_surface,np.int64),
   selected_surface_face_rows=np.asarray(sfrows,np.int64),
   selected_surface_face_node_ids=tri[sfrows],
   selected_global_tet_ids=np.asarray(tids,np.int64),
   selected_parent_tet_ids=tparent[tids],
   selected_tet_parent_surface_face_ids=tface[tids],
   selected_tet_stable_source_face_ids=tstable[tids],
   selected_tet_material_regions=tmat[tids],
   selected_parent_tetrahedra_global_node_ids=Tsel,
   selected_parent_tetrahedra_positive_actual007_global_node_ids=Tpos,
   selected_global_node_ids=nodes,
   selected_node_actual007_world_m=actual[nodes],
   selected_node_reference_world_m=xyz[nodes],
   external_boundary_oriented_global_node_ids=bf,
   external_boundary_owner_tet_rows=bf_owner,
   external_boundary_parent_surface_face_ids=bparent,
   external_boundary_stable_source_face_ids=bstable,
   external_boundary_layer_class=bclass,
   internal_skin_fat_interface_oriented_global_node_ids=iface,
   internal_interface_owner_tet_rows=iface_owner,
   internal_interface_stable_source_face_ids=iface_stable,
   )
 q={"cavity_id":ci,"contains_hand_failure":hand,"seed_component_indices":g["seed_components"],
  "merged_expansion_rings_after_overlap":g["expanded_after_merge"],"stable_source_face_count":len(S),
  "surface_child_face_count":len(sfrows),"selected_tet_count":len(tids),"unique_parent_tet_count":len(set(tparent[tids].tolist())),
  "selected_material_tet_counts":dict(Counter(map(int,tmat[tids]))),
  "parent_tet_row_25186_included":bool(np.any(tparent[tids]==25186)),
  "tet_row_26104_included":bool(26104 in tids),
  "stable89542_included":bool(89542 in S),
  "boundary_facets":len(bf),"boundary_class_counts":dict(Counter(bclass.tolist())),
  "boundary_edge_count":len(ec),"boundary_edge_incidence_hist":dict(Counter(ec.values())),
  "boundary_orientation_bad":int(bo),"boundary_nonmanifold_vertex_link_count":len(bl),
  "internal_skin_fat_interface_facets":len(iface),
  "positive_actual007_F64_min_det_m3":float(dv.min()),
  "positive_actual007_F32_min_det_m3":float(d32.min()),
  "material_volumes":matsums,"diagnostic_mass_sum_unique_nodes_kg_not_donor_debit":float(mass[nodes].sum()),
  "actual007_candidate_npz":str(fname),"sha256":sha(fname),
  "source_field_status":"unmodified baseline cavity topology only; geometry repair and donor-mass transfer are not yet qualified"}
 manifest["cavities"].append(q)
assert manifest["identity"]["hand_failure_included"]
p=OUT/"cavity-batch-manifest.json";p.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps({"manifest":str(p),"sha256":sha(p),"seed_components":len(seedcomps),"exported_cavities":len(final),
 "hand_cavity":next(c for c in manifest["cavities"] if c["contains_hand_failure"]),
 "all_components":[{"id":c["cavity_id"],"hand":c["contains_hand_failure"],"stable":c["stable_source_face_count"],"faces":c["surface_child_face_count"],"tets":c["selected_tet_count"],"boundary_edges":c["boundary_edge_incidence_hist"],"F32minDet":c["positive_actual007_F32_min_det_m3"]} for c in manifest["cavities"]]},indent=2))
