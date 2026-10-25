#!/usr/bin/env python3
import hashlib,json,pathlib
import numpy as np
D=pathlib.Path("/Users/n/Work/numi-human-connected-surface-069/temporary-outer-solid-159/run-004-interval-dp")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def clean(p):
 return [s.split("#",1)[0].strip() for s in p.read_text().splitlines() if s.split("#",1)[0].strip()]
def node(p):
 L=clean(p);n=int(L[0].split()[0]);r={}
 for line in L[1:n+1]:
  t=line.split();r[int(t[0])]=np.array([float(x) for x in t[1:4]],np.float64)
 ids=np.array(sorted(r),np.int64);return ids,np.stack([r[int(i)] for i in ids])
def ele(p):
 L=clean(p);n=int(L[0].split()[0]);ids=[];rows=[]
 for line in L[1:n+1]:
  t=line.split();ids.append(int(t[0]));rows.append([int(x) for x in t[1:5]])
 return np.array(ids,np.int64),np.array(rows,np.int64)
def face(p):
 L=clean(p);n=int(L[0].split()[0]);ids=[];rows=[];marks=[]
 for line in L[1:n+1]:
  t=line.split();ids.append(int(t[0]));rows.append([int(x) for x in t[1:4]]);marks.append(int(t[4]) if len(t)>4 else 0)
 return np.array(ids,np.int64),np.array(rows,np.int64),np.array(marks,np.int64)
srcnpz=D/"interval-dp-capped-solid.npz"
with np.load(srcnpz,allow_pickle=False) as z:
 X=np.asarray(z["source_points_world_m"],np.float64)
 F=np.asarray(z["shell_faces_closed"],np.int64)
 cap=np.asarray(z["temporary_cap_faces"],np.int64)
binary=D/"interval-dp-capped-solid"
ni,P=node(binary.with_suffix(".1.node"));ti,Tids=ele(binary.with_suffix(".1.ele"));fi,Fids,markers=face(binary.with_suffix(".1.face"))
idrow={int(x):i for i,x in enumerate(ni)}
T=np.vectorize(idrow.__getitem__,otypes=[np.int64])(Tids)
Fo=np.vectorize(idrow.__getitem__,otypes=[np.int64])(Fids)
# Exact original coordinates: every input point must occur exactly once and bit-identically.
buckets={}
for i,p in enumerate(P):buckets.setdefault(tuple(p.tolist()),[]).append(i)
matches=[];missing=[];ambig=[]
for i,p in enumerate(X):
 rows=buckets.get(tuple(p.tolist()),[])
 if len(rows)==1:matches.append(rows[0])
 elif not rows:missing.append(i)
 else:ambig.append({"input_row":i,"output_rows":rows})
matched=set(matches)
generated=[i for i in range(len(P)) if i not in matched]
# F32 determinant and owner quality det / longest_edge^3.
P32=P.astype(np.float32).astype(np.float64)
Q=P32[T]
det=np.einsum("ij,ij->i",Q[:,1]-Q[:,0],np.cross(Q[:,2]-Q[:,0],Q[:,3]-Q[:,0]))
maxe2=np.zeros(len(T))
for a in range(4):
 for b in range(a+1,4):
  maxe2=np.maximum(maxe2,np.sum((Q[:,a]-Q[:,b])**2,axis=1))
q=det/np.power(maxe2,1.5)
# Compare every marked PLC facet with exact expected input facet and verify output boundary from tets.
marker_ok=[];marker_bad=[]
for frow,m in enumerate(markers):
 exp=int(m)-1
 if 0<=exp<len(F) and tuple(sorted(map(int,F[exp])))==tuple(sorted(map(int,Fo[frow]))):
  marker_ok.append(int(m))
 else:marker_bad.append({"output_face_row":frow,"marker":int(m),"vertices":Fo[frow].tolist()})
# Unique face incidence derived from tetrahedra.
inc={}
for tet in T:
 for omit in range(4):
  k=tuple(sorted(int(tet[j]) for j in range(4) if j!=omit));inc[k]=inc.get(k,0)+1
tet_boundary={k for k,v in inc.items() if v==1}
plc_faces={tuple(sorted(map(int,t))) for t in Fo}
duptet=len(T)-len({tuple(sorted(map(int,t))) for t in T})
sumvol=float(det.sum()/6.0)
report={
 "status":"AUDITED",
 "input_npz_sha256":sha(srcnpz),
 "tetgen_node_sha256":sha(binary.with_suffix(".1.node")),
 "tetgen_ele_sha256":sha(binary.with_suffix(".1.ele")),
 "tetgen_face_sha256":sha(binary.with_suffix(".1.face")),
 "input_vertices":len(X),"input_closed_faces":len(F),"input_temporary_cap_faces":len(cap),
 "output_nodes":len(P),"output_tetrahedra":len(T),"output_boundary_faces":len(Fo),
 "original_input_points_bitwise_exact_match_count":len(matches),
 "missing_input_points":missing,"ambiguous_input_points":ambig,
 "generated_output_point_count":len(generated),
 "all_original_points_bitwise_exact":not missing and not ambig and len(matches)==len(X),
 "generated_node_row_sample":generated[:20],
 "tet_f32_nonpositive_count":int(np.sum(det<=0)),
 "tet_f32_min_det_m3":float(det.min()),
 "owner_quality_det_over_longest_edge_cubed_min_f32":float(q.min()),
 "owner_quality_below_1e-5_count":int(np.sum(q<1e-5)),
 "owner_quality_below_1e-5_rows":np.flatnonzero(q<1e-5).astype(int).tolist()[:100],
 "duplicate_tetrahedron_count":int(duptet),
 "tetra_boundary_face_count":len(tet_boundary),
 "tet_boundary_equals_plc_exactly":tet_boundary==plc_faces,
 "tet_boundary_missing_plc_faces":len(plc_faces-tet_boundary),
 "tet_boundary_extra_faces":len(tet_boundary-plc_faces),
 "marked_plc_face_count":len(marker_ok),
 "all_plc_markers_exactly_once":len(markers)==len(F) and sorted(map(int,markers))==list(range(1,len(F)+1)),
 "marker_geometry_mismatches":marker_bad[:20],
 "tet_volume_sum_m3":sumvol,
 "input_plc_signed_volume_m3":0.06860425724578324,
 "tet_to_plc_volume_relative_error":abs(sumvol-0.06860425724578324)/0.06860425724578324,
 "construction_only_scope":"single-region temporary cap foundation; no O/I/D layers, material assignment, physical/native qualification"
}
out=D/"tetgen-foundation-audit.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))

