#!/usr/bin/env python3
"""Bounded physical-O microedge contraction and one local cavity remesh.

Diagnostic background-field candidate only.  The physical boundary is allowed
to move by the explicitly measured contraction; ancestry and volume changes
are recorded instead of claiming exact boundary preservation.
"""
import collections, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError

BASE=Path("/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
IN=BASE/"global-connected-inner-field-293-cavity-remesh-010/candidate.npz"
META=BASE/"global-connected-inner-field-293-cavity-remesh-010/candidate.npz"
OUT=BASE/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse"
OUT.mkdir(parents=True,exist_ok=True)
TETGEN=BASE/"arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0/tetgen"
started=time.monotonic()

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def dets(X,T):
 p=X[T].astype(np.float64)
 return np.linalg.det(np.stack((p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]),axis=2))
def quality(X,T):
 p=X[T].astype(np.float64)
 L=np.max(np.stack([np.linalg.norm(p[:,i]-p[:,j],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1),axis=1)
 return np.abs(dets(X,T))/np.maximum(L**3,1e-300)
def volume(X,T): return float(np.sum(dets(X,T))/6.0)
def face_rows(X,T,sel):
 inc={}; ori={}
 for ti in sorted(sel):
  t=list(map(int,T[ti]))
  for opp in range(4):
   f=[t[j] for j in range(4) if j!=opp];k=tuple(sorted(f));inc.setdefault(k,[]).append(ti);ori.setdefault(k,[]).append(tuple(f))
 out=[]
 for k,ids in inc.items():
  if len(ids)==1:
   f=list(ori[k][0]);tet=T[ids[0]];opp=next(int(v) for v in tet if v not in k);p=X[np.asarray(f)].astype(np.float64)
   if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),X[opp].astype(np.float64)-p[0])>0:f[1],f[2]=f[2],f[1]
   out.append(tuple(f))
  elif len(ids)!=2:raise RuntimeError("nonmanifold selected tet face")
 return out,inc
def edge_hist(F):
 c=collections.Counter()
 for f in F:
  for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):c[tuple(sorted((int(a),int(b))))]+=1
 return c
def read_poly(path):
 lines=[x.strip() for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
 n,dim,na,nm=map(int,lines[0].split()[:4]);assert dim==3
 pts=np.empty((n,3),np.float32)
 for line in lines[1:1+n]:
  q=line.split();pts[int(q[0])-1]=np.asarray(q[1:4],np.float32)
 pos=1+n;nf,mark=map(int,lines[pos].split()[:2]);pos+=1;faces=[]
 for _ in range(nf):
  q=lines[pos].split();pos+=1
  if int(q[0])!=1:raise RuntimeError("expected one triangle per PLC facet")
  q=lines[pos].split();pos+=1
  if int(q[0])!=3:raise RuntimeError("expected triangular PLC facet")
  faces.append([int(q[1])-1,int(q[2])-1,int(q[3])-1])
 return pts,np.asarray(faces,np.int64)
def read_nodes(path):
 lines=[x.strip() for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
 n,dim,na,nm=map(int,lines[0].split()[:4]);assert dim==3
 out={}
 for line in lines[1:1+n]:
  q=line.split();out[int(q[0])]=[float(q[1]),float(q[2]),float(q[3])]
 return out
def read_ele(path):
 lines=[x.strip() for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
 n,nv,na=map(int,lines[0].split()[:3]);assert nv==4
 ids=[];attrs=[]
 for line in lines[1:1+n]:
  q=line.split();ids.append([int(v) for v in q[1:5]]);attrs.append(int(float(q[5])) if na else 1)
 return np.asarray(ids,np.int64),np.asarray(attrs,np.int64)
def barycentric(p,tet):
 A=np.column_stack((tet[1]-tet[0],tet[2]-tet[0],tet[3]-tet[0]));b=np.linalg.solve(A,p-tet[0]);return np.r_[1-b.sum(),b]
def tet_intersection_volume(A,B):
 edges=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3));faces=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
 scale=max(max(np.linalg.norm(A[i]-A[j]) for i,j in edges),max(np.linalg.norm(B[i]-B[j]) for i,j in edges))
 if not scale>0:return 0.
 origin=(A.mean(axis=0)+B.mean(axis=0))*0.5;pts=[]
 def bary(t,p):
  M=np.stack((t[0]-t[3],t[1]-t[3],t[2]-t[3]),axis=1)
  try:u=np.linalg.solve(M,p-t[3])
  except np.linalg.LinAlgError:return np.full(4,-np.inf)
  return np.array([u[0],u[1],u[2],1-u.sum()])
 for p in A:
  if bary(B,p).min()>=-2e-10:pts.append((p-origin)/scale)
 for p in B:
  if bary(A,p).min()>=-2e-10:pts.append((p-origin)/scale)
 for src,other in ((A,B),(B,A)):
  for i,j in edges:
   p0,p1=src[i],src[j]
   for face in faces:
    tri=other[list(face)];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);d0=np.dot(n,p0-tri[0]);d1=np.dot(n,p1-tri[0])
    if d0*d1>=0:continue
    s=d0/(d0-d1)
    if s<=1e-12 or s>=1-1e-12:continue
    p=p0+s*(p1-p0)
    if bary(other,p).min()>=-2e-10:pts.append((p-origin)/scale)
 if len(pts)<4:return 0.
 keep=[]
 for p in pts:
  if not any(np.dot(p-q,p-q)<=1e-18 for q in keep):keep.append(p)
 if len(keep)<4:return 0.
 try:return float(ConvexHull(np.asarray(keep),qhull_options="Qt Qx").volume*scale**3)
 except QhullError:return 0.

z=np.load(IN,allow_pickle=False);X0=z["points_world_m"].astype(np.float32,copy=True);T0=z["tetrahedra"].astype(np.int64,copy=True);A0=z["region_attributes"].astype(np.int64,copy=True)
shell_input=z["physical_shell_faces"].astype(np.int64,copy=True);closure_input=z["construction_closure_faces"].astype(np.int64,copy=True)
# Shell face indices are in the original shell-input row domain, while the
# tetrahedra use the compiled/output volume-node domain. Apply the explicit
# typed map before any edge, closure, boundary, or ancestry operation.
source_to_output=z["source_to_output_node_map"].astype(np.int64);source_rows=z["source_map_used_input_rows"].astype(np.int64)
input_to_output={int(a):int(b) for a,b in zip(source_rows,source_to_output)}
if any(int(i) not in input_to_output for i in np.unique(np.vstack((shell_input,closure_input)))):raise RuntimeError("shell input row lacks explicit output-node mapping")
shell0=np.asarray([[input_to_output[int(i)] for i in f] for f in shell_input],np.int64)
closure0=np.asarray([[input_to_output[int(i)] for i in f] for f in closure_input],np.int64)
input_boundary,_=face_rows(X0,T0,set(range(len(T0))))
input_boundary_keys={tuple(sorted(map(int,f))) for f in input_boundary}
mapped_shell_keys={tuple(sorted(map(int,f))) for f in shell0}
if input_boundary_keys!=mapped_shell_keys:
 raise RuntimeError(f"typed source-shell→volume-node map does not reproduce input tet boundary: tet-only={len(input_boundary_keys-mapped_shell_keys)} shell-only={len(mapped_shell_keys-input_boundary_keys)}")
if np.any(dets(X0,T0)<=0):raise RuntimeError("input candidate has nonpositive Float32 cells")
q0=quality(X0,T0);bad0=np.flatnonzero(q0<1e-5)
if bad0.tolist()!=[164002,164034]:raise RuntimeError(f"unexpected input q witnesses {bad0.tolist()}")
closure_keys={tuple(sorted(map(int,f))) for f in closure0}
is_closure=np.asarray([tuple(sorted(map(int,f))) in closure_keys for f in shell0],bool)
phys_edges={tuple(sorted((int(f[k]),int(f[(k+1)%3])))) for i,f in enumerate(shell0) if not is_closure[i] for k in range(3)}
collapse_edges=[(18028,18029),(18023,18024)]
edge_rows=[];alias=np.arange(len(X0),dtype=np.int64);targets=[];keeps=[];drops=[]
for u,v in collapse_edges:
 edge=tuple(sorted((u,v)));elen=float(np.linalg.norm(X0[u].astype(np.float64)-X0[v].astype(np.float64)))
 if edge not in phys_edges:raise RuntimeError(f"candidate collapse edge {edge} is not a mapped physical-O edge")
 incident=int(sum(u in f and v in f for i,f in enumerate(shell0) if not is_closure[i]))
 if incident!=2:raise RuntimeError(f"physical-O edge {edge} has {incident} incident faces")
 if elen>0.0006:raise RuntimeError(f"physical-O collapse edge {edge} exceeds 0.6mm budget: {elen}")
 target=((X0[u].astype(np.float64)+X0[v].astype(np.float64))*0.5).astype(np.float32)
 alias[v]=u;targets.append(target);keeps.append(u);drops.append(v)
 maxdisp=float(max(np.linalg.norm(X0[u].astype(np.float64)-target.astype(np.float64)),np.linalg.norm(X0[v].astype(np.float64)-target.astype(np.float64))))
 edge_rows.append({"global_nodes":[u,v],"length_before_m":elen,"incident_physical_O_faces":incident,"midpoint_world_m":target.astype(float).tolist(),"max_endpoint_displacement_m":maxdisp,"within_0p3mm_from_010":bool(maxdisp<=0.0003)})
if len(set(drops))!=len(drops) or len(set(keeps))!=len(keeps) or set(drops)&set(keeps):raise RuntimeError("collapse edges overlap and require an explicit combined-link proof")
if any(r["max_endpoint_displacement_m"]>0.0003 for r in edge_rows):raise RuntimeError("collapse exceeds 0.3mm construction displacement budget")
Xc=X0.copy()
for k,target in zip(keeps,targets):Xc[k]=target
Tc=alias[T0];tkeep=np.asarray([len(set(map(int,t)))==4 for t in Tc],bool)
collapsed_tet_rows=np.flatnonzero(~tkeep).astype(np.int64);parent_tet_rows=np.flatnonzero(tkeep).astype(np.int64)
Tc=Tc[tkeep];Ac=A0[tkeep];collapsed_parent_rows=parent_tet_rows.copy()
shell_remap=alias[shell0];shell_keep=np.asarray([len(set(map(int,f)))==3 for f in shell_remap],bool);shell_parent_rows=np.flatnonzero(shell_keep).astype(np.int64);shell_new=shell_remap[shell_keep]
role_physical=~is_closure[shell_parent_rows];role_closure=is_closure[shell_parent_rows]
closure_new=shell_new[role_closure];physical_new=shell_new[role_physical]
if any(n!=2 for n in edge_hist(shell_new).values()):raise RuntimeError("batched collapse does not leave a closed full boundary shell")
if len({tuple(sorted(map(int,f))) for f in shell_new})!=len(shell_new):raise RuntimeError("batched collapse creates duplicate shell faces")
# Check each surviving O/closure face against its original oriented parent.
old_kept=shell0[shell_keep].astype(np.int64);newp=Xc[shell_new].astype(np.float64);oldp=X0[old_kept].astype(np.float64)
oldn=np.cross(oldp[:,1]-oldp[:,0],oldp[:,2]-oldp[:,0]);newn=np.cross(newp[:,1]-newp[:,0],newp[:,2]-newp[:,0]);normal_dot=np.einsum("ij,ij->i",oldn,newn)
if np.any(normal_dot<=0):raise RuntimeError(f"batched collapse reverses {int((normal_dot<=0).sum())} retained shell faces")
parent_area2=np.linalg.norm(oldn,axis=1);child_area2=np.linalg.norm(newn,axis=1);area_ratio=child_area2/np.maximum(parent_area2,1e-300)
# Exact boundary-domain and determinant checks precede TetGen.
allbf,_=face_rows(Xc,Tc,set(range(len(Tc))))
allbf_keys={tuple(sorted(map(int,f))) for f in allbf};shell_keys={tuple(sorted(map(int,f))) for f in shell_new}
boundary_delta={"tet_only_count":len(allbf_keys-shell_keys),"shell_only_count":len(shell_keys-allbf_keys),"tet_only_examples":sorted(allbf_keys-shell_keys)[:8],"shell_only_examples":sorted(shell_keys-allbf_keys)[:8]}
if boundary_delta["tet_only_count"] or boundary_delta["shell_only_count"]:raise RuntimeError(f"batched collapsed boundary differs from mapped shell: {boundary_delta}")
qc=quality(Xc,Tc);badc=np.flatnonzero(qc<1e-5)
if np.any(dets(Xc,Tc)<=0):raise RuntimeError("batched collapsed source mesh has a nonpositive Float32 cell")
# Preserve actual input-shell face-row ancestry. This background candidate does
# not carry a direct detailed-source face table; do not invent one from the
# source-node map. Construction closure rows remain separately typed.

# Build local closed cavity around the collapsed edge star and all remaining q witnesses.
face_to_tets=collections.defaultdict(list);edge_to_tets=collections.defaultdict(list)
for ti,t in enumerate(Tc):
 for opp in range(4):face_to_tets[tuple(sorted(int(t[j]) for j in range(4) if j!=opp))].append(ti)
 for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):edge_to_tets[tuple(sorted((int(t[i]),int(t[j]))))].append(ti)
seed_nodes=set(map(int,Tc[badc].ravel()))|set(map(int,keeps))
sel={int(ti) for ti,t in enumerate(Tc) if any(int(vv) in seed_nodes for vv in t)}
def close_edges(sel):
 for _ in range(100):
  bf,_=face_rows(Xc,Tc,sel);problem=[e for e,n in edge_hist(bf).items() if n!=2]
  if not problem:return
  new=set()
  for e in problem:new.update(ti for ti in edge_to_tets[e] if ti not in sel)
  if not new:raise RuntimeError("cannot close cavity boundary edge stars")
  sel.update(new)
 raise RuntimeError("cavity closure did not converge")
close_edges(sel);rings=[]
for r in range(3):
 bf,_=face_rows(Xc,Tc,sel);add=set()
 for f in bf:
  for ti in face_to_tets[tuple(sorted(map(int,f)))]:
   if ti not in sel:add.add(ti)
 sel.update(add);before=len(sel);close_edges(sel)
 rings.append({"ring":r+1,"face_neighbors_added":len(add),"selected_after":len(sel)})
bfaces,_=face_rows(Xc,Tc,sel)
if any(n!=2 for n in edge_hist(bfaces).values()):raise RuntimeError("collapsed cavity boundary is not closed")
local_old_rows=np.asarray(sorted(sel),np.int64);local_old_tets=Tc[local_old_rows]
local_nodes=np.unique(local_old_tets.ravel());node_local={int(g):i+1 for i,g in enumerate(local_nodes)}
old_cavity_volume=volume(Xc,local_old_tets);maxvol=max(old_cavity_volume/500.0,1e-11)
goodseed=next((int(i) for i in local_old_rows if len(set(map(int,Tc[i])))==4),None)
if goodseed is None:raise RuntimeError("no valid TetGen region seed cell")
seed_xyz=Xc[Tc[goodseed]].astype(np.float64).mean(axis=0)
polyout=OUT/"cavity.poly"
with polyout.open("w") as f:
 f.write(f"{len(local_nodes)} 3 0 1\n")
 boundary_nodes={int(vv) for tri in bfaces for vv in tri}
 for lid,gid in enumerate(local_nodes,1):
  p=Xc[int(gid)];f.write(f"{lid} {float(p[0]):.17g} {float(p[1]):.17g} {float(p[2]):.17g} {1 if int(gid) in boundary_nodes else 0}\n")
 f.write(f"{len(bfaces)} 1\n")
 for tri in bfaces:f.write(f"1 0\n3 {node_local[int(tri[0])]} {node_local[int(tri[1])]} {node_local[int(tri[2])]}\n")
 f.write(f"0\n1\n1 {seed_xyz[0]:.17g} {seed_xyz[1]:.17g} {seed_xyz[2]:.17g} {int(Ac[goodseed])} {maxvol:.17g}\n")
if "--preflight" in sys.argv:
 report={"input_sha256":sha(IN),"input_shell_volume_node_map":{"mapping_rows":len(source_rows),"all_shell_input_nodes_mapped":True,"tet_boundary_exact_before_repair":True},"collapse_edges":edge_rows,"shell":{"faces_before":len(shell0),"faces_after":len(shell_new),"physical_faces_after":len(physical_new),"closure_faces_after":len(closure_new),"maximum_endpoint_displacement_m":max(r["max_endpoint_displacement_m"] for r in edge_rows),"min_parent_child_normal_dot":float(normal_dot.min()),"min_retained_face_double_area_m2":float(child_area2.min()),"min_parent_child_face_double_area_ratio":float(area_ratio.min()),"p01_parent_child_face_double_area_ratio":float(np.quantile(area_ratio,.01)),"tet_boundary_delta_after_collapse":boundary_delta,"physical_surface_area_before_m2":float(np.sum(np.linalg.norm(np.cross(X0[shell0[~is_closure,1]]-X0[shell0[~is_closure,0]],X0[shell0[~is_closure,2]]-X0[shell0[~is_closure,0]]),axis=1))/2),"physical_surface_area_after_m2":float(np.sum(np.linalg.norm(np.cross(Xc[physical_new[:,1]]-Xc[physical_new[:,0]],Xc[physical_new[:,2]]-Xc[physical_new[:,0]]),axis=1))/2)},"input":{"nodes":len(X0),"tets":len(T0),"poor_rows_before":bad0.tolist(),"collapsed_zero_tets":collapsed_tet_rows.tolist(),"after_collapse_tets":len(Tc),"nonpositive_after_collapse":int((dets(Xc,Tc)<=0).sum()),"poor_after_collapse_rows":badc.tolist(),"poor_after_collapse_count":int(len(badc)),"min_q_after_collapse":float(qc.min()),"volume_before_m3":volume(X0,T0),"volume_after_collapse_m3":volume(Xc,Tc),"volume_delta_m3":volume(Xc,Tc)-volume(X0,T0)},"cavity":{"selected_tets":len(sel),"selected_input_rows":local_old_rows.tolist(),"rings":rings,"boundary_facets":len(bfaces),"boundary_edge_incidence":{str(k):int(v) for k,v in collections.Counter(edge_hist(bfaces).values()).items()},"old_volume_m3":old_cavity_volume,"tetgen_max_volume_m3":maxvol,"poly_path":str(polyout)}}
 (OUT/"preflight.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2));raise SystemExit(0)

if boundary_delta["tet_only_count"] or boundary_delta["shell_only_count"]:
 raise RuntimeError(f"collapsed tetra topology boundary differs from remapped O/closure shell: {boundary_delta}")

cmd=[str(TETGEN),f"-pYq1.2/10a{maxvol:.17g}AS5000",str(polyout)]
proc=subprocess.run(cmd,cwd=OUT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
(OUT/"tetgen.stdout.txt").write_text(proc.stdout);(OUT/"tetgen.stderr.txt").write_text(proc.stderr)
if proc.returncode!=0:raise RuntimeError(f"TetGen exit {proc.returncode}: {proc.stderr[-500:]}")
stem=OUT/"cavity";nodes=read_nodes(Path(str(stem)+".1.node"));raw_tets,local_attr=read_ele(Path(str(stem)+".1.ele"));order=sorted(nodes);row={v:i for i,v in enumerate(order)}
local_xyz=np.asarray([nodes[i] for i in order],np.float64).astype(np.float32);local_tets=np.asarray([[row[int(v)] for v in t] for t in raw_tets],np.int64)
ld=dets(local_xyz,local_tets)
for k in np.flatnonzero(ld<0):local_tets[k,[0,1]]=local_tets[k,[1,0]]
ld=dets(local_xyz,local_tets);lq=quality(local_xyz,local_tets)
input_match={Xc[int(g)].tobytes():int(g) for g in local_nodes};all_lookup=collections.defaultdict(list)
for gi,p in enumerate(Xc):all_lookup[p.tobytes()].append(gi)
mapped=[];extras=[];nextid=len(Xc)
for p in local_xyz:
 key=p.tobytes()
 if key in input_match:mapped.append(input_match[key])
 elif key in all_lookup:raise RuntimeError(f"TetGen node aliases untouched global node {all_lookup[key][0]}")
 else:mapped.append(nextid);nextid+=1;extras.append(p.copy())
mapped=np.asarray(mapped,np.int64);extra=np.asarray(extras,np.float32).reshape(-1,3);Xf=np.vstack((Xc,extra)).astype(np.float32);newlocal=mapped[local_tets]
_,oinc=face_rows(local_xyz,local_tets,set(range(len(local_tets))));outbf=[f for f,inc in oinc.items() if len(inc)==1]
outkeys={tuple(sorted(int(mapped[vv]) for vv in f)) for f in outbf};expkeys={tuple(sorted(map(int,f))) for f in bfaces}
if outkeys!=expkeys:raise RuntimeError(f"cavity boundary changed: lost {len(expkeys-outkeys)}, gained {len(outkeys-expkeys)}")
newcv=volume(local_xyz,local_tets)
if abs(newcv-old_cavity_volume)>max(old_cavity_volume*2e-6,1e-14):raise RuntimeError(f"TetGen cavity volume residual {newcv-old_cavity_volume:g}")
keepT=np.ones(len(Tc),bool);keepT[local_old_rows]=False;Tf=np.vstack((Tc[keepT],newlocal)).astype(np.int64);Af=np.r_[Ac[keepT],local_attr.astype(np.int64)]
all_d=dets(Xf,Tf);all_q=quality(Xf,Tf)
if np.any(all_d<=0):raise RuntimeError(f"merged output has {int((all_d<=0).sum())} nonpositive Float32 cells")
globalbf,_=face_rows(Xf,Tf,set(range(len(Tf))));globalkeys={tuple(sorted(map(int,f))) for f in globalbf};expected={tuple(sorted(map(int,f))) for f in shell_new}
if globalkeys!=expected:raise RuntimeError(f"merged boundary differs from collapsed shell: lost {len(expected-globalkeys)} gained {len(globalkeys-expected)}")
# Parent simplex ancestry against the immediate collapsed cavity, with exact
# edge-collapse ancestry separately retained for the moved physical boundary.
par_rows=[];par_bary=[];par_min=[];par_ok=[];oldxyz=Xc[Tc[local_old_rows]].astype(np.float64)
mat=np.stack((oldxyz[:,1]-oldxyz[:,0],oldxyz[:,2]-oldxyz[:,0],oldxyz[:,3]-oldxyz[:,0]),axis=2);inv=np.linalg.inv(mat);v0=oldxyz[:,0]
for p in local_xyz.astype(np.float64):
 uvw=np.einsum("tij,tj->ti",inv,p[None,:]-v0);bb=np.column_stack((1-uvw.sum(1),uvw));m=bb.min(1);valid=np.flatnonzero((m>=-2e-6)&(bb.max(1)<=1+2e-6))
 ix=int(valid[np.argmax(m[valid])]) if len(valid) else int(np.argmax(m));par_rows.append(int(local_old_rows[ix]));par_bary.append(bb[ix]);par_min.append(float(m[ix]));par_ok.append(bool(len(valid)))
# Exact geometric overlap CSR against original pre-collapse donor cells.
old_global_rows=np.unique(np.r_[collapsed_parent_rows[local_old_rows],collapsed_tet_rows]).astype(np.int64)
old_global_tets=T0[old_global_rows];old_vol=np.abs(dets(X0,old_global_tets))/6.;new_vol=np.abs(dets(local_xyz,local_tets))/6.
ox=X0[old_global_tets].astype(np.float64);ctr=ox.mean(1);rad=np.max(np.linalg.norm(ox-ctr[:,None,:],axis=2),axis=1);tree=cKDTree(ctr);maxrad=float(rad.max())
rr=[];cc=[];vv=[];paircount=0
for ni,tt in enumerate(local_tets):
 A=local_xyz[tt].astype(np.float64);c=A.mean(0);r=float(np.max(np.linalg.norm(A-c,axis=1)));candidates=tree.query_ball_point(c,r+maxrad+1e-12);paircount+=len(candidates);lo=A.min(0);hi=A.max(0)
 for oi in candidates:
  B=ox[oi]
  if np.any(B.max(0)<lo-1e-12) or np.any(B.min(0)>hi+1e-12):continue
  val=tet_intersection_volume(A,B)
  if val>max(1e-24,old_vol[oi]*1e-14):rr.append(ni);cc.append(oi);vv.append(val)
rr=np.asarray(rr,np.int64);cc=np.asarray(cc,np.int64);vv=np.asarray(vv,np.float64)
rs=np.bincount(rr,weights=vv,minlength=len(local_tets));cs=np.bincount(cc,weights=vv,minlength=len(old_global_rows))
childcov=rs/np.maximum(new_vol,1e-30);parentcov=cs/np.maximum(old_vol,1e-30)
ordr=np.lexsort((cc,rr));rr,cc,vv=rr[ordr],cc[ordr],vv[ordr];cnt=np.bincount(rr,minlength=len(local_tets));offs=np.r_[0,np.cumsum(cnt)]
childfrac=vv/np.maximum(rs[rr],1e-300);parentfrac=vv/np.maximum(cs[cc],1e-300)
# Aligned role and parent surface ancestry after collapse.
np.savez_compressed(OUT/"candidate.npz",points_world_m=Xf,tetrahedra=Tf,region_attributes=Af,
 physical_shell_faces=shell_new,physical_shell_parent_input_face_row=shell_parent_rows,physical_shell_face_is_physical_O=role_physical,
 physical_outer_faces=physical_new,physical_outer_parent_input_face_row=shell_parent_rows[role_physical],
 construction_closure_faces=closure_new,construction_closure_parent_input_face_row=shell_parent_rows[role_closure],
 surface_collapse_edges_global=np.asarray(collapse_edges,np.int64),surface_collapse_alias_pairs=np.column_stack((drops,keeps)).astype(np.int64),surface_collapse_targets_world_m=np.asarray(targets,np.float32),
 surface_collapse_endpoint_parameters=np.full(len(collapse_edges),0.5,np.float64),surface_collapse_endpoint_residual_world_m=np.vstack([np.vstack((target.astype(float)-X0[u].astype(float),target.astype(float)-X0[v].astype(float))) for (u,v),target in zip(collapse_edges,targets)]),
 cavity_replaced_input_collapsed_tet_rows=local_old_rows,cavity_replaced_original_parent_tet_rows=collapsed_parent_rows[local_old_rows],
 cavity_child_tetrahedra_global=newlocal,cavity_child_material_attributes=local_attr,
 cavity_child_node_parent_collapsed_tet_row=np.asarray(par_rows,np.int64),cavity_child_node_parent_original_tet_row=collapsed_parent_rows[np.asarray(par_rows,np.int64)],
 cavity_child_node_parent_barycentric=np.asarray(par_bary,np.float64),cavity_child_node_parent_barycentric_min=np.asarray(par_min,np.float64),cavity_child_node_parent_contains=np.asarray(par_ok,bool),
 overlap_child_offsets=offs,overlap_parent_original_input_tet_row=old_global_rows[cc],overlap_intersection_volume_m3=vv,overlap_child_normalized_fraction=childfrac,overlap_parent_conservative_fraction=parentfrac,
 overlap_child_coverage_ratio=childcov,overlap_parent_coverage_ratio=parentcov,overlap_parent_input_tet_rows=old_global_rows,
 source_to_output_node_map=alias[source_to_output],source_map_used_input_rows=source_rows,collapsed_node_aliases=z["collapsed_node_aliases"],
 prior_replaced_parent_tet_rows=z["replaced_parent_tet_rows"],prior_added_node_parent_tet_rows=z["added_node_parent_tet_rows"],prior_added_node_parent_barycentric=z["added_node_parent_barycentric"],prior_added_node_cavity_id=z["added_node_cavity_id"],
 replaced_parent_tet_rows=np.unique(np.r_[z["replaced_parent_tet_rows"],collapsed_parent_rows[local_old_rows],collapsed_tet_rows]).astype(np.int64),
 added_node_parent_tet_rows=z["added_node_parent_tet_rows"],added_node_parent_barycentric=z["added_node_parent_barycentric"],added_node_cavity_id=z["added_node_cavity_id"])
report={"schema":"numi.global-tetgen-293-physical-O-batched-microedge-collapse-014.v1","status":"diagnostic_candidate_not_physical_admission","input_candidate":str(IN),"input_candidate_sha256":sha(IN),"tetgen_binary":str(TETGEN),"tetgen_sha256":sha(TETGEN),
 "method":"Simultaneously contract two measured physical-O microedges to their midpoints, bounded by 0.3 mm maximum endpoint displacement from candidate010, then locally remesh the complete affected cavity with fixed remapped shell. This background mesh has no direct detailed-source face table; physical O and construction closure retain separate input-shell face ancestry.",
 "surface_microedge_contractions":edge_rows,"maximum_endpoint_displacement_m":max(r["max_endpoint_displacement_m"] for r in edge_rows),"boundary":{"full_shell_faces_before":len(shell0),"full_shell_faces_after":len(shell_new),"physical_O_faces_before":int((~is_closure).sum()),"physical_O_faces_after":len(physical_new),"construction_closure_faces_before":int(is_closure.sum()),"construction_closure_faces_after":len(closure_new),"shell_edge_incidence_histogram":{str(k):int(v) for k,v in collections.Counter(edge_hist(shell_new).values()).items()},"duplicate_faces":0,"min_parent_to_child_normal_dot":float(normal_dot.min()),"physical_surface_area_before_m2":float(np.sum(np.linalg.norm(np.cross(X0[shell0[~is_closure,1]]-X0[shell0[~is_closure,0]],X0[shell0[~is_closure,2]]-X0[shell0[~is_closure,0]]),axis=1))/2),"physical_surface_area_after_m2":float(np.sum(np.linalg.norm(np.cross(Xc[physical_new[:,1]]-Xc[physical_new[:,0]],Xc[physical_new[:,2]]-Xc[physical_new[:,0]]),axis=1))/2),"exact_boundary_claim":False,"ancestry":"surviving shell facets retain their original input-shell face-row identity; facets collapsed to zero area are omitted; closure facets remain tagged separately; no detailed-source face identity is fabricated"},
 "collapse_preflight":{"input_low_quality_rows":bad0.tolist(),"removed_degenerate_volume_tets":collapsed_tet_rows.tolist(),"remaining_low_quality_rows":badc.tolist(),"remaining_low_quality_count":int(len(badc)),"remaining_min_quality":float(qc.min()),"all_remaining_F32_positive":bool(np.all(dets(Xc,Tc)>0)),"global_volume_before_m3":volume(X0,T0),"global_volume_after_m3":volume(Xc,Tc),"global_volume_delta_m3":volume(Xc,Tc)-volume(X0,T0),"cavity_selected_tets":len(sel),"cavity_boundary_facets":len(bfaces),"cavity_rings":rings},
 "cavity":{"tetgen_command":cmd,"returncode":proc.returncode,"nodes":len(local_xyz),"tets":len(local_tets),"nonpositive_F32":int((ld<=0).sum()),"min_quality":float(lq.min()),"below_1e-5":int((lq<1e-5).sum()),"local_boundary_matches_after_collapse":outkeys==expkeys,"regional_volume_before_m3":old_cavity_volume,"regional_volume_after_m3":newcv,"regional_volume_delta_m3":newcv-old_cavity_volume,"child_node_parent_simplex_contained_count":int(np.sum(par_ok)),"child_node_parent_simplex_outside_count":int(len(par_ok)-np.sum(par_ok)),"child_node_worst_bary_min":float(min(par_min))},
 "overlap_mapping":{"raw_intersection_entries":int(len(vv)),"candidate_pair_count":int(paircount),"minimum_child_covered_fraction":float(childcov.min(initial=1)),"maximum_child_coverage_ratio":float(childcov.max(initial=0)),"minimum_old_parent_covered_fraction":float(parentcov.min(initial=1)),"maximum_old_parent_coverage_ratio":float(parentcov.max(initial=0)),"uncovered_child_cells":int(np.sum(rs<=0)),"uncovered_old_parent_cells":int(np.sum(cs<=0)),"child_ancestry_fractions_sum_to_one":bool(np.allclose(np.bincount(rr,weights=childfrac,minlength=len(local_tets)),1,atol=2e-12)),"conservative_parent_transfer_fractions_sum_to_one_for_covered_donors":bool(np.allclose(np.bincount(cc,weights=parentfrac,minlength=len(old_global_rows))[cs>0],1,atol=2e-12)),"raw_overlap_volume_before_normalization_m3":float(vv.sum()),"old_parent_geometric_volume_m3":float(old_vol.sum()),"new_child_geometric_volume_m3":float(new_vol.sum()),"uncovered_old_volume_m3":float(np.sum(np.maximum(old_vol-cs,0))),"uncovered_new_volume_m3":float(np.sum(np.maximum(new_vol-rs,0))),"mass_map":"geometric overlap-derived conservative parent transfer fractions only; material density/mass ledger is not represented by this one-region background mesh"},
 "assembled":{"candidate":str(OUT/"candidate.npz"),"candidate_sha256":sha(OUT/"candidate.npz"),"nodes":len(Xf),"tets":len(Tf),"nonpositive_F32":int((all_d<=0).sum()),"min_quality":float(all_q.min()),"below_1e-5":int((all_q<1e-5).sum()),"boundary_matches_collapsed_shell":globalkeys==expected,"volume_before_m3":volume(X0,T0),"volume_after_m3":volume(Xf,Tf),"volume_delta_m3":volume(Xf,Tf)-volume(X0,T0),"unmodified_outside_tets_preserved":True},"elapsed_s":time.monotonic()-started}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(OUT/"report.json"),"candidate":str(OUT/"candidate.npz"),"collapse":report["surface_microedge_contractions"],"cavity":report["cavity"],"overlap":report["overlap_mapping"],"assembled":report["assembled"],"elapsed_s":report["elapsed_s"]},indent=2))
