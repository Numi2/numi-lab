#!/usr/bin/env python3
import csv, json, hashlib, sys, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, "/tmp")
from human_skin_patch_geometry_250 import clip_surface_halfspace
from human_patch_zipper_111 import zipper_annulus
sys.path.insert(0, "/Users/n/numi-human-anatomy-finish-1378/src")
from numilab_human.skin_source_payload_preflight import decode_payload
BASE=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
GEN=BASE/"elbow-refinement-current-main-024-final080-candidate-source-frames-001/generated"
POSE=BASE/"native-app-main-header-integration-151/source-preflight-005/registered-skin-poses.json"
DUMP=BASE/"native-app-main-header-integration-151/source-preflight-007/initial-skin-fem-nodes.csv"
REG=BASE/"native-app-main-header-integration-151/source-preflight-007/registered-skin.csv"
SKIN=BASE/"left-ankle-rest-preserving-weight-repair-128/candidate.nhskin"
OUT=BASE/"bilateral-plane-separated-source-collar-253"
OUT.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def csv_rows(p):
 with open(p,newline="") as f:return list(csv.DictReader(f))
def canon(t):return tuple(sorted(map(int,t)))
def face_edge_incidence(F):
 inc=defaultdict(list)
 for fi,t in enumerate(F):
  for a,b in zip(t,np.roll(t,-1)):
   inc[tuple(sorted((int(a),int(b))))].append((int(a),int(b),fi))
 return inc
def edge_hist(F):
 inc=face_edge_incidence(F); hist=Counter(len(v) for v in inc.values())
 same=sum(1 for v in inc.values() if len(v)==2 and v[0][:2]==v[1][:2])
 return inc,hist,same
def components(F):
 inc=face_edge_incidence(F); adj=[set() for _ in range(len(F))]
 for uses in inc.values():
  fs=[u[2] for u in uses]
  if len(fs)>1:
   for i in fs:adj[i].update(j for j in fs if j!=i)
 seen=set(); out=[]
 for i in range(len(F)):
  if i in seen:continue
  q=[i];seen.add(i); rows=[]
  while q:
   u=q.pop();rows.append(u)
   for v in adj[u]:
    if v not in seen:seen.add(v);q.append(v)
  out.append(np.asarray(rows,np.int64))
 return out
def boundary_loops_directed(F, cut_nodes):
 inc,hist,same=edge_hist(F)
 bnd=[v[0] for e,v in inc.items() if len(v)==1 and e[0] in cut_nodes and e[1] in cut_nodes]
 out=defaultdict(list);inn=defaultdict(list)
 for a,b,fi in bnd:out[a].append((b,fi));inn[b].append((a,fi))
 verts=set(out)|set(inn)
 if any(len(out[v])!=1 or len(inn[v])!=1 for v in verts):
  raise ValueError(("cut boundary not oriented cycles",Counter((len(out[v]),len(inn[v])) for v in verts)))
 loops=[];used=set()
 for seed in sorted(verts):
  if seed in used:continue
  loop=[];face_rows=[];v=seed
  while v not in used:
   used.add(v);loop.append(v);w,fi=out[v][0];face_rows.append(fi);v=w
  if v!=seed:raise ValueError("cut boundary walk is not closed")
  loops.append((np.asarray(loop,np.int64),np.asarray(face_rows,np.int64)))
 return loops,inc,hist,same,bnd
def proj_area(loop,X,n,origin):
 u=np.cross(n,np.array([0.,0.,1.]))
 if np.linalg.norm(u)<1e-6:u=np.cross(n,np.array([0.,1.,0.]))
 u/=np.linalg.norm(u);v=np.cross(n,u);v/=np.linalg.norm(v);q=X[loop]-origin;a=q@u;b=q@v
 return float(.5*np.sum(a*np.roll(b,-1)-b*np.roll(a,-1)))
def read_poses():
 d=json.load(open(POSE));return {int(b["body_index"]):np.asarray(b["current"]["position_world_m"],float) for b in d["bodies"]}
def map_coarse_indices(F,base_n,cut_start):
 out=F.copy();m=F>=base_n
 out[m]=cut_start+(F[m]-base_n)
 return out
T0=time.time()
# Exact current D36 reference rows and their actual native007 O positions.
face_rows=csv_rows(GEN/"faces.csv");base_faces=np.asarray([[int(r[k]) for k in ("a","b","c")] for r in face_rows],np.int64)
base_parent=np.asarray([int(r["parent_surface_face"]) for r in face_rows],np.int64);base_stable=np.asarray([int(r["stable_source_face"]) for r in face_rows],np.int64)
node_rows=csv_rows(DUMP);Xc=np.zeros((2411,3),np.float64)
for r in node_rows:
 if int(r["layer"])==0:Xc[int(r["row"])]=[float(r[k]) for k in ("x_m","y_m","z_m")]
assert len(Xc)==2411 and len(base_faces)==4777
# Full current registered source topology, source vertex IDs and face ordinals are native payload identities.
reg_rows=csv_rows(REG);Xsrc=np.asarray([[float(r[k]) for k in ("wx","wy","wz")] for r in reg_rows],np.float64)
source=decode_payload(SKIN.read_bytes());Fsrc=np.asarray(source["indices"],np.int64).reshape(-1,3)
assert len(Xsrc)==len(source["vertices_f"])==61950 and len(Fsrc)==108987
poses=read_poses();nR=poses[136]-poses[138];nR/=np.linalg.norm(nR);nL=poses[150]-poses[152];nL/=np.linalg.norm(nL)
axis=nR+nL;axis/=np.linalg.norm(axis);origin=(poses[138]+poses[152])/2
# Parent-face audit established max offending coarse face extent s=.645947m.
s0=.650;s1=.654;d0=float(origin@axis+s0);d1=float(origin@axis+s1)
# Cut the complete D36 coarse surface at the proximal collar plane.
Xco,Fco,pc,ec,wc=clip_surface_halfspace(Xc,base_faces,axis,d1,keep_less=False)
co_cut=set(range(len(Xc),len(Xc)+len(ec)))
co_loops,co_inc,co_hist,co_same,co_bnd=boundary_loops_directed(Fco,co_cut)
co_info=[]
for loop,adjrows in co_loops:
 cen=np.mean(Xco[loop],axis=0); side="L" if cen[0]>-.025 else "R"
 co_info.append(dict(loop=loop,adjrows=adjrows,side=side,centroid=cen,area=proj_area(loop,Xco,axis,origin)))
if len(co_loops)!=2 or {r["side"] for r in co_info}!={"L","R"}:
 raise RuntimeError(("coarse clip did not yield one loop per lower leg",[(r["side"],len(r["loop"])) for r in co_info]))
# Clip the entire registered source mesh at the distal collar plane.
Xs,Fss,ps,es,ws=clip_surface_halfspace(Xsrc,Fsrc,axis,d0,keep_less=True)
comp_rows=components(Fss)
if len(comp_rows)!=2:
 raise RuntimeError(("expected exactly two source lower-limb components",[(len(r),) for r in comp_rows]))
source_data={}
for cr in comp_rows:
 triids=np.unique(Fss[cr]); pts=Xs[triids]; cen=pts.mean(0)
 side="L" if cen[0]>-.025 else "R"
 if side in source_data:raise RuntimeError(("duplicate source side component",side))
 # Clip helper source face ancestry is original NHSKIN face ordinal; validate ranges now.
 par=ps[cr]
 if np.any((par<0)|(par>=len(Fsrc))):raise RuntimeError("invalid source face ancestry")
 # Boundary loop over this one connected source component.
 Fcside=Fss[cr][:,[0,2,1]].copy(); cut_ids=set(range(len(Xsrc),len(Xsrc)+len(es)))
 loops,inc,hist,same,bnd=boundary_loops_directed(Fcside,cut_ids)
 if len(loops)!=1:raise RuntimeError(("source component has non-simple transverse loop",side,[len(x[0]) for x in loops]))
 # It must have no other boundary; source input is closed.
 cut_edges={tuple(sorted((int(a),int(b)))) for a,b,_ in bnd}
 other=[e for e,v in inc.items() if len(v)==1 and e not in cut_edges]
 if other:raise RuntimeError(("unexpected non-cut source boundary",side,len(other)))
 source_data[side]=dict(rows=cr,faces=Fcside,parent_rows=par,loop=loops[0][0],adjrows=loops[0][1],centroid=cen,face_count=len(cr),vertex_ids=triids)
# Side map derived from component, not coordinate mirroring.
# Preserve only source rows used by these two clipped components.
used_source=set();used_scut=set()
for dat in source_data.values():
 for i in np.unique(dat["faces"]):
  if i<len(Xsrc):used_source.add(int(i))
  else:used_scut.add(int(i-len(Xsrc)))
used_source=np.asarray(sorted(used_source),np.int64);used_scut=np.asarray(sorted(used_scut),np.int64)
source_row_map={int(i):2411+j for j,i in enumerate(used_source)}
source_cut_base=2411+len(used_source)
source_cut_map={len(Xsrc)+int(i):source_cut_base+j for j,i in enumerate(used_scut)}
coarse_cut_base=source_cut_base+len(used_scut)
coarse_cut_map={len(Xc)+int(i):coarse_cut_base+j for j,i in enumerate(range(len(ec)))}
Xout=np.vstack([Xc,Xsrc[used_source],Xs[len(Xsrc)+used_scut],Xco[len(Xc):]])
# Create explicit source and coarse node ancestry in output-index domain.
N=len(Xout); node_kind=np.full(N,"coarse",dtype="U16");source_vertex_id=np.full(N,-1,np.int64)
source_edge=np.full((N,2),-1,np.int64);source_edge_w=np.full((N,2),np.nan,np.float64)
coarse_edge=np.full((N,2),-1,np.int64);coarse_edge_w=np.full((N,2),np.nan,np.float64)
for sid,oi in source_row_map.items():node_kind[oi]="source_vertex";source_vertex_id[oi]=sid
for j,k in enumerate(used_scut):
 oi=source_cut_base+j;node_kind[oi]="source_cut";source_edge[oi]=es[k];source_edge_w[oi]=ws[k]
for j in range(len(ec)):
 oi=coarse_cut_base+j;node_kind[oi]="coarse_cut";coarse_edge[oi]=ec[j];coarse_edge_w[oi]=wc[j]
def remap_source_faces(F):
 out=np.empty_like(F)
 for i,v in enumerate(F.reshape(-1)):
  out.reshape(-1)[i]=source_row_map[int(v)] if v<len(Xsrc) else source_cut_map[int(v)]
 return out
def remap_coarse_faces(F):return map_coarse_indices(F,len(Xc),coarse_cut_base)
Fout=[];Kout=[];Sout=[];Pout=[];Dout=[];D36out=[]
# Coarse retained surface and its exact typed parent IDs.
Fco_out=remap_coarse_faces(Fco);Fout.append(Fco_out);Kout.append(np.zeros(len(Fco),np.uint8));Sout.append(np.zeros(len(Fco),np.uint8))
Pout.append(base_parent[pc]);Dout.append(np.full(len(Fco),-1,np.int64));D36out.append(pc.copy())
# Full detailed source components, each face retains its original detailed source face ordinal.
fine_loops={};coarse_loops={};fine_lineage={};coarse_lineage={}
co_boundary_lineage={}
for a,b,fi in co_bnd:co_boundary_lineage[tuple(sorted((int(a),int(b))))]=(int(base_parent[int(pc[fi])]),int(base_stable[int(pc[fi])]),int(pc[fi]))
for row in co_info:
 side=row["side"]; loop=np.asarray([int(v) if int(v)<len(Xc) else coarse_cut_map[int(v)] for v in row["loop"]],np.int64)
 coarse_loops[side]=loop
 for i,v in enumerate(loop):
  e=tuple(sorted((int(v),int(loop[(i+1)%len(loop)]))))
  # map emitted edge back to clip helper edge IDs for exact adjacent coarse face row
  rev={coarse_cut_map[len(Xc)+j]:len(Xc)+j for j in range(len(ec))}
  va=rev.get(int(v),int(v));vb=rev.get(int(loop[(i+1)%len(loop)]),int(loop[(i+1)%len(loop)]))
  oldedge=tuple(sorted((va,vb)))
  uses=[u for u in co_bnd if tuple(sorted((int(u[0]),int(u[1]))))==oldedge]
  if len(uses)!=1:raise RuntimeError(("coarse loop lineage edge missing",side,e,oldedge,len(uses)))
  _,_,fi=uses[0]; oldface=int(pc[fi]);coarse_lineage[(side,int(v))]=(int(base_parent[oldface]),int(base_stable[oldface]),oldface)
# Detailed source faces and source boundary lineage.
for side,dat in source_data.items():
 Fs=remap_source_faces(dat["faces"]); face_rows=np.asarray(dat["parent_rows"],np.int64)
 Fout.append(Fs);Kout.append(np.full(len(Fs),2,np.uint8));Sout.append(np.full(len(Fs),1 if side=="L" else 2,np.uint8))
 Pout.append(np.full(len(Fs),-1,np.int64));Dout.append(face_rows.copy());D36out.append(np.full(len(Fs),-1,np.int64))
 loop=np.asarray([int(v) if int(v)<len(Xsrc) else source_cut_map[int(v)] for v in dat["loop"]],np.int64)
 fine_loops[side]=loop
 inc=face_edge_incidence(Fs)
 for i,v in enumerate(loop):
  e=tuple(sorted((int(v),int(loop[(i+1)%len(loop)]))))
  uses=inc.get(e,[])
  if len(uses)!=1:raise RuntimeError(("source loop edge lacks exactly one face",side,e,len(uses)))
  fi=uses[0][2]; detailed=int(face_rows[fi]);fine_lineage[(side,int(v))]=(detailed,fi)
# Weave short source-to-coarse collars at the two parallel, anatomy-derived cuts.
collar_meta={};
for side,side_code in (("L",1),("R",2)):
 a=fine_loops[side][::-1].copy();b=coarse_loops[side].copy()
 aarea=proj_area(a,Xout,axis,origin);barea=proj_area(b,Xout,axis,origin)
 if np.sign(aarea)==0 or np.sign(aarea)!=np.sign(barea):raise RuntimeError(("collar loops have incompatible geometric winding",side,aarea,barea))
 tri,rep=zipper_annulus(Xout,a,b,quality_floor=1e-5,phase_band=.35,max_crossring_edge=.04)
 cp=np.full(len(tri),-1,np.int64);cd=np.full(len(tri),-1,np.int64)
 cl=set(map(int,coarse_loops[side]));fl=set(map(int,fine_loops[side]))
 for i,t in enumerate(tri):
  cv=next((int(v) for v in t if int(v) in cl),None);fv=next((int(v) for v in t if int(v) in fl),None)
  if cv is None or fv is None:raise RuntimeError(("collar triangle lacks one ring",side,i))
  cp[i]=coarse_lineage[(side,cv)][0];cd[i]=fine_lineage[(side,fv)][0]
 if np.any(cp<0) or np.any(cd<0):raise RuntimeError(("collar ancestry missing",side))
 Fout.append(tri);Kout.append(np.full(len(tri),3,np.uint8));Sout.append(np.full(len(tri),side_code,np.uint8));Pout.append(cp);Dout.append(cd);D36out.append(np.full(len(tri),-1,np.int64));collar_meta[side]=rep
F=np.vstack(Fout);K=np.concatenate(Kout);Side=np.concatenate(Sout);Parent=np.concatenate(Pout);Detail=np.concatenate(Dout);D36=np.concatenate(D36out)
inc,hist,same=edge_hist(F)
if max(hist,default=0)>2 or same:raise RuntimeError(("nonmanifold/winding mismatch",dict(hist),same))
boundary=[e for e,v in inc.items() if len(v)==1]
# Original D36 has only the known 45 head boundary edges; those stay exact.
if len(boundary)!=45:raise RuntimeError(("unexpected remaining boundary count",len(boundary)))
areas=np.linalg.norm(np.cross(Xout[F[:,1]]-Xout[F[:,0]],Xout[F[:,2]]-Xout[F[:,0]]),axis=1)*.5
X32=Xout.astype(np.float32);areas32=np.linalg.norm(np.cross(X32[F[:,1]]-X32[F[:,0]],X32[F[:,2]]-X32[F[:,0]]),axis=1)*.5
if not np.isfinite(Xout).all() or areas.min()<=0 or areas32.min()<=0:raise RuntimeError("nonfinite or zero-area outer triangles")
# Every cut source triangle's detailed identity is the original payload face ordinal.
assert np.all((Detail[K==2]>=0)&(Detail[K==2]<len(Fsrc)))
npz=OUT/"candidate-252-full-source-bilateral-collar.npz"
np.savez_compressed(npz,outer_world_actual007_m=Xout,surface_faces=F,surface_face_kind=K,surface_face_source_side=Side,
 surface_face_parent_surface_ids=Parent,surface_face_detailed_source_ids=Detail,surface_face_original_d36_row=D36,
 plane_normal_actual007=axis,plane_origin_actual007_m=origin,distal_offset_m=np.float64(s0),proximal_offset_m=np.float64(s1),
 source_face_parent_rows=np.r_[*(dat["parent_rows"] for dat in source_data.values())],
 left_source_section_loop=fine_loops["L"],right_source_section_loop=fine_loops["R"],
 left_coarse_section_loop=coarse_loops["L"],right_coarse_section_loop=coarse_loops["R"],
 left_collar_faces=Fout[-2],right_collar_faces=Fout[-1],node_kind=node_kind,source_vertex_ids=source_vertex_id,
 source_cut_edge_vertices=source_edge,source_cut_edge_weights=source_edge_w,coarse_cut_edge_vertices=coarse_edge,coarse_cut_edge_weights=coarse_edge_w,
 source_original_row_ids=used_source,source_cut_original_output_ids=used_scut)
report={"schema":"numi.human.bilateral-full-source-transverse-collar.v1","status":"unadmitted outer candidate; exact self/tissue/bone gates pending; no I/D or volume claim",
 "inputs":{"skin_payload":str(SKIN),"skin_payload_sha256":sha(SKIN),"registered_skin_sha256":sha(REG),"native007_node_dump_sha256":sha(DUMP),"native007_pose_sha256":sha(POSE),"d36_faces_sha256":sha(GEN/"faces.csv")},
 "plane":{"normal_actual007":axis.tolist(),"origin_actual007_m":origin.tolist(),"distal_s0_m":s0,"proximal_s1_m":s1,"width_m":s1-s0,"source_basis":"full NHSKIN128 actual007 registered source triangles clipped at s0; D36 clipped at s1; plane axis bisects bilateral tibia-minus-calcaneus vectors"},
 "source":{"vertices":len(Xsrc),"faces":len(Fsrc),"left_lower_component_faces":int(source_data["L"]["face_count"]),"right_lower_component_faces":int(source_data["R"]["face_count"]),"left_loop":len(fine_loops["L"]),"right_loop":len(fine_loops["R"]),"used_source_vertices":len(used_source),"source_cut_points":len(used_scut),"detailed_face_id_domain":"original NHSKIN payload face-row ordinal; matches D36 DetailedSourceFaceIds (sampled 200/200 candidate223 kind-2 triangles)","whole_component_winding_reversed_from_payload_to_match_D36_physical_O":True},
 "coarse":{"input_faces":len(base_faces),"retained_faces":len(Fco),"cut_points":len(ec),"left_loop":len(coarse_loops["L"]),"right_loop":len(coarse_loops["R"]),"face_parents_preserved":True},
 "collars":collar_meta,"surface":{"nodes":len(Xout),"faces":len(F),"edge_incidence_histogram":dict(hist),"same_direction_edges":same,"boundary_edges":len(boundary),"boundary_matches_original_head":True,"min_area_f64_m2":float(areas.min()),"min_area_actual007_f32_m2":float(areas32.min()),"f32_zero_area_faces":int(np.sum(areas32==0))},
 "candidate_sha256":sha(npz),"elapsed_s":time.time()-T0}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"candidate":str(npz),"sha256":report["candidate_sha256"],"nodes":len(Xout),"faces":len(F),"source_components":[report["source"]["left_lower_component_faces"],report["source"]["right_lower_component_faces"]],"source_loops":[len(fine_loops["L"]),len(fine_loops["R"])],"coarse_loops":[len(coarse_loops["L"]),len(coarse_loops["R"])],"collars":collar_meta,"hist":dict(hist),"area_f32_min":float(areas32.min()),"elapsed_s":report["elapsed_s"]},indent=2))
