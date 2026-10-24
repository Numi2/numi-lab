#!/usr/bin/env python3
# PRIVATE INTEGRATION ONLY: this copy retains 311 legacy BG/SHELL/constraint-prefix inputs.
# It is not qualified for 398 or any revised geometry; do not run before O conflicts clear.
import hashlib,json,math,os,sys,time,runpy,inspect
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path("/Users/n/numi-human-aortic-contact-opt-20261019")
E=ROOT/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001"
OUT=E/"global-inner-p1-field-402-absolute-variable-integration/generated-run-001"
BG=E/"global-connected-inner-field-293-steiner-edge-collapse-001/candidate.npz"
SHELL=E/"boundary-disk-construction-287/global-outer-dome-287.npz"
BANK=E/"registered-surrounding-tissue-native013-191/registered-tissue-banks.npz"
INV=E/"registered-surrounding-tissue-native013-191/inventory.json"
OLDINV=E/"registered-surrounding-tissue-native007-190/inventory.json"
POSE=E/"wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json"
OBS=E/"robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py"
MAT=E/"conforming-p1-material-volume-owner-295/tools/human_skin_p1_material_volume.py"
CERT=E/"conforming-material-boundary-owner-294/tools/human_skin_material_boundary.py"
LIB126=E/"ankle-parent-tet-refinement-026/build_common_volume_126.py"
EXACT=Path("/tmp/exact_level_p1_280.py")
LEVEL=E/"edge-mask-refinement-254/tools/human_skin_patch_geometry.py"
ABS=E/"global-inner-p1-field-402-absolute-variable-integration/absolute_field_402.py"
if OUT.exists(): raise SystemExit(f"refuse overwrite {OUT}")
OUT.mkdir(parents=True)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def log(stage,**kw):
 print(json.dumps({"stage":stage,"elapsed_s":time.monotonic()-t0,**kw},sort_keys=True),flush=True)
def flux(P,F):
 P=np.asarray(P,np.float64);F=np.asarray(F,np.int64)
 if not len(F):return 0.0
 a=P[F[:,0]];b=P[F[:,1]];c=P[F[:,2]]
 return float(np.einsum("ij,ij->i",a,np.cross(b,c)).sum()/6.0)
def tet_det(X,T):
 a,b,c,d=(X[T[:,i]].astype(np.float64) for i in range(4))
 return np.einsum("ij,ij->i",b-a,np.cross(c-a,d-a))
def canonical(F):
 return np.sort(np.asarray(F,np.int64),axis=1)
t0=time.monotonic()
# Load tested helper owners without executing their builders.
lib_source=LIB126.read_text()
lib_old_out='OUT=P/"generated-126-common-volume-partition-001"'
lib_new_out='OUT=P/"generated-126-helper-library-311"'
if lib_old_out not in lib_source: raise RuntimeError("126 helper output pin not found")
lib_source=lib_source.replace(lib_old_out,lib_new_out,1)
lib={"__name__":"global_field_geometry_library","__file__":str(LIB126)}
exec(compile(lib_source,str(LIB126),"exec"),lib)
tri_tet_ns={};exec(compile(OBS.read_text(),str(OBS),"exec"),tri_tet_ns)
tri_tet=tri_tet_ns["triangle_tet_barycentric_polygon"]
joint_ns={};exec(compile((E/"joint-p1-exclusion-owner-285/tools/human_skin_p1_obstacles.py").read_text(),str(OBS),"exec"),joint_ns)
solve_field=joint_ns["solve_exclusion_field"]
mat_ns={};exec(compile(MAT.read_text(),str(MAT),"exec"),mat_ns)
split_bands=mat_ns["split_material_bands"];role_faces=mat_ns["material_role_faces"];source_face_map=mat_ns["exterior_source_face_rows"]
cert_ns={};exec(compile(CERT.read_text(),str(CERT),"exec"),cert_ns);certify=cert_ns["certify_material_boundaries"]
exact_ns={};exec(compile(EXACT.read_text(),str(EXACT),"exec"),exact_ns);extract=exact_ns["extract_level_exact"]
level_ns={};exec(compile(LEVEL.read_text(),str(LEVEL),"exec"),level_ns)
sublevel_fractions=level_ns["p1_tetrahedral_sublevel_fractions"]
toy_values=np.asarray([0.0,1.0,1.0,1.0],np.float64)
toy_tets=np.asarray([[0,1,2,3]],np.int64)
toy_fraction=sublevel_fractions(toy_values,toy_tets,0.5)
if toy_fraction.shape!=(1,) or abs(float(toy_fraction[0])-0.125)>1e-15:
 raise ValueError(("analytic P1 volume preflight failed",toy_fraction.tolist()))
api={k:str(inspect.signature(v)) for k,v in {
 "closest_bary":lib["closest_bary"],"triangle_tet_barycentric_polygon":tri_tet,
 "solve_exclusion_field":solve_field,"split_material_bands":split_bands,
 "material_role_faces":role_faces,"exterior_source_face_rows":source_face_map,
 "certify_material_boundaries":certify,"extract_level_exact":extract,"p1_tetrahedral_sublevel_fractions":sublevel_fractions}.items()}
assert len(api)==9 and all(callable(v) for v in (tri_tet,solve_field,split_bands,role_faces,source_face_map,certify,extract,sublevel_fractions))
# Inputs and exact source-shell -> TetGen node map.
b=np.load(BG,allow_pickle=False);s=np.load(SHELL,allow_pickle=False)
X=np.asarray(b["points_world_m"],np.float32);T=np.asarray(b["tetrahedra"],np.int64)
source_pts=np.asarray(s["points_world_m"],np.float32)
M=np.full(len(source_pts),-1,np.int64)
M[np.asarray(b["source_map_used_input_rows"],np.int64)]=np.asarray(b["source_to_output_node_map"],np.int64)
if np.any(M[np.unique(s["shell_faces"])]<0):raise ValueError("shell source row lacks TetGen node map")
Xmap=X[M[np.unique(s["shell_faces"])]]
Smap=source_pts[np.unique(s["shell_faces"])]
if not np.array_equal(Xmap,Smap):raise ValueError("source shell point map is not exact")
Fsource=np.asarray(s["shell_faces"],np.int64);Fshell=M[Fsource]
Fphys=M[np.asarray(s["physical_outer_faces"],np.int64)]
Fcap=M[np.asarray(s["construction_closure_faces"],np.int64)]
if len(Fshell)!=38082 or len(Fphys)!=36397 or len(Fcap)!=1685:raise ValueError("unexpected pinned shell counts")
# Require exact mapped shell boundary parity with actual positive tetrahedral mesh.
local=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
raw=[];orient=[]
for loc in local:
 q=T[:,loc];orient.append(q);raw.append(np.sort(q,axis=1))
allf=np.concatenate(raw);allfo=np.concatenate(orient)
uf,first,count=np.unique(allf,axis=0,return_index=True,return_counts=True)
Tb=uf[count==1]
if set(map(tuple,Tb))!=set(map(tuple,canonical(Fshell))):raise ValueError("mapped source shell not exact tet boundary")
# Orient outer/cap faces from positive tetrahedra, not the source PLC parity.
bmap={tuple(map(int,uf[i])):tuple(map(int,allfo[first[i]])) for i in np.flatnonzero(count==1)}
Fphys=np.asarray([bmap[tuple(row)] for row in canonical(Fphys)],np.int64)
Fcap=np.asarray([bmap[tuple(row)] for row in canonical(Fcap)],np.int64)
d=tet_det(X,T)
if np.any(d<=0):raise ValueError(("background has nonpositive F32 geometry",int(np.sum(d<=0)),float(d.min())))
outer_ids=np.unique(Fphys)
temp_rows=np.flatnonzero(np.asarray(s["temporary_vertex_mask"],bool))
temp_ids=M[temp_rows];temp_ids=temp_ids[temp_ids>=0]
fixed=outer_ids.copy()
fixed_values=np.zeros(len(fixed),np.float64)
cap_boundary_policy={"physical_outer_nodes":"fixed to zero","temporary_construction_closure_nodes":"free variables; not physical surface; boundary-condition policy provisional and not anatomically certified"}
shellvol=float(np.sum(d)/6)
edges=np.stack([np.linalg.norm(X[T[:,i]]-X[T[:,j]],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1)
qbg=np.abs(d)/np.maximum(edges.max(axis=1)**3,1e-30)
log("mapped_shell_verified",points=len(X),tets=len(T),physical_outer_faces=len(Fphys),construction_cap_faces=len(Fcap),
    mapped_shell_faces=len(Fshell),temporary_fixed_nodes=len(temp_ids),fixed_nodes=len(fixed),
    shell_volume_m3=shellvol,min_f32_det_m3=float(d.min()),min_f32_quality=float(qbg.min()),
    source_shell_sha256=sha(SHELL),background_sha256=sha(BG))
# Load complete 689 banks using old inventory order, then append the 19 additions.
inv=json.loads(INV.read_text())
old_inv=json.loads(OLDINV.read_text())
pose_sha=sha(POSE)
if inv["pose_sha256"]!=pose_sha:raise ValueError("new registered-bank pose hash mismatch")
z=np.load(BANK,allow_pickle=False)
legacy_meta=old_inv["banks"];actual_meta=inv["banks"]
if len(legacy_meta)!=670 or len(actual_meta)!=689 or len(old_inv["skipped"])!=19 or len(inv["skipped"])!=0 or len(z.files)!=1378:
 raise ValueError("bank inventory cardinality mismatch")
def bank_sig(m):return (m["kind"],int(m["stable_id"]),int(m["layer"]),tuple(map(int,m["body_indices"])),int(m["vertices"]),int(m["triangles"]))
actual_by_sig={bank_sig(m):m for m in actual_meta}
if len(actual_by_sig)!=689:raise ValueError("new bank identity signatures are not unique")
legacy_sigs={bank_sig(m) for m in legacy_meta}
if len(legacy_sigs)!=670 or not legacy_sigs.issubset(actual_by_sig):raise ValueError("legacy banks do not map uniquely into complete inventory")
meta=list(legacy_meta)+[m for m in actual_meta if bank_sig(m) not in legacy_sigs]
array_meta=[actual_by_sig[bank_sig(m)] for m in meta]
if len(meta)!=689 or [bank_sig(m) for m in meta[:670]]!=[bank_sig(m) for m in legacy_meta]:raise ValueError("logical bank ordering failed")
# Verify byte identity against old arrays before reusing legacy flat-face IDs/chunks.
zold=np.load((E/"registered-surrounding-tissue-native007-190/registered-tissue-banks.npz"),allow_pickle=False)
for oi,oldm in enumerate(legacy_meta):
 am=array_meta[oi];okey=oldm["key"];akey=am["key"]
 for suffix in ("vertices","faces"):
  aa=zold[okey+"_"+suffix];bb=z[akey+"_"+suffix]
  if aa.dtype!=bb.dtype or aa.shape!=bb.shape or aa.tobytes()!=bb.tobytes():raise ValueError(("legacy bank array changed",oi,okey,akey,suffix))
vs=[];fs=[];vo=[0];fo=[0];bankface=np.empty(sum(int(m["triangles"]) for m in meta),np.uint16)
face_meta=[];vtotal=0;ftotal=0
for bi,(m,am) in enumerate(zip(meta,array_meta)):
 key=am["key"];V=np.asarray(z[key+"_vertices"],np.float32);F=np.asarray(z[key+"_faces"],np.int64)
 if V.shape!=(int(m["vertices"]),3) or F.shape!=(int(m["triangles"]),3):raise ValueError(("bank shape mismatch",m["key"],key))
 vs.append(V);fs.append(F+vtotal);vtotal+=len(V);ftotal+=len(F);vo.append(vtotal);fo.append(ftotal)
 bankface[fo[-2]:fo[-1]]=bi
 face_meta.append({"logical_key":m["key"],"array_key":key,"kind":m["kind"],"stable_id":int(m["stable_id"]),"triangles":len(F),"aabb_min_m":m["aabb_min_m"],"aabb_max_m":m["aabb_max_m"]})
allv=np.vstack(vs);allf=np.vstack(fs).astype(np.int64,copy=False)
# Bounded-memory triangle geometry used by the conservative center/radius broadphase.
tri=allv[allf]
bc=tri.astype(np.float64).mean(axis=1)
br=np.zeros(len(tri),np.float64)
for k in range(3):
 br=np.maximum(br,np.linalg.norm(tri[:,k,:].astype(np.float64)-bc,axis=1))
tri_lo=tri.min(axis=1);tri_hi=tri.max(axis=1)
if tri.shape!=(len(allf),3,3) or not np.all(np.isfinite(bc)) or not np.all(np.isfinite(br)):
 raise ValueError("invalid bank-triangle broadphase arrays")
if len(allf)!=4581231:raise ValueError(("bank face total mismatch",len(allf)))
log("banks_loaded",banks=len(meta),missing=len(inv["skipped"]),vertices=len(allv),triangles=len(allf),
    soft_tissue=sum(m["kind"]=="soft_tissue" for m in meta),anatomy=sum(m["kind"]=="anatomy" for m in meta),
    bank_input_sha256=sha(BANK),inventory_sha256=sha(INV),legacy_inventory_sha256=sha(OLDINV),registered_pose_sha256=pose_sha,
    legacy_prefix_order="verified bitwise vertex/face arrays by kind/stable_id/layer/body_indices",appended_new_banks=len(meta)-670)
# 296b phi is a candidate active-set initializer only. Absolute variables are
# not bounded below by it; any omitted cells must be reactivated and certified.
# This initial subset is therefore never a safe-pruning claim.
legacy_cache=np.load(E/"global-inner-p1-field-296b-allbank-exclusion/global-field-cache.npz",allow_pickle=False)
if not np.array_equal(legacy_cache["points_world_m"],X) or not np.array_equal(legacy_cache["tetrahedra"],T):
 raise ValueError("296b field cache does not match current positive background")
phi0=np.asarray(legacy_cache["phi0"],np.float64)
physical_o_cache=np.asarray(legacy_cache["physical_outer_faces"],np.int64)
if not np.array_equal(canonical(physical_o_cache),canonical(Fphys)):raise ValueError("296b physical O table mismatch")
if not np.array_equal(np.asarray(vo[:671],np.int64),legacy_cache["bank_vertex_offsets"]) or not np.array_equal(np.asarray(fo[:671],np.int64),legacy_cache["bank_face_offsets"]):raise ValueError("logical 670 bank offsets do not preserve the cached flat prefix")
target=.341
active=np.flatnonzero(np.min(phi0[T],axis=1)<target)
if not len(active):raise ValueError("no active background tetrahedra")
field_cache=OUT/"global-field-cache.npz"
np.savez_compressed(field_cache,points_world_m=X,tetrahedra=T,phi0=phi0,
 physical_outer_faces=Fphys,construction_closure_faces=Fcap,physical_outer_node_ids=outer_ids,
 temporary_closure_node_ids=temp_ids,source_shell_map=M,source_shell_faces=Fsource,
 bank_vertex_offsets=np.asarray(vo,np.int64),bank_face_offsets=np.asarray(fo,np.int64),
 legacy_phi_cache_sha256=np.asarray(sha(E/"global-inner-p1-field-296b-allbank-exclusion/global-field-cache.npz")))
log("field_cached",path=str(field_cache),sha256=sha(field_cache),phi_quantiles=np.quantile(phi0,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
    active_tets=len(active),total_tets=len(T),initially_omitted_tets=int(len(T)-len(active)),
    exclusion_target=target,initializer="diagnostic field only; absolute LP has no hard lower bound from initializer; omitted-tet coverage must be certified after solve")
# Initial all-bank triangle/tet clipping shortlist. Postsolve checks every omitted
# tet against the returned absolute field and stops before extraction if coverage is missing.
Ta=T[active];tetxyz=X[Ta].astype(np.float64);tc=tetxyz.mean(axis=1)
tr=np.linalg.norm(tetxyz-tc[:,None,:],axis=2).max(axis=1)
tet_lo=tetxyz.min(axis=1);tet_hi=tetxyz.max(axis=1)
bins=np.asarray([.001,.002,.004,.008,.016,.032,.064,.128,.256,1.0])
tb=np.searchsorted(bins,tr,side="left")
if np.any(tb>=len(bins)):raise ValueError(("tet radius bin overrun",float(tr.max())))
trees=[]
for bi,upper in enumerate(bins):
 loc=np.flatnonzero(tb==bi)
 if len(loc):trees.append((active[loc],cKDTree(tc[loc]),float(upper),loc))
xlo=X.min(axis=0).astype(float)-1e-12;xhi=X.max(axis=0).astype(float)+1e-12
# bank triangles that cannot intersect any field tet by global bbox are omitted safely
face_in_box=np.all(tri_hi>=xlo,axis=1)&np.all(tri_lo<=xhi,axis=1)
eligible_faces=np.flatnonzero(face_in_box)
log("clip_prepared",active_tets=len(active),eligible_bank_faces=len(eligible_faces),total_bank_faces=len(allf),
    tet_radius_max_m=float(tr.max()),tet_radius_bins=[{"upper_m":float(x),"count":int(np.sum(tb==i))} for i,x in enumerate(bins)],
    max_bank_triangle_radius_m=float(br.max()))
part_dir=OUT/"constraint-parts";part_dir.mkdir()
old_part_dir=E/"global-inner-p1-field-296b-allbank-exclusion/constraint-parts"
checkpoint_dir=E/"global-inner-p1-field-310-all689-resume"
checkpoint_part_dir=checkpoint_dir/"constraint-parts"
old_parts=sorted(old_part_dir.glob("part-*.npz"))
checkpoint_text=(checkpoint_dir/"partial-constraint-checkpoint.json").read_text()
if checkpoint_text.endswith("\\n"):checkpoint_text=checkpoint_text[:-2]
checkpoint=json.loads(checkpoint_text)
prior_parts=sorted((p for p in checkpoint_part_dir.glob("part-*.npz") if int(p.stem.split("-")[1])>=64),key=lambda p:int(p.stem.split("-")[1]))
prior_numbers=[int(p.stem.split("-")[1]) for p in prior_parts]
if len(old_parts)!=64 or len(prior_parts)!=int(checkpoint["new_parts"]) or prior_numbers!=list(range(64,64+len(prior_parts))):
 raise ValueError(("unexpected reusable chunk counts/numbering",len(old_parts),len(prior_parts),checkpoint.get("new_parts")))

part_index=[];max_reused_face=-1
import shutil
def copy_checked_parts(sources,origin,max_bank_id):
 global max_reused_face
 for src in sources:
  dst=part_dir/src.name;shutil.copyfile(src,dst)
  q=np.load(dst,allow_pickle=False)
  if int(q["bank_ids"].max())>=max_bank_id:raise ValueError(("chunk bank ID outside expected prefix",src.name))
  if np.any(q["face_ids"]<0) or np.any(q["face_ids"]>=len(bankface)) or not np.array_equal(bankface[q["face_ids"]],q["bank_ids"]):raise ValueError(("chunk flat-face/bank identity mismatch",src.name))
  if np.any(q["tet_ids"]<0) or np.any(q["tet_ids"]>=len(T)):raise ValueError(("chunk tetrahedron identity mismatch",src.name))
  if int(q["face_ids"].min())<=max_reused_face:raise ValueError(("duplicate/out-of-order checkpoint face rows",src.name,int(q["face_ids"].min()),max_reused_face))
  max_reused_face=max(max_reused_face,int(q["face_ids"].max()))
  part_index.append({"file":dst.name,"polygons":int(len(q["tet_ids"])),"vertices":int(len(q["polygon_barycentric"])),
   "sha256":sha(dst),"origin":origin,"helper_sha256":sha(E/"robust-p1-construction-clip-owner-302/tools/human_skin_p1_obstacles.py")})
copy_checked_parts(old_parts,"296b legacy chunks; byte-identical logical 670-bank prefix",670)
if max_reused_face!=321407:raise ValueError(("unexpected legacy face checkpoint",max_reused_face))
copy_checked_parts(prior_parts,"310 all-689 robust-clip partial checkpoint",689)
if max_reused_face!=int(checkpoint["last_face_saved"]):raise ValueError(("310 checkpoint face identity mismatch",max_reused_face,checkpoint["last_face_saved"]))
ct=[];cf=[];cb=[];off=[0];flat=[];clip_counts=Counter();part_no=len(part_index)
def flush_part():
 global part_no,ct,cf,cb,off,flat
 if not ct:return
 path=part_dir/f"part-{part_no:05d}.npz"
 bary=np.asarray(flat,np.float64).reshape(-1,4)
 np.savez_compressed(path,tet_ids=np.asarray(ct,np.int64),face_ids=np.asarray(cf,np.int64),
  bank_ids=np.asarray(cb,np.uint16),poly_offsets=np.asarray(off,np.int64),polygon_barycentric=bary)
 part_index.append({"file":path.name,"polygons":len(ct),"vertices":len(bary),"sha256":sha(path),
   "origin":"311 all-689 continuation; robust exact-fallback clip helper"})
 part_no+=1;ct=[];cf=[];cb=[];off=[0];flat=[]
clip_t=time.monotonic();broad=0;aabb=0;polys=0;tri_checked=0
resume_faces=eligible_faces[eligible_faces>max_reused_face]
wall_deadline=t0+285.0
for fs0 in range(0,len(resume_faces),1024):
 if time.monotonic()>=wall_deadline:
  flush_part()
  ck={"status":"bounded_partial_checkpoint","resume_after_flat_face":int(max_reused_face),"last_face_saved":int(max((int(np.load(part_dir/x["file"],allow_pickle=False)["face_ids"].max()) for x in part_index),default=max_reused_face)),"reused_parts":len(part_index),"new_parts":len(part_index)-64,"elapsed_s":time.monotonic()-t0,"triangles_checked_new":tri_checked,"constraints_new":polys}
  (OUT/"partial-constraint-checkpoint.json").write_text(json.dumps(ck,indent=2,sort_keys=True)+"\n")
  log("bounded_partial_checkpoint",**ck)
  raise SystemExit(0)
 idsface=resume_faces[fs0:fs0+1024]
 C=bc[idsface];Rtri=br[idsface]
 # For each bin query is conservative by center-radius enclosure.
 cand_by=[[] for _ in range(len(idsface))]
 for tet_ids,tree,upper,local_ids in trees:
  q=tree.query_ball_point(C,Rtri+upper+2e-12)
  for jj,rows in enumerate(q):
   if rows:cand_by[jj].extend(tet_ids[np.asarray(rows,np.int64)].tolist())
 for jj,fi in enumerate(idsface):
  tri_checked+=1
  cand=np.asarray(cand_by[jj],np.int64)
  if not len(cand):continue
  broad+=len(cand)
  loccand=np.searchsorted(active,cand)
  mask=np.all(tet_hi[loccand]>=tri_lo[fi]-2e-12,axis=1)&np.all(tet_lo[loccand]<=tri_hi[fi]+2e-12,axis=1)
  cand=cand[mask]
  aabb+=len(cand)
  if not len(cand):continue
  P=tri[fi].astype(np.float64)
  for ti in cand:
   B,Cb=tri_tet(tetxyz[np.searchsorted(active,ti)] if False else X[T[int(ti)]].astype(np.float64),P)
   if len(B):
    ct.append(int(ti));cf.append(int(fi));cb.append(int(bankface[fi]));flat.extend(B.tolist());off.append(len(flat));polys+=1
  if len(ct)>=12000:flush_part()
 if (fs0//50000 != max(-1,(fs0-1024)//50000) or fs0+1024>=len(eligible_faces)):
  log("allbank_clip_progress",triangles_checked=tri_checked,eligible_faces=len(eligible_faces),broad_pairs=broad,
      aabb_pairs=aabb,intersection_polygons=polys,parts=len(part_index),elapsed_clip_s=time.monotonic()-clip_t)
flush_part()
manifest={"schema":"numi.global-p1-obstacle-constraints.402.v1","status":"complete","exclusion_target":target,
 "input_sha256":{"background":sha(BG),"shell":sha(SHELL),"banks":sha(BANK),"inventory":sha(INV),"legacy_inventory":sha(OLDINV),"pose":pose_sha,"obstacle_helper":sha(OBS),"joint_helper":sha(E/"joint-p1-exclusion-owner-285/tools/human_skin_p1_obstacles.py"),"absolute_field_helper":sha(ABS)},
 "inventory":{"available_bank_count":len(meta),"missing_count":len(inv["skipped"]),"missing":inv["skipped"],"legacy_prefix_count":len(old_inv["banks"]),"logical_order":"old 670 inventory rows followed by newly registered rows","triangles":len(allf),"vertices":len(allv)},
 "field":{"points":len(X),"tets":len(T),"initial_constraint_tets_minphi_below_target":len(active),"initially_omitted_tets_requiring_certificate":int(len(T)-len(active)),"phi0_quantiles":np.quantile(phi0,[0,.01,.1,.5,.9,.99,1]).tolist()},
 "constraints":{"eligible_triangles":len(eligible_faces),"triangles_checked":tri_checked+int(np.sum(eligible_faces<=max_reused_face)),"new_triangles_checked":tri_checked,"reused_legacy_parts":64,"reused_checkpoint_parts":len(part_index)-64,"resume_after_face_id":max_reused_face,"circumsphere_candidates":broad,"aabb_candidates":aabb,"intersection_polygons":polys,"polygon_vertices":sum(x["vertices"] for x in part_index),"parts":part_index,"elapsed_s":time.monotonic()-clip_t},
 "api_signatures":api,"scope_limitations":inv["limitations"],"absolute_field_policy":{"objective":"same weighted minimum c^T phi as 311","initializer":"diagnostic only, not bound/objective","fixed_nodes":"physical O=0; temporary closure nodes are free under provisional policy","coverage":"postsolve omitted-tet reactivation certificate required"},"status_note":"complete 689 source-pose banks; no missing body-frame banks; old 670 arrays retained in exact prefix order, new 19 appended logically; clipping is Float64 construction evidence; absolute variable LP uses all-bank rows only on initial active subset and fail-closes if postsolve reactivation is required"}
(OUT/"constraints-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
log("constraints_complete",intersection_polygons=polys,polygon_vertices=manifest["constraints"]["polygon_vertices"],
    aabb_pairs=aabb,broad_pairs=broad,parts=len(part_index),elapsed_clip_s=time.monotonic()-clip_t)
if time.monotonic()>=t0+285.0:
 raise SystemExit("clip exceeded bounded 285s budget after writing complete constraints manifest")
# Flatten every retained polygon vertex, then remove only exact duplicate
# (tet,barycentric) rows. No tolerance welding: source chunks remain untouched.
tet_row_chunks=[];bary_row_chunks=[];raw_polygon_count=0
for part in part_index:
 q=np.load(part_dir/part["file"],allow_pickle=False)
 offs=np.asarray(q["poly_offsets"],np.int64);BB=np.asarray(q["polygon_barycentric"],np.float64)
 if offs.shape!=(len(q["tet_ids"])+1,) or offs[0]!=0 or offs[-1]!=len(BB) or np.any(np.diff(offs)<3):
  raise ValueError(("invalid saved polygon offsets",part["file"]))
 if not np.isfinite(BB).all() or np.any(BB<0) or (len(BB) and np.max(np.abs(BB.sum(axis=1)-1))>2e-12):
  raise ValueError(("invalid cached barycentric polygon",part["file"]))
 raw_polygon_count+=len(q["tet_ids"])
 tet_row_chunks.append(np.repeat(np.asarray(q["tet_ids"],np.int64),np.diff(offs)))
 bary_row_chunks.append(BB)
raw_tet=np.concatenate(tet_row_chunks);raw_B=np.concatenate(bary_row_chunks)
del tet_row_chunks,bary_row_chunks
raw_rows=len(raw_tet)
if len(raw_B)!=raw_rows:raise ValueError(("constraint row alignment failed",raw_rows,len(raw_B)))
order=np.lexsort((raw_B[:,3],raw_B[:,2],raw_B[:,1],raw_B[:,0],raw_tet))
st=raw_tet[order];sb=raw_B[order]
keep=np.ones(raw_rows,dtype=bool)
if raw_rows>1:keep[1:]=(st[1:]!=st[:-1])|np.any(sb[1:]!=sb[:-1],axis=1)
unique_tet=st[keep];unique_B=sb[keep]
unique_rows=len(unique_tet)
if unique_rows>raw_rows:raise AssertionError("exact dedup increased rows")
(OUT/"constraint-row-dedup.json").write_text(json.dumps({"raw_polygon_count":int(raw_polygon_count),
 "raw_vertex_rows":int(raw_rows),"unique_exact_tet_bary_rows":int(unique_rows),
 "duplicate_exact_rows_removed":int(raw_rows-unique_rows),
 "identity":"exact Float64 equality of (tet ID,barycentric tuple); no tolerance welding",
 "source_constraint_parts_retained":len(part_index)},indent=2,sort_keys=True)+"\n")
del order,st,sb,keep,raw_tet,raw_B
# Build the joint minimum-change LP directly from NumPy arrays and CSR,
# avoiding per-row Python coefficient lists in the generic helper.
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
vol=np.abs(tet_det(X,T))/6.0
cost=np.bincount(T.ravel(),weights=np.repeat(vol/4,4),minlength=len(X))
cost=np.maximum(cost,float(np.median(cost[cost>0]))*1e-6)
abs_ns={"__name__":"absolute_field_402","__file__":str(ABS)}
exec(compile(ABS.read_text(),str(ABS),"exec"),abs_ns)
solve_absolute=abs_ns["solve_absolute_obstacle_field"]
reactivate_skipped=abs_ns["reactivate_skipped_tets"]
certify_skipped=abs_ns["certify_skipped_tets"]
def solve_sparse_dedup(p,TT,ids,B,fixed_ids,costs,level,maxval,time_limit):
 return solve_absolute(p,TT,ids,B,fixed_ids,np.zeros(len(fixed_ids),np.float64),costs,level,max_value=maxval,time_limit_s=time_limit)
# Tiny independent check of CSR sign/scaling before using the cached field.
toy_p=np.zeros(4,np.float64);toy_T=np.asarray([[0,1,2,3]],np.int64)
toy_ids=np.asarray([0],np.int64);toy_B=np.asarray([[.25,.25,.25,.25]],np.float64)
toy_out,toy_report=solve_sparse_dedup(toy_p,toy_T,toy_ids,toy_B,np.asarray([],np.int64),np.ones(4),.5,1.0,2.0)
if toy_out is None or not toy_report["pass"] or float(toy_B[0]@toy_out[toy_T[0]])<.5-1e-8:
 raise ValueError(("array CSR LP preflight failed",toy_report))
field,lp=solve_sparse_dedup(phi0,T,unique_tet,unique_B,fixed,cost,target,1.0,
    max(1.0,min(120.0,t0+295.0-time.monotonic())))
lp.update(raw_constraint_rows=int(raw_rows),duplicate_exact_rows=int(raw_rows-unique_rows),raw_intersection_polygons=int(raw_polygon_count))
(OUT/"joint-field-solve.json").write_text(json.dumps(lp,indent=2,sort_keys=True)+"\n")
log("joint_field_solve",pass_=bool(lp.get("pass")),status=lp.get("status"),constraints=lp.get("constraints"),
    active_rows=lp.get("active_rows"),conflicts=len(lp.get("conflicts",[])),changed_nodes=lp.get("changed_nodes"),
    max_field=lp.get("maximum_field_value"))
if field is not None:
 coverage=certify_skipped(T,field,active,target)
 missed=np.asarray(coverage["reactivate_tet_ids"],np.int64)
 lp["constraint_coverage_certificate"]=coverage
 lp["constraint_coverage_reactivation_count"]=int(len(missed))
 lp["constraint_coverage_reactivation_first_tets"]=missed[:64].tolist()
 lp["constraint_coverage_certified"]=bool(coverage["pass"])
 lp["fixed_boundary_values"]=cap_boundary_policy
 if len(missed):
  lp["pass"]=False;lp["status"]="constraint_coverage_incomplete_reactivation_required"
  (OUT/"joint-field-solve.json").write_text(json.dumps(lp,indent=2,sort_keys=True)+"\n")
  raise SystemExit("absolute solve exposed omitted tets requiring renewed obstacle clipping; no field emitted")
if field is None:raise SystemExit("absolute all-bank obstacle solve failed; field and constraints are retained")
field_cache2=OUT/"global-obstacle-constrained-field.npz"
np.savez_compressed(field_cache2,points_world_m=X,tetrahedra=T,phi=field,
 physical_outer_faces=Fphys,construction_closure_faces=Fcap,physical_outer_node_ids=outer_ids,
 temporary_closure_node_ids=temp_ids,source_shell_map=M,source_shell_faces=Fsource)
# Conservative common-field levels: exact P1 sublevel volume fraction per source tet.
vals=field[T]
vol_t=vol
def cdf(level):
 fractions=sublevel_fractions(field,T,float(level))
 return float(np.dot(vol_t,fractions))
# Independent analytic one-tet calibration was checked before the expensive clip.

def bisect_volume(targetv,lo,hi):
 vlo=cdf(lo);vhi=cdf(hi)
 if targetv<vlo or targetv>vhi:return None,{"volume_lo":vlo,"volume_hi":vhi,"reachable":False}
 for _ in range(52):
  mid=(lo+hi)*.5;vm=cdf(mid)
  if vm<targetv:lo=mid
  else:hi=mid
 raw=(lo+hi)*.5;u=np.unique(field)
 j=int(np.searchsorted(u,raw))
 choices=[]
 for k in range(max(0,j-1),min(len(u)-1,j+1)):
  a,b=float(u[k]),float(u[k+1])
  if a<raw<b:choices.append((abs((a+b)/2-raw),-(b-a),(a+b)/2))
 if choices:chosen=sorted(choices)[0][2]
 else:chosen=raw
 return float(chosen),{"volume_lo":vlo,"volume_hi":vhi,"raw_root":raw,"selected_safe_level":chosen,"realized_volume":cdf(chosen),"target_volume":targetv,"delta_percent":100*(cdf(chosen)/targetv-1),"reachable":True}
target_skin=.003452570642
target_fat=.011652343239
li,ri=bisect_volume(target_skin,0.0,min(target,0.95))
if li is None:raise SystemExit(("skin reference volume unreachable below exclusion",ri))
ld,rd=bisect_volume(target_skin+target_fat,li,min(target,0.95))
if ld is None:
 ld=min(target-1e-4,0.3409)
 rd={"reachable":False,"max_allowed_level":ld,"realized_total_skin_plus_fat":cdf(ld),"target_total":target_skin+target_fat}
# Ensure selected levels stay strictly ordered and below obstacle exclusion threshold.
if not li<ld or ld>=target:raise ValueError(("invalid selected common-field levels",li,ld,target))
log("levels_selected",skin_level=li,deep_level=ld,skin=ri,combined=rd,exclusion_target=target)
SI=extract(X,T,field,li);SD=extract(X,T,field,ld)
log("interfaces_extracted",skin_points=len(SI["points"]),skin_faces=len(SI["faces"]),deep_points=len(SD["points"]),deep_faces=len(SD["faces"]),
    skin_volume=SI["signed_volume_m3"],deep_volume=SD["signed_volume_m3"])
volume=split_bands(X,T,field,li,ld)
childX=np.asarray(volume["points_world_m"],np.float32);childT=np.asarray(volume["tetrahedra"],np.int64);mids=np.asarray(volume["material_ids"],np.uint32)
cdet=tet_det(childX,childT);edge=np.stack([np.linalg.norm(childX[childT[:,i]]-childX[childT[:,j]],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1).max(axis=1)
q=cdet/np.maximum(edge**3,1e-30)
vskin=float(np.sum(cdet[mids==1])/6);vfat=float(np.sum(cdet[mids==2])/6)
cdf_skin=cdf(li);cdf_fat=cdf(ld)-cdf(li)
if abs(vskin-cdf_skin)>max(1e-10,1e-7*max(vskin,cdf_skin)) or abs(vfat-cdf_fat)>max(1e-10,1e-7*max(vfat,cdf_fat)):
 raise ValueError(("P1 analytic volume/splitter mismatch",vskin,cdf_skin,vfat,cdf_fat))
log("material_split",nodes=len(childX),tets=len(childT),skin_tets=int(np.sum(mids==1)),fat_tets=int(np.sum(mids==2)),
    nonpositive_f32=int(np.sum(cdet<=0)),min_det_m3=float(cdet.min()),min_quality=float(q.min()),
    q_below_1e5=int(np.sum(q<1e-5)),skin_volume_m3=vskin,fat_volume_m3=vfat,
    skin_delta_percent=100*(vskin/target_skin-1),fat_delta_percent=100*(vfat/target_fat-1))
roles=role_faces(childT,mids)
cap_rows=set(map(tuple,canonical(Fcap)))
face_sources=source_face_map(volume,roles,np.vstack((Fphys,Fcap)))
# Map exact source-surface ancestry to construction cap faces (source rows after physical O).
construction_child=[]
for role in ("outer_faces","interface_faces","deep_faces"):
 rows=face_sources.get(role,np.full(0,-1,np.int64))
 faces=roles[role]
 for i,r in enumerate(rows):
  if int(r)>=len(Fphys):construction_child.append(faces[i])
construction_child=np.asarray(construction_child,np.int64).reshape(-1,3)
cert=certify(childX,childT,mids,outer_faces=roles["outer_faces"],interface_faces=roles["interface_faces"],deep_faces=roles["deep_faces"],
 construction_faces=construction_child,minimum_quality=1e-5)
np.savez_compressed(OUT/"global-skin-fat-material-volume.npz",**volume,
 physical_outer_faces=roles["outer_faces"],skin_fat_interface_faces=roles["interface_faces"],deep_exterior_faces=roles["deep_faces"],
 construction_boundary_faces=construction_child,outer_source_face_rows=face_sources["outer_faces"],deep_source_face_rows=face_sources["deep_faces"])
final={"schema":"numi.global-p1-common-field-material-volume.402.v1","status":"construction-candidate-unadmitted",
 "inputs_sha256":{"background":sha(BG),"shell":sha(SHELL),"banks":sha(BANK),"inventory":sha(INV),"legacy_inventory":sha(OLDINV),"pose":pose_sha,
  "obstacle_helper":sha(OBS),"joint_helper":sha(E/"joint-p1-exclusion-owner-285/tools/human_skin_p1_obstacles.py"),
  "material_splitter":sha(MAT),"boundary_certifier":sha(CERT),"exact_extractor":sha(EXACT)},
 "available_banks":len(meta),"missing_bank_frames":inv["skipped"],"bank_triangle_count":len(allf),
 "absolute_field_policy":{"variables":"absolute phi in [0,1]","objective":"weighted minimum c^T phi","initializer":"diagnostic only, not lower bound or objective","physical_O":"fixed exactly to zero","temporary_cap":"free provisional construction boundary; policy unresolved"},
 "background":{"nodes":len(X),"tets":len(T),"min_background_f32_det_m3":float(d.min()),"shell_volume_m3":shellvol,
  "physical_outer_faces":len(Fphys),"temporary_closure_faces":len(Fcap),"temporary_closure_free_nodes":len(temp_ids),"temporary_closure_policy":"provisional; not a physical boundary condition"},
 "initializer":{"method":"diagnostic-only initializer; not a hard bound or objective",
  "exclusion_target":target,"phi0_quantiles":np.quantile(phi0,[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
  "initial_constraint_tets":len(active),"initially_omitted_tets":int(len(T)-len(active))},
 "constraints_manifest_sha256":sha(OUT/"constraints-manifest.json"),"joint_field_solve":lp,
 "levels":{"skin_phi":li,"deep_phi":ld,"skin_reference_volume_m3":target_skin,"fat_reference_volume_m3":target_fat,
  "skin_volume_selection":ri,"combined_volume_selection":rd},
 "material":{"nodes":len(childX),"tets":len(childT),"skin_tets":int(np.sum(mids==1)),"fat_tets":int(np.sum(mids==2)),
  "skin_volume_m3":vskin,"fat_volume_m3":vfat,"skin_delta_percent":100*(vskin/target_skin-1),"fat_delta_percent":100*(vfat/target_fat-1),
  "nonpositive_f32_tets":int(np.sum(cdet<=0)),"min_signed_det_m3":float(cdet.min()),"min_normalized_det":float(q.min()),
  "q_below_1e-5":int(np.sum(q<1e-5)),"boundary_certificate":cert},
 "limitations":["static registered source-pose construction only; not accepted deformed state or native mechanics","construction-only static source-pose field; not accepted deformed state or native mechanics","the old material volumes are accounting targets, not validated anatomical volumes for this independent field","temporary closure surfaces are construction-only and are excluded from physical O identity; construction facets remain flagged","no geometric-overlap CSR, donor mass ledger, reduced mechanical mapping, or native admission supplied"],
 "api_signatures":api,"elapsed_s":time.monotonic()-t0}
(OUT/"candidate-report.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
log("complete",path=str(OUT),status=final["status"],nodes=len(childX),tets=len(childT),certificate=cert.get("pass"),elapsed_s=final["elapsed_s"])

