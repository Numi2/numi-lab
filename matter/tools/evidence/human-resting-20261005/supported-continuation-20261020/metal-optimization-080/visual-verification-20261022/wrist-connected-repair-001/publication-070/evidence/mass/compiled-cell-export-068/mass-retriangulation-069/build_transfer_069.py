#!/usr/bin/env python3
import ast,csv,collections,hashlib,json,pathlib
import numpy as np
from scipy.spatial import ConvexHull,QhullError
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068"; C=D/"run-004/source-preflight"
AP=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz"
APN=E/"native-candidate020-conforming-prism-030/source-preflight-006/initial-skin-fem-nodes.csv"
HELP=E/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/build_014-replay.py"
OUT=D/"mass-retriangulation-069"
OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def csvrows(p):
 with pathlib.Path(p).open() as f:return list(csv.DictReader(f))
def tetvol(X,T):
 V=X[T].astype(np.float64);A=np.stack((V[:,1]-V[:,0],V[:,2]-V[:,0],V[:,3]-V[:,0]),axis=2);return np.linalg.det(A)/6
def qmetric(X,T):
 V=X[T].astype(np.float64);L=np.max(np.stack([np.linalg.norm(V[:,i]-V[:,j],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1),axis=1);return np.abs(tetvol(X,T)*6)/np.maximum(L**3,1e-300)
def key(t):return tuple(sorted(map(int,t)))
# Load only pure geometry functions from the retained 398 overlap owner.
tree=ast.parse(HELP.read_text());ns={"np":np,"collections":collections,"ConvexHull":ConvexHull,"QhullError":QhullError}
func=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="tet_intersection_volume"]
exec(compile(ast.Module(body=func,type_ignores=[]),str(HELP),"exec"),ns);overlap=ns["tet_intersection_volume"]
cr=csvrows(C/"compiled-skin-fem-tets.csv");nr=sorted(csvrows(C/"compiled-skin-fem-nodes.csv"),key=lambda r:int(r["global_node"]));an=sorted(csvrows(APN),key=lambda r:int(r["global_node"]));a=np.load(AP,allow_pickle=False)
Told=np.asarray(a["old_tetrahedra"],np.int64);Tnew=np.asarray(a["tetrahedra"],np.int64);Mold=np.asarray(a["old_material_ids"],np.int64);Mnew=np.asarray(a["material_ids"],np.int64)
assert len(cr)==len(Told)==len(Tnew)==28632 and len(nr)==len(an)==7233
gids=np.array([int(r["global_node"]) for r in nr],np.int64);assert np.array_equal(gids,[int(r["global_node"]) for r in an]);base=int(gids[0])
global_tet=np.array([int(r["compiled_tet_row"]) for r in cr],np.int64);source_tet=np.array([int(r["source_tet_row"]) for r in cr],np.int64)
obj=np.array([int(r["object_index"]) for r in cr],np.int32);iy=np.array([int(r["compiled_identity_y"]) for r in cr],np.int32)
mat=np.array([int(r["material_index"]) for r in cr],np.int32);auth=np.array([int(r["authored_material_index"]) for r in cr],np.int32)
gen=np.array([int(r["topology_generation"]) for r in cr],np.uint32);flags=np.array([int(r["flags"]) for r in cr],np.uint32);active=np.array([int(r["identity_active_bit"])==1 for r in cr],bool)
ng=np.array([[int(r[f"global_node{i}"]) for i in range(4)] for r in cr],np.int64);nsrc=np.array([[int(r[f"source_node{i}"]) for i in range(4)] for r in cr],np.int64);nl=ng-base
assert np.array_equal(nl,nsrc) and np.array_equal(source_tet,np.arange(len(cr))) and np.array_equal(Told,nl) and np.all(active) and np.array_equal(mat,auth)
W=np.array([float(r["inverse_rest_w_m3"]) for r in cr],np.float64).astype(np.float32);rho=np.array([float(r["compiled_bulk_x"]) for r in cr],np.float64).astype(np.float32)
Wbits=W.view(np.uint32);rbits=rho.view(np.uint32);pmass=W.astype(np.float64)*rho.astype(np.float64)
invbits=np.array([[int(r[f"inverse_rest_bits_r{rr}c{cc}"]) for rr in range(3) for cc in range(3)] for r in cr],np.uint32).reshape(-1,3,3)
nmass=np.array([float(r["mass_kg"]) for r in nr],np.float32)
Xref=np.array([[float(r[k]) for k in ("rest_x","rest_y","rest_z")] for r in nr],np.float32)
Xcur=np.array([[float(r[k]) for k in ("position_x","position_y","position_z")] for r in nr],np.float32)
Xfit=np.array([[float(r[k]) for k in ("x_m","y_m","z_m")] for r in an],np.float32);assert np.array_equal(Xfit.view(np.uint32),Xcur.view(np.uint32))
parentV=np.abs(tetvol(Xref,Told));childV=np.abs(tetvol(Xref,Tnew))
# Match compiler_impl.cpp: F32 node positions, F64 determinant, then F32 f4 storage.
childCookedV=childV.astype(np.float32).astype(np.float64)
ledger=OUT/"native-active-parent-ledger.npz"
np.savez_compressed(ledger,global_tet_rows=global_tet,source_tet_rows=source_tet,object_indices=obj,identity_y=iy,identity_material_x=mat,authored_material_ids=auth,topology_generation=gen,identity_flags_w=flags,active=active,source_node_rows=nsrc,global_node_ids=ng,row0w_reference_volume_m3=W,row0w_f32_bits=Wbits,bulk_x_density_kg_m3=rho,bulk_x_f32_bits=rbits,parent_mass_kg=pmass,inverse_rest_matrix_f32_bits=invbits,compiled_node_global_ids=gids,compiled_node_mass_kg=nmass)
# Exact row/domain match for most cells. Pair by canonical node set; true changes need geometric overlap.
oldmap=collections.defaultdict(list);newmap=collections.defaultdict(list)
for i,t in enumerate(Told):oldmap[key(t)].append(i)
for i,t in enumerate(Tnew):newmap[key(t)].append(i)
new_to_old=np.full(len(Tnew),-1,np.int64);old_to_new=np.full(len(Told),-1,np.int64)
changed_declared=np.asarray(a["changed_tet_rows"],np.int64)
for k,rs in newmap.items():
 for n,o in zip(rs,oldmap.get(k,[])):new_to_old[n]=o;old_to_new[o]=n
cn=np.flatnonzero(new_to_old<0);co=np.flatnonzero(old_to_new<0);assert len(cn)==len(co)==14
oldmat=np.where(mat==15,1,2).astype(np.int64)
assert np.array_equal(Mold,oldmat)
# Exact topological boundary rows, oriented outward from each selected material cavity.
def bound(X,T,sel):
 inc=collections.defaultdict(list)
 for ti in map(int,sel):
  t=list(map(int,T[ti]))
  for opp in range(4):
   f=[t[j] for j in range(4) if j!=opp];inc[tuple(sorted(f))].append((f,int(t[opp])))
 out={}
 for k,u in inc.items():
  if len(u)>2:raise RuntimeError("nonmanifold local cavity face")
  if len(u)==1:
   f,op=u[0];p=X[np.array(f)].astype(np.float64)
   if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),X[op].astype(np.float64)-p[0])>0:f=[f[0],f[2],f[1]]
   out[k]=min(tuple(f[i:]+f[:i]) for i in range(3))
 return out
boundary={}
for m in (1,2):
 io=co[Mold[co]==m];jn=cn[Mnew[cn]==m]
 bo=bound(Xref,Told,io);bn=bound(Xref,Tnew,jn)
 boundary[str(m)]={"old_changed_cells":len(io),"new_changed_cells":len(jn),"old_faces":len(bo),"new_faces":len(bn),"face_set_equal":set(bo)==set(bn),"oriented_boundary_equal":bo==bn,"old_only":[list(x) for x in sorted(set(bo)-set(bn))[:5]],"new_only":[list(x) for x in sorted(set(bn)-set(bo))[:5]]}
quality={}
for label,X in (("reference_f32",Xref),("current_f32",Xcur)):
 for role,T in (("old",Told),("candidate",Tnew)):
  sv=tetvol(X,T);q=qmetric(X,T)
  quality[f"{label}_{role}"]={"nonpositive_count":int(np.count_nonzero(sv<=0)),"min_signed_volume_m3":float(sv.min()),"min_q":float(q.min()),"q_below_1e-5":int(np.count_nonzero(q<1e-5)),"q_definition":"abs(det)/max_edge_length^3"}
# Changed candidate children clipped against full old native mesh, with conservative AABB pruning.
oldlo=Xref[Told].min(axis=1);oldhi=Xref[Told].max(axis=1);rr=[];cc=[];gvv=[];mvv=[];cross=[];pairs=0
for ni in map(int,cn):
 A=Xref[Tnew[ni]].astype(np.float64);lo=A.min(0);hi=A.max(0)
 cand=np.flatnonzero(np.all(oldhi>=lo-1e-15,axis=1)&np.all(oldlo<=hi+1e-15,axis=1))
 for oi in map(int,cand):
  pairs+=1;z=overlap(A,Xref[Told[oi]].astype(np.float64));thr=max(1e-24,min(childV[ni],parentV[oi])*1e-14)
  if z>thr:
   rr.append(ni);cc.append(oi);gvv.append(z);mvv.append(z)
   if Mnew[ni]!=Mold[oi]:cross.append((ni,oi,z))
# Identity entries for exactly retained cells.
for ni,oi in enumerate(new_to_old):
 if oi>=0:rr.append(ni);cc.append(int(oi));gvv.append(float(parentV[oi]));mvv.append(float(W[oi]))
rr=np.asarray(rr,np.int64);cc=np.asarray(cc,np.int64);gvv=np.asarray(gvv,np.float64);mvv=np.asarray(mvv,np.float64);s=np.lexsort((cc,rr));rr,cc,gvv,mvv=rr[s],cc[s],gvv[s],mvv[s]
childcov=np.bincount(rr,weights=gvv,minlength=len(Tnew))/childV
parentcov=np.bincount(cc,weights=gvv,minlength=len(Told))/parentV
fraction=mvv/W.astype(np.float64)[cc]
parent_mass_fraction_sum=np.bincount(cc,weights=fraction,minlength=len(Told))
childmass=np.bincount(rr,weights=pmass[cc]*fraction,minlength=len(Tnew))
childrho=childmass/childCookedV
den={}
for m,bounds in ((1,(900.,1100.)),(2,(850.,950.))):
 v=childrho[Mnew==m]
 den[str(m)]={"count":len(v),"min_kg_m3":float(v.min()),"max_kg_m3":float(v.max()),"bounds_kg_m3":list(bounds),"within_bounds":bool(np.all((v>=bounds[0])&(v<=bounds[1]))),"new_mass_kg":float(childmass[Mnew==m].sum()),"old_mass_kg":float(pmass[mat=={1:15,2:16}[m]].sum())}
offs=np.r_[0,np.cumsum(np.bincount(rr,minlength=len(Tnew)))]
xfer=OUT/"candidate-parent-overlap-transfer.npz"
entry_kind=np.zeros(len(rr),np.uint8);entry_kind[np.isin(rr,cn)]=1
np.savez_compressed(xfer,child_row_offsets=offs,compiled_parent_local_rows=cc,geometric_overlap_volume_m3=gvv,parent_mass_measure_volume_m3=mvv,mapping_entry_kind=entry_kind,parent_mass_fraction=fraction,child_geometric_fraction=gvv/childV[rr],child_material_ids=Mnew,child_reference_volume_m3=childCookedV,child_geometric_volume_m3=childV,child_mass_kg=childmass,child_density_kg_m3=childrho,child_parent_exact_node_set_row=new_to_old,changed_child_rows=cn,changed_parent_rows=co,parent_source_tet_rows=source_tet,parent_global_tet_rows=global_tet)
# Native cooked-node mass reconstruction verifies exact source ledger arithmetic.
rebuild=np.zeros(len(gids),np.float64)
for i,t in enumerate(Told):rebuild[t]+=pmass[i]*.25
rebuild32=rebuild.astype(np.float32)
# Material patch volume / mass accounting.
patch={}
for m in (1,2):
 io=co[Mold[co]==m];jn=cn[Mnew[cn]==m]
 patch[str(m)]={"old_parent_rows":io.tolist(),"new_child_rows":jn.tolist(),"old_geom_volume_m3":float(parentV[io].sum()),"new_geom_volume_m3":float(childV[jn].sum()),"old_parent_mass_kg":float(pmass[io].sum()),"new_child_mass_kg":float(childmass[jn].sum())}
geometry_positive=all(v["nonpositive_count"]==0 for v in quality.values())
all_patch_boundaries_equal=all(v["oriented_boundary_equal"] for v in boundary.values())
max_geom_coverage_error=max(float(np.max(np.abs(childcov-1))),float(np.max(np.abs(parentcov-1))))
local_region_supported=bool(geometry_positive and all_patch_boundaries_equal and not cross and max_geom_coverage_error<1e-10)
report={"schema":"numi.compiled-parent-retriangulation-overlap-069.v1","status":"local_mass_transfer_audit",
"pins":{"AP_npz_sha256":sha(AP),"AP_fit_nodes_csv_sha256":sha(APN),"native_tets_csv_sha256":sha(C/"compiled-skin-fem-tets.csv"),"native_nodes_csv_sha256":sha(C/"compiled-skin-fem-nodes.csv"),"overlap_helper_sha256":sha(HELP),"parent_ledger_sha256":sha(ledger),"transfer_npz_sha256":sha(xfer)},
"parent_identity":{"active_count":len(cr),"all_active":bool(np.all(active)),"global_tet_range":[int(global_tet.min()),int(global_tet.max())],"source_rows_identity":bool(np.array_equal(source_tet,np.arange(len(cr)))),"object_indices":sorted(set(map(int,obj))),"identity_y":sorted(set(map(int,iy))),"materials":{str(int(m)):int(np.count_nonzero(mat==m)) for m in np.unique(mat)},"source_node_tuples_exact":bool(np.array_equal(nl,nsrc)),"compiled_material_matches_authored":bool(np.array_equal(mat,auth)),
"material_table":{"15":{"density_f32":float(np.median(rho[mat==15])),"density_bits":f"0x{int(rbits[mat==15][0]):08x}","volume_m3":float(W[mat==15].sum()),"parent_mass_kg":float(pmass[mat==15].sum())},"16":{"density_f32":float(np.median(rho[mat==16])),"density_bits":f"0x{int(rbits[mat==16][0]):08x}","volume_m3":float(W[mat==16].sum()),"parent_mass_kg":float(pmass[mat==16].sum())}}},
"candidate":{"old_T_matches_native_order":bool(np.array_equal(Told,nl)),"declared_changed_row_count":int(len(changed_declared)),"declared_changed_rows":changed_declared.tolist(),"set_changed_row_count":int(len(co)),"set_changed_rows":co.tolist(),"ordered_tuple_only_rows":[int(i) for i in changed_declared if sorted(Told[i])==sorted(Tnew[i]) and not np.array_equal(Told[i],Tnew[i])],"changed_parent_count":len(co),"changed_child_count":len(cn),"changed_parent_rows":co.tolist(),"changed_child_rows":cn.tolist(),"same_material_partition_for_unchanged":bool(np.array_equal(Mnew[new_to_old>=0],Mold[new_to_old[new_to_old>=0]])),"patch_by_material":patch,"oriented_boundary":boundary,"local_same_positive_material_region_numerically_supported":local_region_supported},
"quality":quality,
"overlap":{"method":"Changed cavity: retained 398 Float64 clipped-polyhedron intersection volumes on F32 compiled-reference coordinates with AABB broadphase. Unchanged exact node-set rows use one-to-one identity ancestry (geometric volume and compiled parent mass-measure recorded separately). No rational predicate claim.","candidate_pairs":pairs,"nonzeros":len(gvv),"cross_material_nonzeros":len(cross),"cross_material_examples":[{"child":int(n),"parent":int(o),"volume_m3":float(v),"new_material":int(Mnew[n]),"old_material":int(Mold[o])} for n,o,v in cross[:12]],"child_coverage_min_max":[float(childcov.min()),float(childcov.max())],"parent_coverage_min_max":[float(parentcov.min()),float(parentcov.max())],"max_child_coverage_error":float(np.max(np.abs(childcov-1))),"max_parent_coverage_error":float(np.max(np.abs(parentcov-1))),"all_children_covered":bool(np.all(childcov>0)),"all_parents_covered":bool(np.all(parentcov>0)),"parent_mass_fraction_sum_min_max":[float(parent_mass_fraction_sum.min()),float(parent_mass_fraction_sum.max())],"max_parent_mass_fraction_closure_error":float(np.max(np.abs(parent_mass_fraction_sum-1))),"per_material_density_bounds":den,"mass_closure_kg":float(childmass.sum()-pmass.sum()),"mass_closure_relative":float((childmass.sum()-pmass.sum())/pmass.sum()),"child_density_formula":"sum(parent native mass * parent_mass_measure_volume / compiled parent inverseRestRow0.w) / child compiler inverseRestRow0.w (F32 node coords, F64 determinant, F32 storage); changed rows use raw geometric overlap as parent measure, unchanged rows use exact identity measure","no_row_normalization":True},
"cooked_ledger_rebuild":{"sum_native_parent_tet_mass_kg":float(pmass.sum()),"sum_native_compiled_node_mass_kg":float(np.sum(nmass.astype(np.float64))),"max_node_mass_error_from_parent_cell_ledger_g":float(np.max(np.abs(rebuild32.astype(np.float64)-nmass.astype(np.float64)))*1000),"nonzero_f32_node_mass_errors":int(np.count_nonzero(rebuild32.view(np.uint32)!=nmass.view(np.uint32)))},
"artifacts":{"native_parent_ledger":str(ledger),"transfer_npz":str(xfer),"report":str(OUT/"transfer-report.json")},
"limitations":["Parent mass is directly reconstructed from active native compiled material identity, F32 bulk.x, and F32 inverseRestRow0.w; it is not inferred from nodal owner buckets.","The parent cell identity is the frozen compiled source_tet_row, and no per-cell anatomical donor ID is present in the cooked tet identity; donor-region ledger qualification remains open.","Overlap uses a retained Float64 clipping construction with tolerances; exact boundary identity is separately checked by oriented node-ID face tables.","No native rebuild or activation is claimed."]}
(OUT/"transfer-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,indent=2))
