#!/usr/bin/env python3
import ast, collections, csv, hashlib, json, pathlib
import numpy as np
from scipy.spatial import ConvexHull, QhullError

E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
LEDGER=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068/mass-retriangulation-069/native-active-parent-ledger.npz"
NODES=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068/run-004/source-preflight/compiled-skin-fem-nodes.csv"
INPUT=E/"connected-volume-repair-032/repair-input.npz"
CHILD=E/"E/connected-volume-refinement-037/run-006/candidate.npz"
RUN=E/"E/connected-volume-refinement-037/run-006"
HELP=E/"global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/build_014-replay.py"
OUT=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068/mass-retriangulation-070"
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def tetvol(X,T):
    V=np.asarray(X)[np.asarray(T,dtype=np.int64)].astype(np.float64)
    A=np.stack((V[:,1]-V[:,0],V[:,2]-V[:,0],V[:,3]-V[:,0]),axis=2)
    return np.linalg.det(A)/6
def qmetric(X,T):
    V=np.asarray(X)[np.asarray(T,dtype=np.int64)].astype(np.float64)
    L=np.max(np.stack([np.linalg.norm(V[:,i]-V[:,j],axis=1) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))],axis=1),axis=1)
    return np.abs(tetvol(X,T)*6)/np.maximum(L**3,1e-300)
def key(t): return tuple(sorted(map(int,t)))
def boundary(X,T,sel):
    inc=collections.defaultdict(list)
    for ti in map(int,sel):
        t=list(map(int,T[ti]))
        for opp in range(4):
            f=[t[j] for j in range(4) if j!=opp]
            inc[tuple(sorted(f))].append((f,int(t[opp])))
    out={}
    for k,u in inc.items():
        if len(u)>2: raise RuntimeError("nonmanifold local cavity face")
        if len(u)==1:
            f,op=u[0]; p=np.asarray(X)[np.asarray(f)].astype(np.float64)
            if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),np.asarray(X)[op].astype(np.float64)-p[0])>0:
                f=[f[0],f[2],f[1]]
            out[k]=min(tuple(f[i:]+f[:i]) for i in range(3))
    return out

# Extract the retained overlap helper without executing the original build script.
tree=ast.parse(HELP.read_text())
ns={"np":np,"collections":collections,"ConvexHull":ConvexHull,"QhullError":QhullError}
func=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="tet_intersection_volume"]
exec(compile(ast.Module(body=func,type_ignores=[]),str(HELP),"exec"),ns)
intersect=ns["tet_intersection_volume"]

L=np.load(LEDGER,allow_pickle=False)
R=np.load(INPUT,allow_pickle=False)
C=np.load(CHILD,allow_pickle=False)
with NODES.open() as f:
    node_rows=sorted(csv.DictReader(f),key=lambda r:int(r["global_node"]))
N=len(node_rows)
Xnative=np.asarray([[float(r[k]) for k in ("rest_x","rest_y","rest_z")] for r in node_rows],np.float32)
Xcur_native=np.asarray([[float(r[k]) for k in ("position_x","position_y","position_z")] for r in node_rows],np.float32)
Tnative=np.asarray(L["source_node_rows"],np.int64)
Pnative=np.asarray(L["parent_mass_kg"],np.float64)
Vnative=np.asarray(L["row0w_reference_volume_m3"],np.float32).astype(np.float64)
rhonative=np.asarray(L["bulk_x_density_kg_m3"],np.float32).astype(np.float64)
Mnative=np.asarray(L["identity_material_x"],np.int32)
global_tet=np.asarray(L["global_tet_rows"],np.int64)
source_tet=np.asarray(L["source_tet_rows"],np.int64)

Tinput=np.asarray(R["tetrahedra"],np.int64)
Min=np.asarray(R["material_ids"],np.int64)
Xinput64=np.asarray(R["reference_points_m"],np.float64)
Xchild32=np.asarray(C["reference_points_f32"],np.float32)
Xchild64=np.asarray(C["canonical_reference_points_f64"],np.float64)
Xcur_child=np.asarray(C["current_points_f32"],np.float32)
Tchild=np.asarray(C["tetrahedra"],np.int64)
Mchild=np.asarray(C["material_ids"],np.int64)
parent_input=np.asarray(C["child_parent_rows"],np.int64)
parent_input_ids=np.asarray(C["child_parent_cell_ids"],np.int64)
frac64=np.asarray(C["child_reference_volume_fraction_of_input_parent_f64"],np.float64)
assert len(Tnative)==len(Tinput)==28632 and len(Tchild)==37935 and N==7233
assert np.array_equal(parent_input,parent_input_ids)
assert np.array_equal(Xinput64.astype(np.float32),Xnative)
assert np.array_equal(Xchild32[:N].view(np.uint32),Xnative.view(np.uint32))
assert np.array_equal(Xcur_child[:N].view(np.uint32),Xcur_native.view(np.uint32))
assert np.array_equal(source_tet,np.arange(len(Tnative)))
assert np.all(np.isin(Mnative,[15,16])) and np.all(np.isin(Min,[1,2]))
assert np.array_equal(Mchild,Min[parent_input])

# Exact node-set ancestry from refinement-input cells to actual cooked parents.
native_by_set=collections.defaultdict(list)
input_by_set=collections.defaultdict(list)
for i,t in enumerate(Tnative): native_by_set[key(t)].append(i)
for i,t in enumerate(Tinput): input_by_set[key(t)].append(i)
input_to_native=np.full(len(Tinput),-1,np.int64)
native_to_input=np.full(len(Tnative),-1,np.int64)
for k,rs in input_by_set.items():
    for ir,nr in zip(rs,native_by_set.get(k,[])):
        input_to_native[ir]=nr
        native_to_input[nr]=ir
changed_input=np.flatnonzero(input_to_native<0)
changed_native=np.flatnonzero(native_to_input<0)
assert len(changed_input)==len(changed_native)==14
assert np.array_equal(Min[input_to_native>=0],
                      np.where(Mnative[input_to_native[input_to_native>=0]]==15,1,2))
# Parent retessellation has same oriented cavity boundary by exact shared original node IDs.
boundary_report={}
for inm,nm in ((1,15),(2,16)):
    ir=changed_input[Min[changed_input]==inm]
    nr=changed_native[Mnative[changed_native]==nm]
    bo=boundary(Xnative,Tnative,nr)
    bi=boundary(Xnative,Tinput,ir)
    boundary_report[str(inm)]={"native_material_id":nm,"native_parent_rows":nr.tolist(),
        "input_parent_rows":ir.tolist(),"native_boundary_faces":len(bo),"input_boundary_faces":len(bi),
        "face_set_equal":set(bo)==set(bi),"oriented_boundary_equal":bo==bi}
    assert bo==bi

is_changed_child=np.isin(parent_input,changed_input)
assert int(is_changed_child.sum())==92
# Canonical-F64 child/parent fractions are used only for exact same-parent refinement.
fraction_sum_input=np.bincount(parent_input,weights=frac64,minlength=len(Tinput))
assert np.all(frac64>0)
assert float(np.max(np.abs(fraction_sum_input-1)))<2e-10

# Actual emitted compiler cell measures: FP32 node positions, F64 determinant, then F32 volume storage.
Vchild_geom32=np.abs(tetvol(Xchild32,Tchild))
Vchild_cooked=Vchild_geom32.astype(np.float32).astype(np.float64)
Vchild_canonical=np.abs(tetvol(Xchild64,Tchild))
Vnative_geom=np.abs(tetvol(Xnative,Tnative))

# Each child of an unchanged AP input cell uses its own direct parent fraction.
# Children descended from any of the 14 changed AP cells are intersected against
# all actual native parent tets; do not compose through the AP retessellation.
cr=[];cc=[];gvol=[];mvol=[];kind=[];cross=[];candidate_pairs=0;dropped_small=0
Xtet=Xnative[Tnative].astype(np.float64)
oldlo=Xtet.min(axis=1); oldhi=Xtet.max(axis=1)
for ci in range(len(Tchild)):
    ap=int(parent_input[ci])
    if input_to_native[ap]>=0:
        pi=int(input_to_native[ap])
        f=float(frac64[ci])
        cr.append(ci);cc.append(pi)
        gvol.append(float(Vchild_canonical[ci]))
        mvol.append(float(f*Vnative[pi]))
        kind.append(0)
        if Mchild[ci] != (1 if Mnative[pi]==15 else 2):
            raise RuntimeError("direct ancestry changed material")
        continue
    A=Xchild32[Tchild[ci]].astype(np.float64)
    lo=A.min(axis=0); hi=A.max(axis=0)
    cand=np.flatnonzero(np.all(oldhi>=lo-1e-15,axis=1)&np.all(oldlo<=hi+1e-15,axis=1))
    any_hit=False
    for pi in map(int,cand):
        candidate_pairs+=1
        z=float(intersect(A,Xtet[pi]))
        threshold=max(1e-27,min(Vchild_geom32[ci],Vnative_geom[pi])*1e-14)
        if z<=threshold:
            if z>0: dropped_small+=1
            continue
        any_hit=True
        cr.append(ci);cc.append(pi);gvol.append(z);mvol.append(z);kind.append(1)
        if Mnative[pi] != (15 if Mchild[ci]==1 else 16):
            cross.append((ci,pi,z))
    if not any_hit:
        raise RuntimeError(f"changed child has no positive native overlap: {ci}")
assert not cross, f"cross-material overlaps: {cross[:5]}"
cr=np.asarray(cr,np.int64);cc=np.asarray(cc,np.int64)
gvol=np.asarray(gvol,np.float64);mvol=np.asarray(mvol,np.float64);kind=np.asarray(kind,np.uint8)
order=np.lexsort((cc,cr));cr,cc,gvol,mvol,kind=cr[order],cc[order],gvol[order],mvol[order],kind[order]
offsets=np.r_[0,np.cumsum(np.bincount(cr,minlength=len(Tchild)))]
parent_mass_fraction=mvol/Vnative[cc]
parent_fraction_sum=np.bincount(cc,weights=parent_mass_fraction,minlength=len(Tnative))
child_mass=np.bincount(cr,weights=Pnative[cc]*parent_mass_fraction,minlength=len(Tchild))
child_density=child_mass/Vchild_cooked

# Quality and volume diagnostics.
quality={}
for name,X,T in (("native_reference_f32",Xnative,Tnative),
                 ("refined_reference_f32",Xchild32,Tchild),
                 ("refined_current_f32",Xcur_child,Tchild),
                 ("refined_canonical_reference_f64",Xchild64,Tchild)):
    vol=tetvol(X,T); q=qmetric(X,T)
    quality[name]={"cell_count":len(T),"nonpositive_count":int(np.count_nonzero(vol<=0)),
        "minimum_signed_volume_m3":float(vol.min()),"minimum_q":float(q.min()),
        "q_below_1e-5":int(np.count_nonzero(q<1e-5))}
density_report={}
for im,nm,bounds in ((1,15,(900.,1100.)),(2,16,(850.,950.))):
    rows=np.flatnonzero(Mchild==im); parents=np.flatnonzero(Mnative==nm)
    d=child_density[rows]
    density_report[str(im)]={"native_material_id":nm,"child_count":len(rows),
        "density_min_kg_m3":float(d.min()),"density_max_kg_m3":float(d.max()),
        "configured_bounds_kg_m3":list(bounds),"within_bounds":bool(np.all((d>=bounds[0])&(d<=bounds[1]))),
        "parent_mass_kg":float(Pnative[parents].sum()),"child_mass_kg":float(child_mass[rows].sum()),
        "mass_delta_kg":float(child_mass[rows].sum()-Pnative[parents].sum()),
        "parent_compiled_volume_m3":float(Vnative[parents].sum()),
        "child_compiled_volume_m3":float(Vchild_cooked[rows].sum())}

# Geometric overlap coverage applies to changed descendants only. Direct children
# are certified by their exact same-parent fraction ancestry, not by a fake overlap.
changed_children=np.flatnonzero(is_changed_child)
local_child_sum=np.bincount(cr[kind==1],weights=gvol[kind==1],minlength=len(Tchild))[changed_children]
local_child_cov=local_child_sum/Vchild_geom32[changed_children]
local_parent_rows=changed_native
local_parent_sum=np.bincount(cc[kind==1],weights=gvol[kind==1],minlength=len(Tnative))[local_parent_rows]
local_parent_cov=local_parent_sum/Vnative_geom[local_parent_rows]
# Canonical raw volume closure and emitted-F32 compiled volume drift remain distinct.
canon_parent_sum=np.bincount(parent_input,weights=Vchild_canonical,minlength=len(Tinput))
input_v64=np.abs(tetvol(Xinput64,Tinput))
canon_parent_drift=np.abs(canon_parent_sum-input_v64)/np.maximum(input_v64,1e-300)
child_vs_canon=np.abs(Vchild_cooked-Vchild_canonical)/np.maximum(Vchild_canonical,1e-300)
mass_delta=float(child_mass.sum()-Pnative.sum())

OUT.mkdir(exist_ok=True)
csr=OUT/"refined-child-native-parent-csr.npz"
np.savez_compressed(csr,child_row_offsets=offsets,parent_native_local_rows=cc,
    parent_global_tet_rows=global_tet[cc],parent_source_tet_rows=source_tet[cc],
    parent_mass_fraction=parent_mass_fraction,parent_mass_measure_volume_m3=mvol,
    geometric_overlap_or_ancestry_volume_m3=gvol,mapping_entry_kind=kind,
    child_material_ids=Mchild,child_compiled_reference_volume_m3=Vchild_cooked,
    child_canonical_reference_volume_m3=Vchild_canonical,child_mass_kg=child_mass,
    child_density_kg_m3=child_density,child_parent_input_rows=parent_input,
    changed_input_parent_rows=changed_input,changed_native_parent_rows=changed_native)
report={"schema":"numi.refined-child-to-native-parent-mass-transfer-070.v1",
 "status":"bounded_direct_parent_transfer_audit",
 "pins":{"native_parent_ledger_sha256":sha(LEDGER),"run006_candidate_sha256":sha(CHILD),
   "run006_manifest_sha256":sha(RUN/"run-manifest.json"),"repair_input_sha256":sha(INPUT),
   "native_nodes_csv_sha256":sha(NODES),"overlap_helper_sha256":sha(HELP),
   "transfer_csr_sha256":sha(csr)},
 "domains":{"active_native_parent_count":len(Tnative),"refined_child_count":len(Tchild),
   "native_source_tet_rows_identity":bool(np.array_equal(source_tet,np.arange(len(Tnative)))),
   "native_parent_global_tet_range":[int(global_tet.min()),int(global_tet.max())],
   "input_reference_f32_matches_cooked_native":bool(np.array_equal(Xinput64.astype(np.float32),Xnative)),
   "refined_original_node_prefix_f32_matches_native":bool(np.array_equal(Xchild32[:N].view(np.uint32),Xnative.view(np.uint32))),
   "refined_current_original_node_prefix_matches_native":bool(np.array_equal(Xcur_child[:N].view(np.uint32),Xcur_native.view(np.uint32))),
   "child_parent_rows_match_cell_ids":bool(np.array_equal(parent_input,parent_input_ids)),
   "child_materials_inherit_input_parent":bool(np.array_equal(Mchild,Min[parent_input]))},
 "ancestry":{"unchanged_input_parent_count":int(np.count_nonzero(input_to_native>=0)),
   "retriangulated_input_parent_count":int(len(changed_input)),
   "retriangulated_native_parent_count":int(len(changed_native)),
   "changed_input_parent_rows":changed_input.tolist(),"changed_native_parent_rows":changed_native.tolist(),
   "descendant_children_requiring_actual_overlap":int(is_changed_child.sum()),
   "same_parent_fraction_max_closure_error":float(np.max(np.abs(fraction_sum_input-1))),
   "same_parent_fraction_source":"run006 child_reference_volume_fraction_of_input_parent_f64, only where canonical AP parent exact node set maps to the active native parent",
   "retriangulated_parent_mapping":"actual F32 child tetrahedra intersected against actual F32 native parent tetrahedra; no mixture composition through the AP parent",
   "changed_cavity_boundaries_by_material":boundary_report},
 "quality":quality,
 "changed_overlap":{"candidate_pairs":candidate_pairs,"stored_overlap_entries":int(np.count_nonzero(kind==1)),
   "subthreshold_positive_intersection_candidates":dropped_small,"cross_material_overlap_entries":len(cross),
   "changed_child_geometric_coverage_min_max":[float(local_child_cov.min()),float(local_child_cov.max())],
   "changed_child_max_coverage_error":float(np.max(np.abs(local_child_cov-1))),
   "changed_native_parent_geometric_coverage_min_max":[float(local_parent_cov.min()),float(local_parent_cov.max())],
   "changed_native_parent_max_coverage_error":float(np.max(np.abs(local_parent_cov-1))),
   "clipper":"Retained Float64 construction clipper on emitted/native F32 reference coordinates; tolerance-based, not exact rational predicates"},
 "volume_diagnostics":{"canonical_child_volume_fraction_closure_max_relative_error":float(canon_parent_drift.max()),
   "maximum_canonical_child_vs_compiler_f32_volume_relative_difference":float(child_vs_canon.max()),
   "canonical_f64_refinement_input_volume_drift_limit":1e-4,
   "emitted_f32_parent_volume_drift_is_not_normalized":True},
 "mass_transfer":{"total_parent_mass_kg":float(Pnative.sum()),"total_child_mass_kg":float(child_mass.sum()),
   "total_mass_delta_kg":mass_delta,"total_mass_delta_relative":mass_delta/float(Pnative.sum()),
   "parent_mass_fraction_sum_min_max":[float(parent_fraction_sum.min()),float(parent_fraction_sum.max())],
   "maximum_parent_mass_fraction_closure_error":float(np.max(np.abs(parent_fraction_sum-1))),
   "no_fraction_normalization":True,"no_density_rescaling":True,
   "per_material":density_report,
   "child_density_formula":"child mass = sum(actual native compiled parent mass * parent mass fraction); child density = child mass / compiler inverseRestRow0.w"},
 "artifacts":{"csr":str(csr),"report":str(OUT/"transfer-report.json"),"ledger":str(LEDGER)},
 "limitations":["Native cooked tet identity provides active source_tet_row and exact node tuple, but no anatomical donor-region ID.",
   "Changed-cavity overlap is a bounded floating construction, not an exact-predicate proof.",
   "This transfer does not claim bone clearance, native emission, or physical admission."]}
report_path=OUT/"transfer-report.json"
report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
report["pins"]["report_sha256"]=sha(report_path)
report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
manifest={"builder_sha256":sha(pathlib.Path(__file__)),"transfer_csr_sha256":sha(csr),
 "report_sha256":sha(report_path),"pins":report["pins"]}
(OUT/"run-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2))

