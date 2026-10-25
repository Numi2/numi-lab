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
DIAG=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068/mass-retriangulation-070/diagnostic-all-overlaps"
OUT=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068/mass-retriangulation-070/canonical-native-material-reference"
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
            f,op=u[0];p=np.asarray(X)[np.asarray(f)].astype(np.float64)
            if np.dot(np.cross(p[1]-p[0],p[2]-p[0]),np.asarray(X)[op].astype(np.float64)-p[0])>0:
                f=[f[0],f[2],f[1]]
            out[k]=min(tuple(f[i:]+f[:i]) for i in range(3))
    return out
tree=ast.parse(HELP.read_text())
ns={"np":np,"collections":collections,"ConvexHull":ConvexHull,"QhullError":QhullError}
func=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="tet_intersection_volume"]
exec(compile(ast.Module(body=func,type_ignores=[]),str(HELP),"exec"),ns)
intersect=ns["tet_intersection_volume"]

L=np.load(LEDGER,allow_pickle=False);R=np.load(INPUT,allow_pickle=False);C=np.load(CHILD,allow_pickle=False)
with NODES.open() as f: node_rows=sorted(csv.DictReader(f),key=lambda r:int(r["global_node"]))
N=len(node_rows)
Xnative=np.asarray([[float(r[k]) for k in ("rest_x","rest_y","rest_z")] for r in node_rows],np.float32)
Xcurrent_native=np.asarray([[float(r[k]) for k in ("position_x","position_y","position_z")] for r in node_rows],np.float32)
Tnative=np.asarray(L["source_node_rows"],np.int64)
Pnative=np.asarray(L["parent_mass_kg"],np.float64)
Vcompiled=np.asarray(L["row0w_reference_volume_m3"],np.float32).astype(np.float64)
Mnative=np.asarray(L["identity_material_x"],np.int32)
global_tet=np.asarray(L["global_tet_rows"],np.int64)
source_tet=np.asarray(L["source_tet_rows"],np.int64)
gids=np.asarray(L["compiled_node_global_ids"],np.int64)
Tinput=np.asarray(R["tetrahedra"],np.int64);Min=np.asarray(R["material_ids"],np.int64)
Xinput64=np.asarray(R["reference_points_m"],np.float64)
Tchild=np.asarray(C["tetrahedra"],np.int64);Mchild=np.asarray(C["material_ids"],np.int64)
Xemitted=np.asarray(C["reference_points_f32"],np.float32)
Xcurrent=np.asarray(C["current_points_f32"],np.float32)
Xcanonical=np.asarray(C["canonical_reference_points_f64"],np.float64)
pinput=np.asarray(C["child_parent_rows"],np.int64)
pinput_ids=np.asarray(C["child_parent_cell_ids"],np.int64)
frac_recorded=np.asarray(C["child_reference_volume_fraction_of_input_parent_f64"],np.float64)
ptr=np.asarray(C["source_vertex_ancestry_indptr"],np.int64)
source_ids=np.asarray(C["source_vertex_ancestry_global_ids"],np.int64)
source_w=np.asarray(C["source_vertex_ancestry_weights"],np.float64)
assert len(Tnative)==len(Tinput)==28632 and len(Tchild)==37935 and N==7233
assert np.array_equal(source_tet,np.arange(len(Tnative))) and np.array_equal(pinput,pinput_ids)
assert np.array_equal(gids,np.asarray(C["global_node_ids_original_prefix"],np.int64))
assert np.array_equal(Xinput64.astype(np.float32),Xnative)
assert np.array_equal(Xemitted[:N].view(np.uint32),Xnative.view(np.uint32))
assert np.array_equal(Xcurrent[:N].view(np.uint32),Xcurrent_native.view(np.uint32))
assert np.all(np.isin(Mnative,[15,16])) and np.all(np.isin(Min,[1,2]))
assert np.array_equal(Mchild,Min[pinput])
assert len(ptr)==len(Xemitted)+1 and ptr[0]==0 and ptr[-1]==len(source_ids)==len(source_w)
assert np.all(np.isfinite(source_w)) and np.all(source_w>=0)
row_sums=np.diff(ptr) # only used for row coverage below
row_sum=np.array([source_w[ptr[i]:ptr[i+1]].sum() for i in range(len(Xemitted))])
assert np.max(np.abs(row_sum-1))<2e-12
# Reconstruct a declared reference material domain from the actual compiled
# original F32 points, promoted exactly to F64, and run-006 source vertex CSR.
gid_to_local={int(g):i for i,g in enumerate(gids)}
Xmaterial=np.zeros((len(Xemitted),3),np.float64)
for i in range(len(Xemitted)):
    a,b=map(int,ptr[i:i+2])
    if a==b: raise RuntimeError(f"node {i} lacks source ancestry")
    for j in range(a,b):
        src=gid_to_local.get(int(source_ids[j]))
        if src is None: raise RuntimeError(f"ancestry ID {source_ids[j]} is not in native node domain")
        Xmaterial[i]+=source_w[j]*Xnative[src].astype(np.float64)
assert np.array_equal(Xmaterial[:N],Xnative.astype(np.float64))
emission_offset=np.linalg.norm(Xmaterial-Xemitted.astype(np.float64),axis=1)
canonical_offset=np.linalg.norm(Xmaterial-Xcanonical,axis=1)
assert np.max(emission_offset[:N])==0.0

# Exact node-set match for unchanged AP parent cells; 14 AP retets require local overlap.
nb=collections.defaultdict(list);ib=collections.defaultdict(list)
for i,t in enumerate(Tnative): nb[key(t)].append(i)
for i,t in enumerate(Tinput): ib[key(t)].append(i)
input_to_native=np.full(len(Tinput),-1,np.int64);native_to_input=np.full(len(Tnative),-1,np.int64)
for k,rs in ib.items():
    for ir,nr in zip(rs,nb.get(k,[])):input_to_native[ir]=nr;native_to_input[nr]=ir
changed_input=np.flatnonzero(input_to_native<0);changed_native=np.flatnonzero(native_to_input<0)
assert len(changed_input)==len(changed_native)==14
assert np.array_equal(Min[input_to_native>=0],np.where(Mnative[input_to_native[input_to_native>=0]]==15,1,2))
changed_child=np.isin(pinput,changed_input);assert int(changed_child.sum())==92
# Material-reference geometry derived from native-F32 source nodes. Validate same
# oriented cavity boundary for the 14 retessellated parent cells.
boundary_report={}
for im,nm in ((1,15),(2,16)):
    ir=changed_input[Min[changed_input]==im];nr=changed_native[Mnative[changed_native]==nm]
    bo=boundary(Xnative,Tnative,nr);bi=boundary(Xmaterial,Tinput,ir)
    boundary_report[str(im)]={"native_material_id":nm,"native_rows":nr.tolist(),"input_rows":ir.tolist(),
        "native_faces":len(bo),"input_faces":len(bi),"oriented_boundary_equal":bo==bi}
    assert bo==bi

Vnative_geo=np.abs(tetvol(Xnative.astype(np.float64),Tnative))
Vinput_geo=np.abs(tetvol(Xmaterial[:N],Tinput))
Vchild_material=np.abs(tetvol(Xmaterial,Tchild))
Vchild_emit=np.abs(tetvol(Xemitted,Tchild))
Vchild_cooked=Vchild_emit.astype(np.float32).astype(np.float64)
Vnative_floor=Vnative_geo
# Use the normalized source-ancestry material domain's raw child/reference
# volume fraction for exact same-parent refinements; validate, never normalize.
frac=np.zeros(len(Tchild),np.float64)
for ci in np.flatnonzero(~changed_child):
    pi=int(input_to_native[pinput[ci]])
    if pi<0: raise RuntimeError("same-parent child has no exact native parent")
    frac[ci]=Vchild_material[ci]/Vnative_geo[pi]
fraction_by_input=np.bincount(pinput[~changed_child],weights=frac[~changed_child],minlength=len(Tinput))
same_input=np.flatnonzero(input_to_native>=0)
assert np.max(np.abs(fraction_by_input[same_input]-1))<2e-10

# All changed input-parent descendants are geometrically clipped against the
# actual native parent tets in this canonical material-reference domain.
xtet=Xnative[Tnative].astype(np.float64);lo0=xtet.min(1);hi0=xtet.max(1)
cr=[];cc=[];gvol=[];mvol=[];kind=[];cross=[];candidate_pairs=0;small=0
for ci in range(len(Tchild)):
    ap=int(pinput[ci])
    if input_to_native[ap]>=0:
        pi=int(input_to_native[ap]);f=float(frac[ci])
        cr.append(ci);cc.append(pi);gvol.append(float(Vchild_material[ci]));mvol.append(f*Vcompiled[pi]);kind.append(0)
        continue
    A=Xmaterial[Tchild[ci]]
    lo=A.min(0);hi=A.max(0)
    cand=np.flatnonzero(np.all(hi0>=lo-1e-15,axis=1)&np.all(lo0<=hi+1e-15,axis=1))
    found=False
    for pi in map(int,cand):
        candidate_pairs+=1
        z=float(intersect(A,xtet[pi]))
        threshold=max(1e-27,min(Vchild_material[ci],Vnative_geo[pi])*1e-14)
        if z<=threshold:
            if z>0:small+=1
            continue
        found=True;cr.append(ci);cc.append(pi);gvol.append(z);mvol.append(z);kind.append(1)
        if Mnative[pi]!=(15 if Mchild[ci]==1 else 16):cross.append((ci,pi,z))
    if not found: raise RuntimeError(f"changed child has no material-reference overlap {ci}")
assert not cross, f"cross-material overlaps in declared material-reference domain: {cross[:5]}"
cr=np.asarray(cr,np.int64);cc=np.asarray(cc,np.int64);gvol=np.asarray(gvol,np.float64);mvol=np.asarray(mvol,np.float64);kind=np.asarray(kind,np.uint8)
order=np.lexsort((cc,cr));cr,cc,gvol,mvol,kind=cr[order],cc[order],gvol[order],mvol[order],kind[order]
offset=np.r_[0,np.cumsum(np.bincount(cr,minlength=len(Tchild)))]
parent_fraction=mvol/Vcompiled[cc]
parent_fraction_sum=np.bincount(cc,weights=parent_fraction,minlength=len(Tnative))
child_mass=np.bincount(cr,weights=Pnative[cc]*parent_fraction,minlength=len(Tchild))
child_density=child_mass/Vchild_cooked
child_geo_coverage=np.bincount(cr,weights=gvol,minlength=len(Tchild))/Vchild_material
parent_geo_coverage=np.bincount(cc,weights=gvol,minlength=len(Tnative))/Vnative_geo
quality={}
for label,X,T in (("native_reference_f32",Xnative,Tnative),("material_reference_f64",Xmaterial,Tchild),
                  ("emitted_reference_f32",Xemitted,Tchild),("current_f32",Xcurrent,Tchild)):
    vol=tetvol(X,T);q=qmetric(X,T)
    quality[label]={"cells":len(T),"nonpositive":int(np.count_nonzero(vol<=0)),
        "min_signed_volume_m3":float(vol.min()),"min_q":float(q.min()),"q_below_1e-5":int(np.count_nonzero(q<1e-5))}
density={}
for im,nm,bounds in ((1,15,(900.,1100.)),(2,16,(850.,950.))):
    rows=np.flatnonzero(Mchild==im);parents=np.flatnonzero(Mnative==nm);d=child_density[rows]
    density[str(im)]={"native_material_id":nm,"children":len(rows),"min_kg_m3":float(d.min()),"max_kg_m3":float(d.max()),
        "bounds_kg_m3":list(bounds),"within_bounds":bool(np.all((d>=bounds[0])&(d<=bounds[1]))),
        "parent_mass_kg":float(Pnative[parents].sum()),"child_mass_kg":float(child_mass[rows].sum()),
        "mass_delta_kg":float(child_mass[rows].sum()-Pnative[parents].sum())}
local_children=np.flatnonzero(changed_child)
local_child_cov=child_geo_coverage[local_children]
local_parents=changed_native
local_parent_cov=parent_geo_coverage[local_parents]
input_fraction_sum=np.bincount(pinput,weights=np.where(changed_child,0.,frac),minlength=len(Tinput))
# Use parent geometry F64 material-domain fractions for unchanged parent ancestry;
# retet descendants use raw overlap fractions.
mass_delta=float(child_mass.sum()-Pnative.sum())
OUT.mkdir(exist_ok=True)
csr=OUT/"refined-child-native-parent-csr.npz"
np.savez_compressed(csr,child_row_offsets=offset,parent_native_local_rows=cc,
    parent_global_tet_rows=global_tet[cc],parent_source_tet_rows=source_tet[cc],
    parent_mass_fraction=parent_fraction,parent_mass_measure_volume_m3=mvol,
    geometric_overlap_or_ancestry_volume_m3=gvol,mapping_entry_kind=kind,
    child_material_ids=Mchild,child_mass_kg=child_mass,child_density_kg_m3=child_density,
    child_material_reference_volume_m3=Vchild_material,child_compiled_reference_volume_m3=Vchild_cooked,
    child_parent_input_rows=pinput,changed_input_parent_rows=changed_input,changed_native_parent_rows=changed_native,
    material_reference_points_f64=Xmaterial,emission_offset_m=emission_offset,
    source_ancestry_offsets=ptr,source_ancestry_global_ids=source_ids,source_ancestry_weights=source_w)
report={"schema":"numi.refined-child-native-parent-material-reference-transfer-070.v1",
 "status":"bounded_material_reference_parent_transfer_audit",
 "pins":{"native_parent_ledger_sha256":sha(LEDGER),"native_nodes_csv_sha256":sha(NODES),
   "refinement_candidate_sha256":sha(CHILD),"refinement_manifest_sha256":sha(RUN/"run-manifest.json"),
   "refinement_input_sha256":sha(INPUT),"overlap_helper_sha256":sha(HELP)},
 "reference_domain":{"definition":"Original native compiled F32 rest coordinates promoted exactly to F64; every refined point reconstructed by run-006 source-vertex CSR weights over those original global node IDs.",
   "source_csr_row_sums_max_error":float(np.max(np.abs(row_sum-1))),
   "original_prefix_exact":bool(np.array_equal(Xmaterial[:N],Xnative.astype(np.float64))),
   "max_new_node_emission_offset_m":float(emission_offset[N:].max()),
   "p50_new_node_emission_offset_m":float(np.quantile(emission_offset[N:],.5)),
   "p95_new_node_emission_offset_m":float(np.quantile(emission_offset[N:],.95)),
   "max_material_vs_run_canonical_f64_offset_m":float(canonical_offset.max()),
   "f32_emission_domain_is_separate":True},
 "ancestry":{"native_parent_count":len(Tnative),"refined_child_count":len(Tchild),
   "same_node_set_input_parent_count":int(np.count_nonzero(input_to_native>=0)),
   "retriangulated_input_and_native_parent_count":14,"changed_input_rows":changed_input.tolist(),
   "changed_native_rows":changed_native.tolist(),"changed_parent_descendant_child_count":int(changed_child.sum()),
   "unchanged_parent_fraction_closure_max_error":float(np.max(np.abs(fraction_by_input[same_input]-1))),
   "changed_cavity_oriented_boundary_by_material":boundary_report,
   "direct_method":"raw canonical material-reference child volume / exact matched native parent material-reference volume",
   "retet_method":"actual local tetra intersection between descendants of the 14 retessellated input parents and actual active native parent cells; no composition through an AP parent mixture"},
 "changed_overlap":{"candidate_pairs":candidate_pairs,"stored_intersection_entries":int(np.count_nonzero(kind==1)),
   "subthreshold_positive_intersection_candidates":small,"cross_material_entries":len(cross),
   "changed_child_coverage_min_max":[float(local_child_cov.min()),float(local_child_cov.max())],
   "changed_child_max_abs_coverage_error":float(np.max(np.abs(local_child_cov-1))),
   "changed_parent_coverage_min_max":[float(local_parent_cov.min()),float(local_parent_cov.max())],
   "changed_parent_max_abs_coverage_error":float(np.max(np.abs(local_parent_cov-1))),
   "clipper":"Retained Float64 clipping construction on dyadic material-reference coordinates, with AABB pruning; not an exact-rational predicate"},
 "quality":quality,
 "volume":{"native_compiled_row0w_sum_m3":float(Vcompiled.sum()),
   "native_material_reference_geometric_volume_m3":float(Vnative_geo.sum()),
   "refined_material_reference_geometric_volume_m3":float(Vchild_material.sum()),
   "refined_compiled_emitted_volume_m3":float(Vchild_cooked.sum()),
   "material_reference_total_volume_relative_drift":float((Vchild_material.sum()-Vnative_geo.sum())/Vnative_geo.sum()),
   "emitted_compiled_total_volume_relative_drift":float((Vchild_cooked.sum()-Vcompiled.sum())/Vcompiled.sum()),
   "max_compiled_emitted_vs_material_child_volume_relative_difference":float(np.max(np.abs(Vchild_cooked-Vchild_material)/Vchild_material))},
 "mass":{"parent_total_kg":float(Pnative.sum()),"child_total_kg":float(child_mass.sum()),"delta_kg":mass_delta,
   "delta_relative":mass_delta/float(Pnative.sum()),"parent_fraction_sum_min_max":[float(parent_fraction_sum.min()),float(parent_fraction_sum.max())],
   "max_parent_fraction_closure_error":float(np.max(np.abs(parent_fraction_sum-1))),
   "no_fraction_normalization":True,"no_density_rescaling":True,"per_material":density,
   "density_formula":"child mass=sum(actual native parent mass * raw parent mass fraction); child density=child mass / emitted compiler row0.w"},
 "artifacts":{"csr":str(csr),"report":str(OUT/"transfer-report.json"),"ledger":str(LEDGER)},
 "limitations":["This is a declared material-reference pullback ledger; its node coordinates are reconstructed from original compiled F32 source nodes and ancestry weights, not the emitted refined F32 midpoint positions.",
   "Maximum new-node emission offset is separately recorded; emitted F32 geometry/current quality and physical boundary gates remain independent.",
   "Cooked identity supplies active source_tet_row and node tuple but no anatomical donor-region ID.",
   "No native build, bone-clearance pass, or physical admission is claimed."]}
rp=OUT/"transfer-report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
report["pins"]={"native_parent_ledger_sha256":sha(LEDGER),"native_nodes_csv_sha256":sha(NODES),
   "refinement_candidate_sha256":sha(CHILD),"refinement_manifest_sha256":sha(RUN/"run-manifest.json"),
   "refinement_input_sha256":sha(INPUT),"overlap_helper_sha256":sha(HELP),"transfer_csr_sha256":sha(csr)}
rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
report["pins"]["report_sha256"]=sha(rp);rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
(OUT/"run-manifest.json").write_text(json.dumps({"builder_sha256":sha(pathlib.Path(__file__)),"transfer_csr_sha256":sha(csr),"report_sha256":sha(rp),"pins":report["pins"]},indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2))

