#!/usr/bin/env python3
"""Build candidate-only conservative native-parent mass ancestry for refinement 037 run-006.
Uses exact active native cell rows from the frozen 039 source-only compiler export.
Changed local retet rows are resolved by tetrahedron intersection in native-F32-promoted-F64 material-reference coordinates.
"""
from pathlib import Path
import csv, hashlib, itertools, json, time
import numpy as np
from scipy.spatial import ConvexHull, QhullError

E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification/20261022/wrist-connected-repair-001")
# Correct spelling pins the root path explicitly; fail before writing if it is absent.
if not E.exists():
    E = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D = E / "connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight"
BASE = E / "route-audit-counter-fix-030/baseline-035.npz"
CAND = E / "E/connected-volume-refinement-037/run-006/candidate.npz"
NODES = D / "compiled-skin-fem-nodes.csv"
TETS = D / "compiled-skin-fem-tets.csv"
CONTRIB = D / "premerge-layer-contributions.csv"
OUT = E / "connected-skin-parent-mass-transfer-067/native-parent-transfer-073"
OUT.mkdir(parents=True, exist_ok=True)
SCRIPT = Path(__file__)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def tet_volume(v):
    return abs(float(np.linalg.det(v[1:] - v[0]))) / 6.0

def halfspaces(v):
    ns, bs = [], []
    for k in range(4):
        ids = [j for j in range(4) if j != k]
        p0, p1, p2 = v[ids]
        n = np.cross(p1-p0, p2-p0)
        ln = np.linalg.norm(n)
        if not np.isfinite(ln) or ln == 0:
            return None, None
        n /= ln
        if np.dot(n, v[k]-p0) > 0:
            n = -n
        ns.append(n)
        bs.append(np.dot(n,p0))
    return np.asarray(ns), np.asarray(bs)

def tetra_intersection_volume(a,b,inside_tol=2e-10,unique_tol=2e-10):
    na,ba=halfspaces(a); nb,bb=halfspaces(b)
    if na is None or nb is None:
        return 0.0
    N=np.vstack((na,nb)); h=np.r_[ba,bb]
    pts=[]
    for ix in itertools.combinations(range(8),3):
        M=N[list(ix)]
        if abs(float(np.linalg.det(M))) < 1e-10:
            continue
        x=np.linalg.solve(M,h[list(ix)])
        if np.max(N@x-h) <= inside_tol:
            if not any(np.max(np.abs(x-y)) < unique_tol for y in pts):
                pts.append(x)
    if len(pts)<4:
        return 0.0
    try:
        return float(ConvexHull(np.asarray(pts)).volume)
    except QhullError:
        return 0.0

def arr_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

t0=time.time()
base=np.load(BASE,allow_pickle=False)
cand=np.load(CAND,allow_pickle=False)
node_rows=arr_csv(NODES)
tet_rows=[x for x in arr_csv(TETS) if int(x["identity_active_bit"])==1]
contrib=arr_csv(CONTRIB)
if len(node_rows)!=7233 or len(tet_rows)!=28632:
    raise RuntimeError(f"unexpected frozen native dimensions nodes={len(node_rows)} active_tets={len(tet_rows)}")
native_global=np.asarray([int(x["global_node"]) for x in node_rows],dtype=np.int64)
native_local=np.asarray([int(x["object_local_node"]) for x in node_rows],dtype=np.int64)
if len(set(native_global.tolist()))!=len(native_global) or set(native_local.tolist())!=set(range(len(node_rows))):
    raise RuntimeError("native compiled node identity domain is not unique dense object-local rows")
global_to_local={int(g):int(i) for g,i in zip(native_global,native_local)}
native_rest=np.asarray([[float(x["rest_x"]),float(x["rest_y"]),float(x["rest_z"])] for x in node_rows],dtype=np.float64)
old_t=np.asarray([[int(x[f"source_node{j}"]) for j in range(4)] for x in tet_rows],dtype=np.int64)
old_mat=np.asarray([int(x["material_index"]) for x in tet_rows],dtype=np.int64)
old_source_row=np.asarray([int(x["source_tet_row"]) for x in tet_rows],dtype=np.int64)
old_cell_volume_compiled=np.asarray([float(x["inverse_rest_w_m3"]) for x in tet_rows],dtype=np.float64)
old_density=np.asarray([float(x["compiled_bulk_x"]) for x in tet_rows],dtype=np.float64)
old_corner_mass=np.asarray([float(x["bulk_density_corner_mass_kg"]) for x in tet_rows],dtype=np.float64)
old_cell_mass=old_cell_volume_compiled*old_density
if not np.array_equal(old_source_row,np.arange(len(tet_rows))):
    raise RuntimeError("active native source_tet_row is not the dense 0..N-1 parent ID")
if not np.allclose(old_corner_mass,old_cell_mass/4,rtol=0,atol=0):
    raise RuntimeError("native exported per-corner cell-mass identity failed")

# Source ancestry reconstructs every output vertex in the retained native-F32 material reference domain.
ptr=cand["source_vertex_ancestry_indptr"].astype(np.int64)
ag=cand["source_vertex_ancestry_global_ids"].astype(np.int64)
aw=cand["source_vertex_ancestry_weights"].astype(np.float64)
if len(ptr)!=len(cand["tetrahedra"])*0+len(cand["current_points_f32"])+1:
    raise RuntimeError("source ancestry row count does not match candidate point count")
ref_points=np.empty((len(ptr)-1,3),dtype=np.float64)
ancestry_sum_err=0.0
for i in range(len(ref_points)):
    a,z=int(ptr[i]),int(ptr[i+1])
    if a==z:
        raise RuntimeError(f"candidate node {i} has empty source ancestry")
    ids=ag[a:z]
    if any(int(g) not in global_to_local for g in ids):
        raise RuntimeError(f"candidate node {i} ancestry references an absent native source node")
    w=aw[a:z]
    ancestry_sum_err=max(ancestry_sum_err,abs(float(w.sum())-1.0))
    loc=np.asarray([global_to_local[int(g)] for g in ids],dtype=np.int64)
    ref_points[i]=(native_rest[loc]*w[:,None]).sum(axis=0)
if ancestry_sum_err>1e-10:
    raise RuntimeError(f"candidate ancestry weights do not sum to one: {ancestry_sum_err}")
candidate_ref_diff=np.max(np.abs(ref_points-cand["canonical_reference_points_f64"].astype(np.float64)),axis=1)
base_global=base["global_node_ids"].astype(np.int64)
if not np.array_equal(cand["global_node_ids_original_prefix"].astype(np.int64),base_global):
    raise RuntimeError("candidate original global-node prefix differs from baseline 035")
if not np.array_equal(base_global,native_global):
    # compare ordered source identity; this must be explicit, not silently remapped
    raise RuntimeError("baseline global node IDs differ from actual native compiled global node IDs")

# Native geometry and authoritative compiled cell mass use different recorded quantities.
old_xyz=native_rest[old_t]
old_geom_volume=np.asarray([tet_volume(v) for v in old_xyz])
old_native_volume_rel=np.abs(old_geom_volume-old_cell_volume_compiled)/old_cell_volume_compiled
new_t=cand["tetrahedra"].astype(np.int64)
new_mat=cand["material_ids"].astype(np.int64)
child_base_parent=cand["child_parent_rows"].astype(np.int64)
child_ref_xyz=ref_points[new_t]
child_geom_volume=np.asarray([tet_volume(v) for v in child_ref_xyz])
if np.any(~np.isfinite(child_geom_volume)) or np.any(child_geom_volume<=0):
    raise RuntimeError("candidate child native-reference geometry has nonpositive/nonfinite cells")
base_t=base["tetrahedra"].astype(np.int64)
base_mat=base["material_ids"].astype(np.int64)
base_to_native_mat={1:15,2:16}
geometric_changed=[]
orientation_only=[]
for i in range(len(tet_rows)):
    if old_mat[i] != base_to_native_mat[int(base_mat[i])]:
        geometric_changed.append(i)
    elif set(old_t[i].tolist()) != set(base_t[i].tolist()):
        geometric_changed.append(i)
    elif tuple(old_t[i].tolist()) != tuple(base_t[i].tolist()):
        orientation_only.append(i)
if len(geometric_changed)!=14 or len(orientation_only)!=2:
    raise RuntimeError(f"native-vs-baseline topology comparison changed unexpectedly: geometric={len(geometric_changed)} orientation={len(orientation_only)}")

# Identity/refinement parents are valid for identical tetrahedra. Only the 14 retet source rows need local overlap.
child_ref_parent_volume=np.zeros(len(new_t),dtype=np.float64)
child_overlap_lists=[None]*len(new_t)
changed_set=set(geometric_changed)
for ci,p in enumerate(child_base_parent):
    if int(p) not in changed_set:
        if new_mat[ci] != base_mat[p] or old_mat[p] != base_to_native_mat[int(new_mat[ci])]:
            raise RuntimeError(f"identity source/material mismatch child={ci} baseline_parent={p}")
        child_overlap_lists[ci]=[(int(p),float(child_geom_volume[ci]))]
        child_ref_parent_volume[ci]=child_geom_volume[ci]

# Index actual native tet AABBs once. The batch is 92 descendants, not a global all-pairs scan.
old_min=old_xyz.min(axis=1); old_max=old_xyz.max(axis=1)
child_changed_rows=np.flatnonzero(np.isin(child_base_parent,geometric_changed))
local_pairs=0; local_nonzero=0; tclip=time.time()
for ci in child_changed_rows:
    vnew=child_ref_xyz[ci]
    lo=vnew.min(axis=0); hi=vnew.max(axis=0)
    candidates=np.flatnonzero(np.all(old_max>=lo-2e-10,axis=1)&np.all(old_min<=hi+2e-10,axis=1))
    hits=[]
    for oi in candidates:
        local_pairs+=1
        ov=tetra_intersection_volume(vnew,old_xyz[oi])
        if ov>max(min(child_geom_volume[ci],old_geom_volume[oi])*1e-13,1e-25):
            hits.append((int(oi),ov))
            local_nonzero+=1
    if not hits:
        raise RuntimeError(f"changed child has no native parent overlap: child={ci} baseline_parent={int(child_base_parent[ci])}")
    child_overlap_lists[int(ci)]=hits
    child_ref_parent_volume[int(ci)]=sum(v for _,v in hits)

# Full child coverage, native parent coverage, materials and source row ancestry.
child_coverage_rel=np.asarray([(sum(v for _,v in row)-child_geom_volume[i])/child_geom_volume[i] for i,row in enumerate(child_overlap_lists)])
if np.max(np.abs(child_coverage_rel))>1e-7:
    bad=np.flatnonzero(np.abs(child_coverage_rel)>1e-7)
    raise RuntimeError(f"child overlap coverage failed {len(bad)} rows; worst={float(np.max(np.abs(child_coverage_rel)))}")
parent_covered=np.zeros(len(tet_rows),dtype=np.float64)
cross_material=[]
for ci,row in enumerate(child_overlap_lists):
    child_native_mat=base_to_native_mat[int(new_mat[ci])]
    for oi,ov in row:
        if old_mat[oi] != child_native_mat:
            cross_material.append((int(ci),int(oi),float(ov),child_native_mat,int(old_mat[oi])))
        parent_covered[oi]+=ov
parent_cover_rel=(parent_covered-old_geom_volume)/old_geom_volume
if np.max(np.abs(parent_cover_rel))>1e-7:
    bad=np.flatnonzero(np.abs(parent_cover_rel)>1e-7)
    raise RuntimeError(f"native parent coverage failed {len(bad)} rows; worst={float(np.max(np.abs(parent_cover_rel)))}")
if cross_material:
    raise RuntimeError(f"candidate/native overlap crosses material boundary in {len(cross_material)} pairs")

# Build sparse child->native-parent CSR with both row- and column-normalized measures.
offsets=[0]; parent_rows=[]; overlap_v=[]; child_parent_frac=[]; parent_mass_frac=[]
for ci,row in enumerate(child_overlap_lists):
    for oi,ov in sorted(row,key=lambda x:x[0]):
        parent_rows.append(int(oi)); overlap_v.append(float(ov))
        child_parent_frac.append(float(ov/child_geom_volume[ci]))
        parent_mass_frac.append(float(ov/old_geom_volume[oi]))
    offsets.append(len(parent_rows))
parent_rows=np.asarray(parent_rows,dtype=np.int32)
overlap_v=np.asarray(overlap_v,dtype=np.float64)
child_parent_frac=np.asarray(child_parent_frac,dtype=np.float64)
parent_mass_frac=np.asarray(parent_mass_frac,dtype=np.float64)
child_mass=np.zeros(len(new_t),dtype=np.float64)
for ci in range(len(new_t)):
    a,z=offsets[ci],offsets[ci+1]
    child_mass[ci]=np.dot(old_cell_mass[parent_rows[a:z]],parent_mass_frac[a:z])
child_effective_density=child_mass/child_geom_volume
density_bounds={15:(900.,1100.),16:(850.,950.)}
density_violations=[]
for m,(lo,hi) in density_bounds.items():
    ix=np.flatnonzero(np.array([base_to_native_mat[int(x)] for x in new_mat])==m)
    bad=ix[(child_effective_density[ix]<lo)|(child_effective_density[ix]>hi)]
    density_violations.extend((int(i),int(m),float(child_effective_density[i])) for i in bad)
# Parent mass conservation by native row and material.
parent_mass_received=np.zeros(len(tet_rows),dtype=np.float64)
np.add.at(parent_mass_received,parent_rows,old_cell_mass[parent_rows]*parent_mass_frac)
parent_mass_rel=(parent_mass_received-old_cell_mass)/old_cell_mass
material_totals={}
for m in (15,16):
    oldsum=float(old_cell_mass[old_mat==m].sum())
    newsum=float(child_mass[np.array([base_to_native_mat[int(x)] for x in new_mat])==m].sum())
    material_totals[str(m)]={"parent_mass_kg":oldsum,"child_mass_kg":newsum,"relative_difference":(newsum-oldsum)/oldsum}

# Native corner provenance: one matching layer contribution row per material corner, identity/local donor info only.
contrib_by_key={}
for i,x in enumerate(contrib):
    key=(int(x["combined_source_global_node_index"]),int(x["source_material_index"]))
    if key in contrib_by_key:
        raise RuntimeError(f"duplicate source contribution identity for {key}")
    contrib_by_key[key]=i
parent_corner_contrib=np.empty((len(tet_rows),4),dtype=np.int32)
parent_corner_body=np.empty((len(tet_rows),4),dtype=np.int32)
parent_corner_local=np.empty((len(tet_rows),4,3),dtype=np.float64)
for oi,t in enumerate(tet_rows):
    mat=int(t["material_index"])
    for j in range(4):
        gid=int(t[f"global_node{j}"])
        key=(gid,mat)
        if key not in contrib_by_key:
            raise RuntimeError(f"missing source donor/frame contribution for native parent corner {oi}/{j}/{key}")
        cr=contrib_by_key[key]; x=contrib[cr]
        parent_corner_contrib[oi,j]=cr
        parent_corner_body[oi,j]=int(x["node_frame_owner_body"])
        parent_corner_local[oi,j]=[float(x[f"donor_local_{k}_m"]) for k in ("x","y","z")]
parent_corner_cell_mass=np.repeat((old_cell_mass/4)[:,None],4,axis=1)

# Save only transfer/ledger arrays; geometry stays pinned in candidate.npz.
map_path=OUT/"native-parent-transfer-073.npz"
np.savez_compressed(map_path,
    child_parent_indptr=np.asarray(offsets,dtype=np.int64),
    child_native_parent_rows=parent_rows,
    overlap_native_reference_volume_m3=overlap_v,
    child_parent_fraction_overlap_over_child_volume=child_parent_frac,
    parent_mass_fraction_overlap_over_native_parent_geometry_volume=parent_mass_frac,
    child_parent_rows_baseline_035=child_base_parent.astype(np.int32),
    child_material_ids_baseline_12=new_mat.astype(np.int8),
    child_native_material_ids=np.asarray([base_to_native_mat[int(x)] for x in new_mat],dtype=np.int16),
    child_native_reference_volume_m3=child_geom_volume,
    child_transferred_mass_kg=child_mass,
    child_effective_reference_density_kg_m3=child_effective_density,
    native_parent_source_tet_rows=old_source_row.astype(np.int32),
    native_parent_material_ids=old_mat.astype(np.int16),
    native_parent_compiled_reference_volume_m3=old_cell_volume_compiled,
    native_parent_native_f32_geometry_volume_m3=old_geom_volume,
    native_parent_compiled_density_bulk_x=old_density,
    native_parent_compiled_mass_kg=old_cell_mass,
    native_parent_corner_source_contribution_row_ids=parent_corner_contrib,
    native_parent_corner_frame_owner_body_ids=parent_corner_body,
    native_parent_corner_donor_local_points_m=parent_corner_local,
    native_parent_corner_lumped_cell_mass_kg=parent_corner_cell_mass,
    geometric_changed_native_parent_rows=np.asarray(geometric_changed,dtype=np.int32),
    orientation_only_native_parent_rows=np.asarray(orientation_only,dtype=np.int32),
)
input_hashes={str(p.name):sha(p) for p in (BASE,CAND,NODES,TETS,CONTRIB)}
report={
    "status":"CANDIDATE_NATIVE_PARENT_MASS_ANCESTRY_BUILT",
    "admission":{"physical_mass_transfer":False,"native_emission":False,"bone_clearance":False,"native_physics":False},
    "method":"identity/refinement parent measure for 28,618 exact baseline/native source-cell matches; native-F32-promoted-F64 tetra intersection for descendants of the 14 geometrically retetrahedralized rows",
    "candidate_sha256":sha(CAND),"candidate_refinement_manifest_sha256":sha(E/"E/connected-volume-refinement-037/run-006/run-manifest.json"),
    "inputs_sha256":input_hashes,
    "dimensions":{"native_nodes":len(node_rows),"native_active_parents":len(tet_rows),"candidate_children":len(new_t),"candidate_nodes":len(ref_points),"overlap_nonzeros":len(overlap_v),"changed_native_geometric_rows":len(geometric_changed),"orientation_only_rows":len(orientation_only),"descendants_of_changed_baseline_rows":len(child_changed_rows)},
    "native_vs_baseline":{"exact_ordered_source_row_matches":len(tet_rows)-len(orientation_only)-len(geometric_changed),"orientation_only_rows":orientation_only,"geometric_changed_rows":geometric_changed,"actual_source_row_domain":"compiled-skin-fem-tets.csv source_tet_row (dense 0..28631)"},
    "native_reference_domain":{"vertices":"compiled native rest_x/rest_y/rest_z F32 promoted exactly to F64","candidate_new_vertices":"source_vertex_ancestry CSR interpolation of original native global-node IDs","max_native_inverseRest_volume_vs_F64_tet_geometry_relative_error":float(old_native_volume_rel.max()),"max_reconstructed_candidate_reference_vs_candidate_canonical_f64_abs_m":float(candidate_ref_diff.max()),"candidate_reference_f64_retained_as_native_material_domain":False},
    "coverage":{"candidate_child_max_abs_relative_error":float(np.max(np.abs(child_coverage_rel))),"native_parent_max_abs_relative_error":float(np.max(np.abs(parent_cover_rel))),"cross_material_overlap_pairs":len(cross_material),"local_aabb_pairs_tested":local_pairs,"positive_local_overlap_pairs":local_nonzero,"local_intersection_seconds":time.time()-tclip,"all_children_covered":bool(np.max(np.abs(child_coverage_rel))<=1e-7),"all_native_parents_covered":bool(np.max(np.abs(parent_cover_rel))<=1e-7),"material_boundary_preserved_in_reference_measure":not cross_material},
    "mass":{"native_parent_mass_formula":"inverse_rest_w_m3 * compiled_bulk_x; compiled bulk_density_corner_mass_kg equals this value / 4 exactly for all active rows","native_parent_mass_total_kg":float(old_cell_mass.sum()),"candidate_child_mass_total_kg":float(child_mass.sum()),"relative_total_mass_error":float((child_mass.sum()-old_cell_mass.sum())/old_cell_mass.sum()),"per_material":material_totals,"max_abs_parent_mass_closure_relative_error":float(np.max(np.abs(parent_mass_rel))),"effective_child_density_bounds_kg_m3":{"15":[900,1100],"16":[850,950]},"effective_child_density_violations":len(density_violations),"minimum_child_effective_density_by_material":{str(m):float(child_effective_density[np.array([base_to_native_mat[int(x)] for x in new_mat])==m].min()) for m in (15,16)},"maximum_child_effective_density_by_material":{str(m):float(child_effective_density[np.array([base_to_native_mat[int(x)] for x in new_mat])==m].max()) for m in (15,16)},"parent_corner_source_provenance":"one premerge row keyed by (combined_source_global_node_index, native material_index) found for every native parent corner; native cell corner mass is parent cell mass / 4. Exported premerge row mass is source-layer nodal mass, not used as cell mass."},
    "limitations":["Reference pullback ledger only. It does not prove current-emitted geometry overlap or material/contact admission.","Compiled row source identity and density fields are pinned to frozen 039 source-only export.","The 14-row changed retet patch is mapped by geometric overlap; all other rows use exact identity/refinement connectivity.","Source corner donor/body/local point provenance is retained; no claim that per-corner cell mass equals the original premerge nodal contribution mass."]
}
report_path=OUT/"report.json"
report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
manifest={"script":str(SCRIPT),"script_sha256":sha(SCRIPT),"map":str(map_path),"map_sha256":sha(map_path),"report":str(report_path),"report_sha256":sha(report_path),"inputs_sha256":input_hashes,"runtime_seconds":time.time()-t0}
(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2))
print("manifest",json.dumps(manifest,sort_keys=True))

