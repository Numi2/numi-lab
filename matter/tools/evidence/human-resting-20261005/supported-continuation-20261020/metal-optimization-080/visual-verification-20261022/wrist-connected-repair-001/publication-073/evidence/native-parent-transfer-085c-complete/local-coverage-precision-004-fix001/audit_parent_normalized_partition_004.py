#!/usr/bin/env python3
"""Offline 9644-row source to cooked-parent to locally precision-corrected 085c child moment audit."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, json
import numpy as np

R=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=R/"connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight"
O=Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded/native-parent-transfer-085c/local-coverage-precision-004-fix001")
OUT=O/"parent-normalized-partition-diagnostic"; OUT.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(newline="") as f:return list(csv.DictReader(f))
src=rows(D/"premerge-layer-contributions.csv")
nodes=rows(D/"compiled-skin-fem-nodes.csv")
tets=[x for x in rows(D/"compiled-skin-fem-tets.csv") if int(x["identity_active_bit"])==1]
assert (len(src),len(nodes),len(tets))==(9644,7233,28632)
assert all(x["mass_semantics"]=="owner_layer_lumped_nodal_mass" for x in src)
gid=np.array([int(x["combined_source_global_node_index"]) for x in src],np.int64)
mat=np.array([int(x["source_material_index"]) for x in src],np.int64)
donor=np.array([int(x["node_frame_owner_body"]) for x in src],np.int64)
raw=np.array([float(x["mass_kg"]) for x in src],np.float64)
xyz=np.array([[float(x[f"donor_local_{a}_m"]) for a in "xyz"] for x in src],np.float64)
grp=defaultdict(list)
for i,k in enumerate(zip(gid,mat)):grp[(int(k[0]),int(k[1]))].append(i)
ixs={k:np.asarray(v,np.int64) for k,v in grp.items()}
rgrp={k:float(raw[ix].sum()) for k,ix in ixs.items()}
assert all(v>0 and np.isfinite(v) for v in rgrp.values())
def moms(w):
    ans={}
    for d,m in sorted(set(zip(donor.tolist(),mat.tolist()))):
        ix=np.flatnonzero((donor==d)&(mat==m)); a=w[ix]; x=xyz[ix]
        ans[(d,m)]={"mass":float(a.sum()),"first":(a[:,None]*x).sum(0),
            "second":np.einsum("n,ni,nj->ij",a,x,x),"scale1":float(np.sum(a*np.linalg.norm(x,axis=1))),
            "scale2":float(np.sum(a*np.sum(x*x,axis=1))),"rows":int(len(ix))}
    return ans
def cmp(a,b):
    z={}
    for k in sorted(set(a)|set(b)):
        aa=a.get(k);bb=b.get(k);am=0 if aa is None else aa["mass"];bm=0 if bb is None else bb["mass"]
        af=np.zeros(3) if aa is None else aa["first"];bf=np.zeros(3) if bb is None else bb["first"]
        a2=np.zeros((3,3)) if aa is None else aa["second"];b2=np.zeros((3,3)) if bb is None else bb["second"]
        s1=max(1e-300,0 if aa is None else aa["scale1"]);s2=max(1e-300,0 if aa is None else aa["scale2"])
        z[f"{k[0]}:{k[1]}"]={"donor_body":k[0],"material_index":k[1],
            "mass_a_kg":float(am),"mass_b_kg":float(bm),"mass_delta_kg":float(bm-am),
            "mass_delta_relative":float((bm-am)/am) if am else None,
            "first_a_kg_m":af.tolist(),"first_b_kg_m":bf.tolist(),"first_delta_kg_m":(bf-af).tolist(),
            "first_delta_norm_relative_abs_scale":float(np.linalg.norm(bf-af)/s1),
            "second_a_kg_m2":a2.tolist(),"second_b_kg_m2":b2.tolist(),"second_delta_kg_m2":(b2-a2).tolist(),
            "second_delta_fro_relative_abs_scale":float(np.linalg.norm(b2-a2)/s2),"source_rows":(aa or bb)["rows"]}
    return z
# Cooked cell mass and corner mass from the actual active compiled CSV.
pt=np.array([[int(x[f"global_node{j}"]) for j in range(4)] for x in tets],np.int64)
pm=np.array([int(x["material_index"]) for x in tets],np.int64)
vol=np.array([float(x["inverse_rest_w_m3"]) for x in tets])
rho=np.array([float(x["compiled_bulk_x"]) for x in tets])
mass=vol*rho
corner=np.array([float(x["bulk_density_corner_mass_kg"]) for x in tets])
assert np.max(np.abs(corner-mass/4))==0
node_mat=defaultdict(float)
for p in range(len(tets)):
    for j in range(4):node_mat[(int(pt[p,j]),int(pm[p]))]+=float(mass[p]/4)
cooked=np.zeros(len(src))
for k,ix in ixs.items():
    if k not in node_mat:raise AssertionError(f"source group absent from compiled tets: {k}")
    cooked[ix]=node_mat[k]*raw[ix]/rgrp[k]
if set(node_mat)-set(ixs):raise AssertionError(f"compiled node/material groups missing source rows: {list(set(node_mat)-set(ixs))[:5]}")
compiled_n={int(x["global_node"]):float(x["mass_kg"]) for x in nodes}
cooked_n=defaultdict(float)
for (g,m),v in node_mat.items():cooked_n[g]+=v
node_err=np.array([abs(cooked_n[g]-v) for g,v in compiled_n.items()])
node_rel=np.array([abs(cooked_n[g]-v)/max(abs(v),1e-30) for g,v in compiled_n.items()])
rawm=moms(raw); cookm=moms(cooked); raw_cooked=cmp(rawm,cookm)
# Exact functional producer scheme: each parent-child transfer mass is split 1/16
# over child corners x parent corners, then by source-row raw-mass proportion.
M=np.load(O/"native-parent-transfer-085c.npz",allow_pickle=False)
C=np.load("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz",allow_pickle=False)
ind=M["child_native_parent_indptr"];pidx=M["child_native_parent_rows"]
overlap=M["overlap_native_reference_volume_m3"]
parent_covered=np.bincount(pidx,weights=overlap,minlength=len(mass))
pfrac=overlap/parent_covered[pidx]
raw_parent_fraction=M["parent_mass_fraction_overlap_over_native_parent_geometry_volume"]
cmass=np.zeros(len(M["candidate_cell_uids"]),dtype=np.float64)
for ci in range(len(cmass)):
    a,z=int(ind[ci]),int(ind[ci+1]); cmass[ci]=np.dot(mass[pidx[a:z]],pfrac[a:z])
ct=C["tetrahedra_zero_based"].astype(np.int64)
cmat=np.array([{1:15,2:16}[int(v)] for v in C["material_ids"]],np.int64)
assert len(ct)==len(cmass) and np.array_equal(C["cell_uids"],M["candidate_cell_uids"]) and np.array_equal(C["material_ids"].astype(np.int8),M["candidate_child_material_ids"])
node_n=len(C["reference_points_m"]); slot={15:0,16:1}
node_expected=np.zeros((node_n,2)); node_received=np.zeros_like(node_expected)
childmass_recalc=np.zeros(len(ct)); childrows=np.zeros(len(src))
for c,tet in enumerate(ct):
    m=int(cmat[c]);s=slot[m]
    for e in range(int(ind[c]),int(ind[c+1])):
        p=int(pidx[e]); transfer=float(mass[p]*pfrac[e]); childmass_recalc[c]+=transfer
        q=transfer/4.
        for cn0 in tet:
            cn=int(cn0);node_expected[cn,s]+=q
            each_parent_corner=q/4.
            for j in range(4):
                key=(int(pt[p,j]),m);ix=ixs[key]
                add=each_parent_corner*raw[ix]/rgrp[key]
                childrows[ix]+=add.sum()
                node_received[cn,s]+=add.sum()
cmass_rel=np.abs(childmass_recalc-cmass)/np.maximum(cmass,1e-30)
nerr=np.abs(node_received-node_expected);nrel=nerr/np.maximum(np.abs(node_expected),1e-30)
rerr=np.abs(childrows-cooked);rrel=rerr/np.maximum(np.abs(cooked),1e-30)
if cmass_rel.max()>1e-10: print("DIAGNOSTIC child parent-CSR mass recompute exceeds 1e-10",float(cmass_rel.max()))
if nrel.max()>1e-10: print("DIAGNOSTIC child node/material closure exceeds 1e-10",float(nrel.max()))
if rrel.max()>1e-10: print("DIAGNOSTIC cooked-to-child source-row mass closure exceeds 1e-10",float(rrel.max()))
childm=moms(childrows);cook_child=cmp(cookm,childm)



def totals(w):return {str(m):float(w[mat==m].sum()) for m in sorted(set(mat.tolist()))}
report={"status":"ROUNDING_NORMALIZED_PARENT_PARTITION_DIAGNOSTIC_NOT_OWNER_MAP",
 "method":"Diagnostic alternative only: divide each native parent's unchanged geometric overlap volumes by their summed covered volume to close its mass partition, then split each transfer across four child corners and four parent corners (1/16 each), then split each parent-corner share among matching raw source rows by raw mass within (combined source node, material). Source donor-local points remain unchanged; the geometric CSR volumes are not modified.",
 "moment_convention":"Per donor body/material, first moment=sum(m*x); second moment is full raw tensor sum(m*x*x^T), where x is the original donor-local source point.",
 "inputs_sha256":{"source_contributions":sha(D/"premerge-layer-contributions.csv"),"compiled_nodes":sha(D/"compiled-skin-fem-nodes.csv"),"compiled_tets":sha(D/"compiled-skin-fem-tets.csv"),"transfer_085c_local":sha(O/"native-parent-transfer-085c.npz"),"precision_csr":sha(O/"geometric-overlap-local-precision-004.npz"),"candidate_084_final":sha("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz"),"transfer_085c_local":sha(O/"native-parent-transfer-085c.npz")},
 "dimensions":{"source_rows":len(src),"node_material_groups":len(ixs),"native_active_parent_tets":len(tets),"child_tets":len(ct),"child_nodes":node_n,"child_parent_csr_nnz":len(pidx),"donor_material_groups":len(rawm)},
 "source_to_cooked":{"raw_mass_by_material_kg":totals(raw),"cooked_parent_mass_by_material_kg":totals(cooked),"per_donor_material_mass_and_moments":raw_cooked,"max_abs_group_mass_delta_kg":max(abs(x["mass_delta_kg"]) for x in raw_cooked.values()),"max_abs_group_mass_delta_relative":max(abs(x["mass_delta_relative"] or 0) for x in raw_cooked.values()),"max_compiled_node_mass_closure_abs_kg":float(node_err.max()),"max_compiled_node_mass_closure_relative":float(node_rel.max())},
 "cooked_to_child":{"cooked_mass_by_material_kg":totals(cooked),"child_source_attributed_mass_by_material_kg":totals(childrows),"per_donor_material_mass_and_moments":cook_child,"max_abs_group_mass_delta_kg":max(abs(x["mass_delta_kg"]) for x in cook_child.values()),"max_abs_group_mass_delta_relative":max(abs(x["mass_delta_relative"] or 0) for x in cook_child.values()),"max_abs_source_row_mass_closure_kg":float(rerr.max()),"max_relative_source_row_mass_closure":float(rrel.max()),"source_row_mass_closure_worst_rows":[{"row":int(i),"combined_global_node":int(gid[i]),"material":int(mat[i]),"donor_body":int(donor[i]),"cooked_mass_kg":float(cooked[i]),"child_attributed_mass_kg":float(childrows[i]),"relative_error":float(rrel[i])} for i in np.argsort(rrel)[-12:][::-1]],"max_abs_child_node_material_closure_kg":float(nerr.max()),"max_relative_child_node_material_closure":float(nrel.max()),"max_child_cell_mass_parent_CSR_recompute_relative_error":float(cmass_rel.max()),"max_abs_parent_partition_fraction_adjustment_relative":float(np.max(np.abs((pfrac-raw_parent_fraction)/np.maximum(np.abs(raw_parent_fraction),1e-300))))},
 "limitations":["This parent-normalized partition is a diagnostic alternative, not the frozen 073 owner map; it must not be promoted without owner review. It does not alter the geometric CSR, whose coverage errors are reported separately.","Source-layer contribution rows are source mass/frame provenance; compiled parent cell mass is authoritative cooked mass.","This is a reference-pullback ledger audit, not current geometry, native mass activation, bone clearance, or physical qualification."]}
out=OUT/"parent-normalized-partition-diagnostic-004.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2,sort_keys=True))
