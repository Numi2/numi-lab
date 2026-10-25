#!/usr/bin/env python3
import csv, json, hashlib, pathlib, collections, sys
import numpy as np
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
AP=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz"
APN=E/"native-candidate020-conforming-prism-030/source-preflight-006/initial-skin-fem-nodes.csv"
C=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068/run-004/source-preflight"
D=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068"
scriptpath=D/"compare_cooked_ap_068.py"
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a=np.load(AP)
tap=np.asarray(a["tetrahedra"],dtype=np.int64)
map_=np.asarray(a["material_ids"],dtype=np.int64)
pap=np.asarray(a["points_world_f32"],dtype=np.float32)
with (C/"compiled-skin-fem-tets.csv").open() as f: cr=list(csv.DictReader(f))
with (C/"compiled-skin-fem-nodes.csv").open() as f: nr=list(csv.DictReader(f))
with APN.open() as f: apnr=sorted(csv.DictReader(f),key=lambda r:int(r["global_node"]))
nodes=sorted(nr,key=lambda r:int(r["global_node"]))
global_ids=np.array([int(r["global_node"]) for r in nodes],dtype=np.int64)
ap_global_ids=np.array([int(r["global_node"]) for r in apnr],dtype=np.int64)
assert len(nodes)==len(apnr)==pap.shape[0] and len(cr)==tap.shape[0]
assert np.array_equal(global_ids,ap_global_ids)
base=int(global_ids[0])
pcomp=np.array([[float(r[k]) for k in ("position_x","position_y","position_z")] for r in nodes],dtype=np.float32)
rest=np.array([[float(r[k]) for k in ("rest_x","rest_y","rest_z")] for r in nodes],dtype=np.float32)
mnode=np.array([float(r["mass_kg"]) for r in nodes],dtype=np.float32)
node_bits_equal=bool(np.array_equal(pcomp.view(np.uint32),pap.view(np.uint32)))
APNpts=np.array([[float(r[k]) for k in ("x_m","y_m","z_m")] for r in apnr],dtype=np.float32)
apn_csv_bits_equal=bool(np.array_equal(pap.view(np.uint32),APNpts.view(np.uint32)))
Tg=np.array([[int(r[f"global_node{i}"])-base for i in range(4)] for r in cr],dtype=np.int64)
Ts=np.array([[int(r[f"source_node{i}"]) for i in range(4)] for r in cr],dtype=np.int64)
mc=np.array([int(r["material_index"]) for r in cr],dtype=np.int64)
ma=np.array([int(r["authored_material_index"]) for r in cr],dtype=np.int64)
flags=np.array([int(r["flags"]) for r in cr],dtype=np.uint64)
identity_obj=np.array([int(r["compiled_identity_y"]) for r in cr],dtype=np.int64)
srcrow=np.array([int(r["source_tet_row"]) for r in cr],dtype=np.int64)
Vcook=np.array([float(r["inverse_rest_w_m3"]) for r in cr],dtype=np.float64)
bulk=np.array([float(r["compiled_bulk_x"]) for r in cr],dtype=np.float64)
inv=np.array([[float(r[f"inverse_rest_r{rr}c{cc}"]) for rr in range(3) for cc in range(3)] for r in cr],dtype=np.float64).reshape(-1,3,3)
invbits=np.array([[int(r[f"inverse_rest_bits_r{rr}c{cc}"]) for rr in range(3) for cc in range(3)] for r in cr],dtype=np.uint32).reshape(-1,3,3)
def detvol(X,T):
 v=X[T]
 D=np.stack([v[:,1]-v[:,0],v[:,2]-v[:,0],v[:,3]-v[:,0]],axis=2)
 return np.linalg.det(D)/6.0,D
vol_ap_signed,Dap=detvol(pap,tap)
vol_rest_signed,Drest=detvol(rest,Tg)
# Canonical per-cell node sets; row order and orientation are assessed separately.
def keyrows(T): return [tuple(sorted(map(int,row))) for row in T]
kap=keyrows(tap); kcomp=keyrows(Tg)
ap_index=collections.defaultdict(list)
for i,k in enumerate(kap): ap_index[k].append(i)
matched=[ap_index.get(k,[]) for k in kcomp]
unique_match=all(len(v)==1 for v in matched)
map_comp_to_ap=np.array([v[0] if len(v)==1 else -1 for v in matched],dtype=np.int64)
same_set_rows=(map_comp_to_ap>=0)
if unique_match:
  Tcell=bool(np.array_equal(np.sort(Tg,axis=1),np.sort(tap[map_comp_to_ap],axis=1)))
  row_same_order=bool(np.array_equal(Tg,tap[map_comp_to_ap]))
  row_same_direct=bool(np.array_equal(Tg,tap))
else:
  Tcell=False; row_same_order=False; row_same_direct=False
  Tcell=int(np.sum(same_set_rows))
orientation={}
if unique_match:
 det_comp,_=detvol(pap,Tg)
 det_ap_mapped=vol_ap_signed[map_comp_to_ap]
 signprod=np.sign(det_comp)*np.sign(det_ap_mapped)
 orientation={"same_sign":int(np.count_nonzero(signprod>0)),"opposite_sign":int(np.count_nonzero(signprod<0)),"zero_or_nonfinite":int(np.count_nonzero(signprod==0))}
# Material mapping by exact matched cell identities
material_cross={}
material_match_map=None
if unique_match:
 for ac in sorted(np.unique(map_).tolist()):
  sub=map_[map_comp_to_ap]==ac
  material_cross[str(ac)]={str(int(x)):int(np.count_nonzero(sub&(mc==x))) for x in np.unique(mc)}
 pairs=collections.Counter((int(map_[ai]),int(ci)) for ci,ai in enumerate(map_comp_to_ap) if ai>=0)
 material_cross_counts={f"AP{apm}->compiled{cm}":int(n) for (apm,cm),n in pairs.items()}
# Nodal mass assembly from actual cooked F32 inverse-rest volume and compiled density.
mass64=np.zeros(len(nodes),dtype=np.float64)
cell_corner=np.zeros(len(cr),dtype=np.float64)
for i,row in enumerate(Tg):
 cell_corner[i]=bulk[i]*Vcook[i]*0.25
 for n in row: mass64[int(n)]+=cell_corner[i]
mass_f32=mass64.astype(np.float32)
delta=mass_f32.astype(np.float64)-mnode.astype(np.float64)
# AP geometry F32 determinant + material->compiled density mapping (only compare if full material correspondence exists)
APmat_to_comp={}
if unique_match:
 for apm in np.unique(map_):
  vals=np.unique(mc[map_[map_comp_to_ap]==apm])
  if len(vals)==1: APmat_to_comp[int(apm)]=int(vals[0])
ap_bulk=np.full(len(tap),np.nan)
if unique_match:
 for i,ai in enumerate(map_comp_to_ap):
  ap_bulk[ai]=bulk[i]
ap_cell_mass=np.zeros(len(tap))
for i,row in enumerate(tap):
 if np.isfinite(ap_bulk[i]): ap_cell_mass[i]=abs(vol_ap_signed[i])*ap_bulk[i]
ap_mass=np.zeros(len(nodes),dtype=np.float64)
if np.all(np.isfinite(ap_bulk)):
 for i,row in enumerate(tap):
  q=ap_cell_mass[i]*0.25
  for n in row: ap_mass[int(n)]+=q
ap_mass_f32=ap_mass.astype(np.float32)
ap_delta=ap_mass_f32.astype(np.float64)-mnode.astype(np.float64)
# Recompute inverse reference matrix from exported rest F32 nodes; compare both conventions.
inv_expected=np.linalg.inv(Drest)
inv_expected_f32=inv_expected.astype(np.float32)
invf_error=np.abs(inv-inv_expected_f32.astype(np.float64))
invt_error=np.abs(inv-np.swapaxes(inv_expected_f32,1,2).astype(np.float64))
# direct volume check from exported rest points
Vrest_abs=np.abs(vol_rest_signed)
# Raw source/compiled headers and source pins
run_manifest=json.loads((C/"export-launch-manifest.json").read_text())
linkcmd=json.loads((D/"build-001/link-command.json").read_text())
linklibs=[x for x in linkcmd if x.endswith(".dylib")]
runtime=pathlib.Path(run_manifest["runtime_library"])/"libmetalrobo.dylib"
metallib=pathlib.Path(run_manifest["metallib"])
stderr=(C/"stderr.txt").read_text(errors="replace")
loaded=[line.strip() for line in stderr.splitlines() if "libmetalrobo.dylib" in line]
report={
 "scope":"source-only compiled 039 object export compared with AP 026/030 candidate; no simulation/native admission",
 "pins":{"AP_candidate_npz_sha256":sha(AP),"AP_node_csv_sha256":sha(APN),"compiled_tets_csv_sha256":sha(C/"compiled-skin-fem-tets.csv"),"compiled_nodes_csv_sha256":sha(C/"compiled-skin-fem-nodes.csv"),"overlay_source_sha256":sha(D/"apps/numilab_human_myosim_visual_probe.mm"),"overlay_binary_sha256":sha(D/"build-001/numi-human-native"),"compile_command_sha256":sha(D/"build-001/compile-command.json"),"link_command_sha256":sha(D/"build-001/link-command.json"),"linked_dylibs":[{"path":p,"sha256":sha(p)} for p in linklibs if pathlib.Path(p).exists()],"runtime_lib_path":str(runtime),"runtime_lib_sha256":sha(runtime) if runtime.exists() else None,"metallib_path":str(metallib),"metallib_sha256":sha(metallib) if metallib.exists() else None,"loaded_libmetalrobo_lines":loaded},
 "compiled_object":{"object_index":int(identity_obj[0]),"compiled_global_tet_range":[int(srcrow[0])+3315,int(srcrow[-1])+3315],"source_rows":len(cr),"all_active":bool(np.all((flags&1)!=0)),"identity_object_ids_unique":sorted(set(map(int,identity_obj))),"identity_material_ids":{str(int(k)):int(v) for k,v in collections.Counter(mc).items()},"authored_source_material_ids":{str(int(k)):int(v) for k,v in collections.Counter(ma).items()},"all_material_ids_match_authored":bool(np.array_equal(mc,ma)),"all_source_to_global_node_rows_match":bool(np.all(Tg==Ts))},
 "node_domain":{"node_count":len(nodes),"global_node_range":[int(global_ids.min()),int(global_ids.max())],"AP_world_points_equal_compiled_position_F32_bits":node_bits_equal,"AP_npz_points_equal_AP_node_csv_F32_bits":apn_csv_bits_equal,"compiled_rest_nodes_exported":True,"rest_position_f32_bit_columns_exported":True},
 "tetrahedra_comparison":{"AP_rows":len(tap),"compiled_rows":len(Tg),"AP_material_ids":{str(int(k)):int(v) for k,v in collections.Counter(map_).items()},"compiled_global_material_ids":{str(int(k)):int(v) for k,v in collections.Counter(mc).items()},"same_node_set_per_cell":bool(unique_match and Tcell),"same_source_row_order_and_node_order":row_same_direct,"same_node_set_per_cell_with_row_reordering":bool(unique_match and Tcell),"row_map_unique":bool(unique_match),"matched_cells":int(np.count_nonzero(same_set_rows)),"orientation_parity_of_corresponding_rows":orientation,"AP_to_compiled_material_crosswalk":material_cross_counts if unique_match else None,"source_rows_identity_order":bool(np.array_equal(srcrow,np.arange(len(cr)))),"max_difference_source_vs_global_local_node_id":int(np.max(np.abs(Tg-Ts)))},
 "reference_operator":{"cooked_f32_volume_total_m3":float(Vcook.sum()),"cooked_f32_volume_by_compiled_material_m3":{str(int(k)):float(Vcook[mc==k].sum()) for k in np.unique(mc)},"AP_F32_geometry_signed_volume_by_material_id_m3":{str(int(k)):float(vol_ap_signed[map_==k].sum()) for k in np.unique(map_)},"max_abs_w_minus_abs_det_compiled_rest_f32_m3":float(np.max(np.abs(Vcook-Vrest_abs))),"max_abs_inverse_coeff_error_vs_inverse_of_exported_rest_nodes_after_f32_cast":float(np.max(invf_error)),"max_abs_inverse_coeff_error_vs_transposed_alternative":float(np.max(invt_error)),"inverse_row0w_volume_formula":{"max_abs_signed_det_over6":float(np.max(np.abs(Vcook-vol_rest_signed))),"max_abs_abs_det_over6":float(np.max(np.abs(Vcook-Vrest_abs)))}},
 "nodal_mass_reconstruction":{"compiled_node_mass_total_kg":float(mnode.astype(np.float64).sum()),"from_cooked_w_bulk_cell_corner_mass_total_kg":float(cell_corner.sum()),"cooked_corner_assembly_then_f32":{"max_abs_node_error_kg":float(np.max(np.abs(delta))),"max_abs_node_error_g":float(np.max(np.abs(delta))*1000),"sum_signed_node_error_kg":float(delta.sum()),"sum_abs_node_error_kg":float(np.abs(delta).sum()),"nodes_nonzero_error":int(np.count_nonzero(delta)),"max_error_node_global_id":int(global_ids[np.argmax(np.abs(delta))]),"max_expected_node_kg":float(mass_f32[np.argmax(np.abs(delta))]),"max_compiled_node_kg":float(mnode[np.argmax(np.abs(delta))])},"AP_geometry_volume_corner_assembly_then_f32":{"available":bool(np.all(np.isfinite(ap_bulk))),"max_abs_node_error_kg":float(np.max(np.abs(ap_delta))) if np.all(np.isfinite(ap_bulk)) else None,"max_abs_node_error_g":float(np.max(np.abs(ap_delta))*1000) if np.all(np.isfinite(ap_bulk)) else None,"sum_signed_node_error_kg":float(ap_delta.sum()) if np.all(np.isfinite(ap_bulk)) else None,"sum_abs_node_error_kg":float(np.abs(ap_delta).sum()) if np.all(np.isfinite(ap_bulk)) else None,"nodes_nonzero_error":int(np.count_nonzero(ap_delta)) if np.all(np.isfinite(ap_bulk)) else None}},
 "sample_cell_matches":[]}
if unique_match:
 for idx in sorted(set([0,len(cr)//2,len(cr)-1,int(np.argmax(np.abs(Vcook-Vrest_abs)))])):
  report["sample_cell_matches"].append({"compiled_source_tet_row":int(srcrow[idx]),"compiled_global_tet_row":int(idx+3315),"AP_tet_row":int(map_comp_to_ap[idx]),"compiled_material":int(mc[idx]),"AP_material":int(map_[map_comp_to_ap[idx]]),"compiled_nodes_local":Tg[idx].tolist(),"AP_nodes_local":tap[map_comp_to_ap[idx]].tolist(),"compiled_F32_volume_m3":float(Vcook[idx]),"AP_F32_signed_volume_m3":float(vol_ap_signed[map_comp_to_ap[idx]]),"compiled_rest_F32_signed_volume_m3":float(vol_rest_signed[idx])})
out=D/"cooked-vs-ap-068"
out.mkdir(exist_ok=True)
(out/"comparison-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2))

