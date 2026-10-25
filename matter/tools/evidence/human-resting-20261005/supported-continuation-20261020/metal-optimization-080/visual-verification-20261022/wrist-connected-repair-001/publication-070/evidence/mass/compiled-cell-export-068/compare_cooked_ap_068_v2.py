import csv,json,hashlib,pathlib,collections,struct
import numpy as np
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
AP=E/"native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz"
APN=E/"native-candidate020-conforming-prism-030/source-preflight-006/initial-skin-fem-nodes.csv"
D=E/"connected-skin-parent-mass-transfer-067/compiled-cell-export-068"; C=D/"run-004/source-preflight"; O=D/"cooked-vs-ap-068-v2"; O.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a=np.load(AP); Ta=np.asarray(a["tetrahedra"],np.int64); ma=np.asarray(a["material_ids"],np.int64)
To=np.asarray(a["old_tetrahedra"],np.int64) if "old_tetrahedra" in a.files else None
with (C/"compiled-skin-fem-tets.csv").open() as f: cr=list(csv.DictReader(f))
with (C/"compiled-skin-fem-nodes.csv").open() as f: nr=sorted(csv.DictReader(f),key=lambda r:int(r["global_node"]))
with APN.open() as f: an=sorted(csv.DictReader(f),key=lambda r:int(r["global_node"]))
assert len(cr)==len(Ta) and len(nr)==len(an)
gid=np.array([int(r["global_node"]) for r in nr],np.int64); assert np.array_equal(gid,[int(r["global_node"]) for r in an])
base=int(gid[0]); Pa=np.array([[float(r[k]) for k in ("x_m","y_m","z_m")] for r in an],np.float32)
Pc=np.array([[float(r[k]) for k in ("position_x","position_y","position_z")] for r in nr],np.float32)
T=np.array([[int(r[f"global_node{i}"])-base for i in range(4)] for r in cr],np.int64)
Ts=np.array([[int(r[f"source_node{i}"]) for i in range(4)] for r in cr],np.int64)
mc=np.array([int(r["material_index"]) for r in cr],np.int64); ms=np.array([int(r["authored_material_index"]) for r in cr],np.int64)
vol=np.array([float(r["inverse_rest_w_m3"]) for r in cr]); rho=np.array([float(r["compiled_bulk_x"]) for r in cr])
flags=np.array([int(r["flags"]) for r in cr],np.uint64); sr=np.array([int(r["source_tet_row"]) for r in cr],np.int64)
mnode=np.array([float(r["mass_kg"]) for r in nr],np.float32)
def key(t):return tuple(sorted(map(int,t)))
def mapT(U):
 d=collections.defaultdict(list)
 for j,t in enumerate(U):d[key(t)].append(j)
 v=np.full(len(T),-1,np.int64); rev=np.full(len(U),-1,np.int64)
 for i,t in enumerate(T):
  cand=d.get(key(t),[])
  if cand:
   j=cand.pop(0);v[i]=j;rev[j]=i
 return v,rev
ma_to_comp,comp_to_ma=mapT(Ta)
match=int(np.count_nonzero(ma_to_comp>=0))
oldstats=None
if To is not None:
 old_to_comp,comp_to_old=mapT(To)
 oldstats={"matched":int(np.count_nonzero(old_to_comp>=0)),"compiled_unmatched":int(np.count_nonzero(comp_to_old<0)),"old_candidate_unmatched":int(np.count_nonzero(old_to_comp<0)),"ordered_equal":bool(np.array_equal(To,T))}
cross=collections.Counter((int(ma[j]),int(mc[i])) for i,j in enumerate(ma_to_comp) if j>=0)
mapmat={}
for am in np.unique(ma):
 vals=sorted({cm for (a0,cm),n in cross.items() if a0==int(am) and n})
 mapmat[int(am)]=vals[0] if len(vals)==1 else vals
assert all(isinstance(x,int) for x in mapmat.values()),mapmat
rho_lookup={int(m):float(np.median(rho[mc==m])) for m in np.unique(mc)}
rho_ap=np.array([rho_lookup[mapmat[int(x)]] for x in ma])
def vols(X,U):
 v=X[U]; return np.linalg.det(np.stack([v[:,1]-v[:,0],v[:,2]-v[:,0],v[:,3]-v[:,0]],axis=2))/6
Va=vols(Pa,Ta)
Vap_npz=vols(np.asarray(a["points_world_f32"],np.float32),Ta)
# actual cooked nodal assembly, double accumulation then f32 output
mass64=np.zeros(len(gid),np.float64); mass_seq=np.zeros(len(gid),np.float32)
cellmass=rho*vol
for i,t in enumerate(T):
 q64=cellmass[i]*.25; mass64[t]+=q64
 q32=np.float32(np.float32(rho[i])*np.float32(vol[i])*np.float32(.25))
 for n in t: mass_seq[n]=np.float32(mass_seq[n]+q32)
mass64f=mass64.astype(np.float32)
# AP fit-node current geometry reconstructed per AP T
mapass=np.zeros(len(gid),np.float64)
apcmass=np.abs(Va)*rho_ap
for i,t in enumerate(Ta):mapass[t]+=apcmass[i]*.25
mapassf=mapass.astype(np.float32)
delta=(mapassf.astype(np.float64)-mnode.astype(np.float64))*1000
idx=np.argsort(np.abs(delta))[::-1][:20]
# direct rows matched volume comparison where AP canonical cell exactly matches cooked
dm=[]
for i,j in enumerate(ma_to_comp):
 if j>=0: dm.append((vol[i],abs(Va[j]),i,int(mc[i])))
# material/density exact bits
dtable={}
for m in np.unique(mc):
 r=float(np.median(rho[mc==m])); f=np.float32(r)
 dtable[str(int(m))]={"active_tets":int(np.count_nonzero(mc==m)),"bulk_x_decimal":r,"bulk_x_f32_hex":f"0x{int(f.view(np.uint32)):08x}","authored_material_ids":sorted(set(map(int,ms[mc==m]))),"cooked_volume_m3":float(vol[mc==m].sum()),"mass_kg":float(cellmass[mc==m].sum())}
mismatch_ap=np.flatnonzero(ma_to_comp<0); mismatch_c=np.flatnonzero(comp_to_ma<0)
report={
"scope":"Read-only comparison: exact active frozen-039 compiled cells/operator and node masses vs AP026 candidate connectivity plus actual AP030 source-preflight-006 F32 fit points. AP NPZ points are only a separate counterfactual; no simulation/native admission.",
"pins":{"AP_npz_sha256":sha(AP),"AP_fit_points_csv_sha256":sha(APN),"compiled_tets_sha256":sha(C/"compiled-skin-fem-tets.csv"),"compiled_nodes_sha256":sha(C/"compiled-skin-fem-nodes.csv"),"exporter_source_sha256":sha(D/"apps/numilab_human_myosim_visual_probe.mm"),"exporter_binary_sha256":sha(D/"build-001/numi-human-native"),"compile_command_sha256":sha(D/"build-001/compile-command.json"),"link_command_sha256":sha(D/"build-001/link-command.json"),"runtime_sha256":sha(pathlib.Path(json.loads((C/"export-launch-manifest.json").read_text())["runtime_library"])/"libmetalrobo.dylib")},
"compiled":{"active_cell_count":len(T),"all_rows_active":bool(np.all(flags&1)),"source_rows_identity":bool(np.array_equal(sr,np.arange(len(T)))),"authored_material_equals_compiled_material":bool(np.array_equal(ms,mc)),"source_nodes_equal_global_local_nodes":bool(np.array_equal(Ts,T)),"compiled_global_tet_range":[int(sr.min()+3315),int(sr.max()+3315)],"material_density_table":dtable},
"point_input":{"all_7233_F32_fit_points_equal_compiled_current_positions":bool(np.array_equal(Pa.view(np.uint32),Pc.view(np.uint32))),"different_rows":int(np.count_nonzero(np.any(Pa.view(np.uint32)!=Pc.view(np.uint32),axis=1))),"candidate_npz_points_equal_actual_fit_csv":bool(np.array_equal(np.asarray(a["points_world_f32"],np.float32).view(np.uint32),Pa.view(np.uint32)))},
"connectivity":{"AP_rows":len(Ta),"compiled_rows":len(T),"matched_exact_unordered_node_sets":match,"AP_unmatched_rows":np.flatnonzero(ma_to_comp<0).astype(int).tolist(),"compiled_unmatched_rows":np.flatnonzero(comp_to_ma<0).astype(int).tolist(),"ordered_tet_rows_equal":bool(np.array_equal(Ta,T)),"old_candidate_topology":oldstats,"material_crosswalk_from_exactmatchedcells":{str(k):v for k,v in mapmat.items()},"AP_unmatched_witnesses":[{"ap_row":int(i),"nodes":Ta[i].tolist(),"material":int(ma[i]),"changed_tet_row":bool(i in set(map(int,a["changed_tet_rows"]))) if "changed_tet_rows" in a.files else None} for i in mismatch_ap[:30]],"compiled_unmatched_witnesses":[{"compiled_row":int(i),"source_row":int(sr[i]),"nodes_local":T[i].tolist(),"nodes_global":(T[i]+base).tolist(),"material":int(mc[i])} for i in mismatch_c[:30]]},
"volume_mass":{"compiled_rest_volume_m3":float(vol.sum()),"compiled_rest_volume_by_material_m3":{str(int(m)):float(vol[mc==m].sum()) for m in np.unique(mc)},"compiled_parent_mass_by_material_kg":{str(int(m)):float(cellmass[mc==m].sum()) for m in np.unique(mc)},"AP_actual_fit_F32_geometry_volume_by_AP_material_m3":{str(int(m)):float(np.abs(Va[ma==m]).sum()) for m in np.unique(ma)},"AP_current_mass_by_material_kg_using_density_crosswalk":{str(int(m)):float(apcmass[ma==m].sum()) for m in np.unique(ma)},"matched_cell_count_for_AP_vs_cooked_row0w":len(dm),"matched_cell_row0w_vs_AP_F32current_volume_absdiff_max_m3":float(max((abs(v-abs(va)) for v,va,_,_ in dm),default=0)),"AP_candidate_NPZ_points_volume_total_m3_not_used_by_fit":float(np.abs(Vap_npz).sum())},
"nodal_mass":{"compiled_nodal_mass_sum_kg":float(mnode.sum()),"cooked_cell_rho_times_row0w_sum_kg":float(cellmass.sum()),"cooked_cell_corner_contribution_sum_times4_kg":float(cellmass.sum()),"F64nodal_accum_then_F32_max_error_g":float(np.max(np.abs(mass64f.astype(np.float64)-mnode.astype(np.float64)))*1000),"F64nodal_accum_nonzero_error_count":int(np.count_nonzero(mass64f.view(np.uint32)!=mnode.view(np.uint32))),"sequential_F32corner_accum_max_error_g":float(np.max(np.abs(mass_seq.astype(np.float64)-mnode.astype(np.float64)))*1000),"AP_T_plus_actual030F32points_max_node_mass_difference_g":float(np.max(np.abs(delta))),"AP_T_plus_actual030F32points_nodes_nonzero_difference":int(np.count_nonzero(delta)),"AP_T_difference_witnesses_top20":[{"local_node":int(i),"global_node":int(gid[i]),"ap_mass_kg":float(mapassf[i]),"compiled_mass_kg":float(mnode[i]),"delta_g":float(delta[i])} for i in idx]},
"native_recipe":"The compiled node mass is rebuilt as sum over active cooked tet corners of bulk.x * inverseRestRow0.w / 4, accumulated in F64 then cast to F32. AP fit comparison uses the actual 030 source-preflight-006 initial-skin-fem-nodes.csv points, matching the 033 producer script; it does not use candidate.npz points_world_f32."
}
(O/"comparison-report-v2.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps(report,indent=2))
