"""Exact bridge from joint mechanical quotient to existing role-volume row domains.
Uses the retained 217 node table. No guessed coordinates, parent rows or mass.
"""
from pathlib import Path
import numpy as np,csv,gzip,json,sys,hashlib,importlib.util
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
d=Path(sys.argv[1]);out=d/"role-volumes-v2";out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=H/"current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz"
with np.load(source) as z:p=z["points_reference_world_m_f64"].astype(np.float32);t=z["volume_tetrahedra"];mat=z["material_index"]
with np.load(d/"candidate-volume.npz") as z:nt=z["volume_tetrahedra"];nm=z["material_index"];off=z["replacement_parent_offsets"];parents=z["replacement_parent_cell_rows"]
report={"status":"RUNNING","physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN","roles":{}}
rp=out/"report.json"
def save():rp.write_text(json.dumps(report,indent=2)+"\n")
save()
bases={}
for role in ["Skin","Fat"]:
 path=H/f"current-native-1024-root/canonical834-current-base/{role.lower()}-canonical-parent-union.npz"
 with np.load(path) as z:pp=z["points_world_m_f32"];tt=z["tetrahedra"]
 used=np.unique(tt);bases[role]={"path":path,"p":pp,"t":tt,"used":used,"to_joint":np.full(len(pp),-1,np.int64)}
node_table=H/"current-role-p1-support-938-1025-active-source-map/actual-001/217-handoff/skin-fat-combined-node-map.csv.gz"
with gzip.open(node_table,"rt",newline="") as f:
 for row in csv.DictReader(f):
  joint=int(row["joint_node_id"])-1
  for partition,key in [(row["source_partition"],"source_tetgen_node_id"),(row["merge_peer_partition"],"merge_peer_tetgen_node_id")]:
   if not partition:continue
   role=partition.capitalize();base=bases[role]
   # 938 preserves canonical source row + 1 as the TetGen node ID;
   # active rows are sparse, not renumbered in ascending active order.
   point=int(row[key])-1
   assert 0<=point<len(base["p"])
   assert np.searchsorted(base["used"],point)<len(base["used"])
   assert base["used"][np.searchsorted(base["used"],point)]==point
   assert base["to_joint"][point] in [-1,joint]
   base["to_joint"][point]=joint
spec=importlib.util.spec_from_file_location("oldtop",H/"transfer_surface_contractions_indexed_379_root.py");top=importlib.util.module_from_spec(spec);spec.loader.exec_module(top)
for role,material in [("Skin",15),("Fat",16)]:
 base=bases[role];forward=base["to_joint"];used=base["used"];assert np.all(forward[used]>=0)
 original_rows=np.flatnonzero(mat==material)
 assert len(original_rows)==len(base["t"])
 assert np.array_equal(np.sort(forward[base["t"]],axis=1),np.sort(t[original_rows],axis=1))
 inv=np.full(len(p),-1,np.int64);inv[forward[used]]=used
 rows=np.flatnonzero(nm==material);tt=inv[nt[rows]];assert np.all(tt>=0)
 det,q=top.m.measurements(base["p"],tt)
 mismatch=np.linalg.norm(base["p"][used].astype(float)-p[forward[used]].astype(float),axis=1)
 offsets=np.r_[0,np.cumsum(np.diff(off)[rows])]
 cp=np.concatenate([parents[off[i]:off[i+1]] for i in rows]);lookup=np.full(len(mat),-1,np.int64);lookup[original_rows]=np.arange(len(original_rows));cp=lookup[cp];assert np.all(cp>=0)
 # This is a direct role volume proposal. Parent CSR is in its authenticated
 # immediate canonical role-volume cell-row domain, not source330 ancestry.
 target=out/f"{role.lower()}-volume-proposal.npz"
 boundary=top.boundary(tt)
 direct=cp[offsets[:-1]]
 retained_mask=(np.diff(offsets)==1)&np.all(tt==base["t"][direct],axis=1)
 ri=np.flatnonzero(retained_mask);ri=ri[np.argsort(direct[ri],kind="stable")]
 changed=np.flatnonzero(~retained_mask);order=np.r_[ri,changed];ordered=tt[order]
 retained=direct[ri];inverse=np.full(len(base["t"]),-1,np.int64);inverse[retained]=np.arange(len(retained))
 replacement_offsets=np.r_[0,np.cumsum(np.diff(offsets)[changed])]
 replacement_rows=np.concatenate([cp[offsets[i]:offsets[i+1]] for i in changed])
 np.savez_compressed(target,points_world_m=base["p"],points_world_m_f32=base["p"],tetrahedra=ordered,material_indices=np.full(len(tt),material,np.int32),boundary_faces=boundary,
  current_geometry_parent_path=np.array(str(base["path"])),current_geometry_parent_sha256=np.array(sha(base["path"])),
  replacement_parent_offsets=replacement_offsets,replacement_parent_cell_rows=replacement_rows,
  retained_input_cell_rows=retained,replacement_tetrahedra=tt[changed],
  output_cell_to_input_cell=direct[order],input_cell_to_output_cell=inverse,
  output_point_to_input_point_row=np.arange(len(base["p"]),dtype=np.int64),
  output_point_parent_offsets=np.zeros(len(base["p"])+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
 report["roles"][role]={"path":str(target),"sha256":sha(target),"cells":len(tt),"active_nodes":len(np.unique(tt)),"nonpositive_cells":int((det<=0).sum()),"native_quality_failures":int((~top.m.passes(det,q)).sum()),"minimum_volume_m3":float(det.min()/6),"minimum_quality":float(q.min()),"joint_source_coordinate_difference_rms_m":float(np.sqrt(np.mean(mismatch**2))),"joint_source_coordinate_difference_max_m":float(mismatch.max()),"original_ordered_cell_domains_proven":True,"immediate_parent_references":len(cp)}
 save()
report["status"]="SOURCE_ROLE_VOLUME_BRIDGE_CHECKED_NOT_PHYSICAL_ADMISSION";save()
print(json.dumps(report,indent=2))
