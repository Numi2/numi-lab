from pathlib import Path
import numpy as np,json,sys,hashlib,importlib.util
w=Path("/Users/numi/Work/human-integrated-gaps-20261104")
h=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
d=Path(sys.argv[1]);r=json.loads((d/"report.json").read_text())
with np.load(h/"current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz") as z:oldp=z["points_reference_world_m_f64"].astype(np.float32);oldmat=z["material_index"]
with np.load(d/"candidate-volume.npz") as z:
 p=z["points_reference_world_m_f64"];t=z["volume_tetrahedra"];mat=z["material_index"];off=z["replacement_parent_offsets"];parents=z["replacement_parent_cell_rows"]
assert np.array_equal(oldp.astype(float),p)
assert np.all(np.isfinite(p)) and t.min()>=0 and t.max()<len(p)
assert len(off)==len(t)+1 and off[0]==0 and off[-1]==len(parents) and np.all(np.diff(off)>0)
assert np.all(parents>=0) and parents.max()<len(oldmat)
# Parent material membership is checked per CSR segment without losing duplicates.
assert np.all(oldmat[parents]==np.repeat(mat,np.diff(off)))
assert len(np.unique(np.sort(t,axis=1),axis=0))==len(t)
spec=importlib.util.spec_from_file_location("top",h/"transfer_surface_contractions_indexed_379_root.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
boundary=m.boundary(t)
# Prove complete oriented volume-boundary agreement with all emitted O/I/D and unchanged walls.
with np.load(h/"current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz") as z:ot=z["volume_tetrahedra"]
with np.load(h/"regional-compensation-1061-root/memory-performance-123/rejection-recovery-125/connected-conditioning-134/fixed-boundary-volume-proposal.npz") as z:baseline=np.concatenate([ot[z["retained_input_cell_rows"]],z["replacement_tetrahedra"]])
quot=np.arange(len(p))
for op in r["operations"]:quot[op["drop"]]=op["keep"]
expected=quot[m.boundary(baseline)];expected=expected[~np.any(np.diff(np.sort(expected,axis=1),axis=1)==0,axis=1)]
assert np.array_equal(m.boundary_identity(boundary),m.boundary_identity(expected))
# Exact dyadic determinant signs of every changed retained cell.
changed_nodes={o["keep"] for o in r["operations"]}
rows=np.flatnonzero(np.isin(t,list(changed_nodes)).any(1))
vv=np.unique(t[rows]);lut={int(i):[float(c).as_integer_ratio() for c in oldp[i]] for i in vv}
exponent=max(den.bit_length()-1 for point in lut.values() for num,den in point)
ip={i:[num<<(exponent-(den.bit_length()-1)) for num,den in v] for i,v in lut.items()}
def exact(tet):
 a,b,c,e=[ip[int(i)] for i in tet];b=[v-u for u,v in zip(a,b)];c=[v-u for u,v in zip(a,c)];e=[v-u for u,v in zip(a,e)]
 return b[0]*(c[1]*e[2]-c[2]*e[1])-b[1]*(c[0]*e[2]-c[2]*e[0])+b[2]*(c[0]*e[1]-c[1]*e[0])
assert all(exact(c)>0 for c in t[rows])
surfaces={}
for role in "OID":
 with np.load(d/f"candidate-{role}-typed-current.npz") as z:
  n=len(z["faces"])
  for ok,lanes in [("quality_star_patch_union_offsets",["quality_star_patch_union_face_source_"+k for k in ["layer","body","bone","face_id","original_id","identity0","identity1"]]),("face_ancestry_offsets",["face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows"])]:
   o=z[ok];assert len(o)==n+1 and o[0]==0 and np.all(np.diff(o)>0)
   assert all(len(z[k])==o[-1] for k in lanes)
  surfaces[role]={"faces":n,"points":len(z["points_world_m_f32"])}
out=dict(status="PASS_INDEPENDENT_LOCAL_TOPOLOGY_TYPED_CSR_AND_EXACT_CELL_CHECKS",physical_admission=False,native_steps=0,all_coordinates_exact=True,complete_oriented_boundary_matches_volume_quotient=True,parent_materials_separated=True,duplicate_cells=0,changed_cells_exactly_positive=len(rows),source_cell_ancestry_entries=len(parents),surfaces=surfaces,remaining=["all185 context and contained solid checks","conservative mass and P1/native compiler","whole native scene physical and performance comparison"])
(d/"independent-checks.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
