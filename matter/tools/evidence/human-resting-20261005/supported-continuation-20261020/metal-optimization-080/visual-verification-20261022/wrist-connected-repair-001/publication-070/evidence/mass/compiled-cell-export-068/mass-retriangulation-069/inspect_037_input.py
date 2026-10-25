import numpy as np, pathlib, csv
E=pathlib.Path(/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001)
z=np.load(E/E/connected-volume-refinement-037/run-006/candidate.npz); r=np.load(E/connected-volume-repair-032/repair-input.npz); ap=np.load(E/native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz)
C=E/connected-skin-parent-mass-transfer-067/compiled-cell-export-068/run-004/source-preflight/compiled-skin-fem-nodes.csv
rows=sorted(csv.DictReader(C.open()),key=lambda x:int(x[global_node])); n=len(rows)
ref=np.array([[float(x[k]) for k in (rest_x,rest_y,rest_z)] for x in rows],dtype=np.float32); cur=np.array([[float(x[k]) for k in (position_x,position_y,position_z)] for x in rows],dtype=np.float32)
print(F64prefix exact repair,np.max(np.abs(z[canonical_reference_points_f64][:n]-r[reference_points_m])),np.array_equal(z[canonical_reference_points_f64][:n],r[reference_points_m]))
print(native restF32 vs runrefF32 prefix,np.array_equal(ref,z[reference_points_f32][:n]),np.max(np.abs(ref-z[reference_points_f32][:n])))
print(native restF32 vs canonicalF64 cast,np.array_equal(ref,z[canonical_reference_points_f64][:n].astype(np.float32)),np.max(np.abs(ref-z[canonical_reference_points_f64][:n].astype(np.float32))))
print(currentprefix,np.array_equal(cur,z[current_points_f32][:n]))
print(parent map,np.array_equal(z[child_parent_rows],z[child_parent_cell_ids]))
changed=np.flatnonzero(np.array([not np.array_equal(np.sort(ap[old_tetrahedra][i]),np.sort(ap[tetrahedra][i])) for i in range(len(ap[tetrahedra]))])); p=z[child_parent_rows]; cnt=np.bincount(p[np.isin(p,changed)],minlength=len(ap[tetrahedra]))
print(changedAPparents,changed.tolist(),descendant total,int(np.isin(p,changed).sum()),per,{int(i):int(cnt[i]) for i in changed})
print(childparent range,int(p.min()),int(p.max()),cellid match,np.array_equal(z[child_parent_cell_ids],p))
