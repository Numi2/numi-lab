from pathlib import Path
import csv,hashlib,json,sys,numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');sys.path.insert(0,str(R/'tools'))
from human_skin_patch_geometry import select_tetrahedra_by_surface_parent,tetrahedral_region_boundary
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
D=E/'ankle-parent-tet-refinement-026/generated-068-local-tet-cavity';G=E/'wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001/generated'
def read(f):
 with f.open() as h:return list(csv.DictReader(h))
faces=read(G/'faces.csv');tets=read(G/'tetrahedra.csv');prior=np.load(D/'cavity-d36-parent-matched-4-rings.npz',allow_pickle=False)
fp=np.array([int(r['parent_surface_face']) for r in faces]);tp=np.array([int(r['parent_surface_face']) for r in tets])
selected_rows=np.flatnonzero(np.isin(fp,prior['selected_parent_surface_face_ids']))
selected=select_tetrahedra_by_surface_parent(fp,selected_rows,tp)
np.testing.assert_array_equal(selected,prior['selected_global_tet_ids'])
T=np.array([[int(tets[i][k]) for k in ['a','b','c','d']] for i in selected]);M=np.array([int(tets[i]['material']) for i in selected])
out=tetrahedral_region_boundary(T,M)
def cyclic(f):return min(tuple(f),tuple(np.roll(f,1)),tuple(np.roll(f,2)))
assert {cyclic(f) for f in out['exterior_faces']}=={cyclic(f) for f in prior['external_boundary_oriented_global_node_ids']}
assert {tuple(sorted(f)) for f in out['interface_faces']}=={tuple(sorted(f)) for f in prior['internal_skin_fat_interface_global_node_ids']}
report={'scope':'Shared parent-identity and oriented-volume-boundary regression on actual D36 ankle cavity; no geometry repair or physical admission.',
 'selected_surface_rows':len(selected_rows),'selected_tets':len(selected),'exterior_faces':len(out['exterior_faces']),'interface_faces':len(out['interface_faces']),
 'oriented_exterior_and_interface_match':True,'physical_admission':False,
 'input_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [G/'faces.csv',G/'tetrahedra.csv',D/'cavity-d36-parent-matched-4-rings.npz',R/'tools/human_skin_patch_geometry.py']}}
p=D/'shared-boundary-owner-d36-regression.json'
with p.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
