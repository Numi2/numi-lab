from pathlib import Path
import csv,json,hashlib,sys,time
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
O=E/'native-app-main-header-integration-151/repair-batch-selection-014'
sys.path.insert(0,str(O/'tools'))
from human_skin_patch_geometry import partition_tetrahedral_repair_batch
G=E/'wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001/generated'
tr=list(csv.DictReader((G/'tetrahedra.csv').open()))
T=np.array([[int(r[k]) for k in ('a','b','c','d')]for r in tr])
F=list(csv.DictReader((G/'faces.csv').open()))
fr=np.array([[int(r[k]) for k in ('a','b','c')]for r in F])
stable=np.array([int(r['stable_source_face'])for r in F])
tstable=np.array([int(r['stable_source_face'])for r in tr])
H=E/'ankle-parent-tet-refinement-026/generated-157-d36-actual007-O-affected-face-rerun/actual007-O-affected-faces.json'
hit=sorted({int(h['d36_face_row'])for h in json.loads(H.read_text())['intersections']})
B=E/'ankle-parent-tet-refinement-026/generated-210-independent-cavity-batch'
m=json.loads((B/'cavity-batch-manifest.json').read_text())
sets=[];moved=[]
for c in m['cavities']:
 z=np.load(c['actual007_candidate_npz'])
 sets.append(z['selected_global_tet_ids'])
 selected_stable=set(z['selected_stable_source_face_ids'].tolist())
 fh=[i for i in hit if stable[i]in selected_stable]
 rows=np.unique(fr[fh].ravel())
 moved.append(np.r_[rows,rows+2411,rows+4822])
start=time.perf_counter(); groups,shared=partition_tetrahedral_repair_batch(T,sets,moved)
elapsed=time.perf_counter()-start
assert len(groups)==len(sets)==10
for g in groups:
 assert len(g['seed_components'])==1
 assert np.array_equal(g['selected_global_tet_ids'],sets[g['seed_components'][0]])
claimed=set(np.concatenate([g['selected_global_tet_ids']for g in groups]).tolist())
for i in hit: assert set(np.flatnonzero(tstable==stable[i]).tolist())<=claimed
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'status':'PASS','scope':'unchanged cavity selection replay; not geometry repair or physical admission','cavity_count':len(groups),'selected_tetrahedra':len(claimed),'witnessed_surface_faces_covered':len(hit),'protected_unique_nodes':len(set(np.concatenate(moved).tolist())),'shared_fixed_stitch_pairs':len(shared),'selection_wall_seconds':elapsed,'inputs':{str(p):sha(p)for p in [G/'tetrahedra.csv',G/'faces.csv',H,B/'cavity-batch-manifest.json']},'helper_sha256':sha(O/'tools/human_skin_patch_geometry.py'),'test_sha256':sha(O/'tests/test_human_skin_patch_geometry.py')}
(O/'actual-d36-replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
