from pathlib import Path
from itertools import product
import hashlib,importlib.util,json,sys
import numpy as np
from directional_bounds636_v2 import directional_bounds
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546');D=N.parent
cand=N/'connected-topology-580/run-009-from772/lineage/candidate.npz';check=N/'connected-topology-580/run-009-from772/full-check/report.json';target=N/'hard-face-feasibility-634/prepared-from766-001/prepared/surface-targets/targets.npz';owner=D/'surface-witness-basis-408/propose_joint_clearance_witnesses_408.py';src=D.parent/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz';here=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('original_bounds408_real636',owner);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
x=np.load(cand,allow_pickle=False);pts=x['points_world_m'].astype(np.float64)*1000;ref,_=m.compose_source_reference(x,src);ref=ref*1000
tri=np.load(target,allow_pickle=False)['face_node_rows'];support=np.unique(tri);a=json.loads(check.read_text())['current_anchor_lineage'];assert a['pass'] and a['anchors']==237;anchors=np.asarray(a['current_output_rows'],np.int64)
oldlo,oldhi,_=m.source_safe_trust_bounds(pts,ref,support,anchors,4,12);lo,hi,r=directional_bounds(pts,ref,support,anchors,4,12)
assert np.all(lo<=oldlo+1e-12) and np.all(hi>=oldhi-1e-12)
lo=lo.reshape(-1,3);hi=hi.reshape(-1,3);max_step=0.;max_source=0.
for bits in product((0,1),repeat=3):
 d=np.where(np.asarray(bits)[None,:],hi,lo)
 max_step=max(max_step,float(np.linalg.norm(d,axis=1).max(initial=0.)))
 max_source=max(max_source,float(np.linalg.norm(pts[support]-ref[support]+d,axis=1).max(initial=0.)))
assert max_step<=4+1e-10 and max_source<=12+1e-10
out=here/'real-766-bounds-check-v2.json'
result={'candidate_sha256':sha(cand),'check_sha256':sha(check),'target_sha256':sha(target),'source_base_sha256':sha(src),'owner408_sha256':sha(owner),'directional_bounds636_sha256':sha(here/'directional_bounds636_v2.py'),'old_cube_contained_for_all_support_nodes':True,'eight_box_corners_checked':True,'max_box_corner_step_mm':max_step,'max_box_corner_source_distance_mm':max_source,**r}
out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(result))
