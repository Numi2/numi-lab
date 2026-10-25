from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from scipy import sparse
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');A=E/'connected-volume-repair-032';D=E/'connected-volume-repair-036'
sys.path.insert(0,str(D/'canonical/matter/tools'));import human_skin_fat_connected_repair as repair
z=np.load(A/'repair-input.npz');start=time.monotonic()
Y,r=repair.coupled_clearance_step(z['points_world_f32'],z['reference_points_m'],z['tetrahedra'],z['free_nodes'],sparse.load_npz(A/'contact-rows.npz'),z['contact_targets'],step_limit_mm=2.,time_limit_s=90.,lower_linear_matrix=sparse.load_npz(A/'layer-rows.npz'),lower_linear_targets=z['layer_targets'])
np.savez_compressed(D/'candidate.npz',points_world_f32=Y,tetrahedra=z['tetrahedra'],material_ids=z['material_ids'])
r.update({'elapsed_s':time.monotonic()-start,'candidate_sha256':hashlib.sha256((D/'candidate.npz').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((D/'canonical/matter/tools/human_skin_fat_connected_repair.py').read_bytes()).hexdigest(),'status':'CONNECTED_STATIC_CONSTRUCTION_NOT_PHYSICAL_ADMISSION','physical_qualification':False})
(D/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
