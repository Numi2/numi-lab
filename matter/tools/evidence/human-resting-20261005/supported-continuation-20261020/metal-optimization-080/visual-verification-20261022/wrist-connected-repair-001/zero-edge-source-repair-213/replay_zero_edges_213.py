from pathlib import Path
import sys,json,hashlib,time
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
O=E/'zero-edge-source-repair-213';sys.path.insert(0,str(O/'tools'))
from human_skin_patch_geometry import collapse_zero_length_surface_edges
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=E/'bilateral-source-skin-cap-reconstruction-211/candidate-surface-unadmitted.npz';assert sha(p)=='9c33d81c4d16985051db78234e4e36d233f3130a02c60ffd8ca7f1d04cc76884'
s=time.monotonic();z=np.load(p);X=z['outer_world_actual007_m'];F=z['surface_faces'];g,a,r=collapse_zero_length_surface_edges(X,F,z['protected_cut_vertex_ids'])
y=np.load(E/'bilateral-source-skin-cap-reconstruction-212/candidate-surface-welded-unadmitted.npz')
assert len(F)-len(g)==60 and (a!=np.arange(len(a))).sum()==30
assert np.array_equal(X[g],X[F[r]])
# Independent implementation preserves the same retained face geometry and order.
assert np.array_equal(X[g],y['outer_world_actual007_m'][y['surface_faces']])
np.savez_compressed(O/'zero-edge-collapse-replay-213.npz',vertex_alias=a,source_face_rows=r,faces=g)
report={'source_sha256':sha(p),'geometry_helper_sha256':sha(O/'tools/human_skin_patch_geometry.py'),'tests':49,'removed_zero_area_faces':60,'explicit_vertex_aliases':30,'retained_triangles_coordinate_exact':True,'independent_212_geometry_parity':True,'wall_seconds':time.monotonic()-s,'physical_admission':False,'known_source_failure':'candidate212 exact changed-patch self audit has548 pairs; cleanup never establishes crossing clearance'}
(O/'report-213.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
