from pathlib import Path
import sys,numpy as np,json,hashlib
from scipy import sparse
D=Path('/Users/numi/Work/numi-human-connected-surface-069');E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
M=D/'publication-074/canonical/matter/tools';sys.path.insert(0,str(M))
sys.path.insert(0,str(E/'E/accepted-transfer-row-adapter-075'))
from load_transfer_075 import load_transfer075
from human_skin_fat_embedded_surface import MappedSkinSurface
K=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001'
Z=K/'candidate-tetrahedra-compacted.npz';z=np.load(Z);O=z['current037_points_m_f32'];W=sparse.load_npz(K/'transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz')
Q=E/'accepted-transfer-row-export-075/run-001/source-preflight'
_,origin,_,faces=load_transfer075(Q/'accepted-transfer-rows.csv',Q/'accepted-coarse-reference.csv',Q/'accepted-coarse-faces.csv',E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin',
 expected_sha256={'transfer_rows':'71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14','coarse_reference':'9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2','coarse_faces':'7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2','detail_payload':'7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'})
S=MappedSkinSurface(O,origin,faces,W,max_dihedral_change_degrees=15.)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'construction_bound_degrees':15.,'bound_is_anatomical_acceptance':False,'source_sha256':{'embedded_owner':sha(M/'human_skin_fat_embedded_surface.py'),'compacted_mesh':sha(Z),'mapped_matrix':sha(K/'transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz')},'baseline':S.report(O),'candidates':{},'physical_steps':0}
assert report['baseline']['mapped_skin_dihedral_abs_change_max_degrees']==0.
for name,p in [('091',D/'whole-reference-tetgen-084-bounded/connected-clearance-091/run-001/iteration-004/candidate.npz'),('093',D/'closest-exit-clearance-093/run-001/iteration-004/candidate.npz')]:
 report['candidates'][name]={'candidate_sha256':sha(p),**S.report(np.load(p)['points_world_f32'])}
assert abs(report['candidates']['091']['mapped_skin_dihedral_abs_change_max_degrees']-76.5408)<.01
P=D/'dihedral-real-map-check-096';P.mkdir(exist_ok=True);(P/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
