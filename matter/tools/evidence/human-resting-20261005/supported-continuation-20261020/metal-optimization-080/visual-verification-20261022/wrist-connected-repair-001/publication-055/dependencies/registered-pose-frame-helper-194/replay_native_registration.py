from pathlib import Path
import sys,json,hashlib
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
O=E/'registered-pose-frame-helper-194'
sys.path.insert(0,str(O/'tools'))
from rebind_human_skin_reference import decode,registered_positions_float32
S=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004'
payload_path=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(payload_path)=='4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5'
payload=decode(payload_path.read_bytes())
snapshot=json.loads((S/'registered-skin-poses.json').read_text())
assert sha(S/'registered-skin-poses.json')=='0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd'
csv=np.genfromtxt(S/'registered-skin.csv',delimiter=',',names=True)
assert np.array_equal(csv['vertex'],np.arange(payload['nv']))
expected=np.column_stack([csv[k] for k in ('wx','wy','wz')]).astype(np.float32)
current=registered_positions_float32(payload,snapshot,frame='current')
rest=registered_positions_float32(payload,snapshot,frame='rest')
err=np.linalg.norm(current.astype(float)-expected.astype(float),axis=1)
report={'status':'pass' if err.max()<=2e-6 else 'fail','scope':'source-only Float32 registration replay; no accepted FEM deformation, geometry or physiology qualification',
 'source_sha256':sha(payload_path),'poses_sha256':sha(S/'registered-skin-poses.json'),
 'native_csv_sha256':sha(S/'registered-skin.csv'),'source_vertices':payload['nv'],
 'current_max_error_m':float(err.max()),'current_rms_error_m':float(np.sqrt(np.mean(err**2))),
 'current_bitwise_equal_vertices':int(np.sum(np.all(current==expected,axis=1))),
 'rest_current_max_distance_m':float(np.linalg.norm(current.astype(float)-rest.astype(float),axis=1).max()),
 'current_finite':bool(np.isfinite(current).all()),'rest_finite':bool(np.isfinite(rest).all()),
 'helper_sha256':sha(O/'tools/rebind_human_skin_reference.py')}
(O/'native-registration-parity.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
assert report['status']=='pass',report
print(json.dumps(report,indent=2))
