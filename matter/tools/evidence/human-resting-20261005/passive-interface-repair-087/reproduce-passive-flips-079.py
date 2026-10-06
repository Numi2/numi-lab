"""Reproduce the four retained offline flips with their frozen helper."""
from pathlib import Path
import hashlib, importlib.util, json, numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005')
out=E/'passive-flip-reproduction-079';out.mkdir(exist_ok=False)
helper=E/'passive-interface-helper-069.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(helper)=='6fa55a8e1f730c17026d08365eaf63295e4e3414de01a813bbdb9ec6919286d6'
spec=importlib.util.spec_from_file_location('frozen_helper_069',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
prepared={}
for sid in [3,13]:
    data=np.load(E/f'passive-organ-interface-conditioned-068/surface-{sid}.npz')
    v=data['vertices'];f=data['faces']
    prepared[sid]=(np.column_stack((v,np.zeros_like(v))),f.copy(),{i:[i] for i in range(len(f))},{})
result=module.improve_registered_surface_sliver_faces(prepared,minimum_altitude_m=1.2e-6,maximum_nonplanarity_m=1e-7)
checks=[]
for sid,(v,f,_,_) in prepared.items():
    reference=E/f'passive-organ-interface-flipped-069/surface-{sid}.npz';data=np.load(reference)
    checks.append({'stable_id':sid,'retained_npz_sha256':sha(reference),'positions_bitwise_equal':v[:,:3].astype('<f4').tobytes()==data['vertices'].astype('<f4').tobytes(),
                   'faces_bitwise_equal':f.astype('<i4').tobytes()==data['faces'].astype('<i4').tobytes()})
assert result['flip_count']==4 and all(row['positions_bitwise_equal'] and row['faces_bitwise_equal'] for row in checks)
report={'scope':'Offline reproduction of retained step069; no new geometry produced or promoted.',
        'driver_sha256':sha(__file__),'helper_sha256':sha(helper),'flips':result,'checks':checks}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(checks),flush=True)
