from pathlib import Path
import numpy as np,json,hashlib
d=Path('/Users/numi/Work/numi-human-connected-surface-069');b=d/'D/postfit-backtrack-424/run-001/backtrack-373/pass-013'
p=d/'D/local-two-parent-retriangulation-425/run-001-pass013';p.mkdir(parents=True,exist_ok=False)
z=np.load(b/'candidate.npz');pairs=np.load(b/'audit-130/surface-self-pairs.npy');seeds=np.unique(pairs).astype(int);details=np.unique(z['parent_fine_detail_face_row'][seeds]).astype(int)
assert len(pairs)==10 and np.all(details>=0)
(p/'seed-faces.json').write_text(json.dumps(seeds.tolist())+'\n');(p/'source-details.json').write_text(json.dumps(details.tolist())+'\n')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
a=['/Users/n/numi-human-prep-venv-20261005/bin/python',str(d/'D/local-two-parent-retriangulation-409-next/exact_global_incremental_flip_search_409_next.py'),'--candidate',str(b/'candidate.npz'),'--source-detail-rows',str(p/'source-details.json'),'--seed-face-rows',str(p/'seed-faces.json'),'--output-dir',str(p/'search'),'--max-accept','6','--max-trials-per-round','32','--max-seconds','60']
pins={str(f):sha(f)for f in [b/'candidate.npz',b/'audit-130/summary.json',b/'audit-130/surface-self-pairs.npy',p/'seed-faces.json',p/'source-details.json',Path(a[1]),Path(a[1]).with_name('f32_area_gate_409_next.py'),Path(a[1]).with_name('incremental_exact_flip_check_408.py')]}
(p/'launch.json').write_text(json.dumps(dict(argv=a,pins=pins,native_admission=False,accepted_physical_steps=0),indent=2)+'\n');print(json.dumps({'pairs':pairs.tolist(),'details':details.tolist(),'plan':str(p/'launch.json')}))

