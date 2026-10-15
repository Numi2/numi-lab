from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

base=Path('/Users/home/cardboard-implementation-20261014')
prior=base/'analysis/medium-tangent-state-sensitivity-1634-001'
out=base/'analysis/medium-finer-fp64-fd-003'
out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bindings=[prior/'report.json',prior/'manifest.json',prior/'replay.py',
    base/'analysis/retained-paper-allpoint-reference-296-001/reference.npz',
    base/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py']
prediction=dict(schema='numi.cardboard.bounded-fp64-derivative-prediction.v1',
    question='Do smaller FP64 perturbations resolve the previously curvature-limited two medium matrices?',
    points=[1634,1638],epsilons=[1e-8,1e-9,1e-10,1e-11,1e-12],
    stopping='One complete declared epsilon sweep at both retained states; no failed point exclusion.',
    gate='relative full-matrix error <=1e-6 against independent MP90; record actual realized input direction error and endpoint branches. Near-zero component and noise-floor gates remain separate.',
    interpretation='Exploratory reference accuracy experiment motivated by separately retained MP90 endpoint sweep; no physical steps.',
    inputs=[dict(path=str(p),sha256=sha(p)) for p in bindings])
(out/'prediction.json').write_text(json.dumps(prediction,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('reference',prior/'replay.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
data=json.loads((prior/'report.json').read_text())
npz=np.load(bindings[3]);keymap={tuple(k):i for i,k in enumerate(npz['keys'])}
results=[]
for row in data['results']:
    tet=row['point']['tet_id'];idx=keymap[(0,tet,1)]
    p=np.asarray(npz['parameters'][idx],dtype=np.float32).astype(float)
    f=np.asarray(row['point']['input_words']['F_widened']).reshape(3,3)
    old=np.asarray(npz['accepted'][idx],float)
    mp=np.asarray(row['nine_direction_check']['mp90_tangent_matrix'])
    sweeps=reference.tangent_matrix_finite_difference_fp64(p,f,old,prediction['epsilons'])
    for result in sweeps:
        matrix=np.asarray(result['matrix'])
        result['relative_matrix_error']=float(np.linalg.norm(matrix-mp)/np.linalg.norm(mp))
        eps=result['epsilon']
        realized=np.stack([(((f+eps*h)-(f-eps*h))/(2*eps)).reshape(9)
            for h in np.eye(9).reshape(9,3,3)],axis=1)
        result['realized_basis_relative_error']=float(np.linalg.norm(realized-np.eye(9))/3)
    results.append(dict(tet_id=tet,results=sweeps))
report=dict(schema='numi.cardboard.bounded-finer-fp64-fd.v1',results=results,
    absolute_noise_floor=None,physical_steps=0,physical_validation=False,
    limits=['Two medium centers only. A resolved matrix norm comparison does not establish componentwise near-zero accuracy.',
        'The existing safeguarded FP64 projection and criteria are unchanged; endpoint results do not qualify native FP32 arithmetic.'])
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(out/'run.py').write_bytes(Path(__file__).read_bytes())
(out/'README.md').write_text('# Finer FP64 finite differences\n\n'
    'Repeats the two retained medium-state sweeps at1e-8 through1e-12 '
    'dimensionless F using the unchanged independent safeguarded FP64 projector. '
    'The prediction is retained before execution. Exact basis realization, every '
    'endpoint branch and convergence record remain in the report; no point is '
    'excluded. This is a bounded numerical reference experiment.\n')
(out/'manifest.json').write_text(json.dumps(dict(inputs=prediction['inputs'],artifacts={
    p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()}),indent=2)+'\n')
print(json.dumps([{ 'tet':r['tet_id'],'sweep':[{ 'epsilon':v['epsilon'],
    'error':v['relative_matrix_error'],'input_error':v['realized_basis_relative_error']}
    for v in r['results']]} for r in results],indent=2))
