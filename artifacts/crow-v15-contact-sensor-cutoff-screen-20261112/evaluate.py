from pathlib import Path
import json,statistics
A=Path('artifacts/crow-v15-contact-sensor-cutoff-screen-20261112');out={}
for seed in (20261126,20261127):
 d=json.loads((A/f'analysis-{seed}.json').read_text());p=json.loads((A/f'contact-analysis-{seed}.json').read_text());rows=[]
 for e in range(8):
  parity=p['lane_parity'][str(e)];same=parity=={'endpoints':0,'substeps':0};assert same,(seed,e,parity)
  if e<2:rows.append({'lane':e,'role':'no-sensor control','physics_parity':same});continue
  vals=[x for x in d['records'] if x['lane']==e];assert len(vals)==194
  errors=sorted(x['absolute_error_m_s'] for x in vals);by_step={}
  for x in vals:by_step.setdefault(x['step'],{})[x['side']]=x['sensor_m_s']
  assert len(by_step)==97 and all(set(x)=={'left','right'} for x in by_step.values())
  unique=len({(x['left'],x['right']) for x in by_step.values()});n=sum(x>.02 for x in errors);med=statistics.median(errors);cutoff=1e-7 if e%2 else 1e-9
  rows.append({'lane':e,'cutoff_ns':cutoff,'physics_parity':same,'samples':len(vals),'unique_sensor_pairs':unique,'median_error_m_s':med,'p95_error_m_s':errors[int(.95*(len(errors)-1))],'max_error_m_s':max(errors),'errors_over_0_02_m_s':n,'passes_provisional_new_cutoff_gate':(n==0 and med<=.002 and unique>=20) if cutoff==1e-7 else None})
 out[str(seed)]=rows
(A/'evaluation.json').write_text(json.dumps(out,indent=2)+'\n')
for seed,rows in out.items():
 for x in rows:print(seed,x)
