import csv, hashlib, json
from pathlib import Path
root=Path('/Users/home/cardboard-evidence-20261006')
out=root/'final-cycle-comparison.json'
if out.exists(): raise FileExistsError(out)
result={'schema':'numi.cardboard.final-cycle-comparison.v1','physical_validation':False,'protocol':'32 load, 16 hold, 32 unload, 64 settling; dt 25 us; both grips clamped after return to zero','plastic_observable':'maximum absolute value of the six named ep components; cells above 1e-6 is a numerical diagnostic','runs':[]}
for name in ['finite-glue-default-final-001','finite-glue-elastic-final-001','finite-glue-unused-angle-final-001']:
 p=root/name
 run={'name':name,'result':json.loads((p/'result.json').read_text()),'inputs':{},'states':[]}
 def read(path):
  raw=path.read_bytes();run['inputs'][path.name]=hashlib.sha256(raw).hexdigest();return raw
 rows=list(csv.DictReader(read(p/'observations.csv').decode().splitlines()))
 run['observation_rows']=len(rows)
 run['all_status_success']=all(int(r['status_code'])==0 for r in rows)
 run['all_step_accepted']=all(int(r['step_accepted'])==1 for r in rows)
 run['final_observations']=rows[-2:]
 for path in sorted(p.glob('material_state_*.json')):
  s=json.loads(read(path)); mats={m['index']:m for m in s['materials']}
  maxima=[]; per={}
  for t in s['tetrahedra']:
   m=mats[t['material_index']];ep=[abs(v) for k,v in zip(m['state_names'],t['state']) if k in {'ep11','ep22','ep33','ep23','ep13','ep12'}]
   if not ep:continue
   val=max(ep);maxima.append(val);per.setdefault(m['name'],[]).append(val)
  run['states'].append({'file':path.name,'paper_cells_with_plastic_state':len(maxima),'max_abs_plastic_strain':max(maxima,default=0),'cells_above_1e_6':sum(v>1e-6 for v in maxima),'materials':{k:{'max_abs_plastic_strain':max(v),'cells_above_1e_6':sum(x>1e-6 for x in v)} for k,v in per.items()}})
 result['runs'].append(run)
out.write_text(json.dumps(result,indent=2)+'\n')
for r in result['runs']:
 print(r['name'],r['result']['status'],r['observation_rows'],r['all_step_accepted'])
 for s in r['states']:
  if '000144' in s['file']:print(s)
 print('final motion',[(x['arm'],x['max_free_displacement_m'],x['max_free_speed_m_s'],x['kinetic_energy_J']) for x in r['final_observations']])
