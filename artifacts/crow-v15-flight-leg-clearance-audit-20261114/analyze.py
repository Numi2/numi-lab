from pathlib import Path
import json
A=Path('artifacts/crow-v15-flight-leg-clearance-audit-20261114')
meta=json.loads((A/'inputs.json').read_text());poses={}
for line in (A/'output.tsv').read_text().splitlines():
 z=line.split()
 if z and z[0]=='body':poses.setdefault(int(z[2]),{})[z[3]]=[float(v) for v in z[4:]]
rows=[]
for r in meta['records']:
 p=poses[r['id']]; toes={n:v[2] for n,v in p.items() if '_toe_' in n};feet={n:v[2] for n,v in p.items() if n.endswith('_foot')}
 assert len(toes)==16 and len(feet)==2
 rows.append({**r,'lowest_toe_origin_z_m':min(toes.values()),'lowest_foot_origin_z_m':min(feet.values()),'lowest_toe':min(toes,key=toes.get)})
base=next(r for r in rows if r['delta']==[0.,0.,0.])
for r in rows:r['toe_origin_raise_vs_original_m']=r['lowest_toe_origin_z_m']-base['lowest_toe_origin_z_m']
out={'source_root':meta['root_q'],'original':base,'best_by_lowest_toe_origin':max(rows,key=lambda r:r['lowest_toe_origin_z_m']),'rows':rows,'limitations':'Static rigid FK at one retained airborne state, toe link origins only; this is not contact surface clearance, a reachable trajectory, or a dynamic trial.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('original','best_by_lowest_toe_origin','limitations')},indent=2))
