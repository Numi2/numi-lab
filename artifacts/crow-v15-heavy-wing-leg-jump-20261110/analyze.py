from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-heavy-wing-leg-jump-20261110')
def load(name):
 out=[{} for _ in range(8)]
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);out[x['lane']][x['step']]=x
 return out
def loaded(x):return any('locomotion_ground' in (c['a_name'],c['b_name']) and c['impulses'][0]>1e-7 for c in x['contacts'])
rows=[]
for seed in (38942573,38942609):
 p=load(str(seed)+'-push-trigger');n=load(str(seed)+'-control-trigger')
 for lane in range(8):
  a,b=p[lane],n[lane];common=sorted(set(a)&set(b));first=next((t for t in common if a[t]['q']!=b[t]['q'] or a[t]['v']!=b[t]['v']),None);window=[t for t in common if 648<=t<=670];assert all(t in a and t in b for t in range(648,671));air=next((t for t in window if not loaded(a[t]) and loaded(b[t])),None);mx=max(window,key=lambda t:a[t]['q'][2]-b[t]['q'][2]);rows.append({'seed':seed,'lane':lane,'first_qv_divergence':first,'first_push_unloaded_with_control_loaded':air,'max_altitude_difference_step':mx,'max_altitude_difference_m':a[mx]['q'][2]-b[mx]['q'][2],'push_altitude_at_max':a[mx]['q'][2],'control_altitude_at_max':b[mx]['q'][2],'push_vz_at_first_unload':a[air]['v'][2] if air else None,'control_vz_at_first_unload':b[air]['v'][2] if air else None})
r={'rows':rows,'unloaded_while_control_loaded_count':sum(x['first_push_unloaded_with_control_loaded']!=None for x in rows),'scope':'Active flapping amplitude held zero until 13.4 s; passive aero and joint drives remain active. Ground contacts are final-substep samples, not whole-interval forces.'};(A/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print('lanes',len(rows),'unloaded push while control loaded',r['unloaded_while_control_loaded_count'],'height delta range',min(x['max_altitude_difference_m'] for x in rows),max(x['max_altitude_difference_m'] for x in rows),'first',rows[0])
