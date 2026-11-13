from pathlib import Path
import gzip,json
A=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-consistent-wing-drive-20261113');analysis=json.loads((A/'analysis.json').read_text());p=A/'push-wing-20261126-rebound-consistent-rebound/run.stderr.gz';counts={}
com={(x['arm'],x['lane']):x for x in json.loads((A/'com-analysis.json').read_text())['rows']}
with gzip.open(p,'rt') as f:
 for l in f:
  z=l.split()
  if z and z[0]=='CROW_TWO_TOE_SUBSTEP':
   t,k,e=map(int,z[1:4]);
   if 340<=t<=347:
    v=list(map(float,z[6:11]));counts[e,t]=counts.get((e,t),False) or sum(v[2:4])>1e-7
rows=[]
for r in analysis['rows'][2:]:
 e=r['lane'];p=r['push'];release=p['first_three_unloaded'];supported=all(counts[e,t] for t in range(340,348));strict=(r['impulse_348_355_delta_ns']>.05 and r['early_impulse_delta_ns']>.05 and supported and release is not None and not p['pre_release_unloaded_intervals'] and not p['later_supported_control_intervals'] and com['push',e]['com_peak_rise_from_350_m']>=.05 and p['nonfoot_impulse_348_400_ns']==0 and p['toe_speed_threshold_substeps']<=1)
 rows.append({'lane':e,'support_340_347':supported,'strict_pass':strict,'first_release':release,'interruptions_before_release':p['pre_release_unloaded_intervals'] if release is not None else None,'later_recontacts':p['later_supported_control_intervals'] if release is not None else None})
(A/'strict-gates.json').write_text(json.dumps({'scope':'Predeclared development screen, not biological or flight qualification.','rows':rows},indent=2)+'\n')
for r in rows:print(r['lane'],'support',r['support_340_347'],'release',r['first_release'],'pre gaps',r['interruptions_before_release'],'recontacts',(r['later_recontacts'] or [])[:8],'pass',r['strict_pass'])
