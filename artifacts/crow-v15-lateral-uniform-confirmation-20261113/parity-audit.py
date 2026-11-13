from pathlib import Path
import gzip,json,sys
A=Path('artifacts/crow-v15-lateral-uniform-confirmation-20261113');P=A/sys.argv[1];seed=next(x for x in ('20261126','20261127') if x in P.name)
R=Path('artifacts/crow-v15-uniform-flight-24s-screen-20261113')/('push-wing-012-'+seed+'-rebound-uniform')
G=Path('artifacts/crow-v15-lateral-gain-screen-20261113/lateral-gains-20261126-rebound-gains')
def read(p,lanes):
 d={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for line in f:
   if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
   x=json.loads(line.split(' ',1)[1]);key=(x['lane'],x['step'])
   if key[0] in lanes:
    if key in d:raise ValueError('duplicate accepted state')
    d[key]={k:x[k] for k in ('q','v','policy_action','contacts')}
 return d
actual=read(P,{0,1,6,7});base=read(R,{0,1});meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];oldmeta=json.loads((R/'run.stdout').read_text())['first_episode_navigation'];rows=[]
for e in (0,1):
 end=oldmeta['steps_by_environment'][e];missing=[t for t in range(1,end+1) if (e,t) not in base or (e,t) not in actual];different=[t for t in range(1,end+1) if (e,t) in base and (e,t) in actual and base[e,t]!=actual[e,t]]
 rows.append({'lane':e,'kind':'no-push-wing-control','reference_end':end,'new_end':meta['steps_by_environment'][e],'missing_ticks':missing,'different_ticks':different,'exact':end==meta['steps_by_environment'][e] and not missing and not different})
if seed=='20261126':
 gain=read(G,{6,7})
 for e in (6,7):
  missing=[t for t in range(1,1201) if (e,t) not in gain or (e,t) not in actual];different=[t for t in range(1,1201) if (e,t) in gain and (e,t) in actual and gain[e,t]!=actual[e,t]]
  rows.append({'lane':e,'kind':'selected-gain-full-episode','missing_ticks':missing,'different_ticks':different,'exact':not missing and not different})
a=json.loads((P/'analysis.json').read_text());candidates=[r for r in a['rows'] if r['lane']>=2]
result={'seed':seed,'rows':rows,'all_required_exact':all(x['exact'] for x in rows),'all_six_preliminary_engineering_pass':len(candidates)==6 and all(x.get('preliminary_engineering_pass',False) for x in candidates),'validated_six_lane_screen_pass':len(candidates)==6 and all(x.get('preliminary_engineering_pass',False) for x in candidates) and all(x['exact'] for x in rows),'scope':'Used-seed obstacle-free screening; not flight qualification or freeze.'}
(P/'parity-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
