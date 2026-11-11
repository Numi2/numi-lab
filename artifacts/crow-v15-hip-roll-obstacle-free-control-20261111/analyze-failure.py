from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-hip-roll-obstacle-free-control-20261111');base=json.loads((A/'flight-analysis.json').read_text());ends={r['lane']:r['steps'] for r in base.get('lanes',[])};last={};fatal={};caps=collections.defaultdict(list)
with gzip.open(A/'flight.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_FLIGHT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);e=x['lane'];t=x['step']
   if e in ends and t<=ends[e]:last[e]=x
  elif l.startswith('CROW_CONTACT_TRACE '):
   x=json.loads(l.split(' ',1)[1]);e=x['lane']
   if e not in fatal and x['step']<=ends.get(e,0)+1:fatal[e]=x
  elif l.startswith('CROW_TOE_V_SUBSTEP '):
   a=l.split();t,sub,e=map(int,a[1:4]);v=list(map(float,a[4:]));
   if t<=ends.get(e,0)+1 and max(map(abs,v))>=11.99:caps[e].append((t,sub))
rows=[]
for e,r in sorted((x['lane'],x) for x in base.get('lanes',[])):
 x=fatal.get(e);s=last.get(e);row={'lane':e,'first_episode_steps':r['steps'],'termination_reason':r['termination_reason'],'physical_failure':r['physical_failure'],'latest_accepted_sample_step':s['step'] if s else None,'latest_sample_q':s['q'] if s else None,'latest_sample_v':s['v'] if s else None,'terminal_contact_step':x['step'] if x else None,'terminal_contacts':x.get('contacts') if x else None,'terminal_pre_q':x.get('pre_q') if x else None,'terminal_pre_v':x.get('pre_v') if x else None,'toe_cap_substep_count_before_or_at_first_end':len(caps[e]),'toe_cap_last_observer_step':caps[e][-1][0] if caps[e] else None};rows.append(row)
out={'rows':rows,'limitations':'Named contacts are endpoint contacts at termination, not necessarily the first fatal contact. Toe sample cutoff allows one observer step for initialization offset; exact phase and within-substep extrema remain unresolved. Obstacle-free course; no autonomous or landing inference.'};(A/'failure-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{'lane':r['lane'],'steps':r['first_episode_steps'],'terminal_contacts':r['terminal_contacts'],'toe_caps':r['toe_cap_substep_count_before_or_at_first_end']} for r in rows],indent=2))
