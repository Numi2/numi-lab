from pathlib import Path
import collections,gzip,json,math,statistics,sys
A=Path('artifacts/crow-v15-airborne-gust-screen-20261114');P=A/sys.argv[1];ref={(x['lane'],x['step']):x for x in json.load(gzip.open('artifacts/crow-v15-flight-command-refactor-20261114/reference-states.json.gz','rt'))};states={};wrenches=collections.defaultdict(list);errors=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);k=(x['lane'],x['step'])
   if k in states:errors.append('duplicate state '+str(k))
   states[k]=x
  elif line.startswith('CROW_PRE_AERO_ROOT '):
   a=line.split();t,sub,e=map(int,a[1:4]);wrenches[e,t,sub].append(list(map(float,a[4:])))
  elif 'OBSERVER_ERROR' in line or line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
base=json.loads((P/'analysis.json').read_text());rows=[]
def sep(a,b):
 x,y=a[3:7],b[3:7];return 2*math.acos(max(-1,min(1,abs(sum(i*j for i,j in zip(x,y))))))
for e in range(8):
 missing=[(t,s) for t in range(695,711) for s in range(16) if len(wrenches[e,t,s])!=1];prefix=[]
 for t in range(1,691):
  a,b=states.get((e,t)),ref.get((e,t))
  if a is None or b is None or any(a[k]!=b[k] for k in ('q','v','policy_action','contacts')):prefix.append(t)
 pulse=[];badw=[];signy=1 if e%2==0 else -1;signx=1 if e&2==0 else -1
 for t in range(695,711):
  vals=[wrenches[e,t,s][0] for s in range(16) if len(wrenches[e,t,s])==1]
  if len(vals)!=16:continue
  if all(abs(x[1]-signy*.08)<1e-6 and abs(x[3]-signx*.004)<1e-6 for x in vals):pulse.append(t)
  elif not all(abs(x[1])<1e-6 and abs(x[3])<1e-6 for x in vals):badw.append(t)
 disturbance=[]
 for t in range(700,716):
  if (e,t) in states and (e,t) in ref:
   a,b=states[e,t]['v'],ref[e,t]['v'];disturbance.append((abs(a[1]-b[1]),abs(a[3]-b[3])))
 recovery=[]
 for t in range(850,901):
  if (e,t) in states and (e,t) in ref:
   a,b=states[e,t],ref[e,t];recovery.append((abs(a['q'][2]-b['q'][2]),sep(a['q'],b['q']),math.sqrt(sum((a['v'][j]-b['v'][j])**2 for j in range(3)))))
 means=[statistics.mean(x[i] for x in recovery) for i in range(3)] if len(recovery)==51 else None
 checks=base['rows'][e]['checks'];physical={k:v for k,v in checks.items() if k!='shared_inputs'}
 rows.append({'lane':e,'wrench_missing_or_duplicate':len(missing),'observed_pulse_steps':pulse,'invalid_wrench_steps':badw,'prefix_mismatch_count':len(prefix),'first_prefix_mismatch':prefix[:5],'max_velocity_or_roll_rate_response':max((max(x) for x in disturbance),default=None),'recovery_samples':len(recovery),'mean_height_attitude_velocity_deviation':means,'physical_gate_failures_excluding_expected_host_identity':[k for k,v in physical.items() if not v], 'pass':not missing and len(pulse)==5 and pulse==list(range(pulse[0],pulse[0]+5)) and pulse[0] in (700,701) and not badw and not prefix and len(disturbance)==16 and max(max(x) for x in disturbance)>=.01 and means is not None and means[0]<=.08 and means[1]<=.20 and means[2]<=.15 and all(physical.values())})
inputs=json.loads((P/'inputs.json').read_text()); reference=json.loads((A/'reference-inputs.json').read_text()); host_hash=json.loads((A/'turn-hashes.json').read_text())['host']; shared_nonhost=all(inputs[k]==reference[k] for k in ('landing_policy_sha256','route_policy_sha256','frozen_flight_sha256')); host_identified=inputs['host_sha256']==host_hash and inputs['host_sha256']!=reference['host_sha256']; height=json.loads((P/'height-analysis.json').read_text());out={'rows':rows,'observer_errors':errors,'original_shared_inputs_checks':[r['checks']['shared_inputs'] for r in base['rows']],'height_lane_gates':[r['passed'] for r in height['rows']],'shared_nonhost_inputs_match':shared_nonhost,'instrumented_host_hash_identified':host_identified,'all_pass':not errors and shared_nonhost and host_identified and all(r['pass'] for r in rows) and all(r['passed'] for r in height['rows']),'scope':'used-seed finite gust engineering screen; does not establish a general recovery envelope or navigation'};(P/'gust-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
