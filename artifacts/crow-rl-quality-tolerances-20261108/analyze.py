from pathlib import Path
import csv,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-rl-quality-tolerances-20261108';B=R/'build/crow-quality-tolerances-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def analyze():
 plan=json.loads((A/'plan.json').read_text());rows=[];same=True
 for arm in plan['arms']:
  observations=[]
  for repeat in range(2):
   name=arm+f'-{repeat}';out=B/'observations'/name
   config=json.loads((out/'effective-config.json').read_text());trace=list(csv.DictReader((out/'trace.tsv').open(),delimiter='\t'))
   failure=json.loads((out/'failure.json').read_text()) if (out/'failure.json').exists() else None
   terminal=json.loads((out/'terminal.json').read_text()) if (out/'terminal.json').exists() else None
   summary=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else None
   assert sum(x is not None for x in [failure,terminal,summary])==1
   penetration=max([0.]+[-float(r[k]) for r in trace for k in ['upper_gap_m','lower_gap_m']])
   capture=B/'observations'/(name+'-capture.bin')
   physical={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='runtime.json'}
   physical['failure_capture']=sha(capture) if capture.exists() else None
   if arm=='strict':assert physical['failure_capture']==plan['predictions']['strict_capture_sha256']
   accepted=len(trace)+(failure['successful_steps'] if failure else 0)
   rec=dict(arm=arm,repeat=repeat,effective=config,accepted_transitions=accepted,trace_rows=len(trace),
     physical_failure=failure,task_termination=terminal,complete=summary is not None,
     max_sampled_jaw_penetration_m=penetration,final_sample_tick=max([int(t['tick']) for t in trace],default=-1),
     physical_files=physical,summary=summary)
   rec['physical_gate_passed']=bool(summary and summary['accepted_transitions']==640 and summary['physics_rejections']==0 and not terminal and penetration<=plan['predictions']['sampled_penetration_max_m'])
   observations.append(rec)
  exact=observations[0]['physical_files']==observations[1]['physical_files'];same &= exact
  for r in observations:r['repeated_physical_files_exact']=exact
  rows.extend(observations)
 base=next(x for x in rows if x['arm']=='api-default' and x['repeat']==0)
 supported=dict(strict_failure_reproduced=all(x['physical_files']['failure_capture']==plan['predictions']['strict_capture_sha256'] for x in rows if x['arm']=='strict'),
   repeated_physical_files_exact=same,api_default_physical_completion=all(x['physical_gate_passed'] for x in rows if x['arm']=='api-default'))
 report=dict(schema='crow-quality-tolerances-result-v1',arms=len(plan['arms']),native_trial_commands=len(rows),
   predictions_supported=supported,api_default_accepted_transitions=base['accepted_transitions'],
   api_default_complete=base['complete'],api_default_max_sampled_penetration_m=base['max_sampled_jaw_penetration_m'],
   policy_or_profile_promoted=False,production_changed=False,goal_complete=False,
   limits='One fixed four-lane surrogate trajectory, two repeats for determinism; sampled control-boundary geometry only; no CCD, biological calibration, feeding or general robustness qualification.')
 write(A/'case-analysis.json',rows);write(A/'result.json',report);print(json.dumps(report),flush=True)
 return report
if __name__=='__main__':analyze()
