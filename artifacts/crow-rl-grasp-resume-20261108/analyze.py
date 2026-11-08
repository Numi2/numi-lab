from pathlib import Path
import csv,hashlib,json
R=Path.cwd();A=R/'artifacts/crow-rl-grasp-resume-20261108';B=R/'build/crow-grasp-resume-build-01';O=B/'observations'
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text());execution=json.loads((A/'execution.json').read_text())
cases=[]
for name,arm in [('normal',0),('open',1),('disconnected',2),('frictionless',3)]:
 p=O/name
 if not p.exists():continue
 summary=json.loads((p/'summary.json').read_text()) if (p/'summary.json').exists() else None
 failure=json.loads((p/'failure.json').read_text()) if (p/'failure.json').exists() else None
 terminal=json.loads((p/'terminal.json').read_text()) if (p/'terminal.json').exists() else None
 rows=[]
 for r in csv.DictReader((p/'trace.tsv').open(),delimiter='\t'):
  try:
   if None in r or any(v is None for v in r.values()):continue
   for v in r.values():float(v)
  except (ValueError,TypeError):continue
  rows.append(r)
 maximum=lambda names:max([0.]+[abs(float(r[k])) for r in rows for k in names])
 penetration=max([0.]+[-float(r[k]) for r in rows for k in ['upper_gap_m','lower_gap_m']])
 ground=max([0.]+[.004-float(r['z']) for r in rows])
 held=[lane['minimum_lift_64_95_m']>=.01 and lane['maximum_head_relative_slip_m']<=.0005 and lane['opposed_contact_samples']>=28 for lane in summary['lanes']] if summary else None
 files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.is_file() and f.name!='runtime.json'}
 cap=O/(name+'-capture.bin')
 cases.append(dict(name=name,arm=arm,summary=summary,failure=failure,terminal=terminal,trace_complete_rows=len(rows),
  held_grasp=held,held_count=sum(held) if held is not None else None,max_sampled_jaw_penetration_m=penetration,max_sampled_ground_penetration_m=ground,
  max_sampled_jaw_normal_impulse=maximum(['upper_Jn','lower_Jn']),max_sampled_jaw_tangent_impulse=maximum(['upper_Jt','lower_Jt']),
  physical_files=files,capture_sha256=hashlib.sha256(cap.read_bytes()).hexdigest() if cap.exists() else None))
by={c['name']:c for c in cases}
complete=len(cases)==4 and all(c['summary'] and c['summary']['accepted_transitions']==640 and c['summary']['physics_rejections']==0 and c['trace_complete_rows']==640 and c['failure'] is None and c['terminal'] is None for c in cases)
gold=B/'reference/normal-gold'
normal_exact='normal' in by and all((O/'normal'/p.name).is_file() and p.read_bytes()==(O/'normal'/p.name).read_bytes() for p in gold.iterdir())
gates=dict(all_arms_complete_640=complete,normal_physical_files_exact=normal_exact,
 enabled_jaw_penetration_bound=len(cases)==4 and all(c['summary'] and c['trace_complete_rows']==640 and c['max_sampled_jaw_penetration_m']<=.0008 for c in cases if c['arm']!=2),
 ground_penetration_bound=len(cases)==4 and all(c['summary'] and c['trace_complete_rows']==640 and c['max_sampled_ground_penetration_m']<=.0008 for c in cases),
 disconnected_jaw_impulses_zero='disconnected' in by and by['disconnected']['summary'] is not None and by['disconnected']['trace_complete_rows']==640 and by['disconnected']['max_sampled_jaw_normal_impulse']<=1e-12 and by['disconnected']['max_sampled_jaw_tangent_impulse']<=1e-12,
 frictionless_jaw_tangent_impulses_zero='frictionless' in by and by['frictionless']['summary'] is not None and by['frictionless']['trace_complete_rows']==640 and by['frictionless']['max_sampled_jaw_tangent_impulse']<=1e-12)
observations={name:c['held_count'] for name,c in by.items()}
hypotheses={name:observations.get(name)==v for name,v in plan['mechanism_predictions_held_count'].items()}
write(A/'analysis.json',dict(cases=cases,normal_gold_files=len(list(gold.iterdir())),sample_scope='Control-boundary evidence; no bound on every unobserved physics substep'))
write(A/'result.json',dict(schema='crow-grasp-controls-result-v1',native_commands=len(execution),correctness_gates=gates,all_correctness_gates_passed=all(gates.values()),
 observed_held_counts=observations,mechanism_hypotheses_supported=hypotheses,all_mechanism_hypotheses_supported=all(hypotheses.values()),production_changed=False,policy_or_profile_promoted=False,goal_complete=False))
print((A/'result.json').read_text(),flush=True)
