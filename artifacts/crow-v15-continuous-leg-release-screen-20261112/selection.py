from pathlib import Path
import json
A=Path("artifacts/crow-v15-continuous-leg-release-screen-20261112");B=Path("artifacts/crow-v15-continuous-leg-release-hold-20261112");rows=[]
for seed in (20261126,20261127):
 pa=json.loads((B/f"paired-{seed}.json").read_text());ar=json.loads((A/f"analysis-{seed}.json").read_text());br=json.loads((B/f"analysis-{seed}.json").read_text());aj=json.loads((A/f"joint-speed-{seed}.json").read_text());bj=json.loads((B/f"joint-speed-{seed}.json").read_text())
 for pair,a,b,x,y in zip(pa,ar['rows'],br['rows'],aj['rows'],bj['rows']):
  assert len({pair['lane'],a['lane'],b['lane'],x['lane'],y['lane']})==1
  delta_i=pair['extension']['windows']['549_556']['vertical_contact_ns']-pair['crouch_hold']['windows']['549_556']['vertical_contact_ns']
  v_ext=pair['extension']['root'].get('556',{}).get('vz_m_s');v_hold=pair['crouch_hold']['root'].get('556',{}).get('vz_m_s');delta_v=None if v_ext is None or v_hold is None else v_ext-v_hold
  def safe(o,j):
   s=o['stats'];return o['end']==625 and not o['failure'] and s.get('contact')==6800 and s.get('toe_coverage')==6800 and s.get('transition_coverage')==240 and s.get('transition_unloaded')==0 and s.get('post_push_coverage')==80 and s.get('post_push_loaded')==0 and s.get('overflow')==0 and s.get('nonfoot')==0 and len(o['caps'])==0 and j['complete'] and j['all_joint_limit_hits']==0
  parity=not ar['errors'] and not br['errors'] and not aj['errors'] and not bj['errors'] and not pair['state_incomplete_steps'] and not pair['contact_incomplete_steps'] and all(pair[k] is None or pair[k]>=549 for k in ('state_first_difference','contact_first_difference'))
  row={'seed':seed,'lane':a['lane'],'duration_s':a['arm'],'delta_vertical_contact_impulse_ns':delta_i,'delta_root_vz_m_s':delta_v,'prefix_and_coverage':parity,'extension_safe':safe(a,x),'hold_safe':safe(b,y),'extension_joint_limit_hits':x['all_joint_limit_hits'],'hold_joint_limit_hits':y['all_joint_limit_hits'],'extension_toe_hits':len(a['caps']),'hold_toe_hits':len(b['caps'])}
  row['pass']=parity and row['extension_safe'] and row['hold_safe'] and delta_i>=.020 and delta_v is not None and delta_v>=.030
  rows.append(row)
groups={str(t):{'all_four_pass':all(r['pass'] for r in rows if r['duration_s']==t),'count':sum(r['duration_s']==t for r in rows)} for t in (.2,.4,.6,.8)}
assert all(x['count']==4 for x in groups.values())
(A/'selection.json').write_text(json.dumps({'rows':rows,'settings':groups,'scope':'Predeclared engineering screen on correlated development starts. It does not establish biological or full-flight qualification.'},indent=2)+'\n')
for r in rows:print(r)
print('settings',groups)
