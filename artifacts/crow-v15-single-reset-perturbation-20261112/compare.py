from pathlib import Path
import json
A=Path('artifacts/crow-v15-single-reset-perturbation-20261112');B=Path('artifacts/crow-v15-slow-push-uniform-dev-20261112')
x=json.loads((B/'analysis-20261127.json').read_text());y=json.loads((A/'analysis-20261127.json').read_text());rows=[]
for a,b in zip(x['rows'],y['rows']):
 def f(z):
  return {'initial_unloaded_substeps':z['unloaded_by_period']['early'],'nonfoot_initial_substeps':z['nonfoot_ground_by_period']['early'],'initial_cap_hits':sum(v['hits'] for v in z['speed_cap_hits'] if v['period']=='early'),'postsettle_cap_hits':sum(v['hits'] for v in z['speed_cap_hits'] if v['period']!='early'),'walking_unloaded_substeps':z['unloaded_by_period']['walk_4_6'],'wp':z['waypoints'],'final_position':z['final_position_m'],'final_velocity':z['final_velocity_m_s'],'reason':z['termination_reason']}
 rows.append({'lane':a['lane'],'double':f(a),'single':f(b)})
(A/'comparison.json').write_text(json.dumps({'rows':rows,'scope':'Not identical starting states: the same random channels are applied once versus twice. Double reference is 16s, single is 10s; compare early/through 10s, not final state.'},indent=2)+'\n')
for z in rows:
 print(z['lane'],'early unloaded double/single',z['double']['initial_unloaded_substeps'],z['single']['initial_unloaded_substeps'],'early caps',z['double']['initial_cap_hits'],z['single']['initial_cap_hits'],'single walk unloaded',z['single']['walking_unloaded_substeps'],'postsettle caps',z['single']['postsettle_cap_hits'],'wp at 10',z['single']['wp'],'z',round(z['single']['final_position'][2],3))
