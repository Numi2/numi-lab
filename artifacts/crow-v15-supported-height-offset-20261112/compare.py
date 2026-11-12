from pathlib import Path
import json
A=Path('artifacts/crow-v15-supported-height-offset-20261112');B=Path('artifacts/crow-v15-single-reset-perturbation-20261112')
x=json.loads((B/'analysis-20261127.json').read_text());y=json.loads((A/'analysis-20261127.json').read_text());rows=[]
for a,b in zip(x['rows'],y['rows']):
 def f(z):
  return {'initial_unloaded_substeps':z['unloaded_by_period']['early'],'nonfoot_initial_substeps':z['nonfoot_ground_by_period']['early'],'initial_cap_hits':sum(v['hits'] for v in z['speed_cap_hits'] if v['period']=='early'),'postsettle_cap_hits':sum(v['hits'] for v in z['speed_cap_hits'] if v['period']!='early'),'walking_unloaded_substeps':z['unloaded_by_period']['walk_4_6'],'wp':z['waypoints'],'final_position':z['final_position_m'],'final_velocity':z['final_velocity_m_s'],'reason':z['termination_reason']}
 rows.append({'lane':a['lane'],'reference':f(a),'supported':f(b)})
(A/'comparison.json').write_text(json.dumps({'rows':rows,'scope':'Not identical starting states: the same random channels are applied once versus twice. Both runs are ten seconds; supported reset q differs while perturbation random channels remain the same.'},indent=2)+'\n')
for z in rows:
 print(z['lane'],'early unloaded original/supported',z['reference']['initial_unloaded_substeps'],z['supported']['initial_unloaded_substeps'],'early caps',z['reference']['initial_cap_hits'],z['supported']['initial_cap_hits'],'single walk unloaded',z['supported']['walking_unloaded_substeps'],'postsettle caps',z['supported']['postsettle_cap_hits'],'wp at 10',z['supported']['wp'],'z',round(z['supported']['final_position'][2],3))
