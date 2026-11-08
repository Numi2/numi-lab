import json,runpy
from pathlib import Path
mod=runpy.run_path('analyze.py');ref=mod['analyze'](json.loads(Path('fixed128.json').read_text()));out={}
for label in ('2e-4','2e-5'):
 d=json.loads(Path('candidate-'+label+'-cap128.json').read_text());r=mod['analyze'](d)
 failed=[(i,s) for i,s in enumerate(d['steps']) if s['world_code']!=0]
 assert all(s['world_code']==11 and s['unchanged_from_previous'] and s['sweeps']==128 for i,s in failed)
 result={k:v for k,v in r.items() if k!='steps'}
 result['first_failure']={'step':failed[0][0],'normal_m_s':failed[0][1]['normal_residual'],'tangent_m_s':failed[0][1]['failure_tangent_diagnostic']} if failed else None
 if len(r['force_bins_N'])==4 and len(ref['force_bins_N'])==4:
  x=[s['ground_normal_impulse_Ns']/.001 for s in r['steps']];y=[s['ground_normal_impulse_Ns']/.001 for s in ref['steps']];peak=max(map(abs,y))
  force=max(abs(a-b) for a,b in zip(x,y))/peak;imp=abs(r['total_ground_impulse_Ns']-ref['total_ground_impulse_Ns'])/ref['total_ground_impulse_Ns']
  result['one_ms_force_error_relative_peak']=force;result['impulse_error_relative']=imp;result['both_below_0_25_percent']=force<=.0025 and imp<=.0025
 else:result['force_comparison']='undefined: trajectory incomplete'
 out[label]=result
Path('cap-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
