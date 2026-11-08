import json,runpy
from pathlib import Path
mod=runpy.run_path('analyze.py');new=mod['analyze'](json.loads(Path('fixed128.json').read_text()));base=json.loads(Path('analysis.json').read_text())['fixed64']
a=[x['ground_normal_impulse_Ns']/.001 for x in base['steps']];b=[x['ground_normal_impulse_Ns']/.001 for x in new['steps']]
assert len(a)==len(b)==20 and all(x['accepted'] for x in base['steps']+new['steps'])
peak=max(map(abs,b));force=max(abs(x-y) for x,y in zip(a,b))/peak
impulse=abs(base['total_ground_impulse_Ns']-new['total_ground_impulse_Ns'])/new['total_ground_impulse_Ns']
out={'force_error_relative_128_peak':force,'impulse_error_relative_128':impulse,'both_below_0_25_percent':force<=.0025 and impulse<=.0025,'fixed128':new}
Path('reference-refinement.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='fixed128'},indent=2))
