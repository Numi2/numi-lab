import json
from pathlib import Path
j=json.loads(Path('analysis.json').read_text());a=j['fixed64'];b=j['bounded']
assert len(a['steps'])==len(b['steps'])==20 and all(x['accepted'] for x in a['steps']+b['steps'])
f=[x['ground_normal_impulse_Ns']/.001 for x in a['steps']];g=[x['ground_normal_impulse_Ns']/.001 for x in b['steps']];peak=max(abs(x) for x in f)
error=max(abs(x-y) for x,y in zip(f,g))/peak
out={'fixed64_force_N':f,'bounded_force_N':g,'reference_peak_N':peak,'max_force_error_relative_peak':error,'force_gate_0_25_percent_pass':error<=.0025}
Path('one-ms.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
