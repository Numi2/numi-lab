from pathlib import Path
import json,sys
p=Path('artifacts/crow-v15-actor-command-flight-20261114')/sys.argv[1]
a=json.loads((p/'analysis.json').read_text())
excluded={'flap_target_reconstructed','pronation_target_reconstructed','signed_heading','lateral_deviation'}
rows=[]
for x in a['rows']:
 checks={k:v for k,v in x['checks'].items() if k not in excluded}
 rows.append({'lane':x['lane'],'safety_checks':checks,'failed_safety_checks':[k for k,v in checks.items() if not v],'observed_height_range_m':x['height_range_m'],'observed_forward_speed_range_m_s':x['forward_speed_range_m_s'],'observed_max_tilt_rad':x['max_tilt_rad'],'observed_max_heading_change_rad':x['max_abs_heading_change_rad']})
out={'excluded_script_specific_checks':sorted(excluded),'lanes':rows,'all_physical_screen_checks_pass':len(rows)==8 and all(not r['failed_safety_checks'] for r in rows),'scope':'used-seed exploratory actor handoff; no arbitrary-command, mission, landing or biological qualification'}
(p/'physical-screen.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
