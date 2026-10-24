"""Aggregate completed accepted intervals without attributing rejected paths to the delivery."""
from pathlib import Path
import csv,json,math
O=Path('/Users/home/Documents/Codex/2026-10-21/i-wa/outputs/plastic_bag_empty_franka_20261024');r=json.loads((O/'delivery_receipt.json').read_text())
a=[];e=[]
for part in ['prefix','combined']:
 a+=list(csv.DictReader((O/part/'solver_accuracy.csv').open()));e+=[x for x in csv.DictReader((O/part/'energy_history.csv').open()) if int(x['step'])>0]
assert len(a)==len(e)==1920 and [int(x['step']) for x in a]==[int(x['step']) for x in e]==list(range(1,1921))
last=e[-1];mass=r['film_mass_kg_from_gpu'];summaries=[json.loads((O/p/'run_summary.json').read_text()) for p in ['first_drop','combined']]
result={'duration_s':8,'native_hz':240,'saved_states':1921,'film_mass_kg':mass,'final_film_kinetic_energy_J':float(last['kinetic_J']),'final_film_mass_weighted_rms_speed_m_s':math.sqrt(2*float(last['kinetic_J'])/mass),'actual_corner_lift_m':r['actual_corner_lift_m'],'coin_count_from_gpu':r['coin_count_from_gpu'],'solver_accuracy_columns':list(a[0]),'accepted_interval_count':len(a),'accepted_substeps':sorted(set(int(x['substeps']) for x in a)),'accepted_material_refinement':sorted(set(int(x['material_refinement']) for x in a)),'total_rejected_trials':sum(int(x['attempt'])-1 for x in a),'accepted_energy_bad_flags':sum(int(x['energy_bad']) for x in a),'accepted_film_bad_flags':sum(int(x['film_bad']) for x in a),'first_drop_and_final_continuation_max_stretch_ratio':max(x['max_edge_stretch_ratio'] for x in summaries),'stretch_scope':'Maxima from completed first-drop and 720–1920 continuation segments; the 240–720 accepted prefix remains in its per-state history. Rejected states do not enter this aggregation.','final_energy_components_J':{k:float(last[k]) for k in ['kinetic_J','gravity_J','membrane_J','bending_J','total_J']}}
(O/'aggregate_metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
