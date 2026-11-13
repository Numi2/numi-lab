from pathlib import Path
import json,sys
P=Path('artifacts/crow-v15-pronation-amplitude-screen-20261113')/sys.argv[1];a=json.loads((P/'analysis.json').read_text());f=json.loads((P/'force-analysis.json').read_text());r={x['lane']:x for x in a['rows']};fr={x['lane']:x for x in f['rows']};par={x['lane']:x['exact'] for x in f['parity']};prereq=bool(f['policy_bytes_match'] and f['scene_mask_matches'] and f['model_matches_expected'] and not f['errors'] and f['duplicate_count']==0 and all(par[i] for i in range(4)) and a['reference_controls_exact'] and all(a['amplitude_target_checks'][str(i)]['passed'] for i in (2,3)))
checks={}
for e in (4,5,6,7):
 x=fr[e];j=x['pronation'];saturation=[v['at_effort_limit_count']/v['samples'] if v['samples'] else None for v in j];valid=bool(prereq and par[e] and r[e]['preliminary_engineering_pass'] and a['amplitude_target_checks'][str(e)]['passed'] and r[e]['pronation_position_range_rad'][0] is not None and r[e]['pronation_position_range_rad'][0]>-.35 and r[e]['pronation_position_range_rad'][1]<.35 and x['complete'] and x['invalid_records']==0 and x['kernel_input_vs_substep_start_max_abs']<1e-7 and x['max_unaccounted_world_force_n']<1e-7 and all(v['samples']==8016 and v['position'] is not None and v['position']['min']>-.35 and v['position']['max']<.35 and v['over_effort_limit_count']==0 for v in j) and all(v is not None and v<=.75 for v in saturation))
 checks[str(e)]={'passed':valid,'effort_limit_fraction_left_right':saturation,'sampled_position_ranges':[[v['position']['min'],v['position']['max']] if v['position'] else None for v in j]}
pairs={str(amp):all(checks[str(e)]['passed'] for e in pair) for amp,pair in ((.12,(4,5)),(.08,(6,7)))};selection=None
if any(pairs.values()):
 cand=[]
 for amp,pair in ((.12,(4,5)),(.08,(6,7))):
  if pairs[str(amp)]:
   worst=max(v for e in pair for v in checks[str(e)]['effort_limit_fraction_left_right']);margin=min(min(r[e]['height_range_m'][0]-.25,1.5-r[e]['height_range_m'][1]) for e in pair);cand.append((worst,-margin,amp))
 selection=min(cand)[2]
out={'common_prerequisites':prereq,'lanes':checks,'pair_pass':pairs,'selected_amplitude_rad':selection,'scope':'One used-seed obstacle-free engineering screen, not biological calibration, qualified flight, or freeze.'};(P/'screen.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
