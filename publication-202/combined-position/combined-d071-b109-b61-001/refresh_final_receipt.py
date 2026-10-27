from pathlib import Path
import hashlib,json
OUT=Path(__file__).resolve().parent
P=OUT/'final-receipt.json';R=json.loads(P.read_text());C=json.loads((OUT/'full-comparison.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
R['target_inside_depth_vs_d071']=C['target_inside_nearest_distance_vs_d071']
R['nearest_distance_displacement_bound_check_vs_d071']=C['nearest_distance_comparison_vs_d071']
R['plane_signed_distance_diagnostic_vs_d071']=C['plane_signed_distance_diagnostic_vs_d071']
R['depth_metric_rule']='For inside rows use classification-signed nearest_distance_m (negative inside). All 13,181 shared query records are checked against per-node movement plus 1e-5 mm. signed_face_distance_m is a selected-triangle-plane metric and is diagnostic only.'
R['identity_comparison']['comparison_report_path']=str(OUT/'full-comparison.json')
R['identity_comparison']['comparison_report_sha256']=sha(OUT/'full-comparison.json')
R['evidence_files']['full_comparison']={'path':str(OUT/'full-comparison.json'),'sha256':sha(OUT/'full-comparison.json')}
R['evidence_files']['plane_metric_diagnostic']={'path':str(OUT/'full-comparison-plane-distance-diagnostic.json'),'sha256':sha(OUT/'full-comparison-plane-distance-diagnostic.json')}
R['evidence_files']['comparison_script']={'path':str(OUT/'compare_full.py'),'sha256':sha(OUT/'compare_full.py')}
R['limitations'].append('signed_face_distance_m reports distance to a selected triangle plane, not nearest clearance; use the separate classification-signed nearest_distance_m check for depth comparisons.')
P.write_text(json.dumps(R,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'receipt_path':str(P),'receipt_sha256':sha(P),'comparison_sha256':sha(OUT/'full-comparison.json'),'compare_script_sha256':sha(OUT/'compare_full.py'),'nearest_shared_rows':R['nearest_distance_displacement_bound_check_vs_d071']['shared_query_records'],'nearest_bound_violations':R['nearest_distance_displacement_bound_check_vs_d071']['bound_violation_count'],'depth_rows':{k:{'n':v['inside_rows_compared'],'min_mm':v['minimum_progress_mm'],'sum_mm':v['sum_progress_mm']} for k,v in R['target_inside_depth_vs_d071'].items()},'plane_diagnostic_examples':R['plane_signed_distance_diagnostic_vs_d071']['rows_exceeding_motion_bound_or_triangle_changed']},indent=2))
