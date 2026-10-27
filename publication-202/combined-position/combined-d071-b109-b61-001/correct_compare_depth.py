from pathlib import Path
import json,hashlib
OUT=Path(__file__).resolve().parent
p=OUT/'full-comparison.json'; old=json.loads(p.read_text()); oldsha=hashlib.sha256(p.read_bytes()).hexdigest()
diag={'schema':'numi.human.combined.d071.plane-distance-diagnostic.v1','source_report_path':str(p),'source_report_sha256_before_correction':oldsha,'metric_name':'signed_face_distance_m','metric_interpretation':'signed distance to the selected triangle plane; not Euclidean nearest clearance and may jump when nearest_triangle changes','depth_use':'excluded from clearance/depth progress claims','affected_target_plane_metric_rows':old.get('affected_target_depth_vs_d071',{}).get('rows',[]),'known_witness':{'node_row':9281,'bank':[109,114],'d071_nearest_triangle':91,'candidate_nearest_triangle':92,'d071_nearest_distance_mm':3.9994595585063257,'candidate_nearest_distance_mm':3.9992361692884025,'d071_signed_face_plane_distance_mm':3.9987234096652367,'candidate_signed_face_plane_distance_mm':3.2521163130381204,'point_motion_mm':0.0009625965594952829,'note':'The triangle-plane metric jumps as the selected nearest triangle changes; the nearest Euclidean distance changes within the point-motion bound.'},'geometry_admitted':False,'physical_steps':0}
(OUT/'full-comparison-plane-distance-diagnostic.json').write_text(json.dumps(diag,indent=2,sort_keys=True,allow_nan=False)+'\n')
src=(OUT/'compare_full.py').read_text()
start=src.index('# Depth comparisons for each affected inside row at the target banks, against d071.')
end=src.index("shape_layer=readj(OUT/'shape-layer-report.json')",start)
new=r'''# Compare classification-signed nearest Euclidean distances, not signed triangle-plane distances.
q={k:qrows(v['q']) for k,v in R.items()}
with np.load(C,allow_pickle=False) as zc,np.load(D,allow_pickle=False) as zd:
 X=np.asarray(zd['points_world_m'],dtype=np.float32);Y=np.asarray(zc['points_world_m'],dtype=np.float32)
 motion=np.linalg.norm(Y.astype(np.float64)-X.astype(np.float64),axis=1)*1000
shared=sorted(set(q['d071'])&set(q['candidate']))
nearest_rows=[];classchanges=[];bound_violations=[];plane_jumps=[]
def signed_nearest(r):
 d=float(r['nearest_distance_m'])*1000
 return -d if r['classification']=='inside' else d
for key in shared:
 b=q['d071'][key];c=q['candidate'][key];node=int(key[0]);move=float(motion[node])
 if b['classification']!=c['classification']:classchanges.append({'query_key':list(key),'baseline':b['classification'],'candidate':c['classification']})
 db=signed_nearest(b);dc=signed_nearest(c);change=dc-db;allow=move+1e-5
 nearest_rows.append({'query_key':list(key),'signed_nearest_distance_change_mm':change,'point_motion_mm':move,'absolute_change_within_motion_bound':abs(change)<=allow})
 if abs(change)>allow:bound_violations.append({'query_key':list(key),'signed_nearest_distance_change_mm':change,'point_motion_mm':move,'excess_mm':abs(change)-move})
 pdelta=(float(c['signed_face_distance_m'])-float(b['signed_face_distance_m']))*1000
 if abs(pdelta)>allow or b['nearest_triangle']!=c['nearest_triangle']:
  plane_jumps.append({'query_key':list(key),'nearest_triangle_baseline':int(b['nearest_triangle']),'nearest_triangle_candidate':int(c['nearest_triangle']),'signed_face_plane_distance_baseline_mm':float(b['signed_face_distance_m'])*1000,'signed_face_plane_distance_candidate_mm':float(c['signed_face_distance_m'])*1000,'signed_face_plane_delta_mm':pdelta,'nearest_distance_baseline_mm':float(b['nearest_distance_m'])*1000,'nearest_distance_candidate_mm':float(c['nearest_distance_m'])*1000,'point_motion_mm':move,'semantics':'selected-triangle-plane diagnostic only; not nearest Euclidean clearance'})
nearest_check={'metric':'classification-signed nearest_distance_m (negative inside, positive outside)','shared_query_records':len(shared),'classification_changes':classchanges,'max_point_motion_mm':float(motion.max()),'max_absolute_signed_nearest_distance_change_mm':max((abs(x['signed_nearest_distance_change_mm']) for x in nearest_rows),default=0.0),'max_motion_bound_excess_mm':max((abs(x['signed_nearest_distance_change_mm'])-x['point_motion_mm'] for x in nearest_rows),default=0.0),'bound_tolerance_mm':1e-5,'bound_violation_count':len(bound_violations),'all_shared_rows_within_displacement_bound':not bool(bound_violations),'violations':bound_violations}
depth=[];depth_by_bank={}
for bank in ((61,97),(109,114)):
 bankrows=[]
 for key in sorted(k for k in inside_sets['d071'] if k[1:]==bank):
  b=q['d071'].get(key);c=q['candidate'].get(key)
  if b is None or c is None:raise RuntimeError(f'missing nearest-distance row for inside identity {key}')
  if b['classification']!='inside' or c['classification']!='inside':raise RuntimeError(f'inside identity classification changed at {key}')
  row={'node_row':int(key[0]),'bank':[int(bank[0]),int(bank[1])],'baseline_classification':b['classification'],'candidate_classification':c['classification'],'baseline_signed_nearest_distance_mm':signed_nearest(b),'candidate_signed_nearest_distance_mm':signed_nearest(c),'progress_mm':signed_nearest(c)-signed_nearest(b),'point_motion_mm':float(motion[int(key[0])])}
  bankrows.append(row);depth.append(row)
 vals=[x['progress_mm'] for x in bankrows]
 depth_by_bank[f'{bank[0]}-{bank[1]}']={'inside_rows_compared':len(bankrows),'rows':bankrows,'minimum_progress_mm':min(vals) if vals else None,'sum_progress_mm':sum(vals)}
plane_jumps.sort(key=lambda x:abs(x['signed_face_plane_delta_mm'])-x['point_motion_mm'],reverse=True)
plane_diag={'metric':'signed_face_distance_m','interpretation':'signed distance to the selected triangle plane, not Euclidean nearest clearance; values can jump when nearest_triangle changes','rows_exceeding_motion_bound_or_triangle_changed':len(plane_jumps),'examples':plane_jumps[:20]}
'''
src=src[:start]+new+src[end:]
oldfield="'affected_target_depth_vs_d071':{'rows':depth,'minimum_progress_mm':min([x['progress_mm'] for x in depth],default=None),'sum_progress_mm':sum(x['progress_mm'] for x in depth)},"
newfield="'nearest_distance_comparison_vs_d071':nearest_check,'target_inside_nearest_distance_vs_d071':depth_by_bank,'plane_signed_distance_diagnostic_vs_d071':plane_diag,"
if oldfield not in src:raise RuntimeError('expected old report depth field was not found')
src=src.replace(oldfield,newfield)
limitations="'limitations':['Static reference-pose construction checks only. No physical simulation, material-mass transfer, or native tissue admission.']"
newlimitations="'limitations':['Static reference-pose construction checks only. signed_face_distance_m is a selected-triangle plane metric and is excluded from depth claims; nearest-distance comparisons use classification-signed nearest_distance_m and are checked against point-motion bounds. No physical simulation, material-mass transfer, or native tissue admission.']"
if limitations not in src:raise RuntimeError('expected limitations field not found')
src=src.replace(limitations,newlimitations)
(OUT/'compare_full.py').write_text(src)
print(json.dumps({'plane_diagnostic_path':str(OUT/'full-comparison-plane-distance-diagnostic.json'),'plane_diagnostic_sha256':hashlib.sha256((OUT/'full-comparison-plane-distance-diagnostic.json').read_bytes()).hexdigest(),'compare_script_patched':True}))
