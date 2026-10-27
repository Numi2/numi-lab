from pathlib import Path
import csv,hashlib,json
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027'
OUT=Path(__file__).resolve().parent
C=OUT/'candidate.npz';D=S/'root-topology-a69-safe-subset-001/candidate.npz';A=ROOT/'D/publication-200-stage-001/evidence/retained/candidate.npz'
R={
 'candidate':{'130':OUT/'audit-130/summary.json','q':OUT/'audit-130/query/pairs.csv','self':OUT/'audit-130/surface-self-pairs.npy','159':OUT/'audit-159/report.json'},
 'd071':{'130':S/'root-topology-a69-safe-subset-001/audit-130/summary.json','q':S/'root-topology-a69-safe-subset-001/audit-130/query/pairs.csv','self':S/'root-topology-a69-safe-subset-001/audit-130/surface-self-pairs.npy','159':S/'root-topology-a69-safe-subset-001/audit-159/report.json'},
 'a69':{'130':ROOT/'D/publication-200-stage-001/evidence/retained/audit-130/summary.json','q':ROOT/'D/publication-200-stage-001/evidence/retained/audit-130/query/pairs.csv','self':ROOT/'D/publication-200-stage-001/evidence/retained/audit-130/surface-self-pairs.npy','159':ROOT/'D/publication-200-stage-001/evidence/retained/audit-159/report.json'}}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def inside(path):
 with Path(path).open(newline='') as f:
  return {(int(x['query_row']),int(x['bank_body']),int(x['bank_stable_id'])) for x in csv.DictReader(f) if x['classification']=='inside'}
def qrows(path):
 out={}
 with Path(path).open(newline='') as f:
  for r in csv.DictReader(f):out[(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id']))]=r
 return out
def crossdoc(path):
 d=readj(path)
 return {(int(b['body']),int(b['stable_bone_id']),int(sf),int(bf)) for b in d['banks'] for sf,bf in ((b.get('result') or {}).get('triangle_pairs') or [])}
def mapparent(cand):
 with np.load(cand,allow_pickle=False) as z:return np.asarray(z['refinement_face_parent_face_row'],dtype=np.int64).copy()
def parents(raw,parent):
 return {(b,k,int(parent[sf]),bf) for b,k,sf,bf in raw}
def pairs_array(path):return {tuple(map(int,x)) for x in np.asarray(np.load(path,allow_pickle=False),dtype=np.int64).reshape(-1,2)}
# Verify every full report binds to the exact emitted candidate.
for label,p in [('candidate',C),('d071',D),('a69',A)]:
 d130=readj(R[label]['130']);d159=readj(R[label]['159'])
 if d130.get('candidate_sha256')!=sha(p) or d159.get('candidate_sha256')!=sha(p):raise RuntimeError(f'{label} audit candidate hash mismatch')
 if not d130.get('complete_surface_audit') or int(d130.get('exact_PLC_surface_pairs',-1))!=0:raise RuntimeError(f'{label} audit130 did not pass complete self screen')
 if not d159.get('complete') or int(d159.get('bank_count',-1))!=185:raise RuntimeError(f'{label} audit159 incomplete')
inside_sets={k:inside(v['q']) for k,v in R.items()}
self_sets={k:pairs_array(v['self']) for k,v in R.items()}
raw_sets={k:crossdoc(v['159']) for k,v in R.items()}
parent_maps={k:mapparent({'candidate':C,'d071':D,'a69':A}[k]) for k in R}
parent_sets={k:parents(raw_sets[k],parent_maps[k]) for k in R}
reports={k:{'audit130':readj(v['130']),'audit159':readj(v['159'])} for k,v in R.items()}
summary={}
for comp,base in [('candidate','d071'),('candidate','a69')]:
 ri=raw_sets[comp];rb=raw_sets[base];pi=parent_sets[comp];pb=parent_sets[base];ii=inside_sets[comp];ib=inside_sets[base];si=self_sets[comp];sb=self_sets[base]
 added=ri-rb;removed=rb-ri;padded=pi-pb;premoved=pb-pi;iadd=ii-ib;irem=ib-ii;sadd=si-sb;srem=sb-si
 bcounts={(int(x['body']),int(x['stable_bone_id'])):len((x.get('result') or {}).get('triangle_pairs') or []) for x in reports[base]['audit159']['banks']}
 ccounts={(int(x['body']),int(x['stable_bone_id'])):len((x.get('result') or {}).get('triangle_pairs') or []) for x in reports[comp]['audit159']['banks']}
 bank_changes=[{'bank':[b,k],'baseline_raw_count':bcounts[(b,k)],'candidate_raw_count':ccounts[(b,k)]} for b,k in sorted(bcounts) if bcounts[(b,k)]!=ccounts[(b,k)]]
 by_target={}
 for bank in ((61,97),(109,114)):
  ba={(x[2],x[3]) for x in rb if x[:2]==bank};ca={(x[2],x[3]) for x in ri if x[:2]==bank}
  by_target[f'{bank[0]}-{bank[1]}']={'baseline_raw_count':len(ba),'candidate_raw_count':len(ca),'added_count':len(ca-ba),'removed_count':len(ba-ca),'added':[list(x) for x in sorted(ca-ba)],'removed':[list(x) for x in sorted(ba-ca)]}
 summary[f'{comp}_vs_{base}']={
  'raw_crossings':{'baseline_count':len(rb),'candidate_count':len(ri),'added_count':len(added),'removed_count':len(removed),'added':[list(x) for x in sorted(added)],'removed':[list(x) for x in sorted(removed)],'banks_with_changed_counts':bank_changes},
  'parent_pairs':{'baseline_count':len(pb),'candidate_count':len(pi),'added_count':len(padded),'removed_count':len(premoved),'added':[list(x) for x in sorted(padded)],'removed':[list(x) for x in sorted(premoved)]},
  'inside_node_bank_identities':{'baseline_count':len(ib),'candidate_count':len(ii),'added_count':len(iadd),'removed_count':len(irem),'added':[list(x) for x in sorted(iadd)],'removed':[list(x) for x in sorted(irem)]},
  'exact_surface_self_pairs':{'baseline_count':len(sb),'candidate_count':len(si),'added_count':len(sadd),'removed_count':len(srem),'added':[list(x) for x in sorted(sadd)],'removed':[list(x) for x in sorted(srem)]},'target_banks':by_target}
# Compare classification-signed nearest Euclidean distances, not signed triangle-plane distances.
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
shape_layer=readj(OUT/'shape-layer-report.json');execution=readj(OUT/'full-audits-execution.json')
report={'schema':'numi.human.combined.d071.b109_114.b61_97.full-comparison.v1','status':'COMPLETE_STATIC_CONSTRUCTION_COMPARISON','pins':{'candidate_path':str(C),'candidate_sha256':sha(C),'d071_path':str(D),'d071_sha256':sha(D),'a69_path':str(A),'a69_sha256':sha(A),'preflight_path':str(OUT/'preflight.json'),'preflight_sha256':sha(OUT/'preflight.json'),'shape_layer_report_path':str(OUT/'shape-layer-report.json'),'shape_layer_report_sha256':sha(OUT/'shape-layer-report.json'),'audit130_path':str(R['candidate']['130']),'audit130_sha256':sha(R['candidate']['130']),'audit159_path':str(R['candidate']['159']),'audit159_sha256':sha(R['candidate']['159']),'execution_path':str(OUT/'full-audits-execution.json'),'execution_sha256':sha(OUT/'full-audits-execution.json')},'audit130_candidate':{k:reports['candidate']['audit130'].get(k) for k in ('status','points','faces','complete_surface_audit','exact_PLC_surface_pairs','inside_nodes','inside_node_bank_pairs','query_pair_count')},'audit159_candidate':{k:reports['candidate']['audit159'].get(k) for k in ('status','surface_point_count','surface_face_count','bank_count','expected_bank_count','complete')},'comparisons':summary,'shape':shape_layer['shape'],'layer':shape_layer['layer'],'nearest_distance_comparison_vs_d071':nearest_check,'target_inside_nearest_distance_vs_d071':depth_by_bank,'plane_signed_distance_diagnostic_vs_d071':plane_diag,'geometry_admitted':False,'physical_steps':0,'limitations':['Static reference-pose construction checks only. signed_face_distance_m is a selected-triangle plane metric and is excluded from depth claims; nearest-distance comparisons use classification-signed nearest_distance_m and are checked against point-motion bounds. No physical simulation, material-mass transfer, or native tissue admission.']}
out=OUT/'full-comparison.json';out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'candidate_sha256':sha(C),'comparison_sha256':sha(out),'candidate_self_pairs':len(self_sets['candidate']),'inside_counts':{k:len(v) for k,v in inside_sets.items()},'raw_counts':{k:len(v) for k,v in raw_sets.items()},'parent_counts':{k:len(v) for k,v in parent_sets.items()},'d071_comparison':{k:v for k,v in summary['candidate_vs_d071'].items() if k!='raw_crossings'},'changed_target_depth_rows':len(depth),'full_comparison_path':str(out)},indent=2))
