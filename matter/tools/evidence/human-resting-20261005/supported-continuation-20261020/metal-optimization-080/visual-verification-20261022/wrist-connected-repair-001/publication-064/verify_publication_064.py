from pathlib import Path
import hashlib,json,sys
import numpy as np
P=Path(__file__).resolve().parent;D=P/'payload'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
listed=json.loads((P/'payload-files-064.json').read_text())['files']
actual={f.relative_to(D).as_posix():f for f in D.rglob('*')if f.is_file()}
assert set(actual)=={r['path']for r in listed}
for r in listed:
 f=actual[r['path']];assert sha(f)==r['sha256']and f.stat().st_size==r['size_bytes'],r['path']
m=json.loads((P/'publication-manifest-064.json').read_text())
assert m['native_or_physical_admission']is False and m['accepted_physical_steps']==0
assert m['payload_file_list_sha256']==sha(P/'payload-files-064.json')
a=json.loads((D/'route-audit-consolidation-019/run-022-matched-query-023-corrected-001/route-report.json').read_text())
assert a['counts']['hit_segment_bank_pairs']==16 and a['counts']['proper_hits']==20 and a['construction_planes']==0
for w in a['route_witnesses']:
 tri=np.asarray(w['triangle_body_local_f32_m']);b=np.asarray(w['exact_first_hit']['barycentric'])
 s=np.asarray(w['segment_start_query_local_m']);t=np.asarray(w['segment_end_query_local_m'])
 assert np.linalg.norm(b@tri-(s+w['t']*(t-s)))<1e-12
r=json.loads((D/'root-independent-static-query-023.json').read_text())
assert r['exact_f32_matches_candidate']is True and len(r['inside_witnesses'])==17
assert r['initial_node_export_is_pre_override_baseline']is True and r['initial_export_mismatch_rows_vs_candidate']==344
r24=json.loads((D/'o12-outward-repair-024/report.json').read_text())
assert r24['candidate']['affected_q_below_1e-5']==0 and r24['candidate']['plane_violations_below_minus_0.1um']==48
lp=json.loads((D/'o12-outward-repair-024-lp-feasibility-001/lp-feasibility-report.json').read_text())
assert lp['plane_only_phaseI']['objective_sum_plane_slack_mm']>.9
assert lp['q_hard_phaseI']['objective_sum_plane_slack_mm']>.9
z=np.load(D/'o12-outward-repair-024/candidate.npz',allow_pickle=False)
X=z['points_world_f32'].astype(float);T=z['tetrahedra'];V=X[T]
det=np.linalg.det(np.stack([V[:,i]-V[:,0]for i in(1,2,3)],axis=2))
assert np.all(det>0)
print(json.dumps({'status':'PASS_RETAINED_FAILURES_AND_EXACT_WITNESSES','payload_files':len(listed),'payload_bytes':sum(r['size_bytes']for r in listed),'native_physical_steps':0,'python':sys.version}))
