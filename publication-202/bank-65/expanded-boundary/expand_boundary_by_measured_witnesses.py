#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import hashlib,json,os,subprocess,sys
import numpy as np
D=Path(__file__).resolve().parent
B=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b65-99')
S=B.parent.parent
BASE=B/'root-layer-guard-002/backtrack-0p25/candidate.npz'
TOPO=B/'attempt-002/fan-hole-ancestry-check.json'
WIT=B/'attempt-004-bone-aware-connected-patch-prep-001/witness-preflight.json'
INITIAL=B/'attempt-020-248-boundary-remesh-preflight-001/preflight-retry001.json'
CHECKER=D/'check_expanded_boundary.py'
OUT=D/'run-001'
MAX_ROUNDS=3;MAX_ADDED_FACES=32

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 pins={'candidate':sha(BASE),'topology':sha(TOPO),'witness_preflight':sha(WIT),'initial_boundary_report':sha(INITIAL),'checker':sha(CHECKER)}
 if pins['candidate']!='a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db':raise ValueError('current candidate pin changed')
 topo=json.loads(TOPO.read_text());wit=json.loads(WIT.read_text());initial=json.loads(INITIAL.read_text())
 rows=set(map(int,topo['two_ring_patch_summary'][0]['face_rows']))|set(map(int,topo['fan_expansion'][0]['fan_face_rows']))|set(map(int,wit['measured_expansion']['added_face_rows']))
 if len(rows)!=248:raise ValueError(f'expected the pinned 248-face patch, got {len(rows)}')
 with np.load(BASE,allow_pickle=False) as z:F=np.asarray(z['source_O_faces'],np.int64);ind=np.asarray(z['face_source_ancestry_indptr'],np.int64);anc=np.asarray(z['face_source_ancestry_face_rows'],np.int64)
 edge_faces=defaultdict(list)
 for r,f in enumerate(F):
  for u,v in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):edge_faces[tuple(sorted((int(u),int(v))))].append(int(r))
 current_report=initial;rounds=[];added_total=[];status='STARTED';failure=None
 for it in range(1,MAX_ROUNDS+1):
  bad_edges=set()
  for x in current_report['boundary_edge_screen']['edge_clearance_violations']:
   bad_edges.add(tuple(sorted(map(int,x['edge_nodes']))))
  for x in current_report['boundary_edge_screen']['exact_hits']+current_report['boundary_edge_screen']['uncertain']:
   bad_edges.add(tuple(sorted(map(int,x['edge_nodes']))))
  inside_nodes={int(x[0]) for x in current_report['boundary_point_screen']['inside_node_bank_pairs']}
  for u,v in current_report['boundary_edges']:
   if int(u) in inside_nodes or int(v) in inside_nodes:bad_edges.add(tuple(sorted([int(u),int(v)])))
  if not bad_edges:
   status='BOUNDARY_CLEAR_READY_FOR_REMESH_PREFLIGHT';break
  witnesses=sorted(bad_edges);to_add=set();edge_records=[];blockers=[]
  for e in witnesses:
   inc=edge_faces.get(e,[]);in_patch=[r for r in inc if r in rows];outside=[r for r in inc if r not in rows]
   edge_records.append({'edge_nodes':list(e),'incident_faces':inc,'patch_incident_faces':in_patch,'outside_adjacent_faces':outside})
   if len(inc)!=2 or len(in_patch)!=1 or len(outside)!=1:
    blockers.append({'edge_nodes':list(e),'incident_faces':inc,'patch_incident_faces':in_patch,'outside_adjacent_faces':outside,'reason':'not exactly one patch-side and one external face'})
    continue
   to_add.add(outside[0])
  if blockers:
   status='STOP_NO_CONFORMING_EXPANSION_FACE';failure={'blockers':blockers,'bad_edge_count':len(witnesses)};break
  if not to_add:
   status='STOP_NO_EXPANSION_FACES';failure={'bad_edges':witnesses};break
  if len(set(added_total)|to_add)>MAX_ADDED_FACES:
   status='STOP_EXPANSION_FACE_CAP';failure={'proposed_additional_face_rows':sorted(to_add),'cap':MAX_ADDED_FACES};break
  missing=[r for r in sorted(to_add) if ind[r+1]<=ind[r]]
  if missing:
   status='STOP_MISSING_SOURCE_ANCESTRY';failure={'face_rows':missing};break
  next_rows=sorted(rows|to_add);run_dir=OUT/f'iteration-{it:03d}';run_dir.mkdir(parents=True)
  row_file=run_dir/'rows.json';row_file.write_text(json.dumps(next_rows,separators=(',',':'))+'\n')
  plan={'iteration':it,'input_patch_face_count':len(rows),'output_patch_face_count':len(next_rows),'measured_bad_edges':witnesses,'edge_adjacency_records':edge_records,'added_face_rows':sorted(to_add),'added_faces_have_source_csr':True,'total_added_face_count':len(set(added_total)|to_add),'rows_sha256':hashlib.sha256(np.asarray(next_rows,np.int64).tobytes()).hexdigest(),'rows_json_sha256':sha(row_file)}
  (run_dir/'expansion-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
  env=os.environ.copy();env['BOUNDARY_RUN_DIR']=str(run_dir);env['BOUNDARY_ROWS_JSON']=str(row_file)
  cp=subprocess.run([sys.executable,str(CHECKER)],env=env,capture_output=True,text=True)
  (run_dir/'checker.log').write_text(cp.stdout+cp.stderr)
  if cp.returncode!=0:
   status='STOP_EXPANDED_BOUNDARY_CHECK_FAILED';failure={'iteration':it,'returncode':cp.returncode,'tail':(cp.stdout+cp.stderr)[-4000:]};break
  result=json.loads((run_dir/'preflight.json').read_text())
  rounds.append({'iteration':it,'plan_path':str(run_dir/'expansion-plan.json'),'plan_sha256':sha(run_dir/'expansion-plan.json'),'preflight_path':str(run_dir/'preflight.json'),'preflight_sha256':sha(run_dir/'preflight.json'),'patch_face_count':result['patch_face_count'],'boundary_edge_count':result['boundary_edge_count'],'boundary_node_count':result['boundary_node_count'],'boundary_gate_pass':result['boundary_gate_pass'],'inside_count':result['boundary_point_screen']['inside_count'],'exact_hit_count':result['boundary_edge_screen']['exact_hit_count'],'uncertain_count':result['boundary_edge_screen']['uncertain_count'],'min_boundary_edge_distance_mm':result['boundary_edge_screen']['min_measured_edge_to_bone_triangle_distance_mm'],'clearance_violation_count':result['boundary_edge_screen']['edge_clearance_violation_count']})
  rows=set(next_rows);added_total.extend(to_add);current_report=result
  if result['boundary_gate_pass']:
   status='BOUNDARY_CLEAR_READY_FOR_REMESH_PREFLIGHT';break
 else:
  if status=='STARTED':status='STOP_BOUNDARY_NOT_CLEAR_WITHIN_ROUND_CAP'
 receipt={'schema':'numi.human.b65-99.measured-boundary-expansion.v1','status':status,'base_candidate_path':str(BASE),'base_candidate_sha256':pins['candidate'],'initial_patch_face_count':248,'final_patch_face_count':len(rows),'boundary_edges_at_last_check':current_report['boundary_edge_count'],'boundary_nodes_at_last_check':current_report['boundary_node_count'],'total_added_face_rows':sorted(set(added_total)),'total_added_face_count':len(set(added_total)),'maximum_expansion_rounds':MAX_ROUNDS,'maximum_added_face_cap':MAX_ADDED_FACES,'pins':pins,'rounds':rounds,'final_boundary_gate_pass':bool(current_report.get('boundary_gate_pass',False)),'final_boundary_report_path':str((OUT/f"iteration-{rounds[-1]['iteration']:03d}/preflight.json") if rounds else INITIAL),'final_boundary_report_sha256':sha((OUT/f"iteration-{rounds[-1]['iteration']:03d}/preflight.json") if rounds else INITIAL),'failure':failure,'candidate_remesh_constructed':False,'disposition':'No remesh candidate is built unless the expanded fixed boundary passes all-bank point-outside, exact edge-hit-free, and 0.05mm clearance gates.'}
 out=OUT/'expansion-receipt.json';out.write_text(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False));print('receipt_sha256',sha(out))
if __name__=='__main__':main()
