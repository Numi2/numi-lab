from pathlib import Path
import csv,hashlib,json
OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027'
D071=S/'root-topology-a69-safe-subset-001/candidate.npz';C=OUT/'candidate.npz'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def readj(p):return json.loads(Path(p).read_text())
def query(p):
 with Path(p).open(newline='') as f:return {(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])):r for r in csv.DictReader(f)}
def inside_ids(rows):return {k for k,v in rows.items() if v['classification']=='inside'}
comp=readj(OUT/'full-comparison.json');sl=readj(OUT/'shape-layer-report.json');ex=readj(OUT/'full-audits-execution.json')
b=query(S/'root-topology-a69-safe-subset-001/audit-130/query/pairs.csv');c=query(OUT/'audit-130/query/pairs.csv')
assert inside_ids(b)==inside_ids(c)
depth_by_bank={}
for bank in ((61,97),(109,114)):
 rows=[]
 for key in sorted(x for x in inside_ids(b) if x[1:]==bank):
  br=b[key];cr=c[key]
  rows.append({'node_row':key[0],'baseline_classification':br['classification'],'candidate_classification':cr['classification'],'baseline_signed_distance_mm':float(br['signed_face_distance_m'])*1000,'candidate_signed_distance_mm':float(cr['signed_face_distance_m'])*1000,'progress_mm':(float(cr['signed_face_distance_m'])-float(br['signed_face_distance_m']))*1000})
 vals=[x['progress_mm'] for x in rows]
 depth_by_bank[f'{bank[0]}-{bank[1]}']={'inside_rows_compared':len(rows),'rows':rows,'minimum_progress_mm':min(vals) if vals else None,'sum_progress_mm':sum(vals)}
# Assert the stored candidate-specific reports pass all required construction gates.
assert sl['shape']['pass'] and sl['layer']['no_new_failure_identities'] and sl['layer']['added_failure_count']==0
assert ex['source_pins_unchanged'] and all(x['returncode']==0 for x in ex['runs'])
assert comp['comparisons']['candidate_vs_d071']['raw_crossings']['added_count']==0 and comp['comparisons']['candidate_vs_d071']['raw_crossings']['removed_count']==0
assert comp['comparisons']['candidate_vs_d071']['parent_pairs']['added_count']==0 and comp['comparisons']['candidate_vs_d071']['parent_pairs']['removed_count']==0
assert comp['comparisons']['candidate_vs_d071']['inside_node_bank_identities']['added_count']==0 and comp['comparisons']['candidate_vs_d071']['inside_node_bank_identities']['removed_count']==0
assert comp['comparisons']['candidate_vs_d071']['exact_surface_self_pairs']['candidate_count']==0
report={'schema':'numi.human.combined.d071.b109_114.b61_97.static-increment-receipt.v1','status':'STATIC_CONSTRUCTION_INCREMENT_CHECKED_ON_D071','candidate_path':str(C),'candidate_sha256':sha(C),'base_d071_path':str(D071),'base_d071_sha256':sha(D071),'source_a69_sha256':'a69f3b0db4c84cfae63177dd5aa96ec339b128870f4d0261c43d56e39d2a64db','input_position_patches':{'b109_114_candidate_sha256':'468254c2d44472bbcaf91b86f968536477fa2847b8581bc69003c3eb2395e0e7','b61_97_candidate_sha256':'c65ba99363fe98d403d2ce2bbf6fea2eff89b6ae1758bcd40e7274e6a2b9f8ed'},'integration':{'changed_nodes_b109_114':1074,'changed_nodes_b61_97':78,'overlap_nodes':0,'union_changed_nodes':1152,'changed_connectivity_faces_from_d071':12,'d071_topology_and_nonposition_metadata_preserved':True},'shape':sl['shape'],'layer':sl['layer'],'full_audit':{'audit130':comp['audit130_candidate'],'audit159':comp['audit159_candidate'],'execution_receipt_path':str(OUT/'full-audits-execution.json'),'execution_receipt_sha256':sha(OUT/'full-audits-execution.json')},'identity_comparison':{'versus_d071':comp['comparisons']['candidate_vs_d071'],'versus_a69':comp['comparisons']['candidate_vs_a69'],'comparison_report_path':str(OUT/'full-comparison.json'),'comparison_report_sha256':sha(OUT/'full-comparison.json')},'target_inside_depth_vs_d071':depth_by_bank,'evidence_files':{'preflight':{'path':str(OUT/'preflight.json'),'sha256':sha(OUT/'preflight.json')},'shape_layer':{'path':str(OUT/'shape-layer-report.json'),'sha256':sha(OUT/'shape-layer-report.json')},'audit130':{'path':str(OUT/'audit-130/summary.json'),'sha256':sha(OUT/'audit-130/summary.json')},'audit159':{'path':str(OUT/'audit-159/report.json'),'sha256':sha(OUT/'audit-159/report.json')}},'geometry_admitted':False,'physical_steps':0,'limitations':['Static reference-pose construction checks only. No physical simulation, conservative material-mass transfer, closed-solid containment, or native tissue admission.']}
p=OUT/'final-receipt.json';p.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'receipt_path':str(p),'receipt_sha256':sha(p),'candidate_sha256':sha(C),'inside_depth_by_bank':{k:{'rows':v['inside_rows_compared'],'min_mm':v['minimum_progress_mm'],'sum_mm':v['sum_progress_mm']} for k,v in depth_by_bank.items()},'audit130_self_pairs':report['full_audit']['audit130']['exact_PLC_surface_pairs'],'audit159_banks':report['full_audit']['audit159']['bank_count'],'vs_d071_raw_added_removed':[comp['comparisons']['candidate_vs_d071']['raw_crossings']['added_count'],comp['comparisons']['candidate_vs_d071']['raw_crossings']['removed_count']],'vs_d071_parent_added_removed':[comp['comparisons']['candidate_vs_d071']['parent_pairs']['added_count'],comp['comparisons']['candidate_vs_d071']['parent_pairs']['removed_count']]},indent=2))
