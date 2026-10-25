from pathlib import Path
import csv,gzip,json,hashlib,collections,statistics
import numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069/connected-reference-capacity-112')
R1=D/'run-001';R2=D/'run-002';Q=Path('/Users/numi/Work/numi-human-connected-surface-069/connected-cell-witness-103/run-001/baseline/query')
REF=Path('/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/accepted-transfer-row-export-075/run-001/source-preflight/accepted-coarse-reference.csv')
REF_SHA='9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(REF)!=REF_SHA: raise RuntimeError('coarse reference input pin mismatch')
def read_csv(p): return list(csv.DictReader(Path(p).open(newline='')))
a=read_csv(R1/'column-capacity-proposals.csv');b=read_csv(R2/'column-capacity-proposals.csv');ref=read_csv(REF)
if len(a)!=2411 or len(b)!=2411 or len(ref)!=2411: raise RuntimeError('row coverage mismatch')
for i,(x,y,z) in enumerate(zip(a,b,ref)):
 if int(x['column_row'])!=i or int(y['column_row'])!=i or int(z['coarse_row'])!=i: raise RuntimeError('row domain mismatch')
# Compare all proposal decisions and coordinates despite the clearer run-002 headings.
for i,(x,y) in enumerate(zip(a,b)):
 for k0,k1 in [('capacity_status','capacity_status'),('OD_first_entry_body','OD_first_entry_body'),
  ('OD_first_entry_bone','OD_first_entry_bone'),('OD_first_entry_triangle','OD_first_entry_triangle'),
  ('OD_first_entry_t','OD_first_entry_t'),('OD_soft_path_capacity_m_after_50um_construction_clearance','OD_soft_path_capacity_m_after_50um_construction_clearance'),
  ('ID_first_entry_body','ID_first_entry_body'),('ID_first_entry_bone','ID_first_entry_bone'),
  ('ID_first_entry_triangle','ID_first_entry_triangle'),('ID_first_entry_distance_m','ID_first_entry_distance_m'),
  ('D_action','D_action'),('D_safe_parameter_from_I','D_safe_parameter_from_I'),('D_safe_x_m','D_safe_x_m'),
  ('D_safe_y_m','D_safe_y_m'),('D_safe_z_m','D_safe_z_m'),('exact_segment_failures','exact_segment_failures')]:
  if x[k0]!=y[k1]: raise RuntimeError(f'proposal mismatch row {i} field {k0}: {x[k0]} != {y[k1]}')
 if int(x['OI_crosses_bone'])!=int(y['OI_path_intersects_or_endpoint_inside']): raise RuntimeError(f'OI policy mismatch {i}')
 if int(x['OI_equal_t_tie'])!=int(y['OI_equal_t_tie']) or int(x['ID_equal_t_tie'])!=int(y['ID_equal_t_tie']) or int(x['OD_equal_t_tie'])!=int(y['OD_equal_t_tie']): raise RuntimeError(f'tie mismatch {i}')
 if int(x['O_to_I_crossing_count'])!=int(y['O_to_I_first_entry_bank_count']): raise RuntimeError(f'OI entry count mismatch {i}')
# Content-level exact audit equality; gzip metadata itself may differ.
def records(path):
 with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
seg1=records(R1/'exact-segment-audits.jsonl.gz');seg2=records(R2/'exact-segment-audits.jsonl.gz')
if seg1!=seg2: raise RuntimeError('exact segment audit contents differ between 001/002')
pairs={(int(x['query_row']),int(x['bank_body']),int(x['bank_stable_id'])):x['classification'] for x in read_csv(Q/'pairs.csv')}
by_status=collections.Counter(x['capacity_status'] for x in b)
by_donor=collections.defaultdict(collections.Counter);by_src=collections.defaultdict(collections.Counter)
for i,x in enumerate(b):
 by_donor[int(ref[i]['donor_body'])][x['capacity_status']]+=1
 by_src[int(ref[i]['source_dominant_body'])][x['capacity_status']]+=1
# First-entry and exit event detail for affected rows. Each classifier hit is
# retained with rational t and barycentrics; event orientation alternates from
# its exact endpoint classification.
per_segment=collections.defaultdict(list);segment_status=collections.Counter();first_entry_counts=collections.Counter()
for rec in seg2:
 segment_status[(rec['route'],rec['status'])]+=1
 if rec['status']!='PASS': raise RuntimeError('non-PASS exact segment')
 result=rec['result']; st=result['start_status']; L=result['segment_length_m']
 if result['first_entry'] is not None: first_entry_counts[(rec['column_row'],rec['route'])]+=1
 for h in result['proper_hits']:
  kind='entry' if st=='outside' else 'exit'
  per_segment[(rec['column_row'],rec['route'])].append({'body':rec['body'],'stable_bone_id':rec['bone'],
    'event':kind,'triangle_stable_id':h['triangle_stable_id'],'t':h['t'],
    't_numerator':h['t_numerator'],'t_denominator':h['t_denominator'],
    'distance_from_route_start_m':h['t']*L,'hit_world_m':h['hit_world'],'barycentric':h['barycentric']})
  st='inside' if st=='outside' else 'outside'
 if st!=result['end_status']: raise RuntimeError('event sequence parity mismatch')
for i,y in enumerate(b):
 if int(y['I_to_D_first_entry_bank_count'])!=first_entry_counts[(i,'ID')]: raise RuntimeError(f'ID first-entry bank count mismatch {i}')
 if int(y['O_to_D_first_entry_bank_count'])!=first_entry_counts[(i,'OD')]: raise RuntimeError(f'OD first-entry bank count mismatch {i}')
 if int(y['O_to_I_first_entry_bank_count'])!=first_entry_counts[(i,'OI')]: raise RuntimeError(f'OI first-entry bank count mismatch {i}')
endpoint_inside=[]
for row in b:
 i=int(row['column_row']); o=sorted([[body,bone] for (q,body,bone),v in pairs.items() if q==i and v=='inside'])
 ii=sorted([[body,bone] for (q,body,bone),v in pairs.items() if q==i+2411 and v=='inside'])
 d=sorted([[body,bone] for (q,body,bone),v in pairs.items() if q==i+4822 and v=='inside'])
 if row['capacity_status']=='REQUIRES_SKIN_RECONSTRUCTION':
  endpoint_inside.append({'column_row':i,'donor_body':int(ref[i]['donor_body']),
   'source_dominant_body':int(ref[i]['source_dominant_body']),
   'O_endpoint_inside_banks':o,'I_endpoint_inside_banks':ii,'D_endpoint_inside_banks':d,
   'O_to_I_exact_events':per_segment.get((i,'OI'),[]),
   'I_to_D_exact_events':per_segment.get((i,'ID'),[]),
   'O_to_D_exact_events':per_segment.get((i,'OD'),[])})
proposed=[x for x in b if x['capacity_status']=='PROPOSED_D_SHORTENING']
z=np.load(Path('/Users/numi/Work/numi-human-connected-surface-069/connected-cell-witness-103/run-001/baseline/candidate.npz'),allow_pickle=False)
pts=np.asarray(z['points_world_f32'],dtype=np.float64);alphas=[];distances=[];left=[]
for x in proposed:
 i=int(x['column_row']);alpha=float(x['D_safe_parameter_from_I']);ID=float(x['ID_first_entry_distance_m'])
 I=pts[i+2411];D0=pts[i+4822];length=float(np.linalg.norm(D0-I));safe=np.asarray([float(x[k]) for k in ('D_safe_x_m','D_safe_y_m','D_safe_z_m')])
 if not (0<alpha<1) or np.linalg.norm(safe-(I+alpha*(D0-I)))>2e-12: raise RuntimeError(f'D proposal not on I-D {i}')
 remains=ID-alpha*length
 if abs(remains-5e-5)>3e-9: raise RuntimeError(f'D proposal clearance mismatch {i}: {remains}')
 alphas.append(alpha);distances.append(ID);left.append(remains)
tight=[x for x in b if x['capacity_status']=='NO_FEASIBLE_D_SHORTENING']
if len(tight)!=1: raise RuntimeError('expected one clearance-exhausted row')
x=tight[0];i=int(x['column_row']); tight_events=per_segment[(i,'ID')]
entry=[h for h in tight_events if h['event']=='entry']
if len(entry)!=1: raise RuntimeError('tight row does not have unique exact entry')
h=entry[0]
summary={
 'schema':'connected-reference-capacity-112-verified-summary-v2',
 'summary_generator_sha256':sha(Path(__file__).resolve()),
 'run001_report_sha256':sha(R1/'report.json'),'run002_report_sha256':sha(R2/'report.json'),
 'run002_source_sha256':sha(D/'run_capacity_audit_112_run002.py'),
 'run002_helper_sha256':sha(D/'human_skin_fat_reference_capacity.py'),
 'run002_test_sha256':sha(D/'test_human_skin_fat_reference_capacity.py'),
 'run002_input_pins_sha256':sha(R2/'input-pins.json'),
 'run002_proposals_sha256':sha(R2/'column-capacity-proposals.csv'),
 'run002_exact_segment_receipts_sha256':sha(R2/'exact-segment-audits.jsonl.gz'),
 'run001_original_source_hashes_verified':json.loads((R1/'input-pins.json').read_text()),
 'paired_run_comparison':{'all_2411_capacity_states_and_D_proposals_identical':True,
    'all_exact_segment_receipts_content_identical':True,'run001_geometry_helper_semantics_preserved':True,
    'run002_clearer_endpoint_inside_columns_present':True},
 'capacity_status_counts':dict(by_status),
 'counts_by_donor_body':{str(k):dict(v) for k,v in sorted(by_donor.items())},
 'counts_by_source_dominant_body':{str(k):dict(v) for k,v in sorted(by_src.items())},
 'exact_route_bank_audit_counts':{'OI_candidate_segments_PASS':segment_status[('OI','PASS')],
  'ID_candidate_segments_PASS':segment_status[('ID','PASS')],'OD_candidate_segments_PASS':segment_status[('OD','PASS')],
  'nonpass':sum(v for (r,s),v in segment_status.items() if s!='PASS'),
  'AABB_disjoint_proofs':json.loads((R2/'report.json').read_text())['aabb_disjoint_segment_proofs'],
  'cross_bank_equal_rational_parameter_ties':0},
 'D_shortening':{'proposal_count':len(proposed),'parameter_from_I_min_median_max':[min(alphas),statistics.median(alphas),max(alphas)],
  'I_to_first_entry_distance_m_min_median_max':[min(distances),statistics.median(distances),max(distances)],
  'residual_clearance_m_min_max':[min(left),max(left)],
  'verified_original_I_D_segment':True,'verified_50um_clearance':True},
 'skin_reconstruction_columns':endpoint_inside,
 'clearance_exhausted_witness':{'column_row':i,'donor_body':int(ref[i]['donor_body']),
  'source_dominant_body':int(ref[i]['source_dominant_body']),'bank_body':h['body'],
  'stable_bone_id':h['stable_bone_id'],'triangle_stable_id':h['triangle_stable_id'],
  'exact_t':{'numerator':h['t_numerator'],'denominator':h['t_denominator']},
  'distance_from_I_m':h['distance_from_route_start_m'],'remaining_after_50um_m':h['distance_from_route_start_m']-5e-5,
  'hit_world_m':h['hit_world_m'],'barycentric':h['barycentric']},
 'interpretation':{'reconstruction':'Any O/I endpoint inside a typed bone bank or a proper O-I crossing requires skin reconstruction; no D clamp is offered for these columns.',
  'shortening':'D proposals preserve O/I and lie on the original I-D segment, stopping the requested 50 µm before the first exact entry. These are path-capacity diagnostics only, not tissue-thickness prescriptions.',
  'body_groups':'Integer body/source-body IDs from 075 are reported without inferred anatomical names.'},
 'limitations':['The source helper does not prove each bank mesh is watertight or self-intersection-free.',
  'No mesh, topology, mass, native state, or source geometry was modified.',
  'No tissue thickness or physiological adequacy is claimed.']}
summary_path=R2/'verified-capacity-summary.json';summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print('summary',summary_path,sha(summary_path))
print(json.dumps({'counts':dict(by_status),'skin_rows':len(endpoint_inside),'donor_groups':len(by_donor),
 'tight':summary['clearance_exhausted_witness'],'summary_sha256':sha(summary_path)},indent=2))
