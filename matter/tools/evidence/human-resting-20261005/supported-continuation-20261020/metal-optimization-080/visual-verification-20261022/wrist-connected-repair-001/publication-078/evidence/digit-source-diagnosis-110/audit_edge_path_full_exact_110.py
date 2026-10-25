from pathlib import Path
import csv,json,hashlib,sys,time
import numpy as np
sys.path.insert(0,'/Users/numi/Work/numi-human-connected-surface-069/publication-075/canonical/matter')
from root_exact_segment_witness import classify_segment_triangle

W=Path('/Users/numi/Work/numi-human-connected-surface-069')
E=Path('/Users/numi/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=W/'digit-source-diagnosis-110'
BASE=W/'connected-cell-witness-103/run-001/baseline'
COMPACT=W/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
REPORTPATH=D/'report-002.json'
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
OUT=D/'report-004.json'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

t0=time.monotonic()
with np.load(BASE/'candidate.npz',allow_pickle=False) as z: coarse=np.asarray(z['points_world_f32'])
with np.load(COMPACT,allow_pickle=False) as z: ccurrent=np.asarray(z['current037_points_m_f32'])
if not np.array_equal(coarse,ccurrent):raise RuntimeError('baseline103 endpoints no longer bitwise equal compact current037')
path_report=json.load(open(REPORTPATH))['detailed_surface_patch']['shortest_surface_edge_path']
path=np.asarray([v['world_rest_m'] for v in path_report['vertices']],dtype=np.float64)
if path.shape!=(19,3):raise RuntimeError('unexpected fine path shape')
# exact-emitted bank export; values are decimal round trips of native F32 components.
bodycsv=BANKDIR/'exact-emitted.bodies.csv'; tric=BANKDIR/'exact-emitted.triangles.csv'
bodies={}
with bodycsv.open(newline='') as f:
 for row in csv.DictReader(f):
  b=int(row['body_index'])
  if b in (65,66):
   p=np.array([float(row[f'position_{a}_m']) for a in 'xyz'],dtype=np.float32).astype(np.float64)
   q=np.array([float(row[f'orientation_q{a}']) for a in 'xyzw'],dtype=np.float32).astype(np.float64)
   # Use the exact emitted world-to-body matrix checked in root audit 039.
   M=np.array([float(row[f'world_to_body_m{i}{j}']) for i in range(3) for j in range(3)],dtype=np.float64).reshape(3,3)
   off=np.array([float(row[f'com_offset_{a}_m']) for a in 'xyz'],dtype=np.float64)
   bodies[b]={'position':p,'M':M,'offset':off,'raw':row}
banks={(65,99):[],(66,100):[]}
with tric.open(newline='') as f:
 for row in csv.DictReader(f):
  key=(int(row['body_index']),int(row['bone_stable_id']))
  if key not in banks:continue
  corners=np.array([[float(row[f'v{i}{a}_m']) for a in 'xyz'] for i in range(3)],dtype=np.float32).astype(np.float64)
  banks[key].append((int(row['triangle_stable_id']),corners))
expected={(65,99):342,(66,100):174}
for k,n in expected.items():
 if len(banks[k])!=n:raise RuntimeError(f'bank {k} count {len(banks[k])} != pinned {n}')

def localize(world,body):
 b=bodies[body]
 return (np.asarray(world,dtype=np.float64)-b['position'])@b['M'].T-b['offset']

def audit_segments(name,world_points):
 result={}
 for (body,bone),items in banks.items():
  local=localize(world_points,body)
  hits=[];counts={'proper_interior_hit':0,'boundary_hit':0,'coplanar_requires_2d_test':0,'invalid_degenerate_triangle':0,'parallel_disjoint':0,'disjoint':0};aabb_pairs=0;tested_pairs=0
  # The hit source is small: 18 polyline segments x 2 complete typed banks.
  for si,(a,b) in enumerate(zip(local[:-1],local[1:])):
   alo=np.minimum(a,b);ahi=np.maximum(a,b)
   for tid,tri in items:
    if not (np.any(np.max(tri,axis=0)<alo) or np.any(np.min(tri,axis=0)>ahi)):aabb_pairs+=1
    hit=classify_segment_triangle(a,b,tri);tested_pairs+=1
    cls=hit['classification'];counts[cls]=counts.get(cls,0)+1
    if cls not in ('disjoint','parallel_disjoint'):
     hits.append({'segment_index':si,'triangle_stable_id':tid,'classification':cls,
       't':hit.get('t'),'barycentric':hit.get('barycentric'), 'endpoint_a_local_m':a.tolist(), 'endpoint_b_local_m':b.tolist()})
  result[f'{body}:{bone}']={'body_index':body,'bone_stable_id':bone,'bank_triangle_count':len(items),
     'segment_count':len(world_points)-1,'all_segment_bank_pairs':(len(world_points)-1)*len(items),
     'aabb_candidate_pairs':aabb_pairs,'exact_predicate_tests':tested_pairs,'aabb_rejected_pairs':(len(world_points)-1)*len(items)-aabb_pairs,
     'classification_counts_among_exact_tests':counts,'witnesses':hits}
 return result

# Coarse straight chord and fine detailed surface path are tested in the same world frame,
# then transformed into each exact native bank's query-local coordinates.
coarse_pair=np.asarray([coarse[329],coarse[340]],dtype=np.float64)
coarse_by_bank={}
for (body,bone),items in banks.items():
 a,b=localize(coarse_pair,body)
 hits=[];counts={'proper_interior_hit':0,'boundary_hit':0,'coplanar_requires_2d_test':0,'invalid_degenerate_triangle':0,'parallel_disjoint':0,'disjoint':0};aabb_pairs=0;tested_pairs=0
 for tid,tri in items:
  if not (np.any(np.max(tri,axis=0)<np.minimum(a,b)) or np.any(np.min(tri,axis=0)>np.maximum(a,b))):aabb_pairs+=1
  h=classify_segment_triangle(a,b,tri);tested_pairs+=1;counts[h['classification']]=counts.get(h['classification'],0)+1
  if h['classification'] not in ('disjoint','parallel_disjoint'):
   hits.append({'triangle_stable_id':tid,'classification':h['classification'],'t':h.get('t'),'barycentric':h.get('barycentric')})
 coarse_by_bank[f'{body}:{bone}']={'body_index':body,'bone_stable_id':bone,'bank_triangle_count':len(items),
   'segment_count':1,'all_segment_bank_pairs':len(items),'aabb_candidate_pairs':aabb_pairs,'exact_predicate_tests':tested_pairs,
   'aabb_rejected_pairs':len(items)-aabb_pairs,'classification_counts_among_exact_tests':counts,'witnesses':hits}
fine_by_bank=audit_segments('fine',path)
anchors={'coarse_row_329_to_detail_row':int(path_report['vertices'][0]['detail_row']),
         'coarse_row_340_to_detail_row':int(path_report['vertices'][-1]['detail_row'])}
anchor_deltas_mm=[float(np.linalg.norm(path[0]-coarse[329].astype(np.float64))*1000),float(np.linalg.norm(path[-1]-coarse[340].astype(np.float64))*1000)]
report={'status':'SAVED_GEOMETRY_EXACT_SEGMENT_COMPARISON','scope':'Exact binary-rational segment/triangle narrowphase on one saved O-O chord and all 18 segments of its mapped detailed NHSKIN edge path against every triangle in complete exact-emitted banks (65:99, 66:100). No broadphase/query rerun, geometry mutation, or physical admission.',
 'input_pins':{'baseline103_candidate_sha256':sha(BASE/'candidate.npz'),'compact084_mesh_sha256':sha(COMPACT),
  'path_report_sha256':sha(REPORTPATH),'detailed_payload_sha256':sha(E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'),
  'accepted_transfer_rows_sha256':sha(E/'accepted-transfer-row-export-075/run-001/source-preflight/accepted-transfer-rows.csv'),
  'transfer_W_sha256':sha(W/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002/Wnew-detail-to-084-nodes.npz'),
  'W_semantics_correction_sha256':sha(W/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002/transfer-report-correction.json'),
  'exact_bank_triangles_sha256':sha(tric),'exact_bank_bodies_sha256':sha(bodycsv),
  'exact_classifier_sha256':sha(W/'publication-075/canonical/matter/root_exact_segment_witness.py')},
 'frame':{'transform':'the audited native 039 world-to-body matrix from emitted float32 quaternion, and emitted COM offset; local=(world-position)@M.T-offset','anchor_residual_to_baseline_coarse_endpoint_mm':anchor_deltas_mm,
  'fine_path_source':'accepted-transfer-rows.csv detail_rest_world_m coordinates; exact predicates use parsed binary64 coordinates as supplied','coarse_source':'baseline103 points_world_f32; exact predicates use emitted binary32 values promoted to binary64','W_role':'Wnew is a displacement prolongation operator; it is not multiplied by absolute positions. At the pinned zero-displacement baseline the accepted transfer detail_rest_world coordinates define the fine path.'},
 'coarse_chord':coarse_by_bank,'fine_detailed_path':fine_by_bank,
 'interpretation_limits':['A proper interior hit is an exact segment/triangle intersection. Boundary hits and coplanar cases are retained separately and are not silently treated as disjoint.',
  'The fine NHSKIN path coordinates come from the accepted transfer row export; this audit did not run a full detailed-mesh vs bone-bank surface intersection test.',
  'This two-bank, one-edge witness is local mechanism evidence only; it is not all-bank clearance or physical qualification.'],
 'elapsed_seconds':time.monotonic()-t0}
OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(OUT),'coarse':{k:{'aabb_candidate_pairs':v['aabb_candidate_pairs'],'exact_predicate_tests':v['exact_predicate_tests'],'counts':v['classification_counts_among_exact_tests']} for k,v in coarse_by_bank.items()},'fine':{k:{'all_pairs':v['all_segment_bank_pairs'],'aabb_candidates':v['aabb_candidate_pairs'],'exact_tests':v['exact_predicate_tests'],'counts':v['classification_counts_among_exact_tests']} for k,v in fine_by_bank.items()},'elapsed_seconds':report['elapsed_seconds'],'anchor_residual_to_baseline_coarse_endpoint_mm':anchor_deltas_mm},indent=2))
