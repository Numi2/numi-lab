from pathlib import Path
import json,hashlib,csv,sys,subprocess,time,collections,os,gzip,traceback
import numpy as np
from scipy import sparse
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=Path('/Users/n/Work/numi-human-connected-surface-069');M=D/'publication-075/canonical/matter/tools'
sys.path.insert(0,str(M))
import human_skin_fat_connected_repair as repair
from human_skin_fat_embedded_surface import MappedSkinSurface
from human_skin_fat_boundary_workset import select_boundary_witness_workset
HELPER_DIR=M
RAY_DIR=M
import human_skin_fat_bone_clearance as outward
from human_skin_fat_bone_clearance import (build_edge_span_midpoint_samples, validate_sample_query_rows,
    validate_sample_points_csv)
# Set to the frozen identity-domain audit extension after its tests complete.
AUDIT_PATH=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
sys.path.insert(0,str(AUDIT_PATH.parent))
import human_skin_fat_bone_audit as audit
P=D/'connected-exterior-clearance-102/run-001'
assert not P.exists();P.mkdir(parents=True)
INPUT=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
EXPECTED='5a35470223ce12343cc217be7822d182802e99f1c423bb52b8409f153f9a49db'
BANK=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
BASEQUERY=E/'E/generic-static-query-042/run-037006-current-001'
BASEAUDIT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/E/generic-static-query-043/run-037006-query-row-edges-001/tet-edge-report.json")
BIN=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query')
EXPORT=E/'E/generic-static-query-042/prepare_generic_query_points.py'
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(INPUT)==EXPECTED
z=np.load(INPUT);O=z['current037_points_m_f32'];X=O.copy();R=z['reference_points_m'];T=z['tetrahedra_zero_based'];mat=z['material_ids']
assert X.dtype==np.float32 and X.shape==(7360,3)
n=len(X)
# Fresh 084 start only; no 089/091 or later displacement is loaded.
assert n==7360 and T.shape==(32796,4)
ADAPTER_DIR=E/'E/accepted-transfer-row-adapter-075'
sys.path.insert(0,str(ADAPTER_DIR))
from load_transfer_075 import load_transfer075
EXPORT075=E/'accepted-transfer-row-export-075/run-001/source-preflight'
TRANSFER_ROWS=EXPORT075/'accepted-transfer-rows.csv'
COARSE_REFERENCE=EXPORT075/'accepted-coarse-reference.csv'
COARSE_FACES=EXPORT075/'accepted-coarse-faces.csv'
DETAIL_PAYLOAD=E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'
assert sha(TRANSFER_ROWS)=='71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14'
assert sha(COARSE_REFERENCE)=='9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2'
assert sha(COARSE_FACES)=='7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2'
assert sha(DETAIL_PAYLOAD)=='7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'
W075,detail_origin,coarse_origin,detail_faces=load_transfer075(
    TRANSFER_ROWS,COARSE_REFERENCE,COARSE_FACES,DETAIL_PAYLOAD,
    expected_sha256={'transfer_rows':'71e0ff115f74f1c072dc39e25d8c0de869c892dbfaf3424c792c2f674172fb14',
      'coarse_reference':'9e26673335c571d5920f5cd43e52daf4b240de72feadf6d9e1cef92cf80807a2',
      'coarse_faces':'7311267392275649abeb3ecaf6a5ca43eb4d67cf2a9b7f13a9c692587c36b1e2',
      'detail_payload':'7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'})
assert W075.shape==(61950,2411) and detail_origin.shape==(61950,3) and detail_faces.shape==(108987,3)
assert np.array_equal(coarse_origin,O[:2411]),'075 coarse F32 origin differs from 037 mechanics prefix'
MAP=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/transfer-map-075-084-002'
map_report=json.loads((MAP/'transfer-report.json').read_text())
assert map_report['inputs_sha256']['candidate']==EXPECTED
W=sparse.load_npz(MAP/'Wnew-detail-to-084-nodes.npz').tocsr()
assert sha(MAP/'Wnew-detail-to-084-nodes.npz')==map_report['outputs_sha256']['Wnew-detail-to-084-nodes.npz']
assert W.shape==(61950,n) and not np.any(W.data<0)

embedded=MappedSkinSurface(O.copy(),detail_origin,detail_faces,W,max_dihedral_change_degrees=15.)
assert embedded.passes(X)
assert np.array_equal(embedded.positions(O).view(np.uint32),detail_origin.view(np.uint32))

RAYFIELD=D/'closest-exit-clearance-098/source-field-audit-002/source-outward-field.npz'
RAYFIELD_REPORT=D/'closest-exit-clearance-098/source-field-audit-002/source-field-audit.json'
assert sha(RAYFIELD)=='263d3be786bea0192019d0070060218fe4aea6a66b3b804a87cdcaa0715c767d'
assert sha(RAYFIELD_REPORT)=='976b029e2d97451a926ebd7bd592b6b248930952fba28dacf6cd1e380f452e2f'
ray_field_npz=np.load(RAYFIELD,allow_pickle=False)
source_field=np.asarray(ray_field_npz['source_outward_direction_by_compact_output_row'],dtype=np.float64)
assert source_field.shape==(n,3) and np.isfinite(source_field).all()
assert np.array_equal(ray_field_npz['source_input_row_by_compact_output_row'],z['source_input_row'])
assert np.array_equal(ray_field_npz['source_global_node_ids'],z['source_global_node_ids'])
field_audit=json.loads(RAYFIELD_REPORT.read_text())
assert field_audit['geometry_gates']['ray_queries_run'] is False
assert field_audit['field_summary']['finite_rows']==n
assert field_audit['field_summary']['generated_direction_disagreement_count']==0
# Qualify the direction field in the exact active reference frame.
source_R=np.empty((7233,3),dtype=np.float64)
direct=np.flatnonzero(z['source_input_row']>=0)
source_R[z['source_input_row'][direct]]=R[direct]
source_expected=source_R[:2411]-source_R[4822:7233]
source_expected/=np.linalg.norm(source_expected,axis=1)[:,None]
assert np.max(np.linalg.norm(source_field[direct]-source_expected[z['source_input_row'][direct]%2411],axis=1))<1e-10
assert sha(RAY_DIR/'human_skin_fat_bone_clearance.py')=='c66d53a10ec7a5c93be1f691d3eb4428dc235f3343361219c69e21943acd654e'
SOURCE_ANCESTRY=E/'native-candidate020-failure-witness-026/physical-cell-quality-001/palm-prism-rank-retriangulation-003/candidate.npz'
assert sha(SOURCE_ANCESTRY)=='640b7bc56013748b26d5e4e12790f85a472602ee6a4732f5773243df6e2147cc'
parent_source=np.load(SOURCE_ANCESTRY,allow_pickle=False)
rebuilt_field,rebuilt_report=outward.build_reference_outward_field(R,
    source_global_node_ids=z['source_global_node_ids'],output_source_input_row=z['source_input_row'],
    native_parent_tetrahedra=parent_source['old_tetrahedra'],
    generated_output_rows=z['newnode_ancestry_output_row'],
    generated_parent_tet_rows=z['newnode_ancestry_source_tet_row'],
    generated_parent_barycentric=z['newnode_ancestry_barycentric'])
assert np.array_equal(rebuilt_field,source_field)
(P/'reference-outward-field-parity.json').write_text(json.dumps({
    'bitwise_field_parity_with_corrected_002':True,
    'source_ancestry_sha256':sha(SOURCE_ANCESTRY),'report':rebuilt_report},indent=2)+'\n')
ray_receipts=[]
(P/'execution-owner-pins.json').write_text(json.dumps({
 'candidate_sha256':sha(INPUT),'fresh_start':'compact 084 current037_points_m_f32; X=O bitwise',
 'mapping_sha256':sha(MAP/'Wnew-detail-to-084-nodes.npz'),
 'mapping_report_sha256':sha(MAP/'transfer-report.json'),'construction_owner_sha256':sha(M/'human_skin_fat_connected_repair.py'),
 'embedded_owner_sha256':sha(M/'human_skin_fat_embedded_surface.py'),
 'surface_separator_owner_sha256':sha(M/'human_skin_fat_surface_separation.py'),
 'generated_thickness_owner_sha256':sha(M/'human_skin_fat_generated_layer_constraints.py'),
 'source_field_helper_sha256':sha(RAY_DIR/'human_skin_fat_bone_clearance.py'),
 'source_field_npz_sha256':sha(RAYFIELD),'source_field_report_sha256':sha(RAYFIELD_REPORT),
 'source_field_semantics':'098 audited field; unit O-D direction indexed by source input row modulo 2411 and prolonged to generated nodes through all declared native-parent barycentric choices; construction direction only',
 'starting_candidate_sha256':sha(INPUT),'mapped_skin_max_dihedral_change_degrees':15.,
 'outer_authoring_workset':'exact outer-edge/node witnesses and two connected boundary rings; remaining proposed inner nodes stay free; not a runtime constraint',
 'step_limit_mm':2.,'max_incremental_gradient':None,'dihedral_active_fraction':.75,'dihedral_constraint_passes':3,'iteration_limit':4,
 'static_authoring_only':True,'accepted_physical_steps':0},indent=2)+'\n')
tri,ids,oldgids,bodies,rot,off,pos,keys,pins,exact=audit.read_bank_inputs(BANK,E/'route-audit-counter-fix-030/canonical/matter')
faces=np.concatenate([T[:,list(k)]for k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]])
sf=np.sort(faces,axis=1);_,first,counts=np.unique(sf,axis=0,return_index=True,return_counts=True)
assert not np.any(counts>2)
BF=faces[first[counts==1]]
np.save(P/'boundary-faces.npy',BF)
# Preserve the already declared original O-I/I-D projected thickness envelope.
rr=[];cc=[];vv=[];lb=[]
for col in range(2411):
 for layer in [0,1]:
  a=layer*2411+col;b=a+2411;d=O[a].astype(float)-O[b];length=np.linalg.norm(d);normal=d/length
  for sign,bound in [(1,.5*length),(-1,-1.5*length)]:
   row=len(lb)
   for node,w in [(a,sign),(b,-sign)]:
    for k in range(3):rr.append(row);cc.append(3*node+k);vv.append(w*normal[k])
   lb.append(bound)
Loriginal=sparse.coo_matrix((vv,(rr,cc)),shape=(len(lb),3*n)).tocsr();lboriginal=np.asarray(lb)
from human_skin_fat_generated_layer_constraints import build_generated_boundary_layer_inequalities
marker_rows=list(csv.DictReader((D/'whole-reference-tetgen-084-bounded/facet-marker-map.csv').open()))
marker_parent={int(r['marker']):(r['surface'],int(r['source_face_row'])) for r in marker_rows}
outer_boundary_faces=z['surface_faces_zero_based'][np.array([marker_parent[int(m)][0]=='O' for m in z['surface_face_markers']])]
outer_boundary_nodes=np.unique(outer_boundary_faces)
assert np.isin(np.unique(W.indices),outer_boundary_nodes).all()
Lgenerated,lbgenerated,generated_report=build_generated_boundary_layer_inequalities(
 R,O,z['source_input_row'],{s:z['source_'+s+'_faces'] for s in ['O','I','D']},
 z['surface_faces_zero_based'],z['surface_face_markers'],marker_parent)
assert generated_report['generated_boundary_nodes_covered']==125
assert generated_report['generated_interior_nodes_excluded']==2
(P/'generated-layer-constraints.json').write_text(json.dumps(generated_report,indent=2)+'\n')
sparse.save_npz(P/'generated-layer-rows.npz',Lgenerated);np.save(P/'generated-layer-targets.npy',lbgenerated)
Lbase=sparse.vstack([Loriginal,Lgenerated],format='csr');lbbase=np.r_[lboriginal,lbgenerated]
from human_skin_fat_surface_separation import build_pair_inequalities
sep_report=json.loads((D/'E/embedded-self-separator-078/run-002/report.json').read_text())
sep_rows=[];sep_targets=[];sep_pro=[]
def add_separator_pair(fa,fb):
 A,b,pro=build_pair_inequalities(detail_faces[fa],detail_faces[fb],detail_origin,W,O,O,pair_id=[int(fa),int(fb)])
 sep_rows.append(A);sep_targets.extend(b.tolist());sep_pro.append(pro)
for pair in sep_report['pairs']:add_separator_pair(*pair['face_rows'])
def hard_rows():
 S=sparse.vstack(sep_rows,format='csr') if sep_rows else sparse.csr_matrix((0,3*n))
 return sparse.vstack([Lbase,S],format='csr'),np.r_[lbbase,np.asarray(sep_targets)]
L,lb=hard_rows()
(P/'separator-provenance.json').write_text(json.dumps(sep_pro,indent=2)+'\n')

sparse.save_npz(P/'layer-rows.npz',L);np.save(P/'layer-targets.npy',lb)
lookup={(int(r[0]),int(r[1]),int(r[3])):k for k,r in enumerate(ids)}
BANK_GEOMETRY={}
for body,bone in keys:
 body=int(body);bone=int(bone);mask=(ids[:,0]==body)&(ids[:,1]==bone)
 bank_tri=tri[mask];bank_ids=ids[mask,3].astype(np.int64)
 if len(bank_tri)==0 or len(np.unique(bank_ids))!=len(bank_ids):raise ValueError('empty or duplicate complete-bank rows')
 BANK_GEOMETRY[(body,bone)]={'triangles':bank_tri,'triangle_ids':bank_ids,
   'aabb_min':bank_tri.min(axis=(0,1)),'aabb_max':bank_tri.max(axis=(0,1))}
EXACT_PREDICATE_SHA='d6a2efe451e7743b659d74c274d1372a5ed2a765511a8671b2c1a2370b7d03a2'
def write_ray_receipts(folder):
 path=folder/'first-outward-exit-receipts.json.gz'
 with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as f:
  json.dump({'schema':'first-outward-exit-receipts-102-v1','source_field_sha256':sha(RAYFIELD),
    'helper_sha256':sha(RAY_DIR/'human_skin_fat_bone_clearance.py'),'exact_classifier_sha256':EXACT_PREDICATE_SHA,
    'receipt_count':len(ray_receipts),'receipts':ray_receipts},f,separators=(',',':'))
 return path
def first_exit_contact(*,world_start,local_start,direction_world,body,bone,start_status,ray_context):
 body=int(body);bone=int(bone)
 if (body,bone) not in BANK_GEOMETRY:raise ValueError('ray target bank is not in pinned complete-bank inventory')
 bank=BANK_GEOMETRY[(body,bone)]
 direction_world=np.asarray(direction_world,dtype=np.float64)
 if direction_world.shape!=(3,) or not np.isfinite(direction_world).all() or np.linalg.norm(direction_world)==0:
  raise ValueError('invalid source O-D direction')
 direction_world=direction_world/np.linalg.norm(direction_world)
 direction_local=rot[body]@direction_world
 end_local,aabb_witness=outward.ray_endpoint_outside_aabb(local_start,direction_local,
    bank['aabb_min'],bank['aabb_max'])
 report=outward.certify_first_outward_exit(
    start_status=start_status,start_local=local_start,end_local=end_local,
    source_direction_local=direction_local,triangles_local=bank['triangles'],
    triangle_stable_ids=bank['triangle_ids'],bank_aabb_min_local=bank['aabb_min'],
    bank_aabb_max_local=bank['aabb_max'],expected_bank_triangle_count=len(bank['triangles']),
    world_start=world_start,world_to_local=rot[body],clearance_m=5e-5,
    exact_classifier=exact['classify_segment_triangle'],exact_classifier_sha256=EXACT_PREDICATE_SHA)
 record={'context':ray_context,'body':body,'bone_stable_id':bone,'aabb_exit':aabb_witness,
    'complete_bank_triangle_count':len(bank['triangles']),'status':report['status'],'report':report}
 receipt_id=len(ray_receipts);ray_receipts.append(record)
 if report['status']!='PASS':
  raise ValueError(f"098 first outward ray exit rejected: context={ray_context}, reason={report.get('reason')}, receipt={receipt_id}")
 plane=report['plane']
 return {'normal':np.asarray(plane['unit_normal_world'],dtype=np.float64),
         'target':float(plane['minimum_signed_distance_m']),
         'point':np.asarray(plane['point_world'],dtype=np.float64),
         'receipt_id':receipt_id,'report':report}
def locations(Y):return {body:(Y.astype(float)-pos[body])@rot[body].T-off[body]for body in rot}
def read_query(Q,Y,*,records=False,query_domain='mechanics_node_row'):
 pairs=list(csv.DictReader((Q/'pairs.csv').open()))
 noderecs=list(csv.DictReader((Q/'nodes.csv').open()))
 assert len(noderecs)==len(Y)
 for i,r in enumerate(noderecs):
  assert int(r['query_row'])==i and int(r['total_bone_banks'])==185
  assert sum(int(r[k])for k in ['aabb_proven_outside_banks','parity_outside_banks','inside_banks','ambiguous_banks','invalid_banks'])==185
  assert int(r['ambiguous_banks'])==0 and int(r['invalid_banks'])==0
 qmap={}
 for r in pairs:
  k=(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id']))
  assert k not in qmap and 0<=k[0]<len(Y) and int(r['ambiguous'])==0
  assert r['classification'] in ['inside','outside']
  qmap[k]=r if records else r['classification']
 actual=set();local=locations(Y)
 for body,bone in keys:
  body=int(body);bone=int(bone);v=tri[(ids[:,0]==body)&(ids[:,1]==bone)]
  inside=np.all(local[body]>=v.min(axis=(0,1)),axis=1)&np.all(local[body]<=v.max(axis=(0,1)),axis=1)
  actual.update((int(row),body,bone)for row in np.flatnonzero(inside))
 assert actual==set(qmap),'query AABB coverage mismatch'
 return pairs,qmap
def do_query(C,I,*,query_name='query',query_domain='mechanics_node_row'):
 Q=I/query_name;Q.mkdir()
 args=[PY,str(EXPORT),'--candidate-npz',str(C),'--input-sha256',sha(C),'--field','points_world_f32','--output',str(Q/'points.csv')]
 r=subprocess.run(args,capture_output=True,text=True);(Q/'prepare.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 args=[str(BIN),str(BANK/'exact-emitted.triangles.csv'),str(BANK/'exact-emitted.bodies.csv'),str(Q/'points.csv'),str(Q/'pairs.csv'),str(Q/'nodes.csv'),'185']
 r=subprocess.run(args,capture_output=True,text=True);(Q/'query.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 query_rows=len(np.load(C,allow_pickle=False)['points_world_f32'])
 (Q/'query-receipt.json').write_text(json.dumps({'argv':args,'binary_sha256':sha(BIN),'points_sha256':sha(Q/'points.csv'),'pairs_sha256':sha(Q/'pairs.csv'),'nodes_sha256':sha(Q/'nodes.csv'),'triangles_sha256':pins['exact-emitted.triangles.csv'],'bodies_sha256':pins['exact-emitted.bodies.csv'],'query_rows':query_rows,'identity_domain':query_domain,'physical_steps':0},indent=2)+'\n')
 return Q
def query_inside_span_midpoints(edge_report,node_qmap,Y,I):
 # Mechanics query rows and source node rows are joined explicitly. Sample query
 # rows live in a separate batch and are never used as mechanics node IDs.
 query_row_to_node=np.arange(n,dtype=np.int64)
 counts=edge_report.get('counts',{})
 if int(counts.get('ambiguous',0)) or int(counts.get('boundary_hits',0)):
  report={'candidate_stopped':True,'reason':'edge audit contains unresolved boundary/ambiguous hits','counts':counts}
  (I/'midpoint-query-report.json').write_text(json.dumps(report,indent=2)+'\n')
  raise ValueError('edge audit contains unresolved boundary/ambiguous hits; midpoint spans cannot be inferred')
 samples,sample_build_report=build_edge_span_midpoint_samples(
    edge_report.get('witnesses',[]),edge_report.get('uncertain_witnesses',[]),
    node_qmap,query_row_to_node,Y)
 # Independently check exact crossing barycentrics against the candidate edge and
 # the pinned bank-local source triangle before creating query midpoint rows.
 loc_by_body=locations(Y);crossing_residuals=[]
 for w in edge_report.get('witnesses',[]):
  a,b=map(int,w['endpoint_local_nodes']);body=int(w['body']);bone=int(w['bone_stable_id'])
  if tuple(map(int,w['endpoint_query_rows']))!=(a,b):
   raise ValueError('mechanics node row / edge query row identity mismatch')
  t=float(w['exact_result']['t']);bary=np.asarray(w['exact_result']['barycentric'],dtype=np.float64)
  ti=lookup[(body,bone,int(w['triangle_stable_id']))]
  from_edge=(1.-t)*loc_by_body[body][a]+t*loc_by_body[body][b]
  from_triangle=bary@tri[ti]
  crossing_residuals.append(float(np.linalg.norm(from_edge-from_triangle)))
 max_crossing_residual=max(crossing_residuals,default=0.)
 sample_build_report['crossing_barycentric_identity']={'witnesses':len(crossing_residuals),
    'max_edge_to_triangle_residual_m':max_crossing_residual,'tolerance_m':5e-9,'passed':max_crossing_residual<=5e-9}
 points=np.asarray([s['point_world_f32']for s in samples],dtype=np.float32).reshape((-1,3))
 map_path=I/'inside-edge-span-samples.json'
 map_path.write_text(json.dumps({'sample_query_domain':'sample_query_row','mechanics_node_domain':'mechanics_node_row',
    'query_row_to_mechanics_node':query_row_to_node.tolist(),'samples':samples,'build_report':sample_build_report},indent=2)+'\n')
 if max_crossing_residual>5e-9:
  report={'candidate_stopped':True,'reason':'crossing barycentric identity failed','sample_build':sample_build_report}
  (I/'midpoint-query-report.json').write_text(json.dumps(report,indent=2)+'\n')
  raise ValueError('exact crossing barycentric identity failed; see midpoint-query-report.json')
 if not samples:
  report={'sample_build':sample_build_report,'sample_query_rows':0,'nearest_exit_constraints':0,'query_skipped_reason':'no exact-hit edge spans'}
  (I/'midpoint-query-report.json').write_text(json.dumps(report,indent=2)+'\n')
  return [],{},report
 sample_npz=I/'inside-edge-span-samples.npz'
 np.savez_compressed(sample_npz,points_world_f32=points)
 Q=do_query(sample_npz,I,query_name='midpoint-query',query_domain='sample_query_row')
 csv_identity=validate_sample_points_csv(Q/'points.csv',samples,points)
 sample_pairs,sample_record_map=read_query(Q,points,records=True,query_domain='sample_query_row')
 sample_nodes=list(csv.DictReader((Q/'nodes.csv').open()))
 sample_qmap,target_status=validate_sample_query_rows(samples,points,sample_nodes,sample_pairs,
    expected_bank_count=185,aabb_candidate_keys=set(sample_record_map))
 local=locations(points);max_local_identity_error=0.0
 for r in sample_pairs:
  qr=int(r['query_row']);body=int(r['bank_body'])
  actual=np.asarray([float(r['query_local_'+c])for c in 'xyz'],dtype=np.float64)
  error=float(np.linalg.norm(actual-local[body][qr]))
  max_local_identity_error=max(max_local_identity_error,error)
  if error>2e-8:raise ValueError(f'midpoint query local-coordinate identity failure row={qr} body={body} err_m={error}')
 contacts=[];expected_actual_disagreements=[];inside_bank_contacts=0
 sample_contacts_by_query_row=collections.defaultdict(list)
 for (pair_qr,body,bone),r in sample_qmap.items():
  if r['classification']=='inside':sample_contacts_by_query_row[int(pair_qr)].append((int(body),int(bone),r))
 for rows in sample_contacts_by_query_row.values():rows.sort(key=lambda x:(x[0],x[1]))
 for sample in samples:
  sid=int(sample['sample_id']);target_class=target_status[sid]
  if bool(sample['expected_inside_by_exact_parity'])!=(target_class=='inside'):
   expected_actual_disagreements.append({'sample_id':sid,'sample_query_row':int(sample['sample_query_row']),
      'edge_index':int(sample['edge_index']),'body':int(sample['body']),'bone_stable_id':int(sample['bone_stable_id']),
      'expected_inside_by_exact_parity':bool(sample['expected_inside_by_exact_parity']),'actual_target_classification':target_class})
  qr=int(sample['sample_query_row'])
  for body,bone,r in sample_contacts_by_query_row.get(qr,[]):
   a=int(sample['mechanics_node_a']);b=int(sample['mechanics_node_b']);tm=float(sample['t_mid'])
   direction=outward.interpolate_source_outward_direction(source_field,a,b,tm)
   local_start=np.asarray([float(r['query_local_'+c]) for c in 'xyz'],dtype=np.float64)
   exit_contact=first_exit_contact(world_start=points[qr],local_start=local_start,direction_world=direction,
      body=body,bone=bone,start_status='inside',
      ray_context={'domain':'sample_query_row','sample_id':sid,'sample_query_row':qr,
                   'mechanics_node_a':a,'mechanics_node_b':b,'t_mid':tm})
   normal=exit_contact['normal']
   contacts.append({'coefficients':[{'node':a,'weight':1.-tm,'normal_world':normal.tolist()},
       {'node':b,'weight':tm,'normal_world':normal.tolist()}],
       'target_m':exit_contact['target'],'exit_world_normal':normal.tolist(),
       'exit_world_point_m':exit_contact['point'].tolist(),'sample_query_row':qr,
       'source_edge_index':int(sample['edge_index']),'mechanics_node_rows':[a,b],'t_mid':tm,
       'body':body,'bone_stable_id':bone,'ray_receipt_id':exit_contact['receipt_id'],
       'kind':'source_od_first_outward_ray_exit'})
   inside_bank_contacts+=1
 actual_counts=collections.Counter(target_status.values())
 report={'sample_build':sample_build_report,'sample_query_rows':len(samples),
    'target_inside_midpoints':int(actual_counts.get('inside',0)),
    'target_outside_midpoints':int(actual_counts.get('outside',0)),
    'target_aabb_outside_midpoints':int(actual_counts.get('outside_aabb_proven',0)),
    'actual_inside_bank_first_outward_ray_exit_constraints':inside_bank_contacts,
    'sample_query_pairs':len(sample_pairs),'aabb_coverage':'complete over all 185 banks',
    'sample_points_csv_identity':csv_identity,'max_query_local_transform_residual_m':max_local_identity_error,
    'certified_first_outward_ray_exits':inside_bank_contacts,
    'expected_actual_classification_disagreements':expected_actual_disagreements,
    'f32_collapsed_sample_query_rows':sample_build_report['f32_collapsed_sample_query_rows'],
    'query_receipt_sha256':sha(Q/'query-receipt.json'),'sample_mapping_sha256':sha(map_path),
    'sample_npz_sha256':sha(sample_npz),'interpretation':'every nonzero exact-hit interval is queried; actual inside pairs receive 098 source O-D first-exit planes against complete bank AABBs and exact segment classification; no nearest-point fallback; outside intervals receive no plane'}
 (I/'midpoint-query-report.json').write_text(json.dumps(report,indent=2)+'\n')
 if sample_build_report['endpoint_parity_contradiction_count'] or expected_actual_disagreements:
  raise ValueError('exact crossing parity and actual midpoint classifications disagree; see midpoint-query-report.json')
 if sample_build_report['f32_collapsed_sample_query_rows']:
  raise ValueError('one or more exact-hit spans collapse in Float32; see midpoint-query-report.json')
 return contacts,sample_qmap,report

def quality(Y):
 j,_,q,_=repair.cell_values_gradients(Y,R,T)
 _,_,area,cos=repair.boundary_face_values_gradients(Y,O,BF)
 ids_dihedral,delta_dihedral=embedded.dihedral_change_metrics(Y)
 return {'J_min':float(j.min()),'J_max':float(j.max()),'q_min':float(q.min()),'invalid_cells':int((j<=0).sum()),'q_below_1e_minus5':int((q<1e-5).sum()),'area_ratio_min':float(area.min()),'area_ratio_max':float(area.max()),'normal_cosine_min':float(cos.min()),'mapped_dihedral_edges':int(len(ids_dihedral)),'mapped_dihedral_abs_change_max_degrees':float(np.degrees(np.abs(delta_dihedral)).max()) if len(delta_dihedral) else 0.,'mapped_dihedral_pass':bool(embedded.dihedral_passes(Y)),'max_total_move_mm':float(np.linalg.norm(Y.astype(float)-O,axis=1).max()*1000),**embedded.report(Y)}
history=[];start=time.monotonic()
BI=P/'baseline';BI.mkdir()
BC=BI/'candidate.npz';np.savez_compressed(BC,points_world_f32=X,tetrahedra=T,material_ids=mat)
BASEQUERY=do_query(BC,BI)
edge_report=audit.audit_unique_tet_edges(np.arange(n),T,ids,tri,locations(X),exact['classify_segment_triangle'],node_identity_domain='query_row')
BASEAUDIT=BI/'tet-edge-report.json'
BASEAUDIT.write_text(json.dumps(edge_report,indent=2)+'\n')
query=BASEQUERY
assert quality(X)['q_below_1e_minus5']==0
# Use the unchanged exact owner315 predicate. Shared indexed features alone
# are allowed; adjacency is not a collision exemption.
import importlib.util
DEPS=E/'incremental-exact-audit-owner-253/src';sys.path.insert(0,str(DEPS))
PRED=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
spec=importlib.util.spec_from_file_location('numilab_human._connected_086_predicate',PRED)
selfowner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=selfowner;spec.loader.exec_module(selfowner)
def self_audit(Y,folder):
 rec,_=selfowner.float32_surface_records(embedded.positions(Y),detail_faces,source_seam_groups=())
 result=selfowner._audit_self_pairs_touching_faces(rec,set(range(len(detail_faces))))
 np.save(folder/'self-pairs.npy',np.asarray(result['triangle_pairs'],dtype=np.int64).reshape(-1,2))
 out={k:result[k] for k in ['count','aabb_candidate_pairs','allowed_shared_vertex_or_edge_pairs','complete_surface_audit']}
 out['predicate_sha256']=sha(PRED);out['mapped_positions_sha256']=hashlib.sha256(embedded.positions(Y).tobytes()).hexdigest()
 (folder/'self-report.json').write_text(json.dumps(out,indent=2)+'\n')
 return result['triangle_pairs']
baseline_self_pairs=self_audit(X,BI)
for fa,fb in baseline_self_pairs:add_separator_pair(int(fa),int(fb))
L,lb=hard_rows()
(P/'baseline-self-separator-provenance.json').write_text(json.dumps(sep_pro,indent=2)+'\n')

base_pairs,base_qmap=read_query(query,X)
base_inside=[r for r in base_pairs if r['classification']=='inside']
baseline_summary={'query_rows':len(X),'inside_pairs':len(base_inside),
 'inside_nodes':len({int(r['query_row']) for r in base_inside}),
 'edge_bank_groups_with_hits':edge_report['edge_bank_groups_with_hits'],
 'inside_signed_face_distance_min_mm':1000*min((float(r['signed_face_distance_m']) for r in base_inside),default=0.),
 'inside_nearest_distance_max_mm':1000*max((float(r['nearest_distance_m']) for r in base_inside),default=0.),
 'note':'signed/nearest distances are source-query diagnostics, not a certified solid-penetration depth',
 'baseline_detailed_self_intersections':len(baseline_self_pairs),'mapped_skin_baseline':embedded.report(X)}
(P/'input-pins.json').write_text(json.dumps({'input_npz':str(INPUT),'input_sha256':EXPECTED,'starting_candidate_path':str(INPUT),'starting_candidate_sha256':sha(INPUT),'starting_state_field':'current037_points_m_f32',
 'baseline_query_pairs_sha256':sha(BASEQUERY/'pairs.csv'),'baseline_query_nodes_sha256':sha(BASEQUERY/'nodes.csv'),
 'baseline_query_identity_domain':'mechanics_node_row; query row i maps to candidate mechanics node i',
 'baseline_edge_audit_sha256':sha(BASEAUDIT),'transfer_rows_sha256':sha(TRANSFER_ROWS),
 'coarse_reference_sha256':sha(COARSE_REFERENCE),'coarse_faces_sha256':sha(COARSE_FACES),
 'detail_payload_sha256':sha(DETAIL_PAYLOAD),'adapter_sha256':sha(ADAPTER_DIR/'load_transfer_075.py'),
 'solver_sha256':sha(M/'human_skin_fat_connected_repair.py'),
 'embedded_surface_sha256':sha(M/'human_skin_fat_embedded_surface.py'),
 'closest_exit_helper_sha256':sha(HELPER_DIR/'human_skin_fat_bone_clearance.py'),
 'outward_ray_exit_098_sha256':sha(RAY_DIR/'human_skin_fat_bone_clearance.py'),
 'source_field_npz_sha256':sha(RAYFIELD),
 'edge_contact_method':'query each exact-hit interval F32 midpoint against all 185 banks; actual inside node and edge samples use source-directed first proper exits; outside spans add no plane',
 'baseline':baseline_summary,'active_set_passes':0,'iteration_limit':4,'step_limit_mm':2.,'time_limit_s':60.,'max_incremental_gradient':None,'dihedral_active_fraction':.75,'dihedral_constraint_passes':3,'physical_steps':0,'geometry_admitted':False,
 'scope':'connected whole skin-fat remesh with same-parent detail embedding; static authoring, not native physical qualification'},indent=2)+'\n')
print(json.dumps({'baseline':baseline_summary,'free_nodes_include_outer_O':True,'max_iterations':4,'active_set_passes':0}),flush=True)
previous_score=baseline_summary['edge_bank_groups_with_hits']+4*baseline_summary['inside_nodes']
def write_continuation(folder,history,baseline,started,first_failure=None):
 payload={'history':history,'baseline':baseline,'active_set_passes':0,'iteration_limit':4,
  'accepted_physical_steps':0,'geometry_admitted':False,'elapsed_s':time.monotonic()-started,
  'input_sha256':EXPECTED,'mapping_sha256':sha(MAP/'Wnew-detail-to-084-nodes.npz'),
  'solver_sha256':sha(M/'human_skin_fat_connected_repair.py'),
  'embedded_owner_sha256':sha(M/'human_skin_fat_embedded_surface.py'),
  'audit_sha256':sha(AUDIT_PATH),'source_ray_helper_sha256':sha(RAY_DIR/'human_skin_fat_bone_clearance.py'),
  'source_field_sha256':sha(RAYFIELD),'max_dihedral_change_degrees':15.,
  'first_failure':first_failure,'limitations':['Static connected whole-mesh asset construction only.',
  'Fresh compact 084 current037 state is the starting state; no 089/091 displacement candidate is loaded.',
  'Source O-D direction and first ray exits are construction constraints, not anatomical outward or clearance proof.',
  'All-bank/node and exact edge checks remain separate required gates; no native physical qualification.',
  'Thickness, mass and reference inputs are pinned and not redefined.']}
 temp=folder/'continuation.json.tmp';temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(folder/'continuation.json')
history=[];start=time.monotonic();first_failure=None;current_iteration=None;current_stage='baseline'

try:
 for it in range(1,5):
  I=P/('iteration-%03d'%it);I.mkdir();current_iteration=it;current_stage='query mapping';ray_receipts=[]
  pairs,qmap=read_query(query,X)
  rows=[];targets=[];provenance=[]
  def add(nodes,weights,normal,target,pro):
   row=len(targets)
   for v,w in zip(nodes,weights):
    for k in range(3):rows.append((row,3*int(v)+k,float(w*normal[k])))
   targets.append(target);provenance.append(pro)
  current_stage='validated midpoint query and source-ray exits'
  midpoint_contacts,midpoint_qmap,midpoint_report=query_inside_span_midpoints(edge_report,qmap,X,I)
  for contact in midpoint_contacts:
   coeff=contact['coefficients']
   add([v['node']for v in coeff],[v['weight']for v in coeff],
       np.asarray(contact['exit_world_normal'],dtype=np.float64),contact['target_m'],
       {'kind':'source_od_first_outward_ray_exit',**contact})
  for r in pairs:
   if r['classification']!='inside':continue
   node=int(r['query_row']);body=int(r['bank_body']);bone=int(r['bank_stable_id'])
   local_start=np.asarray([float(r['query_local_'+c]) for c in 'xyz'],dtype=np.float64)
   exit_contact=first_exit_contact(world_start=X[node],local_start=local_start,direction_world=source_field[node],
       body=body,bone=bone,start_status='inside',
       ray_context={'domain':'mechanics_node_row','query_row':node})
   normal=exit_contact['normal']
   add([node],[1],normal,exit_contact['target'],
       {'kind':'source_od_first_outward_ray_exit','query_row':node,'body':body,'bone':bone,
        'ray_receipt_id':exit_contact['receipt_id'],'exit_point_world':exit_contact['point'].tolist()})
  current_stage='serialize all node and midpoint ray receipts'
  write_ray_receipts(I)
  if not targets:
   write_continuation(P,history,baseline_summary,start,first_failure)
   break
  rr,cc,vv=zip(*rows);C=sparse.coo_matrix((vv,(rr,cc)),shape=(len(targets),3*n)).tocsr();target=np.asarray(targets)
  free=np.unique(C.indices//3)
  for ring in range(2):free=np.unique(T[np.isin(T,free).any(axis=1)])
  crossing_edges=np.asarray([w['endpoint_query_rows'] for w in edge_report['witnesses']],dtype=np.int64).reshape(-1,2)
  inside_nodes=np.unique(np.asarray([int(r['query_row']) for r in pairs if r['classification']=='inside'],dtype=np.int64))
  free,workset_report=select_boundary_witness_workset(n,free,outer_boundary_faces,
      crossing_edges,inside_nodes,C,patch_rings=2)
  workset_report['owner_sha256']=sha(M/'human_skin_fat_boundary_workset.py')
  workset_report['source']='exact all-tet-edge and all-185-bank current witnesses; not nearest points'
  (I/'boundary-workset-report.json').write_text(json.dumps(workset_report,indent=2)+'\n')
  sparse.save_npz(I/'contact-rows.npz',C)
  (I/'contact-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
  np.savez_compressed(I/'input.npz',points_world_f32=X,reference_points_m=R,construction_origin_f32=O,tetrahedra=T,material_ids=mat,free_nodes=free,contact_targets=target)
  current_stage='connected constrained solver'
  Y,report=repair.coupled_clearance_step(X,R,T,free,C,target,step_limit_mm=2.,time_limit_s=60.,
     lower_linear_matrix=L,lower_linear_targets=lb,max_incremental_gradient=None,dihedral_active_fraction=.75,dihedral_constraint_passes=3,
     construction_origin=O,cumulative_displacement_weight=.002,boundary_faces=BF,
     boundary_area_bounds=(.5,2.),boundary_normal_cosine_min=.5,embedded_surface=embedded,linear_method='highs-ipm')
  (I/'construction-report.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'iteration':it,'construction':{k:v for k,v in report.items() if k not in ['mapped_skin_linearization_face_ids','mapped_skin_dihedral_linearization_edge_ids']}}),flush=True)
  CAND=I/'candidate.npz';np.savez_compressed(CAND,points_world_f32=Y,tetrahedra=T,material_ids=mat)
  current_stage='exact mapped-surface self audit'
  self_pairs=self_audit(Y,I)
  if self_pairs:
   for fa,fb in self_pairs:add_separator_pair(int(fa),int(fb))
   L,lb=hard_rows()
   (I/'stop-or-repair.json').write_text(json.dumps({'candidate_not_admitted':True,'reason':'new detailed self-intersection','pairs':len(self_pairs),'next_change':'hard baseline SAT rows added for the witnessed face pairs; retry from last passing connected state'},indent=2)+'\n')
   print(json.dumps({'iteration':it,'self_crossings':len(self_pairs),'action':'retry_with_witness_constraints'}),flush=True)
   continue
  current_stage='185-bank candidate query'
  query=do_query(CAND,I);newpairs,_=read_query(query,Y)
  current_stage='all-tet-edge exact audit'
  edge_report=audit.audit_unique_tet_edges(np.arange(n),T,ids,tri,locations(Y),exact['classify_segment_triangle'],node_identity_domain='query_row')
  (I/'tet-edge-report.json').write_text(json.dumps(edge_report,indent=2)+'\n')
  new_inside=[r for r in newpairs if r['classification']=='inside']
  metric={'iteration':it,'inside_edge_span_sample_count':midpoint_report['sample_query_rows'],'source_directed_edge_constraints':midpoint_report.get('target_inside_midpoints',0),'midpoint_query_report_sha256':sha(I/'midpoint-query-report.json'),'inside_signed_face_distance_min_mm':1000*min((float(r['signed_face_distance_m']) for r in new_inside),default=0.),'inside_nearest_distance_max_mm':1000*max((float(r['nearest_distance_m']) for r in new_inside),default=0.),'edge_bank_groups':edge_report['edge_bank_groups_with_hits'],'edges':edge_report['edges_with_hits'],'inside_nodes':len({int(r['query_row'])for r in newpairs if r['classification']=='inside'}),'inside_pairs':sum(r['classification']=='inside'for r in newpairs),'ambiguous':edge_report['counts']['ambiguous'],'candidate_sha256':sha(CAND),'quality':quality(Y),'candidate_path':str(CAND),'query_path':str(query),'audit_path':str(I/'tet-edge-report.json')}
  quality_now=metric['quality']
  metric['cell_gate_pass']=bool(quality_now['invalid_cells']==0 and quality_now['q_below_1e_minus5']==0 and quality_now['J_min']>=.35 and quality_now['J_max']<=2.5)
  metric['boundary_gate_pass']=bool(quality_now['area_ratio_min']>=.5 and quality_now['area_ratio_max']<=2. and quality_now['normal_cosine_min']>=.5)
  metric['detail_gate_pass']=bool(quality_now['mapped_skin_geometry_pass'] and quality_now['mapped_skin_nonpositive_normal_faces']==0 and quality_now['mapped_skin_area_ratio_min']>=.5 and quality_now['mapped_skin_area_ratio_max']<=2. and quality_now['mapped_skin_normal_cosine_min']>=.5 and quality_now['mapped_dihedral_pass'])
  metric['all_authoring_gates_pass']=bool(metric['cell_gate_pass'] and metric['boundary_gate_pass'] and metric['detail_gate_pass'] and metric['ambiguous']==0)
  score=metric['edge_bank_groups']+4*metric['inside_nodes']
  metric['progress_score']=score
  metric['improved_over_previous']=bool(score<previous_score)
  if not metric['all_authoring_gates_pass']:
   metric['stop_reason']='cell_or_boundary_or_mapped_detail_gate_failed'
  elif not metric['improved_over_previous']:
   metric['stop_reason']='no_progress_vs_previous_iteration'
  history.append(metric)
  write_continuation(P,history,baseline_summary,start,first_failure)
  print(json.dumps(metric),flush=True)
  if 'stop_reason' in metric:break
  previous_score=score
  X=Y
except Exception as exc:
 if first_failure is None:
  first_failure={'iteration':current_iteration,'stage':current_stage,'exception_type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
 faildir=P/('iteration-%03d'%int(current_iteration)) if current_iteration is not None else P
 if faildir.is_dir():
  (faildir/'first-failure.json').write_text(json.dumps(first_failure,indent=2)+'\n')
 raise
finally:
 if current_iteration is not None and (P/('iteration-%03d'%int(current_iteration))).is_dir():
  try: write_ray_receipts(P/('iteration-%03d'%int(current_iteration)))
  except Exception as receipt_exc:
   if first_failure is None:first_failure={'iteration':current_iteration,'stage':'ray receipt finalization','exception_type':type(receipt_exc).__name__,'message':str(receipt_exc)}
 write_continuation(P,history,baseline_summary,start,first_failure)
print('CONNECTED_EXTERIOR_CLEARANCE_102_ENDED',flush=True)
