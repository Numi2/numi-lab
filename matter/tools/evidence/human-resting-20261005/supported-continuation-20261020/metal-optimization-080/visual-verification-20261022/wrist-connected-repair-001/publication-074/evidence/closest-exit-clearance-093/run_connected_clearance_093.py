from pathlib import Path
import json,hashlib,csv,sys,subprocess,time,collections,os
import numpy as np
from scipy import sparse
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=Path('/Users/n/Work/numi-human-connected-surface-069');M=D/'publication-074/canonical/matter/tools'
sys.path.insert(0,str(M))
import human_skin_fat_connected_repair as repair
from human_skin_fat_embedded_surface import MappedSkinSurface
HELPER_DIR=D/'closest-exit-clearance-093';sys.path.insert(0,str(HELPER_DIR))
from closest_exit_clearance_093 import (build_edge_span_midpoint_samples, validate_sample_query_rows,
    validate_sample_points_csv, nearest_exit_contact)
# Set to the frozen identity-domain audit extension after its tests complete.
AUDIT_PATH=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
sys.path.insert(0,str(AUDIT_PATH.parent))
import human_skin_fat_bone_audit as audit
P=D/'closest-exit-clearance-093/run-001'
assert not P.exists();P.mkdir()
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
START=D/'whole-reference-tetgen-084-bounded/connected-clearance-089/run-001/iteration-004/candidate.npz'
assert sha(START)=='30f761426a48b26c5bccf613d93ad8153ddd1db7ed2a5837ca667e2efa3e72a9'
startz=np.load(START);X=startz['points_world_f32'].copy()
assert X.dtype==np.float32 and X.shape==O.shape and np.array_equal(startz['tetrahedra'],T) and np.array_equal(startz['material_ids'],mat)
n=len(X)
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

embedded=MappedSkinSurface(O.copy(),detail_origin,detail_faces,W)
assert embedded.passes(X)
assert np.array_equal(embedded.positions(O).view(np.uint32),detail_origin.view(np.uint32))

(P/'execution-owner-pins.json').write_text(json.dumps({
 'candidate_sha256':sha(INPUT),'mapping_sha256':sha(MAP/'Wnew-detail-to-084-nodes.npz'),
 'mapping_report_sha256':sha(MAP/'transfer-report.json'),'construction_owner_sha256':sha(M/'human_skin_fat_connected_repair.py'),
 'embedded_owner_sha256':sha(M/'human_skin_fat_embedded_surface.py'),
 'surface_separator_owner_sha256':sha(M/'human_skin_fat_surface_separation.py'),
 'generated_thickness_owner_sha256':sha(M/'human_skin_fat_generated_layer_constraints.py'),
 'starting_candidate_sha256':sha(START),'static_authoring_only':True,'accepted_physical_steps':0},indent=2)+'\n')
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
  actual_inside=target_class=='inside'
  if bool(sample['expected_inside_by_exact_parity'])!=actual_inside:
   expected_actual_disagreements.append({'sample_id':sid,'sample_query_row':int(sample['sample_query_row']),
      'edge_index':int(sample['edge_index']),'body':int(sample['body']),'bone_stable_id':int(sample['bone_stable_id']),
      'expected_inside_by_exact_parity':bool(sample['expected_inside_by_exact_parity']),'actual_target_classification':target_class})
  qr=int(sample['sample_query_row'])
  for body,bone,r in sample_contacts_by_query_row.get(qr,[]):
   bank_sample={**sample,'body':body,'bone_stable_id':bone}
   tri_index=lookup[(body,bone,int(r['nearest_triangle']))]
   contact=nearest_exit_contact(bank_sample,r,tri[tri_index],rot[body],off[body],pos[body],node_count=n)
   contact['kind']='measured_edge_span_nearest_exit'
   contact['tagged_edge_body_bone']=[int(sample['body']),int(sample['bone_stable_id'])]
   contacts.append(contact);inside_bank_contacts+=1
 actual_counts=collections.Counter(target_status.values())
 report={'sample_build':sample_build_report,'sample_query_rows':len(samples),
    'target_inside_midpoints':int(actual_counts.get('inside',0)),
    'target_outside_midpoints':int(actual_counts.get('outside',0)),
    'target_aabb_outside_midpoints':int(actual_counts.get('outside_aabb_proven',0)),
    'actual_inside_bank_nearest_exit_constraints':inside_bank_contacts,
    'sample_query_pairs':len(sample_pairs),'aabb_coverage':'complete over all 185 banks',
    'sample_points_csv_identity':csv_identity,'max_query_local_transform_residual_m':max_local_identity_error,
    'nearest_exit_barycentric_residual_max_m':max((c['closest_barycentric_residual_m']for c in contacts),default=0.),
    'expected_actual_classification_disagreements':expected_actual_disagreements,
    'f32_collapsed_sample_query_rows':sample_build_report['f32_collapsed_sample_query_rows'],
    'query_receipt_sha256':sha(Q/'query-receipt.json'),'sample_mapping_sha256':sha(map_path),
    'sample_npz_sha256':sha(sample_npz),'interpretation':'every nonzero exact-hit interval is queried; actual inside pairs receive measured nearest-exit planes; actual outside intervals receive no plane'}
 (I/'midpoint-query-report.json').write_text(json.dumps(report,indent=2)+'\n')
 if sample_build_report['endpoint_parity_contradiction_count'] or expected_actual_disagreements:
  raise ValueError('exact crossing parity and actual midpoint classifications disagree; see midpoint-query-report.json')
 if sample_build_report['f32_collapsed_sample_query_rows']:
  raise ValueError('one or more exact-hit spans collapse in Float32; see midpoint-query-report.json')
 return contacts,sample_qmap,report

def quality(Y):
 j,_,q,_=repair.cell_values_gradients(Y,R,T)
 _,_,area,cos=repair.boundary_face_values_gradients(Y,O,BF)
 return {'J_min':float(j.min()),'J_max':float(j.max()),'q_min':float(q.min()),'invalid_cells':int((j<=0).sum()),'q_below_1e_minus5':int((q<1e-5).sum()),'area_ratio_min':float(area.min()),'area_ratio_max':float(area.max()),'normal_cosine_min':float(cos.min()),'max_total_move_mm':float(np.linalg.norm(Y.astype(float)-O,axis=1).max()*1000),**embedded.report(Y)}
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
assert not self_audit(X,BI), 'new origin has existing detailed self crossings'

base_pairs,base_qmap=read_query(query,X)
base_inside=[r for r in base_pairs if r['classification']=='inside']
baseline_summary={'query_rows':len(X),'inside_pairs':len(base_inside),
 'inside_nodes':len({int(r['query_row']) for r in base_inside}),
 'edge_bank_groups_with_hits':edge_report['edge_bank_groups_with_hits'],
 'inside_signed_face_distance_min_mm':1000*min((float(r['signed_face_distance_m']) for r in base_inside),default=0.),
 'inside_nearest_distance_max_mm':1000*max((float(r['nearest_distance_m']) for r in base_inside),default=0.),
 'note':'signed/nearest distances are source-query diagnostics, not a certified solid-penetration depth',
 'mapped_skin_baseline':embedded.report(X)}
(P/'input-pins.json').write_text(json.dumps({'input_npz':str(INPUT),'input_sha256':EXPECTED,'starting_candidate_path':str(START),'starting_candidate_sha256':sha(START),
 'baseline_query_pairs_sha256':sha(BASEQUERY/'pairs.csv'),'baseline_query_nodes_sha256':sha(BASEQUERY/'nodes.csv'),
 'baseline_query_identity_domain':'mechanics_node_row; query row i maps to candidate mechanics node i',
 'baseline_edge_audit_sha256':sha(BASEAUDIT),'transfer_rows_sha256':sha(TRANSFER_ROWS),
 'coarse_reference_sha256':sha(COARSE_REFERENCE),'coarse_faces_sha256':sha(COARSE_FACES),
 'detail_payload_sha256':sha(DETAIL_PAYLOAD),'adapter_sha256':sha(ADAPTER_DIR/'load_transfer_075.py'),
 'solver_sha256':sha(M/'human_skin_fat_connected_repair.py'),
 'embedded_surface_sha256':sha(M/'human_skin_fat_embedded_surface.py'),
 'closest_exit_helper_sha256':sha(HELPER_DIR/'closest_exit_clearance_093.py'),
 'edge_contact_method':'query each exact-hit edge-span F32 midpoint against all 185 banks; actual inside pairs use measured nearest-exit point/covector; actual outside spans add no plane',
 'baseline':baseline_summary,'active_set_passes':0,'iteration_limit':4,'physical_steps':0,'geometry_admitted':False,
 'scope':'connected whole skin-fat remesh with same-parent detail embedding; static authoring, not native physical qualification'},indent=2)+'\n')
print(json.dumps({'baseline':baseline_summary,'free_nodes_include_outer_O':True,'max_iterations':4,'active_set_passes':0}),flush=True)
previous_score=baseline_summary['edge_bank_groups_with_hits']+4*baseline_summary['inside_nodes']
for it in range(1,5):
 I=P/('iteration-%03d'%it);I.mkdir()
 pairs,qmap=read_query(query,X)
 rows=[];targets=[];provenance=[]
 def add(nodes,weights,normal,target,pro):
  row=len(targets)
  for v,w in zip(nodes,weights):
   for k in range(3):rows.append((row,3*int(v)+k,float(w*normal[k])))
  targets.append(target);provenance.append(pro)
 midpoint_contacts,midpoint_qmap,midpoint_report=query_inside_span_midpoints(edge_report,qmap,X,I)
 for contact in midpoint_contacts:
  coeff=contact['coefficients']
  add([v['node']for v in coeff],[v['weight']for v in coeff],
      np.asarray(contact['nearest_exit_world_normal'],dtype=np.float64),contact['target_m'],
      {'kind':'measured_inside_edge_span_nearest_exit',**contact})
 for r in pairs:
  if r['classification']!='inside':continue
  node=int(r['query_row']);body=int(r['bank_body']);bone=int(r['bank_stable_id'])
  p=np.asarray([float(r['query_local_'+c])for c in 'xyz']);hl=np.asarray([float(r['closest_'+c])for c in 'xyz'])
  normal=rot[body].T@(hl-p);normal/=np.linalg.norm(normal)
  h=pos[body]+np.linalg.solve(rot[body],hl+off[body])
  add([node],[1],normal,float(normal@h+5e-5),{'kind':'native_owner_inside_exit','query_row':node,'body':body,'bone':bone})
 if not targets:break
 rr,cc,vv=zip(*rows);C=sparse.coo_matrix((vv,(rr,cc)),shape=(len(targets),3*n)).tocsr();target=np.asarray(targets)
 free=np.unique(C.indices//3)
 for ring in range(2):free=np.unique(T[np.isin(T,free).any(axis=1)])
 sparse.save_npz(I/'contact-rows.npz',C)
 (I/'contact-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 np.savez_compressed(I/'input.npz',points_world_f32=X,reference_points_m=R,construction_origin_f32=O,tetrahedra=T,material_ids=mat,free_nodes=free,contact_targets=target)
 Y,report=repair.coupled_clearance_step(X,R,T,free,C,target,step_limit_mm=2.,time_limit_s=60.,
    lower_linear_matrix=L,lower_linear_targets=lb,max_incremental_gradient=None,
    construction_origin=O,cumulative_displacement_weight=.002,boundary_faces=BF,
    boundary_area_bounds=(.5,2.),boundary_normal_cosine_min=.5,embedded_surface=embedded,linear_method='highs-ipm')
 (I/'construction-report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'iteration':it,'construction':report}),flush=True)
 CAND=I/'candidate.npz';np.savez_compressed(CAND,points_world_f32=Y,tetrahedra=T,material_ids=mat)
 self_pairs=self_audit(Y,I)
 if self_pairs:
  for fa,fb in self_pairs:add_separator_pair(int(fa),int(fb))
  L,lb=hard_rows()
  (I/'stop-or-repair.json').write_text(json.dumps({'candidate_not_admitted':True,'reason':'new detailed self-intersection','pairs':len(self_pairs),'next_change':'hard baseline SAT rows added for the witnessed face pairs; retry from last passing connected state'},indent=2)+'\n')
  print(json.dumps({'iteration':it,'self_crossings':len(self_pairs),'action':'retry_with_witness_constraints'}),flush=True)
  continue
 query=do_query(CAND,I);newpairs,_=read_query(query,Y)
 edge_report=audit.audit_unique_tet_edges(np.arange(n),T,ids,tri,locations(Y),exact['classify_segment_triangle'],node_identity_domain='query_row')
 (I/'tet-edge-report.json').write_text(json.dumps(edge_report,indent=2)+'\n')
 new_inside=[r for r in newpairs if r['classification']=='inside']
 metric={'iteration':it,'inside_edge_span_sample_count':midpoint_report['sample_query_rows'],'nearest_exit_edge_constraints':midpoint_report.get('target_inside_midpoints',0),'midpoint_query_report_sha256':sha(I/'midpoint-query-report.json'),'inside_signed_face_distance_min_mm':1000*min((float(r['signed_face_distance_m']) for r in new_inside),default=0.),'inside_nearest_distance_max_mm':1000*max((float(r['nearest_distance_m']) for r in new_inside),default=0.),'edge_bank_groups':edge_report['edge_bank_groups_with_hits'],'edges':edge_report['edges_with_hits'],'inside_nodes':len({int(r['query_row'])for r in newpairs if r['classification']=='inside'}),'inside_pairs':sum(r['classification']=='inside'for r in newpairs),'ambiguous':edge_report['counts']['ambiguous'],'candidate_sha256':sha(CAND),'quality':quality(Y),'candidate_path':str(CAND),'query_path':str(query),'audit_path':str(I/'tet-edge-report.json')}
 quality_now=metric['quality']
 metric['cell_gate_pass']=bool(quality_now['invalid_cells']==0 and quality_now['q_below_1e_minus5']==0 and quality_now['J_min']>=.35 and quality_now['J_max']<=2.5)
 metric['boundary_gate_pass']=bool(quality_now['area_ratio_min']>=.5 and quality_now['area_ratio_max']<=2. and quality_now['normal_cosine_min']>=.5)
 metric['detail_gate_pass']=bool(quality_now['mapped_skin_geometry_pass'] and quality_now['mapped_skin_nonpositive_normal_faces']==0 and quality_now['mapped_skin_area_ratio_min']>=.5 and quality_now['mapped_skin_area_ratio_max']<=2. and quality_now['mapped_skin_normal_cosine_min']>=.5)
 metric['all_authoring_gates_pass']=bool(metric['cell_gate_pass'] and metric['boundary_gate_pass'] and metric['detail_gate_pass'] and metric['ambiguous']==0)
 score=metric['edge_bank_groups']+4*metric['inside_nodes']
 metric['progress_score']=score
 metric['improved_over_previous']=bool(score<previous_score)
 if not metric['all_authoring_gates_pass']:
  metric['stop_reason']='cell_or_boundary_or_mapped_detail_gate_failed'
 elif not metric['improved_over_previous']:
  metric['stop_reason']='no_progress_vs_previous_iteration'
 history.append(metric)
 (P/'continuation.json').write_text(json.dumps({'history':history,'baseline':baseline_summary,'active_set_passes':0,'iteration_limit':4,'accepted_physical_steps':0,'geometry_admitted':False,'elapsed_s':time.monotonic()-start,'input_sha256':EXPECTED,'mapping_sha256':sha(MAP/'Wnew-detail-to-084-nodes.npz'),'solver_sha256':sha(M/'human_skin_fat_connected_repair.py'),'audit_sha256':sha(AUDIT_PATH),'limitations':['Static connected asset authoring only.','075 mapped detail skin was included as a constrained surface; this does not certify all-bank surface clearance or full solid self-crossings.','Mass transfer is checked separately against exact native cooked source ownership; no native physical qualification.']},indent=2)+'\n')
 print(json.dumps(metric),flush=True)
 if 'stop_reason' in metric:break
 previous_score=score
 X=Y
print('CONNECTED_WHOLE_REMESH_CONSTRUCTION_ENDED',flush=True)
