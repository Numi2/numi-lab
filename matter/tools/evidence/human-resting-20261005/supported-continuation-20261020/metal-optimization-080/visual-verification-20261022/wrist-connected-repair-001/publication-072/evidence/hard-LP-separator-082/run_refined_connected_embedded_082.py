from pathlib import Path
import json,hashlib,csv,sys,subprocess,time,collections,os
import numpy as np
from scipy import sparse
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=Path('/Users/n/Work/numi-human-connected-surface-069/direct-authoring-082');M=D/'canonical/matter/tools'
sys.path.insert(0,str(M))
import human_skin_fat_connected_repair as repair
from human_skin_fat_embedded_surface import MappedSkinSurface
# Set to the frozen identity-domain audit extension after its tests complete.
AUDIT_PATH=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
sys.path.insert(0,str(AUDIT_PATH.parent))
import human_skin_fat_bone_audit as audit
P=D/'repair-run-embedded-082'
assert not P.exists();P.mkdir()
INPUT=E/'E/connected-volume-refinement-037/run-006/candidate.npz'
EXPECTED='52c81dbe323eab2240bc315f7b7bde605a201fa8e928206f27000de7e5d7a492'
BANK=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
BASEQUERY=E/'E/generic-static-query-042/run-037006-current-001'
BASEAUDIT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/E/generic-static-query-043/run-037006-query-row-edges-001/tet-edge-report.json")
BIN=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query')
EXPORT=E/'E/generic-static-query-042/prepare_generic_query_points.py'
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(INPUT)==EXPECTED
z=np.load(INPUT);O=z['current_points_f32'];X=O.copy();R=z['canonical_reference_points_f64'];T=z['tetrahedra'];mat=z['material_ids']
n=len(X)
assert n==9320
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
W=sparse.hstack([W075,sparse.csr_matrix((W075.shape[0],n-W075.shape[1]))],format='csr')
embedded=MappedSkinSurface(O.copy(),detail_origin,detail_faces,W)
assert embedded.passes(X)

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
Lbase=sparse.coo_matrix((vv,(rr,cc)),shape=(len(lb),3*n)).tocsr();lbbase=np.asarray(lb)
SEP=Path('/Users/n/Work/numi-human-connected-surface-069/E/embedded-self-separator-078/run-002')
SEP_A=SEP/'separator-inequalities.npz';SEP_T=SEP/'separator-targets.npz'
assert sha(SEP_A)=='130a0a8aa68067ec1dd16af9653b2f87af7eb47787aa15f311b24ad8508b9cd8'
assert sha(SEP_T)=='1425e5f2af00963698c3a2f63569b592878d581fda956af687adb1c1ff87a325'
S_sep=sparse.load_npz(SEP_A).tocsr();sepdata=np.load(SEP_T)
sep_targets=sepdata['target'];sep_base_slack=sepdata['baseline_slack_m']
assert S_sep.shape==(25,3*n) and sep_targets.shape==(25,) and sep_base_slack.shape==(25,)
actual_base_slack=np.asarray(S_sep@O.astype(float).ravel()-sep_targets).ravel()
assert np.max(np.abs(actual_base_slack-sep_base_slack))<2e-7, 'separator origin/baseline slack mismatch'
assert np.min(actual_base_slack)>=-1e-9, 'separator rows fail at original 037 origin'
L=sparse.vstack([Lbase,S_sep],format='csr');lb=np.r_[lbbase,sep_targets]
sparse.save_npz(P/'layer-rows.npz',Lbase);np.save(P/'layer-targets.npy',lbbase)
sparse.save_npz(P/'separator-rows.npz',S_sep);np.save(P/'separator-targets.npy',sep_targets)
sparse.save_npz(P/'combined-hard-linear-rows.npz',L);np.save(P/'combined-hard-linear-targets.npy',lb)
lookup={(int(r[0]),int(r[1]),int(r[3])):k for k,r in enumerate(ids)}
def locations(Y):return {body:(Y.astype(float)-pos[body])@rot[body].T-off[body]for body in rot}
def read_query(Q,Y):
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
  assert k not in qmap and 0<=k[0]<n and int(r['ambiguous'])==0
  assert r['classification'] in ['inside','outside']
  qmap[k]=r['classification']
 actual=set();local=locations(Y)
 for body,bone in keys:
  body=int(body);bone=int(bone);v=tri[(ids[:,0]==body)&(ids[:,1]==bone)]
  inside=np.all(local[body]>=v.min(axis=(0,1)),axis=1)&np.all(local[body]<=v.max(axis=(0,1)),axis=1)
  actual.update((int(row),body,bone)for row in np.flatnonzero(inside))
 assert actual==set(qmap),'query AABB coverage mismatch'
 return pairs,qmap
def do_query(C,I):
 Q=I/'query';Q.mkdir()
 args=[PY,str(EXPORT),'--candidate-npz',str(C),'--input-sha256',sha(C),'--field','points_world_f32','--output',str(Q/'points.csv')]
 r=subprocess.run(args,capture_output=True,text=True);(Q/'prepare.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 args=[str(BIN),str(BANK/'exact-emitted.triangles.csv'),str(BANK/'exact-emitted.bodies.csv'),str(Q/'points.csv'),str(Q/'pairs.csv'),str(Q/'nodes.csv'),'185']
 r=subprocess.run(args,capture_output=True,text=True);(Q/'query.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 (Q/'query-receipt.json').write_text(json.dumps({'argv':args,'binary_sha256':sha(BIN),'points_sha256':sha(Q/'points.csv'),'pairs_sha256':sha(Q/'pairs.csv'),'nodes_sha256':sha(Q/'nodes.csv'),'triangles_sha256':pins['exact-emitted.triangles.csv'],'bodies_sha256':pins['exact-emitted.bodies.csv'],'query_rows':n,'identity_domain':'query_row','physical_steps':0},indent=2)+'\n')
 return Q
def quality(Y):
 j,_,q,_=repair.cell_values_gradients(Y,R,T)
 _,_,area,cos=repair.boundary_face_values_gradients(Y,O,BF)
 return {'J_min':float(j.min()),'J_max':float(j.max()),'q_min':float(q.min()),'invalid_cells':int((j<=0).sum()),'q_below_1e_minus5':int((q<1e-5).sum()),'area_ratio_min':float(area.min()),'area_ratio_max':float(area.max()),'normal_cosine_min':float(cos.min()),'max_total_move_mm':float(np.linalg.norm(Y.astype(float)-O,axis=1).max()*1000),**embedded.report(Y)}
query=BASEQUERY;edge_report=json.loads(BASEAUDIT.read_text());history=[];start=time.monotonic()
base_pairs,base_qmap=read_query(query,X)
base_inside=[r for r in base_pairs if r['classification']=='inside']
baseline_summary={'query_rows':len(X),'inside_pairs':len(base_inside),
 'inside_nodes':len({int(r['query_row']) for r in base_inside}),
 'edge_bank_groups_with_hits':edge_report['edge_bank_groups_with_hits'],
 'inside_signed_face_distance_min_mm':1000*min((float(r['signed_face_distance_m']) for r in base_inside),default=0.),
 'inside_nearest_distance_max_mm':1000*max((float(r['nearest_distance_m']) for r in base_inside),default=0.),
 'note':'signed/nearest distances are source-query diagnostics, not a certified solid-penetration depth',
 'mapped_skin_baseline':embedded.report(X)}
(P/'input-pins.json').write_text(json.dumps({'input_npz':str(INPUT),'input_sha256':EXPECTED,
 'baseline_query_pairs_sha256':sha(BASEQUERY/'pairs.csv'),'baseline_query_nodes_sha256':sha(BASEQUERY/'nodes.csv'),
 'baseline_edge_audit_sha256':sha(BASEAUDIT),'transfer_rows_sha256':sha(TRANSFER_ROWS),
 'coarse_reference_sha256':sha(COARSE_REFERENCE),'coarse_faces_sha256':sha(COARSE_FACES),
 'detail_payload_sha256':sha(DETAIL_PAYLOAD),'separator_rows_sha256':sha(SEP_A),'separator_targets_sha256':sha(SEP_T),'separator_rows':int(S_sep.shape[0]),'separator_min_baseline_slack_m':float(actual_base_slack.min()),'adapter_sha256':sha(ADAPTER_DIR/'load_transfer_075.py'),
 'solver_sha256':sha(M/'human_skin_fat_connected_repair.py'),
 'embedded_surface_sha256':sha(M/'human_skin_fat_embedded_surface.py'),
 'baseline':baseline_summary,'active_set_passes':0,'iteration_limit':1,'separator_rows':25,'physical_steps':0,'geometry_admitted':False,
 'scope':'one static connected authoring iteration from original refined-037 baseline; full solid self-crossings and native qualification not covered'},indent=2)+'\n')
print(json.dumps({'baseline':baseline_summary,'free_nodes_include_outer_O':True,'max_iterations':1,'active_set_passes':0}),flush=True)
previous_score=baseline_summary['edge_bank_groups_with_hits']+4*baseline_summary['inside_nodes']
for it in range(1,2):
 I=P/('iteration-%03d'%it);I.mkdir()
 pairs,qmap=read_query(query,X)
 groups=collections.defaultdict(list)
 for w in edge_report['witnesses']:groups[(w['edge_index'],w['body'],w['bone_stable_id'])].append(w)
 rows=[];targets=[];provenance=[]
 def add(nodes,weights,normal,target,pro):
  row=len(targets)
  for v,w in zip(nodes,weights):
   for k in range(3):rows.append((row,3*int(v)+k,float(w*normal[k])))
  targets.append(target);provenance.append(pro)
 for key,ws in sorted(groups.items()):
  ws.sort(key=lambda w:(w['exact_result']['t'],w['triangle_stable_id']))
  a,b=ws[0]['endpoint_local_nodes'];body,bone=key[1:]
  ca=qmap.get((a,body,bone),'outside_AABB_proven');cb=qmap.get((b,body,bone),'outside_AABB_proven')
  if ca.startswith('outside'):
   spans=[(ws[j],ws[j+1]['exact_result']['t']if j+1<len(ws)else 1.)for j in range(0,len(ws),2)];toward=X[a].astype(float)
  elif cb.startswith('outside'):spans=[(ws[-1],0.)];toward=X[b].astype(float)
  else:continue
  for w,tend in spans:
   v=tri[lookup[(body,bone,w['triangle_stable_id'])]];hlocal=np.asarray(w['exact_result']['barycentric'])@v
   h=pos[body]+np.linalg.solve(rot[body],hlocal+off[body])
   normal=rot[body].T@np.cross(v[1]-v[0],v[2]-v[0]);normal/=np.linalg.norm(normal)
   if normal@(toward-h)<0:normal=-normal
   tm=.5*(w['exact_result']['t']+tend)
   add([a,b],[1-tm,tm],normal,float(normal@h+5e-5),{'kind':'exact_bone_intrusion','edge':key[0],'query_rows':[a,b],'body':body,'bone':bone,'triangle':w['triangle_stable_id'],'t':tm})
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
 def capture_lp(raw_x,raw_free,metadata):
  meta={'solver':metadata,'separator_rows':int(S_sep.shape[0]),'contact_rows':int(C.shape[0]),
        'hard_linear_rows':int(L.shape[0]),'direction_source':'unrounded scipy linprog result.x'}
  (I/'unrounded-lp-solution.json').write_text(json.dumps(meta,indent=2)+'\n')
  if raw_x is not None:np.savez_compressed(I/'unrounded-lp-solution.npz',result_x=raw_x,free_nodes=raw_free)
 Y,report=repair.coupled_clearance_step(X,R,T,free,C,target,step_limit_mm=2.,time_limit_s=60.,
    lower_linear_matrix=L,lower_linear_targets=lb,max_incremental_gradient=None,
    construction_origin=O,cumulative_displacement_weight=.002,boundary_faces=BF,
    boundary_area_bounds=(.5,2.),boundary_normal_cosine_min=.5,
    embedded_surface=embedded,linear_method='highs-ipm',lp_result_callback=capture_lp)
 (I/'construction-report.json').write_text(json.dumps(report,indent=2)+'\n')
 if not report['lp_success'] or not (I/'unrounded-lp-solution.npz').exists():
  replay_report={'direction_saved':False,'new_solver_calls':0,'status':'LP returned no successful primal solution'}
  (I/'unrounded-line-search-replay.json').write_text(json.dumps(replay_report,indent=2)+'\n')
  raise RuntimeError('LP did not produce a successful unrounded solution; stopped before candidate/query')
 else:
  # Replay exact unrounded LP variables through the same component backtracking; no new solve.
  lp=np.load(I/'unrounded-lp-solution.npz');rawx=lp['result_x'];lpfree=lp['free_nodes'];nd=3*len(lpfree)
  assert np.array_equal(lpfree,free)
  delta=rawx[:nd].reshape(-1,3)*.001
  lookupF=np.full(len(X),-1,dtype=np.int64);lookupF[free]=np.arange(len(free))
  er=[];ec=[]
  for tnodes in T:
   ix=lookupF[tnodes];ix=ix[ix>=0]
   for j0 in ix[1:]:er.extend([ix[0],j0]);ec.extend([j0,ix[0]])
  for B in [C,L]:
   for row0 in range(B.shape[0]):
    ix=np.unique(lookupF[B.indices[B.indptr[row0]:B.indptr[row0+1]]//3]);ix=ix[ix>=0]
    for j0 in ix[1:]:er.extend([ix[0],j0]);ec.extend([j0,ix[0]])
  for face0 in BF:
   ix=lookupF[face0];ix=ix[ix>=0]
   for j0 in ix[1:]:er.extend([ix[0],j0]);ec.extend([j0,ix[0]])
  Sup=embedded.support
  for row0 in range(Sup.shape[0]):
   ix=lookupF[Sup.indices[Sup.indptr[row0]:Sup.indptr[row0+1]]];ix=ix[ix>=0]
   for j0 in ix[1:]:er.extend([ix[0],j0]);ec.extend([j0,ix[0]])
  from scipy.sparse.csgraph import connected_components
  graph=sparse.coo_matrix((np.ones(len(er)),(er,ec)),shape=(len(free),len(free))).tocsr()
  comp_count,comp_labels=connected_components(graph,directed=False)
  replay_best=X.copy();replay_components=[]
  def line_gates(Y0,original_deficit):
   jj0,_,qq0,_=repair.cell_values_gradients(Y0,R,T)
   hardslack=np.asarray(L@Y0.astype(float).ravel()-lb).ravel()
   baselayer=np.asarray(Lbase@Y0.astype(float).ravel()-lbbase).ravel()
   sepslack=np.asarray(S_sep@Y0.astype(float).ravel()-sep_targets).ravel()
   _,_,ca0,cc0=repair.boundary_face_values_gradients(Y0,O,BF)
   _,_,da0,dc0=embedded.metrics(Y0)
   deficit=float(np.maximum(target-C@Y0.astype(float).ravel(),0).sum())
   checks=[
    ('q',float(qq0.min())>=1e-5,{'min':float(qq0.min()),'row':int(np.argmin(qq0))}),
    ('J',float(jj0.min())>=.35 and float(jj0.max())<=2.5,{'min':float(jj0.min()),'min_row':int(np.argmin(jj0)),'max':float(jj0.max()),'max_row':int(np.argmax(jj0))}),
    ('hard_linear',float(hardslack.min())>=-1e-9,{'min':float(hardslack.min()),'row':int(np.argmin(hardslack)),'layer_min':float(baselayer.min()),'separator_min':float(sepslack.min()),'separator_row':int(np.argmin(sepslack))}),
    ('coarse_area',float(ca0.min())>=.5 and float(ca0.max())<=2.,{'min':float(ca0.min()),'max':float(ca0.max()),'min_row':int(np.argmin(ca0)),'max_row':int(np.argmax(ca0))}),
    ('coarse_normal',float(cc0.min())>=.5,{'min':float(cc0.min()),'row':int(np.argmin(cc0))}),
    ('detail_area',float(da0.min())>=.5 and float(da0.max())<=2.,{'min':float(da0.min()),'max':float(da0.max()),'min_row':int(np.argmin(da0)),'max_row':int(np.argmax(da0))}),
    ('detail_normal',float(dc0.min())>=.5,{'min':float(dc0.min()),'row':int(np.argmin(dc0))}),
    ('contact_progress',deficit<original_deficit-1e-12,{'deficit_mm':deficit*1000,'starting_mm':original_deficit*1000}),
   ]
   return next((name for name,ok,_ in checks if not ok),None),checks
  for comp0 in range(comp_count):
   local=np.flatnonzero(comp_labels==comp0);nodes=free[local]
   initial=float(np.maximum(target-C@replay_best.astype(float).ravel(),0).sum())
   attempts=[];accepted=False
   for kk in range(15):
    alpha=2.**-kk;trial=replay_best.copy()
    trial[nodes]=(X[nodes].astype(float)+alpha*delta[local]).astype(np.float32)
    first,checks=line_gates(trial,initial)
    attempts.append({'fraction':alpha,'first_failed_gate':first,'gates':{nm:{'pass':bool(ok),**vals}for nm,ok,vals in checks}})
    if first is None:
     replay_best=trial;accepted=True;break
   replay_components.append({'component':comp0,'nodes':len(nodes),'accepted':accepted,'attempts':attempts})
  replay_match=bool(np.array_equal(replay_best.view(np.uint32),Y.view(np.uint32)))
  replay_report={'direction':'exact unrounded scipy linprog result.x','new_solver_calls':0,
    'candidate_bitwise_match':replay_match,'free_nodes_saved':int(len(lpfree)),
    'components':replay_components}
  if not replay_match:raise AssertionError('exact LP line replay differs from emitted candidate')
  (I/'unrounded-line-search-replay.json').write_text(json.dumps(replay_report,indent=2)+'\n')
  print(json.dumps({'iteration':it,'construction':report,'line_search_replay':replay_report}),flush=True)
 CAND=I/'candidate.npz';np.savez_compressed(CAND,points_world_f32=Y,tetrahedra=T,material_ids=mat)
 query=do_query(CAND,I);newpairs,_=read_query(query,Y)
 edge_report=audit.audit_unique_tet_edges(np.arange(n),T,ids,tri,locations(Y),exact['classify_segment_triangle'],node_identity_domain='query_row')
 (I/'tet-edge-report.json').write_text(json.dumps(edge_report,indent=2)+'\n')
 new_inside=[r for r in newpairs if r['classification']=='inside']
 metric={'iteration':it,'inside_signed_face_distance_min_mm':1000*min((float(r['signed_face_distance_m']) for r in new_inside),default=0.),'inside_nearest_distance_max_mm':1000*max((float(r['nearest_distance_m']) for r in new_inside),default=0.),'edge_bank_groups':edge_report['edge_bank_groups_with_hits'],'edges':edge_report['edges_with_hits'],'inside_nodes':len({int(r['query_row'])for r in newpairs if r['classification']=='inside'}),'inside_pairs':sum(r['classification']=='inside'for r in newpairs),'ambiguous':edge_report['counts']['ambiguous'],'candidate_sha256':sha(CAND),'quality':quality(Y),'candidate_path':str(CAND),'query_path':str(query),'audit_path':str(I/'tet-edge-report.json')}
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
 (P/'continuation.json').write_text(json.dumps({'history':history,'baseline':baseline_summary,'active_set_passes':0,'iteration_limit':1,'separator_rows':25,'accepted_physical_steps':0,'geometry_admitted':False,'elapsed_s':time.monotonic()-start,'input_sha256':EXPECTED,'solver_sha256':sha(M/'human_skin_fat_connected_repair.py'),'audit_sha256':sha(AUDIT_PATH),'limitations':['Static connected asset authoring only.','075 mapped detail skin was included as a constrained surface; this does not certify all-bank surface clearance or full solid self-crossings.','No mass, material or native physical qualification.','25 fine-skin separator inequalities are hard linear rows appended to thickness rows, without contact slack.']},indent=2)+'\n')
 print(json.dumps(metric),flush=True)
 if 'stop_reason' in metric:break
 previous_score=score
 X=Y
print('CONNECTED_REFINED_CONSTRUCTION_ENDED',flush=True)
