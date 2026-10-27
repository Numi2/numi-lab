#!/usr/bin/env python3
"""Direct sparse contact-feature projection for 43/16 on exact pinned 806."""
from pathlib import Path
import csv,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';B=W/'combined-d071-b109-b61-001';OUT=Path(__file__).resolve().parent
FIT= W/'b43-16/attempt-031-contact-feature-fit-001/fit_contact_features.py'
SCREEN=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b43-16/attempt-030-radius-aware-position-001/rebase-024-on-806/deep-backtrack-001/screen_rebased_direction.py')
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
DPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/point-depth-barriers-357/point_depth_barriers_357.py')
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
BANKHELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight';LAYER=B/'skin-fat-layers-541.npz'
EXPORT=E/'E/generic-static-query-042/prepare_generic_query_points.py';QUERYBIN=Path('/tmp/generic-query-overlay-build-042-work/matter/numi-matter-human-skin-fat-static-query')
EXPECTED='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
ALPHAS=[1.0,.5,.25,.125,.0625];CONTACT_MARGIN=.02;POINT_MARGIN=.03

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def point_query_nodes(Y,nodes,outdir):
 outdir.mkdir(parents=True,exist_ok=True);pnpz=outdir/'points-world.npz';np.savez_compressed(pnpz,points_world_f32=np.asarray(Y,dtype=np.float32)[nodes])
 points=outdir/'points.csv';args=[sys.executable,str(EXPORT),'--candidate-npz',str(pnpz),'--input-sha256',sha(pnpz),'--field','points_world_f32','--output',str(points)]
 pr=subprocess.run(args,capture_output=True,text=True);(outdir/'prepare.log').write_text(pr.stdout+pr.stderr)
 if pr.returncode:raise RuntimeError('point exporter failed')
 pairs=outdir/'pairs.csv';nodes_csv=outdir/'nodes.csv';cmd=[str(QUERYBIN),str(BANKDIR/'exact-emitted.triangles.csv'),str(BANKDIR/'exact-emitted.bodies.csv'),str(points),str(pairs),str(nodes_csv),'185']
 run=subprocess.run(cmd,capture_output=True,text=True);(outdir/'query.log').write_text(run.stdout+run.stderr)
 if run.returncode:raise RuntimeError('exact all-bank point query failed')
 return list(csv.DictReader(points.open())),list(csv.DictReader(pairs.open())),list(csv.DictReader(nodes_csv.open())),pnpz

def main():
 t=time.monotonic();basepath=B/'candidate.npz';summary=B/'audit-130/summary.json';auditpath=B/'audit-159/report.json'
 if sha(basepath)!=EXPECTED:raise ValueError('pinned 806 SHA mismatch')
 with np.load(basepath,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
 X=payload['points_world_m'].astype(np.float32);F=payload['source_O_faces'].astype(np.int64);ref=payload['area_reference_037_points_world_m_f64'];anchors=payload['typed_coarse_anchor_output_rows_018'].astype(np.int64)
 doc=json.load(open(summary));auditdoc=json.load(open(auditpath))
 if doc.get('candidate_sha256')!=EXPECTED or auditdoc.get('candidate_sha256')!=EXPECTED or not auditdoc.get('complete'):raise ValueError('baseline audits stale')
 wits=[r for r in doc['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id']))==(43,16) and r.get('classification')=='inside' and int(r.get('ambiguous',0))==0];selected=np.asarray(sorted(int(r['query_row']) for r in wits),dtype=np.int64)
 if len(selected)!=43:raise ValueError('43/16 inside witness row mismatch')
 b43=next(r for r in auditdoc['banks'] if (int(r['body']),int(r['stable_bone_id']))==(43,16));rawbase={(int(f),int(bt)) for f,bt in (b43.get('result') or {}).get('triangle_pairs',[])}
 if len(rawbase)!=247:raise ValueError('target raw pair pin mismatch')
 ip=payload['refinement_face_parent_ancestry_indptr'];pr=payload['refinement_face_parent_ancestry_face_rows'];mapbase=mod('proj_screen',SCREEN).map_pairs(rawbase,ip,pr)
 # Reconstruct actual contact endpoints from the already defined 806 exact pair set.
 fitter=mod('proj_contact_fit',FIT);helper=mod('proj_bank_helper',BANKHELPER);Tbank,bids,_,_,M,offset,pos,_,_,_=helper.read_bank_inputs(BANKDIR,E)
 trirow={int(bids[i,3]):int(i) for i in np.flatnonzero((bids[:,0]==43)&(bids[:,1]==16))};Xlocal=(X.astype(np.float64)-pos[43])@M[43].T-offset[43]
 features=[];counts={}
 for sf,bt in sorted(rawbase):
  if bt not in trirow:raise ValueError('missing target typed bank triangle')
  loc=trirow[bt];feat=fitter.pair_contact_features(Xlocal[F[sf]],Tbank[loc])
  if not feat:raise ValueError(f'no true contact feature reconstructed for {sf}/{bt}')
  counts[len(feat)]=counts.get(len(feat),0)+1
  nl=np.cross(Tbank[loc][1]-Tbank[loc][0],Tbank[loc][2]-Tbank[loc][0]);nl/=np.linalg.norm(nl);nw=nl@M[43];nw/=np.linalg.norm(nw)
  for p,bary in feat:features.append((int(sf),int(bt),np.asarray(bary,float),np.asarray(nw,float)))
 if len(features)<247:raise ValueError('contact endpoint feature count incomplete')
 # Four-ring connected support around all actual point penetrations plus known near-self guard face nodes.
 edges=np.unique(np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
 adj=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(X),len(X)))
 support=set(map(int,selected));front=set(support)
 for _ in range(4):front=set(map(int,adj[list(front)].indices))-support;support.update(front)
 selfguard=set()
 for p in [W/'b43-16/attempt-023-original-patch-layer-depth-001/exact-self-check.json',W/'b43-16/attempt-024-witness-constrained-fit-001/exact-self-check.json']:
  for a,b in json.load(open(p))['exact_self_pairs']:selfguard.add(tuple(sorted((int(a),int(b)))))
 selfguard=np.asarray(sorted(selfguard),dtype=np.int64).reshape(-1,2);support.update(map(int,F[selfguard.reshape(-1)].reshape(-1)))
 support=np.asarray(sorted(support),dtype=np.int64);loc={int(v):i for i,v in enumerate(support)};affected=np.flatnonzero(np.any(np.isin(F,support),axis=1)).astype(np.int64)
 if any(int(v) not in loc for f,bt,bary,nw in features for v in F[f]):raise ValueError('actual crossing face is outside connected support')
 n=3*len(support);rr=[];cc=[];vv=[]
 for j,(f,bt,bary,nw) in enumerate(features):
  for k,node in enumerate(F[f]):
   for a in range(3):rr.append(j);cc.append(3*loc[int(node)]+a);vv.append(float(bary[k]*nw[a]))
 C=sparse.coo_matrix((vv,(rr,cc)),shape=(len(features),n)).tocsr()
 # Exact point-depth direction rows for all 43 inside skin nodes.
 q=mod('proj_q',QPATH);depth=mod('proj_depth',DPATH);area=mod('proj_area',q.DEFAULT_AREA_OWNER)
 with np.load(basepath,allow_pickle=False) as za:source_mm=q.compose_source_reference(za,Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'))[0]*1000.
 bm=X.astype(np.float64)*1000.;query=B/'audit-130/query';spec={'points_csv':query/'points.csv','pairs_csv':query/'pairs.csv','nodes_csv':query/'nodes.csv','query_receipt_json':query/'query-receipt.json','kind':'inside','require_points_match':True}
 outside=[];supportset=set(map(int,support))
 with (query/'pairs.csv').open(newline='') as fh:
  for r in csv.DictReader(fh):
   if r['classification']=='outside' and int(r['query_row']) in supportset and float(r['nearest_distance_m'])<=.0011:outside.append((int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id']),int(r['nearest_triangle'])))
 ns,norms,low,typed,records,depthreceipt=depth.build_point_depth_rows(X,[spec,{**spec,'kind':'outside','selected_typed_pairs':outside}],BANKDIR/'exact-emitted.bodies.csv')
 prrows,prmeta=q.build_scoped_output_node_direction_rows(len(X),support,anchors,ns,norms,low,typed)
 ik=np.asarray([r['kind']=='inside_nonregression' for r in records],bool);take=np.flatnonzero((typed[:,1]==43)&(typed[:,2]==16)&ik&np.isin(ns,support)&(~np.isin(ns,anchors)))
 if len(take)!=43:raise ValueError(f'point target count {len(take)}')
 rr=[];cc=[];vv=[]
 for i,k in enumerate(take):
  for a in range(3):rr.append(i);cc.append(3*loc[int(ns[k])]+a);vv.append(float(norms[k,a]))
 Tp=sparse.coo_matrix((vv,(rr,cc)),shape=(len(take),n)).tocsr()
 # Sparse graph strain operator, inverse edge-length weighted.
 er=[];ec=[];ev=[];rnum=0
 for u,v in edges:
  iu=loc.get(int(u));iv=loc.get(int(v))
  if iu is None and iv is None:continue
  weight=1.0/np.sqrt(max(float(np.linalg.norm(bm[u]-bm[v])),.05))
  for a in range(3):
   if iu is not None:er.append(rnum);ec.append(3*iu+a);ev.append(weight)
   if iv is not None:er.append(rnum);ec.append(3*iv+a);ev.append(-weight)
   rnum+=1
 E0=sparse.coo_matrix((ev,(er,ec)),shape=(rnum,n)).tocsr()
 # Direct sparse normal equations; no iterative optimizer loop.
 H=(C.T@C)+.15*(E0.T@E0)+10.0*(Tp.T@Tp)+.01*sparse.eye(n,format='csr')
 rhs=C.T@np.full(len(features),.02)+10.0*Tp.T@np.full(len(take),.03)
 d=spsolve(H.tocsc(),np.asarray(rhs).reshape(-1)).reshape(-1,3)
 np.savez_compressed(OUT/'contact-feature-direction.npz',support_nodes=support,direction_delta_mm=d,contact_target_mm=np.full(len(features),.02),point_target_mm=np.full(len(take),.03))
 maxmove=float(np.linalg.norm(d,axis=1).max(initial=0))
 if maxmove>.5:d*=.5/maxmove
 contactpred=np.asarray(C@d.reshape(-1));pointpred=np.asarray(Tp@d.reshape(-1))
 solver={'method':'sparse_normal_equation_contact_endpoint_projection','success':bool(np.isfinite(d).all()),'variables':n,'contact_feature_rows':len(features),'raw_crossing_pairs':len(rawbase),'endpoint_count_distribution':{str(k):v for k,v in counts.items()},'point_depth_target_rows':len(take),'point_depth_guard_rows':len(prrows),'support_nodes':len(support),'affected_faces':len(affected),'graph_rows':rnum,'max_step_mm':maxmove,'contact_predicted_min_mm':float(contactpred.min(initial=np.inf)),'contact_predicted_sum_mm':float(contactpred.sum()),'contact_rows_at_or_above_0_02mm':int(np.count_nonzero(contactpred>=.02)),'point_predicted_min_mm':float(pointpred.min(initial=np.inf)),'point_predicted_sum_mm':float(pointpred.sum()),'point_rows_at_or_above_0_03mm':int(np.count_nonzero(pointpred>=.03))}
 screen=mod('proj_screen2',SCREEN);tf=mod('proj_transfer',W/'b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py')
 with np.load(LAYER,allow_pickle=False) as z:layerz={k:z[k].copy() for k in z.files}
 raw_trials=[];firstexact=None;selected_candidate=None;artifact=None
 for alpha in ALPHAS:
  Y=X.copy();Y[support]=(X[support].astype(np.float64)+(alpha*d)/1000.).astype(np.float32)
  changed=np.flatnonzero(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64);chf=np.flatnonzero(np.any(np.isin(F,changed),axis=1)).astype(np.int64)
  outside_rows=np.setdiff1d(np.arange(len(X),dtype=np.int64),support);out_same=bool(np.array_equal(Y[outside_rows].view(np.uint32),X[outside_rows].view(np.uint32)));anc_same=bool(np.array_equal(Y[anchors].view(np.uint32),X[anchors].view(np.uint32)))
  shapep,sm,ag=fitter.shape_pass(X,Y,F,chf,screen,q,ref);lg=screen.layer_check(X,Y,F,chf,layerz,tf)
  tr={'alpha':alpha,'changed_nodes':int(len(changed)),'changed_faces':int(len(chf)),'outside_bitwise_unchanged':out_same,'anchors_bitwise_unchanged':anc_same,'shape_pass':shapep,'shape':sm,'area_gate':ag,'layer_gate':lg,'exact_checked':False}
  if not(out_same and anc_same and shapep and ag['pass'] and lg['pass']):tr['status']='REJECTED_SHAPE_OR_LAYER';raw_trials.append(tr);continue
  # Exact self and target-bank screens are run only after emitted-F32 shape/layer checks.
  aud=mod('proj_audit',AUD159);exact=aud.exact_module();bankhelper=mod('proj_bankhelper_again',BANKHELPER);BT,bi,_,_,MM,off,ps,keys,_,_=bankhelper.read_bank_inputs(BANKDIR,E)
  baseline_bank=next(r for r in auditdoc['banks'] if (int(r['body']),int(r['stable_bone_id']))==(43,16));stored={(int(f),int(bt)) for f,bt in baseline_bank['result']['triangle_pairs']}
  before,box0=screen.audit_target(X.astype(np.float64),F,chf,BT,bi,MM,off,ps,exact,aud);expected={(f,bt) for f,bt in stored if f in set(map(int,chf))}
  if before!=expected:raise ValueError('baseline exact local target control mismatch')
  afterlocal,box1=screen.audit_target(Y.astype(np.float64),F,chf,BT,bi,MM,off,ps,exact,aud);after=(stored-expected)|afterlocal;mapafter=screen.map_pairs(after,ip,pr);newraw=after-stored;removedraw=stored-after;newmap=mapafter-mapbase;removedmap=mapbase-mapafter
  records,_=exact.float32_surface_records(Y,F,source_seam_groups=());selfres=exact._audit_self_pairs_touching_faces(records,set(map(int,chf)));selfpairs={(int(a),int(b)) for a,b in (selfres.get('triangle_pairs') or [])}
  useful=bool(not newraw and not newmap and not selfpairs and len(mapafter)<len(mapbase))
  tr.update(status='EXACT_USEFUL' if useful else 'REJECTED_EXACT_NO_MAPPED_REDUCTION',exact_checked=True,exact_target={'raw_before':len(stored),'raw_after':len(after),'new_raw':sorted([list(x) for x in newraw]),'removed_raw':sorted([list(x) for x in removedraw]),'mapped_before':len(mapbase),'mapped_after':len(mapafter),'new_mapped':sorted([list(x) for x in newmap]),'removed_mapped':sorted([list(x) for x in removedmap]),'local_aabb_before':box0,'local_aabb_after':box1},exact_self={'count':len(selfpairs),'pairs':sorted([list(x) for x in selfpairs])})
  raw_trials.append(tr)
  if firstexact is None:
   firstexact=(Y,chf,changed,tr);artifact=OUT/'candidate-first-exact-private.npz';saved=dict(payload);saved['points_world_m']=Y
   with artifact.open('xb') as f:np.savez_compressed(f,**saved)
  if useful:selected_candidate=(Y,chf,changed,tr);break
 # Exact all-185 point classification/depth for 43 inside witnesses at the first shape/layer-pass backtrack.
 point_depth_report=None
 if firstexact is not None:
  Y0,_,_,_=firstexact;qd=OUT/'inside-query-43';prows,ppairs,nrows,pnpz=point_query_nodes(Y0,selected,qd)
  if len(prows)!=len(selected) or len(nrows)!=len(selected):raise ValueError('43-node exact all-185 point query cardinality mismatch')
  cand_inside=set();node_records=[]
  for qi,nr in enumerate(nrows):
   if int(nr['total_bone_banks'])!=185 or int(nr['ambiguous_banks']) or int(nr['invalid_banks']):raise ValueError('43-node all-bank query invalid/ambiguous')
  for row in ppairs:
   if row['classification']=='inside':cand_inside.add((int(selected[int(row['query_row'])]),int(row['bank_body']),int(row['bank_stable_id'])))
  baseinside={(int(r['query_row']),int(r['bank_body']),int(r['bank_stable_id'])) for r in doc['inside_witnesses'] if int(r['query_row']) in set(map(int,selected))}
  targetrows=[r for r in ppairs if (int(selected[int(r['query_row'])]),int(r['bank_body']),int(r['bank_stable_id'])) in {(int(x),43,16) for x in selected}]
  depthvals=[abs(float(r['signed_face_distance_m']))*1000. for r in targetrows if r['classification']=='inside' and r['signed_face_distance_m'] not in ('','None')]
  base_depth=[abs(float(r['signed_face_distance_m']))*1000. for r in wits]
  point_depth_report={'node_count':len(selected),'bank_count_per_node':[int(r['total_bone_banks']) for r in nrows],'baseline_target_inside_count':len(baseinside),'candidate_target_inside_count':sum(r['classification']=='inside' for r in targetrows),'baseline_target_signed_abs_depth_mm':{'min':float(np.min(base_depth)),'median':float(np.median(base_depth)),'max':float(np.max(base_depth))},'candidate_target_signed_abs_depth_mm':{'min':float(np.min(depthvals)) if depthvals else None,'median':float(np.median(depthvals)) if depthvals else None,'max':float(np.max(depthvals)) if depthvals else None},'baseline_inside_pairs':sorted([list(x) for x in baseinside]),'candidate_inside_pairs':sorted([list(x) for x in cand_inside]),'new_inside_pairs':sorted([list(x) for x in cand_inside-baseinside]),'removed_inside_pairs':sorted([list(x) for x in baseinside-cand_inside]),'all_185_ambiguous_or_invalid':0,'query_points_npz_sha256':sha(pnpz)}
 report={'schema':'numi.human.b43-16.contact-feature-projection-806.v1','status':'EXACT_USEFUL_CANDIDATE_PENDING_ALL_BANK_FACE_AUDIT' if selected_candidate else ('REJECTED_NO_EXACT_USEFUL_BACKTRACK' if firstexact else 'REJECTED_NO_SHAPE_LAYER_PASS'),'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'base_candidate_sha256':EXPECTED,'audit130_sha256':sha(summary),'audit159_sha256':sha(auditpath),'layer_bundle_sha256':sha(LAYER),'baseline_target_raw_pairs':len(rawbase),'baseline_target_mapped_identities':len(mapbase),'baseline_inside_nodes':selected.tolist(),'baseline_depth_mm':{'min':min(abs(float(r['signed_face_distance_m']))*1000 for r in wits),'median':float(np.median([abs(float(r['signed_face_distance_m']))*1000 for r in wits])),'max':max(abs(float(r['signed_face_distance_m']))*1000 for r in wits)},'actual_intersection_feature_count':len(features),'contact_endpoint_count_distribution':{str(k):v for k,v in counts.items()},'support_nodes':len(support),'affected_faces':len(affected),'self_guard_pair_count':len(selfguard),'solver':solver,'backtracks':raw_trials,'inside_depth_query':point_depth_report,'first_exact_artifact_path':str(artifact) if artifact else None,'elapsed_s':time.monotonic()-t}
 if firstexact and artifact:report['first_exact_artifact_sha256']=sha(artifact)
 if selected_candidate:
  Y,chf,changed,tr=selected_candidate;path=OUT/'candidate-exact-private.npz';saved=dict(payload);saved['points_world_m']=Y
  with path.open('xb') as f:np.savez_compressed(f,**saved)
  report.update(candidate_path=str(path),candidate_sha256=sha(path),selected_alpha=tr['alpha'],changed_nodes=changed.tolist(),changed_face_rows=chf.tolist(),selected_trial=tr)
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'solver':solver,'baseline_raw':len(rawbase),'baseline_mapped':len(mapbase),'endpoint_count_distribution':counts,'backtracks':[{'alpha':r['alpha'],'status':r['status'],'shape':r['shape_pass'],'layer':r['layer_gate']['pass'],'exact':r['exact_checked'],'raw_after':(r.get('exact_target') or {}).get('raw_after'),'mapped_after':(r.get('exact_target') or {}).get('mapped_after'),'self':(r.get('exact_self') or {}).get('count')} for r in raw_trials],'inside_depth_query':point_depth_report,'candidate':report.get('candidate_path'),'sha':report.get('candidate_sha256')},indent=2))
if __name__=='__main__':main()
