#!/usr/bin/env python3
"""Bounded 43/16 contact-feature clearance fit on exact pinned 806; private diagnostic only."""
from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';B=W/'combined-d071-b109-b61-001';OUT=Path(__file__).resolve().parent
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
DPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/point-depth-barriers-357/point_depth_barriers_357.py')
FIT024=W/'b43-16/attempt-024-witness-constrained-fit-001/fit_43_16_witness_constraints.py'
SCREEN=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b43-16/attempt-030-radius-aware-position-001/rebase-024-on-806/deep-backtrack-001/screen_rebased_direction.py')
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
BANKHELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
SOURCE=Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz')
LAYER=B/'skin-fat-layers-541.npz'
EXPECTED='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
CONTACT_MARGIN_MM=.02; POINT_MARGIN_MM=.03; EDGE_LIMIT_MM=.03; TRUST_MM=.5
ALPHAS=[1.0,.5,.25,.125,.0625]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def barycentric(p,t):
 a,b,c=t;v0=b-a;v1=c-a;v2=p-a;d00=v0@v0;d01=v0@v1;d11=v1@v1;d20=v2@v0;d21=v2@v1;den=d00*d11-d01*d01
 if abs(den)<1e-30:return None
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;return np.array([1-v-w,v,w])
def inside_bary(w,tol=2e-8):return w is not None and float(np.min(w))>=-tol and float(np.max(w))<=1+tol
def pair_contact_features(skin,bone):
 """Return barycentric skin points at actual triangle/triangle plane intersections."""
 scale=max(float(np.max(np.linalg.norm(np.diff(np.vstack([skin,skin[:1]]),axis=0),axis=1))),float(np.max(np.linalg.norm(np.diff(np.vstack([bone,bone[:1]]),axis=0),axis=1))),1e-6)
 tol=max(1e-12,scale*1e-8)
 nb=np.cross(bone[1]-bone[0],bone[2]-bone[0]);nn=float(np.linalg.norm(nb))
 if nn<=1e-15:return []
 nb=nb/nn; ds=(skin-bone[0])@nb; found=[]
 def add_point(p,ws):
  wb=barycentric(p,bone)
  if inside_bary(wb,5e-7):found.append((np.asarray(p),np.asarray(ws,dtype=np.float64)))
 for i in range(3):
  j=(i+1)%3;di,dj=float(ds[i]),float(ds[j])
  if abs(di)<=tol:add_point(skin[i],np.eye(3)[i])
  if di*dj<0:
   t=di/(di-dj);p=skin[i]+t*(skin[j]-skin[i]);w=np.zeros(3);w[i]=1-t;w[j]=t;add_point(p,w)
 # Also intersect the bone edges with the skin plane; this recovers endpoints when skin edges are coplanar.
 ns=np.cross(skin[1]-skin[0],skin[2]-skin[0]);sn=float(np.linalg.norm(ns))
 if sn>1e-15:
  ns=ns/sn; db=(bone-skin[0])@ns
  for i in range(3):
   j=(i+1)%3;di,dj=float(db[i]),float(db[j])
   pts=[]
   if abs(di)<=tol:pts.append(bone[i])
   if di*dj<0:
    t=di/(di-dj);pts.append(bone[i]+t*(bone[j]-bone[i]))
   for p in pts:
    w=barycentric(p,skin)
    if inside_bary(w,5e-7):found.append((np.asarray(p),w))
 # Deduplicate contact endpoints while preserving the corresponding skin barycentrics.
 unique=[]
 for p,w in found:
  if not any(float(np.linalg.norm(p-q))<=max(5e-10,scale*2e-7) for q,_ in unique):unique.append((p,w))
 if len(unique)>2:
  pairs=[(float(np.linalg.norm(unique[i][0]-unique[j][0])),i,j) for i in range(len(unique)) for j in range(i+1,len(unique))]
  _,i,j=max(pairs);unique=[unique[i],unique[j]]
 return unique
def shape_pass(X,Y,F,rows,screen,q,ref):
 m=screen.shape_metrics(X,Y,F,rows);area=q.full_f32_area_gate(Y,X.astype(np.float64)*1000.,ref.astype(np.float64)*1000.,F,.1)
 passed=bool(m['zero_area_after']==0 and m['minimum_area_ratio']>=screen.AREA_MIN and m['maximum_normal_turn_deg']<=screen.NORMAL_MAX and m['minimum_min_angle_delta_deg']>=screen.MIN_ANGLE_DELTA and m['maximum_aspect_ratio_ratio']<=screen.ASPECT_MAX and area['pass'])
 return passed,m,area
def mapped(raw,ip,pr):
 out=set()
 for f,t in raw:
  for p in pr[int(ip[f]):int(ip[f+1])]:out.add((int(p),int(t)))
 return out
def main():
 t=time.monotonic();basepath=B/'candidate.npz';summary=B/'audit-130/summary.json';auditpath=B/'audit-159/report.json';query=B/'audit-130/query'
 if sha(basepath)!=EXPECTED:raise ValueError('pinned 806 candidate SHA mismatch')
 with np.load(basepath,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
 X=np.asarray(payload['points_world_m'],dtype=np.float32);F=np.asarray(payload['source_O_faces'],dtype=np.int64);ref=payload['area_reference_037_points_world_m_f64'];anchors=np.asarray(payload['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
 doc=json.load(open(summary));aud=json.load(open(auditpath))
 if doc.get('candidate_sha256')!=EXPECTED or not doc.get('complete_surface_audit') or aud.get('candidate_sha256')!=EXPECTED or not aud.get('complete') or int(aud.get('expected_bank_count',0))!=185:raise ValueError('pinned complete 806 audits unavailable')
 wits=[r for r in doc['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id']))==(43,16) and r.get('classification')=='inside' and int(r.get('ambiguous',0))==0]
 selected=np.asarray(sorted(int(r['query_row']) for r in wits),dtype=np.int64)
 if len(selected)!=43:raise ValueError(f'expected 43 exact 43/16 inside rows, found {len(selected)}')
 bankrec=next(r for r in aud['banks'] if (int(r['body']),int(r['stable_bone_id']))==(43,16))
 rawbase={(int(f),int(bt)) for f,bt in (bankrec.get('result') or {}).get('triangle_pairs',[])}
 if len(rawbase)!=247:raise ValueError(f'expected 247 exact target pairs, found {len(rawbase)}')
 ip=np.asarray(payload['refinement_face_parent_ancestry_indptr'],dtype=np.int64);pr=np.asarray(payload['refinement_face_parent_ancestry_face_rows'],dtype=np.int64);mapbase=mapped(rawbase,ip,pr)
 edges=np.unique(np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
 adj=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(X),len(X)))
 support=set(map(int,selected));front=set(support)
 for _ in range(4):front=set(map(int,adj[list(front)].indices))-support;support.update(front)
 # Include face vertices from retained failed-fit self witnesses as barriers; this adds only local ring vertices.
 selfpaths=[W/'b43-16/attempt-023-original-patch-layer-depth-001/exact-self-check.json',W/'b43-16/attempt-024-witness-constrained-fit-001/exact-self-check.json']
 selfguard=set()
 for p in selfpaths:
  for a,b in json.load(open(p))['exact_self_pairs']:selfguard.add(tuple(sorted((int(a),int(b)))))
 selfguard=np.asarray(sorted(selfguard),dtype=np.int64).reshape(-1,2);support.update(map(int,F[selfguard.reshape(-1)].reshape(-1)))
 support=np.asarray(sorted(support),dtype=np.int64);loc={int(n):i for i,n in enumerate(support)}
 affected=np.flatnonzero(np.any(np.isin(F,support),axis=1)).astype(np.int64);affected_set=set(map(int,affected))
 # Load the exact typed bank triangles and recover contact segment endpoints for every exact pair.
 bankmod=mod('contact_bank_helper',BANKHELPER);bankT,bankids,_,_,M,offset,pos,bankkeys,bankpins,_=bankmod.read_bank_inputs(BANKDIR,E)
 trirow={int(bankids[i,3]):int(i) for i in np.flatnonzero((bankids[:,0]==43)&(bankids[:,1]==16))}
 Xlocal=(X.astype(np.float64)-pos[43])@M[43].T-offset[43]
 contact_features=[];miss=[];feature_counts={}
 for sf,bt in sorted(rawbase):
  if int(bt) not in trirow:miss.append([int(sf),int(bt)]);continue
  feat=pair_contact_features(Xlocal[F[int(sf)]],bankT[trirow[int(bt)]])
  if not feat:miss.append([int(sf),int(bt)]);continue
  feature_counts[len(feat)]=feature_counts.get(len(feat),0)+1
  normal=np.cross(bankT[trirow[int(bt)]][1]-bankT[trirow[int(bt)]][0],bankT[trirow[int(bt)]][2]-bankT[trirow[int(bt)]][0]);normal/=max(float(np.linalg.norm(normal)),1e-300)
  nworld=normal@M[43];nworld/=max(float(np.linalg.norm(nworld)),1e-300)
  for point,bary in feat:contact_features.append({'skin_face':int(sf),'bone_triangle':int(bt),'point_local_m':point.tolist(),'skin_bary':bary.tolist(),'normal_world':nworld.tolist()})
 if miss:raise ValueError(f'could not reconstruct actual contact features for {len(miss)} of 247 exact pair rows; first {miss[:8]}')
 if len(contact_features)<247:raise ValueError('contact feature inventory is incomplete')
 for cf in contact_features:
  if any(int(v) not in loc for v in F[cf['skin_face']]):raise ValueError(f'crossing face escaped connected support {cf["skin_face"]}')
 # Confirm the typed bone winding points outward from all inside query witnesses.
 votes=[]
 for r in wits:
  tri=bankT[trirow[int(r['nearest_triangle'])]];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);ql=np.asarray([float(r['query_local_x']),float(r['query_local_y']),float(r['query_local_z'])]);cl=np.asarray([float(r['closest_x']),float(r['closest_y']),float(r['closest_z'])]);votes.append(float((ql-cl)@n))
 if len(votes)!=43 or not all(v<0 for v in votes):raise ValueError('typed bank outward winding is not confirmed by inside witnesses')
 # Point-depth rows: exact inside witnesses plus nearby outside controls for every typed bank in the connected support.
 q=mod('contact_q',QPATH);depth=mod('contact_depth',DPATH);area=mod('contact_area',q.DEFAULT_AREA_OWNER);q.verify_area_linearization_layout(area)
 with np.load(basepath,allow_pickle=False) as za:source_mm=q.compose_source_reference(za,SOURCE)[0]*1000.
 bm=X.astype(np.float64)*1000.
 spec={'points_csv':query/'points.csv','pairs_csv':query/'pairs.csv','nodes_csv':query/'nodes.csv','query_receipt_json':query/'query-receipt.json','kind':'inside','require_points_match':True}
 supportset=set(map(int,support));outside_pairs=[]
 with (query/'pairs.csv').open(newline='') as fh:
  for row in csv.DictReader(fh):
   if row['classification']=='outside' and int(row['query_row']) in supportset and float(row['nearest_distance_m'])<=.0011:outside_pairs.append((int(row['query_row']),int(row['bank_body']),int(row['bank_stable_id']),int(row['nearest_triangle'])))
 ns,norms,low,typed,records,depth_receipt=depth.build_point_depth_rows(X,[spec,{**spec,'kind':'outside','selected_typed_pairs':outside_pairs}],BANKDIR/'exact-emitted.bodies.csv')
 prrows,prmeta=q.build_scoped_output_node_direction_rows(len(X),support,anchors,ns,norms,low,typed)
 inside_kind=np.asarray([r['kind']=='inside_nonregression' for r in records],dtype=bool)
 ptarg=np.flatnonzero((typed[:,1]==43)&(typed[:,2]==16)&inside_kind&np.isin(ns,support)&(~np.isin(ns,anchors)))
 if len(ptarg)!=43:raise ValueError(f'point-depth 43/16 rows changed: {len(ptarg)}')
 n=3*len(support);rr=[];cc=[];vv=[]
 for j,k in enumerate(ptarg):
  for a in range(3):rr.append(j);cc.append(3*loc[int(ns[k])]+a);vv.append(float(norms[k,a]))
 Tpoint=sparse.coo_matrix((vv,(rr,cc)),shape=(len(ptarg),n)).tocsr()
 rr=[];cc=[];vv=[]
 for j,cf in enumerate(contact_features):
  f=int(cf['skin_face']);ba=np.asarray(cf['skin_bary'],dtype=np.float64);nw=np.asarray(cf['normal_world'],dtype=np.float64)
  for k,node in enumerate(F[f]):
   for a in range(3):rr.append(j);cc.append(3*loc[int(node)]+a);vv.append(float(ba[k]*nw[a]))
 Ccontact=sparse.coo_matrix((vv,(rr,cc)),shape=(len(contact_features),n)).tocsr()
 # Preserve shape, source chart, skin/fat prisms and known near-self witnesses.
 normal,_,_,req=q.fixed_reference_area_requirements(bm,ref,F)
 G,rhs,_,_=area.area_linearization(bm,F,affected,normal[affected],req[affected],minimum_ratio=.1)
 tri=bm[F[affected]];n0=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);a0=np.linalg.norm(n0,axis=1)
 if np.any(a0<=0):raise ValueError('baseline affected face degenerate')
 Gu,lo_area,cur,_=area.area_linearization(bm,F,affected,n0/a0[:,None],a0,minimum_ratio=.76)
 Gn,_,proj,_=area.area_linearization(bm,F,affected,n0/a0[:,None],a0,minimum_ratio=.76)
 G=sparse.vstack((G,Gu,-Gu,Gn-.91*Gu),format='csr');rhs=np.r_[rhs,lo_area,cur-1.29*a0,.91*a0-proj]
 sep=mod('contact_self_sep',q.DEFAULT_PAIR_OWNER);selfrows,selfmeta=sep.build_pair_rows(bm,F,selfguard,margin_fraction=.25)
 A,b=q._owner_matrix(prrows+selfrows,G,rhs,support)
 layerowner=mod('contact_old_layer_fit',FIT024);LG,lb,layermeta=layerowner.make_layer_guard(LAYER,F,affected,support,{})
 A=sparse.vstack((A,LG),format='csr');b=np.r_[b,lb]
 lo,hi,boundmeta=q.source_safe_trust_bounds(bm,source_mm,support,anchors,.5,12.)
 lo=np.maximum(lo,-TRUST_MM);hi=np.minimum(hi,TRUST_MM)
 if np.any(lo>hi):raise ValueError('source and local trust bounds conflict')
 EG,eb,edgemeta=q.build_edge_gradient_constraints(bm,F,support,EDGE_LIMIT_MM)
 # Sparse L1 continuation: exact crossing-feature and inside depth rows are soft;
 # area, source, self, anchors and skin/fat guards remain hard.
 na=n;ne=EG.shape[0];nc=Ccontact.shape[0];npnt=Tpoint.shape[0]
 I=sparse.eye(n,format='csr');Ie=sparse.eye(ne,format='csr');Ic=sparse.eye(nc,format='csr');Ip=sparse.eye(npnt,format='csr');Z=lambda r,c:sparse.csr_matrix((r,c))
 blocks=[sparse.hstack((-A,Z(A.shape[0],n),Z(A.shape[0],ne),Z(A.shape[0],nc),Z(A.shape[0],npnt))),
  sparse.hstack((I,-I,Z(n,ne),Z(n,nc),Z(n,npnt))),sparse.hstack((-I,-I,Z(n,ne),Z(n,nc),Z(n,npnt))),
  sparse.hstack((EG,Z(ne,n),-Ie,Z(ne,nc),Z(ne,npnt))),
  sparse.hstack((-Ccontact,Z(nc,n),Z(nc,ne),-Ic,Z(nc,npnt))),
  sparse.hstack((-Tpoint,Z(npnt,n),Z(npnt,ne),Z(npnt,nc),-Ip))]
 Aub=sparse.vstack(blocks,format='csr');bub=np.r_[-b,np.zeros(2*n),eb,-np.full(nc,CONTACT_MARGIN_MM),-np.full(npnt,POINT_MARGIN_MM)]
 objective=np.r_[np.zeros(n),np.full(n,.01),np.full(ne,10.),np.full(nc,5000.),np.full(npnt,1000.)]
 bounds=list(zip(lo,hi))+[(0,None)]*(n+ne+nc+npnt)
 sol=linprog(objective,A_ub=Aub,b_ub=bub,bounds=bounds,method='highs',options={'maxiter':10000})
 lp={'success':bool(sol.success),'status':int(sol.status),'message':str(sol.message),'iterations':int(getattr(sol,'nit',0)),'objective':float(sol.fun) if sol.success else None,'variables':len(objective),'constraints':int(Aub.shape[0]),'nonzeros':int(Aub.nnz),'support_nodes':len(support),'coordinate_dofs':n,'contact_feature_rows':nc,'raw_pair_rows':len(rawbase),'feature_endpoint_count_distribution':{str(k):v for k,v in feature_counts.items()},'point_target_rows':npnt,'point_depth_guard_rows':int(len(prrows)),'self_guard_pairs':len(selfguard),'self_guard_rows':int(len(selfrows)),'layer_guard_rows':int(LG.shape[0]),'affected_faces':len(affected),'edge_gradient_rows':ne,'hard_constraint_rows':int(A.shape[0])}
 report={'schema':'numi.human.b43-16.contact-feature-fit-806.v1','status':'LP_FAILED' if not sol.success else 'SOLVED_PENDING_EMISSION','baseline_sha256':EXPECTED,'audit130_sha256':sha(summary),'audit159_sha256':sha(auditpath),'layer_sha256':sha(LAYER),'baseline_raw_target_pairs':len(rawbase),'baseline_mapped_parent_identities':len(mapbase),'inside_node_rows':selected.tolist(),'baseline_exact_self_pairs':0,'outside_selected_typed_pairs':len(outside_pairs),'contact_feature_reconstruction':{'pair_count':len(rawbase),'feature_count':len(contact_features),'feature_counts_per_pair':{str(k):v for k,v in feature_counts.items()},'unresolved_pair_count':len(miss),'contact_margin_mm':CONTACT_MARGIN_MM,'surface_skin_face_rows':sorted({int(r['skin_face']) for r in contact_features}),'bone_triangles':sorted({int(r['bone_triangle']) for r in contact_features})},'solver':lp,'depth_receipt':depth_receipt,'point_depth_receipt':prmeta,'layer_linear_guard':layermeta,'edge_guard':edgemeta,'geometry_admitted':False,'native_admission':False,'accepted_physical_steps':0,'physical_transfer_steps':0,'self_guard_pair_rows':selfguard.tolist(),'backtracks':[],'elapsed_s':time.monotonic()-t}
 if not sol.success:
  (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ['status','baseline_raw_target_pairs','baseline_mapped_parent_identities','contact_feature_reconstruction','solver','elapsed_s']},indent=2));return
 d=np.asarray(sol.x[:n]).reshape(-1,3);cslack=np.asarray(sol.x[n+ne:n+ne+nc]);pslack=np.asarray(sol.x[n+ne+nc:])
 dpath=OUT/'contact-feature-direction.npz';np.savez_compressed(dpath,support_nodes=support,direction_delta_mm=d,contact_slack_mm=cslack,point_slack_mm=pslack,contact_features_json=np.frombuffer(json.dumps(contact_features,separators=(',',':')).encode(),dtype=np.uint8))
 screen=mod('contact_screen',SCREEN);exactaudit=mod('contact_audit159',AUD159);exact=exactaudit.exact_module();bankhelper=mod('contact_bank_helper_again',BANKHELPER)
 bT,bids,_,_,M2,off2,pos2,_,_,_=bankhelper.read_bank_inputs(BANKDIR,E)
 before_faces=next(r for r in aud['banks'] if int(r['body'])==43 and int(r['stable_bone_id'])==16)
 trial_records=[];selected_trial=None;rejected=None
 for alpha in ALPHAS:
  Y=X.copy();Y[support]=(X[support].astype(np.float64)+(alpha*d)/1000.).astype(np.float32)
  changed=np.flatnonzero(np.any(Y.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64)
  chf=np.flatnonzero(np.any(np.isin(F,changed),axis=1)).astype(np.int64)
  outside=np.setdiff1d(np.arange(len(X),dtype=np.int64),support)
  outside_same=bool(np.array_equal(Y[outside].view(np.uint32),X[outside].view(np.uint32)))
  anchors_same=bool(np.array_equal(Y[anchors].view(np.uint32),X[anchors].view(np.uint32)))
  shapep,shapem,areag=shape_pass(X,Y,F,chf,screen,q,ref)
  lay=screen.layer_check(X,Y,F,chf,{k:v for k,v in payload.items()} if False else np.load(LAYER,allow_pickle=False),mod('contact_transfer_gate',W/'b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py'))
  tr={'alpha':alpha,'changed_nodes':int(len(changed)),'changed_faces':int(len(chf)),'outside_bitwise_unchanged':outside_same,'anchors_bitwise_unchanged':anchors_same,'shape_pass':shapep,'shape':shapem,'area_gate':areag,'layer_gate':lay,'exact_checked':False}
  if not (outside_same and anchors_same and shapep and areag['pass'] and lay['pass']):
   tr['status']='REJECTED_SHAPE_OR_LAYER';trial_records.append(tr);continue
  tr['exact_checked']=True
  rawlocal,box=screen.audit_target(Y.astype(np.float64),F,chf,bT,bids,M2,off2,pos2,exact,exactaudit)
  expected_local={(f,t) for f,t in rawbase if f in set(map(int,chf))}
  if rawlocal!=expected_local:raise ValueError('zero-motion baseline target exact subset mismatch')
  after_raw=(rawbase-expected_local)|rawlocal
  after_map=mapped(after_raw,ip,pr)
  records,_=exact.float32_surface_records(Y,F,source_seam_groups=());selfres=exact._audit_self_pairs_touching_faces(records,set(map(int,chf)));selfpairs={(int(a),int(b)) for a,b in (selfres.get('triangle_pairs') or [])}
  newraw=after_raw-rawbase;removedraw=rawbase-after_raw;newmap=after_map-mapbase;removedmap=mapbase-after_map
  useful=bool(not newraw and not newmap and not selfpairs and len(after_map)<len(mapbase))
  tr.update(status='EXACT_USEFUL' if useful else 'REJECTED_EXACT_NO_MAPPED_REDUCTION',exact_target={'raw_before':len(rawbase),'raw_after':len(after_raw),'new_raw':sorted([list(x) for x in newraw]),'removed_raw':sorted([list(x) for x in removedraw]),'mapped_before':len(mapbase),'mapped_after':len(after_map),'new_mapped':sorted([list(x) for x in newmap]),'removed_mapped':sorted([list(x) for x in removedmap]),'local_aabb':box},exact_self={'count':len(selfpairs),'pairs':sorted([list(x) for x in selfpairs])})
  trial_records.append(tr)
  if rejected is None:
   rejected=OUT/'candidate-first-exact-rejection-private.npz';outpayload=dict(payload);outpayload['points_world_m']=Y
   with rejected.open('xb') as f:np.savez_compressed(f,**outpayload)
  if useful:selected_trial=(Y,alpha,tr,changed,chf);break
 report.update(status='EXACT_USEFUL_CANDIDATE_PENDING_ALL_BANK_AUDIT' if selected_trial else 'REJECTED_NO_EXACT_USEFUL_BACKTRACK',direction_path=str(dpath),direction_sha256=sha(dpath),target_soft_slack_sum_mm=float(cslack.sum()),point_soft_slack_sum_mm=float(pslack.sum()),backtracks=trial_records,rejected_artifact_path=str(rejected) if rejected else None,elapsed_s=time.monotonic()-t)
 if selected_trial:
  Y,alpha,tr,changed,chf=selected_trial;out=OUT/'candidate-exact-private.npz';outpayload=dict(payload);outpayload['points_world_m']=Y
  with out.open('xb') as f:np.savez_compressed(f,**outpayload)
  report.update(candidate_path=str(out),candidate_sha256=sha(out),selected_alpha=alpha,changed_nodes=changed.tolist(),changed_face_rows=chf.tolist(),selected_trial=tr)
 elif rejected:report['rejected_artifact_sha256']=sha(rejected)
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'solver':lp,'target_slack_sum_mm':report.get('target_soft_slack_sum_mm'),'point_slack_sum_mm':report.get('point_soft_slack_sum_mm'),'target_before_raw':len(rawbase),'target_before_mapped':len(mapbase),'trials':[{'alpha':x['alpha'],'status':x['status'],'shape':x['shape_pass'],'layer':x['layer_gate']['pass'],'exact':x['exact_checked'],'raw_after':(x.get('exact_target') or {}).get('raw_after'),'mapped_after':(x.get('exact_target') or {}).get('mapped_after'),'self':(x.get('exact_self') or {}).get('count')} for x in trial_records],'candidate':report.get('candidate_path'),'sha':report.get('candidate_sha256')},indent=2))
if __name__=='__main__':main()
