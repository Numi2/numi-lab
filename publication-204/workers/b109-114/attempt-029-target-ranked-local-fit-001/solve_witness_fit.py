#!/usr/bin/env python3
from pathlib import Path
from fractions import Fraction
from collections import defaultdict,deque
import hashlib,importlib.util,json,sys,time
import numpy as np
from scipy import sparse

ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638'
W=S/'per-bank-workers-20261027';D=W/'b109-114/attempt-029-target-ranked-local-fit-001'
BASE=W/'combined-d071-b109-b61-001/candidate.npz'
AUD159=W/'combined-d071-b109-b61-001/audit-159/report.json'
AUD130=W/'combined-d071-b109-b61-001/audit-130/summary.json'
LAYER=W/'combined-d071-b109-b61-001/skin-fat-layers-541.npz'
LAYER_OWNER=W/'b65-99/root-layer-guard-002/fit.py'
QPATH=ROOT/'D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
SOURCE=ROOT/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
BANK_E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=BANK_E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT=ROOT/'audit_skin_bone_surfaces_159.py'
OSQP_DEPS=Path('/Users/numi/numi-human-fds132-m6-1350/attempt-311-posed-area-sqp-003/osqp-deps')
BASE_SHA='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
LAYER_SHA='eb41bf0d1c11f1ee0a94672058063c51350e8e0dca1483a6ae7568cb03a78b57'
BODY=109;BONE=114;SKIN_FACE=34547;BONE_ROWS=[111,112,116]
MARGIN_MM=0.02;RINGS=4;TRUST_MM=.15;SOURCE_CAP_MM=12.0;EDGE_GRAD=.1
BACKTRACKS=[1.0,.5,.25,.125,.0625]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(name,p):
 spec=importlib.util.spec_from_file_location(name,p);x=importlib.util.module_from_spec(spec);sys.modules[name]=x;spec.loader.exec_module(x);return x
def tri_metrics(points,faces):
 T=np.asarray(points,dtype=np.float64)[np.asarray(faces,dtype=np.int64)]
 e1=T[:,1]-T[:,0];e2=T[:,2]-T[:,0];e3=T[:,2]-T[:,1]
 n=np.cross(e1,e2);twice=np.linalg.norm(n,axis=1)
 lengths=np.stack((np.linalg.norm(e1,axis=1),np.linalg.norm(e2,axis=1),np.linalg.norm(e3,axis=1)),axis=1)
 ang=[]
 for i in range(3):
  u=T[:,(i+1)%3]-T[:,i];v=T[:,(i+2)%3]-T[:,i]
  c=np.einsum('ij,ij->i',u,v)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-300)
  ang.append(np.degrees(np.arccos(np.clip(c,-1,1))))
 return {'area':twice*.5,'angle':np.min(np.stack(ang,axis=1),axis=1),'aspect':np.max(lengths,axis=1)**2/(2*np.sqrt(3)*np.maximum(twice,1e-300)),'normal':n}
def strict_shape(base,trial,faces,affected):
 a=tri_metrics(base,faces[affected]);b=tri_metrics(trial,faces[affected])
 ar=b['area']/np.maximum(a['area'],1e-300);an=b['angle']/np.maximum(a['angle'],1e-300);asp=b['aspect']/np.maximum(a['aspect'],1e-300)
 dot=np.einsum('ij,ij->i',a['normal'],b['normal'])/np.maximum(np.linalg.norm(a['normal'],axis=1)*np.linalg.norm(b['normal'],axis=1),1e-300)
 bad=(ar<.95)|(an<.95)|(asp>1.05)|(dot<np.cos(np.deg2rad(18.0)))
 return {'pass':bool(not bad.any()),'affected_face_count':int(len(affected)),'violating_face_count':int(bad.sum()),'min_area_ratio':float(ar.min(initial=np.inf)),'min_min_angle_ratio':float(an.min(initial=np.inf)),'max_aspect_ratio':float(asp.max(initial=-np.inf)),'min_normal_dot':float(dot.min(initial=np.inf)),'max_normal_turn_deg':float(np.degrees(np.arccos(np.clip(dot,-1,1))).max(initial=0.0)),'violating_face_rows':affected[bad].astype(int).tolist(),'limits':{'area_ratio':.95,'min_angle_ratio':.95,'aspect_ratio':1.05,'normal_dot':float(np.cos(np.deg2rad(18.0)))}}
def exact_bary(point,tri):
 p=np.asarray(point,dtype=np.float64);t=np.asarray(tri,dtype=np.float64);v0=t[1]-t[0];v1=t[2]-t[0];v2=p-t[0]
 d00=np.dot(v0,v0);d01=np.dot(v0,v1);d11=np.dot(v1,v1);d20=np.dot(v2,v0);d21=np.dot(v2,v1);den=d00*d11-d01*d01
 if abs(den)<1e-24:raise ValueError('degenerate skin triangle for exact witness barycentrics')
 v=(d11*d20-d01*d21)/den;w=(d00*d21-d01*d20)/den;return np.array([1-v-w,v,w],np.float64)
def main():
 start=time.monotonic()
 for p,expected in ((BASE,BASE_SHA),(LAYER,LAYER_SHA)):
  if sha(p)!=expected:raise ValueError('pinned input mismatch '+str(p))
 adoc=json.loads(AUD159.read_text());brec=next(b for b in adoc['banks'] if (int(b['body']),int(b['stable_bone_id']))==(BODY,BONE))
 exact_pairs={(int(a),int(b)) for a,b in (brec.get('result') or {}).get('triangle_pairs',[])}
 if (SKIN_FACE,BONE_ROWS[0]) not in exact_pairs:raise ValueError('target ranked pair no longer exists')
 with np.load(BASE,allow_pickle=False) as z:payload={k:z[k].copy() for k in z.files}
 x=np.asarray(payload['points_world_m'],dtype=np.float32);F=np.asarray(payload['source_O_faces'],dtype=np.int64)
 if len(x)!=21212 or len(F)!=42379:raise ValueError('baseline shape mismatch')
 helper=mod('bank_helper_029',HELPER);audit=mod('skin_bone_audit_029',AUDIT);exact=audit.exact_module()
 T,ids,gids,bodies,M,offset,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,BANK_E)
 mask=(ids[:,0]==BODY)&(ids[:,1]==BONE);BT=np.asarray(T[mask],np.float64)*1000.;stable=np.asarray(ids[mask,3],np.int64)
 if len(BT)!=392 or max(BONE_ROWS)>=len(BT):raise ValueError('unexpected target bone row domain')
 local=((x.astype(np.float64)-pos[BODY])@M[BODY].T-offset[BODY])*1000.
 # Verify the report's second identity field is the bank-local triangle row.
 # Rebuild the saved baseline target pair set using explicit row IDs.
 baseline_local=audit.intersect_bank(local,F,BT,np.arange(len(BT),dtype=np.int64),exact)
 baseline_rows={(int(a),int(b)) for a,b in (baseline_local.get('result') or {}).get('triangle_pairs',[])}
 if baseline_rows!=exact_pairs:raise ValueError(f'zero-motion pair control mismatch report={len(exact_pairs)} local={len(baseline_rows)}')
 # Orient the exact Float32 bone shell from canonical shared vertex keys.
 def key(v):return tuple(np.asarray(v,dtype=np.float64).view(np.uint64).tolist())
 verts={};ti=np.empty((len(BT),3),np.int64)
 for i,tr in enumerate(BT):
  for j,v in enumerate(tr):ti[i,j]=verts.setdefault(key(v),len(verts))
 uses=defaultdict(list)
 for i,t in enumerate(ti):
  for j in range(3):
   a,b=int(t[j]),int(t[(j+1)%3]);k=tuple(sorted((a,b)));sgn=1 if (a,b)==k else -1;uses[k].append((i,sgn))
 if any(len(v)!=2 for v in uses.values()):raise ValueError('target bone triangle rows are not a closed two-manifold')
 adj=[[] for _ in BT]
 for items in uses.values():
  (i,si),(j,sj)=items;rel=-si*sj;adj[i].append((j,rel));adj[j].append((i,rel))
 orient=np.zeros(len(BT),np.int8);components=[];conflicts=0
 from collections import deque
 for root in range(len(BT)):
  if orient[root]:continue
  orient[root]=1;q0=deque([root]);comp=[]
  while q0:
   i=q0.popleft();comp.append(i)
   for j,rel in adj[i]:
    want=orient[i]*rel
    if orient[j]==0:orient[j]=want;q0.append(j)
    elif orient[j]!=want:conflicts+=1
  components.append(comp)
 volumes=[]
 for comp in components:
  center=BT[np.asarray(comp)].mean(axis=(0,1));vol=0.
  for i in comp:
   tr=BT[i]*int(orient[i])-center;vol+=float(np.dot(tr[0],np.cross(tr[1],tr[2]))/6.)
  if vol<0:orient[np.asarray(comp,np.int64)]*=-1;vol=-vol
  volumes.append(vol)
 if conflicts or len(components)!=1:raise ValueError('target bone shell orientation is inconsistent')
 # Reconstruct exact intersection endpoints and barycentric skin witnesses for exactly the three selected pairs.
 witness_rows=[];witness_receipts=[]
 for brow in BONE_ROWS:
  skin_local=local[F[SKIN_FACE]];bone_local=BT[brow]
  rat_skin=[tuple(Fraction.from_float(float(v)) for v in p) for p in skin_local]
  rat_bone=[tuple(Fraction.from_float(float(v)) for v in p) for p in bone_local]
  exact_points=exact.triangle_intersection_points(rat_skin,rat_bone)
  uniq=[]
  for p in exact_points:
   if p not in uniq:uniq.append(p)
  if not uniq:raise ValueError(f'exact intersection has no witness for {(SKIN_FACE,brow)}')
  nlocal=np.cross(bone_local[1]-bone_local[0],bone_local[2]-bone_local[0])*int(orient[brow]);nlocal/=np.linalg.norm(nlocal)
  nworld=M[BODY].T@nlocal;nworld/=np.linalg.norm(nworld)
  points=[]
  for pi,p in enumerate(uniq):
   point=np.asarray([float(v) for v in p],np.float64);bary=exact_bary(point,skin_local)
   if abs(float(bary.sum())-1)>1e-10 or bary.min()<-1e-8 or bary.max()>1+1e-8:raise ValueError('intersection endpoint barycentric outside skin face')
   row={'nodes':F[SKIN_FACE].astype(int).tolist(),'coefficients':[(float(w)*nworld).tolist() for w in bary],'lower':MARGIN_MM,'label':f'exact_intersection:{SKIN_FACE}:{brow}:endpoint{pi}'}
   witness_rows.append(row)
   points.append({'exact_local_mm':[[int(v.numerator),int(v.denominator)] for v in p],'skin_barycentric':bary.tolist(),'outward_normal_local':nlocal.tolist(),'outward_normal_world':nworld.tolist()})
  witness_receipts.append({'skin_face_row':SKIN_FACE,'bone_triangle_row':brow,'bone_triangle_stable_id_in_039':int(stable[brow]),'exact_intersection_vertex_count':len(uniq),'endpoints':points})
 # Connected support around the source skin face, fixed outside ring.
 edges=np.unique(np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0),axis=1),axis=0)
 node_adj=[set() for _ in range(len(x))]
 for u,v in edges:node_adj[int(u)].add(int(v));node_adj[int(v)].add(int(u))
 target_nodes=set(map(int,F[SKIN_FACE]));dist={n:0 for n in target_nodes};q0=deque(target_nodes)
 while q0:
  u=q0.popleft()
  if dist[u]>=RINGS:continue
  for v in node_adj[u]:
   if v not in dist:dist[v]=dist[u]+1;q0.append(v)
 support=np.asarray(sorted(dist),np.int64);support_set=set(map(int,support));locidx={int(v):i for i,v in enumerate(support)}
 affected=np.flatnonzero(np.any(np.isin(F,support),axis=1))
 anchors=np.asarray(payload['typed_coarse_anchor_output_rows_018'],np.int64)
 bm=x.astype(np.float64)*1000.;ref=np.asarray(payload['area_reference_037_points_world_m_f64'],np.float64)*1000.
 q=mod('witness_q_029',QPATH);area=mod('witness_area_029',q.DEFAULT_AREA_OWNER)
 q.verify_area_linearization_layout(area)
 with np.load(BASE,allow_pickle=False) as a:srcmm=q.compose_source_reference(a,SOURCE)[0]*1000.
 lo,hi,bounds=q.source_safe_trust_bounds(bm,srcmm,support,anchors,TRUST_MM,SOURCE_CAP_MM)
 normals,current_area,reference_area,required=q.fixed_reference_area_requirements(bm,ref,F)
 # Per-face area floors relative to this exact baseline.
 G,grhs,_,_=area.area_linearization(bm,F,affected,normals[affected],current_area[affected],minimum_ratio=.95)
 # Per-face first-order normal-turn constraints: bound both tangent components of
 # the oriented area-normal variation to 10 degrees at each affected face.
 normal_rows=[];angle=10*np.pi/180.
 for fi in affected:
  a,b,c=map(int,F[int(fi)]);p0,p1,p2=bm[[a,b,c]];e1=p1-p0;e2=p2-p0;n=np.cross(e1,e2);area2=float(np.linalg.norm(n))
  if area2<=0:raise ValueError(f'degenerate baseline face {fi}')
  nhat=n/area2;t1=e1/np.linalg.norm(e1);t2=np.cross(nhat,t1);t2/=np.linalg.norm(t2)
  for t in (t1,t2):
   c0=-np.cross(e2,t)-np.cross(t,e1);c1=np.cross(e2,t);c2=np.cross(t,e1)
   cap=np.tan(angle)*area2/np.sqrt(2.)
   nodes=[a,b,c];coeffs=[c0,c1,c2]
   kept=[(node,coef) for node,coef in zip(nodes,coeffs) if node in locidx]
   if kept:
    normal_rows.append({'nodes':[node for node,_ in kept],'coefficients':[coef.tolist() for _,coef in kept],'lower':-cap,'label':f'normal-turn+:{int(fi)}'})
    normal_rows.append({'nodes':[node for node,_ in kept],'coefficients':[(-coef).tolist() for _,coef in kept],'lower':-cap,'label':f'normal-turn-:{int(fi)}'})
 # Add the existing layer-prism linear guard using the pinned 541 baseline layer.
 layer_owner=mod('layer_owner_029',LAYER_OWNER)
 LG,lb,layer_receipt=layer_owner.make_layer_guard(LAYER,F,affected,support)
 # Check support against actual nonfixed displacement coordinates. A zero
 # movable row is discarded only when its zero-motion slack is nonnegative.
 movable=np.asarray(hi,dtype=np.float64)-np.asarray(lo,dtype=np.float64)>1e-15
 zero_row_receipt={'schema':'numi.human.b109-114.zero-row-preflight.v1','tolerance':1e-14,'pair_rows':[],'area_rows':[],'status':'PASS'}
 def filter_pair_rows(rows,kind):
  kept=[]
  for row in rows:
   vec=np.zeros(3*len(support),np.float64)
   for node,coef in zip(row['nodes'],row['coefficients']):
    li=locidx.get(int(node))
    if li is None:raise ValueError(f'{kind} row outside support: {row.get("label")}')
    vec[3*li:3*li+3]+=np.asarray(coef,dtype=np.float64)
   norm=float(np.linalg.norm(vec*movable));slack=-float(row['lower'])
   if not np.isfinite(norm) or not np.isfinite(slack):raise ValueError(f'nonfinite {kind} row {row.get("label")}')
   if norm<=1e-14:
    rec={'label':row.get('label'),'movable_coefficient_norm':norm,'lower':float(row['lower']),'zero_motion_slack':slack}
    zero_row_receipt['pair_rows'].append(rec)
    if slack < -1e-12:
     zero_row_receipt['status']='INFEASIBLE_ZERO_MOVABLE_SUPPORT';zero_row_receipt['unsatisfied']=rec
     (D/'zero-row-preflight.json').write_text(json.dumps(zero_row_receipt,indent=2,sort_keys=True)+'\n')
     raise ValueError(f'{kind} row has no movable support and is unsatisfied: {rec}')
   else:kept.append(row)
  return kept
 witness_rows=filter_pair_rows(witness_rows,'exact-witness')
 normal_rows=filter_pair_rows(normal_rows,'normal-shape')
 local_cols=q.interleaved_coordinate_columns(support)
 local_area=G[:,local_cols].tocsr()
 movable_area=local_area.multiply(movable.reshape(1,-1))
 area_norms=np.sqrt(np.asarray(movable_area.multiply(movable_area).sum(axis=1)).ravel())
 keep_area=[]
 for ii,norm in enumerate(area_norms):
  if not np.isfinite(norm):raise ValueError(f'nonfinite area row {int(affected[ii])}')
  if norm<=1e-14:
   slack=-float(grhs[ii]);rec={'face_row':int(affected[ii]),'movable_coefficient_norm':float(norm),'lower':float(grhs[ii]),'zero_motion_slack':slack}
   zero_row_receipt['area_rows'].append(rec)
   if slack < -1e-12:
    zero_row_receipt['status']='INFEASIBLE_ZERO_MOVABLE_SUPPORT';zero_row_receipt['unsatisfied']=rec
    (D/'zero-row-preflight.json').write_text(json.dumps(zero_row_receipt,indent=2,sort_keys=True)+'\n')
    raise ValueError(f'area row has no movable support and is unsatisfied: {rec}')
  else:keep_area.append(ii)
 if len(keep_area)!=G.shape[0]:G=G[keep_area];grhs=grhs[keep_area]
 (D/'zero-row-preflight.json').write_text(json.dumps(zero_row_receipt,indent=2,sort_keys=True)+'\n')
 A,b=q._owner_matrix(witness_rows+normal_rows,G,grhs,support)
 EG,eb,edge_receipt=q.build_edge_gradient_constraints(bm,F,support,EDGE_GRAD)
 A=sparse.vstack((A,LG,EG),format='csr');b=np.r_[b,lb,eb]
 # Graph-strain quadratic, inverse-edge weighting with bounded edge gradients.
 eidx=edges[np.isin(edges,support).any(axis=1)]
 rr=[];cc=[];vv=[];nr=0
 for u,v in eidx:
  iu=locidx.get(int(u));iv=locidx.get(int(v));weight=1./max(float(np.linalg.norm(bm[u]-bm[v])),.05)
  for k in range(3):
   if iu is not None:rr.append(nr);cc.append(3*iu+k);vv.append(weight)
   if iv is not None:rr.append(nr);cc.append(3*iv+k);vv.append(-weight)
   nr+=1
 E0=sparse.coo_matrix((vv,(rr,cc)),shape=(nr,3*len(support))).tocsr()
 n=3*len(support);H=5.0*(E0.T@E0)+.02*sparse.eye(n,format='csr')
 sys.path.insert(0,str(OSQP_DEPS));import osqp
 I=sparse.eye(n,format='csr');K=sparse.vstack((A,I),format='csc')
 lower=np.r_[b,lo];upper=np.r_[np.full(A.shape[0],np.inf),hi]
 opt=osqp.OSQP();opt.setup(P=H.tocsc(),q=np.zeros(n),A=K,l=lower,u=upper,verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=150000,polishing=True,adaptive_rho=True,scaling=30,scaled_termination=True)
 sol=opt.solve();solved=int(sol.info.status_val)==1
 report={'schema':'numi.human.b109-114.exact-three-witness-connected-fit.v1','status':'QP_FAILED','baseline_sha256':BASE_SHA,'baseline_audit159_sha256':sha(AUD159),'baseline_layer_sha256':sha(LAYER),'target_bank':[BODY,BONE],'target_skin_face_row':SKIN_FACE,'target_bone_triangle_rows':BONE_ROWS,'target_exact_witnesses':witness_receipts,'target_row_count':len(witness_rows),'support_rings':RINGS,'support_nodes':len(support),'affected_face_count':len(affected),'fixed_anchor_count':int(np.count_nonzero(np.isin(support,anchors))),'trust_mm':TRUST_MM,'source_cap_mm':SOURCE_CAP_MM,'source_bound_receipt':bounds,'edge_gradient_cap':EDGE_GRAD,'edge_gradient_receipt':edge_receipt,'area_guard_rows':int(G.shape[0]),'normal_turn_rows':len(normal_rows),'normal_turn_bound_degrees':10.0,'layer_guard_receipt':layer_receipt,'solver':{'status':str(sol.info.status),'status_val':int(sol.info.status_val),'iterations':int(sol.info.iter),'primal_residual':float(sol.info.prim_res),'dual_residual':float(sol.info.dual_res),'success':bool(solved)},'backtrack_trials':[],'geometry_admitted':False,'physical_steps':0,'elapsed_solve_s':time.monotonic()-start}
 if solved:
  d=np.asarray(sol.x,dtype=np.float64).reshape(-1,3)
  for alpha in BACKTRACKS:
   y=x.copy();y[support]=(x[support].astype(np.float64)+alpha*d/1000.).astype(np.float32)
   changed=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1));disp=np.linalg.norm((y.astype(np.float64)-x.astype(np.float64))*1000.,axis=1)
   shape=strict_shape(bm,y.astype(np.float64)*1000.,F,affected)
   progress=[]
   for wr in witness_rows:
    val=0.
    for node,coeff in zip(wr['nodes'],wr['coefficients']):
     val+=float(np.dot(np.asarray(coeff),((y[int(node)].astype(np.float64)-x[int(node)].astype(np.float64))*1000.)))
    progress.append(val)
   rec={'alpha':alpha,'shape':shape,'changed_nodes':int(len(changed)),'max_step_mm':float(disp.max(initial=0.0)),'min_target_witness_progress_mm':float(min(progress,default=0.0)),'target_witness_progress_mm':progress,'candidate_path':None,'candidate_sha256':None}
   if shape['pass'] and min(progress,default=0.0)>0.0005:
    cp=D/f'candidate-alpha-{str(alpha).replace(".","p")}.npz';payload['points_world_m']=y
    n0=len(x);m0=len(F)
    payload['remesh806_old_to_new_node_rows']=np.arange(n0,dtype=np.int64);payload['remesh806_new_to_old_node_rows']=np.arange(n0,dtype=np.int64)
    payload['remesh806_baseline_sha256']=np.asarray([BASE_SHA],dtype='<U64')
    payload['remesh806_face_parent_indptr']=np.arange(m0+1,dtype=np.int64);payload['remesh806_face_parent_face_rows']=np.arange(m0,dtype=np.int64)
    with cp.open('xb') as f:np.savez_compressed(f,**payload)
    rec['candidate_path']=str(cp);rec['candidate_sha256']=sha(cp)
   report['backtrack_trials'].append(rec)
  passing=[r for r in report['backtrack_trials'] if r['candidate_path']]
  if passing:
   report['status']='SHAPE_PASS_CANDIDATES_EMITTED';report['candidate_candidates']=passing
  else:report['status']='NO_BACKTRACK_PASSED_SHAPE_AND_WITNESS_PROGRESS'
 rep=D/'fit-report.json';rep.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'candidate_sha256':BASE_SHA,'target_pair_rows':BONE_ROWS,'exact_witnesses':witness_receipts,'support_nodes':len(support),'affected_faces':len(affected),'area_rows':int(G.shape[0]),'normal_rows':len(normal_rows),'layer_rows':int(LG.shape[0]),'solver':report['solver'],'backtracks':[{'alpha':r['alpha'],'shape_pass':r['shape']['pass'],'changed_nodes':r['changed_nodes'],'max_step_mm':r['max_step_mm'],'min_witness_progress_mm':r['min_target_witness_progress_mm'],'candidate_path':r['candidate_path'],'candidate_sha256':r['candidate_sha256'],'shape':{k:r['shape'][k] for k in ('min_area_ratio','min_min_angle_ratio','max_aspect_ratio','min_normal_dot','max_normal_turn_deg','violating_face_count')} if 'violating_face_count' in r['shape'] else {k:r['shape'][k] for k in ('min_area_ratio','min_min_angle_ratio','max_aspect_ratio','min_normal_dot','max_normal_turn_deg')}} for r in report['backtrack_trials']],'report':str(rep),'report_sha256':sha(rep)},indent=2,allow_nan=False))
if __name__=='__main__':main()
