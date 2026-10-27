#!/usr/bin/env python3
"""One bounded 109/114 solve with 32 exact self and 15 bone separation witnesses."""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys, time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
B=S/'partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001'
OUT=Path(__file__).parent
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
DPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/point-depth-barriers-357/point_depth_barriers_357.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANK=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
SOURCE=Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz')
SELF_PAIR_FILE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-016-self-barrier-union-001/witness-pairs-union.npy')
SELF_PAIR_SHA='7f461cb82f9d972cfba9dc3bcc46867cb9ff2f38a3533ddf298efa8627887815'
BONE_ANALYSIS=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-017-bone-witness-constraints-001/bone-separation-analysis.json')
BONE_ROWS_FILE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b109-114/attempt-017-bone-witness-constraints-001/bone-separation-rows.json')
BONE_ANALYSIS_SHA='353ab76f96aee0e5e84e0d338afbcdfb18d448c5d52b0a1651be725ff3e69388'
BONE_ROWS_SHA='e89ba266de21a653108c84a6bd7a0b4c7397ccb38a0ae013d43e7a15a2c7a7d5'
BASE130=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/partial-midpoint-continuation-root-001/combined-43-57-65smooth-109-124-001/audit-130/summary.json')
BASE159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-182-stage-001/combined-653/audit-159/report.json')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p); x=importlib.util.module_from_spec(sp);sys.modules[name]=x;sp.loader.exec_module(x);return x

def tri_metrics(T):
 T=np.asarray(T,np.float64);a=T[:,1]-T[:,0];b=T[:,2]-T[:,0];c=T[:,2]-T[:,1]
 normal=np.cross(a,b);twice=np.linalg.norm(normal,axis=1)
 lengths=np.stack((np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)),axis=1)
 angles=[]
 for i in range(3):
  u=T[:,(i+1)%3]-T[:,i];v=T[:,(i+2)%3]-T[:,i]
  dot=np.einsum('ij,ij->i',u,v)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-300)
  angles.append(np.degrees(np.arccos(np.clip(dot,-1,1))))
 return {'area':twice*.5,'min_angle':np.min(np.stack(angles,axis=1),axis=1),'aspect':np.max(lengths,axis=1)**2/(2*np.sqrt(3)*np.maximum(twice,1e-300)),'normal':normal}

def strict_shape_report(before,after,faces):
 changed=np.flatnonzero(np.any(np.asarray(before,dtype=np.float32).view(np.uint32)!=np.asarray(after,dtype=np.float32).view(np.uint32),axis=1))
 affected=np.flatnonzero(np.any(np.isin(faces,changed),axis=1))
 A=tri_metrics(before[faces[affected]].astype(np.float64));B=tri_metrics(after[faces[affected]].astype(np.float64))
 ar=B['area']/np.maximum(A['area'],1e-300);ang=B['min_angle']/np.maximum(A['min_angle'],1e-300);asp=B['aspect']/np.maximum(A['aspect'],1e-300)
 dot=np.einsum('ij,ij->i',A['normal'],B['normal'])/np.maximum(np.linalg.norm(A['normal'],axis=1)*np.linalg.norm(B['normal'],axis=1),1e-300)
 flips=np.einsum('ij,ij->i',A['normal'],B['normal'])<=0
 bad=(ar<.95)|(ang<.95)|(asp>1.05)|flips
 return {'pass':bool(not bad.any()),'affected_face_count':int(len(affected)),'changed_node_count':int(len(changed)),
  'violating_face_count':int(bad.sum()),'minimum_area_ratio':float(ar.min()),'minimum_angle_ratio':float(ang.min()),
  'maximum_aspect_ratio':float(asp.max()),'minimum_normal_dot':float(dot.min()),
  'maximum_normal_rotation_degrees':float(np.degrees(np.arccos(np.clip(dot,-1,1))).max()),
  'thresholds':{'area_ratio_floor':.95,'minimum_angle_ratio_floor':.95,'aspect_ratio_ceiling':1.05,'positive_orientation':True},
  'affected_face_rows':affected.astype(int).tolist()}

def main():
 t=time.monotonic(); base=B/'candidate.npz'; summary=B/'audit-130/summary.json'; query=B/'audit-130/query'
 expected='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
 if sha(base)!=expected:raise ValueError('baseline pin mismatch')
 with np.load(base,allow_pickle=False) as z: payload={k:z[k].copy() for k in z.files}
 x=payload['points_world_m']; faces=payload['source_O_faces'].astype(np.int64); ref=payload['area_reference_037_points_world_m_f64']*1000.; anchors=payload['typed_coarse_anchor_output_rows_018'].astype(np.int64)
 doc=json.loads(summary.read_text()); assert doc['candidate_sha256']==expected and doc['complete_surface_audit'] and doc['exact_PLC_surface_pairs']==0
 audit159=B/'audit-159/report.json'; auditdoc=json.loads(audit159.read_text())
 assert auditdoc.get('complete') and auditdoc.get('candidate_sha256')==expected and auditdoc.get('bank_count')==185
 witness=[r for r in doc['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id']))==(109,114)]
 selected=np.array(sorted({int(r['query_row']) for r in witness}),dtype=np.int64)
 if not len(selected):raise ValueError('no target interior witnesses')
 edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
 adj=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(x),len(x)))
 support=set(map(int,selected)); frontier=set(support)
 for _ in range(6):
  frontier=set(map(int,adj[list(frontier)].indices))-support; support.update(frontier)
 moving_support=set(support)
 if sha(SELF_PAIR_FILE)!=SELF_PAIR_SHA:raise ValueError('exact self-witness pin mismatch')
 self_pairs=np.asarray(np.load(SELF_PAIR_FILE,allow_pickle=False),dtype=np.int64).reshape(-1,2)
 if self_pairs.shape!=(32,2):raise ValueError(f'expected the 32-pair self witness union, got {self_pairs.shape}')
 witness_nodes=set(map(int,faces[self_pairs.ravel()].ravel()))
 support.update(witness_nodes);fixed_extra=sorted(support-moving_support)
 support=np.array(sorted(support),dtype=np.int64); loc={int(v):i for i,v in enumerate(support)}
 affected=np.flatnonzero(np.any(np.isin(faces,support),axis=1)); q=mod('depth_cont_q',QPATH); depth=mod('depth_cont_d',DPATH); area=mod('depth_cont_area',q.DEFAULT_AREA_OWNER)
 q.verify_area_linearization_layout(area)
 with np.load(base,allow_pickle=False) as source_archive: srcmm=q.compose_source_reference(source_archive,SOURCE)[0]*1000.
 bm=x.astype(np.float64)*1000.
 spec={'points_csv':query/'points.csv','pairs_csv':query/'pairs.csv','nodes_csv':query/'nodes.csv','query_receipt_json':query/'query-receipt.json','kind':'inside','require_points_match':True}
 sup=set(map(int,support)); outs=[]
 with (query/'pairs.csv').open(newline='') as f:
  for row in csv.DictReader(f):
   if row['classification']=='outside' and int(row['query_row']) in sup and float(row['nearest_distance_m'])<=.0011:
    outs.append((int(row['query_row']),int(row['bank_body']),int(row['bank_stable_id']),int(row['nearest_triangle'])))
 ns,norms,low,ids,records,receipt=depth.build_point_depth_rows(x,[spec,{**spec,'kind':'outside','selected_typed_pairs':outs}],BANK/'exact-emitted.bodies.csv')
 prs,prmeta=q.build_scoped_output_node_direction_rows(len(x),support,anchors,ns,norms,low,ids)
 normals,current_area,reference_area,req=q.fixed_reference_area_requirements(bm,ref,faces)
 G,rhs,_,_=area.area_linearization(bm,faces,affected,normals[affected],req[affected],minimum_ratio=.1)
 Gsource,rhssource,_,_=area.area_linearization(bm,faces,affected,normals[affected],current_area[affected],minimum_ratio=.95)
 G=sparse.vstack((G,Gsource),format='csr');rhs=np.r_[rhs,rhssource]
 sep=mod('depth_self_sep',q.DEFAULT_PAIR_OWNER)
 try:
  self_rows,self_records=sep.build_pair_rows(bm,faces,self_pairs,margin_fraction=.25)
 except Exception as exc:
  report={'status':'SELF_BARRIER_BASELINE_INFEASIBLE','baseline_sha256':expected,'bank':[109,114],
   'self_witness_pairs':self_pairs.tolist(),'self_witness_file_sha256':SELF_PAIR_SHA,
   'helper_path':str(q.DEFAULT_PAIR_OWNER),'helper_sha256':sha(q.DEFAULT_PAIR_OWNER),
   'error_type':type(exc).__name__,'error':str(exc),'geometry_admitted':False,'physical_steps':0}
  (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
  print(json.dumps(report,indent=2),flush=True);return
 if sha(BONE_ANALYSIS)!=BONE_ANALYSIS_SHA or sha(BONE_ROWS_FILE)!=BONE_ROWS_SHA:raise ValueError('bone separation witness pin mismatch')
 bone_doc=json.loads(BONE_ANALYSIS.read_text())
 rows_doc=json.loads(BONE_ROWS_FILE.read_text())
 if bone_doc.get('status')!='ROWS_READY' or bone_doc.get('target_bank')!=[109,114] or bone_doc.get('baseline_candidate_sha256')!=expected:raise ValueError('bone witness analysis does not match pinned 653 target')
 bone_rows=[]
 for r in rows_doc.get('rows',[]):
  row={'nodes':np.asarray(r['nodes'],dtype=np.int64),'coefficients':np.asarray(r['coefficients'],dtype=np.float64),'lower':float(r['lower']),'label':str(r['label'])}
  if not set(map(int,row['nodes'])).issubset(set(map(int,support))):raise ValueError('bone witness row node is outside QP support')
  bone_rows.append(row)
 if len(bone_rows)!=45:raise ValueError(f'expected 45 target bone separation rows, got {len(bone_rows)}')
 A,b=q._owner_matrix(prs+self_rows+bone_rows,G,rhs,support); EG,eb,_=q.build_edge_gradient_constraints(bm,faces,support,.1); A=sparse.vstack((A,EG),format='csr');b=np.r_[b,eb]
 lo,hi,boundsmeta=q.source_safe_trust_bounds(bm,srcmm,support,anchors,.5,12.)
 for vertex in fixed_extra:
  j=3*loc[int(vertex)];lo[j:j+3]=0.;hi[j:j+3]=0.
 inside_kind=np.array([r['kind']=='inside_nonregression' for r in records],dtype=bool)
 target=np.flatnonzero((ids[:,1]==109)&(ids[:,2]==114)&inside_kind&np.isin(ns,support)&(~np.isin(ns,anchors)))
 if len(target)==0:raise ValueError('all target witnesses fixed')
 n=3*len(support);m=len(target); rr=[];cc=[];vv=[]
 for i,k in enumerate(target):
  for j in range(3):rr.append(i);cc.append(3*loc[int(ns[k])]+j);vv.append(norms[k,j])
 T=sparse.coo_matrix((vv,(rr,cc)),shape=(m,n)).tocsr()
 er=[];ec=[];ev=[]; nr=0
 for u,v in edges:
  iu=loc.get(int(u));iv=loc.get(int(v))
  if iu is None and iv is None:continue
  weight=1./max(float(np.linalg.norm(bm[u]-bm[v])),.05)
  for k in range(3):
   if iu is not None:er.append(nr);ec.append(3*iu+k);ev.append(weight)
   if iv is not None:er.append(nr);ec.append(3*iv+k);ev.append(-weight)
   nr+=1
 E0=sparse.coo_matrix((ev,(er,ec)),shape=(nr,n)).tocsr(); I=sparse.eye(n,format='csr'); J=sparse.eye(nr,format='csr'); U=sparse.eye(m,format='csr')
 Z=lambda a,c:sparse.csr_matrix((a,c))
 blocks=[sparse.hstack((-A,Z(A.shape[0],n),Z(A.shape[0],nr),Z(A.shape[0],m))),
         sparse.hstack((I,-I,Z(n,nr),Z(n,m))), sparse.hstack((-I,-I,Z(n,nr),Z(n,m))),
         sparse.hstack((E0,Z(nr,n),-J,Z(nr,m))), sparse.hstack((-E0,Z(nr,n),-J,Z(nr,m))),
         sparse.hstack((-T,Z(m,n),Z(m,nr),-U))]
 Am=sparse.vstack(blocks,format='csr'); rhsall=np.r_[-b,np.zeros(2*n+2*nr),-np.full(m,.15)]
 c=np.r_[np.zeros(n),np.full(n,.02),np.full(nr,5.),np.full(m,1000.)]
 bounds=list(zip(lo,hi))+[(0,None)]*(n+nr+m)
 # First solve a scaled LP over the same pinned hard rows and bounds. The
 # auxiliary t maximizes the minimum projected progress over all 24 target rows.
 from types import SimpleNamespace
 tcol=sparse.csr_matrix(np.ones((m,1),dtype=np.float64))
 lpA=sparse.vstack((sparse.hstack((-A,sparse.csr_matrix((A.shape[0],1)))),
                    sparse.hstack((-T,tcol))),format='csr')
 lpb=np.r_[-b,np.zeros(m)]
 lp=linprog(np.r_[np.zeros(n),-1.0],A_ub=lpA,b_ub=lpb,
            bounds=list(zip(lo.tolist(),hi.tolist()))+[(None,None)],
            method='highs',options={'primal_feasibility_tolerance':1e-7,'dual_feasibility_tolerance':1e-7})
 lp_report={'success':bool(lp.success),'status':int(lp.status),'message':str(lp.message),
  'maximum_common_target_progress_mm':None,'minimum_hard_row_slack_normalized':None,
  'minimum_target_row_slack_mm':None,'maximum_bound_violation_mm':None,
  'minimum_bone_row_slack_mm':None,'target_progress_mm':[],
  'variables':int(n),'hard_rows':int(A.shape[0]),'target_rows':int(m),
  'hard_row_scale':'all owner pair/area rows normalized by q._owner_matrix; edge rows normalized by their owner builder',
  'progress_objective':'maximize t subject to every target projection T_i d >= t'}
 if lp.success:
  lpd=np.asarray(lp.x[:n],dtype=np.float64); tstar=float(lp.x[-1])
  hard_slack=np.asarray(A@lpd-b).ravel()
  target_prog=np.asarray(T@lpd).ravel()
  bound_violation=max(0.0,float(np.max(lo-lpd,initial=0.0)),float(np.max(lpd-hi,initial=0.0)))
  loc_lp={int(row):i for i,row in enumerate(support)}
  bone_slack=[]
  for row in bone_rows:
   val=0.0
   for node,coeff in zip(row['nodes'],row['coefficients']):
    val+=float(np.asarray(coeff)@lpd[3*loc_lp[int(node)]:3*loc_lp[int(node)]+3])
   bone_slack.append(val-float(row['lower']))
  lp_report.update({'maximum_common_target_progress_mm':tstar,
   'minimum_hard_row_slack_normalized':float(hard_slack.min(initial=np.inf)),
   'minimum_target_row_slack_mm':float((target_prog-tstar).min(initial=np.inf)),
   'maximum_bound_violation_mm':bound_violation,
   'minimum_bone_row_slack_mm':float(min(bone_slack,default=np.inf)),
   'target_progress_mm':target_prog.tolist(),
   'maximum_absolute_component_move_mm':float(np.max(np.abs(lpd),initial=0.0)),
   'hard_rows_independently_feasible':bool(hard_slack.min(initial=np.inf)>=-2e-7 and bound_violation<=1e-8),
   'all_target_rows_positive':bool(target_prog.min(initial=np.inf)>0.0)})
 if not lp.success or not lp_report['hard_rows_independently_feasible'] or not lp_report['all_target_rows_positive']:
  early={'status':'HARD_CONSTRAINTS_INFEASIBLE' if not lp.success or not lp_report.get('hard_rows_independently_feasible',False) else 'NO_POSITIVE_COMMON_TARGET_PROGRESS',
   'baseline_sha256':expected,'bank':[109,114],'bone_separation_analysis_sha256':BONE_ANALYSIS_SHA,
   'bone_separation_rows_sha256':BONE_ROWS_SHA,'self_witness_sha256':SELF_PAIR_SHA,
   'feasibility_lp':lp_report,'geometry_admitted':False,'physical_steps':0}
  (OUT/'feasibility-report.json').write_text(json.dumps(early,indent=2,sort_keys=True,allow_nan=False)+'\n')
  print(json.dumps(early,indent=2,allow_nan=False),flush=True);return
 # Scale the complete QP objective by 1/100000 (same minimizer) so the
 # stiffness and slack blocks are O(1), and ask OSQP for one bounded scaled run.
 sys.path.insert(0,'/Users/numi/numi-human-fds132-m6-1350/attempt-311-posed-area-sqp-003/osqp-deps')
 import osqp
 H=5.0*(E0.T@E0)+0.02*sparse.eye(n,format='csr')
 P=sparse.block_diag((H/100000.0,sparse.eye(m,format='csr')),format='csc')
 K=sparse.vstack((sparse.hstack((A,Z(A.shape[0],m))),
                  sparse.hstack((T,U)),
                  sparse.hstack((I,Z(n,m))),
                  sparse.hstack((Z(m,n),U))),format='csc')
 lower=np.r_[b,np.full(m,.15),lo,np.zeros(m)]
 upper=np.r_[np.full(A.shape[0]+m,np.inf),hi,np.full(m,np.inf)]
 opt=osqp.OSQP()
 opt.setup(P=P,q=np.zeros(n+m),A=K,l=lower,u=upper,verbose=False,
           eps_abs=1e-6,eps_rel=1e-6,max_iter=120000,polishing=True,
           scaling=20,adaptive_rho=True,scaled_termination=True)
 result=opt.solve()
 solved=result.info.status_val==1
 sol=SimpleNamespace(success=solved,status=int(result.info.status_val),
                     message=str(result.info.status),x=result.x if solved else None)
 scaled_solver_report={'status':str(result.info.status),'status_val':int(result.info.status_val),
 'iterations':int(result.info.iter),'primal_residual':float(result.info.prim_res),
 'dual_residual':float(result.info.dual_res),'scaled_objective_factor':1e-5,
 'eps_abs':1e-6,'eps_rel':1e-6,'max_iter':120000,'scaling':20,'adaptive_rho':True,
 'scaled_termination':True,'lp_maximum_common_target_progress_mm':lp_report['maximum_common_target_progress_mm']}

 rep={'status':'QP_FAILED','baseline_sha256':expected,'baseline_audit130_sha256':sha(BASE130),'baseline_audit159_sha256':sha(BASE159),'bank':[109,114],'target_inside_nodes':selected.tolist(),'support_rings':6,'edge_gradient_cap':0.1,'quadratic_graph_weight':5.0,'source_oriented_area_floor':0.95,'source_area_guard_rows':int(len(affected)),'target_movable_witnesses':int(m),'support_nodes':len(support),'moving_support_nodes':int(len(moving_support)),'affected_faces':int(len(affected)),'point_depth_active_rows':len(prs),'point_depth_report':prmeta,'exact_self_witness_pairs':self_pairs.tolist(),'exact_self_witness_file_sha256':SELF_PAIR_SHA,'self_separation_helper':str(q.DEFAULT_PAIR_OWNER),'self_separation_helper_sha256':sha(q.DEFAULT_PAIR_OWNER),'self_barrier_row_count':len(self_rows),'self_barrier_records':self_records,'bone_barrier_row_count':len(bone_rows),'bone_separation_analysis_sha256':BONE_ANALYSIS_SHA,'bone_separation_rows_sha256':BONE_ROWS_SHA,'bone_separation_pair_count':len(bone_doc['added_exact_pairs']),'bone_separation_pairs':bone_doc['added_exact_pairs'],'bone_separation_projection_summary':bone_doc['intended_correction_projection_summary'],'fixed_witness_nodes':fixed_extra,'feasibility_lp':lp_report,'scaled_solver':scaled_solver_report,'solver':{'success':bool(sol.success),'status':int(sol.status),'message':str(sol.message)},'geometry_admitted':False,'physical_steps':0}
 if sol.success:
  d=sol.x[:n].reshape(-1,3); y=x.copy(); y[support]=(x[support].astype(np.float64)+d/1000).astype(np.float32)
  actual=(y.astype(np.float64)-x.astype(np.float64))*1000.; ar=q.full_f32_area_gate(y,bm,ref,faces,.1); sl=float(np.min(A@actual[support].reshape(-1)-b,initial=0.)); tar=np.asarray(T@actual[support].reshape(-1)); source_max=float(np.linalg.norm(y.astype(float)*1000-srcmm,axis=1).max()); maxmove=float(np.linalg.norm(actual,axis=1).max())
  changed=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1)); rep.update(status='PRIVATE_CANDIDATE_PENDING_EXACT_CHECKS', area_pass=bool(ar['pass']),min_linear_slack=sl,source_max_mm=source_max,max_move_mm=maxmove, target_projected_progress_mm=tar.tolist(), target_projected_progress_sum_mm=float(tar.sum()), target_progress_min_mm=float(tar.min()), target_slack_sum_mm=float(sol.x[-m:].sum()), changed_nodes=changed.tolist(), elapsed_s=time.monotonic()-t)
  if not ar['pass'] or sl< -1e-4 or source_max>12.000001 or maxmove>.500001 or not np.array_equal(y[anchors],x[anchors]) or len(changed)==0:
   rep['status']='REJECTED_EMISSION_GATES'
  else:
   payload['points_world_m']=y; targetfile=OUT/'candidate-private.npz'
   with targetfile.open('xb') as f:np.savez_compressed(f,**payload)
   rep['candidate_path']=str(targetfile);rep['candidate_sha256']=sha(targetfile)
 if rep.get('candidate_path'):
  candidate_path=Path(rep['candidate_path'])
  with np.load(candidate_path,allow_pickle=False) as z: candidate_payload={k:z[k].copy() for k in z.files}
  cand_points=np.asarray(candidate_payload['points_world_m'],dtype=np.float32)
  shape=strict_shape_report(x,cand_points,faces)
  rep['strict_shape_gate']=shape
  rep['nonposition_metadata_identical']=all(payload[k].dtype==candidate_payload[k].dtype and payload[k].shape==candidate_payload[k].shape and payload[k].tobytes()==candidate_payload[k].tobytes() for k in payload if k!='points_world_m')
  rep['typed_anchors_bitwise_unchanged']=bool(np.array_equal(cand_points[anchors].view(np.uint32),x[anchors].view(np.uint32)))
  rep['max_emitted_node_displacement_mm']=float(np.linalg.norm(cand_points.astype(np.float64)-x.astype(np.float64),axis=1).max()*1000)
  if not shape['pass'] or not rep['nonposition_metadata_identical'] or not rep['typed_anchors_bitwise_unchanged']:
   rep['status']='REJECTED_STRICT_SHAPE_OR_METADATA'
  else:
   rep['status']='PRIVATE_CANDIDATE_PENDING_BACKTRACK_EXACT_CHECKS'
 rep['elapsed_s']=time.monotonic()-t
 (OUT/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:rep.get(k) for k in ['status','solver','support_nodes','moving_support_nodes','fixed_witness_nodes','affected_faces','self_barrier_row_count','area_pass','strict_shape_gate','min_linear_slack','source_max_mm','max_move_mm','target_progress_min_mm','target_projected_progress_sum_mm','candidate_sha256','scoped_exact_self','full_audits','elapsed_s']},indent=2,default=lambda v:v.item() if hasattr(v,'item') else v.tolist() if hasattr(v,'tolist') else str(v)),flush=True)
if __name__=='__main__':main()
