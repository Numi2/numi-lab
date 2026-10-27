#!/usr/bin/env python3
"""Private joint source-aware depth continuation for four banks; no physical admission."""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys, time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
B=S/'per-bank-workers-20261027/b124-123/root-depth-quadratic-strain-001'
TARGETS={(57,89),(114,117),(59,95),(64,98)}
OUT=Path(__file__).parent
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
DPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/point-depth-barriers-357/point_depth_barriers_357.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANK=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
SOURCE=Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p); x=importlib.util.module_from_spec(sp);sys.modules[name]=x;sp.loader.exec_module(x);return x

def main():
 t=time.monotonic(); base=B/'candidate-private.npz'; summary=B/'audit-130/summary.json'; query=B/'audit-130/query'
 expected='e111fa302a41f7c80e278219c43451a8e56452b563d638855c0e477744a83be1'
 if sha(base)!=expected:raise ValueError('baseline pin mismatch')
 with np.load(base,allow_pickle=False) as z: payload={k:z[k].copy() for k in z.files}
 x=payload['points_world_m']; faces=payload['source_O_faces'].astype(np.int64); ref=payload['area_reference_037_points_world_m_f64']*1000.; anchors=payload['typed_coarse_anchor_output_rows_018'].astype(np.int64)
 doc=json.loads(summary.read_text()); assert doc['candidate_sha256']==expected and doc['complete_surface_audit'] and doc['exact_PLC_surface_pairs']==0
 witness=[r for r in doc['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id'])) in TARGETS]
 selected=np.array(sorted({int(r['query_row']) for r in witness}),dtype=np.int64)
 if not len(selected):raise ValueError('no target interior witnesses')
 edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
 adj=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(x),len(x)))
 support=set(map(int,selected)); frontier=set(support)
 for _ in range(4):
  frontier=set(map(int,adj[list(frontier)].indices))-support; support.update(frontier)
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
 normals,_,_,req=q.fixed_reference_area_requirements(bm,ref,faces)
 G,rhs,_,_=area.area_linearization(bm,faces,affected,normals[affected],req[affected],minimum_ratio=.1)
 A,b=q._owner_matrix(prs,G,rhs,support); EG,eb,_=q.build_edge_gradient_constraints(bm,faces,support,.2); A=sparse.vstack((A,EG),format='csr');b=np.r_[b,eb]
 lo,hi,boundsmeta=q.source_safe_trust_bounds(bm,srcmm,support,anchors,.5,12.)
 inside_kind=np.array([r['kind']=='inside_nonregression' for r in records],dtype=bool)
 target_banks=np.array([(int(b),int(c)) in TARGETS for _,b,c,_ in ids],dtype=bool)
 target=np.flatnonzero(target_banks&inside_kind&np.isin(ns,support)&(~np.isin(ns,anchors)))
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
 # Reuse the previously installed OSQP package for a private construction test.
 # Minimize squared neighboring displacement differences with source/depth
 # guards hard and target progress represented by nonnegative soft slack.
 from types import SimpleNamespace
 sys.path.insert(0,'/Users/numi/numi-human-fds132-m6-1350/attempt-311-posed-area-sqp-003/osqp-deps')
 import osqp
 H=(E0.T@E0)+0.02*sparse.eye(n,format='csr')
 P=sparse.block_diag((H,100000*sparse.eye(m,format='csr')),format='csc')
 K=sparse.vstack((sparse.hstack((A,Z(A.shape[0],m))),
                  sparse.hstack((T,U)),
                  sparse.hstack((I,Z(n,m))),
                  sparse.hstack((Z(m,n),U))),format='csc')
 lower=np.r_[b,np.full(m,.15),lo,np.zeros(m)]
 upper=np.r_[np.full(A.shape[0]+m,np.inf),hi,np.full(m,np.inf)]
 opt=osqp.OSQP()
 opt.setup(P=P,q=np.zeros(n+m),A=K,l=lower,u=upper,verbose=False,
           eps_abs=1e-7,eps_rel=1e-7,max_iter=40000,polishing=True)
 result=opt.solve()
 solved=result.info.status_val==1
 sol=SimpleNamespace(success=solved,status=int(result.info.status_val),
                     message=str(result.info.status),x=result.x if solved else None)

 rep={'status':'LP_FAILED','baseline_sha256':expected,'bank':[list(v) for v in sorted(TARGETS)],'target_inside_nodes':selected.tolist(),'target_movable_witnesses':int(m),'support_nodes':len(support),'affected_faces':len(affected),'point_depth_active_rows':len(prs),'point_depth_report':prmeta,'solver':{'success':bool(sol.success),'status':int(sol.status),'message':str(sol.message)},'geometry_admitted':False,'physical_steps':0}
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
 (OUT/'report.json').write_text(json.dumps(rep,indent=2,allow_nan=False)+'\n'); print(json.dumps({k:rep.get(k) for k in ['status','solver','target_movable_witnesses','support_nodes','affected_faces','area_pass','min_linear_slack','source_max_mm','max_move_mm','target_projected_progress_sum_mm','target_progress_min_mm','target_slack_sum_mm','candidate_sha256','elapsed_s']},indent=2))
if __name__=='__main__':main()
