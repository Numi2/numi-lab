#!/usr/bin/env python3
"""Private radius-aware connected position repair for the pinned 806 checkpoint; no physical admission."""
from pathlib import Path
import csv, hashlib, importlib.util, json, sys, time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
B=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/combined-d071-b109-b61-001')
OUT=Path(__file__).parent
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
DPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/point-depth-barriers-357/point_depth_barriers_357.py')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANK=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
BANKHELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
SOURCE=Path('/Users/numi/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz')
SELF_RECEIPT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b43-16/attempt-023-original-patch-layer-depth-001/exact-self-check.json')
SELF_PIN='391eb0ecf1dd7acdaf4089667b6abae0be00b5e5f43fede521a3094e435895c2'
LAYER_RECEIPT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b43-16/attempt-023-original-patch-layer-depth-001/layer-identity-check-all-positive.json')
LAYER_RECEIPT_PIN='4a0f15109a47b06701d7fad3737d6657bf88b1b5d0ff3a82dab2c7bb03925f8d'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p); x=importlib.util.module_from_spec(sp);sys.modules[name]=x;sp.loader.exec_module(x);return x


def make_layer_guard(layer_file, faces, affected, support, special_reserve):
 # Different layer points with the same vertex row receive the same outer
 # displacement because the owner interpolates fixed source layer offsets.
 with np.load(layer_file,allow_pickle=False) as z:
  assert np.array_equal(z['outer_faces'],faces[:,[0,2,1]])
  planes=[z['outer_points_f64']*1000.,z['inner_points_f64']*1000.,z['deep_points_f64']*1000.]
  oriented=z['outer_faces']
 local={int(node):i for i,node in enumerate(support)}
 R=[];C=[];V=[];bounds=[];skipped=0;nonpositive=0
 for li in (0,1):
  out,inn=planes[li],planes[li+1]
  for fi in affected:
   original=oriented[fi];a,b,c=sorted(map(int,original))
   parity=(int(original[0]>original[1])+int(original[0]>original[2])+int(original[1]>original[2]))&1
   expected=1. if parity else -1.
   tetra=[([(a,out[a]),(b,out[b]),(c,out[c]),(c,inn[c])],expected),
          ([(a,out[a]),(b,out[b]),(b,inn[b]),(c,inn[c])],-expected),
          ([(a,out[a]),(a,inn[a]),(b,inn[b]),(c,inn[c])],expected)]
   for slot,(vertices,sign) in enumerate(tetra):
    p0,p1,p2,p3=[v[1] for v in vertices]
    u=p1-p0;v=p2-p0;w=p3-p0
    signed=float(sign*np.dot(u,np.cross(v,w))/6.)
    grads=[None,sign*np.cross(v,w)/6.,sign*np.cross(w,u)/6.,sign*np.cross(u,v)/6.]
    grads[0]=-grads[1]-grads[2]-grads[3]
    combined={}
    for (node,_),grad in zip(vertices,grads):
     if node in local:combined[node]=combined.get(node,np.zeros(3))+grad
    nrm=float(np.sqrt(sum(np.dot(g,g) for g in combined.values())))
    if nrm<=1e-13:
     skipped+=1
     continue
    if signed<=0:nonpositive+=1
    # Positive oriented tetrahedra retain at least 20% of current signed
    # volume in the linearization; existing bad ones may not worsen.
    lower_volume=(-.8*signed if signed>0 else 0.)
    reserve=float(special_reserve.get((int(fi),int(li),int(slot)),0.))
    lower=(lower_volume+reserve)/nrm
    row=len(bounds);bounds.append(lower)
    for node,grad in combined.items():
     for axis in range(3):
      R.append(row);C.append(3*local[node]+axis);V.append(grad[axis]/nrm)
 mat=sparse.coo_matrix((V,(R,C)),shape=(len(bounds),3*len(support))).tocsr()
 return mat,np.asarray(bounds),{'rows':len(bounds),'baseline_nonpositive_rows':nonpositive,'skipped_fixed_or_zero_rows':skipped,'reinforced_positive_cells':{str(k):float(v) for k,v in special_reserve.items()}}

def main():
 t=time.monotonic(); base=B/'candidate.npz'; summary=B/'audit-130/summary.json'; query=B/'audit-130/query'
 expected='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
 if sha(base)!=expected:raise ValueError('baseline pin mismatch')
 if np.load(B/'audit-130/surface-self-pairs.npy',allow_pickle=False).shape!=(0,2):raise ValueError('pinned 806 baseline self pair inventory is nonempty')
 with np.load(base,allow_pickle=False) as z: payload={k:z[k].copy() for k in z.files}
 x=payload['points_world_m']; faces=payload['source_O_faces'].astype(np.int64); ref=payload['area_reference_037_points_world_m_f64']*1000.; anchors=payload['typed_coarse_anchor_output_rows_018'].astype(np.int64)
 doc=json.loads(summary.read_text()); assert doc['candidate_sha256']==expected and doc['complete_surface_audit'] and doc['exact_PLC_surface_pairs']==0
 audit159_path=B/'audit-159/report.json'
 
 audit159=json.loads(audit159_path.read_text())
 if audit159.get('candidate_sha256')!=expected or not audit159.get('complete') or int(audit159.get('expected_bank_count',0))!=185:raise ValueError('pinned 806 full bank audit missing or stale')
 bank43=next(v for v in audit159['banks'] if (int(v['body']),int(v['stable_bone_id']))==(43,16))
 raw_baseline_pairs={(int(f),int(t)) for f,t in (bank43.get('result') or {}).get('triangle_pairs',[])}
 ref_i=payload['refinement_face_parent_ancestry_indptr'];ref_r=payload['refinement_face_parent_ancestry_face_rows']
 mapped_baseline_pairs={(int(p),int(t)) for f,t in raw_baseline_pairs for p in ref_r[int(ref_i[f]):int(ref_i[f+1])] }
 if not raw_baseline_pairs or not mapped_baseline_pairs:raise ValueError(f'pinned 806 target crossings missing: {len(raw_baseline_pairs)}/{len(mapped_baseline_pairs)}')
 witness=[r for r in doc['inside_witnesses'] if (int(r['bank_body']),int(r['bank_stable_id']))==(43,16) and r.get('classification')=='inside' and int(r.get('ambiguous',0))==0]
 all_selected={int(r['query_row']) for r in witness}
 if len(all_selected)!=43:raise ValueError(f'expected all 43 exact inside nodes for 43/16, got {len(all_selected)}')
 selected=np.array(sorted(all_selected),dtype=np.int64)
 if not len(selected):raise ValueError('no target interior witnesses')
 edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
 adj=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(x),len(x)))
 support=set(map(int,selected)); frontier=set(support)
 for _ in range(4):
  frontier=set(map(int,adj[list(frontier)].indices))-support; support.update(frontier)
 moving_support=set(support)
 pair_path=B/'audit-130/surface-self-pairs.npy'
 baseline_self=np.load(pair_path,allow_pickle=False)
 if baseline_self.shape!=(0,2):raise ValueError('pinned 806 baseline self inventory is not empty')
 # Failed-position-fit self pairs are treated as guard witnesses, not as current collisions.
 failed_self_paths=[S/'per-bank-workers-20261027/b43-16/attempt-023-original-patch-layer-depth-001/exact-self-check.json',S/'per-bank-workers-20261027/b43-16/attempt-024-witness-constrained-fit-001/exact-self-check.json']
 self_guard=set()
 for fp in failed_self_paths:
  docself=json.loads(fp.read_text())
  for pair in docself['exact_self_pairs']:
   self_guard.add(tuple(sorted((int(pair[0]),int(pair[1])))))
 self_guard_pairs=np.asarray(sorted(self_guard),dtype=np.int64).reshape(-1,2)
 support.update(map(int,faces[self_guard_pairs.reshape(-1)].reshape(-1)))
 fixed_extra=[]
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
 # Restrict further shape change against the pinned starting candidate.
 origpath=base
 if sha(origpath)!=expected:raise ValueError('shape baseline is not pinned 806')
 with np.load(origpath,allow_pickle=False) as z: orig=z['points_world_m'].astype(float)*1000.
 t0=orig[faces[affected]];n0=np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]);a0=np.linalg.norm(n0,axis=1)
 if np.any(a0<=0):raise ValueError('degenerate original face')
 tri=bm[faces[affected]]
 nc=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);ac=np.linalg.norm(nc,axis=1)
 if np.any(ac<=0):raise ValueError('degenerate current face')
 Gu,lo_area,cur,_=area.area_linearization(bm,faces,affected,nc/ac[:,None],a0,minimum_ratio=.76)
 Gn,_,proj,_=area.area_linearization(bm,faces,affected,n0/a0[:,None],a0,minimum_ratio=.76)
 G=sparse.vstack((G,Gu,-Gu,Gn-.91*Gu),format='csr')
 rhs=np.r_[rhs,lo_area,cur-1.29*a0,.91*ac-proj]
 sep=mod('depth_self_sep',q.DEFAULT_PAIR_OWNER)
 self_rows,self_records=sep.build_pair_rows(bm,faces,self_guard_pairs,margin_fraction=.25)
 A,b=q._owner_matrix(prs+self_rows,G,rhs,support); EG,eb,_=q.build_edge_gradient_constraints(bm,faces,support,.2)
 layer_file=B/'skin-fat-layers-541.npz'
 if not layer_file.is_file():raise ValueError('matching pinned 806 layer bundle missing')
 failed_layer_path=S/'per-bank-workers-20261027/b43-16/attempt-024-witness-constrained-fit-001/layer-transfer-check.json'
 failed_layer_doc=json.loads(failed_layer_path.read_text())
 layer_witnesses=failed_layer_doc['proposed_same_node_layer_transfer']['violations_first_64']
 affected_set=set(map(int,affected))
 special_reserve={}
 for rr0 in layer_witnesses:
  key=(int(rr0['face_row']),int(rr0['layer_interval']),int(rr0['tetra_slot']))
  if key[0] in affected_set:
   special_reserve[key]=2.0*float(rr0['violation_mm3'])+1e-7
 LG,lb,layer_receipt=make_layer_guard(layer_file,faces,affected,support,special_reserve)
 A=sparse.vstack((A,EG,LG),format='csr');b=np.r_[b,eb,lb]
 lo,hi,boundsmeta=q.source_safe_trust_bounds(bm,srcmm,support,anchors,.5,12.)
 for vertex in fixed_extra:
  j=3*loc[vertex];lo[j:j+3]=0.;hi[j:j+3]=0.
 inside_kind=np.array([r['kind']=='inside_nonregression' for r in records],dtype=bool)
 target=np.flatnonzero((ids[:,1]==43)&(ids[:,2]==16)&inside_kind&np.isin(ns,support)&(~np.isin(ns,anchors)))
 if len(target)==0:raise ValueError('all target witnesses fixed')
 n=3*len(support);m=len(target); rr=[];cc=[];vv=[]
 for i,k in enumerate(target):
  for j in range(3):rr.append(i);cc.append(3*loc[int(ns[k])]+j);vv.append(norms[k,j])
 Tpoint=sparse.coo_matrix((vv,(rr,cc)),shape=(m,n)).tocsr()
 # Build pair-specific local tangent witnesses from the 247 exact crossing pairs.
 bank_tri,bank_ids,bank_gids,bank_bodies,bank_M,bank_offset,bank_pos,bank_keys,bank_pins,_=mod('radius_bank_helper',BANKHELPER).read_bank_inputs(BANK,E)
 target_bank_rows=np.flatnonzero((bank_ids[:,0]==43)&(bank_ids[:,1]==16))
 tri_by_id={int(bank_ids[i,3]):int(i) for i in target_bank_rows}
 normal_votes=[]
 for wr in witness:
  ti=tri_by_id[int(wr['nearest_triangle'])];tri=bank_tri[ti]
  nn=np.cross(tri[1]-tri[0],tri[2]-tri[0]); nn=nn/max(float(np.linalg.norm(nn)),1e-300)
  ql=np.asarray([float(wr['query_local_x']),float(wr['query_local_y']),float(wr['query_local_z'])])
  cl=np.asarray([float(wr['closest_x']),float(wr['closest_y']),float(wr['closest_z'])])
  normal_votes.append(float(np.dot(ql-cl,nn)))
 if len(normal_votes)!=43 or not all(v<0 for v in normal_votes):raise ValueError('43/16 outward triangle winding is not confirmed by exact inside witnesses')
 crossing_rows=[]; crossing_rhs=[]; crossing_meta=[]
 for skin_row,bone_id in sorted(raw_baseline_pairs):
  if int(bone_id) not in tri_by_id:raise ValueError(f'crossing bone triangle absent from typed bank: {bone_id}')
  bt=bank_tri[tri_by_id[int(bone_id)]]
  nl=np.cross(bt[1]-bt[0],bt[2]-bt[0]); nl=nl/max(float(np.linalg.norm(nl)),1e-300)
  nw=nl@bank_M[43]; nw=nw/max(float(np.linalg.norm(nw)),1e-300)
  tri_local=(bm[faces[int(skin_row)]]/1000.0-bank_pos[43])@bank_M[43].T-bank_offset[43]
  signed=((tri_local-bt[0])@nl)*1000.0
  mean_d=float(np.mean(signed)); needed=0.03-mean_d
  # One barrier per exact intersecting face/bone-triangle pair; this asks only
  # for local centroid separation on that actual witness, with explicit slack.
  crossing_rows.append((int(skin_row),int(bone_id),nw.copy()))
  crossing_rhs.append(needed)
  crossing_meta.append({'skin_face_row':int(skin_row),'bank_triangle_id':int(bone_id),'baseline_centroid_signed_mm':mean_d,'required_centroid_mm':0.03,'required_delta_mm':needed})
 cr=[];cidx=[];cv=[]
 for row,(skin_row,bone_id,nw) in enumerate(crossing_rows):
  for node in faces[skin_row]:
   j=loc.get(int(node))
   if j is not None:
    for axis in range(3):cr.append(row);cidx.append(3*j+axis);cv.append(float(nw[axis])/3.0)
 Ccross=sparse.coo_matrix((cv,(cr,cidx)),shape=(len(crossing_rows),n)).tocsr()
 T=sparse.vstack((Tpoint,Ccross),format='csr')
 target_req=np.r_[np.full(m,0.15),np.asarray(crossing_rhs,dtype=np.float64)]
 point_target_count=int(m); cross_target_count=int(len(crossing_rows)); m=int(len(target_req))
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
 Am=sparse.vstack(blocks,format='csr'); rhsall=np.r_[-b,np.zeros(2*n+2*nr),-target_req]
 c=np.r_[np.zeros(n),np.full(n,.02),np.full(nr,5.),np.full(m,1000.)]
 bounds=list(zip(lo,hi))+[(0,None)]*(n+nr+m)
 # Reuse the previously installed OSQP package for a private construction test.
 # Minimize squared neighboring displacement differences with source/depth
 # guards hard and target progress represented by nonnegative soft slack.
 from types import SimpleNamespace
 sys.path.insert(0,'/Users/numi/numi-human-fds132-m6-1350/attempt-311-posed-area-sqp-003/osqp-deps')
 import osqp
 H=(E0.T@E0)+0.02*sparse.eye(n,format='csr')
 P=sparse.block_diag((H,5000*sparse.eye(m,format='csr')),format='csc')
 K=sparse.vstack((sparse.hstack((A,Z(A.shape[0],m))),
                  sparse.hstack((T,U)),
                  sparse.hstack((I,Z(n,m))),
                  sparse.hstack((Z(m,n),U))),format='csc')
 lower=np.r_[b,target_req,lo,np.zeros(m)]
 upper=np.r_[np.full(A.shape[0]+m,np.inf),hi,np.full(m,np.inf)]
 opt=osqp.OSQP()
 opt.setup(P=P,q=np.zeros(n+m),A=K,l=lower,u=upper,verbose=False,
           eps_abs=5e-6,eps_rel=5e-6,max_iter=100000,polishing=True)
 result=opt.solve()
 val=np.asarray(K@result.x) if result.x is not None else np.full(len(lower),np.nan)
 lowerr=float(np.max(np.maximum(lower-val,0),initial=0))
 upperr=float(np.max(np.maximum(val-upper,0),initial=0))
 solved=(result.info.status_val==1 and np.isfinite(val).all() and lowerr<=1e-5 and upperr<=1e-5)
 sol=SimpleNamespace(success=solved,status=int(result.info.status_val),
                     message=str(result.info.status),x=result.x if solved else None)

 rep={'status':'LP_FAILED','baseline_sha256':expected,'bank':[43,16],'input_pins':{'audit130_sha256':sha(summary),'audit159_sha256':sha(audit159_path),'layer_sha256':sha(layer_file),'self_pairs_sha256':sha(pair_path),'failed_self_witness_sha256':sha(failed_self_paths[1]),'failed_layer_witness_sha256':sha(failed_layer_path)},'baseline_target_raw_pairs':len(raw_baseline_pairs),'baseline_target_mapped_parent_pairs':len(mapped_baseline_pairs),'target_inside_node_count':len(selected),'target_inside_nodes':selected.tolist(),'target_movable_witnesses':int(point_target_count),'crossing_tangent_witness_count':int(cross_target_count),'crossing_tangent_witnesses':crossing_meta,'support_nodes':len(support),'affected_faces':len(affected),'layer_guard':layer_receipt,'layer_failure_witness_count':len(special_reserve),'layer_failure_witnesses':{str(k):v for k,v in special_reserve.items()},'failed_self_guard_pairs':self_guard_pairs.tolist(),'self_witness_pair_count':len(self_guard_pairs),'self_barrier_row_count':len(self_rows),'fixed_witness_nodes':len(fixed_extra),'point_depth_active_rows':len(prs),'point_depth_report':prmeta,'solver':{'success':bool(sol.success),'status':int(sol.status),'message':str(sol.message),'iterations':int(result.info.iter),'primal_residual':float(result.info.prim_res),'dual_residual':float(result.info.dual_res),'max_lower_violation':lowerr,'max_upper_violation':upperr},'geometry_admitted':False,'physical_steps':0}
 if sol.success:
  direction=sol.x[:n].reshape(-1,3)
  direction_path=OUT/'solved-direction.npz'
  np.savez_compressed(direction_path,support_nodes=support,direction_delta_mm=direction,
    target_slack_mm=sol.x[-m:],target_required_mm=target_req,target_matrix=T.toarray() if m<100 else np.empty((0,0)))
  trial_receipts=[]; selected=None; last_trial=None
  for exponent in range(9):
   alpha=0.5**exponent
   scaled=direction*alpha
   y=x.copy();y[support]=(x[support].astype(np.float64)+scaled/1000.0).astype(np.float32)
   actual=(y.astype(np.float64)-x.astype(np.float64))*1000.0
   ar=q.full_f32_area_gate(y,bm,ref,faces,.1)
   linear_slack=float(np.min(A@actual[support].reshape(-1)-b,initial=0.0))
   tar=np.asarray(T@actual[support].reshape(-1))
   source_max=float(np.linalg.norm(y.astype(np.float64)*1000.0-srcmm,axis=1).max())
   maxmove=float(np.linalg.norm(actual,axis=1).max())
   anchors_same=bool(np.array_equal(y[anchors].view(np.uint32),x[anchors].view(np.uint32)))
   outside_same=bool(np.array_equal(y[np.setdiff1d(np.arange(len(x)),support)].view(np.uint32),x[np.setdiff1d(np.arange(len(x)),support)].view(np.uint32)))
   hard_pass=bool(ar['pass'] and linear_slack>=-1e-4 and source_max<=12.000001 and maxmove<=.500001 and anchors_same and outside_same)
   trial={'alpha':float(alpha),'area_gate':ar,'linear_min_slack':linear_slack,'source_max_mm':source_max,'max_move_mm':maxmove,
      'target_progress_min_mm':float(tar.min(initial=np.inf)),'target_progress_sum_mm':float(tar.sum()),
      'target_residual_sum_mm':float(np.maximum(target_req-tar,0).sum()),'hard_emission_pass':hard_pass}
   trial_receipts.append(trial);last_trial=(y,actual,ar,linear_slack,tar,source_max,maxmove,alpha)
   if hard_pass:
    selected=last_trial;break
  rep.update(status='REJECTED_EMISSION_GATES',direction_path=str(direction_path),direction_sha256=sha(direction_path),
    shape_backtrack_trials=trial_receipts,shape_backtrack_selected_alpha=None,
    full_step_area_gate=trial_receipts[0]['area_gate'],model_dimensions={'support_nodes':len(support),'coordinate_dofs':n,
      'target_progress_rows':m,'affected_faces':len(affected),'layer_guard_rows':int(LG.shape[0]),
      'self_barrier_rows':int(self_rows.shape[0]) if hasattr(self_rows,'shape') else len(self_rows),
      'point_depth_rows':int(prs.shape[0]) if hasattr(prs,'shape') else len(prs),
      'constraint_matrix_rows':int(K.shape[0]),'constraint_matrix_nnz':int(K.nnz),'objective_variables':int(n+m)})
  if selected is None:
   y,actual,ar,sl,tar,source_max,maxmove,alpha=last_trial
   failed_payload=dict(payload);failed_payload['points_world_m']=y
   failpath=OUT/'candidate-shape-rejected-private.npz'
   with failpath.open('xb') as f:np.savez_compressed(f,**failed_payload)
   rep.update(status='REJECTED_NO_SHAPE_PASS_WITHIN_1_OVER_256',shape_rejected_artifact=str(failpath),
      shape_rejected_sha256=sha(failpath),last_alpha=float(alpha),area_gate=ar,
      min_linear_slack=sl,source_max_mm=source_max,max_move_mm=maxmove,
      target_projected_progress_mm=tar.tolist(),target_projected_progress_sum_mm=float(tar.sum()),
      target_progress_min_mm=float(tar.min(initial=np.inf)),target_residual_sum_mm=float(np.maximum(0.15-tar,0).sum()),
      elapsed_s=time.monotonic()-t)
  else:
   y,actual,ar,sl,tar,source_max,maxmove,alpha=selected
   changed=np.flatnonzero(np.any(y.view(np.uint32)!=x.view(np.uint32),axis=1))
   rep.update(status='PRIVATE_CANDIDATE_PENDING_EXACT_CHECKS',shape_backtrack_selected_alpha=float(alpha),
      area_pass=bool(ar['pass']),area_gate=ar,min_linear_slack=sl,source_max_mm=source_max,max_move_mm=maxmove,
      target_projected_progress_mm=tar.tolist(),target_projected_progress_sum_mm=float(tar.sum()),
      target_progress_min_mm=float(tar.min(initial=np.inf)),target_residual_sum_mm=float(np.maximum(0.15-tar,0).sum()),
      changed_nodes=changed.tolist(),elapsed_s=time.monotonic()-t)
   targetfile=OUT/'candidate-private.npz'
   with targetfile.open('xb') as f:
    payload['points_world_m']=y;np.savez_compressed(f,**payload)
   rep['candidate_path']=str(targetfile);rep['candidate_sha256']=sha(targetfile)
 (OUT/'report.json').write_text(json.dumps(rep,indent=2,allow_nan=False)+'\n'); print(json.dumps({k:rep.get(k) for k in ['status','solver','target_movable_witnesses','support_nodes','affected_faces','area_pass','min_linear_slack','source_max_mm','max_move_mm','target_projected_progress_sum_mm','target_progress_min_mm','target_slack_sum_mm','candidate_sha256','elapsed_s']},indent=2))
if __name__=='__main__':main()
