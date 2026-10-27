#!/usr/bin/env python3
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
from scipy.optimize import linprog
HERE=Path(__file__).resolve().parent
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';BASE=W/'combined-d071-b109-b61-001';OLD=W/'b43-16/attempt-024-witness-constrained-fit-001'
PREV=W/'b43-16/attempt-030-radius-aware-position-001/targeted-fat-cell-001'
SCREEN=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b43-16/attempt-030-radius-aware-position-001/rebase-024-on-806/deep-backtrack-001/screen_rebased_direction.py')
QPATH=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py')
TRANSFER=W/'b65-99/attempt-024-topology-independent-layer-transfer-001/transfer_topology_independent_541.py'
AUD159=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-181-stage-001/sources/checks/audit_skin_bone_surfaces_159.py')
BANKHELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
BANKDIR=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight')
E=BANKDIR.parents[0].parents[0].parents[0]
BASE_SHA='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
DIR_SHA='b1550f73cae5d4b705add5467309ae3843341fec13a1b74a3516ebc308daf504'
PREV_CAND_SHA='1d8cb93675092095fb60cd0823393df365ddfaf667152f6e286081496e78dcb6'
ALPHA=.00390625; TARGET_FACE=19873; TARGET_SLOT=1; BONE_TRI=140
TRUST=.05

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def closest_pt_tri(p,a,b,c):
 ab=b-a;ac=c-a;ap=p-a;d1=ab@ap;d2=ac@ap
 if d1<=0 and d2<=0:return a,np.array([1.,0.,0.])
 bp=p-b;d3=ab@bp;d4=ac@bp
 if d3>=0 and d4<=d3:return b,np.array([0.,1.,0.])
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:
  v=d1/(d1-d3);return a+v*ab,np.array([1-v,v,0.])
 cp=p-c;d5=ab@cp;d6=ac@cp
 if d6>=0 and d5<=d6:return c,np.array([0.,0.,1.])
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:
  w=d2/(d2-d6);return a+w*ac,np.array([1-w,0.,w])
 va=d3*d6-d5*d4
 if va<=0 and (d4-d3)>=0 and (d5-d6)>=0:
  w=(d4-d3)/((d4-d3)+(d5-d6));return b+w*(c-b),np.array([0.,1-w,w])
 den=1./(va+vb+vc);v=vb*den;w=vc*den;return a+ab*v+ac*w,np.array([1-v-w,v,w])
def segseg(p1,q1,p2,q2):
 d1=q1-p1;d2=q2-p2;r=p1-p2;a=d1@d1;e=d2@d2;f=d2@r
 if a<=1e-30 and e<=1e-30:return p1,p2,0.,0.
 if a<=1e-30:s=0.;t=np.clip(f/e,0.,1.)
 else:
  c=d1@r
  if e<=1e-30:t=0.;s=np.clip(-c/a,0.,1.)
  else:
   b=d1@d2;den=a*e-b*b;s=np.clip((b*f-c*e)/den,0.,1.) if den>1e-30 else 0.;t=(b*s+f)/e
   if t<0:t=0.;s=np.clip(-c/a,0.,1.)
   elif t>1:t=1.;s=np.clip((b-c)/a,0.,1.)
 return p1+s*d1,p2+t*d2,float(s),float(t)
def closest_tri_tri(A,B):
 cand=[]
 for i in range(3):
  q,w=closest_pt_tri(A[i],*B);ba=np.eye(3)[i];cand.append((float(np.linalg.norm(A[i]-q)),A[i],q,ba,w))
 for j in range(3):
  q,w=closest_pt_tri(B[j],*A);bb=np.eye(3)[j];cand.append((float(np.linalg.norm(B[j]-q)),q,B[j],w,bb))
 for i in range(3):
  for j in range(3):
   p,q,s,t=segseg(A[i],A[(i+1)%3],B[j],B[(j+1)%3]);wa=np.zeros(3);wb=np.zeros(3);wa[i]=1-s;wa[(i+1)%3]=s;wb[j]=1-t;wb[(j+1)%3]=t
   cand.append((float(np.linalg.norm(p-q)),p,q,wa,wb))
 d,p,q,wa,wb=min(cand,key=lambda x:x[0]);return d,p,q,wa,wb
def volume_mm3(outer,inner,F,face,slot):
 ori=F[face][[0,2,1]];a,b,c=sorted(map(int,ori));sets=[[(a,outer[a]),(b,outer[b]),(c,outer[c]),(c,inner[c])],[(a,outer[a]),(b,outer[b]),(b,inner[b]),(c,inner[c])],[(a,outer[a]),(a,inner[a]),(b,inner[b]),(c,inner[c])]]
 pts=np.asarray([v for _,v in sets[slot]],dtype=np.float64);v=float(np.dot(pts[1]-pts[0],np.cross(pts[2]-pts[0],pts[3]-pts[0]))/6)
 if v<0:pts[[0,1]]=pts[[1,0]]
 p=pts.astype(np.float32).astype(np.float64);return float(np.dot(p[1]-p[0],np.cross(p[2]-p[0],p[3]-p[0]))/6*1e9)
def fat_gradient(I,D,F,face,slot=1):
 ori=F[face][[0,2,1]];a,b,c=sorted(map(int,ori));p0=I[a]*1000;p1=I[b]*1000;p2=D[b]*1000;p3=D[c]*1000
 v=float(np.dot(p1-p0,np.cross(p2-p0,p3-p0))/6);sgn=1. if v>=0 else -1.
 g1=np.cross(p2-p0,p3-p0)/6;g2=np.cross(p3-p0,p1-p0)/6;g3=np.cross(p1-p0,p2-p0)/6;g0=-g1-g2-g3
 out={a:sgn*g0,b:sgn*(g1+g2),c:sgn*g3};return out,sgn*v

def main():
 started=time.monotonic();basepath=BASE/'candidate.npz';dirpath=OLD/'solved-direction.npz';prev=PREV/'candidate-exact-rejected-private.npz';prevrep=PREV/'report.json';layerpath=BASE/'skin-fat-layers-541.npz'
 if sha(basepath)!=BASE_SHA or sha(dirpath)!=DIR_SHA or sha(prev)!=PREV_CAND_SHA:raise ValueError('base/direction/candidate pin mismatch')
 screen=mod('cross_screen',SCREEN)
 with np.load(basepath,allow_pickle=False) as z:base={k:z[k].copy() for k in z.files}
 with np.load(dirpath,allow_pickle=False) as z:support=z['support_nodes'].astype(np.int64);direction=z['direction_delta_mm'].astype(np.float64)
 previous_report=json.load(open(prevrep)); correction=previous_report['target_cell']
 with np.load(prev,allow_pickle=False) as z:Yprev=z['points_world_m'].copy()
 X=base['points_world_m'].astype(np.float32);F=base['source_O_faces'].astype(np.int64);alpha=ALPHA
 loc={int(n):i for i,n in enumerate(support)}
 # Reconstruct the saved 1/256 direction plus its one-node measured fat-cell correction and require exact parity.
 corr=np.zeros(3);corr[int(correction['correction_axis'])]=float(correction['correction_sign'])*float(correction['correction_mm']);node=int(correction['correction_node'])
 Y=X.copy();Y[support]=(X[support].astype(np.float64)+(alpha*direction)/1000.).astype(np.float32);Y[node]=(X[node].astype(np.float64)+(alpha*direction[loc[node]]+corr)/1000.).astype(np.float32)
 if not np.array_equal(Y.view(np.uint32),Yprev.view(np.uint32)):raise ValueError('saved corrected 1/256 direction failed bitwise reconstruction')
 helper=mod('cross_bank_helper',screen.BANKHELPER);T,bank_ids,_,_,M,offset,pos,_,_,_=helper.read_bank_inputs(screen.BANKDIR,screen.E)
 tri_idx={int(bank_ids[i,3]):int(i) for i in np.flatnonzero((bank_ids[:,0]==43)&(bank_ids[:,1]==16))}
 target_rows=[19203,19214]; previous_pairs=previous_report['exact_target']['new_raw']
 if {(int(a),int(b)) for a,b in previous_pairs}!={(19203,140),(19214,140)}:raise ValueError('two new exact pair pins changed')
 with np.load(layerpath,allow_pickle=False) as z:layer={k:z[k].copy() for k in z.files}
 Xlocal=(X.astype(np.float64)-pos[43])@M[43].T-offset[43]
 Ylocal=(Y.astype(np.float64)-pos[43])@M[43].T-offset[43]
 constraints=[]
 for face,bid in [(19203,140),(19214,140)]:
  Btri=T[tri_idx[bid]];Atri=Xlocal[F[face]];gap,bp,spoint,bary_skin,bary_bone=closest_tri_tri(Atri,Btri)
  if gap<=1e-12:raise ValueError(f'baseline pair {face}/{bid} has no positive separation gap')
  n=(spoint-bp)/gap
  if float(np.dot(n,np.cross(Btri[1]-Btri[0],Btri[2]-Btri[0])))<0:pass
  cur_point=bary_skin@Ylocal[F[face]];curproj=float(np.dot(cur_point-bp,n))*1000.
  gapmm=float(gap*1000.)
  nw=n@M[43];nw=nw/max(float(np.linalg.norm(nw)),1e-300)
  constraints.append({'face':face,'bone_triangle':bid,'gap_mm':gapmm,'current_projected_mm':curproj,'required_nonregression_mm':gapmm,'rhs_mm':gapmm-curproj,'bary_skin':bary_skin.tolist(),'bary_bone':bary_bone.tolist(),'normal_world':nw.tolist(),'bone_point_local_m':bp.tolist()})
 # Tiny local correction variables are limited to nodes in the two crossing faces and fat witness face.
 nodes=np.asarray(sorted(set(int(n) for f in target_rows+[TARGET_FACE] for n in F[f])),dtype=np.int64)
 if not set(map(int,nodes)).issubset(loc):raise ValueError('target correction nodes fall outside retained direction support')
 nvar=3*len(nodes);nloc={int(v):i for i,v in enumerate(nodes)}
 Crow=[];Ccol=[];Cval=[];rhs=[]
 for row,c in enumerate(constraints):
  for k,nodeid in enumerate(F[int(c['face'])]):
   j=nloc[int(nodeid)]
   for axis in range(3):Crow.append(row);Ccol.append(3*j+axis);Cval.append(float(c['bary_skin'][k])*float(c['normal_world'][axis]))
  rhs.append(float(c['rhs_mm']))
 C=np.zeros((len(constraints),nvar));C[np.asarray(Crow),np.asarray(Ccol)]=Cval
 with np.load(layerpath,allow_pickle=False) as z:I0=z['inner_points_f64'].astype(np.float64);D0=z['deep_points_f64'].astype(np.float64)
 delta=Y.astype(np.float64)-X.astype(np.float64);Icur=I0+delta;Dcur=D0+delta
 base_vol=volume_mm3(I0,D0,F,TARGET_FACE,TARGET_SLOT);cur_vol=volume_mm3(Icur,Dcur,F,TARGET_FACE,TARGET_SLOT)
 gradmap,_=fat_gradient(Icur,Dcur,F,TARGET_FACE,TARGET_SLOT);g=np.zeros(nvar)
 for nodeid,gvec in gradmap.items():
  if int(nodeid) in nloc:g[3*nloc[int(nodeid)]:3*nloc[int(nodeid)]+3]=gvec
 # Keep the measured improved cell at least at its pinned 806 volume, with a 1e-9 mm^3 numerical margin.
 fat_rhs=float(base_vol+1e-9-cur_vol)
 A_ub=np.vstack([-C,-g[None,:],np.eye(nvar),-np.eye(nvar)])
 b_ub=np.r_[-np.asarray(rhs),-fat_rhs,np.zeros(2*nvar)]
 c=np.ones(2*nvar);A_lp=np.vstack([np.hstack([A_ub,np.zeros((len(b_ub),nvar))]),np.hstack([np.eye(nvar),-np.eye(nvar)]),np.hstack([-np.eye(nvar),-np.eye(nvar)])])
 b_lp=np.r_[b_ub,np.zeros(2*nvar)]
 result=linprog(c,A_ub=A_lp,b_ub=b_lp,bounds=[(-TRUST,TRUST)]*nvar+[(0,None)]*nvar,method='highs')
 rep={'schema':'numi.human.b43-16.two-new-crossing-pair-local-nonregression.v1','base_sha256':BASE_SHA,'direction_sha256':DIR_SHA,'starting_candidate_sha256':PREV_CAND_SHA,'starting_candidate_source':'saved 1/256 rebase plus measured face-19873 fat-cell correction','alpha':ALPHA,'geometry_admitted':False,'accepted_physical_steps':0,'crossing_nonregression_constraints':constraints,'fat_cell_constraint':{'face_row':TARGET_FACE,'interval':'fat','tetra_slot':TARGET_SLOT,'baseline_mm3':base_vol,'starting_mm3':cur_vol,'minimum_allowed_mm3':base_vol+1e-9,'gradient_norm':float(np.linalg.norm(g)),'linear_rhs_mm3':fat_rhs},'local_correction_nodes':nodes.tolist(),'local_correction_dofs':nvar,'trust_per_component_mm':TRUST,'solver':{'success':bool(result.success),'status':int(result.status),'message':result.message,'objective_l1_mm':float(result.fun) if result.success else None},'elapsed_s':time.monotonic()-started}
 if not result.success:
  rep['status']='REJECTED_LOCAL_CORRECTION_INFEASIBLE'
  (HERE/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps(rep,indent=2));return
 fix=result.x[:nvar].reshape(-1,3);Ynew=Y.copy();totaldelta=(alpha*direction).copy();totaldelta[loc[node]]+=corr
 for j,nodeid in enumerate(nodes):totaldelta[loc[int(nodeid)]]+=fix[j]
 Ynew[support]=(X[support].astype(np.float64)+totaldelta/1000.).astype(np.float32)
 changed=np.flatnonzero(np.any(Ynew.view(np.uint32)!=X.view(np.uint32),axis=1)).astype(np.int64);changed_faces=np.flatnonzero(np.any(np.isin(F,changed),axis=1)).astype(np.int64);affected=np.flatnonzero(np.any(np.isin(F,support),axis=1)).astype(np.int64)
 shape=screen.shape_metrics(X,Ynew,F,changed_faces);q=mod('cross_area',QPATH);area=q.full_f32_area_gate(Ynew,X.astype(np.float64)*1000.,base['area_reference_037_points_world_m_f64'].astype(np.float64)*1000.,F,.1)
 tf=mod('cross_layer',TRANSFER);layercheck=screen.layer_check(X,Ynew,F,changed_faces,layer,tf)
 shape_pass=bool(shape['zero_area_after']==0 and shape['minimum_area_ratio']>=screen.AREA_MIN and shape['maximum_normal_turn_deg']<=screen.NORMAL_MAX and shape['minimum_min_angle_delta_deg']>=screen.MIN_ANGLE_DELTA and shape['maximum_aspect_ratio_ratio']<=screen.ASPECT_MAX)
 rep.update(status='REJECTED_SHAPE_OR_LAYER',local_correction_mm=fix.tolist(),local_correction_max_mm=float(np.linalg.norm(fix,axis=1).max(initial=0)),changed_nodes=changed.tolist(),changed_face_rows=changed_faces.tolist(),shape_metrics=shape,shape_pass=shape_pass,full_area_gate_pass=bool(area['pass']),layer_gate=layercheck)
 if shape_pass and area['pass'] and layercheck['pass']:
  audit=mod('cross_audit159',AUD159);exact=audit.exact_module();BTT,ids,_,_,MM,off,ps,_,_,_=helper.read_bank_inputs(screen.BANKDIR,screen.E)
  auddoc=json.load(open(BASE/'audit-159/report.json'));bank=next(x for x in auddoc['banks'] if int(x['body'])==43 and int(x['stable_bone_id'])==16);rawbase={(int(f),int(t)) for f,t in bank['result']['triangle_pairs']};ip=base['refinement_face_parent_ancestry_indptr'];pr=base['refinement_face_parent_ancestry_face_rows'];mappedbase=screen.map_pairs(rawbase,ip,pr)
  rawlocal,box0=screen.audit_target(X.astype(np.float64),F,changed_faces,BTT,ids,MM,off,ps,exact,audit);expect={(f,t) for f,t in rawbase if f in set(map(int,changed_faces))}
  if rawlocal!=expect:raise ValueError('baseline target local exact control failed')
  postlocal,box1=screen.audit_target(Ynew.astype(np.float64),F,changed_faces,BTT,ids,MM,off,ps,exact,audit);postraw=(rawbase-expect)|postlocal;postmap=screen.map_pairs(postraw,ip,pr)
  records,_=exact.float32_surface_records(Ynew,F,source_seam_groups=());sres=exact._audit_self_pairs_touching_faces(records,set(map(int,changed_faces)));selfpairs={(int(a),int(b)) for a,b in (sres.get('triangle_pairs') or [])}
  rep.update(status='EXACT_USEFUL_CANDIDATE_PENDING_ALL_BANK_AUDIT' if not (postraw-rawbase) and not (postmap-mappedbase) and not selfpairs and len(postmap)<len(mappedbase) else 'REJECTED_EXACT_NO_NET_IDENTITY_IMPROVEMENT',exact_checked=True,exact_target={'baseline_raw':len(rawbase),'after_raw':len(postraw),'new_raw':sorted([list(x) for x in postraw-rawbase]),'removed_raw':sorted([list(x) for x in rawbase-postraw]),'baseline_mapped':len(mappedbase),'after_mapped':len(postmap),'new_mapped':sorted([list(x) for x in postmap-mappedbase]),'removed_mapped':sorted([list(x) for x in mappedbase-postmap]),'local_aabb_before':box0,'local_aabb_after':box1},exact_self={'pairs':sorted([list(x) for x in selfpairs]),'count':len(selfpairs)})
 dest=HERE/('candidate-exact-private.npz' if rep['status']=='EXACT_USEFUL_CANDIDATE_PENDING_ALL_BANK_AUDIT' else 'candidate-exact-rejected-private.npz');payload=dict(base);payload['points_world_m']=Ynew
 with dest.open('xb') as f:np.savez_compressed(f,**payload)
 rep.update(candidate_path=str(dest),candidate_sha256=sha(dest),elapsed_s=time.monotonic()-started)
 (HERE/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:rep.get(k) for k in ['status','solver','crossing_nonregression_constraints','fat_cell_constraint','local_correction_max_mm','shape_metrics','shape_pass','layer_gate','exact_target','exact_self','candidate_path','candidate_sha256','elapsed_s']},indent=2))
if __name__=='__main__':main()
