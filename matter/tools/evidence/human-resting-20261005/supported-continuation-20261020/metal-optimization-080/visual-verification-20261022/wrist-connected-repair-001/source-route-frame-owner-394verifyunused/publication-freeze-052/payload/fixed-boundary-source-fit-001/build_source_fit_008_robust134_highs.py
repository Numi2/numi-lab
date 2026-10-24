#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
from scipy.optimize import linprog
E=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'source-route-frame-owner-394verifyunused'; OUT=D/'fixed-boundary-source-fit-001'/'run-008-134-robust-normal-highs'
SEED_CAND=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
BASE=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
EMIT=E/'source-route-frame-owner-387/current_pose_preflight.py'
IDENT=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
POSE_DIR=pathlib.Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001')
FK=('right-ring-pip-57.json','right-little-pip-61.json','left-ring-pip-95.json','left-little-pip-99.json')
CANON=E/'source-route-frame-owner-396/tools/rebind_human_tissue_reference.py'
IDS=(134,); SEEDS={131:(12677,12679),134:(63329,63331)}
ALT_FLOOR_MM=0.0005; FIT_ALT_FLOOR_MM=0.00065; WITNESS_GAP_FLOOR_MM=0.00005; CONTACTS=D/'fixed-boundary-source-fit-001'/'run-004'/'current191-new-pair-witnesses.json'; TRUST_MM=0.25; SMOOTH=2.0

def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def rotmat(q):
 x,y,z,w=map(float,q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)], [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)], [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)
def pose_map(doc,frame):
 if frame in ('rest','current'):return {int(x['body_index']):x[frame] for x in doc['all_bodies']}
 return {int(x['core_body']):{'position_world_m':x['position'],'orientation_world_xyzw':x['quaternion_xyzw']} for x in doc['source_poses']}
def load_emitter():
 src=EMIT.read_text();marker='base=emit_selected(BASE)'
 if src.count(marker)!=1:raise RuntimeError('unexpected emitter marker')
 ns={};exec(compile(src.split(marker)[0],str(EMIT),'exec'),ns);return ns
def patch_rows(F,seeds,nrings=3):
 ef={}
 for fi,t in enumerate(F):
  for a,b in ((int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))):ef.setdefault(tuple(sorted((a,b))),[]).append(fi)
 selected=set(map(int,seeds))
 for _ in range(nrings):
  add=set()
  for rows in ef.values():
   if any(r in selected for r in rows):add.update(rows)
  selected.update(add)
 rows=np.array(sorted(selected),np.int64);patch=F[rows];counts={}
 for t in patch:
  for a,b in ((int(t[0]),int(t[1])),(int(t[1]),int(t[2])),(int(t[2]),int(t[0]))):counts[tuple(sorted((int(a),int(b))))]=counts.get(tuple(sorted((int(a),int(b)))),0)+1
 bedges=[e for e,n in counts.items() if n==1];allv=set(map(int,patch.ravel()));bverts=set(v for e in bedges for v in e);free=np.array(sorted(allv-bverts),np.int64)
 return rows,patch,free,bedges
def affine_maps(dec,sid,poses):
 rec=next(r for r in dec['records'] if int(r[6])==sid);fb,bc,fv,vc,fi,ic,*_=map(int,rec);v=dec['vertices'][fv:fv+vc];B=dec['bindings'][fb:fb+bc]
 mats={};offs={}
 for pname,pose_by in poses.items():
  A=np.zeros((vc,3,3),np.float64);b=np.zeros((vc,3),np.float64)
  for i in range(vc):
   for s in range(4):
    w=float(v['w'][i,s]);bi=int(v['binding'][i,s])
    if w<=2e-6:continue
    if bi>=bc:raise RuntimeError(f'{sid}:{i}: invalid binding lane')
    bind=B[bi];body=int(bind['body'])
    if body not in pose_by:raise RuntimeError(f'{pname}: body {body} unavailable')
    T=bind['transform'];Rp=rotmat(pose_by[body]['orientation_world_xyzw']);Rb=rotmat(T[3:7]);pos=np.asarray(pose_by[body]['position_world_m'],np.float64)
    A[i]+=w*float(T[7])*(Rp@Rb);b[i]+=w*(pos+Rp@np.asarray(T[:3],np.float64))
  mats[pname]=A;offs[pname]=b
 return mats,offs,(fv,vc,fi,ic)
def get_world(A,b,p):return np.einsum('nij,nj->ni',A,p)+b
def constraint_evaluator(base_local,A_local,patch_vertices,patch_faces,free_global):
 lookup={int(v):i for i,v in enumerate(free_global)};local={int(v):i for i,v in enumerate(patch_vertices)};nvar=3*len(free_global)
 pose_data=[]
 for name in base_local:
  tri=base_local[name][patch_faces]*1000.0;u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];cr=np.cross(u,v);norm=np.linalg.norm(cr,axis=1)
  if np.any(norm==0):raise RuntimeError(f'{name}: zero-area baseline patch face')
  pose_data.append((name,tri,cr/norm[:,None],norm,A_local[name]))
 def evaluate(x,want_jac):
  d=x.reshape(-1,3);values=[];jacrows=[]
  dmap=np.zeros((len(patch_vertices),3),np.float64)
  for k,gid in enumerate(patch_vertices):
   ii=lookup.get(int(gid))
   if ii is not None:dmap[k]=d[ii]
  for name,tri0,nhat,S0,Av in pose_data:
   dworld=np.einsum('nij,nj->ni',Av,dmap)
   disp=np.asarray([[dworld[local[int(g)]] for g in face] for face in patch_faces])
   tri=tri0+disp;u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];w=tri[:,2]-tri[:,1]
   cr=np.cross(u,v);N=np.linalg.norm(cr,axis=1);Larr=np.stack([np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1),np.linalg.norm(w,axis=1)],axis=1);which=np.argmax(Larr,axis=1);L=np.maximum(Larr[np.arange(len(Larr)),which],1e-15)
   alt=N/L;signed=np.einsum('ij,ij->i',cr,nhat)
   values.extend((alt-FIT_ALT_FLOOR_MM).tolist());values.extend((signed-0.2*S0).tolist())
   if not want_jac:continue
   safeN=np.maximum(N,1e-30);nh=cr/safeN[:,None]
   gu=np.cross(v,nh);gv=np.cross(nh,u);gN=np.stack([-gu-gv,gu,gv],axis=1)
   gL=np.zeros((len(tri),3,3),np.float64)
   for ei,(ia,ib) in enumerate(((0,1),(0,2),(1,2))):
    mask=which==ei;edge=tri[mask,ib]-tri[mask,ia];eg=edge/np.maximum(np.linalg.norm(edge,axis=1)[:,None],1e-15);gL[mask,ia]-=eg;gL[mask,ib]+=eg
   galt=gN/L[:,None,None]-(N/(L*L))[:,None,None]*gL
   gs=np.stack([-np.cross(v,nhat),np.cross(v,nhat),np.cross(nhat,u)],axis=1);gs[:,0]=-gs[:,1]-gs[:,2]
   for gxyz in (galt,gs):
    M=np.zeros((len(tri),nvar),np.float64)
    for fi,face in enumerate(patch_faces):
     for corner,gid in enumerate(face):
      vi=lookup.get(int(gid))
      if vi is not None:M[fi,3*vi:3*vi+3]+=gxyz[fi,corner]@Av[local[int(gid)]]
    jacrows.extend(M)
  return np.asarray(values),np.asarray(jacrows) if want_jac else None
 evaluate.constraint_scales=np.concatenate([np.concatenate([np.full(len(tri),0.001,np.float64),np.maximum(S0,1e-12)]) for _,tri,_,S0,_ in pose_data])
 return evaluate

def main():
 t0=time.time();OUT.mkdir(parents=True,exist_ok=False)
 for p in (BASE,SEED_CAND,EMIT,IDENT,CANON,*[POSE_DIR/f for f in FK]):
  if not p.is_file():raise RuntimeError(f'missing pinned input: {p}')
 spec=importlib.util.spec_from_file_location('canon',CANON);canon=importlib.util.module_from_spec(spec);spec.loader.exec_module(canon)
 raw=BASE.read_bytes();
 if hashlib.sha256(raw).hexdigest()!='8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042':raise RuntimeError('baseline source hash mismatch')
 dec=canon.decode_nhtiss(raw);seed_raw=SEED_CAND.read_bytes();seed_dec=canon.decode_nhtiss(seed_raw);ns=load_emitter();ident=json.loads(IDENT.read_text())
 pose_specs={'current191':pose_map(ident,'current'),'rest003':pose_map(ident,'rest')}
 for fn in FK:pose_specs[fn[:-5]]=pose_map(json.loads((POSE_DIR/fn).read_text()),'fk')
 base_emitted={}
 for name,poses in pose_specs.items():ns['poses']=poses;base_emitted[name]=ns['emit_selected'](BASE)
 results=[];vertex_changes={};patch_global_vertices={};fitted_sids=set()
 for sid in IDS:
  rec=next(r for r in dec['records'] if int(r[6])==sid);fb,bc,fv,vc,fi,ic,*_=map(int,rec)
  ix=dec['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);F=ix-fv if ix.min()>=fv and ix.max()<fv+vc else ix
  if np.any(F<0) or np.any(F>=vc):raise RuntimeError(f'{sid}: invalid local face indices')
  rows,pfaces,free,bedges=patch_rows(F,SEEDS[sid],3)
  if len(free)!=6:raise RuntimeError(f'{sid}: expected six free interior nodes, got {len(free)}')
  patch_vertices=np.unique(pfaces);mats,offs,_=affine_maps(dec,sid,pose_specs);baseV={};A_local={}
  for name in pose_specs:
   A_local[name]=mats[name][patch_vertices];baseV[name]=base_emitted[name][sid][0].astype(np.float64)
   pred=get_world(mats[name],offs[name],dec['vertices'][fv:fv+vc]['p'].astype(np.float64));err=float(np.max(np.abs(pred-baseV[name])))
   if err>2e-6:raise RuntimeError(f'{sid}:{name}: affine baseline parity {err}m')
  local={int(g):i for i,g in enumerate(patch_vertices)};pfaces_local=np.asarray([[local[int(x)] for x in face] for face in pfaces],np.int64)
  free_map={int(g):i for i,g in enumerate(free)};edge_set=set()
  for face in pfaces:
   for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):edge_set.add(tuple(sorted((int(a),int(b)))))
  L=np.zeros((len(free),len(free)),np.float64)
  for a,b in edge_set:
   ia=free_map.get(a);ib=free_map.get(b)
   if ia is not None:L[ia,ia]+=1
   if ib is not None:L[ib,ib]+=1
   if ia is not None and ib is not None:L[ia,ib]-=1;L[ib,ia]-=1
  H=np.eye(len(free))+SMOOTH*L
  base_local={name:baseV[name][patch_vertices] for name in pose_specs};patch_local=np.arange(len(patch_vertices),dtype=np.int64);free_local=np.asarray([local[int(g)] for g in free],np.int64)
  # Face 63330 is below the Float32 coordinate ULP in the pinned poses. Its
  # emitted cross-product direction is unreliable in four FK poses. Construct
  # an independent area-weighted one-ring source normal and transport it by a
  # least-squares local cofactor map for this one face only.
  target_row=63330
  target_patch_rows=np.flatnonzero(rows==target_row)
  if len(target_patch_rows)!=1:raise RuntimeError(f'{sid}: robust target face row not unique')
  target_patch_ordinal=int(target_patch_rows[0])
  if target_patch_ordinal!=13:raise RuntimeError(f'{sid}: expected active face at patch ordinal 13, got {target_patch_ordinal}')
  target_face=np.asarray(pfaces[target_patch_ordinal],np.int64)
  incident=np.flatnonzero(np.any(np.isin(F,target_face),axis=1));ringverts=np.unique(F[incident].ravel())
  pref=dec['vertices'][fv:fv+vc]['p'].astype(np.float64)
  source_ring_tri=pref[F[incident]]
  source_ring_cross=np.cross(source_ring_tri[:,1]-source_ring_tri[:,0],source_ring_tri[:,2]-source_ring_tri[:,0])
  source_ring_sum=source_ring_cross.sum(axis=0)
  source_normal=source_ring_sum/max(float(np.linalg.norm(source_ring_sum)),1e-300)
  source_area_sum=float(np.linalg.norm(source_ring_cross,axis=1).sum())
  source_cancel=float(np.linalg.norm(source_ring_sum)/max(source_area_sum,1e-300))
  robust_by_pose={};robust_area_by_pose={};robust_rows=[]
  for pname in pose_specs:
   Y=get_world(mats[pname],offs[pname],pref)
   Xr=pref[ringverts];Yr=Y[ringverts];xm=Xr.mean(axis=0);ym=Yr.mean(axis=0)
   rowmap=np.linalg.lstsq(Xr-xm,Yr-ym,rcond=None)[0];Afit=rowmap.T
   detfit=float(np.linalg.det(Afit));condfit=float(np.linalg.cond(Afit))
   cof=detfit*np.linalg.inv(Afit).T@source_normal
   cof=cof/max(float(np.linalg.norm(cof)),1e-300)
   tri64=Y[target_face]*1000.0
   cross64=np.cross(tri64[1]-tri64[0],tri64[2]-tri64[0])
   robust_area=float(np.dot(cross64,cof))
   if robust_area<=0:raise RuntimeError(f'{pname}: source cofactor orientation not positive on target face ({robust_area})')
   rn=np.zeros((len(pfaces),3),np.float64);rn[target_patch_ordinal]=cof
   robust_by_pose[pname]=rn;robust_area_by_pose[pname]=robust_area
   tri_ring=Y[F[incident]]*1000.0;ring_cross=np.cross(tri_ring[:,1]-tri_ring[:,0],tri_ring[:,2]-tri_ring[:,0]);ring_n=ring_cross.sum(axis=0);ring_n/=max(float(np.linalg.norm(ring_n)),1e-300)
   robust_rows.append({'pose':pname,'normal':cof.tolist(),'local_fit_det':detfit,'local_fit_condition':condfit,'robust_baseline_oriented_area_mm2':robust_area,'cofactor_vs_fp64_one_ring_dot':float(np.dot(cof,ring_n))})
  # Wrap the existing analytic evaluator, replacing only row 13's altitude and
  # oriented-area derivatives with the independently justified fixed normal.
  evalbase=constraint_evaluator(base_local,A_local,patch_local,pfaces_local,free_local)
  nface=len(pfaces_local);free_index={int(g):i for i,g in enumerate(free)}
  def evalc(x,want_jac):
   values,jac=evalbase(x,want_jac)
   disp=x.reshape(-1,3);dmap=np.zeros((len(patch_vertices),3),np.float64)
   for loc_i,gid in enumerate(patch_vertices):
    fidx=free_index.get(int(gid))
    if fidx is not None:dmap[loc_i]=disp[fidx]
   for pidx,pname in enumerate(pose_specs):
    face_local=pfaces_local[target_patch_ordinal]
    tri=base_local[pname][face_local]*1000.0
    for corner,loc_i in enumerate(face_local):
     fidx=free_index.get(int(patch_vertices[loc_i]))
     if fidx is not None:tri[corner]+=A_local[pname][loc_i]@disp[fidx]
    u=tri[1]-tri[0];v=tri[2]-tri[0];edge_pairs=((0,1),(0,2),(1,2))
    lengths=np.asarray([np.linalg.norm(tri[b]-tri[a]) for a,b in edge_pairs])
    edge_i=int(np.argmax(lengths));L=max(float(lengths[edge_i]),1e-15)
    n=robust_by_pose[pname][target_patch_ordinal]
    signed=float(np.dot(np.cross(u,v),n));s0=robust_area_by_pose[pname]
    rid_alt=pidx*2*nface+target_patch_ordinal
    rid_area=pidx*2*nface+nface+target_patch_ordinal
    values[rid_alt]=signed/L-FIT_ALT_FLOOR_MM
    values[rid_area]=signed-0.2*s0
    if want_jac:
     g1=np.cross(v,n);g2=np.cross(n,u);gS=np.stack([-g1-g2,g1,g2])
     gL=np.zeros((3,3),np.float64);a,b=edge_pairs[edge_i];unit=(tri[b]-tri[a])/L;gL[a]-=unit;gL[b]+=unit
     gAlt=gS/L-(signed/(L*L))*gL
     for rowid,grow in ((rid_alt,gAlt),(rid_area,gS)):
      jac[rowid]=0.0
      for corner,loc_i in enumerate(face_local):
       fidx=free_index.get(int(patch_vertices[loc_i]))
       if fidx is not None:jac[rowid,3*fidx:3*fidx+3]+=grow[corner]@A_local[pname][loc_i]
   return values,jac
  evalc.constraint_scales=evalbase.constraint_scales
  xcheck=np.linspace(-1e-5,1e-5,3*len(free),dtype=np.float64)
  vv,jj=evalc(xcheck,True);fdmax=0.0;fdmax_rel=0.0
  for vi in range(len(xcheck)):
   dx=np.zeros_like(xcheck);dx[vi]=1e-7
   vp=evalc(xcheck+dx,False)[0];vm=evalc(xcheck-dx,False)[0]
   fd=(vp-vm)/(2e-7)
   fdmax=max(fdmax,float(np.max(np.abs(fd-jj[:,vi]))))
   fdmax_rel=max(fdmax_rel,float(np.max(np.abs(fd-jj[:,vi])/np.maximum(1.0,np.abs(fd)))))
  if fdmax_rel>2e-5:raise RuntimeError(f'robust signed constraint derivative failed FD: {fdmax_rel}')
  nvar=3*len(free)
  # Sequential linear feasibility in micrometres. The evaluator's x is in mm;
  # LP variables are converted by 0.001 mm/um. Every row is normalized by its
  # local Jacobian norm so the feasibility system is conditioned in micrometres.
  seedp=seed_dec['vertices'][fv:fv+vc]['p'].astype(np.float64)
  basep=dec['vertices'][fv:fv+vc]['p'].astype(np.float64)
  start=((seedp-basep)*1000.0)[free].reshape(-1)
  xcur=start.copy()
  quality_scales=evalc.constraint_scales
  nface=len(pfaces_local); npose=len(pose_specs)
  if len(quality_scales)!=2*nface*npose:raise RuntimeError('unexpected quality row count')
  def row_descriptions(raw_values, row_norm_um):
   bad=np.flatnonzero(raw_values < 0.0)
   out=[]
   for rid in bad:
    pose_i=int(rid//(2*nface)); within=int(rid%(2*nface))
    kind='altitude' if within<nface else 'oriented_double_area'
    face_i=within if within<nface else within-nface
    out.append({'row':int(rid),'pose':list(pose_specs)[pose_i],'kind':kind,
                'source_face_row':int(rows[face_i]),
                'source_vertices_local':[int(x) for x in pfaces[face_i]],
                'physical_residual':float(raw_values[rid]),
                'constraint_scale':float(quality_scales[rid]),
                'jacobian_l2_per_um':float(row_norm_um[rid]),
                'normalized_residual':float(raw_values[rid]/quality_scales[rid])})
   return out
  lp_iterations=[]; stop_reason='maximum_linearizations_reached'; phase1_records=[]
  per_step_trust_um=2.0
  for lp_iter in range(6):
   raw_values, jac_mm=evalc(xcur,True)
   scaled=raw_values/quality_scales
   jac_um=(jac_mm/quality_scales[:,None])*0.001
   rownorm=np.linalg.norm(jac_um,axis=1)
   fixed=np.flatnonzero((scaled < -1e-12) & (rownorm < 1e-14))
   if len(fixed):
    stop_reason='fixed_only_nonlinear_violations'
    lp_iterations.append({'iteration':lp_iter,'status':stop_reason,'fixed_only_rows':row_descriptions(raw_values,rownorm)})
    break
   if np.all(raw_values>=0.0):
    stop_reason='all_nonlinear_rows_satisfied'
    lp_iterations.append({'iteration':lp_iter,'status':stop_reason,'minimum_scaled_residual':float(scaled.min())})
    break
   active=rownorm>=1e-14
   Jn=jac_um[active]/rownorm[active,None]
   hn=scaled[active]/rownorm[active]
   m=len(hn)
   # L1-minimum feasible increment with a hard 2 um trust box. Variables are
   # [delta_um, abs_delta_um]. Quality rows are -J delta <= current margin.
   Aquality=np.hstack([-Jn,np.zeros((m,nvar),np.float64)])
   bquality=hn.copy()
   I=np.eye(nvar,dtype=np.float64)
   Aabs=np.vstack([np.hstack([I,-I]),np.hstack([-I,-I])])
   babs=np.zeros(2*nvar,np.float64)
   Aub=np.vstack([Aquality,Aabs]); bub=np.concatenate([bquality,babs])
   objective=np.concatenate([np.zeros(nvar,np.float64),np.ones(nvar,np.float64)])
   bounds=[(-per_step_trust_um,per_step_trust_um)]*nvar+[(0.0,per_step_trust_um)]*nvar
   lp=linprog(objective,A_ub=Aub,b_ub=bub,bounds=bounds,method='highs')
   if not lp.success:
    # Phase-I minimizes the sum of nonnegative, row-normalized feasibility
    # slacks under exactly the same trust box.
    Aphase=np.hstack([-Jn,np.zeros((m,nvar),np.float64),-np.eye(m,dtype=np.float64)])
    Aphase_abs=np.hstack([Aabs,np.zeros((2*nvar,m),np.float64)])
    phase_obj=np.concatenate([np.zeros(2*nvar,np.float64),np.ones(m,np.float64)])
    phase_bounds=bounds+[(0.0,None)]*m
    phase=linprog(phase_obj,A_ub=np.vstack([Aphase,Aphase_abs]),b_ub=np.concatenate([bquality,babs]),bounds=phase_bounds,method='highs')
    pinfo={'iteration':lp_iter,'feasibility_lp_status':int(lp.status),'feasibility_lp_message':str(lp.message),
           'phase_i_success':bool(phase.success),'phase_i_status':int(phase.status),'phase_i_message':str(phase.message),
           'phase_i_sum_normalized_slack_um':float(phase.fun) if phase.success else None,
           'phase_i_max_normalized_slack_um':float(np.max(phase.x[2*nvar:])) if phase.success else None,
           'linearized_rows':int(m),'trust_um':per_step_trust_um,
           'current_nonlinear_rows':row_descriptions(raw_values,rownorm)}
    phase1_records.append(pinfo);stop_reason='linearized_lp_infeasible_within_trust'
    lp_iterations.append({'iteration':lp_iter,'status':stop_reason,'phase_i':pinfo})
    break
   delta_um=lp.x[:nvar]
   current_merit=float(np.maximum(0.0,-scaled/np.maximum(rownorm,1e-30)).sum())
   accepted=None; line_trials=[]
   for alpha in (1.0,0.5,0.25,0.125,0.0625,0.03125,0.015625):
    trial=xcur+alpha*delta_um*0.001
    trial_raw=evalc(trial,False)[0]
    trial_scaled=trial_raw/quality_scales
    trial_merit=float(np.maximum(0.0,-trial_scaled/np.maximum(rownorm,1e-30)).sum())
    all_pass=bool(np.all(trial_raw>=0.0))
    line_trials.append({'alpha':alpha,'merit_um':trial_merit,'minimum_scaled_residual':float(trial_scaled.min()),'all_rows_pass':all_pass})
    if all_pass or trial_merit < current_merit-max(1e-12,1e-7*current_merit):
     accepted=(trial,trial_raw,trial_merit,alpha);break
   iter_record={'iteration':lp_iter,'feasibility_lp_status':int(lp.status),'feasibility_lp_message':str(lp.message),
      'objective_l1_increment_um':float(lp.fun),'step_linf_um':float(np.max(np.abs(delta_um))),
      'predicted_minimum_linearized_normalized_residual':float(np.min(scaled+((jac_mm/quality_scales[:,None])@(delta_um*0.001)))),
      'current_minimum_physical_residual':float(raw_values.min()),'current_minimum_scaled_residual':float(scaled.min()),
      'current_violation_merit_um':current_merit,'line_search':line_trials,
      'accepted_alpha':accepted[3] if accepted else None}
   lp_iterations.append(iter_record)
   if accepted is None:
    stop_reason='nonlinear_backtracking_stagnation';break
   xcur=accepted[0]
   if np.all(accepted[1]>=0.0):
    stop_reason='all_nonlinear_rows_satisfied';break
  final_raw=evalc(xcur,False)[0]
  final_scaled=final_raw/quality_scales
  final_jac_mm=evalc(xcur,True)[1]
  final_jac_um=(final_jac_mm/quality_scales[:,None])*0.001
  final_rownorm=np.linalg.norm(final_jac_um,axis=1)
  final_bad=row_descriptions(final_raw,final_rownorm)
  minslack=float(final_scaled.min())
  d=xcur.reshape(-1,3)
  lp_feasible=bool(np.all(final_raw>=0.0))
  results.append({'stable_id':sid,'fixed_only_violated_constraint_rows_at_start':[],
    'seed_faces':list(SEEDS[sid]),'patch_faces':len(rows),'patch_vertices':len(patch_vertices),
    'fixed_boundary_edges':len(bedges),'fixed_boundary_vertices':len(patch_vertices)-len(free),
    'free_vertices_local':free.tolist(),'optimizer':'scipy.optimize.linprog(method=highs)',
    'lp_feasible_nonlinear_end_state':lp_feasible,'lp_stop_reason':stop_reason,
    'linearization_count':len(lp_iterations),'linearization_history':lp_iterations,
    'phase_i_records':phase1_records,'minimum_final_normalized_constraint_residual':minslack,
    'minimum_final_physical_constraint_residual':float(final_raw.min()),
    'negative_nonlinear_rows_at_end':final_bad,
    'movement_mm':{'max':float(np.linalg.norm(d,axis=1).max()),'rms':float(np.sqrt(np.mean(np.sum(d*d,axis=1))))},
    'constraints':{'poses':list(pose_specs),'emitted_altitude_target_mm':ALT_FLOOR_MM,
      'pre_emission_fit_altitude_margin_mm':FIT_ALT_FLOOR_MM,'oriented_double_area_fraction':0.2,
      'per_step_trust_um':per_step_trust_um,'maximum_linearizations':6,
      'row_scaling':'each normalized quality row divided by its current Jacobian L2 norm per micrometre'},
    'robust_microface_normal_contract':{'source_face_row':target_row,'patch_face_ordinal':target_patch_ordinal,
      'incident_one_ring_faces':int(len(incident)),'source_area_cancellation_ratio':source_cancel,
      'pose_normals':robust_rows,'central_difference_max_abs':fdmax,'central_difference_max_relative':fdmax_rel},
    'collision_contact_row_count':0,'collision_constraints_included':False})
  if not lp_feasible:continue
  patch_global_vertices[sid]=[fv+int(g) for g in patch_vertices];fitted_sids.add(sid)
  for i,gid in enumerate(free):vertex_changes[fv+int(gid)]=d[i]*1e-3
  patch_global_vertices[sid]=[fv+int(g) for g in patch_vertices];fitted_sids.add(sid)
  for i,gid in enumerate(free):vertex_changes[fv+int(gid)]=d[i]*1e-3
 feasible=len(vertex_changes)>0
 report={'schema':'numi.human.fixed-boundary-source-position-fit.preflight.v1','status':'PARTIAL_POSITION_FIT_EMITTED_QUALITY_PENDING' if feasible and len(fitted_sids)<len(IDS) else 'POSITION_FIT_EMITTED_QUALITY_PENDING' if feasible else 'LP_NO_CANDIDATE','baseline_path':str(BASE),'baseline_sha256':sha(BASE),'builder_path':str(pathlib.Path(__file__).resolve()),'builder_sha256':sha(__file__),'emitter_sha256':sha(EMIT),'identity003_sha256':sha(IDENT),'poses':{('current191' if k=='current191' else 'rest003' if k=='rest003' else k):sha(IDENT if k in ('current191','rest003') else POSE_DIR/(k+'.json')) for k in pose_specs},'patches':results,'modified_vertex_count':len(vertex_changes),'outside_patch_payload_bytes_unchanged':True,'patch_normals_regenerated_from_area_weighted_source_faces':True,'method':'actual8874 stable-134-only quality counterfactual; fixed outer ring; six-pose altitude/oriented-area inequalities solved by sequential row-normalized HiGHS L1 feasibility in micrometres; source-face row 63330 uses independently justified one-ring source normal transported by local cofactor; collision constraints were not included','limitations':['This stable-134 quality counterfactual does not include a collision barrier and is not an admitted asset. Exact local collision screening and full tissue qualification remain separate.']}
 if feasible:
  out=bytearray(raw);vo=dec['vertex_offset']
  for gi,delta in vertex_changes.items():
   recbytes=bytearray(out[vo+gi*56:vo+(gi+1)*56]);p=np.frombuffer(recbytes,dtype='<f4',count=3,offset=0).copy();p=(p.astype(np.float64)+delta).astype(np.float32);recbytes[:12]=p.tobytes();out[vo+gi*56:vo+(gi+1)*56]=recbytes
  # Recompute area-weighted source-frame normals for every vertex in each edited patch.
  normal_updates={};normal_metrics={}
  for sid in sorted(patch_global_vertices):
   rec=next(r for r in dec['records'] if int(r[6])==sid);fv,vc,fi,ic=map(int,(rec[2],rec[3],rec[4],rec[5]));ix=dec['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);F=ix-fv if ix.min()>=fv and ix.max()<fv+vc else ix
   P=dec['vertices'][fv:fv+vc]['p'].astype(np.float64).copy()
   for gi,delta in vertex_changes.items():
    if fv<=gi<fv+vc:P[gi-fv]+=delta
   accum=np.zeros_like(P)
   tri=P[F];face_n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
   for c in range(3):np.add.at(accum,F[:,c],face_n)
   affected=np.asarray([g-fv for g in patch_global_vertices[sid]],np.int64)
   norms=np.linalg.norm(accum[affected],axis=1)
   if np.any(norms<=1e-20):raise RuntimeError(f'{sid}: undefined fitted source normal')
   new=accum[affected]/norms[:,None];old=dec['vertices'][fv+affected]['n'].astype(np.float64);old/=np.maximum(np.linalg.norm(old,axis=1)[:,None],1e-30)
   flip=np.einsum('ij,ij->i',new,old)<0;new[flip]*=-1
   dots=np.einsum('ij,ij->i',new,old)
   for local_i,nvec in zip(affected,new):
    gi=fv+int(local_i);normal_updates[gi]=nvec.astype(np.float32)
   normal_metrics[str(sid)]={'patch_vertex_count':int(len(affected)),'minimum_alignment_to_original_normal':float(dots.min()),'maximum_normal_change_degrees':float(np.degrees(np.arccos(np.clip(dots,-1,1))).max())}
  for gi,nvec in normal_updates.items():
   vo=dec['vertex_offset'];start=vo+gi*56+12;out[start:start+12]=nvec.tobytes()
  report['regenerated_normal_metrics']=normal_metrics
  cand=OUT/'candidate-8874-local-position-fit.nhtissue';cand.write_bytes(out);canon.decode_nhtiss(cand.read_bytes())
  emitted={}
  for name,poses in pose_specs.items():ns['poses']=poses;emitted[name]=ns['emit_selected'](cand)
  quality=[]
  for item in results:
   sid=item['stable_id']
   if sid not in fitted_sids:continue
   rec=next(r for r in dec['records'] if int(r[6])==sid);fv,vc,fi,ic=map(int,(rec[2],rec[3],rec[4],rec[5]));ix=dec['indices'][fi:fi+ic].astype(np.int64).reshape(-1,3);F=ix-fv if ix.min()>=fv and ix.max()<fv+vc else ix;_,pf,_,_=patch_rows(F,item['seed_faces'],3)
   for name in pose_specs:
    V=emitted[name][sid][0].astype(np.float64);tri=V[pf];u=tri[:,1]-tri[:,0];v=tri[:,2]-tri[:,0];edge=tri[:,2]-tri[:,1];cr=np.cross(u,v);N=np.linalg.norm(cr,axis=1);L=np.maximum.reduce([np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1),np.linalg.norm(edge,axis=1)]);alt=np.divide(N,L,out=np.zeros_like(N),where=L>0)*1000
    ulp=np.max(np.abs(np.spacing(V.astype(np.float32)[pf])),axis=(1,2))*1000
    target_face_local=pfaces_local[target_patch_ordinal]
    target_tri=V[patch_vertices[target_face_local]].astype(np.float64)*1000.0
    target_cross=np.cross(target_tri[1]-target_tri[0],target_tri[2]-target_tri[0])
    target_edges=np.asarray([np.linalg.norm(target_tri[1]-target_tri[0]),np.linalg.norm(target_tri[2]-target_tri[0]),np.linalg.norm(target_tri[2]-target_tri[1])])
    robust_signed_alt=float(np.dot(target_cross,robust_by_pose[name][target_patch_ordinal])/max(float(target_edges.max()),1e-15))
    quality.append({'stable_id':sid,'pose':name,'min_altitude_mm':float(alt.min()),'min_altitude_to_vertex_ulp':float(np.min(alt/np.maximum(ulp,1e-30))),'zero_area_faces':int(np.count_nonzero(N==0)),'target_source_face_row':target_row,'target_robust_signed_altitude_mm':robust_signed_alt,'target_cofactor_normal':robust_by_pose[name][target_patch_ordinal].tolist()})
  passed=all(q['min_altitude_mm']>=ALT_FLOOR_MM-1e-7 and q['zero_area_faces']==0 and q['target_robust_signed_altitude_mm']>=ALT_FLOOR_MM-1e-7 for q in quality)
  qstatus=('PARTIAL_POSITION_FIT_QUALITY_PASS_EXACT_LOCAL_GATE_PENDING' if len(fitted_sids)<len(IDS) else 'POSITION_FIT_QUALITY_PASS_EXACT_LOCAL_GATE_PENDING') if passed else 'FAIL_EMITTED_FLOAT32_QUALITY'
  report.update(candidate_patch_stable_ids=sorted(fitted_sids),candidate_path=str(cand),candidate_sha256=sha(cand),emitted_float32_quality=quality,status=qstatus,exact_collision_gate='not_run')
 outp=OUT/'source-fit-report.json';outp.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'builder_path':report['builder_path'],'builder_sha256':report['builder_sha256'],'candidate_path':report.get('candidate_path'),'candidate_sha256':report.get('candidate_sha256'),'patches':results,'emitted_float32_quality':report.get('emitted_float32_quality'),'elapsed_s':time.time()-t0},indent=2))
if __name__=='__main__':main()
