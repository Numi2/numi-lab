#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
from scipy.optimize import minimize
E=pathlib.Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'source-route-frame-owner-394verifyunused'; OUT=D/'fixed-boundary-source-fit-001'/'run-003'
BASE=pathlib.Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/muscle158-precision-002/composed-receipt/composed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
EMIT=E/'source-route-frame-owner-387/current_pose_preflight.py'
IDENT=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
POSE_DIR=pathlib.Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001')
FK=('right-ring-pip-57.json','right-little-pip-61.json','left-ring-pip-95.json','left-little-pip-99.json')
CANON=E/'source-route-frame-owner-396/tools/rebind_human_tissue_reference.py'
IDS=(131,134); SEEDS={131:(12677,12679),134:(63329,63331)}
ALT_FLOOR_MM=0.0005; TRUST_MM=0.25; SMOOTH=2.0

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
   values.extend((alt-ALT_FLOOR_MM).tolist());values.extend((signed-0.2*S0).tolist())
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
 return evaluate

def main():
 t0=time.time();OUT.mkdir(parents=True,exist_ok=False)
 for p in (BASE,EMIT,IDENT,CANON,*[POSE_DIR/f for f in FK]):
  if not p.is_file():raise RuntimeError(f'missing pinned input: {p}')
 spec=importlib.util.spec_from_file_location('canon',CANON);canon=importlib.util.module_from_spec(spec);spec.loader.exec_module(canon)
 raw=BASE.read_bytes();dec=canon.decode_nhtiss(raw);ns=load_emitter();ident=json.loads(IDENT.read_text())
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
  base_local={name:baseV[name][patch_vertices] for name in pose_specs};patch_local=np.arange(len(patch_vertices),dtype=np.int64);free_local=np.asarray([local[int(g)] for g in free],np.int64);evalc=constraint_evaluator(base_local,A_local,patch_local,pfaces_local,free_local)
  nvar=3*len(free)
  def fun(x):d=x.reshape(-1,3);return .5*float(np.sum(d*(H@d)))
  def jacfun(x):return (H@x.reshape(-1,3)).reshape(-1)
  def cons(x):return evalc(x,False)[0]
  def consjac(x):return evalc(x,True)[1]
  initial_values,initial_jac=evalc(np.zeros(nvar),True)
  fixed_bad=np.flatnonzero((initial_values < -1e-12) & (np.linalg.norm(initial_jac,axis=1)<1e-10)).astype(int).tolist()
  opt=minimize(fun,np.zeros(nvar),jac=jacfun,method='SLSQP',bounds=[(-TRUST_MM,TRUST_MM)]*nvar,constraints=[{'type':'ineq','fun':cons,'jac':consjac}],options={'maxiter':300,'ftol':1e-12,'disp':False})
  cv=cons(opt.x);minslack=float(cv.min());d=opt.x.reshape(-1,3)
  results.append({'stable_id':sid,'fixed_only_violated_constraint_rows_at_start':fixed_bad,'seed_faces':list(SEEDS[sid]),'patch_faces':len(rows),'patch_vertices':len(patch_vertices),'fixed_boundary_edges':len(bedges),'fixed_boundary_vertices':len(patch_vertices)-len(free),'free_vertices_local':free.tolist(),'optimizer_success':bool(opt.success),'optimizer_message':str(opt.message),'iterations':int(opt.nit),'objective_mm2':float(opt.fun),'minimum_linearized_constraint_slack':minslack,'movement_mm':{'max':float(np.linalg.norm(d,axis=1).max()),'rms':float(np.sqrt(np.mean(np.sum(d*d,axis=1))))},'constraints':{'poses':list(pose_specs),'altitude_floor_mm':ALT_FLOOR_MM,'oriented_double_area_fraction':0.2,'trust_mm':TRUST_MM}})
  if minslack< -1e-7:continue
  patch_global_vertices[sid]=[fv+int(g) for g in patch_vertices];fitted_sids.add(sid)
  for i,gid in enumerate(free):vertex_changes[fv+int(gid)]=d[i]*1e-3
 feasible=len(vertex_changes)>0
 report={'schema':'numi.human.fixed-boundary-source-position-fit.preflight.v1','status':'PARTIAL_POSITION_FIT_EMITTED_QUALITY_PENDING' if feasible and len(fitted_sids)<len(IDS) else 'POSITION_FIT_EMITTED_QUALITY_PENDING' if feasible else 'SOURCE_FIT_INFEASIBLE_NO_CANDIDATE','baseline_path':str(BASE),'baseline_sha256':sha(BASE),'builder_path':str(pathlib.Path(__file__).resolve()),'builder_sha256':sha(__file__),'emitter_sha256':sha(EMIT),'identity003_sha256':sha(IDENT),'poses':{('current191' if k=='current191' else 'rest003' if k=='rest003' else k):sha(IDENT if k in ('current191','rest003') else POSE_DIR/(k+'.json')) for k in pose_specs},'patches':results,'modified_vertex_count':len(vertex_changes),'outside_patch_payload_bytes_unchanged':True,'patch_normals_regenerated_from_area_weighted_source_faces':True,'method':'actual8874 source-space smooth minimum-displacement fit on connected three-ring face patches, fixed outer ring, joint six-pose altitude/oriented-area inequalities','limitations':['The solver stage constrains six-pose geometry quality only; it does not impose collision constraints. Candidate is not admitted until exact local collision checks pass.']}
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
    quality.append({'stable_id':sid,'pose':name,'min_altitude_mm':float(alt.min()),'min_altitude_to_vertex_ulp':float(np.min(alt/np.maximum(ulp,1e-30))),'zero_area_faces':int(np.count_nonzero(N==0))})
  passed=all(q['min_altitude_mm']>=ALT_FLOOR_MM-1e-7 and q['zero_area_faces']==0 for q in quality)
  qstatus=('PARTIAL_POSITION_FIT_QUALITY_PASS_EXACT_LOCAL_GATE_PENDING' if len(fitted_sids)<len(IDS) else 'POSITION_FIT_QUALITY_PASS_EXACT_LOCAL_GATE_PENDING') if passed else 'FAIL_EMITTED_FLOAT32_QUALITY'
  report.update(candidate_patch_stable_ids=sorted(fitted_sids),candidate_path=str(cand),candidate_sha256=sha(cand),emitted_float32_quality=quality,status=qstatus,exact_collision_gate='not_run')
 outp=OUT/'source-fit-report.json';outp.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'builder_path':report['builder_path'],'builder_sha256':report['builder_sha256'],'candidate_path':report.get('candidate_path'),'candidate_sha256':report.get('candidate_sha256'),'patches':results,'emitted_float32_quality':report.get('emitted_float32_quality'),'elapsed_s':time.time()-t0},indent=2))
if __name__=='__main__':main()
