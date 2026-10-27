#!/usr/bin/env python3
"""One bounded bone-aware partial 109/114 fan fit with explicit crossing slacks."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib,importlib.util,json,sys,time,traceback
import numpy as np
from scipy.optimize import minimize
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
D=S/'per-bank-workers-20261027/b109-114/attempt-004-bone-aware-centroid-fan-001'
C=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz'
AUDIT159=C.parent/'audit-159/report.json'
PRE=D/'selective-pre-fit-patch.npz';W=D/'actual-crossing-witness-map.json';PREF=D/'selective-fit-preflight.json'
HELPER=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
AUDIT_TOOL=ROOT/'audit_skin_bone_surfaces_159.py'
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANKDIR=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
FIT_MM=3.0; SCALE_M=0.001; CLEARANCE_MM=0.05; AREA_FLOOR=0.10; MIN_ANGLE=0.5; MAX_ASPECT=50.0; MAXITER=250; SMOOTH=0.5; SLACK_PENALTY=1000.0
BASE_SHA='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
AUDIT_SHA='c480d498bc1fb0f361e817b7f785b7b425ba9f40c0e181f7176ae375103cce8e'
WITNESS_SHA='a81eef5507def4d62f36ec507f01f1cbfff4d5f0ec96627a8b780dcebe676d35'
PRE_SHA='5ecf9d3b4c4e7c9bbb00c8ab69aeb0154671c2f801e40cb5d8d1f8330392aa7a'
OUT=D/'private-bone-aware-selective-patch.npz'; REPORT=D/'selective-fit-report.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def frac(x):return Fraction.from_float(float(x))
def exact_tri(tri):return tuple(tuple(frac(x) for x in p) for p in np.asarray(tri).reshape(3,3))
def closest_point_triangle(p,a,b,c):
 ab=b-a;ac=c-a;ap=p-a;d1=float(ab@ap);d2=float(ac@ap)
 if d1<=0 and d2<=0:return a.copy()
 bp=p-b;d3=float(ab@bp);d4=float(ac@bp)
 if d3>=0 and d4<=d3:return b.copy()
 vc=d1*d4-d3*d2
 if vc<=0 and d1>=0 and d3<=0:return a+(d1/(d1-d3))*ab
 cp=p-c;d5=float(ab@cp);d6=float(ac@cp)
 if d6>=0 and d5<=d6:return c.copy()
 vb=d5*d2-d1*d6
 if vb<=0 and d2>=0 and d6<=0:return a+(d2/(d2-d6))*ac
 va=d3*d6-d5*d4
 if va<=0 and d4-d3>=0 and d5-d6>=0:return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
 den=1.0/(va+vb+vc);v=vb*den;w=vc*den
 return a+ab*v+ac*w
def tri_metrics(tris):
 T=np.asarray(tris,np.float64);areas=np.linalg.norm(np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=1)
 angles=[];aspects=[]
 for t,twice in zip(T,areas):
  lens=[float(np.linalg.norm(t[(i+1)%3]-t[i])) for i in range(3)];aa=[]
  for i in range(3):
   u=t[(i+1)%3]-t[i];v=t[(i+2)%3]-t[i]
   aa.append(float(np.degrees(np.arccos(np.clip((u@v)/max(np.linalg.norm(u)*np.linalg.norm(v),1e-300),-1,1)))))
  angles.append(min(aa));aspects.append(max(lens)**2/(2*np.sqrt(3)*max(twice,1e-300)))
 return np.asarray(angles),np.asarray(aspects),areas
def exact_records_mesh(points,faces,face_ids,vertex_ids,exact):
 pts=[tuple(frac(x) for x in p) for p in np.asarray(points).reshape(-1,3)];out=[]
 for fid,face in zip(face_ids,faces):
  ids=tuple(map(int,face));tri=tuple(pts[i] for i in ids);n=exact._cross(exact._sub(tri[1],tri[0]),exact._sub(tri[2],tri[0]))
  if not any(n):raise ValueError(f'exact degenerate emitted triangle {fid}')
  lo=tuple(min(p[k] for p in tri) for k in range(3));hi=tuple(max(p[k] for p in tri) for k in range(3))
  out.append((tri,lo,hi,int(fid),tuple(int(vertex_ids[i]) for i in ids)))
 return out
def metrics(points,faces,parents,parent_points,parent_faces):
 tri=np.asarray(points,np.float32)[faces].astype(np.float64);par=np.asarray(parent_points,np.float64)[parent_faces[parents]]
 pn=np.cross(par[:,1]-par[:,0],par[:,2]-par[:,0]);cn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
 den=np.einsum('ij,ij->i',pn,pn);ratio=np.linalg.norm(cn,axis=1)/np.sqrt(den);ori=np.einsum('ij,ij->i',pn,cn)/den
 ang,asp,_=tri_metrics(tri)
 return {'face_count':int(len(faces)),'min_parent_area_ratio':float(ratio.min()),'min_orientation_ratio':float(ori.min()),'orientation_positive':bool(np.all(ori>0)),'min_angle_degrees':float(ang.min()),'max_aspect_measure':float(asp.max())}
def main():
 started=time.monotonic();report={'schema':'numi.human.b109_114.selective_bone_aware_fan_fit.v1','status':'STARTED','source_candidate_sha256':BASE_SHA,'partial_fit':True,'physical_steps':0,'admitted':False,'max_node_trust_mm':FIT_MM,'crossing_slacks_are_explicit':True}
 try:
  if sha(C)!=BASE_SHA or sha(AUDIT159)!=AUDIT_SHA or sha(W)!=WITNESS_SHA or sha(PREF)!=PRE_SHA:raise RuntimeError('pinned inputs changed')
  Z=np.load(C,allow_pickle=False);PP=np.load(PRE,allow_pickle=False);pref=json.loads(PREF.read_text());wmap=json.loads(W.read_text())
  P=Z['points_world_m'].astype(np.float32);F=Z['source_O_faces'].astype(np.int64)
  init_world=PP['points_world_m_f32'].astype(np.float32);faces=PP['patch_faces'].astype(np.int64)
  face_parent=PP['face_parent_candidate_rows'].astype(np.int64);face_bary=PP['face_parent_barycentric_f64'].astype(np.float64)
  srcrows=PP['node_source_candidate_rows'].astype(np.int64);node_parent=PP['node_parent_candidate_face_rows'].astype(np.int64);node_bary=PP['node_parent_candidate_barycentric_f64'].astype(np.float64)
  original_parent=PP['original_patch_face_rows'].astype(np.int64);selected=set(map(int,PP['selected_crossing_parent_face_rows']))
  if len(init_world)!=267 or len(faces)!=467 or len(selected)!=95:raise RuntimeError('selective pre-fit topology inventory changed')
  helper=loadmod('selective_bone_bank_helper',HELPER);audit_tool=loadmod('selective_bone_audit_owner',AUDIT_TOOL);exact=audit_tool.exact_module()
  T,ids,gids,bodies,M,off,pos,bankkeys,bankpins,unused=helper.read_bank_inputs(BANKDIR,E)
  body,bone_id=109,114;mask=(ids[:,0]==body)&(ids[:,1]==bone_id)
  bone_raw=np.asarray(T[mask],np.float64);bone_stable=np.asarray(ids[mask,3],np.int64)
  if not len(bone_raw) or len(np.unique(bone_stable))!=len(bone_stable):raise RuntimeError('target bank triangle inventory invalid')
  bone_map={int(v):i for i,v in enumerate(bone_stable)}
  bone_normals=[];signed6=0.0;edge_uses=defaultdict(list);bone_keys=[]
  for bi,tri in enumerate(bone_raw):
   a,b,c=tri;n=np.cross(b-a,c-a);ln=float(np.linalg.norm(n))
   if not np.isfinite(ln) or ln<=0:raise RuntimeError(f'degenerate bone triangle {bone_stable[bi]}')
   bone_normals.append(n/ln);signed6+=float(a@np.cross(b,c));keys=tuple(exact.float32_point_lattice_key(v) for v in tri);bone_keys.append(keys)
   for i,j in ((0,1),(1,2),(2,0)):edge_uses[tuple(sorted((keys[i],keys[j])))].append((bi,keys[i],keys[j]))
  if any(len(v)!=2 for v in edge_uses.values()):raise RuntimeError('target bone is not a closed two-manifold')
  if any(v[0][1:]==v[1][1:] for v in edge_uses.values()):raise RuntimeError('target bone has inconsistent edge orientation')
  if abs(signed6)<1e-18:raise RuntimeError('target bone signed volume unresolved')
  outward=np.asarray(bone_normals)*(1.0 if signed6>0 else -1.0)
  bone_records=audit_tool.records(bone_raw,bone_stable,exact);prepared_bone=exact.prepare_point_location(bone_records)
  skin_local=(P.astype(np.float64)-pos[body])@M[body].T-off[body]
  init_local=(init_world.astype(np.float64)-pos[body])@M[body].T-off[body]
  source_node_to_patch={int(v):i for i,v in enumerate(srcrows) if v>=0}
  fixed_source=set(map(int,PP['fixed_boundary_source_node_rows']))|set(map(int,PP['frozen_unsafe_parent_vertex_rows']))
  fixed_nodes={source_node_to_patch[v] for v in fixed_source}
  movable=np.asarray([i for i in range(len(init_world)) if i not in fixed_nodes],np.int64)
  if len(fixed_nodes)!=75 or len(movable)!=192:raise RuntimeError(f'fixed/movable inventory changed: {len(fixed_nodes)}/{len(movable)}')
  # Use exact surface records as signed outward halfspaces; normals average across shared bone edges/corners.
  def normal_for_bary(bb,bi):
   zero=[i for i,x in enumerate(bb) if x==0];incident={bi}
   if len(zero)>=2:
    ix=next(i for i,x in enumerate(bb) if x==1);key=bone_keys[bi][ix]
    incident.update(k for k,ks in enumerate(bone_keys) if key in ks)
   elif len(zero)==1:
    ids2=[i for i in range(3) if i!=zero[0]];edge={bone_keys[bi][ids2[0]],bone_keys[bi][ids2[1]]}
    incident.update(k for k,ks in enumerate(bone_keys) if edge.issubset(set(ks)))
   nn=np.sum(outward[sorted(incident)],axis=0);ln=float(np.linalg.norm(nn))
   if ln<1e-10:raise RuntimeError('bone corner normals cancel at exact witness')
   return nn/ln
  child_by_parent=defaultdict(list)
  for oi,fi in enumerate(face_parent):
   if int(fi) in selected:child_by_parent[int(fi)].append(int(oi))
  witnesses=[x for x in wmap['witnesses'] if int(x['skin_face_row']) in selected]
  if len(witnesses)!=318:raise RuntimeError(f'selected exact witness count changed: {len(witnesses)}')
  crossing=[];mapped=[]
  for x in witnesses:
   sf=int(x['skin_face_row']);tid=int(x['bone_face_row']);bi=bone_map.get(tid)
   if bi is None:raise RuntimeError(f'missing target stable bone face {tid}')
   var=int(x['fan_child_face_index'])%3;childids=child_by_parent[sf]
   if len(childids)!=3:raise RuntimeError(f'selected parent {sf} does not have exactly three fan children')
   child=childids[var];nodes=faces[child];weights=np.asarray([float(Fraction(v)) for v in x['fan_child_weights']],np.float64)
   if abs(weights.sum()-1)>1e-9 or weights.min()<-1e-9:raise RuntimeError('invalid rational fan witness weights')
   q=np.asarray([float(Fraction(v)) for v in x['local_m']],np.float64);bb=tuple(Fraction(v) for v in x['bone_barycentrics']);n=normal_for_bary(bb,bi)
   p0=np.sum(init_local[nodes]*weights[:,None],axis=0);resid=float(np.linalg.norm(p0-q))
   if resid>2e-7:raise RuntimeError(f'exact witness-to-partial-fan mapping residual too large: {resid}')
   crossing.append((nodes,weights,q,n,'crossing',sf,tid,child));mapped.append({'pair_index':int(x['pair_index']),'root':int(x['root']),'endpoint_record_id':int(x['endpoint_record_id']),'skin_face_row':sf,'bone_face_row':tid,'child_face_row':child,'mapping_residual_m':resid})
  # Exact point-location witnesses for every local source/generated point, including explicit frozen-node reporting.
  inside=[];inside_half=[];frozen_inside=[]
  for ni,p in enumerate(init_local):
   loc=exact.point_location_prepared(tuple(frac(v) for v in p),prepared_bone)['location']
   if loc not in ('inside','boundary'):continue
   best=np.inf;bestq=None;bestbi=-1
   for bi,tri in enumerate(bone_raw):
    q=closest_point_triangle(p,*tri);d=float(np.linalg.norm(q-p))
    if d<best:best=d;bestq=q;bestbi=bi
   item={'patch_node':int(ni),'source_candidate_node_row':int(srcrows[ni]),'parent_candidate_face_row':int(node_parent[ni]),'location':loc,'nearest_bone_face_row':int(bone_stable[bestbi]),'nearest_distance_mm':best*1000.0}
   inside.append(item)
   if ni in fixed_nodes:frozen_inside.append(item)
   else:inside_half.append((np.asarray([ni]),np.asarray([1.0]),bestq,outward[bestbi],'inside',int(ni),int(bone_stable[bestbi]),-1))
  report['preflight']={'bank':[body,bone_id],'bone_triangle_count':len(bone_raw),'bone_closed_oriented':True,'local_nodes':len(init_world),'local_faces':len(faces),'selected_crossing_parent_faces':len(selected),'selected_exact_endpoint_constraints':len(crossing),'exact_witness_mapping_max_residual_m':max((x['mapping_residual_m'] for x in mapped),default=0.0),'frozen_source_node_count':len(fixed_nodes),'movable_node_count':len(movable),'inside_or_boundary_node_count':len(inside),'inside_or_boundary_nodes':inside,'frozen_inside_or_boundary_nodes':frozen_inside,'fit_source_path':str(PRE),'fit_source_sha256':sha(PRE)}
  report['preflight']['frozen_inside_policy']='preserve these fixed nodes as explicit unresolved witnesses; continue fitting movable safe-crossing support and report exact residuals'
  if not crossing:raise RuntimeError('no exact endpoint constraints selected')
  # Smooth local displacement while preserving all patch parent-area ratios and 4 mm node trust.
  edges=set()
  for tri in faces:
   for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):edges.add(tuple(sorted((int(a),int(b)))) )
  vmap={int(n):i for i,n in enumerate(movable)};nv=len(movable);L=np.zeros((nv,nv),np.float64)
  for a,b in edges:
   ia=vmap.get(a);ib=vmap.get(b)
   if ia is not None:L[ia,ia]+=1
   if ib is not None:L[ib,ib]+=1
   if ia is not None and ib is not None:L[ia,ib]-=1;L[ib,ia]-=1
  H=np.eye(nv)+SMOOTH*L;H3=np.kron(H,np.eye(3));nd=3*nv;ns=len(crossing)
  xparents=skin_local[F[face_parent]];pn=np.cross(xparents[:,1]-xparents[:,0],xparents[:,2]-xparents[:,0]);plen=np.linalg.norm(pn,axis=1)
  if np.any(plen==0):raise RuntimeError('degenerate source parent face')
  punit=pn/plen[:,None]
  def unpack(z):
   x=init_local.copy();x[movable]+=np.asarray(z[:nd]).reshape(nv,3)*SCALE_M;return x
  def objective(z):
   y=np.asarray(z[:nd]);s=np.asarray(z[nd:]);return float(.5*y@H3@y+SLACK_PENALTY*s.sum()+50.0*s@s)
  def objective_jac(z):
   y=np.asarray(z[:nd]);s=np.asarray(z[nd:]);return np.r_[H3@y,SLACK_PENALTY+100.0*s]
  def bone_values(z):
   x=unpack(z);v=[]
   for nodes,w,q,n,*_ in crossing:v.append((float(n@(np.sum(x[nodes]*w[:,None],axis=0)-q))*1000.0-CLEARANCE_MM)/1.0+float(z[nd+len(v)]))
   for nodes,w,q,n,*_ in inside_half:v.append(float(n@(np.sum(x[nodes]*w[:,None],axis=0)-q))*1000.0-CLEARANCE_MM)
   return np.asarray(v,np.float64)
  def bone_jac(z):
   J=np.zeros((ns+len(inside_half),nd+ns),np.float64)
   for r,(nodes,w,q,n,*_) in enumerate(crossing):
    for node,weight in zip(nodes,w):
     j=vmap.get(int(node))
     if j is not None:J[r,3*j:3*j+3]+=float(weight)*n
    J[r,nd+r]=1.0
   for r,(nodes,w,q,n,*_) in enumerate(inside_half,start=ns):
    for node,weight in zip(nodes,w):
     j=vmap.get(int(node))
     if j is not None:J[r,3*j:3*j+3]+=float(weight)*n
   return J
  def area_fun(z):
   x=unpack(z);out=[]
   for i,f in enumerate(faces):
    a,b,c=x[f];out.append(float(punit[i]@np.cross(b-a,c-a))/plen[i]-AREA_FLOOR)
   return np.asarray(out,np.float64)
  def area_jac(z):
   x=unpack(z);J=np.zeros((len(faces),nd+ns),np.float64)
   for i,f in enumerate(faces):
    ia,ib,ic=map(int,f);a,b,c=x[ia],x[ib],x[ic];n=punit[i];den=plen[i]
    grads=(np.cross(n,c-b),np.cross(c-a,n),np.cross(n,b-a))
    for node,g in zip((ia,ib,ic),grads):
     j=vmap.get(node)
     if j is not None:J[i,3*j:3*j+3]=SCALE_M*g/den
   return J
  def trust_fun(z):return FIT_MM**2-np.sum(np.asarray(z[:nd]).reshape(nv,3)**2,axis=1)
  def trust_jac(z):
   J=np.zeros((nv,nd+ns),np.float64);Y=np.asarray(z[:nd]).reshape(nv,3)
   for i in range(nv):J[i,3*i:3*i+3]=-2*Y[i]
   return J
  z0=np.zeros(nd+ns,np.float64);z0[nd:]=0
  half0=bone_values(z0);area0=area_fun(z0)
  report['optimizer_preflight']={'crossing_halfspaces':ns,'inside_hard_halfspaces':len(inside_half),'shape_area_halfspaces':len(faces),'trust_balls':nv,'initial_min_crossing_clearance_slack_mm':float(half0[:ns].min()),'initial_min_area_constraint':float(area0.min()),'max_iterations':MAXITER,'slack_penalty':SLACK_PENALTY}
  res=minimize(objective,z0,jac=objective_jac,method='SLSQP',constraints=[{'type':'ineq','fun':bone_values,'jac':bone_jac},{'type':'ineq','fun':area_fun,'jac':area_jac},{'type':'ineq','fun':trust_fun,'jac':trust_jac}],bounds=[(-FIT_MM,FIT_MM)]*nd+[(0.0,FIT_MM)]*ns,options={'maxiter':MAXITER,'ftol':1e-9,'disp':False})
  fit_local=unpack(res.x);fit_world=((fit_local+off[body])@M[body]+pos[body]).astype(np.float32);fit_world[list(fixed_nodes)]=init_world[list(fixed_nodes)]
  fit_path=OUT
  save={k:PP[k] for k in PP.files};save['points_world_m_f32']=fit_world;save['node_displacement_body_local_m_f64']=fit_local-init_local
  save['source_candidate_sha256']=np.asarray([BASE_SHA]);save['target_bank_body_bone']=np.asarray([body,bone_id],np.int64)
  save['selected_exact_endpoint_witness_count']=np.asarray([len(crossing)],np.int64);save['crossing_clearance_slack_mm_f64']=np.asarray(res.x[nd:],np.float64)
  np.savez_compressed(fit_path,**save)
  slack=np.asarray(res.x[nd:],np.float64)
  slack_rows=[]
  for i,s in enumerate(slack):
   if s>1e-7:
    m=mapped[i];slack_rows.append({'pair_index':m['pair_index'],'root':m['root'],'endpoint_record_id':m['endpoint_record_id'],'skin_parent_face_row':m['skin_face_row'],'bone_face_row':m['bone_face_row'],'slack_mm':float(s)})
  disp=np.linalg.norm(fit_local-init_local,axis=1)*1000.0
  report['optimizer']={'success':bool(res.success),'message':str(res.message),'iterations':int(res.nit),'objective':float(res.fun),'movable_node_count':nv,'fixed_node_count':len(fixed_nodes),'max_displacement_mm':float(disp.max()),'rms_displacement_mm':float(np.sqrt(np.mean(disp**2))),'max_component_displacement_mm':float(np.max(np.abs(fit_local-init_local))*1000),'crossing_constraint_count':ns,'positive_crossing_slack_count':int(np.count_nonzero(slack>1e-7)),'max_crossing_slack_mm':float(slack.max(initial=0)),'crossing_slack_records':slack_rows,'minimum_hard_inside_margin_mm':float(bone_values(res.x)[ns:].min()) if len(inside_half) else None,'minimum_area_constraint':float(area_fun(res.x).min()),'minimum_trust_slack_mm2':float(trust_fun(res.x).min()),'private_patch_path':str(fit_path),'private_patch_sha256':sha(fit_path)}
  if not res.success:
   report['status']='PRIVATE_REJECTED_SLSQP';raise RuntimeError(f'SLSQP failed: {res.message}')
  if disp.max()>FIT_MM+1e-5 or trust_fun(res.x).min()<-1e-6:
   report['status']='PRIVATE_REJECTED_TRUST';raise RuntimeError('max node displacement exceeds 4 mm construction trust')
  if bone_values(res.x)[ns:].min(initial=0)<-1e-6:
   report['status']='PRIVATE_REJECTED_INSIDE_HALFSPACE';raise RuntimeError('hard inside-node halfspace infeasible')
  # Shape gate: generated faces meet fixed quality floors; every affected source face is non-worse than its own baseline.
  emitted_local=(fit_world.astype(np.float64)-pos[body])@M[body].T-off[body]
  newq=metrics(emitted_local,faces,face_parent,skin_local,F)
  fanmask=np.isin(face_parent,np.asarray(sorted(selected),np.int64));fanq=metrics(emitted_local,faces[fanmask],face_parent[fanmask],skin_local,F)
  src_rows_move=set(int(srcrows[i]) for i in movable if srcrows[i]>=0)
  affected_global=np.flatnonzero(np.any(np.isin(F,np.asarray(sorted(src_rows_move),np.int64)),axis=1))
  base_ang,base_asp,base_area=tri_metrics(P[F[affected_global]].astype(np.float64))
  final_global= P.copy();
  for i,sr in enumerate(srcrows):
   if sr>=0:final_global[int(sr)]=fit_world[i]
  final_ang,final_asp,final_area=tri_metrics(final_global[F[affected_global]].astype(np.float64))
  ori=np.einsum('ij,ij->i',np.cross(P[F[affected_global,1]]-P[F[affected_global,0]],P[F[affected_global,2]]-P[F[affected_global,0]]),np.cross(final_global[F[affected_global,1]]-final_global[F[affected_global,0]],final_global[F[affected_global,2]]-final_global[F[affected_global,0]]))
  area_ratio=final_area/np.maximum(base_area,1e-300);angle_tol=1e-5;aspect_tol=np.maximum(1e-6,base_asp*1e-6)
  worsened=(final_ang+angle_tol<base_ang)|(final_asp>base_asp+aspect_tol)|(area_ratio<0.10)|(ori<=0)
  worse_rows=[{'source_face_row':int(fi),'baseline_min_angle_deg':float(ba),'emitted_min_angle_deg':float(fa),'baseline_aspect':float(basp),'emitted_aspect':float(fasp),'area_ratio':float(ar),'orientation_dot':float(o)} for fi,ba,fa,basp,fasp,ar,o in zip(affected_global[ worsened],base_ang[worsened],final_ang[worsened],base_asp[worsened],final_asp[worsened],area_ratio[worsened],ori[worsened])]
  fanpass=fanq['orientation_positive'] and fanq['min_parent_area_ratio']>=AREA_FLOOR and fanq['min_angle_degrees']>=MIN_ANGLE and fanq['max_aspect_measure']<=MAX_ASPECT
  no_worse=not len(worse_rows)
  shape_pass=fanpass and no_worse and newq['orientation_positive']
  report['shape_gate']={'pass':bool(shape_pass),'generated_fan':fanq,'generated_fan_thresholds':{'min_angle_deg':MIN_ANGLE,'max_aspect':MAX_ASPECT,'min_parent_area_ratio':AREA_FLOOR},'affected_global_source_face_count':len(affected_global),'nonworsening_tolerances':{'min_angle_degrees':angle_tol,'aspect_relative':1e-6,'min_area_ratio':0.10},'minimum_affected_angle_delta_deg':float(np.min(final_ang-base_ang)),'maximum_affected_aspect_ratio_to_baseline':float(np.max(final_asp/np.maximum(base_asp,1e-300))),'minimum_affected_area_ratio':float(np.min(area_ratio)),'worsened_face_count':len(worse_rows),'worsened_faces':worse_rows[:256],'local_mixed_patch_shape':newq}
  if not shape_pass:
   report['status']='PRIVATE_REJECTED_SHAPE_NO_DETERIORATION';raise RuntimeError('shape-first gate failed; exact geometry screens not run')
  # Exact target-bone gate on all local replacement children, then exact point location.
  skin_records=audit_tool.records(emitted_local[faces],np.arange(len(faces),dtype=np.int64),exact)
  bone_pairs=exact._audit_pair(skin_records,bone_records,same_surface=False)
  loc_counts=Counter(exact.point_location_prepared(tuple(frac(x) for x in p),prepared_bone)['location'] for p in emitted_local)
  pair_items=[]
  for a,b in bone_pairs['triangle_pairs']:
   pair_items.append({'patch_face_row':int(a),'skin_parent_face_row':int(face_parent[int(a)]),'bone_face_row':int(b)})
  base_a=json.loads(AUDIT159.read_text());bank=next(x for x in base_a['banks'] if (int(x['body']),int(x['stable_bone_id']))==(body,bone_id))['result']
  baseline_pairs={tuple(map(int,x)) for x in bank.get('triangle_pairs',[])}
  emitted_pairs={(x['skin_parent_face_row'],x['bone_face_row']) for x in pair_items}
  report['exact_local_bone_gate']={'bank':[body,bone_id],'pair_count':int(bone_pairs['count']),'pairs':pair_items,'patch_vertex_location_counts':dict(loc_counts),'all_patch_vertices_outside':loc_counts.get('inside',0)==0 and loc_counts.get('boundary',0)==0,'baseline_pair_count':len(baseline_pairs),'emitted_parent_bone_pair_count':len(emitted_pairs),'cleared_baseline_parent_bone_pairs':[list(x) for x in sorted(baseline_pairs-emitted_pairs)],'remaining_baseline_parent_bone_pairs':[list(x) for x in sorted(baseline_pairs&emitted_pairs)],'new_parent_bone_pairs':[list(x) for x in sorted(emitted_pairs-baseline_pairs)],'selected_parent_residual_pair_count':sum(1 for p in emitted_pairs if p[0] in selected),'pass':int(bone_pairs['count'])==0 and loc_counts.get('inside',0)==0 and loc_counts.get('boundary',0)==0}
  # Exact local self screen against the patch and all candidate complement faces, including moved source vertices.
  patch_vertex_ids=[int(v) if v>=0 else int(len(P)+i) for i,v in enumerate(srcrows)]
  patch_records=exact_records_mesh(emitted_local,faces,np.arange(len(faces)),patch_vertex_ids,exact)
  self_internal=exact._audit_pair(patch_records,patch_records,same_surface=True)
  patch_rows=set(map(int,original_parent));comp_rows=np.asarray([i for i in range(len(F)) if i not in patch_rows],np.int64)
  global_local=(final_global.astype(np.float64)-pos[body])@M[body].T-off[body]
  tpatch=emitted_local[faces];lo=tpatch.min(axis=(0,1));hi=tpatch.max(axis=(0,1));ctri=global_local[F[comp_rows]];clo=ctri.min(axis=1);chi=ctri.max(axis=1)
  active=np.flatnonzero(np.all(chi>=lo,axis=1)&np.all(clo<=hi,axis=1));comp_ids=comp_rows[active]
  comp_records=exact_records_mesh(global_local,F[comp_ids],len(faces)+comp_ids,np.arange(len(P),dtype=np.int64),exact)
  self_comp=exact._audit_pair(patch_records,comp_records,same_surface=True)
  report['exact_local_self_gate']={'patch_internal_pair_count':int(self_internal['count']),'patch_internal_pairs':self_internal['triangle_pairs'],'complement_aabb_active_count':int(len(active)),'patch_vs_complement_pair_count':int(self_comp['count']),'patch_vs_complement_pairs':self_comp['triangle_pairs'],'pass':int(self_internal['count'])==0 and int(self_comp['count'])==0}
  allpass=report['exact_local_bone_gate']['pass'] and report['exact_local_self_gate']['pass']
  report['status']='PRIVATE_PARTIAL_FIT_LOCAL_GATES_PASS' if allpass else 'PRIVATE_REJECTED_EXACT_LOCAL_GATES'
  report['accepted_candidate_modified']=False;report['geometry_admitted']=False
 except Exception as e:
  if report.get('status')=='STARTED':report['status']='PRIVATE_SELECTIVE_FIT_ABORTED'
  report['failure']=repr(e);report['traceback']=traceback.format_exc()
 report['elapsed_seconds']=time.monotonic()-started;report['script_path']=str(Path(__file__));report['script_sha256']=sha(Path(__file__))
 REPORT.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('traceback',)},indent=2,allow_nan=False))
if __name__=='__main__':main()
