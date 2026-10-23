#!/usr/bin/env python3
"""Separate geometric retarget proposal; no active payload or header writes."""
import csv,hashlib,json,struct,sys
from collections import Counter
from pathlib import Path
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=E/'capitate-attachment-correspondence-001/geometric-inferred-retarget-003'
C=E/'joint-bmap-refinement-candidate-011-mechanical119-integration-001/source-C-reference-offset-diagnostic-014-exact-f32-residual-001'
CORR=E/'capitate-attachment-correspondence-001/capitate-attachment-correspondence.json'
FRAME=E/'capitate-attachment-correspondence-001/coordinate-frame-review-001/coordinate-frame-review.json'
REG=E/'native-pose-checks/connected-hand-registration069i/registered-skin.csv'
PAY=E/'finger-profile-069i-left-ring-web-retreat/replacement-bilateral-hands038.nhskin'
BIND=E/'arm-skin-weight-audit.json'
BM=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4/bodyparts3d-myosim-major-bones.manifest.json')
BP=BM.parent/'bodyparts3d-myosim-major-bones.nhbones'
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src');sys.path.insert(0,str(E/'source-exact-hand-bone-clearance-001'))
from numilab_human.skin_source_payload_preflight import decode_payload
from bone_point_queries import TrianglePointQuery
RADII=(.002,.005,.010,.020); BOUND=2e-5
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def family(n,s):
 n=n.lower()
 return n.endswith('_r' if s=='right' else '_l') and ('mc_' in n or 'metacarp' in n or any(x in n for x in ('capitate','scaphoid','lunate','triquetrum','pisiform','hamate','trapezium','trapezoid')))
def closest_all(p,T):
 a,b,c=T[:,0],T[:,1],T[:,2];u=b-a;v=c-a;w=p-a
 d00=np.einsum('ij,ij->i',u,u);d01=np.einsum('ij,ij->i',u,v);d11=np.einsum('ij,ij->i',v,v);d20=np.einsum('ij,ij->i',w,u);d21=np.einsum('ij,ij->i',w,v);den=d00*d11-d01*d01;ok=den>1e-28
 x=np.divide(d11*d20-d01*d21,den,out=np.zeros_like(den),where=ok);y=np.divide(d00*d21-d01*d20,den,out=np.zeros_like(den),where=ok);z=1-x-y
 B=np.stack((z,x,y),1);Q=a+x[:,None]*u+y[:,None]*v;d2=np.einsum('ij,ij->i',Q-p,Q-p);inside=ok&(z>=-1e-12)&(x>=-1e-12)&(y>=-1e-12);d2=np.where(inside,d2,np.inf);B=np.where(inside[:,None],B,np.nan)
 for i,j in ((0,1),(1,2),(2,0)):
  e=T[:,j]-T[:,i];l=np.einsum('ij,ij->i',e,e);t=np.clip(np.divide(np.einsum('ij,ij->i',p-T[:,i],e),l,out=np.zeros_like(l),where=l>1e-28),0,1);q=T[:,i]+t[:,None]*e;d=np.einsum('ij,ij->i',q-p,q-p);bb=np.zeros_like(B);bb[:,i]=1-t;bb[:,j]=t;take=d<d2;d2=np.where(take,d,d2);B=np.where(take[:,None],bb,B)
 return np.sqrt(d2),B
def frames_from_csv(rows):
 out={}
 for b in set(int(r['donor']) for r in rows):
  rr=[r for r in rows if int(r['donor'])==b];M=np.array([[[float(r[f'm{i}{j}']) for j in range(3)] for i in range(3)] for r in rr]);w=np.array([[float(r[k]) for k in ('wx','wy','wz')] for r in rr]);l=np.array([[float(r[k]) for k in ('lx','ly','lz')] for r in rr]);t=l-np.einsum('nij,nj->ni',M,w)
  ms=float(np.max(abs(M-M[0])));ts=float(np.max(abs(t-t[0])))
  if ms>1e-6 or ts>2e-5:raise RuntimeError(f'inconsistent frame donor {b}: {ms} {ts}')
  out[b]=(M[0],t[0],len(rr),ms,ts)
 return out
def parity(p,T,d):
 d=np.asarray(d,float);d/=np.linalg.norm(d);a=T[:,0];e1=T[:,1]-a;e2=T[:,2]-a;h=np.cross(np.broadcast_to(d,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h);good=abs(det)>1e-13;inv=np.divide(1.,det,out=np.zeros_like(det),where=good);s=p-a;u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1);v=inv*np.einsum('ij,ij->i',np.broadcast_to(d,q.shape),q);t=inv*np.einsum('ij,ij->i',e2,q)
 return int(np.count_nonzero(good&(t>1e-8)&(u>1e-8)&(v>1e-8)&(u+v<1-1e-8)))
def winding(p,T):
 a=T[:,0]-p;b=T[:,1]-p;c=T[:,2]-p;la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1);lc=np.linalg.norm(c,axis=1);det=np.einsum('ij,ij->i',a,np.cross(b,c));den=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
 return float(abs(np.sum(2*np.arctan2(det,den))/(4*np.pi)))
def load_bones(frames):
 m=json.loads(BM.read_text());raw=BP.read_bytes();head=struct.calcsize('<8s5I32s');magic,abi,bc,vc,ic,fp,source=struct.unpack_from('<8s5I32s',raw)
 if abi!=3 or source.hex()!=m['source']['myosim_source_archive_sha256']:raise RuntimeError('bone payload/source hash mismatch')
 rs=struct.calcsize('<6I8fI');records=[struct.unpack_from('<6I8fI',raw,head+i*rs) for i in range(bc)];off=head+bc*rs;bv=np.asarray([struct.unpack_from('<6f',raw,off+i*24)[:3] for i in range(vc)],np.float32);off+=vc*24;bi=np.frombuffer(raw,dtype='<u4',count=ic,offset=off).reshape(-1,3);out={}
 for r,a in zip(records,m['source']['anchors']):
  body=int(a['core_body_index']);name=str(a['myosim_body']).lower();side='right' if name.endswith('_r') else 'left'
  if body not in frames or not family(name,side):continue
  bi0,fv,nv,fi,ni,st,tx,ty,tz,qx,qy,qz,qw,scale,sr=r;qq=np.array((qx,qy,qz,qw),float);qq/=np.linalg.norm(qq);x,y,z,w=qq
  R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
  local=(bv[fv:fv+nv].astype(float)*scale)@R.T+np.array((tx,ty,tz));M,t=frames[body][:2];world=((local-t)@M).astype(np.float32).astype(float);f=bi[fi//3:(fi+ni)//3].astype(np.int64)-fv;T=world[f]
  ed=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1);ec=Counter(map(tuple,ed.tolist()))
  out[body]={'name':name,'T':T,'query':TrianglePointQuery(T),'lo':T.min(axis=(0,1)),'hi':T.max(axis=(0,1)),'nt':len(T),'boundary':int(sum(v==1 for v in ec.values())),'frame_rows':frames[body][2]}
 return out,m,source.hex()
def solid_witness(mesh,p,d):
 if d<=BOUND:return {'classification':'near_bone_surface','distance_mm':d*1000}
 if np.any(p<mesh['lo']) or np.any(p>mesh['hi']):return {'classification':'outside_bone_AABB','distance_mm':d*1000}
 T=mesh['T'];counts=[parity(p,T,x) for x in ((1,.3713907,.529113),(.217391,1,.618034),(.57721,.271828,1))];par=[x%2 for x in counts];wn=winding(p,T)
 if all(x==1 for x in par) and wn>=.55:c='inside_open_mesh_consensus'
 elif all(x==0 for x in par) and wn<=.12:c='outside_open_mesh_consensus'
 else:c='indeterminate_open_mesh'
 return {'classification':c,'distance_mm':d*1000,'ray_counts':counts,'ray_parity':par,'abs_winding':wn}
def main():
 corr=json.loads(CORR.read_text());frev=json.loads(FRAME.read_text());csvrows=list(csv.DictReader(REG.open()));frames=frames_from_csv(csvrows)
 for b in (49,99):
  ex=frev['body49_99'][str(b)]
  if np.max(abs(frames[b][0]-np.asarray(ex['M'])))>1e-8 or np.max(abs(frames[b][1]-np.asarray(ex['t'])))>1e-7:raise RuntimeError('capitate donor frame mismatch')
 names={int(x['i']):str(x['name']) for x in json.loads(BIND.read_text())['bindings']};payload=decode_payload(PAY.read_bytes());bindids=np.asarray(payload['bindings_u'][:,0],int)
 with np.load(C/'coarse-surface.npz',allow_pickle=False) as z:F=z['faces'].astype(int);parent=z['source_face_parent'].astype(int);kind=z['source_face_kind'].astype(int);inf=z['source_face_parent_inferred_mask'].astype(bool);ap=z['source_face_parent_ancestry_indptr'].astype(int);ai=z['source_face_parent_ancestry_ids'].astype(int)
 with np.load(C/'candidate-layers-C014.npz',allow_pickle=False) as z:D=z['deep_world'].astype(float);N=z['outward_normals'].astype(float)
 with np.load(C/'coarse-to-source-C014.npz',allow_pickle=False) as z:W=z['expanded_source_weights86'].astype(float)
 if F.shape!=(17595,3) or D.shape!=(8820,3):raise RuntimeError('unexpected C014 shape')
 owner=np.asarray([names[int(bindids[np.argmax(W[i])])] for i in range(len(D))],object);fown=[[owner[v] for v in tri] for tri in F];T=D[F]
 area=np.linalg.norm(np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=1)*.5;fn=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);fn/=np.maximum(np.linalg.norm(fn,axis=1)[:,None],1e-30);mn=N[F].mean(axis=1);mn/=np.maximum(np.linalg.norm(mn,axis=1,keepdims=True),1e-30);nd=np.einsum('ij,ij->i',fn,mn)
 exact=(~inf)&((ap[1:]-ap[:-1])==1)&(parent==ai[np.minimum(ap[:-1],len(ai)-1)])&(area>1e-12);bones,bmeta,bsha=load_bones(frames);out_sites=[]
 for s in corr['sites']:
  si=int(s['legacy_site_index'])
  if si not in (2,3,9,10,16,17):continue
  donor=int(s['donor_body']);side='right' if donor==49 else 'left';suffix='_r' if side=='right' else '_l';target=np.asarray(s['target_world_in_069i_frame_m']);M,t=frames[donor][:2];d,bary=closest_all(target,T)
  same_region=np.asarray([all(family(n,side) for n in a) for a in fown]);eligible_base=exact&same_region;eligible=eligible_base&(nd>=.90);allids=np.flatnonzero(eligible);nearestall=int(allids[np.argmin(d[allids])]) if len(allids) else None;regionids=np.flatnonzero(eligible_base);nearestregion=int(regionids[np.argmin(d[regionids])]) if len(regionids) else None;pool=np.flatnonzero(eligible&(d<=.020+1e-12));pool=pool[np.argsort(d[pool],kind='stable')];col=int(np.flatnonzero(bindids==donor)[0]);chosen=None;attempt=[]
  for fi0 in pool:
   fi=int(fi0);q=bary[fi]@T[fi];wi=[];inside=[];uncertain=[];surface=[]
   for body,m in bones.items():
    if not m['name'].endswith(suffix):continue
    db,qb,it,ncan=m['query'].query(q);solid=solid_witness(m,q,db); tri=m['T'][it];bn=np.cross(tri[1]-tri[0],tri[2]-tri[0]);bn/=max(np.linalg.norm(bn),1e-30)
    wi.append({'body':body,'name':m['name'],'distance_mm':db*1000,'closest_bone_point_world_m':np.asarray(qb).tolist(),'triangle':it,'bone_normal':bn.tolist(),'D_minus_bone_dot_authored_outward_mm':float(np.dot(q-qb,mn[fi])*1000),'solid_witness':solid})
    c=solid['classification']
    if c=='inside_open_mesh_consensus':inside.append(m['name'])
    if c=='indeterminate_open_mesh':uncertain.append(m['name'])
    if c=='near_bone_surface':surface.append(m['name'])
   wi.sort(key=lambda x:x['distance_mm']);owned=[x for x in wi if x['name'] in set(fown[fi])];ow=min(owned,key=lambda x:x['distance_mm']) if owned else None;capname='capitate_r' if side=='right' else 'capitate_l';cap=next((x for x in wi if x['name']==capname),None);status='proposal_candidate';why=[]
   if inside:status='rejected_inside_bone';why.append('inside consensus against an available same-side bone mesh')
   if uncertain:status='rejected_containment_indeterminate';why.append('within bone AABB but open-mesh containment indeterminate')
   if surface:status='rejected_bone_surface_coincidence';why.append('within numeric distance tolerance of a bone face')
   if ow and ow['D_minus_bone_dot_authored_outward_mm'] < -BOUND*1000:status='rejected_normal_points_into_bone';why.append('owner-bone to D vector projects inward on authored outward normal')
   rec={'face':fi,'nodes':F[fi].tolist(),'barycentric':bary[fi].tolist(),'point_world_m':q.tolist(),'displacement_m':float(d[fi]),'source_face_parent':int(parent[fi]),'source_face_kind':int(kind[fi]),'source_face_parent_inferred':bool(inf[fi]),'source_parent_ancestry_ids':ai[ap[fi]:ap[fi+1]].tolist(),'corner_dominant_bone_names':fown[fi],'corner_dominant_bone_ids':[next((int(k) for k,v in names.items() if v==n),-1) for n in fown[fi]],'triangle_area_m2':float(area[fi]),'triangle_normal':fn[fi].tolist(),'authored_outward_normal':mn[fi].tolist(),'normal_dot':float(nd[fi]),'capitate_skin_weight_context_only':float(bary[fi]@W[F[fi],col]),'bone_landmark_witnesses':wi,'nearest_owner_bone_witness':ow,'capitate_bone_witness':cap,'status':status,'rejection_reasons':why};attempt.append(rec)
   if status=='proposal_candidate':chosen=rec;break
  ladder=[]
  for radius in RADII:
   ids=np.flatnonzero(eligible&(d<=radius+1e-12));nearest=int(ids[np.argmin(d[ids])]) if len(ids) else None;prop=chosen if chosen and chosen['displacement_m']<=radius+1e-12 else None;pd=None
   if prop:
    q=np.asarray(prop['point_world_m']);nl=M@q+t;pd={'new_stable_id':'whole_body_skin_deep_attachment:new-inferred-C014-site'+str(si)+'-donor'+str(donor)+'-face'+str(prop['face']),'migrates_from_stable_id':s['stable_id'],'historical_source_vertex_id_not_identity':int(s['source_vertex']),'source_identity_recovery_claim':False,'donor_body_preserved':donor,'donor_name_preserved':capname,'old_zero_strain_target_world_m':target.tolist(),'old_donor_local_target_m':s['original_donor_local_target_m'],'new_zero_strain_target_world_m':q.tolist(),'new_donor_local_target_m':nl.tolist(),'local_roundtrip_error_m':float(np.linalg.norm(M.T@(nl-t)-q)),'displacement_mm':float(np.linalg.norm(q-target)*1000),'original_kA_N_per_m_preserved':float(s['original_kA_N_per_m']),'original_area_m2_preserved':float(s['baseline_tributary_area_m2']),'surface_binding':{k:prop[k] for k in ('face','nodes','barycentric','source_face_parent','source_face_kind','source_face_parent_inferred','source_parent_ancestry_ids')},'anatomical_witness':{k:prop[k] for k in ('corner_dominant_bone_names','corner_dominant_bone_ids','triangle_normal','authored_outward_normal','normal_dot','capitate_skin_weight_context_only','bone_landmark_witnesses','nearest_owner_bone_witness','capitate_bone_witness')},'status':'new reference inference only; requires compiler and physical validation'}
   ladder.append({'max_displacement_mm':radius*1000,'nearest_raw_exact_same_side_face':nearest,'nearest_raw_displacement_mm':None if nearest is None else float(d[nearest]*1000),'proposal':pd,'status':'proposal inside displacement limit' if pd else ('no exact same-side region face in radius' if nearest is None else 'no bone-screened proposal inside radius')})
  raw=int(pool[0]) if len(pool) else None
  out_sites.append({'legacy_site_index':si,'legacy_stable_id':s['stable_id'],'historical_source_vertex_id_not_current_identity':int(s['source_vertex']),'donor_body_preserved':donor,'donor_name_preserved':'capitate_r' if side=='right' else 'capitate_l','old_world_target_m':target.tolist(),'old_donor_local_target_m':s['original_donor_local_target_m'],'donor_frame':{'M':M.tolist(),'t':t.tolist(),'rows':frames[donor][2],'M_spread':frames[donor][3],'t_spread_m':frames[donor][4],'target_roundtrip_error_m':float(np.linalg.norm(M.T@(M@target+t-t)-target))},'original_kA_N_per_m':float(s['original_kA_N_per_m']),'original_area_m2':float(s['baseline_tributary_area_m2']),'capitate_render_weight_not_used_for_eligibility':True,'nearest_normal_screened_face_any_distance':None if nearestall is None else {'face':nearestall,'distance_mm':float(d[nearestall]*1000),'normal_dot':float(nd[nearestall]),'owners':fown[nearestall]},'nearest_exact_same_side_region_face_any_normal':None if nearestregion is None else {'face':nearestregion,'distance_mm':float(d[nearestregion]*1000),'point_world_m':(bary[nearestregion]@T[nearestregion]).tolist(),'nodes':F[nearestregion].tolist(),'barycentric':bary[nearestregion].tolist(),'normal_dot':float(nd[nearestregion]),'owners':fown[nearestregion],'source_face_parent':int(parent[nearestregion]),'source_face_kind':int(kind[nearestregion]),'source_parent_ancestry_ids':ai[ap[nearestregion]:ap[nearestregion+1]].tolist()},'nearest_raw_candidate':None if raw is None else {'face':raw,'distance_mm':float(d[raw]*1000),'owners':fown[raw],'capitate_weight_context_only':float(bary[raw]@W[F[raw],col]),'normal_dot':float(nd[raw]),'parent':int(parent[raw]),'kind':int(kind[raw]),'ancestry':ai[ap[raw]:ap[raw+1]].tolist()},'candidate_attempts':len(attempt),'nearest_bone-screened_candidate':chosen,'rejected_leading_candidates':attempt[:8] if chosen is None else [x for x in attempt[:8] if x['status']!='proposal_candidate'],'displacement_ladder':ladder})
 inputs=[CORR,FRAME,REG,PAY,BIND,C/'coarse-surface.npz',C/'candidate-layers-C014.npz',C/'coarse-to-source-C014.npz',BM,BP]
 result={'schema':'numi.human.capitate.geometric-inferred-retarget-ladder.v1','status':'separate geometric proposals/rejections only; no header, compiler record or mechanical admission','inputs':{str(p):sha(p) for p in inputs},'bone_payload_source_archive_sha256':bsha,'method':{'surface':'C014 deep boundary D from candidate011','eligibility':'exact single-parent source ancestry; all three dominant named owners are same-side carpal/metacarpal; capitate skin weights not eligibility','normal':'D-face normal dot mean authored outward >=0.90; proposal search screen','bone':'point-to-triangle distance against same-side registered capitate/carpal/metacarpal geometry; record surface point/normal and open-mesh ray/winding witness; reject inside, indeterminate, or surface-coincident','frame':'registered CSV local=M@world+t, validated against independent capitate transform review','ladder_mm':[2,5,10,20],'physical_gates':'not evaluated: global surfaces, all layer cells, clearance, mass/strain, solver admission'},'bone_meshes':[{'body':b,'name':m['name'],'triangles':m['nt'],'boundary_edges':m['boundary'],'frame_rows':m['frame_rows']} for b,m in sorted(bones.items())],'sites':out_sites,'prior_evidence':'The original strict identity match remains 0/6 and prior result was not modified. New records use explicit new identities/targets and preserve body49/99, kA and area.','limits':['No active headers/runtime mutated.','No proposal is physical/solver admission.','Bone meshes are open, so containment is a conservative ray/winding witness, not watertight proof.','Only same-side bone meshes with transforms in the exact registered CSV were queried.']}
 op=OUT/'geometric-retarget-ladder.json';op.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');back=json.loads(op.read_text())
 for s in back['sites']:
  for x in s['displacement_ladder']:
   p=x['proposal']
   if p:
    bb=np.asarray(p['surface_binding']['barycentric']);assert np.isfinite(bb).all() and abs(bb.sum()-1)<1e-9 and bb.min()>=-1e-9
    assert p['donor_body_preserved']==s['donor_body_preserved'] and p['original_kA_N_per_m_preserved']==s['original_kA_N_per_m'] and p['original_area_m2_preserved']==s['original_area_m2']
 print(json.dumps({'output':str(op),'sha256':sha(op),'sites':[{'site':s['legacy_site_index'],'attempts':s['candidate_attempts'],'selected_face':None if s['nearest_bone-screened_candidate'] is None else s['nearest_bone-screened_candidate']['face'],'shift_mm':None if s['nearest_bone-screened_candidate'] is None else s['nearest_bone-screened_candidate']['displacement_m']*1000,'ladder':[x['max_displacement_mm'] for x in s['displacement_ladder'] if x['proposal']]} for s in back['sites']]},indent=2))
if __name__=='__main__':main()

