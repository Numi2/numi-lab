import sys, json, struct, hashlib, math
from pathlib import Path
import numpy as np
root=Path('/Users/numi/numi-human-aortic-contact-opt-20261019')
sys.path.insert(0,str(root/'tools'))
import audit_registered_ligament_geometry as ag
import audit_ligament_regional_paths as lp
import audit_actual_pose_registered_ligament_catalog as ap
import audit_final_registered_ligament_catalog as base
invpath=root/'build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-ligaments26-009/invocation.json'
inv=json.loads(invpath.read_text()); argv=inv['argv']; ag.RIGID=Path(argv[1]);ag.BONES=Path(argv[3])
rb,bb,rh,bh,joints,dq,sourcePose,records,raw,vertex0,index0=ag.read_payloads()
q=list(map(float,dq)); overrides=[]
for i,a in enumerate(argv):
 if a=='--pose-q':
  qidx=int(argv[i+1]);value=float(argv[i+2]);q[qidx]=value;overrides.append((qidx,value))
eqpath=Path(argv[argv.index('--joint-equality-payload')+1]);eqb,eqh,q,corr=ap.parse_nheq(eqpath,q)
poses,roots,loops=ap.forward_fk(joints,sourcePose,q)
posemap={i:p for i,p in enumerate(poses) if p is not None}
mp=Path(argv[2]); mb=mp.read_bytes(); mh=struct.unpack_from('<8s9I32s',mb,0); _,abi,nbody,nm,ns,nw,nr,nt,_,_,_=mh
site0=76; sites=[struct.unpack_from('<I3f',mb,site0+16*i) for i in range(ns)]
route0=site0+16*ns+64*nw; routes=[struct.unpack_from('<4I',mb,route0+16*i) for i in range(nr)]
muscle0=route0+16*nr; muscles=[struct.unpack_from('<4I37f',mb,muscle0+164*i) for i in range(nm)]
manifest=json.loads(Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/recovered-artifact/myosim-fullbody-reference.manifest.json').read_text())
name_to_i={m['name']:i for i,m in enumerate(manifest['muscles'])}
def qrot(q,v):return np.asarray(ag.qrot(tuple(q),tuple(v)),dtype=np.float64)
def site_world(siteidx):
 body,x,y,z=sites[siteidx]; p=posemap[body]
 return np.asarray(p[:3])+qrot(p[3:],(x,y,z))
def to_frame(frame,world):return qrot((-frame['q'][0],-frame['q'][1],-frame['q'][2],frame['q'][3]),world-np.asarray(frame['origin']))
def route_sites(mname,frame):
 idx=name_to_i[mname]; m=muscles[idx]; off,count=m[1:3]; out=[]
 for ri in range(off,off+count):
  typ,target,sideSite,_=routes[ri]
  if typ==1:
   body,*_=sites[target]
   out.append({'routeNode':ri,'site':target,'body':body,'p':to_frame(frame,site_world(target))})
 return out
frames={};meshes={};normals={}
for side in (0,1):
 pair=(91,92) if side==0 else (41,42)
 ji,frame=ap.body_frame(joints,poses,pair);frames[side]=(ji,frame)
 meshes[side]=lp.all_meshes(frame,bb,posemap,records,vertex0,index0)
 normals[side]={sid:base.load_vertex_normals(sid,frame,bb,posemap,records,vertex0) for sid in meshes[side]}
def tri_candidates(side,stable,center,radius=0.012,limit=48):
 mesh=meshes[side][stable]; vv=np.asarray(mesh[0]); tt=np.asarray(mesh[1]); pts=vv[tt].mean(axis=1)
 d=np.linalg.norm(pts-center,axis=1); ids=np.flatnonzero(d<=radius); ids=ids[np.argsort(d[ids])[:limit]]
 return [(pts[i].astype(np.float64),int(i),float(d[i]),base.interpolated_normal(mesh,normals[side][stable],int(i),pts[i])) for i in ids]
def nohits(a,b,side):
 lo=np.minimum(a,b)-1e-9; hi=np.maximum(a,b)+1e-9; near=[]; hits=[]
 for sid,(vv0,tt0) in meshes[side].items():
  vv=np.asarray(vv0)
  if np.any(np.max(vv,axis=0)<lo) or np.any(np.min(vv,axis=0)>hi):continue
  near.append(sid); hs=lp.vector_hits(a,b,lp.mesh_tuple(vv0,tt0))
  if hs:hits.append((sid,hs))
 return near,hits
configs={
 'elbow_mcl_anterior':{'humerus':{'L':'FCU_l','R':'FCU'},'distal':{'L':'BRA_l','R':'BRA'},'stable0':{0:13,1:12},'stable1':{0:15,1:14},'bodies':{0:(91,92),1:(41,42)}},
 'elbow_mcl_posterior':{'humerus':{'L':'PT_l','R':'PT'},'distal':{'L':'TRIlong_l','R':'TRIlong'},'stable0':{0:13,1:12},'stable1':{0:15,1:14},'bodies':{0:(91,92),1:(41,42)}},
 'elbow_lucl':{'humerus':{'L':'ECRB_l','R':'ECRB'},'distal':{'L':'SUP_l','R':'SUP'},'stable0':{0:13,1:12},'stable1':{0:15,1:14},'bodies':{0:(91,92),1:(41,42)}},
}
for name,cfg in configs.items():
 for side in (0,1):
  label='L' if side==0 else 'R'; ji,frame=frames[side]; st0=cfg['stable0'][side];st1=cfg['stable1'][side]
  r0=route_sites(cfg['humerus'][label],frame);r1=route_sites(cfg['distal'][label],frame)
  print('\n',name,label,'frame',ji,'sites0',[(x['routeNode'],x['site'],x['body'],np.round(x['p']*1000,3).tolist()) for x in r0],'sites1',[(x['routeNode'],x['site'],x['body'],np.round(x['p']*1000,3).tolist()) for x in r1])
  if not r0 or not r1:continue
  want0,want1=cfg['bodies'][side];rr0=next(x for x in r0 if x['body']==want0);rr1=next(x for x in r1 if x['body']==want1)
  c0,d0,t0=ag.project(tuple(rr0['p']),meshes[side][st0]);c1,d1,t1=ag.project(tuple(rr1['p']),meshes[side][st1])
  c0=np.asarray(c0);c1=np.asarray(c1)
  if name.startswith('elbow_mcl_'):
   medialSign=1.0 if side==0 else -1.0
   target=c1+np.array([0.0,0.0,medialSign*7.0/1000.0])
   c1,d1,t1=ag.project(tuple(target),meshes[side][st1]);c1=np.asarray(c1)
  elif name=='elbow_lucl':
   lateralSign=-1.0 if side==0 else 1.0
   target=c1+np.array([0.0,0.0,lateralSign*6.0/1000.0])
   c1,d1,t1=ag.project(tuple(target),meshes[side][st1]);c1=np.asarray(c1)
  print(' targets',np.round(c0*1000,3).tolist(),d0*1000,t0,np.round(c1*1000,3).tolist(),d1*1000,t1)
  cand0=tri_candidates(side,st0,c0,.012,100);cand1=tri_candidates(side,st1,c1,.012,100)
  found=[];tested=0
  for a,ta,da,na in cand0:
   for b,tb,db,nb in cand1:
    d=b-a;L=float(np.linalg.norm(d))
    if L<.005 or L>.06:continue
    axis=d/L;cosa=float(np.dot(na,axis));cosb=float(np.dot(nb,-axis))
    if min(cosa,cosb)<-.0001:continue
    tested+=1;near,hits=nohits(a,b,side)
    if not hits:found.append((da+db,L,-min(cosa,cosb),a,b,ta,tb,cosa,cosb,near))
  found.sort(key=lambda x:(x[0],x[2],x[1]))
  print(' candidateCounts',len(cand0),len(cand1),'tested',tested,'admitted',len(found))
  for x in found[:5]:
   _,L,_,a,b,ta,tb,cosa,cosb,near=x
   print('  witness tri',ta,tb,'anchors_mm',np.round(a*1000,6).tolist(),np.round(b*1000,6).tolist(),'offset_mm',round(np.linalg.norm(a-c0)*1000,3),round(np.linalg.norm(b-c1)*1000,3),'L_mm',round(L*1000,3),'cos',round(cosa,6),round(cosb,6),'near',near)
# RCL uses a humerus-ulna frame but a radius-head source landmark from the ulna-radius descriptor.
for side in (0,1):
 label='L' if side==0 else 'R'; ji,frame=frames[side]; st0=13 if side==0 else 12; st1=17 if side==0 else 16
 muscle='ECRB_l' if side==0 else 'ECRB'; r0=route_sites(muscle,frame);rr=next(x for x in r0 if x['body']==(91 if side==0 else 41))
 c0,d0,t0=ag.project(tuple(rr['p']),meshes[side][st0]); c0=np.asarray(c0)
 pair=(92,93) if side==0 else (42,43); jidx,jframe=ap.body_frame(joints,poses,pair)
 radialCenter=to_frame(frame,np.asarray(jframe['origin']))
 c1,d1,t1=ag.project(tuple(radialCenter),meshes[side][st1]);c1=np.asarray(c1)
 print('\n elbow_rcl',label,'frame',ji,'radialJoint',jidx,'targets',np.round(c0*1000,3).tolist(),round(d0*1000,3),np.round(c1*1000,3).tolist(),round(d1*1000,3),np.round(radialCenter*1000,3).tolist())
 cands0=tri_candidates(side,st0,c0,.012,50);cands1=tri_candidates(side,st1,c1,.010,40)
 found=[];tested=0
 for a,ta,da,na in cands0:
  for b,tb,db,nb in cands1:
   d=b-a;L=float(np.linalg.norm(d))
   if L<.005 or L>.05:continue
   axis=d/L;cosa=float(np.dot(na,axis));cosb=float(np.dot(nb,-axis))
   if min(cosa,cosb)<-.0001:continue
   tested+=1;near,hits=nohits(a,b,side)
   if not hits:found.append((da+db,L,-min(cosa,cosb),a,b,ta,tb,cosa,cosb,near))
 found.sort(key=lambda x:(x[0],x[2],x[1]))
 print(' candidateCounts',len(cands0),len(cands1),'tested',tested,'admitted',len(found))
 for x in found[:5]:
  _,L,_,a,b,ta,tb,cosa,cosb,near=x
  print('  witness tri',ta,tb,'anchors_mm',np.round(a*1000,6).tolist(),np.round(b*1000,6).tolist(),'offset_mm',round(np.linalg.norm(a-c0)*1000,3),round(np.linalg.norm(b-c1)*1000,3),'L_mm',round(L*1000,3),'cos',round(cosa,6),round(cosb,6),'near',near)

