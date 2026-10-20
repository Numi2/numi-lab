import sys,json,struct,math
from pathlib import Path
import numpy as np
R=Path('/Users/numi/numi-human-aortic-contact-opt-20261019');sys.path.insert(0,str(R/'tools'))
import audit_registered_ligament_geometry as ag
import audit_actual_pose_registered_ligament_catalog as ap
import audit_ligament_regional_paths as lp
import audit_final_registered_ligament_catalog as fc
invp=R/'build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-cadence-1ms-044/invocation.json'
inv=json.loads(invp.read_text());argv=inv['argv'];ag.RIGID=Path(argv[1]);ag.BONES=Path(argv[3])
rb,bb,rh,bh,js,dq,sourcePose,recs,_,v0,i0=ag.read_payloads()
q=list(map(float,dq))
for i,x in enumerate(argv):
 if x=='--pose-q':q[int(argv[i+1])]=float(argv[i+2])
_,_,q,corr=ap.parse_nheq(Path(argv[argv.index('--joint-equality-payload')+1]),q)
poses,roots,loops=ap.forward_fk(js,sourcePose,q);bp={i:p for i,p in enumerate(poses) if p is not None}
def qr(q,v):return np.asarray(ag.qrot(tuple(q),tuple(v)),float)
def tf(frame,w):return qr((-frame['q'][0],-frame['q'][1],-frame['q'][2],frame['q'][3]),np.asarray(w)-np.asarray(frame['origin']))
mb=Path(argv[2]).read_bytes();_,abi,nb,nm,ns,nw,nr,nt,*_=struct.unpack_from('<8s9I32s',mb,0)
site0=76;sites=[struct.unpack_from('<I3f',mb,site0+16*i) for i in range(ns)]
route0=site0+16*ns+64*nw;routes=[struct.unpack_from('<4I',mb,route0+16*i) for i in range(nr)]
muscle0=route0+16*nr;muscles=[struct.unpack_from('<4I37f',mb,muscle0+164*i) for i in range(nm)]
man=json.loads(Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/recovered-artifact/myosim-fullbody-reference.manifest.json').read_text());n2i={x['name']:i for i,x in enumerate(man['muscles'])}
def site_world(si):
 b,x,y,z=sites[si];p=bp[b];return np.asarray(p[:3])+qr(p[3:],(x,y,z))
def route(name):
 m=muscles[n2i[name]];off,c=m[1:3];out=[]
 for ri in range(off,off+c):
  typ,si,_,_=routes[ri]
  if typ==1:
   b,*_=sites[si];out.append((si,b,site_world(si)))
 return out
def mesh_surf(sid,frame):return ag.build_mesh(sid,frame,recs,bb,v0,i0,bp)
def candidate_triangles(mesh,normals,target,radius,maxn=100,box=None):
 vv=np.asarray(mesh[0]);tt=np.asarray(mesh[1]);pts=vv[tt].mean(1);d=np.linalg.norm(pts-np.asarray(target),axis=1)
 mask=d<=radius
 if box is not None:
  lo=np.asarray(box[0]);hi=np.asarray(box[1]);mask&=np.all((pts>=lo)&(pts<=hi),axis=1)
 ids=np.flatnonzero(mask);ids=ids[np.argsort(d[ids])[:maxn]]
 return [(pts[i],int(i),float(d[i]),fc.interpolated_normal(mesh,normals,int(i),pts[i])) for i in ids]
def allclear(a,b,allm):
 lo=np.minimum(a,b)-1e-9;hi=np.maximum(a,b)+1e-9;near=[];hits=[]
 for sid,(vv,tt) in allm.items():
  v=np.asarray(vv)
  if np.any(np.max(v,0)<lo) or np.any(np.min(v,0)>hi):continue
  near.append(sid);hs=lp.vector_hits(a,b,lp.mesh_tuple(vv,tt))
  if hs:hits.extend((sid,t,tri) for t,tri in hs)
 return near,hits
def search(side,frame,allm,id0,id1,center0,center1,r0,r1,bounds0=None,bounds1=None,max0=160,max1=160,minL=.005,maxL=.08):
 m0=allm[id0];m1=allm[id1]
 n0=fc.load_vertex_normals(id0,frame,bb,bp,recs,v0);n1=fc.load_vertex_normals(id1,frame,bb,bp,recs,v0)
 c0=candidate_triangles(m0,n0,center0,r0,max0,bounds0);c1=candidate_triangles(m1,n1,center1,r1,max1,bounds1)
 pairs=sorted(((a[2]+b[2],a,b) for a in c0 for b in c1),key=lambda z:z[0])
 tested=0;best=[]
 for score,a,b in pairs:
  p0=np.asarray(a[0],np.float32).astype(float);p1=np.asarray(b[0],np.float32).astype(float);d=p1-p0;L=np.linalg.norm(d)
  if L<minL or L>maxL:continue
  cs=(float(np.dot(a[3],d/L)),float(np.dot(b[3],-d/L)))
  if min(cs)<-1e-4:continue
  tested+=1;near,hits=allclear(p0,p1,allm)
  if not hits:best.append((score,L,a,b,cs,near))
  if len(best)>=8:break
 print('search',('R' if side==0 else 'L'),id0,id1,'targets_mm',np.round(np.asarray(center0)*1000,2).tolist(),np.round(np.asarray(center1)*1000,2).tolist(),'cands',len(c0),len(c1),'pairs',len(pairs),'noninward/tested',tested,'clear',len(best))
 for score,L,a,b,cs,near in best:
  print('  ',{'anchors_mm':(np.round(a[0]*1000,3).tolist(),np.round(b[0]*1000,3).tolist()),'tris':(a[1],b[1]),'resid':(round(a[2]*1000,2),round(b[2]*1000,2)),'length':round(L*1000,3),'cos':tuple(round(x,4) for x in cs),'near':near})
 return best
for side,(parent,child,clav,scap,sidcl,sidsc,corbname,delname) in enumerate([(33,34,29,34,8,10,'CORB','DELT1'),(83,84,79,84,9,11,'CORB_l','DELT1_l')]):
 ji,frame=ap.body_frame(js,poses,(parent,child))
 allm=lp.all_meshes(frame,bb,bp,recs,v0,i0)
 assert len(allm)==185
 print('\nSIDE',('R' if side==0 else 'L'),'joint',ji,'mesh_count',len(allm))
 corb=route(corbname)[0];corbtarget=tf(frame,corb[2]);corbproj,cord,ctri=ag.project(tuple(corbtarget),allm[sidsc])
 print('CORB site',corb[0],corb[1],'frame_mm',np.round(corbtarget*1000,2).tolist(),'projection_mm',np.round(np.asarray(corbproj)*1000,2).tolist(),'residual',round(cord*1000,3),'tri',ctri)
 delt=route(delname)
 acsc=next(x for x in delt if x[1]==scap);accl=next(x for x in delt if x[1]==clav)
 ac0=tf(frame,accl[2]);ac1=tf(frame,acsc[2]);p0,d0,t0=ag.project(tuple(ac0),allm[sidcl]);p1,d1,t1=ag.project(tuple(ac1),allm[sidsc])
 print('DELT AC-region source sites',np.round(ac0*1000,2).tolist(),np.round(ac1*1000,2).tolist(),'proj',np.round(np.asarray(p0)*1000,2).tolist(),np.round(np.asarray(p1)*1000,2).tolist(),'res',round(d0*1000,3),round(d1*1000,3),'tris',t0,t1)
 # Mesh/PCA lateral end analysis with mesh vertices: nearest pair between clavicle and scapula source vertices.
 cv=np.asarray(allm[sidcl][0]);sv=np.asarray(allm[sidsc][0])
 # chunks avoid a large full pairwise matrix
 best=(1e9,None,None)
 for st in range(0,len(cv),128):
  dd=cv[st:st+128,None,:]-sv[None,:,:];d2=np.einsum('ijk,ijk->ij',dd,dd);ix=np.unravel_index(np.argmin(d2),d2.shape)
  if d2[ix]<best[0]:best=(float(d2[ix]),st+ix[0],ix[1])
 print('nearest clav-scap vertex pair mm',math.sqrt(best[0])*1000,'clav',np.round(cv[best[1]]*1000,2).tolist(),'scap',np.round(sv[best[2]]*1000,2).tolist())
 vals,vec=np.linalg.eigh(np.cov(cv.T));axis=vec[:,-1];proj=(cv-cv.mean(0))@axis;end0=cv[proj.argmin()];end1=cv[proj.argmax()]
 # identify nearest axial end to scapula vertices
 def near2(p):
  dd=sv-p;return float(np.min(np.einsum('ij,ij->i',dd,dd)))
 lat=end0 if near2(end0)<near2(end1) else end1;med=end1 if lat is end0 else end0
 axis=med-lat;axis/=np.linalg.norm(axis);length=float(np.max(proj)-np.min(proj))
 print('clav PCA length',length*1000,'lateralend',np.round(lat*1000,2).tolist(),'medialend',np.round(med*1000,2).tolist(),'axis',np.round(axis,4).tolist(),'dist nearestend to scap mm',round(math.sqrt(min(near2(end0),near2(end1)))*1000,2))
 for frac in (.17,.24):
  center=lat+frac*length*axis
  # nearest surface projection to clavicle. Use a site on the inferior side selected from nearby vertices.
  m=allm[sidcl]; target=tuple(center)
  pp,dd,tt=ag.project(target,m)
  print('clav insertion center frac',frac,'axisPoint_mm',np.round(center*1000,2).tolist(),'nearestSurface_mm',np.round(np.asarray(pp)*1000,2).tolist(),'res',round(dd*1000,3),'tri',tt)
  # band surface region: patch around projected point 8mm
  # scap origins near CORB route site projected, constrain to 12mm spherical + small bounding.
  search(side,frame,allm,sidsc,sidcl,corbproj,np.asarray(pp),.012,.012,max0=100,max1=100,minL=.005,maxL=.075)
 # Search for AC capsuloligamentous proxy around DELT1 clav/scap origins.
 search(side,frame,allm,sidsc,sidcl,np.asarray(p1),np.asarray(p0),.012,.012,max0=100,max1=100,minL=.001,maxL=.035)

print("\\nAGGREGATE CC COMPLEX, CORACOID-BASE REGIONAL SEARCH")
for side,(parent,child,clav,scap,sidcl,sidsc,corbname,delname) in enumerate([(33,34,29,34,8,10,'CORB','DELT1'),(83,84,79,84,9,11,'CORB_l','DELT1_l')]):
 ji,frame=ap.body_frame(js,poses,(parent,child));allm=lp.all_meshes(frame,bb,bp,recs,v0,i0)
 sm=allm[sidsc];cv=np.asarray(allm[sidcl][0]);sv=np.asarray(sm[0])
 csite=route(corbname)[0];cworld=tf(frame,csite[2]);ctip,_,_=ag.project(tuple(cworld),sm);ctip=np.asarray(ctip)
 # The coracoid-base region is an explicitly inferred 20 mm vector from the source coracobrachialis/short-head tip landmark toward the scapular-body centroid, then projected to source cortex.
 targetBase=ctip+0.020*(sv.mean(0)-ctip)/np.linalg.norm(sv.mean(0)-ctip)
 pbase,baseResidual,baseTri=ag.project(tuple(targetBase),sm);pbase=np.asarray(pbase)
 vals,vec=np.linalg.eigh(np.cov(cv.T));axis=vec[:,-1];pr=(cv-cv.mean(0))@axis;e0=cv[pr.argmin()];e1=cv[pr.argmax()]
 def near2(p):
  dd=sv-p;return float(np.min(np.einsum('ij,ij->i',dd,dd)))
 lat=e0 if near2(e0)<near2(e1) else e1;med=e1 if lat is e0 else e0;axis=med-lat;axis/=np.linalg.norm(axis);L=float(np.max(pr)-np.min(pr))
 clavTarget=lat+.20*L*axis;pc,clavResidual,clavTri=ag.project(tuple(clavTarget),allm[sidcl]);pc=np.asarray(pc)
 print('SIDE',('R' if side==0 else 'L'),'framejoint',ji,'tip',np.round(ctip*1000,3).tolist(),'baseTarget',np.round(targetBase*1000,3).tolist(),'baseProjection',np.round(pbase*1000,3).tolist(),'baseResidual',round(baseResidual*1000,3),'tri',baseTri,'clavTarget',np.round(clavTarget*1000,3).tolist(),'clavProjection',np.round(pc*1000,3).tolist(),'clavResidual',round(clavResidual*1000,3),'tri',clavTri,'lengthAxis',round(L*1000,3))
 found=search(side,frame,allm,sidsc,sidcl,pbase,pc,.010,.010,max0=100,max1=100,minL=.005,maxL=.06)
