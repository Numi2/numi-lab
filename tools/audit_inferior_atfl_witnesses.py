#!/usr/bin/env python3
"""Find source-bound inferior ATFL regional proxies distinct from accepted superior witnesses."""
import hashlib,json,sys,re
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/numi-human-aortic-contact-opt-20261019');sys.path.insert(0,str(ROOT/'tools'))
import audit_registered_ligament_geometry as ag
import audit_ligament_regional_paths as lp
INV=ROOT/'build-integrated-validation-20261019/supported-skin-ligaments-20261020/native-supported-003/invocation.json'
HEADER=ROOT/'apps/NumiHumanRegisteredLigaments.hpp'
def unit(v):
 n=np.linalg.norm(v,axis=1);return v/np.maximum(n[:,None],1e-30)
def interpolated_normal(mesh,normals,ti,p):
 ids=mesh[1][ti];a,b,c=mesh[0][ids];ab=b-a;ac=c-a;ap=p-a
 d00=ab@ab;d01=ab@ac;d11=ac@ac;d20=ap@ab;d21=ap@ac;den=d00*d11-d01*d01
 wb=(d11*d20-d01*d21)/den;wc=(d00*d21-d01*d20)/den;wa=1-wb-wc
 n=wa*normals[ids[0]]+wb*normals[ids[1]]+wc*normals[ids[2]]
 return n/np.linalg.norm(n)
def vertex_normals(sid,frame,bb,bp,records,vertex0):
 r=records[sid];raw=np.frombuffer(bb,dtype='<f4',count=r['vertexCount']*6,offset=vertex0+24*r['firstVertex']).reshape(-1,6)
 ns=raw[:,3:].astype(float)
 ns=np.asarray([ag.qinvrot(frame['q'],ag.qrot(bp[r['body']][3:],ag.qrot(r['boneQ'],n))) for n in ns])
 return unit(ns)
def candidate_rows(mesh,lo,hi,upper_y,old_point,limit=300):
 v=mesh[0];tr=mesh[1];pts=v[tr].mean(axis=1)
 mask=np.all((pts>=lo)&(pts<=hi),axis=1)&(pts[:,1] < upper_y-0.0005)
 ids=np.flatnonzero(mask);d=np.linalg.norm(pts[ids]-old_point,axis=1);ids=ids[np.argsort(d)[:limit]]
 return [(pts[i],int(i),float(np.linalg.norm(pts[i]-old_point))) for i in ids]
def main():
 invb=INV.read_bytes();inv=json.loads(invb);argv=inv['argv']
 if Path(argv[1])!=ag.RIGID or Path(argv[3])!=ag.BONES:raise RuntimeError('not exact native-supported-003 source')
 rb,bb,rh,bh,js,dq,bp,records,_,v0,i0=ag.read_payloads()
 text=HEADER.read_text();wpat=re.compile(r'if\(n=="anterior_talofibular_superior"\) witness\(side==0\?V3\{([^}]*)\}:V3\{([^}]*)\},side==0\?V3\{([^}]*)\}:V3\{([^}]*)\}\);')
 m=wpat.search(text)
 if not m:raise RuntimeError('cannot parse current superior ATFL side-specific witnesses')
 parse=lambda z:np.array([float(x.strip()) for x in z.split(',')]);sup={0:(parse(m.group(1)),parse(m.group(3))),1:(parse(m.group(2)),parse(m.group(4)))}
 out={'audit':'inferior_atfl_reference_region_proxy_search_v1','classification':'inferred lower-fascicle regional witnesses; not validated anatomical entheses','source':{'invocationSha256':hashlib.sha256(invb).hexdigest(),'invocationSourceSha256':inv.get('source_sha256'),'rigidSha256':hashlib.sha256(rb).hexdigest(),'bonesSha256':hashlib.sha256(bb).hexdigest(),'headerSha256':hashlib.sha256(text.encode()).hexdigest(),'boneCount':bh[2],'vertexCount':bh[3],'indexCount':bh[4]},'criteria':{'regionBoundsMm':'fibula x=[0,22], y=[-12,5], |z|=[18,32]; talus x=[6,25], y=[-14,2], |z|=[6,20]','inferior':'both projected endpoint frame-y values at least 0.5 mm below current superior ATFL witnesses','distinct':'each endpoint at least 0.5 mm separated and on a different source triangle from superior witness','normals':'source normals barycentrically interpolated at projection; reject inward cosine below -1e-4 only','clearance':'scale-relative triangle intersections against all 185 source bone meshes with AABB culling; open-surface caveat retained'},'rows':[]}
 for side in (0,1):
  s0,s1=(21,23) if side==0 else (20,22);parent,child=(150,151) if side==0 else (136,137)
  frame=ag.make_frame(js,dq,bp,parent,child);allm=lp.all_meshes(frame,bb,bp,records,v0,i0);m0,m1=allm[s0],allm[s1]
  n0=vertex_normals(s0,frame,bb,bp,records,v0);n1=vertex_normals(s1,frame,bb,bp,records,v0)
  old0=ag.project(tuple(sup[side][0]),m0)[0];old1=ag.project(tuple(sup[side][1]),m1)[0]
  lo0=np.array([0,-12,18.])/1000;hi0=np.array([22,5,32.])/1000;lo1=np.array([6,-14,6.])/1000;hi1=np.array([25,2,20.])/1000
  if side==0:lo0[2],hi0[2]=-hi0[2],-lo0[2];lo1[2],hi1[2]=-hi1[2],-lo1[2]
  c0=candidate_rows(m0,lo0,hi0,old0[1],old0);c1=candidate_rows(m1,lo1,hi1,old1[1],old1)
  superiorTri0=ag.project(tuple(old0),m0)[2];superiorTri1=ag.project(tuple(old1),m1)[2]
  pairs=[]
  for p0,t0,d0 in c0:
   if t0==superiorTri0:continue
   for p1,t1,d1 in c1:
    if t1==superiorTri1 or p0[1]>=old0[1]-.0005 or p1[1]>=old1[1]-.0005:continue
    if np.linalg.norm(p0-old0)<.0005 or np.linalg.norm(p1-old1)<.0005:continue
    score=d0*d0+d1*d1+.05*np.linalg.norm(p1-p0)**2;pairs.append((score,p0,t0,d0,p1,t1,d1))
  pairs.sort(key=lambda x:x[0]);best=None;tested=0
  for score,p0,t0,d0,p1,t1,d1 in pairs:
   tested+=1;p0f=p0.astype(np.float32).astype(float);p1f=p1.astype(np.float32).astype(float)
   q0,dproj0,ti0=ag.project(tuple(p0f),m0);q1,dproj1,ti1=ag.project(tuple(p1f),m1)
   if ti0!=t0 or ti1!=t1:continue
   body0,body1=records[s0]['body'],records[s1]['body']
   def to_local(point,body):
    world=ag.add(frame['origin'],ag.qrot(frame['q'],point));pose=bp[body];return ag.qinvrot(pose[3:],ag.sub(world,pose[:3]))
   def from_local(local,body):
    pose=bp[body];world=ag.add(pose[:3],ag.qrot(pose[3:],local));return ag.qinvrot(frame['q'],ag.sub(world,frame['origin']))
   l0=np.asarray(to_local(q0,body0),np.float32).astype(float);l1=np.asarray(to_local(q1,body1),np.float32).astype(float)
   a=np.asarray(from_local(l0,body0));b=np.asarray(from_local(l1,body1));delta=b-a;length=np.linalg.norm(delta)
   if not length:continue
   cos0=float(interpolated_normal(m0,n0,ti0,q0)@delta/length);cos1=float(interpolated_normal(m1,n1,ti1,q1)@(-delta)/length)
   if min(cos0,cos1)<-1e-4 or np.linalg.norm(a-old0)<.0005 or np.linalg.norm(b-old1)<.0005:continue
   nearby=[];hits=[];seglo=np.minimum(a,b)-1e-9;seghi=np.maximum(a,b)+1e-9
   for sid,(vv,tt) in allm.items():
    if np.all(np.max(vv,axis=0)>=seglo) and np.all(np.min(vv,axis=0)<=seghi):
     nearby.append(sid);hh=lp.vector_hits(a,b,lp.mesh_tuple(vv,tt))
     if hh:hits.append({'stableId':sid,'sourceRecord':records[sid]['sourceRecord'],'hits':[{'t':round(float(t),9),'triangle':int(tri)} for t,tri in hh]})
   if hits:continue
   best={'side':'L' if side==0 else 'R','stableIds':[s0,s1],'sourceRecords':[records[s0]['sourceRecord'],records[s1]['sourceRecord']],'frameJointPair':[parent,child],'triangleIds':[ti0,ti1],
    'seedFrameM0':[float(x) for x in p0f],'seedFrameM1':[float(x) for x in p1f],'gpuLocalAnchorFloat32M0':[float(x) for x in l0],'gpuLocalAnchorFloat32M1':[float(x) for x in l1],
    'anchorFrameMm0':[round(float(x*1000),6) for x in a],'anchorFrameMm1':[round(float(x*1000),6) for x in b],'superiorWitnessFrameMm0':[round(float(x*1000),6) for x in old0],'superiorWitnessFrameMm1':[round(float(x*1000),6) for x in old1],
    'inferiorMinusSuperiorYmm':[round(float((a[1]-old0[1])*1000),4),round(float((b[1]-old1[1])*1000),4)],'separationFromSuperiorMm':[round(float(np.linalg.norm(a-old0)*1000),4),round(float(np.linalg.norm(b-old1)*1000),4)],'float32AnchorToSourceResidualMm':[round(float(ag.project(tuple(a),m0)[1]*1000),6),round(float(ag.project(tuple(b),m1)[1]*1000),6)],
    'lengthMm':round(float(length*1000),4),'outwardNormalCos':[round(cos0,6),round(cos1,6)],'aabbNearbyStableIds':nearby,'allRegisteredBoneMeshesTested':len(allm),'segmentTriangleHits':hits,'testedPairs':tested}
   break
  row=best or {'side':'L' if side==0 else 'R','stableIds':[s0,s1],'frameJointPair':[parent,child],'candidateCounts':[len(c0),len(c1)],'testedPairs':tested,'accepted':False}
  if best:row['accepted']=True
  out['rows'].append(row)
 out['sha256']=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 dst=ROOT/'tools/audit_inferior_atfl_witnesses.json';dst.write_text(json.dumps(out,indent=2)+'\n')
 print('receipt',dst,'sha256',out['sha256'])
 for x in out['rows']:print(x)
if __name__=='__main__':main()
