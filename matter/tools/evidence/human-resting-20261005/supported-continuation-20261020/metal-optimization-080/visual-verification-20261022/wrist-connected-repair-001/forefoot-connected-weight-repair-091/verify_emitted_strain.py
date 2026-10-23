from pathlib import Path
import json,struct,hashlib,csv,subprocess,numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
P=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/finger-joint-response-001')
O=E/'forefoot-connected-weight-repair-091/emitted-strain';O.mkdir(exist_ok=False)
asset=E/'forefoot-connected-weight-repair-091/candidate.nhskin'
raw=asset.read_bytes();_,abi,nb,nv,ni,fp=struct.unpack_from('<8s5I',raw);assert abi==5
bind=np.frombuffer(raw,dtype=[('body','<u4'),('data','<f4',(8,))],count=nb,offset=60)
vo=60+nb*36;io=vo+nv*56;wo=io+ni*4
F=np.frombuffer(raw,dtype='<u4',count=ni,offset=io).reshape(-1,3)
W=np.frombuffer(raw,dtype='<f4',count=nv*nb,offset=wo).reshape(nv,nb)
bmap={int(b):i for i,b in enumerate(bind['body'])}
def xyz(p):
 x=np.empty((nv,3))
 with open(p) as h:
  for r in csv.DictReader(h):x[int(r['vertex'])]=[float(r[k]) for k in ('wx','wy','wz')]
 return x
ref=P/'final074-neutral-reference-002/registered-skin.csv';X=xyz(ref)
regions={}
for side,fem,tib,cal,toe in [('right',131,136,138,139),('left',145,150,152,153)]:
 for name,ids in [('knee',[fem,tib]),('forefoot',[cal,toe])]:
  lanes=[bmap[i] for i in ids];s=W[:,lanes]
  mask=(s[:,0]>.02)&(s[:,1]>.02)&(s.sum(axis=1)>.5)&np.isin(W.argmax(axis=1),lanes)
  # Include direct neighbours but exclude all faces wholly outside the transition.
  faces=np.flatnonzero(np.any(mask[F],axis=1))
  regions[side+'_'+name]={'body_ids':ids,'transition_vertices':np.flatnonzero(mask),'faces':faces}
def rest(Fi):
 v=X[Fi];A=np.stack((v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=2)
 lengths=np.linalg.norm(np.stack((v[:,1]-v[:,0],v[:,2]-v[:,0],v[:,2]-v[:,1]),axis=1),axis=2)
 a2=np.linalg.norm(np.cross(A[:,:,0],A[:,:,1]),axis=1)
 q=2*np.sqrt(3)*a2/np.sum(lengths**2,axis=1)
 return A,q,a2/lengths.max(axis=1),a2/2
report={'asset':str(asset),'asset_sha256':hashlib.sha256(raw).hexdigest(),'scope':'Source joint-transition quality and deformation; no bone/contact or physical admission claim','regions':{},'poses':{}}
for name,r in regions.items():
 A,q,alt,area=rest(F[r['faces']]);ids=r['faces']
 report['regions'][name]={'body_ids':r['body_ids'],'transition_vertex_count':len(r['transition_vertices']),'face_count':len(ids),'min_quality':float(q.min()),'min_altitude_m':float(alt.min()),'sliver_faces_below_0p005':ids[q<.005].tolist(),'worst_face':int(ids[np.argmin(q)]),'source_face_ids':ids.tolist()}
old=json.loads((P/'joint-surface-audit-final074-001/report.json').read_text())
old['poses'].update({n:{'path':str(E/'joint-regional-quality-088-expanded-poses'/(n+'.json'))} for n in ['mtp_m0p5','mtp_0p5']})
for name,pp in old['poses'].items():
 if not (name.startswith('knee') or name.startswith('mtp')):continue
 out=O/(name+'.csv');z=subprocess.run([str(P/'export-pose-lbs-csv'),str(asset),pp['path'],str(ref),str(out)],text=True,capture_output=True);assert z.returncode==0,z.stderr
 Y=xyz(out);item={}
 for reg,r in regions.items():
  if ('knee' in name)!=('knee' in reg):continue
  ids=r['faces'];Fi=F[ids];A,q,alt,area=rest(Fi);v=Y[Fi];B=np.stack((v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=2)
  # reference QR maps a triangle's tangent coordinates into R^3; singular values are rigid-transform invariant.
  _,RR=np.linalg.qr(A,mode='reduced');D=np.linalg.solve(RR.transpose(0,2,1),B.transpose(0,2,1)).transpose(0,2,1)
  ss=np.linalg.svd(D,compute_uv=False);ratio=np.linalg.norm(np.cross(B[:,:,0],B[:,:,1]),axis=1)/(2*area)
  item[reg]={'min_stretch':float(ss[:,-1].min()),'max_stretch':float(ss[:,0].max()),'min_area_ratio':float(ratio.min()),'min_stretch_below_0p35_faces':ids[ss[:,-1]<.35].tolist(),'max_stretch_above_2p4_faces':ids[ss[:,0]>2.4].tolist(),'min_area_below_0p25_faces':ids[ratio<.25].tolist()}
 report['poses'][name]=item;out.unlink()
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{x:y for x,y in v.items() if x!='source_face_ids'} for k,v in report['regions'].items()}));print(json.dumps(report['poses']))
