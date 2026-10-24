from pathlib import Path
import json,struct,hashlib,time
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
D=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003'
OUT=E/'registered-surrounding-tissue-native013-191'
OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((D/'launch-manifest.snapshot.json').read_text()); argv=manifest['argv']
posepath=D/'registered-skin-poses.json'
poses={int(x['body_index']):x['current'] for x in json.loads(posepath.read_text())['all_bodies']}
def rotate(q,p):
 q=np.asarray(q,dtype=np.float32); p=np.asarray(p,dtype=np.float32)
 x,y,z=np.moveaxis(p,-1,0); a,b,c,w=q
 u=2*(b*z-c*y);v=2*(c*x-a*z);t=2*(a*y-b*x)
 return np.stack([x+((w*u+b*t)-c*v),y+((w*v+c*u)-a*t),z+((w*t+a*v)-b*u)],axis=-1).astype(np.float32)
def bodyworld(body,p):
 pose=poses[int(body)]
 return (np.asarray(pose['position_world_m'],dtype=np.float32)+rotate(pose['orientation_world_xyzw'],p)).astype(np.float32)
arrays={};banks=[];skipped=[];inputs=[]
def store(kind,stable,layer,body_ids,V,F):
 assert np.isfinite(V).all() and len(V)>0 and len(F)>0 and F.min()>=0 and F.max()<len(V)
 i=len(banks);key=f'bank_{i:04d}'
 arrays[key+'_vertices']=V.astype(np.float32);arrays[key+'_faces']=F.astype(np.uint32)
 banks.append(dict(key=key,kind=kind,stable_id=int(stable),layer=int(layer),body_indices=list(map(int,body_ids)),vertices=len(V),triangles=len(F),aabb_min_m=V.min(0).tolist(),aabb_max_m=V.max(0).tolist()))
# Existing NHTISS4 ABI5 route/sparse ownership. Never infer missing body frames.
p=Path(argv[argv.index('--soft-tissue-payload')+1]);raw=p.read_bytes()
magic,abi,nr,nb,nv,ni,reg,source=struct.unpack_from('<8s6I32s',raw)
assert magic==b'NHTISS4\0' and abi==5
off=64
records=np.frombuffer(raw,dtype='<u4',count=nr*8,offset=off).reshape(-1,8);off+=nr*32
bindtype=np.dtype([('body','<u4'),('transform','<f4',(8,))])
bindings=np.frombuffer(raw,dtype=bindtype,count=nb,offset=off);off+=nb*36
vtype=np.dtype([('p','<f4',(3,)),('n','<f4',(3,)),('binding','<u4',(4,)),('w','<f4',(4,))])
verts=np.frombuffer(raw,dtype=vtype,count=nv,offset=off);off+=nv*56
indices=np.frombuffer(raw,dtype='<u4',count=ni,offset=off);off+=ni*4
assert off==len(raw)
for rec in records:
 fb,bc,fv,vc,fi,ic,stable,layer=map(int,rec); B=bindings[fb:fb+bc];v=verts[fv:fv+vc]
 assert len(B)==bc and len(v)==vc and fi+ic<=ni and ic%3==0
 F=indices[fi:fi+ic].reshape(-1,3).astype(np.int64)-fv
 active=v['w']>2e-6
 assert np.all(np.abs(np.where(active,v['w'],0).sum(1)-1)<2e-3)
 assert np.all(v['binding'][active]<bc)
 used=np.unique(v['binding'][active]);bodyids=np.unique(B['body'][used])
 missing=[int(b) for b in bodyids if int(b) not in poses]
 if missing:
  skipped.append(dict(kind='soft_tissue',stable_id=stable,reason='native body frame not captured',missing_bodies=missing));continue
 result=np.zeros((vc,3),dtype=np.float32)
 for slot in range(4):
  for local_binding in np.unique(v['binding'][active[:,slot],slot]):
   rows=np.flatnonzero(active[:,slot]&(v['binding'][:,slot]==local_binding))
   binding=B[int(local_binding)];t=binding['transform']
   local=(t[:3]+rotate(t[3:7],v['p'][rows])*t[7]).astype(np.float32)
   world=bodyworld(binding['body'],local)
   result[rows]=(result[rows]+v['w'][rows,slot,None]*world).astype(np.float32)
 store('soft_tissue',stable,layer,bodyids,result,F)
inputs.append(dict(kind='soft_tissue',path=str(p),sha256=sha(p),record_count=nr))
# Existing NHANAT ABI5 carries typed registered organ/vessel/neural surface IDs.
p=Path(argv[argv.index('--torso-anatomy-payload')+1]);raw=p.read_bytes()
magic,abi,nr,nv,ni,reg,source=struct.unpack_from('<8s5I32s',raw)
assert magic==b'NHANAT1\0' and abi==5
records=np.frombuffer(raw,dtype='<u4',count=nr*8,offset=60).reshape(-1,8)
off=60+nr*32
vertices=np.frombuffer(raw,dtype='<f4',count=nv*6,offset=off).reshape(-1,6);off+=nv*24
indices=np.frombuffer(raw,dtype='<u4',count=ni,offset=off);off+=ni*4
assert off==len(raw)
for rec in records:
 body,fv,vc,fi,ic,stable,layer,reserved=map(int,rec);assert reserved==0 and ic%3==0
 if body not in poses:
  skipped.append(dict(kind='anatomy',stable_id=stable,reason='native body frame not captured',missing_bodies=[body]));continue
 store('anatomy',stable,layer,[body],bodyworld(body,vertices[fv:fv+vc,:3]),indices[fi:fi+ic].reshape(-1,3).astype(np.int64)-fv)
inputs.append(dict(kind='anatomy',path=str(p),sha256=sha(p),record_count=nr))
assert not skipped, skipped
np.savez_compressed(OUT/'registered-tissue-banks.npz',**arrays)
report=dict(scope='Static obstacle authoring cache from exact source payloads and retained native013 source-only all157 body frames; not accepted deformed tissue state or physical qualification',
 pose_path=str(posepath),pose_sha256=sha(posepath),inputs=inputs,banks=banks,skipped=skipped,
 cache_path=str(OUT/'registered-tissue-banks.npz'),cache_sha256=sha(OUT/'registered-tissue-banks.npz'),
 limitations=['Float32 host arithmetic follows native transform expression order but has not been compared to a native packed-vertex dump.',
 'All source payload body frames must be present; no inferred pose fallback is used.',
 'Respiratory/cardiac and accepted FEM deformation are not applied; this is a source-pose neighborhood cache.',
 'No collision predicates or anatomical interface classifications are performed by this preparation script.'])
(OUT/'inventory.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(banks=len(banks),skipped=len(skipped),arrays=len(arrays),cache_sha256=report['cache_sha256'],path=str(OUT)),indent=2))

