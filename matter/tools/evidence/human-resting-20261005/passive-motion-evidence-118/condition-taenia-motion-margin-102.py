"""Bound the source clearance of a native deformation-created local crossing."""
from pathlib import Path
import hashlib,json,numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');source=E/'taenia-conditioned-audit-086/surface-457.npz'
out=E/'taenia-motion-margin-102';out.mkdir(exist_ok=False)
d=np.load(source);before=d['vertices'].astype(float);v=before.copy();f=d['faces']
vertex=4199;neighbor_face=4232;p=v[f[neighbor_face]]
normal=np.cross(p[1]-p[0],p[2]-p[0]);normal/=np.linalg.norm(normal)
distance=float((v[vertex]-p[0])@normal);assert -10e-6<distance<0
target=-40e-6
v[vertex]=(v[vertex]+(target-distance)*normal).astype('<f4').astype(float)
movement=float(np.linalg.norm(v[vertex]-before[vertex]));assert movement<50e-6
def metrics(x):
    p=x[f];n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    edge=np.maximum.reduce([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)])
    return n,float(np.min(np.linalg.norm(n,axis=1)/edge)),float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6)
old_n,_,old_vol=metrics(before);n,altitude,volume=metrics(v)
assert np.all(np.einsum('ij,ij->i',old_n,n)>0) and altitude>=1.2e-6 and abs(volume-old_vol)<1e-10
summed=np.zeros_like(v)
for k in range(3):np.add.at(summed,f[:,k],n)
length=np.linalg.norm(summed,axis=1);assert np.all(length>0)
path=out/'surface-457.npz';np.savez_compressed(path,vertices=v.astype('<f4'),normals=(summed/length[:,None]).astype('<f4'),faces=f.astype('<i4'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Offline inferred numerical conditioning of one passive source vertex. Native094 created three intersections as this vertex crossed a neighboring face plane by0.66micrometres. No mass, state, force or solver change.',
        'driver_sha256':sha(__file__),'source_sha256':sha(source),'native_diagnostic_sha256':sha(E/'taenia-native-diagnostic-099/report.json'),
        'vertex':vertex,'neighbor_face':neighbor_face,'source_signed_distance_m':distance,'target_signed_distance_m':target,
        'actual_signed_distance_m':float((v[vertex]-p[0])@normal),'movement_m':movement,'movement_bound_m':50e-6,
        'minimum_altitude_m':altitude,'volume_m3':volume,'volume_delta_m3':volume-old_vol,
        'all_face_orientations_preserved':True,'output_sha256':sha(path),
        'qualification':'Exact source and native-cycle audits are required. A40micrometre margin is an inferred numerical choice, not a measured anatomical boundary.'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
