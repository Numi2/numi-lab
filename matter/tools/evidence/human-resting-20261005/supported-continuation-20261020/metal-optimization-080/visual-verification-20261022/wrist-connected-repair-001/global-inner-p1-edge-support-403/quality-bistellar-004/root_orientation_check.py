import numpy as np,json,hashlib
from pathlib import Path
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'global-inner-p1-edge-support-403';p=D/'quality-bistellar-004/candidate-geometry-only.npz';z=np.load(p);T=z['tetrahedra'];X=z['points_world_m'];F=z['physical_shell_faces']
f=np.concatenate([T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]])
keys=np.sort(f,axis=1);u,ix,inv,c=np.unique(keys,axis=0,return_index=True,return_inverse=True,return_counts=True);r=np.argmin(f,axis=1);fc=np.stack([f[np.arange(len(f)),(r+i)%3] for i in range(3)],axis=1)
sign=np.where(fc[:,1]<fc[:,2],1,-1);sums=np.bincount(inv,weights=sign,minlength=len(u));bad=int(np.count_nonzero(sums[c==2]))
rep={'candidate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'nonmanifold_faces':int(np.sum(c>2)),'internal_oriented_mismatches':bad,'boundary_faces':int((c==1).sum()),'pass':bool(not bad and np.all(c<=2))}
out=D/'quality-bistellar-004/root-orientation-check.json';out.write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep))

