#!/usr/bin/env python3
"""Connected-component winding diagnostic for corrected 398/tet-88190 witness."""
import importlib.util, json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
ASSET = Path("/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin")
POSE = ROOT / "wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json"
REBIND = ROOT / "regional-refinement-compose-255/tools/rebind_human_skin_reference.py"
CAND = ROOT / "global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse/local-quality-cavities-398/candidate-with-transfer.npz"
spec=importlib.util.spec_from_file_location("rebind_human_skin_reference",REBIND)
rb=importlib.util.module_from_spec(spec);spec.loader.exec_module(rb)
payload=rb.decode(ASSET.read_bytes())
X=rb.registered_positions_float32(payload,json.loads(POSE.read_text()),frame="current").astype(np.float64)
F=np.asarray(payload["inds"],dtype=np.int64)
mesh=np.load(CAND); ids=mesh["tetrahedra"][88190]
q=np.array([.7473772446,0,.2526227554,0],dtype=np.float64)@mesh["points_world_m"][ids]
tri=X[F[40538]]
normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
edges=np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1)
U,inv,cnt=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
faceids=np.repeat(np.arange(len(F)),3); a=[];b=[]
for ei in np.flatnonzero(cnt==2):
    fs=faceids[np.flatnonzero(inv==ei)]
    if len(fs)==2: a.extend((int(fs[0]),int(fs[1])));b.extend((int(fs[1]),int(fs[0])))
G=coo_matrix((np.ones(len(a)),(a,b)),shape=(len(F),len(F))).tocsr()
_,labels=connected_components(G,directed=False,return_labels=True)
rows=np.flatnonzero(labels==labels[40538]); FF=F[rows]
def winding(p):
    t=X[FF]-p[None,None,:];a,b,c=t[:,0],t[:,1],t[:,2]
    la,lb,lc=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)
    num=np.einsum("ij,ij->i",a,np.cross(b,c))
    den=la*lb*lc+np.einsum("ij,ij->i",a,b)*lc+np.einsum("ij,ij->i",b,c)*la+np.einsum("ij,ij->i",c,a)*lb
    return float((2*np.arctan2(num,den)).sum()/(4*np.pi))
owner=np.full(len(U),-1,dtype=np.int64)
for ei in np.flatnonzero(cnt==1): owner[ei]=faceids[np.flatnonzero(inv==ei)[0]]
boundary_ids=np.flatnonzero((cnt==1)&np.isin(owner,rows)); seg=X[U[boundary_ids]]
v=seg[:,1]-seg[:,0]; s=np.clip(np.einsum("ij,ij->i",q-seg[:,0],v)/np.einsum("ij,ij->i",v,v),0,1)
dist=np.linalg.norm(seg[:,0]+s[:,None]*v-q,axis=1)
print(json.dumps({"component_faces":int(len(rows)),"component_boundary_edges":int(len(boundary_ids)),
 "nearest_component_boundary_edge_distance_m":float(dist.min()),
 "boundary_edges_within_10mm":int(np.sum(dist<.01)),
 "winding_at_query":winding(q),
 "winding_at_offsets_m":{str(d):winding(q+d*normal) for d in (-.001,-.0001,.0001,.001)}},indent=2,sort_keys=True))

