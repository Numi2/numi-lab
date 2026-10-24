#!/usr/bin/env python3
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
CAND=ROOT/'global-inner-p1-edge-support-403/quality-coordinate-005/candidate.npz'
FIELD=ROOT/'global-inner-p1-field-owner-integration-007/replay-005-bank0004-face2562-002/field-and-material-map-005.npz'
POSE=ROOT/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-003/registered-skin-poses.json'
REBIND=ROOT/'regional-refinement-compose-255/tools/rebind_human_skin_reference.py'
BODYADM=ROOT/'exact-plane-filter-owner-264/src/numilab_human/bodyparts_adm.py'
ASSETS=[
 ROOT/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin',
 Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin')
]
EXPECTED={
 str(CAND):'5b54ffd34d74d717579fcb14677cbf2bfba01f4d4562652605b9207e65f8f698',
 str(FIELD):'78eb50d2c141028ef8a45ff63b2b16e0a05fb731b3bfef5273b530c6a6e90ef1',
 str(POSE):'0e476dcacbf89bc79b986a312c75b0dca3894972db87ecff0f9e2d95f3ce48cd'
}
def sha(data): return hashlib.sha256(data).hexdigest()
for p,h in EXPECTED.items():
 got=sha(Path(p).read_bytes())
 if got!=h: raise RuntimeError(f'hash mismatch {p}: {got}')
sys.path.insert(0,str(BODYADM.parent.parent))
spec=importlib.util.spec_from_file_location('rebind_human_skin_reference',REBIND)
rebind=importlib.util.module_from_spec(spec);spec.loader.exec_module(rebind)
from numilab_human.bodyparts_adm import prepare_nearest_surface_index
z=np.load(CAND,allow_pickle=False);f=np.load(FIELD,allow_pickle=False)
X=z['points_world_m'];T=z['tetrahedra'];F=z['physical_outer_faces'];O=set(map(int,z['physical_outer_node_ids']))
ids=f['clip_constraint_tet_ids'];offs=f['clip_constraint_offsets'];barys=f['clip_constraint_tet_barycentric']
records=[]
for g in range(len(offs)-1):
 trow=int(ids[offs[g]])
 nodes=T[trow]
 b=barys[offs[g]:offs[g+1]]
 free=np.array([int(n) not in O for n in nodes])
 caps=b[:,free].sum(axis=1) if free.any() else np.zeros(len(b))
 for j,(bb,cap) in enumerate(zip(b,caps)):
  records.append((float(cap),int(trow),int(j),bb.copy(),nodes.copy(),int(g)))
records.sort(key=lambda x:x[0])
cap,trow,j,b,nodes,g=records[0]
q=b@X[nodes].astype(np.float64)
# Exact closest triangle query over the physical O boundary and matched registered skin assets.
idxO=prepare_nearest_surface_index(X.astype(np.float32),F.astype(np.int64))
def nearest_record(idx,points,faces,q):
 dist,closest,fr=idx.closest(q); close=np.asarray(closest,dtype=np.float64); tri=np.asarray(points[faces[int(fr)]],dtype=np.float64)
 n=np.cross(tri[1]-tri[0],tri[2]-tri[0]); nlen=float(np.linalg.norm(n)); n=n/nlen if nlen else n
 return {'distance_m':float(dist),'closest_world_m':close.tolist(),'face_row':int(fr),'face_vertex_ids':faces[int(fr)].tolist(),'oriented_normal':n.tolist(),'signed_plane_offset_m':float((q-close)@n),'triangle_vertices_world_m':tri.tolist()}
out={'schema':'numi.human.fixed-O-capacity-witness-007.v1','status':'bounded diagnostic; no candidate mutation/admission','pins':{p:h for p,h in EXPECTED.items()},'field_candidate':{'path':str(CAND),'tet_count':len(T),'node_count':len(X),'physical_O_node_count':len(O),'physical_O_face_count':len(F)},'minimum_capacity_witness':{'capacity':cap,'target':0.341,'tet_row_005':trow,'tet_node_rows':nodes.tolist(),'polygon_vertex_index_within_tet_constraint':j,'constraint_row':g,'barycentric_in_tet':b.tolist(),'world_point_m':q.tolist(),'tet_node_is_physical_O':[int(n) in O for n in nodes],'free_capacity_definition':'sum of this point barycentric weights on non-physical-O tet nodes, with phi<=1; necessary P1 support bound only'},'nearest_physical_O':nearest_record(idxO,X,F,q),'registered_detail_assets':[]}
snapshot=json.loads(POSE.read_text())
for asset in ASSETS:
 raw=asset.read_bytes(); payload=rebind.decode(raw); xyz=rebind.registered_positions_float32(payload,snapshot,frame='current'); FF=np.asarray(payload['inds'],dtype=np.int64); idx=prepare_nearest_surface_index(xyz,FF)
 nrec=nearest_record(idx,xyz,FF,q)
 # Generalized winding around this exact point and small normal offsets, diagnostic only.
 def winding(point):
  rel=xyz[FF].astype(np.float64)-point[None,None,:]
  a,bv,c=rel[:,0],rel[:,1],rel[:,2]
  la,lb,lc=np.linalg.norm(a,axis=1),np.linalg.norm(bv,axis=1),np.linalg.norm(c,axis=1)
  num=np.einsum('ij,ij->i',a,np.cross(bv,c))
  den=la*lb*lc+np.einsum('ij,ij->i',a,bv)*lc+np.einsum('ij,ij->i',bv,c)*la+np.einsum('ij,ij->i',c,a)*lb
  return float((2*np.arctan2(num,den)).sum()/(4*np.pi))
 n=np.asarray(nrec['oriented_normal'])
 nrec.update({'asset_path':str(asset),'asset_sha256':sha(raw),'registered_current_xyz_sha256':sha(xyz.tobytes()),'vertex_count':int(payload['nv']),'face_count':int(len(FF)),'generalized_winding_at_point':winding(q),'generalized_winding_offsets_m':{str(d):winding(q+d*n) for d in (-.001,-.0001,.0001,.001)},'winding_caveat':'diagnostic; not a certified solid predicate'})
 out['registered_detail_assets'].append(nrec)
print(json.dumps(out,indent=2,sort_keys=True))
