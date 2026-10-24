#!/usr/bin/env python3
"""Pinned nonvisual gate for the398 bounded background repair; no native admission."""
from pathlib import Path
import json,numpy as np,hashlib,time
D=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/global-connected-inner-field-293-cavity-remesh-014-batched-microedge-collapse')
O=D/'local-quality-cavities-398';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
t=time.monotonic();tr=json.loads((O/'transfer-report.json').read_text());mr=json.loads((O/'material-parent-014-conservative-report.json').read_text())
assert tr['geometric_transfer_gate_pass'] and mr['pass']
p=O/'candidate-with-transfer.npz';mp=O/'material-parent-014-conservative-map.npz'
assert sha(p)==tr['output_sha256'] and sha(mp)==mr['map_sha256']
a=np.load(p,allow_pickle=False);m=np.load(mp,allow_pickle=False)
X=a['points_world_m'];T=a['tetrahedra'];A=a['region_attributes'];F=a['physical_shell_faces']
assert X.dtype==np.float32 and np.isfinite(X).all()
assert T.ndim==2 and T.shape[1]==4 and np.min(T)>=0 and np.max(T)<len(X)
assert len(A)==len(T) and np.all(A==1)
p0=X[T].astype(float);dm=np.linalg.det(np.stack([p0[:,1]-p0[:,0],p0[:,2]-p0[:,0],p0[:,3]-p0[:,0]],axis=2))
maxL=np.zeros(len(T))
for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:maxL=np.maximum(maxL,np.linalg.norm(p0[:,i]-p0[:,j],axis=1))
q=dm/maxL**3
faces=np.concatenate([T[:,[1,2,3]],T[:,[0,3,2]],T[:,[0,1,3]],T[:,[0,2,1]]])
keys=np.sort(faces,axis=1)
unique,ix,counts=np.unique(keys,axis=0,return_index=True,return_counts=True)
assert np.all((counts==1)|(counts==2))
boundary=faces[ix[counts==1]];actual=np.sort(boundary,axis=1);expected=np.sort(F,axis=1)
def lex(x):return x[np.lexsort((x[:,2],x[:,1],x[:,0]))]
assert np.array_equal(lex(actual),lex(expected))
assert len(np.unique(np.sort(T,axis=1),axis=0))==len(T)
def canonical_oriented(f):
 k=np.argmin(f,axis=1);return np.stack([f[np.arange(len(f)),(k+i)%3] for i in range(3)],axis=1)
bc=canonical_oriented(boundary);fc=canonical_oriented(F)
bmap={tuple(sorted(row)):tuple(can) for row,can in zip(boundary,bc)}
orientation_mismatches=sum(bmap[tuple(sorted(row))]!=tuple(can) for row,can in zip(F,fc))
offsets=m['overlap_child_offsets'];par=m['overlap_parent_original_input_tet_row'];v=m['overlap_parent_conservative_fraction']
assert len(offsets)==len(T)+1 and offsets[0]==0 and offsets[-1]==len(v)==len(par)
assert np.all(np.diff(offsets)>0) and np.all(v>0) and np.isfinite(v).all()
colsum=np.bincount(par,weights=v,minlength=164176)
assert len(colsum)==164176 and np.max(abs(colsum-1))<2e-12
old=np.load(D/'p1-interior-face-support-395/refined-background-checkpoint.npz',allow_pickle=False)
assert np.array_equal(X,old['points_world_m'])
physicalset={tuple(row) for row in np.sort(a['physical_outer_faces'],axis=1)}
closureset={tuple(row) for row in np.sort(a['construction_closure_faces'],axis=1)}
assert not physicalset&closureset and physicalset|closureset=={tuple(row) for row in expected}
report={'candidate_sha256':sha(p),'material_map_sha256':sha(mp),'nodes':len(X),'active_nodes':len(np.unique(T)),'unreferenced_coordinate_rows':len(X)-len(np.unique(T)),'tets':len(T),'nonpositive':int((dm<=0).sum()),'minimum_quality':float(q.min()),'boundary_faces':len(boundary),'boundary_exact':True,'oriented_boundary_mismatches':orientation_mismatches,'physical_O_faces':len(physicalset),'construction_closure_faces':len(closureset),'all_old_coordinates_bitwise_unchanged':True,'total_volume_m3':float(dm.sum()/6),'max_conservative_parent_column_error':float(np.max(abs(colsum-1))),'elapsed_s':time.monotonic()-t,'native_admission':False,'limitations':['Existing construction caps do not certify physical skin aperture closure.','Unreferenced coordinate rows need compaction in the consuming mechanical compiler.','No absolute physical kg ledger or solved layer field is certified.']}
report['gate_pass']=bool(np.all(dm>0) and np.min(q)>=1e-5 and orientation_mismatches==0)
(O/'validation-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
assert report['gate_pass']
