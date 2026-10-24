#!/usr/bin/env python3
"""Derive construction-plane covectors from frozen 035 local inside rows."""
import hashlib, json, struct
from pathlib import Path
import numpy as np
ROOT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BASE=ROOT/'E/native-candidate020-all-bank-affine-roundtrip-036'
OUT=BASE/'covector-plane-037'; CAP=ROOT/'native-candidate020-all-bank-containment-035/source-preflight-003'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f32(x): return struct.unpack('<f',struct.pack('<f',float(x)))[0]
def rotate(q,v):
    x,y,z,w=q; a,b,c=v
    t=(2.0*(y*c-z*b),2.0*(z*a-x*c),2.0*(x*b-y*a))
    cross=(y*t[2]-z*t[1],z*t[0]-x*t[2],x*t[1]-y*t[0])
    return (a+w*t[0]+cross[0],b+w*t[1]+cross[1],c+w*t[2]+cross[2])
def world_to_local_matrix(q):
    qi=(-q[0],-q[1],-q[2],q[3])
    columns=[rotate(qi,e) for e in ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))]
    return np.asarray([[columns[j][i] for j in range(3)] for i in range(3)],dtype=np.float64)

snapshot=json.loads((CAP/'registered-skin-poses.json').read_text())
bodies={int(x['body_index']):x['current'] for x in snapshot['all_bodies']}
rows=np.load(BASE/'all-bank-local-to-world-rows.npz')
inside=np.flatnonzero(rows['classification_code']==1); n=len(rows['classification_code'])
local_n=np.full((n,3),np.nan,dtype=np.float64)
world_vector=np.full((n,3),np.nan,dtype=np.float64)
world_covector=np.full((n,3),np.nan,dtype=np.float64)
qdev=[]; dets=[]; vector_covector_delta=[]; plane_invariance=[]; inverse_residual=[]
for i in inside:
    body=int(rows['body_index'][i]); pose=bodies[body]
    q=np.asarray([f32(x) for x in pose['orientation_world_xyzw']],dtype=np.float64)
    qdev.append(abs(float(np.linalg.norm(q))-1.0))
    M=world_to_local_matrix(q); det=float(np.linalg.det(M)); dets.append(det)
    if not np.isfinite(det) or abs(det)<1e-8: raise ValueError(f'singular raw-F32 affine matrix at row {i}, body {body}')
    vlocal=np.asarray(rows['closest_local_m'][i]-rows['query_local_m'][i],dtype=np.float64)
    length=float(np.linalg.norm(vlocal))
    if not np.isfinite(length) or length<=0: raise ValueError(f'zero/nonfinite inside escape at row {i}')
    nl=vlocal/length
    # x_local=M*x_world+c means plane covectors transform with M.T.
    covector=M.T@nl; covector/=np.linalg.norm(covector)
    # Escape displacement is a vector and transforms with M^{-1}; keep distinct.
    vector=np.linalg.solve(M,nl); vector/=np.linalg.norm(vector)
    delta_world=np.linalg.solve(M,vlocal)
    raw_covector=M.T@nl
    plane_invariance.append(abs(float(raw_covector@delta_world-nl@vlocal)))
    inverse_residual.append(float(np.linalg.norm(M@delta_world-vlocal)))
    vector_covector_delta.append(float(np.linalg.norm(vector-covector)))
    local_n[i]=nl; world_vector[i]=vector; world_covector[i]=covector
# A shear regression proves covector and vector transforms must not be conflated.
S=np.asarray([[1.,.2,0.],[0.,1.,0.],[0.,0.,1.]])
nt=np.asarray([1.,0.,0.]); expected=S.T@nt; expected/=np.linalg.norm(expected)
wrong=np.linalg.solve(S,nt); wrong/=np.linalg.norm(wrong)
shear_separation=float(np.linalg.norm(expected-wrong))
assert shear_separation>.1
assert len(inside)==139 and len(np.unique(rows['global_node'][inside]))==139
assert min(dets)>.999 and max(dets)<1.001
assert max(inverse_residual)<1e-12 and max(plane_invariance)<1e-15
outnpz=OUT/'inside-row-covector-planes.npz'
np.savez_compressed(outnpz,
    global_node=rows['global_node'], layer=rows['layer'], row=rows['row'], role_mask=rows['role_mask'],
    body_index=rows['body_index'], bone_stable_id=rows['bone_stable_id'], nearest_triangle=rows['nearest_triangle'],
    classification_code=rows['classification_code'], unit_nearest_escape_local=local_n,
    unit_escape_vector_world=world_vector, unit_plane_normal_world=world_covector)
report={
 'schema':'numi.human.d37.affine-covector-plane-receipt.v1',
 'scope':'source-only direction conversion for 035 captured inside rows; no geometry or physical/native admission',
 'transform':'x_local=M*x_world+c, where M is built from rotate(conjugate(raw Float32 xyzw), basis columns), matching NumiHumanInitialSkinBoneContactWitness.hpp',
 'covector_rule':'unit_plane_normal_world=normalize(M.T @ unit_nearest_escape_local)',
 'vector_rule':'unit_escape_vector_world=normalize(solve(M, unit_nearest_escape_local)); this is not the covector rule',
 'input_sha256':{'035_pose_snapshot':sha(CAP/'registered-skin-poses.json'),'036_source_rows':sha(BASE/'all-bank-local-to-world-rows.npz'),'035_all_bank_csv':sha(CAP/'all-bank-containment.csv')},
 'rows_total':n,'inside_rows':len(inside),'unique_inside_nodes':int(len(np.unique(rows['global_node'][inside]))),
 'max_abs_raw_quaternion_norm_minus_one':max(qdev),'det_M_min':min(dets),'det_M_max':max(dets),
 'max_vector_covector_unit_difference':max(vector_covector_delta),
 'max_local_escape_inverse_residual_m':max(inverse_residual),
 'max_unnormalized_plane_dot_invariance_residual_m':max(plane_invariance),
 'synthetic_shear_vector_covector_separation':shear_separation,
 'output_path':str(outnpz),'output_sha256':sha(outnpz),
 'limitation':'Nearest-point query-to-closest direction supplies a construction plane covector; it is not the original triangle face normal or a force normal. Inside/outside classification is inherited from 035.'}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
