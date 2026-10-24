import pathlib,numpy as np,csv,json,hashlib
e=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");d=e/'native-candidate020-all-bank-containment-035/source-preflight-003'
f=e/'E/native-candidate020-all-bank-affine-roundtrip-036/all-bank-local-to-world-rows.npz'
assert hashlib.sha256(f.read_bytes()).hexdigest()=='e13b0aea27be54051386252c2f140a3fa7146b74a782e094c808dab1d38109bb'
a=np.load(f)
with(d/'all-bank-containment.csv').open()as fi:rr=list(csv.DictReader(fi))
poses={int(r['body_index']):r['current'] for r in json.loads((d/'registered-skin-poses.json').read_text())['all_bodies']}
def matrix(quat):
 x,y,z,w=np.asarray(quat,dtype=np.float32).astype(float);x=-x;y=-y;z=-z
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
err=[];back=[];normaldiff=[];offsets={}
for i,r in enumerate(rr):
 b=int(r['bank_body']);M=matrix(poses[b]['orientation_world_xyzw'])
 ql=np.array([float(r['query_local_'+k])for k in 'xyz']);cl=np.array([float(r['closest_'+k])for k in 'xyz'])
 assert int(a['global_node'][i])==int(r['global_node']) and int(a['body_index'][i])==b and int(a['bone_stable_id'][i])==int(r['bank_stable_id'])
 cp=a['query_world_m'][i]+np.linalg.solve(M,cl-ql)
 err.append(np.linalg.norm(cp-a['closest_world_m'][i]))
 back.append(np.linalg.norm(M@(a['closest_world_m'][i]-a['query_world_m'][i])-(cl-ql)))
 n=(cl-ql)/np.linalg.norm(cl-ql);nw=M.T@n;nw/=np.linalg.norm(nw)
 normaldiff.append(np.linalg.norm(nw-a['unit_escape_world'][i]) if r['classification']=='inside' else np.linalg.norm(-nw-a['unit_escape_world'][i]))
 off=M@(a['query_world_m'][i]-np.asarray(poses[b]['position_world_m'],np.float32).astype(float))-ql
 offsets.setdefault(b,[]).append(off)
spread=max(np.max(np.linalg.norm(np.array(v)-np.mean(v,axis=0),axis=1)) for v in offsets.values())
result={'status':'PASS_AFFINE_CLOSEST_POINTS_ONLY','rows':len(rr),'max_closest_world_error_m':max(err),'max_inverse_delta_roundtrip_m':max(back),'max_effective_offset_spread_m':float(spread),'max_covector_vs_consistently_oriented_escape_direction_difference':max(normaldiff),'plane_normals_require_covector_transport':True,'physical_qualification':False}
assert max(err)<1e-12 and max(back)<1e-12 and spread<1e-8
(e/'root-independent-affine-review-036.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
