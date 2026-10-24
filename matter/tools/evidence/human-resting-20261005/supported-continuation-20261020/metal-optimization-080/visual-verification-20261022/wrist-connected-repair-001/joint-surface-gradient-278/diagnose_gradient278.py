from pathlib import Path
exec(Path('/tmp/audit_joint_gradient278.py').read_text().split('out={')[0].replace("O.mkdir(exist_ok=False)","O.mkdir(exist_ok=True)"))
out={}
for case,path in poses.items():
 p=json.loads(path.read_text())
 if 'source_poses' in p:
  p2=copy.deepcopy(base);by={int(x['core_body']):x for x in p['source_poses']}
  for row in p2.get('all_bodies',p2.get('bodies')):
   x=by[int(row['body_index'])];row['current']={'position_world_m':x['position'],'orientation_world_xyzw':x['quaternion_xyzw']}
  p=p2
 X=r.registered_positions_float32(b,p,frame='current').astype(float)
 E1=np.stack([X[F[:,1]]-X[F[:,0]],X[F[:,2]]-X[F[:,0]]],axis=2)
 sv=np.linalg.svd(E1@Ri,compute_uv=False)
 rows=np.unique(np.r_[regions['left_upper_arm'],regions['left_forearm'],regions['right_upper_arm'],regions['right_forearm']])
 worst=rows[np.argsort(sv[rows,0])[-12:]][::-1];rec=[]
 for f in worst:
  ids=F[f];ww=b['weights'][ids].astype(float)
  rr={'face':int(f),'source_vertices':ids.tolist(),'min_max_stretch':sv[f].tolist(),'rest_m':X0[ids].tolist(),'posed_m':X[ids].tolist(),'rest_edges_mm':(np.linalg.norm(X0[ids]-np.roll(X0[ids],1,axis=0),axis=1)*1000).tolist(),'posed_edges_mm':(np.linalg.norm(X[ids]-np.roll(X[ids],1,axis=0),axis=1)*1000).tolist(),'weight_l1_jumps':[float(np.abs(ww[i]-ww[(i+1)%3]).sum()) for i in range(3)],'dominant_bodies':[names.get(int(dom[v]),str(dom[v])) for v in ids]}
  rr['weights']=[{names.get(int(b['binds'][l]['body']),str(b['binds'][l]['body'])):float(w[l]) for l in np.flatnonzero(w)} for w in ww];rec.append(rr)
 out[case]=rec
(O/'worst-stretch-witnesses.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v[:3] for k,v in out.items()}))

