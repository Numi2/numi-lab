from pathlib import Path
from collections import deque
import numpy as np, json, hashlib, csv
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
B=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
source=B/'all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz'
volume=B/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
pin=sha(source);assert pin=='10169962e298586d63794bf94515ecc3a9b1ce91e66cc60566a6919d7c856ac3'
w=json.loads((W/'late-skin-fold-localization-56.json').read_text())
rows=w['source_rows'];byJoint={int(r['joint_node_id']):r for r in rows}
point=22677;tri=np.array([22743,22679,22675],dtype=np.int64);nodes=set([point,*map(int,tri)])
with np.load(volume,allow_pickle=False) as v:
 sourceToJoint=v['current_typed_role_source_row_to_joint_node_O']
 volumePoints=v['points_reference_world_m_f64'].astype(np.float32)
with np.load(source,allow_pickle=False) as z:
 rawP=z['points_world_m_f32'];rawF=z['faces'];F=sourceToJoint[rawF]
 P=np.full((int(sourceToJoint.max())+1,3),np.nan,dtype=np.float32)
 mapPath=Path(w['source_map'])
 assert sha(mapPath)==w['source_map_sha256']
 with mapPath.open() as mf:
  for row in csv.DictReader(mf):
   if row['registered_surface_layer']=='O':
    local=int(row['registered_surface_vertex_row'])
    P[local]=[float(row['current_'+c+'_m']) for c in 'xyz']
 activeFaces=np.all(F>=0,axis=1)
 activeFaces &= np.all(np.isfinite(P[np.maximum(F,0)]),axis=(1,2))
 F=F.copy()

 assert F.ndim==2 and F.shape[1]==3 and P.shape[1]==3
 assert np.array_equal(P[list(nodes)],volumePoints[list(nodes)])
 for joint,row in byJoint.items():
  local=int(row['registered_surface_vertex_row'])
  expected=np.array([float(row['current_'+c+'_m']) for c in 'xyz'],dtype=np.float32)
  assert np.array_equal(P[local],expected),(local,P[local],expected)
 target=np.flatnonzero(activeFaces & np.all(np.sort(F,axis=1)==np.sort(tri),axis=1))
 incident=np.flatnonzero(activeFaces & np.any(np.isin(F,list(nodes)),axis=1))
 adj={}
 for f in F[activeFaces]:
  a,b,c=map(int,f)
  for u,v in ((a,b),(b,c),(c,a)):
   adj.setdefault(u,set()).add(v);adj.setdefault(v,set()).add(u)
 distances={point:0};queue=deque([point])
 while queue:
  u=queue.popleft()
  if distances[u]>=8:continue
  for v in sorted(adj.get(u,())):
   if v not in distances:distances[v]=distances[u]+1;queue.append(v)
 faceData=[]
 ancestry=['face_source_layer','face_source_body','face_source_bone','face_source_face_id','face_source_original_id','source_input_face_rows','source_collapse_operation_rows','source_diagonal_flip_operation_rows']
 attrs={k:z[k] for k in ancestry if k in z.files}
 for row in incident:
  indices=F[row];v=P[indices].astype(np.float64)
  lengths=np.linalg.norm(v[[1,2,0]]-v[[0,1,2]],axis=1)
  area2=float(np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[0])))
  item=dict(row=int(row),nodes=indices.tolist(),original_role_vertex_rows=rawF[row].tolist(),area_m2=area2/2,
            min_edge_m=float(lengths.min()),max_edge_m=float(lengths.max()),
            triangle_mean_ratio=float(2*np.sqrt(3)*area2/np.sum(lengths**2)),
            point_incident=bool(point in indices),
            target_face=bool(row in target),
            ancestry={k:int(a[row]) for k,a in attrs.items()})
  faceData.append(item)
 collisions=np.flatnonzero(np.all(P==P[point],axis=1))
 out=dict(scope='Nonvisual exact Float32 source-surface topology audit; no geometry, contact exclusions, or physics changed',
          source_asset=str(source),source_sha256=pin,registered_points_source=w['source_map'],registered_points_sha256=w['source_map_sha256'],active_registered_faces=int(activeFaces.sum()),source_joint_to_surface_row='Source faces remapped by current_typed_role_source_row_to_joint_node_O before adjacency; mapped registered coordinates equal volume Float32 at all witness nodes',source_index_map_volume=str(volume),source_index_map_volume_sha256=sha(volume),
          source_point=point,source_triangle=tri.tolist(),target_face_rows=target.tolist(),
          shortest_surface_edge_distance_from_point={str(int(i)):distances.get(int(i),'>8') for i in tri},
          point_one_ring=sorted(adj[point]),point_exact_float32_duplicate_rows=collisions.tolist(),
          incident_faces=faceData)
(W/'late-skin-fold-topology-56.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='incident_faces'},indent=2))
print('incident_face_count',len(faceData),'worst_mean_ratio',min(d['triangle_mean_ratio'] for d in faceData))
for f in faceData:
 if f['target_face'] or f['point_incident']:print(json.dumps(f))
