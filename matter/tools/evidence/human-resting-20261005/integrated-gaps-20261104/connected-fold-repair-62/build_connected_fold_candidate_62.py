from pathlib import Path
import numpy as np,json,hashlib,ctypes,itertools
w=Path('/Users/numi/Work/human-integrated-gaps-20261104');d=w/'connected-fold-repair-62'
b=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
source=b/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'
cavity=b/'regional-compensation-1061-root/memory-performance-123/rejection-recovery-125/connected-conditioning-134/fixed-boundary-volume-proposal.npz'
outer=b/'all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
with np.load(source,allow_pickle=False) as z:
 p=z['points_reference_world_m_f64'].astype(np.float32)
 old=z['volume_tetrahedra'];mat=z['material_index'];role=z['current_typed_role_source_row_to_joint_node_O']
with np.load(cavity,allow_pickle=False) as z:
 assert str(z['source_volume_sha256'].item())==sha(source)
 retained=z['retained_input_cell_rows'];replaced=z['replacement_tetrahedra']
 po=z['replacement_parent_offsets'];pr=z['replacement_parent_cell_rows']
 t=np.concatenate([old[retained],replaced])
 # Fixed-cavity admission already requires a single semantic material per patch.
 replacement_mat=mat[pr[po[:-1]]]
with np.load(outer,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
f=role[raw['faces']];assert np.all(f>=0)
matall=np.r_[mat[retained],replacement_mat]
lib=ctypes.CDLL(str(d/'exact-pair.dylib'))
U=ctypes.POINTER(ctypes.c_uint32);D=ctypes.POINTER(ctypes.c_double)
lib.beyond.argtypes=[D,U,U];lib.beyond.restype=ctypes.c_uint
def exact_checks(points,faces,changed):
 xyz=points[faces];lo=xyz.min(1);hi=xyz.max(1);witness=[];count=0
 for row in changed:
  candidates=np.flatnonzero(np.all(hi>=lo[row],axis=1)&np.all(lo<=hi[row],axis=1))
  a=np.asarray(faces[row],np.uint32)
  for other in candidates:
   if row==other or (other in changed and other<row):continue
   bb=np.asarray(faces[other],np.uint32);count+=1
   if lib.beyond(points.ctypes.data_as(D),a.ctypes.data_as(U),bb.ctypes.data_as(U)):
    witness.append([int(row),int(other)])
 return {'broadphase_exact_pairs':count,'crossings_beyond_shared_features':len(witness),'witnesses':witness}
def quality(cells):
 q=p[cells].astype(float)
 det=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])
 edges=np.stack([q[:,i]-q[:,j] for i,j in itertools.combinations(range(4),2)])
 maxedge=np.linalg.norm(edges,axis=2).max(0)
 return det,det/maxedge**3
result=[]
for drop,keep in [(22677,22679),(22679,22677)]:
 fd=f.copy();fd[fd==drop]=keep
 facekeep=~np.any(np.diff(np.sort(fd,axis=1),axis=1)==0,axis=1)
 changedold=np.any(f==drop,axis=1)&facekeep
 changed=np.flatnonzero(changedold[facekeep]);fd=fd[facekeep]
 registered=np.ascontiguousarray(p,dtype=np.float64)
 rawpoints=registered.copy();active=role>=0
 rawpoints[role[active]]=raw['points_world_m_f32'][active].astype(float)
 checks={name:exact_checks(pp,fd,changed) for name,pp in [('registered',registered),('unposed_source',rawpoints)]}
 incident=np.flatnonzero(np.any(t==drop,axis=1))
 cells=t[incident].copy();cells[cells==drop]=keep
 cellkeep=~np.any(np.diff(np.sort(cells,axis=1),axis=1)==0,axis=1)
 det,q=quality(cells[cellkeep])
 oldq=quality(t[incident])[1]
 changes=[]
 for name,pp in [('registered',registered),('unposed_source',rawpoints)]:
  fa=pp[f[changedold]];fb=pp[fd[changed]]
  na=np.cross(fa[:,1]-fa[:,0],fa[:,2]-fa[:,0]);nb=np.cross(fb[:,1]-fb[:,0],fb[:,2]-fb[:,0])
  cos=np.einsum('ij,ij->i',na,nb)/(np.linalg.norm(na,axis=1)*np.linalg.norm(nb,axis=1))
  changes.append({'frame':name,'minimum_normal_cosine':float(cos.min())})
 item={'drop':drop,'keep':keep,'changed_retained_faces':len(changed),'removed_faces':int((~facekeep).sum()),
 'affected_drop_incident_tets':len(incident),'removed_tets':int((~cellkeep).sum()),
 'minimum_retained_volume_m3':float(det.min()/6),'minimum_quality':float(q.min()),
 'minimum_quality_before':float(oldq.min()),'nonpositive_cells':int(np.sum(det<=0)),
 'below_unchanged_native_volume_gate':int(np.sum(det/6<=1e-18)),
 'below_unchanged_native_quality_gate':int(np.sum(q<np.float32(1e-5))),
 'normal_changes':changes,'intersection_checks':checks}
 result.append(item)
 if any(x['crossings_beyond_shared_features'] for x in checks.values()):continue
 if np.any(det/6<=1e-18) or np.any(q<np.float32(1e-5)):continue
 # This is a patch proposal, deliberately not a fabricated fully rebound volume.
 patch=d/f'collapse-{drop}-to-{keep}.npz'
 affected=np.flatnonzero(np.any((t==drop)|(t==keep),axis=1))
 nextcells=t[affected].copy();nextcells[nextcells==drop]=keep
 alive=~np.any(np.diff(np.sort(nextcells,axis=1),axis=1)==0,axis=1)
 localnodes=np.unique(t[affected])
 mp=np.full(len(p),-1,np.int64);mp[localnodes]=np.arange(len(localnodes))
 np.savez_compressed(patch,points_world_m=p[localnodes],
  tetrahedra=mp[nextcells[alive]],material_indices=matall[affected[alive]],
  source_input_vertex_rows=localnodes,source_input_cell_rows=affected,
  retained_input_cell_rows=affected[alive],source_input_faces=f[changedold],
  candidate_faces=fd[changed],source_input_retired_face_rows=np.flatnonzero(~facekeep),
  source_vertex_quotient=np.array([[drop,keep]],np.int64))
 item['local_patch_proposal']={'path':str(patch),'sha256':sha(patch)}
report={'status':'OFFLINE_LOCAL_CONTRACTION_CHECK_ONLY','physical_admission':False,'native_steps':0,
 'inputs':{name:{'path':str(path),'sha256':sha(path)} for name,path in [('source',source),('cavity',cavity),('outer',outer)]},
 'predicates':{'path':'/Users/numi/Work/skin-fat-native-integration-209/apps/NumiHumanRegisteredSurfaceIntersection.hpp',
 'sha256':sha(Path('/Users/numi/Work/skin-fat-native-integration-209/apps/NumiHumanRegisteredSurfaceIntersection.hpp'))},
 'trials':result,
 'remaining':['full anatomical context and contained-volume checks','typed parent-union regeneration','conservative source mass/P1 rebinding','full native physical and motion checks']}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
