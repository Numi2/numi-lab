"""Offline source-weight sensitivity at two retained native poses; no physical stepping."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
from scipy.spatial.transform import Rotation
E=Path('/Users/n/numi-human-resting-evidence-20261005')
A=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005')
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-001/src')
sys.path.insert(0,'/Users/n/numi-human-resting-vascular-structure-source-013/matter/tools')
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.skin_geodesic_visual_candidate import compile_candidate
from numilab_human.surface_topology_audit import exact_embedding
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
import accepted_mrvpack_surface_audit as mrv
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out=E/'resting-hand-geodesic-sensitivity-001';out.mkdir(exist_ok=False)
base=A/'Docs/media/skin-weight-heldout-20261004/inputs/base/bodyparts3d-myosim-skinned-shell.nhskin'
manifest=base.with_suffix('.manifest.json')
active=A/'output/skin-boundaries-001/bodyparts3d-myosim-skinned-shell.nhskin'
solution=E/'bodyparts3d-skin-binding-solution-source.npz'
b=decode_payload(base.read_bytes()); a=decode_payload(active.read_bytes())
nv,nb=b['vertex_count'],b['binding_count'];nf=b['index_count']//3
assert np.array_equal(b['vertices_f'],a['vertices_f'])
assert np.array_equal(b['bindings_u'],a['bindings_u'])
assert np.array_equal(b['full_weights'],a['full_weights'])
faces=a['indices'].reshape(-1,3); bf=b['indices'].reshape(-1,3)
assert np.array_equal(a['vertices_f'][faces[:nf],:3],b['vertices_f'][bf,:3])
owners=a['bindings_u'][:,0][a['full_weights'].argmax(axis=1)]
subsets={side:np.flatnonzero(((owners[faces]>=lo)&(owners[faces]<=hi)).any(axis=1))
 for side,lo,hi in [('right',45,76),('left',95,126)]}
audit_path=E/'resting-hand-skin-audit-002/report.json'; old=json.loads(audit_path.read_text())
cases=[]
for row in old['cases']:
 if row['case']=='initial':continue
 chosen=subsets[row['side']]
 pairs=[[int(chosen[i]) for i in p] for p in row['exact']['triangle_pairs']]
 assert all(max(p)<nf for p in pairs), 'Witness includes new cap; cannot assign base face'
 cases.append({'case':row['case']+'_'+row['side'], 'exact_intersection_pair_count':len(pairs), 'exact_intersection_pairs':pairs})
attribution={'schema':'numi.human.native-skin-crossing-attribution.v1',
 'source':{'payload_sha256':sha(base),'manifest_sha256':sha(manifest),'binding_solution_sha256':sha(solution)},
 'cases':cases,'total_exact_intersection_pairs':sum(len(x['exact_intersection_pairs']) for x in cases),
 'source_audit_sha256':sha(audit_path), 'scope':'Exact source dominant-owner hand subsets at recorded native poses, mapped to coordinate-identical source prefix faces.'}
attr=out/'attribution.json';attr.write_text(json.dumps(attribution,indent=2)+'\n')
weights={0.:a['full_weights']}
candidate_meta={}
for blend in [.1,.25,.5,1.]:
 d=out/f'blend-{blend:g}'
 meta=compile_candidate(base,manifest,solution,attr,d,blend=blend,radius_m=.12)
 weights[blend]=decode_payload((d/meta['payload']['file']).read_bytes())['full_weights']
 candidate_meta[str(blend)]=meta
 print('compiled',blend,meta['payload']['sha256'],flush=True)
result={'script_sha256':sha(__file__), 'base_sha256':sha(base),'active_sha256':sha(active),
 'candidate_manifests':candidate_meta,'cases':[],
 'scope':'Offline full-weight geometry sensitivity only; no native candidate execution, no contact validation, no new physical model. Actual body poses are fixed from prior accepted native terminal states.'}
def save(): (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
source=a['vertices_f'][:,:3].astype(float)
local=[]
for j in range(nb):
 bind=a['bindings_f'][j];local.append(source@Rotation.from_quat(bind[4:8]).as_matrix().T*bind[8]+bind[1:4])
def pose_points(pose,which):
 by={r['body_index']:r[which] for r in pose['bodies']}
 return [v@Rotation.from_quat(by[int(body)]['orientation_world_xyzw']).as_matrix().T+by[int(body)]['position_world_m'] for body,v in zip(a['bindings_u'][:,0],local)]
def blend_points(points,w):
 v=np.zeros((nv,3))
 for j,p in enumerate(points):v+=p*w[:,j,None]
 return v
for run in ['024','025']:
 folder=E/f'terminal-posture-inspection-{run}'
 stem='myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell'
 pack=folder/(stem+'.mrvpack');posepath=folder/(stem+'.skin-poses.json')
 pose=json.loads(posepath.read_text())
 raw,st,off,s=mrv.read_pack(pack);gf=np.asarray(s[(51007,1)]['faces']);lo=int(gf.min());hi=int(gf.max())
 native=np.ndarray((hi-lo+1,20),dtype='<f4',buffer=raw,offset=off+lo*80)[:,:3].copy()
 assert np.array_equal(gf-lo,faces)
 raw.close();st.close()
 current=pose_points(pose,'current');rest=pose_points(pose,'rest')
 predicted=blend_points(current,weights[0.]);reference=blend_points(rest,weights[0.])
 error=float(np.linalg.norm(predicted[:len(native)]-native,axis=1).max())
 assert error<=2e-6, f'Native position oracle {error}'
 for blend,w in weights.items():
  world=blend_points(current,w).astype('<f4');restworld=blend_points(rest,w)
  delta=np.linalg.norm(restworld-reference,axis=1)
  for side,chosen in subsets.items():
   f=faces[chosen];used=np.unique(f);remap=np.full(nv,-1);remap[used]=np.arange(len(used));lf=remap[f]
   t=time.monotonic();qv,qf,_=coordinate_quotient(world[used].tolist(),lf.tolist());exact=exact_embedding(qv,qf)
   edges=np.unique(np.sort(np.concatenate([lf[:,[0,1]],lf[:,[1,2]],lf[:,[2,0]]]),axis=1),axis=0)
   d0=np.linalg.norm(reference[used][edges[:,1]]-reference[used][edges[:,0]],axis=1);nz=d0>1e-9
   ratio=np.linalg.norm(world[used][edges[:,1]]-world[used][edges[:,0]],axis=1)[nz]/d0[nz]
   r={'run':run,'side':side,'blend':blend,'pack_sha256':sha(pack),'pose_sha256':sha(posepath),
    'native_position_oracle_max_error_m':error,'rest_world_change_m_quantiles':np.quantile(delta,[0,.5,.99,1]).tolist(),
    'exact':exact,'source_face_witnesses':[[int(chosen[i]) for i in pair] for pair in exact['triangle_pairs'][:16]],
    'edge_stretch_quantiles':np.quantile(ratio,[0,.01,.5,.99,1]).tolist(),'elapsed_s':time.monotonic()-t}
   result['cases'].append(r);save()
   print(run,side,blend,'pairs',exact['count'],'rest_delta',delta.max(),'oracle',error,flush=True)
save()
