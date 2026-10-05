from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy.spatial.transform import Rotation
E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'rigid-digits-cpu-preview-001'
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-001/src')
sys.path.insert(0,'/Users/n/numi-human-resting-vascular-structure-source-013/matter/tools')
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.surface_topology_audit import exact_embedding
from numilab_human.compiled_quotient_embeddedness import coordinate_quotient
import accepted_mrvpack_surface_audit as a
skinpath=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/skin-boundaries-001/bodyparts3d-myosim-skinned-shell.nhskin')
skin=decode_payload(skinpath.read_bytes());weights=skin['full_weights'];faces=skin['indices'].reshape(-1,3)
owners=skin['bindings_u'][:,0][weights.argmax(axis=1)];nv=len(weights)
source=skin['vertices_f'][:,:3].astype(float)
local=[source@Rotation.from_quat(b[4:8]).as_matrix().T*b[8]+b[1:4] for b in skin['bindings_f']]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'skin_sha256':sha(skinpath),'script_sha256':sha(__file__),'cases':[],
 'scope':'Static counterfactual preview using existing CPU kinematics at two recorded poses. Not a dynamic result or native reduction validation. Root and wrist coordinates remain recorded; only digit q are reset to source qpos0.'}
for run in ['024','025']:
 posepath=out/f'poses-{run}.json';poses=json.loads(posepath.read_text())
 pack=E/f'terminal-posture-inspection-{run}/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell.mrvpack'
 raw,st,off,s=a.read_pack(pack);f=np.asarray(s[(51007,1)]['faces']);lo=int(f.min());hi=int(f.max())
 native=np.ndarray((hi-lo+1,20),dtype='<f4',buffer=raw,offset=off+lo*80)[:,:3].copy();assert np.array_equal(f-lo,faces)
 raw.close();st.close()
 for mode in ['retained_original','rigid_digits']:
  by={p['body_index']:p for p in poses[mode]};world=np.zeros((nv,3))
  for j,(body,p) in enumerate(zip(skin['bindings_u'][:,0],local)):
   b=by[int(body)];world+=(p@Rotation.from_quat(b['orientation_world_xyzw']).as_matrix().T+b['position_world_m'])*weights[:,j,None]
  if mode=='retained_original':
   oracle=float(np.linalg.norm(world[:len(native)]-native,axis=1).max());assert oracle<2e-6,oracle
  for side,low,high in [('right',45,76),('left',95,126)]:
   selected=np.flatnonzero(((owners[faces]>=low)&(owners[faces]<=high)).any(axis=1));f=faces[selected];used=np.unique(f);remap=np.full(nv,-1);remap[used]=np.arange(len(used));lf=remap[f]
   t=time.monotonic();qv,qf,_=coordinate_quotient(world[used].astype('<f4').tolist(),lf.tolist());exact=exact_embedding(qv,qf)
   report['cases'].append({'run':run,'mode':mode,'side':side,'pose_sha256':sha(posepath),'pack_sha256':sha(pack),
    'retained_pose_native_oracle_max_error_m':oracle,'exact':exact,'elapsed_s':time.monotonic()-t,
    'first_source_face_witnesses':[[int(selected[i]) for i in pair] for pair in exact['triangle_pairs'][:16]]})
   (out/'geometry-audit.json').write_text(json.dumps(report,indent=2)+'\n')
   print(run,mode,side,exact['count'],'oracle',oracle,flush=True)
