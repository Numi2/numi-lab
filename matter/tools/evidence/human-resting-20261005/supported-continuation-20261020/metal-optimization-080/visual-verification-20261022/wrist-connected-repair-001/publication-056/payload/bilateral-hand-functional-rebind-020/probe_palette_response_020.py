"""Asset-level palette response probe; not a physical joint-motion test."""
from pathlib import Path
import copy,hashlib,json,sys
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=Path(__file__).resolve().parent
sys.path.insert(0,str(E/'registered-pose-frame-helper-194/tools'));import rebind_human_skin_reference as rb
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 current=Path('/Users/n/numi-human-separation-20261018/C/wholebody-skin-topology-repair-294/candidate-topology-001/bodyparts3d-myosim-skinned-shell.nhskin')
 candidate=D/'run-001/candidate.nhskin'
 pose=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json'
 cp=rb.decode(current.read_bytes());npay=rb.decode(candidate.read_bytes());snap=json.loads(pose.read_text())
 masks=dict(np.load(D/'run-001/selected-hand-rows.npz'));results=[]
 for label,body in [('right',54),('left',104)]:
  perturbed=copy.deepcopy(snap)
  for r in perturbed['all_bodies']:
   if r['body_index']==body:r['current']['position_world_m'][0]+=0.001
  own=masks[label+'_source_vertex_ids'];opp=masks[('left' if label=='right' else 'right')+'_source_vertex_ids']
  metrics={}
  for tag,payload in [('294',cp),('candidate020',npay)]:
   x0=rb.registered_positions_float32(payload,snap,frame='current')
   x1=rb.registered_positions_float32(payload,perturbed,frame='current')
   delta=np.linalg.norm(x1.astype(float)-x0.astype(float),axis=1)
   metrics[tag]=dict(own_selected_response_max_m=float(delta[own].max()),own_selected_responding_rows_over_1um=int((delta[own]>1e-6).sum()),
     opposite_selected_response_max_m=float(delta[opp].max()))
  passed=metrics['294']['own_selected_response_max_m']==0 and metrics['candidate020']['own_selected_response_max_m']>.0001 and metrics['candidate020']['opposite_selected_response_max_m']<1e-6
  results.append(dict(side=label,body_index=body,probe_translation_world_m=[.001,0,0],metrics=metrics,pass_asset_response_gate=bool(passed)))
 report=dict(status='PASS_ASSET_INFLUENCE_RESPONSE' if all(r['pass_asset_response_gate'] for r in results) else 'FAIL_ASSET_INFLUENCE_RESPONSE',
  scope='Synthetic one-body palette translation through existing source replay. It tests skin input connectivity, not an anatomically admissible joint pose, dynamics, physical forces, or native execution.',
  results=results,input_sha256={str(p):sha(p) for p in (current,candidate,pose,Path(__file__),Path(rb.__file__))},
  physical_admission=False,native_admission=False)
 (D/'run-001/palette-response.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
