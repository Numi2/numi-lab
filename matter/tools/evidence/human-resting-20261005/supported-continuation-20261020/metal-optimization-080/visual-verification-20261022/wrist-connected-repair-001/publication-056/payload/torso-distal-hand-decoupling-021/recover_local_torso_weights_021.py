"""Offline counterfactual: remove distal hand influence from torso-owned skin.
Preserve registered resting geometry through owner194 inverse rebind. Not a physical lock.
"""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'torso-distal-hand-decoupling-021'
sys.path.insert(0,str(E/'bilateral-hand-functional-rebind-020'))
from recover_hand_joint_influences_020 import rb,reference_palette,sha
sys.path.insert(0,str(E/'global-inner-p1-full-domain-clipping-009/exact-O-pair-011'))
from audit_exact_pairs_011 import ci
def remove_nonlocal_hand_weights(payload):
 bodies=payload['binds']['body'].astype(int);w=payload['weights'].copy()
 torso=bodies[w.argmax(1)]==20
 distal=((bodies>=49)&(bodies<=76))|((bodies>=99)&(bodies<=126))
 selected=torso & np.any(w[:,distal]>0,axis=1)
 w[np.ix_(selected,distal)]=0
 w[selected]=(w[selected].astype(float)/w[selected].astype(float).sum(1)[:,None]).astype(np.float32)
 rb.validate_weights(w);return w,selected
def main():
 out=D/'run-001';out.mkdir(exist_ok=False)
 src=E/'bilateral-hand-functional-rebind-020/run-001/candidate.nhskin'
 assert sha(src)=='7456bd4e1c646207ac69978a31a0298464d1e157b96ac4d97683051120738ddb'
 x=rb.decode(src.read_bytes())
 pp=E/'wholebody-tet-quality-batch-152/owner-unpaired-volume-163/app-preflight-013/source-preflight-identity-004/registered-skin-poses.json'
 snap=json.loads(pp.read_text());weights,selected=remove_nonlocal_hand_weights(x)
 A,B=reference_palette(x,snap,'current');blob,diag=rb.rebind(x,weights,A,B);y=rb.decode(blob)
 a,b=[rb.registered_positions_float32(z,snap,frame='current') for z in (x,y)]
 err=np.linalg.norm(a.astype(float)-b.astype(float),axis=1)
 assert err.max()<2e-6
 assert np.array_equal(x['inds'],y['inds'])
 assert np.array_equal(x['verts'][~selected],y['verts'][~selected])
 assert np.array_equal(x['weights'][~selected],y['weights'][~selected])
 asset=out/'candidate.nhskin';asset.write_bytes(blob)
 np.savez_compressed(out/'selected-torso-rows.npz',source_vertex_ids=np.flatnonzero(selected))
 report=dict(status='UNADMITTED_SOURCE_WEIGHT_COUNTERFACTUAL',input_sha256={str(p):sha(p) for p in (src,pp,Path(rb.__file__),Path(__file__))},
 candidate_sha256=sha(asset),changed_rows=int(selected.sum()),reference_error_max_m=float(err.max()),
 reference_error_rms_m=float(np.sqrt(np.mean(err**2))),owner_rebind=diag,
 policy='Only torso-dominant fullweight rows: remove registered wrist/finger body49..76 and99..126 influences, renormalize remaining proximal/axial weights. Existing owner194 inverse rebind preserves identity004 current surface.',
 limitations=['Source binding locality experiment only; no physical constraints/forces changed.','Not independent anatomical evidence.','Needs positive layer mapping, full tissue clearance, pose and native qualification.'],
 physical_admission=False,native_admission=False)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='owner_rebind'}),flush=True)
 # First causal probe: right-flex is the original seven-pair case.
 p=next((E/'native-pose-checks/repair-006/right-flex').glob('*.skin-poses.json'));pose=json.loads(p.read_text())
 pa,pb=[rb.registered_positions_float32(z,pose,frame='current') for z in (x,y)]
 ra,_=ci.float32_surface_records(pa,x['inds']);rr,_=ci.float32_surface_records(pb,y['inds'])
 cache,base=ci._prepare_complete_self_pair_baseline(ra);_,new=ci._update_complete_self_pair_baseline(cache,rr)
 old=set(map(tuple,base['triangle_pairs']));now=set(map(tuple,new['triangle_pairs']))
 result=dict(status='PASS_CLEAR_PROBE' if not now else 'UNRESOLVED_OR_REGRESSED_PROBE',scope='Exact source-only causal probe, not physical/native qualification',
 snapshot_path=str(p),snapshot_sha256=sha(p),baseline_pairs=[list(z) for z in sorted(old)],candidate_pairs=[list(z) for z in sorted(now)],added_pairs=[list(z) for z in sorted(now-old)],
 removed_pairs=[list(z) for z in sorted(old-now)],physical_admission=False,native_admission=False)
 (out/'right-flex-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()

