from pathlib import Path
import json,hashlib,numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';P=D/'D/postfit-backtrack-418';P.mkdir(exist_ok=True);out=P/'run-001';out.mkdir(exist_ok=False)
cfg=json.load(open(D/'D/postfit-backtrack-406/run-001/backtrack-config.json'))
base=D/'D/postfit-backtrack-410/run-001/backtrack-373/pass-002/candidate.npz';audit=base.parent
fit=H/'surface-witness-basis-415/run-001';report=json.load(open(fit/'report.json'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(base)=='141ca8ff15a4377092e198c9d7d52d08034541cbf295979e7b539c34d508a090'
assert report['inputs']['base_candidate_sha256']==sha(base)
assert report['status']=='CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED' and report['native_admission'] is False and report['accepted_physical_steps']==0
z=np.load(base);t=np.load(fit/'candidate.npz');changed_metadata=[];unchanged=[]
for k in z.files:
 if k=='points_world_m':continue
 a,b=z[k],t[k]
 if a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes():unchanged.append(k);continue
 assert k.startswith(('local_witness_cluster_222_','local_witness_cluster_230_','local_witness_cluster_354_','local_witness_cluster_315_')),k
 changed_metadata.append(k)
# These source/reference arrays remain immutable even though diagnostic receipts update.
for k in ['local_witness_cluster_222_source_reference_R0_m','area_reference_037_source_O_faces','area_reference_037_points_world_m_f64','source_O_faces','typed_coarse_anchor_output_rows_018']:
 assert z[k].tobytes()==t[k].tobytes(),k
X=z['points_world_m'];Y=t['points_world_m'];support=t['local_witness_cluster_222_support_rows'];changed=np.flatnonzero(np.any(X.view(np.uint32)!=Y.view(np.uint32),axis=1));assert np.isin(changed,support).all()
anchors=z['typed_coarse_anchor_output_rows_018'];assert X[anchors].tobytes()==Y[anchors].tobytes()
cfg['state']={k:str(v)for k,v in dict(candidate=base,point_summary=audit/'audit-130/summary.json',surface_report=audit/'audit-159/report.json',parent_fit=fit/'report.json').items()}
cfg.update(trial=str(fit/'candidate.npz'),out=str(out/'backtrack-373'),parent_fit=str(fit/'report.json'),fit_candidate_sha256=sha(fit/'candidate.npz'),fit_report_sha256=sha(fit/'report.json'),fit_support_rows=support.tolist(),area_reference_npz=str(base),area_reference_sha256=sha(base),self_pair_seed_path=str(D/'D/prospective-self-guards-412/active-selection-413-4x/selected-pair-ids.npy'),self_pair_seed_sha256=sha(D/'D/prospective-self-guards-412/active-selection-413-4x/selected-pair-ids.npy'),self_pair_seed_report_sha256=sha(D/'D/prospective-self-guards-412/active-selection-413-4x/report.json'),self_pair_seed_count=22144)
cfg['uniform_failure_backtracking']=False
cp=out/'backtrack-config.json';cp.write_text(json.dumps(cfg,indent=2)+'\n')
proof=dict(status='PASS_PINNED_418_LOCAL_POSTFIT_OF_415_AND_410_BASELINE',base_sha256=sha(base),fit_sha256=sha(fit/'candidate.npz'),fit_report_sha256=sha(fit/'report.json'),unchanged_preexisting_arrays=unchanged,updated_proposal_diagnostic_arrays=changed_metadata,anchors_unchanged=len(anchors),changed_nodes=len(changed),changed_within_support=True,support_nodes=len(support),config_sha256=sha(cp),native_admission=False,accepted_physical_steps=0)
(out/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items()if not isinstance(v,list)}))

