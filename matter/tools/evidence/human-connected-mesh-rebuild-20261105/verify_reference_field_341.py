
from pathlib import Path
import json,sys,numpy as np,hashlib
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'reference-screened-interior-341'
from compare_remeshed_native_310 import capture,strain_stats,delta,sha
from compare_matched_deformation_339 import regional_delta,gate_counts
ref=W/'matched-ccd-baseline-335/install-motion.stdout.log.gz';a=capture(ref)
skin=a['objects']['BodyParts3D whole-body skin-fat bonded shell'];si=np.flatnonzero(a['obj']==skin);tm=a['tobj']==skin
with np.load(D/'candidate-volume.npz') as z:
 p=z['points_reference_world_m_f64'].astype(np.float32)
 assert np.array_equal(p,a['rest'][si].astype(np.float32))
 assert np.array_equal(si,np.arange(si[0],si[0]+len(si)))
 tt=z['volume_tetrahedra']+si[0];tr=z['geometric_donor_body']
 b=dict(a,tets=np.concatenate([a['tets'][~tm],tt]),treg=np.r_[a['treg'][~tm],tr],tobj=np.r_[a['tobj'][~tm],np.full(len(tt),skin)])
sa,sb=strain_stats(a),strain_stats(b);metrics=regional_delta(sa,sb);g=gate_counts(metrics)
report=dict(status='PASS_FULL_REBUILD_MATCHED_REFERENCE_FIELD' if g['over_5percent']==0 else 'FAIL_FULL_REBUILD_MATCHED_REFERENCE_FIELD',
            inputs={str(p):sha(p) for p in [ref,D/'candidate-volume.npz',D/'report.json']},
            summary=g,regions=metrics,all_source_coordinates_exact=True,force_or_material_energy_qualification=False,
            native_candidate_executed=False,physical_admission=False,
            scope='Independent whole-output recomputation using emitted original335 displacement and retained exact nodes. Uses geometric donor labels exactly as declared by the legacy native capture. Does not predict candidate trajectory, constitutive force, or material energy.')
(D/'independent-reference-field.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='regions'},indent=2))
assert g['over_5percent']==0
