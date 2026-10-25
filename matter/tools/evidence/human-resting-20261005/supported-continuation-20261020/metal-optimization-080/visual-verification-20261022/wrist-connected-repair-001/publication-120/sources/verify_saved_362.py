from pathlib import Path
import json,hashlib,importlib.util,numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';r=H/'coherent-support-continuation-362';p=r/'uniform-scale-recovery/backtrack/selected-result.json';f=D/'D/coherent-support-continuation-360/run_coherent_target_continuation_360.py'
sp=importlib.util.spec_from_file_location('independent362',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
j=m.read(p);state,candidate,faces,anchors=m.validate_resume_state(p,expected_state_sha256=m.sha(p),expected_surface_report_sha256=m.sha(j['surface_report']))
npz=np.load(candidate,allow_pickle=False);raw=np.load(r/'round-001/attempt-001/proposal/raw-lp-candidate.npz',allow_pickle=False)
k=[k for k in raw.files if k.endswith('_lp_delta_mm_f64')][0];support=raw[k[:-len('delta_mm_f64')]+'support_rows']
scope=m.ensure_fit_scope(Path(m.read(H/'joint-boundary-continuation-322/selected-result.json')['candidate']),candidate,support)
out=dict(status='INDEPENDENT_RESUME_AND_SCOPE_PASS',state=str(p),state_sha256=m.sha(p),candidate_sha256=m.sha(candidate),surface_report_sha256=m.sha(j['surface_report']),point_summary_sha256=m.sha(j['point_summary']),resume_validation=state['_resume_verified_hashes'],scope=scope,source_fit_report_failed=True,saved_coordinates_recovered_without_lp_rerun=True,native_admission=False,accepted_physical_steps=0)
(r/'independent-selected-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
