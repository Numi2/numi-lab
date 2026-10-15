from pathlib import Path
import importlib.util,json,hashlib
B=Path('/Users/home/cardboard-implementation-20261014');A=B/'analysis/armijo-all-trial-summary-analyzer-001';O=B/'analysis/valid-summary-native396-verification-001';O.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('audit395',A/'analyze.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
N=B/'native/full-contact-valid-trial-summary-396';P=B/'native/full-contact-padded-trial-summary-395';OFF=B/'native/dimensionless-policy-off-full-contact-391'
raw=(N/'trial-summary.summary.json').read_text();m.strict_json_text(raw)
capture=m.verify_payload(N,raw);n=m.validate_run(N);p=m.validate_run(P);off=m.validate_run(OFF)
comparisons={'396_vs_395':m.compare(N,P,n,p),'396_vs_391':m.compare(N,OFF,n,off)}
assert all(v['production_payload_parity'] for v in comparisons.values())
assert (N/'trial-summary.summary.bin').read_bytes()==(P/'trial-summary.summary.bin').read_bytes()
assert len(capture['trials'])==24 and [r['trial_index'] for r in capture['trials'] if r['all_stage_statuses_success']]==[22,23]
report=dict(status='PASS_NATIVE_STRICT_JSON_ALL24_PAYLOAD_AND_PRODUCTION_PARITY_EXPECTED_PHYSICAL_FAILURE',analyzer_sha256=m.sha_file(A/'analyze.py'),raw_manifest_sha256=m.sha_file(N/'trial-summary.summary.json'),raw_manifest_requires_repair=False,payload_byte_identical395=True,capture=capture,comparisons=comparisons,receipt=n['receipt'],source_build_sha256=n['source_build_sha256'],physical_contact_qualification=False)
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');(O/'verify.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:m.sha_file(p) for p in O.iterdir() if p.is_file()},indent=2)+'\n');print(report['status'])
