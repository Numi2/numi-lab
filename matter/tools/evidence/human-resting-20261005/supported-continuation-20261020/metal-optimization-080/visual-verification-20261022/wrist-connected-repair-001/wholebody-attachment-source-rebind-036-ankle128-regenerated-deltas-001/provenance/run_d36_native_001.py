from pathlib import Path
import json,hashlib,subprocess,os,time,datetime
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019')
E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001'
A=E/'wholebody-attachment-source-rebind-035-knee100-regenerated-deltas-001';D=E/'wholebody-attachment-source-rebind-036-ankle128-regenerated-deltas-001'
old=A/'native-diagnostic-035-1ms-008x125us-001';N=D/'native-diagnostic-036-1ms-008x125us-001';N.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((old/'launch-manifest.snapshot.json').read_text())
def replace(s):
 return str(s).replace(str(old),str(N)).replace(str(A/'build-main-de561-002'),str(D/'build-main-0a693-004')).replace(str(A),str(D)).replace('bilateral-knee-connected-weight-repair-100/candidate.nhskin','left-ankle-rest-preserving-weight-repair-128/candidate.nhskin')
m['argv']=[replace(s) for s in m['argv']];m['environment']={k:replace(v) for k,v in m['environment'].items()}
m['environment']['NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_ID']=D.name
m['environment']['NUMI_HUMAN_EXPERIMENTAL_UNADMITTED_LAYER_CANDIDATE_SHA256']=sha(D/'D36-candidate-identity.json')
m['run_id']=N.name;m['candidate_id']=D.name;m['candidate_identity_sha256']=sha(D/'D36-candidate-identity.json')
m['binary_sha256']=sha(m['argv'][0]);m['candidate_payload_sha256']=sha(m['argv'][m['argv'].index('--skin-payload')+1])
m['base_launch_snapshot_path']=str(old/'launch-manifest.snapshot.json');m['base_launch_snapshot_sha256']=sha(old/'launch-manifest.snapshot.json')
for k in list(m):
 if k.startswith('source_only_preflight_'):del m[k]
for k,f in [('candidate_build_manifest_sha256',D/'D36-build-manifest.json'),('compile_result_sha256',D/'build-main-0a693-004/compile-result.json'),('link_result_sha256',D/'build-main-0a693-004/link-result.json')]:m[k]=sha(f)
# Retain every historical input separately from actual current launch inputs.
m['base_input_hashes']=m.pop('input_hashes')
inputs={p for p in m['argv'] if Path(p).is_file()}
inputs.update(v for v in m['environment'].values() if Path(v).is_file())
inputs.update(str(p) for p in (D/'source-overlay/apps').iterdir() if p.is_file())
inputs.update([str(D/'D36-candidate-identity.json'),str(D/'D36-build-manifest.json')])
m['input_hashes']={p:sha(p) for p in sorted(inputs)}
m['argv_sha256']=hashlib.sha256(json.dumps(m['argv'],sort_keys=True).encode()).hexdigest()
m['environment_sha256']=hashlib.sha256(json.dumps(m['environment'],sort_keys=True).encode()).hexdigest()
m['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
snap=N/'launch-manifest.snapshot.json';snap.write_text(json.dumps(m,indent=2)+'\n')
env=dict(os.environ,**m['environment']);t=time.monotonic();err=None
with (N/'stdout.log').open('wb') as out,(N/'stderr.log').open('wb') as er:
 try:r=subprocess.run(m['argv'],env=env,cwd=R,stdout=out,stderr=er);rc=r.returncode
 except Exception as ex:err=repr(ex);rc=None
receipt={'launch_manifest_sha256':sha(snap),'returncode':rc,'launch_error':err,'elapsed_seconds':time.monotonic()-t,'stdout_sha256':sha(N/'stdout.log'),'stderr_sha256':sha(N/'stderr.log'),'physical_qualification':False}
(N/'execution-receipt.raw.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
