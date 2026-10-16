from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-best-of-cgls-255-001';b=q/'build-source255-best-of-005';c=q/'native/best-of-cgls-production-selection-606';d=q/'analysis/native606-best-of-selection-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='1efd6df0382dfe74e45d453fb22a81848d1fb251e99c2ab5cf6a8b8a5b062fc4'
assert sha(b/'source-build.json')=='9d80b01f7a3910d64daa9a8a7f05cc1a54553d92f66e674d21a409d5852b7567'
f=json.loads((s/'frozen-source.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text());bm=json.loads((b/'source-build.json').read_text());assert bm['success']
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(s/n)==h for n,h in cfg['files'].items()) and all(sha(Path(n))==h for n,h in bm['artifacts_sha256'].items())
assert verify();assert not c.exists();d.mkdir(exist_ok=False)
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')];assert not workers,workers
spec={'schema':'numi.cardboard.native606-command.v1','argv':[str(b/'matter/numi-matter-fem-cgls-best-of-kernel-check')],'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'question':'Do actual Metal best-of transaction kernels select a strictly better accepted CGLS trial and restore all rows and metadata for losing, invalid and unselected environments?','prediction':'Ten adversarial cases exercise win, tie, worse, nonfinite, unhealthy status, rejected Armijo, alpha mismatch, ineligible Newton, inconsistent Newton alpha and legacy fallback. All 2EN unknown rows and unselected metadata are deliberately mutated. This is a production selection-kernel component test with zero physical transactions, not full FEM convergence.'}
sp=d/'command.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','120','--question',spec['question'],'--prediction',spec['prediction'],'--input-file',str(sp),'--']+spec['argv']
(d/'launch.json').write_text(json.dumps({'argv':cmd,'gpu_workers_before':workers,'script_sha256':sha(__file__)},indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};clean['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
t=time.monotonic()
with (d/'wrapper.log').open('xb') as out:r=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,env=clean,timeout=180)
receipt={'exit_code':r.returncode,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));print((d/'wrapper.log').read_text()[-1200:]);sys.exit(r.returncode)
