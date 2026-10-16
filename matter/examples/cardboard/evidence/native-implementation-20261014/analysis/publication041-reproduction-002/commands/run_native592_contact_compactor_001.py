from pathlib import Path
import json,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-shipping-contact-sample-pages-235-007';b=q/'analysis/source235-sample-compactor-007/build';d=q/'analysis/native592-contact-compactor-command-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='2fe9d0436f498f34a8bd67c51c30eca090cc24332a3e4de8d65df0aa28aae759'
assert sha(b/'source-build.json')=='4418188007412b5fc875efe2fa55f116ddf8b9af805b908830471cd7384256a7'
bm=json.loads((b/'source-build.json').read_text())
assert bm['success'] and all(sha(s/n)==h for n,h in bm['source_sha256'].items())
assert all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
assert (q/'solver/strict-state-source240-newton128-full-coupon-585/receipt.json').exists()
live=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if len(l.split())>=3 and Path(l.split(maxsplit=2)[2]).name.startswith('numi-matter-') and not l.split()[1].startswith('Z')]
assert not live,live
d.mkdir(exist_ok=False);c=q/'native/production-contact-compactor-592';assert not c.exists()
argv=[str(b/'numi-matter-cardboard-contact-sample-compaction-check'),str(b/'shaders/cardboard_contact_sample_compaction_check.metallib')]
spec={'schema':'numi.cardboard.native592-command.v1','source':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'full_build_source_count':len(bm['source_sha256']),'argv':argv,'prediction':'Require actual production contact tuple/residual/rigid/history parity and overflow failure; manufactured component only, no complete FEM or shipping trajectory qualification.','gpu_workers':1}
sp=d/'command-spec.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','180','--question','Does the actual compact-contact sample storage preserve canonical native contact actions/history and fail closed on overflow?','--prediction',spec['prediction'],'--input-file',str(sp),'--input-file',str(q/'analysis/source235-sample-compactor-007/source-overlay.json'),'--',*argv]
(d/'argv.json').write_text(json.dumps(cmd,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['PYTHONDONTWRITEBYTECODE']='1'
with (d/'launcher.log').open('xb') as out:p=subprocess.run(cmd,cwd=s,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=240)
r={'exit_code':p.returncode,'receipt_sha256':sha(c/'receipt.json') if (c/'receipt.json').exists() else None,'full_native_FEM_qualified':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(p.returncode)
