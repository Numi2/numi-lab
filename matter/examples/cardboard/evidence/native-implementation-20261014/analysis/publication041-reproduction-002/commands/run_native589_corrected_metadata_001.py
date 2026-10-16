from pathlib import Path
import json,subprocess,os,hashlib,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-strict-refined-contact-observers-241-004';b=q/'build-source241-integrated-observers-001';d=q/'analysis/native589-corrected-true-residual-metadata-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='f64e6c5b12a0d69b944e002eb4c499533ac9b6cd6b852a96d869ce053c41c1db'
assert sha(b/'source-build.json')=='4255148cd4d29d8a0a2dbb5048a32cdd8a5c6a04a4d3e74a3b70a757638969fc'
assert (q/'solver/strict-state-source240-newton128-full-coupon-585/receipt.json').exists()
ps=subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True)
workers=[l for l in ps.splitlines() if len(l.split())>=3 and Path(l.split(maxsplit=2)[2]).name.startswith('numi-matter-') and not l.split()[1].startswith('Z')]
assert not workers,workers
d.mkdir(exist_ok=False);case=q/'native/corrected-refined-true-residual-metadata-589';assert not case.exists()
spec={'source':str(s),'build':str(b),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'target':'numi-matter-fgmres-true-residual-native-check','arguments':['--refined-coupon'],'question':'Does the native true-residual checker report each actual measured node/row count after its numerical assertions, including20682 and32768?','prediction':'The correction changes stdout metadata only; no full FEM evolution or shipping capacity qualification.'}
sp=d/'command-spec.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(case),'--timeout','180','--question',spec['question'],'--prediction',spec['prediction'],'--input-file',str(sp),'--',str(b/'matter'/spec['target']),'--refined-coupon']
(d/'argv.json').write_text(json.dumps(a,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['PYTHONDONTWRITEBYTECODE']='1'
with (d/'launcher.log').open('xb') as out:p=subprocess.run(a,cwd=s,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=240)
receipt=json.loads((case/'receipt.json').read_text())
text=(case/'stdout.log').read_text();rows=[]
for l in text.splitlines():
 if l.startswith('fgmres_true_residual_measurement='):rows.append(json.loads(l.split('=',1)[1]))
# Keep actual emitted records intact; parser/gate is separately source-audited.
r={'schema':'numi.cardboard.native589-corrected-node-metadata.v1','exit_code':p.returncode,'native_receipt':receipt,'stdout_sha256':sha(case/'stdout.log'),'stdout':text,'recognized_records':rows,'physical_FEM_evolution_qualified':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode,'stdout':text}),flush=True);sys.exit(p.returncode)
