from pathlib import Path
import json,hashlib,subprocess,os,time,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-mortar-first-uncertified-audit-257-006';b=q/'build-source257-mortar-audit-007';d=q/'analysis/native604-605-mortar-inline-trace-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='0f6d693d369c3cada323cd0b92e755fd45cb58381063f5e1c8a3117fb5ad6725'
assert sha(b/'source-build.json')=='2529a5022374cf7d89a18aebd2002aeedce06832f54511707983dd2b53f2fa8e'
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text());assert bm['success']
assert all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(p)==h for p,h in bm['artifacts_sha256'].items())
d.mkdir(exist_ok=False)
spec={'schema':'numi.cardboard.native604-605-inline-mortar-trace-command.v1','source':str(s),'build':str(b),'source_sha256':sha(s/'frozen-source.json'),'build_sha256':sha(b/'source-build.json'),'material':str(s/'matter/materials/starch2007_finite_glue_elastic.nmatter'),'policy':'Fresh identical native591 free-body fixture, full activation residual descent enabled. 604 has observers disabled;605 adds only bounded fullprivate trial0 post-rebuild and inline first-uncertified mortar CCD trace at1202:N7 env0 slot78 pair16/99. Require all production transaction files bitwise identical to each other and591. This locates a failure; it does not qualify contact or assembly.','selector':'1202:7:0:0:78:16:99:3:2','gpu_workers':1}
sp=d/'command-spec.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
deadline=time.monotonic()+1800
while not (q/'analysis/native603-command-001/receipt.json').exists():
 assert time.monotonic()<deadline,'603 receipt wait expired'
 time.sleep(5)
env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['PYTHONDONTWRITEBYTECODE']='1'
results=[]
for number,on in [(604,False),(605,True)]:
 ps=subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True)
 live=[l for l in ps.splitlines() if len(l.split())>=3 and Path(l.split(maxsplit=2)[2]).name.startswith('numi-matter-') and not l.split()[1].startswith('Z')]
 assert not live,live
 c=q/'native'/('mortar-inline-observer-'+('on' if on else 'off')+'-'+str(number));assert not c.exists()
 argv=[str(b/'matter/numi-matter-cardboard-full-contact-armijo-check'),spec['material'],'--sparse-free-body-parity','--diagnostic-state-output',str(c/'output'),'--first-failure-dispatch-root','1202','--full-contact-activation-residual-armijo']
 cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','300','--question','Which exact mortar complement query first makes accepted-history fullcontact trial1202:N7 uncertified, with observer parity?','--prediction',spec['policy'],'--input-file',str(sp)]
 if on:cmd+=['--diagnostic-run','--diagnostic-capture','NM_FEM_COUPLED_TRIAL_CAPTURE_PATH','--env','NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR=1202:7:0:both','--env','NM_FEM_COUPLED_TRIAL_VTCCD_AUDIT_SELECTOR='+spec['selector']]
 cmd+=['--']+argv
 (d/(str(number)+'.argv.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (d/(str(number)+'.launcher.log')).open('xb') as out:p=subprocess.run(cmd,cwd=s,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=360)
 r={'number':number,'case':str(c),'observer_on':on,'exit_code':p.returncode,'receipt_sha256':sha(c/'receipt.json') if (c/'receipt.json').exists() else None};results.append(r);print(json.dumps(r),flush=True)
(d/'report.json').write_text(json.dumps({'results':results,'source_inputs_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()),'artifacts_unchanged':all(sha(p)==h for p,h in bm['artifacts_sha256'].items()),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
