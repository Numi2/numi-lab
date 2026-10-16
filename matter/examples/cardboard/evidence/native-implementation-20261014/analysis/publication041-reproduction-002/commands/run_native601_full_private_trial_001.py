from pathlib import Path
import json,hashlib,subprocess,os,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-activation-full-residual-armijo-244-002';b=q/'build-source244-activation-002';d=q/'analysis/native601-freebody-newton7-full-trial-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='80dd260b4e2e5d6c617cac932d8eec2f3c021d8c318b2f93ceb8a563e7bb24a4'
assert sha(b/'source-build.json')=='39bef01c8cd0b62d708ffae18a5d03c4ef5e2ed2061bccc68e1c55f03d0dcbcf'
assert (q/'solver/strict-state-source240-newton128-full-coupon-585/receipt.json').exists()
d.mkdir(exist_ok=False)
cases=[('freebody-newton7-full-private-trial-601',True)]
spec={'schema':'numi.cardboard.native601-command.v1','source':str(s),'build':str(b),'cases':cases,'material':str(s/'matter/materials/starch2007_finite_glue_elastic.nmatter'),'policy':'Replay native591 unchanged solver/fixture/operator. Add source-bound private trial summary at control1202 Newton7 and first-failure dispatch observer at1202. Activation1199/controls1200-1201accepted; Newton7 fullprivate residual reported nonfinite for all8trials. Locate first failedstage and verify accepted histories+rollback against591. Diagnostic-only, no shippingqualification.','no_full_shipping_qualification':True}
sp=d/'command-spec.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
results=[];env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['PYTHONDONTWRITEBYTECODE']='1'
for name,on in cases:
 ps=subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True)
 live=[l for l in ps.splitlines() if len(l.split())>=3 and Path(l.split(maxsplit=2)[2]).name.startswith('numi-matter-') and not l.split()[1].startswith('Z')]
 assert not live,live
 c=q/'native'/name;assert not c.exists()
 argv=[str(b/'matter/numi-matter-cardboard-full-contact-armijo-check'),spec['material'],'--sparse-free-body-parity','--diagnostic-state-output',str(c/'output'),'--first-failure-dispatch-root','1202']
 if on:argv+=['--full-contact-activation-residual-armijo']
 cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','300','--diagnostic-run','--diagnostic-capture','NM_FEM_COUPLED_TRIAL_CAPTURE_PATH','--env','NM_FEM_COUPLED_TRIAL_CAPTURE_SELECTOR=1202:7:0:both','--question','Capture the complete control1202 Newton7 trial0 before and after rebuild at the first identified self-narrowphase mortar certification failure; preserve591/597 production history.','--prediction',spec['policy'],'--input-file',str(sp),'--',*argv]
 (d/(name+'.argv.json')).write_text(json.dumps(cmd,indent=2)+'\n')
 with (d/(name+'.launcher.log')).open('xb') as out:p=subprocess.run(cmd,cwd=s,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=360)
 r={'case':str(c),'full_activation_residual':on,'exit_code':p.returncode,'receipt_sha256':sha(c/'receipt.json') if (c/'receipt.json').exists() else None,'stdout_sha256':sha(c/'stdout.log') if (c/'stdout.log').exists() else None};results.append(r);print(json.dumps(r),flush=True)
(d/'report.json').write_text(json.dumps({'results':results,'no_result_assumed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
