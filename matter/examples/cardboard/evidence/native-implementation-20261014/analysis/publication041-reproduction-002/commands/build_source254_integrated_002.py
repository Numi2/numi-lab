from pathlib import Path
import json,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-integrated-measured-coupon-contact-254-002';d=q/'analysis/source254-build-002';b=q/'build-source254-integrated-002';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='6b35572e45d27c534d8671dbad22106720542c13e7b8bfafca66c918862afbb1'
f=json.loads((s/'frozen-source.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text());old=q/'snapshots/source-integrated-measured-coupon-contact-254-001';oldf=json.loads((old/'frozen-source.json').read_text())
delta=[n for n,h in f['source_sha256'].items() if oldf['source_sha256'].get(n)!=h];assert delta==['apps/NumiHumanRestingVisual.hpp']
prior=q/'analysis/source254-build-001/report.json';pr=json.loads(prior.read_text());assert pr['all_pass'] and pr['all_inputs_unchanged']
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
a=list(pr['commands'][0]['argv'])
a=[v.replace(str(old),str(s)).replace(str(q/'build-source254-integrated-001'),str(b)) for v in a]
with (d/'build.log').open('xb') as out:p=subprocess.run(a,cwd=s,env=env,stdout=out,stderr=subprocess.STDOUT)
r={'schema':'numi.cardboard.source254-integrated-upstream-build.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'commands':[{'argv':a,'exit_code':p.returncode}],'all_pass':p.returncode==0,'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'unchanged_cardboard_CPU_checks':{'prior_report':str(prior),'prior_report_sha256':sha(prior),'all_14_prior_build_and_check_commands_pass':len(pr['commands'])==14 and pr['all_pass'],'sole_selected_delta':delta,'rationale':'No cardboard source or acceptance policy changed after the complete integrated CPU checks; fresh compilation binds all eight targets to the preserved upstream Human source.'},'gpu_execution':False,'native_composed_trajectory_qualified':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(int(not r['all_pass']))
