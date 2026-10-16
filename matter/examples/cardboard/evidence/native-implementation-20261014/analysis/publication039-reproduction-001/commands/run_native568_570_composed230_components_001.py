from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(number):
 if number in ['568','569','570']:
  s=q/'snapshots/source-composed-pure-proxy-replay-230-001';b=q/'build-source230-integrated-001';sh='f0800b754a820f791c7d97c1126e2e653f1b2f7064bd2b9d5ca443cf102ac682';bh='8da7edadd170c15597718fb11b4398f103c9d9de0d25c29d7fe6990d6071a7e8';lib=b/'matter/shaders/NumiMatter.metallib'
  names={'568':('composed230-prescribed-proxies-568','numi-matter-cardboard-coupled-free-box-operator-check'),'569':('composed230-true-residual-8192-569','numi-matter-fgmres-true-residual-native-check'),'570':('composed230-pure-cgls-8192-570','numi-matter-fem-cgls-arena-8192-check')}
  name,target=names[number];binary=b/'matter'/target
 elif number=='559':
  s=q/'snapshots/source216-integrated-configure-support-006';b=q/'build-source216-integrated-006';name='prescribed-box-capsule-freebox-operator-fixed-559';binary=b/'matter/numi-matter-cardboard-coupled-free-box-operator-check'
  sh='26636e0efcd70b8aef95afb82475c1d5f69eec7a1f26c7191d355d4fe55de67e';bh='5eff6f77c79bcb88ba55519bb2f3128667768b124b50782c6f84333f4b613727';lib=b/'matter/shaders/NumiMatter.metallib'
 elif number in ['560','561']:
  s=q/'snapshots/source-composed-pure-8192-on222-226-001';b=q/'build-source226-pure-arenas-001';sh='313f5a81883162bb5917773b3b3f576cf8991932ff4eed2e8af11e36cb13cba8';bh='b177434e32ebbcf4e1f0fd5199a0d715ae5a0e9efe8061ef3bd893e575384c73';lib=b/'shaders/NumiMatter.metallib'
  if number=='560':name='composed226-true-residual-8192-560';binary=b/'numi-matter-fgmres-true-residual-native-check'
  else:name='composed226-pure-cgls-8192-561';binary=b/'numi-matter-fem-cgls-arena-8192-check'
 elif number=='562':
  s=q/'snapshots/source-shipping-contact-pages-225-002';b=q/'build-source-shipping-contact-pages-225-004';name='shipping-page-ordinal-coverage-562';binary=b/'numi-matter-cardboard-shipping-contact-page-native-check';sh='99e4e77a6801688f10cfbc8af0fd61c2dc6da4ac791469ee49af8b4d4035ea26';bh='1d9405960e92a0afaf257c9430f74e704902463eba8351efecf05875a3954fed';lib=None
 else:raise ValueError(number)
 assert sha(s/'frozen-source.json')==sh and sha(b/'source-build.json')==bh
 f=json.loads((s/'frozen-source.json').read_text());j=json.loads((b/'source-build.json').read_text());assert j['success']
 cfg=json.loads((s/'configure-support-source.json').read_text())['files'] if (s/'configure-support-source.json').exists() else {}
 artifacts=j['artifacts_sha256']
 def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in artifacts.items()) and all(sha(s/n)==h for n,h in cfg.items())
 assert verify()
 workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')]
 assert not workers,workers
 c=q/'native'/name;c.mkdir(exist_ok=False);a=[str(binary)];env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/Users/n','NUMI_MATTER_METALLIB':str(lib)}
 if lib is None:env.pop('NUMI_MATTER_METALLIB')
 cmd={'schema':'numi.cardboard.native-component-command.v2','case':name,'argv':a,'environment':env,'source_snapshot':str(s),'source_frozen_sha256':sh,'source_build':str(b/'source-build.json'),'source_build_sha256':bh,'artifacts':artifacts,'all_inputs_verified':True,'workers':workers,'physical_transactions':0,'prediction':'No pass assumed; manufactured operator or kernel contract only; no full nonlinear trajectory or shipping qualification.'}
 (c/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 t=time.monotonic();timed=False;code=None
 with (c/'stdout.log').open('xb') as out,(c/'stderr.log').open('xb') as err:
  child=subprocess.Popen(a,stdout=out,stderr=err,env=env,start_new_session=True)
  try:code=child.wait(timeout=120)
  except subprocess.TimeoutExpired:
   timed=True;os.killpg(child.pid,signal.SIGTERM)
   try:code=child.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
 r={'schema':'numi.cardboard.native-component-receipt.v2','case':name,'exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'stdout_sha256':sha(c/'stdout.log'),'stderr_sha256':sha(c/'stderr.log'),'physical_transactions':0,'full_solver_or_shipping_qualification':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (c/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
 return code or int(timed or not r['all_inputs_unchanged'])
for number in sys.argv[1:]:
 code=run(number)
 if code:sys.exit(code)
