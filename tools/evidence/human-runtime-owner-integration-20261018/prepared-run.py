from pathlib import Path
import os,subprocess,json,hashlib,time
p=Path(__file__).parent.parent;b=Path(__file__).parent
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
e=os.environ.copy()
for k,n in [('NUMI_MATTER_TEST_SOURCE','MatterSource.metal'),('NUMANX_HUMAN_MATTER_TEST_SOURCE','HumanMatterSource.metal'),('NUMI_HUMAN_STAND_SOURCE','StandSource.metal'),('METALROBO_TEST_ARTICULATED_SOURCE','operator-modules.tsv')]:
 q=p/'shaders'/n;e[k]=str(q);e[k+'_SHA256']=sha(q)
q=Path('/Users/n/numi-human-separation-20261018/C/humanmatter-extension-181/w-current-build-001/build/deferred-probe-matched-current-main-004/NumanXCoupledHuman-181-current-main.metal');e['NUMANX_COUPLED_HUMAN_TEST_SOURCE']=str(q);e['NUMANX_COUPLED_HUMAN_TEST_SOURCE_SHA256']=sha(q)
e['NUMI_MATTER_METALLIB']=str(p/'shaders/MatterSource.metal')
t=time.monotonic();a=[str(b/'probe')];r=subprocess.run(a,env=e,cwd=b,capture_output=True,text=True);(b/'native.log').write_text(r.stdout+r.stderr);(b/'native-receipt.json').write_text(json.dumps({'argv':a,'environment':{k:v for k,v in e.items() if k.startswith(('NUMI_','NUMANX_','METALROBO_'))},'binary_sha256':sha(b/'probe'),'library_sha256':sha(p/'full-build/lib/libmetalrobo.dylib'),'seconds':time.monotonic()-t,'exit':r.returncode},indent=2));print(r.returncode,r.stdout[-6000:],r.stderr[-2000:]);raise SystemExit(r.returncode)
