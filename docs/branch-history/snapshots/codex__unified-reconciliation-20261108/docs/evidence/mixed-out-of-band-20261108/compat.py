from pathlib import Path
import hashlib,json,subprocess
p=Path(__file__).resolve().parent;repo=p.parents[2];prior=repo/'docs/evidence/mixed-joint-long-rod-20261108';work=Path('/Users/numi/Work/mixed-joint-long-rod-artifacts-20261108')
exe=work/'rod9';lib=Path('/Users/numi/numi-reconciliation-build-20261108/lib/libmetalrobo.dylib')
arms={'long':work/'long.metallib','old':Path('/Users/numi/Work/mixed-joint-contact-v2-artifacts-20261108/joint.metallib')}
def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(exe)=='330c4503b49e918ae765ef897f0ce282715a7214929bca228ddb72ba64d26538'
assert sha(lib)=='d8bd1d10177b72004f29360e2ec979b4c72ff3b8da2a708dfddfd0aa85e64e61'
for name,f in arms.items():
 manifest=prior/'artifacts.sha256' if name=='long' else prior.parent/'mixed-joint-contact-candidate-20261108/v2-artifacts.sha256'
 assert any(line.split()[0]==sha(f) and line.split(maxsplit=1)[1]==str(f) for line in manifest.read_text().splitlines())
def exposed(x):
 if isinstance(x,dict):return {k:exposed(v) for k,v in x.items() if k not in ('scratch','row_trace')}
 if isinstance(x,list):return [exposed(v) for v in x]
 return x
out={}
(p/'compat-inputs.sha256').write_text(''.join(sha(f)+'  '+str(f)+'\n' for f in (exe,lib,*arms.values())))
for name,shader in arms.items():
 r=subprocess.run([str(exe),str(shader),'2'],cwd=repo,capture_output=True,text=True)
 (p/('compat-'+name+'.stdout')).write_text(r.stdout);(p/('compat-'+name+'.stderr')).write_text(r.stderr)
 if r.returncode:out[name]={'returncode':r.returncode,'match':False};continue
 fresh=json.loads(r.stdout);old=json.loads((prior/'runs'/('rod9-'+name+'.stdout')).read_text())
 out[name]={'returncode':r.returncode,'match':exposed(fresh)==exposed(old),'success':fresh['success'],'failed':fresh['failed'],'repeat':fresh['repeat'],'exact_trace':fresh['exact_trace'],'capture_noninvasive':fresh['capture_noninvasive']}
(p/'compat-analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
if not all(v.get('match') and v.get('repeat') and v.get('exact_trace') and v.get('capture_noninvasive') for v in out.values()):raise SystemExit(1)
