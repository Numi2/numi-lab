from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess
b=Path('/Users/home/cardboard-implementation-20261014');old=b/'snapshots/source-current-newton-restart-augmentation-096';new=b/'snapshots/source-exact-mortar-seam-diagnostics-097';p=b/'analysis/private-mortar-ccd-314-overlay-001';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert h(p/'manifest.json')=='95142a62279f0f03a594ed58ff47fb9fd766be2ab12e9a9ee873ad77a4880140'
m=json.loads((p/'manifest.json').read_text());prior=json.loads((old/'frozen-source.json').read_text())
for rel,expected in prior['source_sha256'].items():assert h(old/rel)==expected,rel
for rel,r in m['files'].items():assert h(p/rel)==r['sha256'],rel
assert not new.exists();shutil.copytree(old,new)
r=subprocess.run(['patch','--batch','--forward','-p1','-i',str(p/'private-mortar-ccd-314.patch')],cwd=new,text=True,capture_output=True);assert r.returncode==0,r.stderr+r.stdout
spec=importlib.util.spec_from_file_location('bound',new/'matter/tools/cardboard_source_bound_build.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);bindings=module.source_bindings(new)
scope='Source096 plus exact paired mortar seam identity, conservative exact shared-edge guard, corrected positive-gap synthetic fixtures, initial-overlap negative control and partition-stage diagnostics. No contact tolerances changed; native qualification pending.'
(new/'frozen-source.json').write_text(json.dumps({**prior,'base_snapshot':str(old),'source_sha256':bindings,'scope':scope},indent=2)+'\n')
(b/'source-overlay-097.json').write_text(json.dumps(dict(schema='numi.cardboard.integrated-source-overlay.v1',base_snapshot=str(old),snapshot=str(new),packages=[dict(path=str(p),manifest_sha256=h(p/'manifest.json'))],patch_result=dict(stdout=r.stdout,stderr=r.stderr,exit_code=r.returncode),changed={k:dict(before=prior['source_sha256'].get(k),after=v) for k,v in bindings.items() if prior['source_sha256'].get(k)!=v},scope=scope),indent=2)+'\n')
cmd=[str(Path('/Users/home/.pyenv/versions/3.10.12/bin/python3')),str(new/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(new),'--build-dir','/Users/home/numi-cardboard-20261006/build-cardboard-provenance-097','--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-full-contact-armijo-check','--target','numi-matter-cardboard-probe'];(b/'build-command-097.json').write_text(json.dumps(cmd,indent=2)+'\n');print('frozen097',len(bindings),flush=True)
with (b/'build097-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
print('build097exit',code,flush=True);raise SystemExit(code)
