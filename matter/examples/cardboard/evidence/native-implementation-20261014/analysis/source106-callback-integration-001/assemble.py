from pathlib import Path
import hashlib
import json
import shutil
import subprocess

B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
OLD=B/'snapshots/source-true-residual-certification-105'
NEW=B/'snapshots/source-capture-callback-lifetime-106'
P=B/'analysis/source105-capture-callback-lifetime-overlay-001'
O=B/'analysis/source106-callback-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'manifest.json')=='459015461a2cb0e0eaf255d877645bbf3497fe88748051b9dbeec46fca3b3f47'
m=json.loads((P/'manifest.json').read_text())
for rel,row in m['files'].items():
 assert sha(P/rel)==row['sha256'] and (P/rel).stat().st_size==row['bytes'],rel
assert sha(OLD/'frozen-source.json')==m['base_source_manifest_sha256']
assert sha(OLD/'matter/src/runtime.mm')==m['base_runtime_sha256']
old=json.loads((OLD/'frozen-source.json').read_text())
support=json.loads((OLD/'configure-support-source.json').read_text())
for rel,h in {**old['source_sha256'],**support['files']}.items():assert sha(OLD/rel)==h,rel
assert not NEW.exists();O.mkdir(exist_ok=False)
shutil.copytree(OLD,NEW)
q=subprocess.run(['patch','-p1','--batch','--forward','-i',str(P/'source-overlay.patch')],cwd=NEW,capture_output=True,text=True)
(O/'patch.stdout.log').write_text(q.stdout);(O/'patch.stderr.log').write_text(q.stderr)
assert q.returncode==0,q.stdout+q.stderr
assert sha(NEW/'matter/src/runtime.mm')==m['output_runtime_sha256']
assert sha(NEW/'matter/src/coupled_capture_manifest_context.hpp')==m['output_context_header_sha256']
selected={rel:sha(NEW/rel) for rel in old['source_sha256']}
selected['matter/src/coupled_capture_manifest_context.hpp']=m['output_context_header_sha256']
scope='Own async coupled-capture metadata and snapshot lifetime; preserves solver105 operator and opt-in certification behavior. No physical change. Native metadata and production parity remain required.'
(NEW/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(OLD),'scope':scope,'source_sha256':selected},indent=2)+'\n')
extra={str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file() and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']}
(NEW/'configure-support-source.json').write_text(json.dumps(dict(schema='numi.cardboard.configure-support-source.v1',scope='Unchanged source105 root-CMake context',files=extra),indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(schema='numi.cardboard.source106-callback-integration.v1',snapshot=str(NEW),base_snapshot=str(OLD),scope=scope,selected_source_files=len(selected),support_files=len(extra),changes={p:dict(before=old['source_sha256'].get(p),after=h) for p,h in selected.items() if h!=old['source_sha256'].get(p)},overlay_manifest_sha256=sha(P/'manifest.json'),patch_sha256=sha(P/'source-overlay.patch'),native_qualification='pending'),indent=2)+'\n')
shutil.copy2(__file__,O/'assemble.py')
cmd=[v.replace(str(OLD),str(NEW)).replace('provenance-105','provenance-106') for v in json.loads((B/'build-command-105.json').read_text())]
(B/'build-command-106.json').write_text(json.dumps(cmd,indent=2)+'\n')
print('source106frozen',len(selected),len(extra),flush=True)
with (B/'build106-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for rel,h in {**selected,**extra}.items():assert sha(NEW/rel)==h,rel
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-106/source-build.json')),indent=2)+'\n')
print('build106',code,flush=True);raise SystemExit(code)
