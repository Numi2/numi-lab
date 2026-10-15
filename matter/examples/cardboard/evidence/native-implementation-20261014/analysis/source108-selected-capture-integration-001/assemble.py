from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
OLD=B/'snapshots/source-ee-ccd-continuation-107';NEW=B/'snapshots/source-selected-capture-tail-reuse-108'
P=B/'analysis/graph-fd-selected-rebuild-overlay-108';O=B/'analysis/source108-selected-capture-integration-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'package_manifest.json')=='be2b503e33f207b3afe1321dbf9f40589040f03c54508c932ddb2e1fd30f090f'
assert sha(P/'source-overlay.json')=='f8229e9a9536319e969c1581320215c3a95444b4a82b2abc86237c2e537de915'
m=json.loads((P/'source-overlay.json').read_text())
assert sha(P/m['patch']['path'])==m['patch']['sha256']
old=json.loads((OLD/'frozen-source.json').read_text());support=json.loads((OLD/'configure-support-source.json').read_text())
for rel,h in {**old['source_sha256'],**support['files']}.items():assert sha(OLD/rel)==h,rel
for rel,h in m['base_files'].items():assert sha(OLD/rel)==h,rel
assert not NEW.exists();O.mkdir(exist_ok=False);shutil.copytree(OLD,NEW)
r=subprocess.run(['patch','-p1','--batch','--forward','-i',str(P/m['patch']['path'])],cwd=NEW,capture_output=True,text=True)
(O/'patch.stdout.log').write_text(r.stdout);(O/'patch.stderr.log').write_text(r.stderr);assert r.returncode==0
for rel,row in m['overlay_files'].items():assert sha(NEW/rel)==row['sha256']==sha(P/row['path']),rel
selected={rel:sha(NEW/rel) for rel in old['source_sha256']}
assert {rel for rel,h in selected.items() if h!=old['source_sha256'][rel]}==set(m['overlay_files'])
scope='Restrict diagnostic-induced graph-factor rebuild suppression to the exact pending selected root/Newton iteration. Default path and Metal kernels unchanged. Native failed-trajectory parity pending.'
(NEW/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(OLD),'scope':scope,'source_sha256':selected},indent=2)+'\n')
extra={str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file() and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']}
assert extra==support['files']
(NEW/'configure-support-source.json').write_text(json.dumps(dict(schema='numi.cardboard.configure-support-source.v1',scope='Unchanged source107 root-CMake context',files=extra),indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(schema='numi.cardboard.source108-selected-capture-integration.v1',snapshot=str(NEW),base_snapshot=str(OLD),scope=scope,selected_source_files=len(selected),support_files=len(extra),changes={p:dict(before=old['source_sha256'][p],after=selected[p]) for p in m['overlay_files']},overlay_manifest_sha256=sha(P/'package_manifest.json'),patch_sha256=sha(P/m['patch']['path']),native_qualification='pending'),indent=2)+'\n')
shutil.copy2(__file__,O/'assemble.py')
cmd=[v.replace(str(OLD),str(NEW)).replace('provenance-107','provenance-108') for v in json.loads((B/'build-command-107.json').read_text())]
(B/'build-command-108.json').write_text(json.dumps(cmd,indent=2)+'\n');print('source108 frozen',len(selected),len(extra),flush=True)
with (B/'build108-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for rel,h in {**selected,**extra}.items():assert sha(NEW/rel)==h,rel
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-108/source-build.json')),indent=2)+'\n')
print('build108',code,flush=True);raise SystemExit(code)
