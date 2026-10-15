from pathlib import Path
import hashlib,json,shutil,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');OLD=B/'snapshots/source-capture-callback-lifetime-106';NEW=B/'snapshots/source-ee-ccd-continuation-107';P=B/'analysis/source106-ccd33-direct-regression-overlay-001';O=B/'analysis/source107-ccd-integration-001';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'manifest.json')=='3d52b03d22bcc20af9669331b0215643734398e88c4bb43b99e59bd822e823c4'
m=json.loads((P/'manifest.json').read_text())
for rel,row in m['files'].items():assert sha(P/rel)==row['sha256'],rel
assert sha(OLD/'frozen-source.json')==m['base_frozen_source_json_sha256']
old=json.loads((OLD/'frozen-source.json').read_text());support=json.loads((OLD/'configure-support-source.json').read_text())
for rel,h in {**old['source_sha256'],**support['files']}.items():assert sha(OLD/rel)==h,rel
assert not NEW.exists();O.mkdir(exist_ok=False);shutil.copytree(OLD,NEW)
r=subprocess.run(['patch','-p1','--batch','--forward','-i',str(P/'source106-ccd33.patch')],cwd=NEW,capture_output=True,text=True);(O/'patch.stdout.log').write_text(r.stdout);(O/'patch.stderr.log').write_text(r.stderr);assert r.returncode==0
changes=['matter/src/metal/contact.metalinc','matter/tools/cardboard_contact_ccd_native_check.mm']
for rel in changes:assert sha(NEW/rel)==sha(P/'overlay'/rel)
selected={rel:sha(NEW/rel) for rel in old['source_sha256']};assert {rel for rel,h in selected.items() if h!=old['source_sha256'][rel]}==set(changes)
scope='Increase only conservative EE-CCD evaluation bound17to33; exact captured roundoff-band ray regression added. VT16 and distance/speed/roundoff/failclosed recurrence unchanged. Direct/native fullcontact same-build checks pending.'
(NEW/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(OLD),'scope':scope,'source_sha256':selected},indent=2)+'\n')
extra={str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file() and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']}
(NEW/'configure-support-source.json').write_text(json.dumps(dict(schema='numi.cardboard.configure-support-source.v1',scope='Unchanged source106 root-CMake context',files=extra),indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(schema='numi.cardboard.source107-ccd-integration.v1',snapshot=str(NEW),base_snapshot=str(OLD),scope=scope,selected_source_files=len(selected),support_files=len(extra),changes={p:dict(before=old['source_sha256'][p],after=selected[p]) for p in changes},overlay_manifest_sha256=sha(P/'manifest.json'),patch_sha256=sha(P/'source106-ccd33.patch'),native_qualification='pending'),indent=2)+'\n');shutil.copy2(__file__,O/'assemble.py')
cmd=[v.replace(str(OLD),str(NEW)).replace('provenance-106','provenance-107') for v in json.loads((B/'build-command-106.json').read_text())];(B/'build-command-107.json').write_text(json.dumps(cmd,indent=2)+'\n');print('source107frozen',len(selected),len(extra),flush=True)
with (B/'build107-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for rel,h in {**selected,**extra}.items():assert sha(NEW/rel)==h,rel
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-107/source-build.json')),indent=2)+'\n');print('build107',code,flush=True);raise SystemExit(code)
