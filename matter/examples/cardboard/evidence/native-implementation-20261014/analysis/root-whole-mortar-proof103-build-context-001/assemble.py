from pathlib import Path
import hashlib,json,subprocess,shutil,sys
b=Path('/Users/home/cardboard-implementation-20261014');repo=Path('/Users/home/numi-cardboard-20261006');old=b/'snapshots/source-boundary-reset-order-101';incomplete=b/'snapshots/source-whole-mortar-proof-102';new=b/'snapshots/source-whole-mortar-proof-103';o=b/'analysis/root-whole-mortar-proof103-build-context-001';o.mkdir(exist_ok=False);assert not new.exists();h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=json.loads((old/'frozen-source.json').read_text());selected=json.loads((incomplete/'frozen-source.json').read_text());assert set(base['source_sha256'])==set(selected['source_sha256'])
for rel,sha in base['source_sha256'].items():assert h(old/rel)==sha,rel
shutil.copytree(old,new,symlinks=False)
changes={}
for rel,sha in selected['source_sha256'].items():
 assert h(incomplete/rel)==sha,rel
 if sha!=base['source_sha256'][rel]:
  shutil.copy2(incomplete/rel,new/rel);changes[rel]=dict(before=base['source_sha256'][rel],after=sha)
scope=selected['scope']+' Snapshot102 configure failed because its narrow selected-source copy omitted unrelated root-CMake target source files. Source103 retains exact101 configure context and exact102 selected source bytes; failure102 remains retained.'
(new/'frozen-source.json').write_text(json.dumps({**selected,'base_snapshot':str(old),'scope':scope},indent=2)+'\n')
extra={str(p.relative_to(new)):h(p)for p in sorted(new.rglob('*'))if p.is_file()and str(p.relative_to(new))not in selected['source_sha256']and p.name!='frozen-source.json'}
(new/'configure-support-source.json').write_text(json.dumps(dict(schema='numi.cardboard.configure-support-source.v1',scope='All additional files copied from101 solely to retain root CMake configuration context; selected compile/runtime source is separately bound by frozen-source and source-build.',files=extra),indent=2)+'\n')
receipt=dict(schema='numi.cardboard.configure-context-repair.v1',failed_snapshot=str(incomplete),failed_build=str(repo/'build-cardboard-provenance-102/source-build.json'),failed_build_sha256=h(repo/'build-cardboard-provenance-102/source-build.json'),snapshot=str(new),selected_source_identical_to102=True,base101_manifest_sha256=h(old/'frozen-source.json'),changed_from101=changes,extra_configure_context_files=len(extra),extra_manifest_sha256=h(new/'configure-support-source.json'),scope=scope)
(b/'source-overlay-103.json').write_text(json.dumps(receipt,indent=2)+'\n');(o/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');(o/'assemble.py').write_bytes(Path(__file__).read_bytes());cmd=json.loads((b/'build-command-102.json').read_text());cmd=[v.replace(str(incomplete),str(new)).replace('build-cardboard-provenance-102','build-cardboard-provenance-103')for v in cmd];(b/'build-command-103.json').write_text(json.dumps(cmd,indent=2)+'\n');print('frozen103 selected778 extra',len(extra),flush=True)
with(b/'build103-runner.log').open('x')as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
for rel,sha in extra.items():assert h(new/rel)==sha,rel
for rel,sha in selected['source_sha256'].items():assert h(new/rel)==sha,rel
(o/'post-build-context-check.json').write_text(json.dumps(dict(all_selected_and_support_inputs_unchanged=True,build_exit_code=code,build_receipt_sha256=h(repo/'build-cardboard-provenance-103/source-build.json')),indent=2)+'\n');print('build103',code,flush=True);raise SystemExit(code)
