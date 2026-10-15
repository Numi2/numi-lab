from pathlib import Path
import hashlib, json, shutil, subprocess

B = Path('/Users/home/cardboard-implementation-20261014')
REPO = Path('/Users/home/numi-cardboard-20261006')
OLD = B / 'snapshots/source-contact-health-diagnostics-104'
NEW = B / 'snapshots/source-true-residual-certification-105'
P = B / 'analysis/true-residual-certification-overlay-105-v2'
O = B / 'analysis/source105-certification-integration-001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'manifest-v2.json') == '000a882a0c3b9402048bf8a8cc5bd91f6672ae4ac9fbd755a5b4e6bd50a77cae'
m = json.loads((P/'manifest-v2.json').read_text())
assert sha(P/'source-overlay.patch') == m['patch_sha256'] == '5c519fb8e86e884616aad687be1009b939c162e2d374e3f295c0d9ed1b8b68dc'
old = json.loads((OLD/'frozen-source.json').read_text())
support = json.loads((OLD/'configure-support-source.json').read_text())
for relative,digest in {**old['source_sha256'],**support['files']}.items():
    assert sha(OLD/relative) == digest,relative
for row in m['files']:
    assert sha(P/'source'/row['path']) == row['candidate_sha256'],row['path']
    if row['base_sha256'] is not None:
        assert sha(OLD/row['path']) == row['base_sha256'],row['path']
for row in m['validation_files']:
    assert sha(P/row['path']) == row['sha256'] and (P/row['path']).stat().st_size == row['bytes']
assert not NEW.exists(); O.mkdir(exist_ok=False)
shutil.copytree(OLD,NEW,symlinks=False)
r = subprocess.run(['patch','-p1','--batch','--forward','-i',str(P/'source-overlay.patch')],cwd=NEW,capture_output=True,text=True)
(O/'patch.stdout.log').write_text(r.stdout); (O/'patch.stderr.log').write_text(r.stderr)
assert r.returncode == 0,r.stdout+r.stderr
for row in m['files']:
    assert sha(NEW/row['path']) == row['candidate_sha256'],row['path']
selected = {relative:sha(NEW/relative) for relative in old['source_sha256']}
for row in m['files']: selected[row['path']] = sha(NEW/row['path'])
changes = {relative:dict(before=old['source_sha256'].get(relative),after=digest)
    for relative,digest in selected.items() if old['source_sha256'].get(relative) != digest}
assert set(changes) == {row['path'] for row in m['files']}
scope = ('Opt-in bounded native FGMRES true-residual certification and restart replacement. '
    'Same configured cached operator, unchanged forcing/absolute thresholds and final transactional certificates. '
    'Default mode unchanged; no full FEM/native convergence qualification from compilation or kernel fixture.')
(NEW/'frozen-source.json').write_text(json.dumps({**old,'base_snapshot':str(OLD),
    'scope':scope,'source_sha256':selected},indent=2)+'\n')
extra = {str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file()
    and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']}
(NEW/'configure-support-source.json').write_text(json.dumps(dict(schema='numi.cardboard.configure-support-source.v1',
    scope='Unchanged source104 additional root-CMake configuration context',files=extra),indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(schema='numi.cardboard.source105-certification-integration.v1',
    base_snapshot=str(OLD),snapshot=str(NEW),scope=scope,selected_source_files=len(selected),
    support_files=len(extra),changes=changes,overlay_manifest_sha256=sha(P/'manifest-v2.json'),
    patch_sha256=sha(P/'source-overlay.patch'),native_qualification='pending'),indent=2)+'\n')
shutil.copy2(__file__,O/'assemble.py')
cmd = [v.replace(str(OLD),str(NEW)).replace('build-cardboard-provenance-104','build-cardboard-provenance-105')
    for v in json.loads((B/'build-command-104.json').read_text())]
cmd.extend(['--target','numi-matter-fgmres-true-residual-native-check'])
(B/'build-command-105.json').write_text(json.dumps(cmd,indent=2)+'\n')
print('source105 frozen',len(selected),'selected files',len(extra),'support files',flush=True)
with (B/'build105-runner.log').open('x') as log:
    code = subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
for relative,digest in {**selected,**extra}.items(): assert sha(NEW/relative) == digest,relative
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,
    source_build_sha256=sha(REPO/'build-cardboard-provenance-105/source-build.json')),indent=2)+'\n')
print('build105',code,flush=True); raise SystemExit(code)
