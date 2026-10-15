from pathlib import Path
import hashlib
import json
import shutil
import subprocess

BASE = Path('/Users/home/cardboard-implementation-20261014')
REPO = Path('/Users/home/numi-cardboard-20261006')
OLD = BASE / 'snapshots/source-whole-mortar-proof-103'
NEW = BASE / 'snapshots/source-contact-health-diagnostics-104'
OUT = BASE / 'analysis/source104-diagnostic-integration-001'
HEALTH = BASE / 'analysis/armijo-health-metadata-overlay-002'
MARKER = BASE / 'analysis/source103-vt-failure-marker-001'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert not NEW.exists()
OUT.mkdir(exist_ok=False)
old_manifest = json.loads((OLD / 'frozen-source.json').read_text())
support = json.loads((OLD / 'configure-support-source.json').read_text())
for rel, digest in old_manifest['source_sha256'].items():
    assert sha(OLD / rel) == digest, rel
for rel, digest in support['files'].items():
    assert sha(OLD / rel) == digest, rel
health = json.loads((HEALTH / 'manifest.json').read_text())
for row in health['artifact_inventory']['files']:
    path = HEALTH / row['path']
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
marker = json.loads((MARKER / 'manifest.json').read_text())
assert sha(MARKER / 'patch.diff') == marker['sha256']['patch']
assert sha(MARKER / 'overlay/matter/src/metal/contact.metalinc') == marker['sha256']['overlay_contact_metal']

shutil.copytree(OLD, NEW, symlinks=False)
patch_records = []
for patch in [HEALTH / 'overlay.patch', MARKER / 'patch.diff']:
    result = subprocess.run(['patch', '-p1', '--batch', '--forward', '-i', str(patch)], cwd=NEW,
                            capture_output=True, text=True)
    patch_records.append(dict(path=str(patch), sha256=sha(patch), exit_code=result.returncode,
                              stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0, result.stderr + result.stdout
for rel, digest in health['overlay_files'].items():
    if rel.startswith('matter/src/'):
        assert sha(NEW / rel) == digest, rel
assert sha(NEW / 'matter/src/metal/contact.metalinc') == marker['sha256']['overlay_contact_metal']

selected = {rel: sha(NEW / rel) for rel in old_manifest['source_sha256']}
changes = {rel: dict(before=old_manifest['source_sha256'][rel], after=digest)
           for rel, digest in selected.items() if old_manifest['source_sha256'][rel] != digest}
assert set(changes) == {'matter/src/runtime.mm', 'matter/src/metal/fem.metalinc', 'matter/src/metal/contact.metalinc'}
scope = ('Diagnostic-only integration of pre-Armijo solver status snapshots, health-aware raw residual labels, '
         'and disjoint generic VT failure markers. No physical operator, CCD threshold, solver tolerance, '
         'contact iteration budget, acceptance rule, or history update changes. Native parity remains required.')
(NEW / 'frozen-source.json').write_text(json.dumps({**old_manifest, 'parent_revision':
    '1648fe0e0483b5b05ce3c1e64d78cbec4885899e', 'base_snapshot': str(OLD),
    'scope': scope, 'source_sha256': selected}, indent=2) + '\n')
extra = {str(p.relative_to(NEW)): sha(p) for p in sorted(NEW.rglob('*')) if p.is_file()
         and str(p.relative_to(NEW)) not in selected
         and str(p.relative_to(NEW)) not in {'frozen-source.json', 'configure-support-source.json'}}
(NEW / 'configure-support-source.json').write_text(json.dumps(dict(
    schema='numi.cardboard.configure-support-source.v1',
    scope='Additional unchanged source103 root-CMake configuration context; selected source bound separately.',
    files=extra), indent=2) + '\n')
record = dict(schema='numi.cardboard.source104-diagnostic-integration.v1', scope=scope,
    base_snapshot=str(OLD), base_manifest_sha256=sha(OLD / 'frozen-source.json'), snapshot=str(NEW),
    selected_source_files=len(selected), support_files=len(extra), changes=changes,
    patches=patch_records, health_manifest_sha256=sha(HEALTH / 'manifest.json'),
    vt_marker_manifest_sha256=sha(MARKER / 'manifest.json'), native_parity='pending')
(OUT / 'integration.json').write_text(json.dumps(record, indent=2) + '\n')
shutil.copy2(__file__, OUT / 'assemble.py')
cmd = [v.replace(str(OLD), str(NEW)).replace('build-cardboard-provenance-103', 'build-cardboard-provenance-104')
       for v in json.loads((BASE / 'build-command-103.json').read_text())]
(BASE / 'build-command-104.json').write_text(json.dumps(cmd, indent=2) + '\n')
print('source104 frozen:', len(selected), 'selected files;', len(extra), 'support files', flush=True)
with (BASE / 'build104-runner.log').open('x') as log:
    code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
for rel, digest in {**selected, **extra}.items():
    assert sha(NEW / rel) == digest, rel
(OUT / 'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,
    source_inputs_unchanged=True, source_build_sha256=sha(REPO / 'build-cardboard-provenance-104/source-build.json')),
    indent=2) + '\n')
print('build104', code, flush=True)
raise SystemExit(code)
