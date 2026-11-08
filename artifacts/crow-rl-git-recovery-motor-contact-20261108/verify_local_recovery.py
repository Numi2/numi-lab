from pathlib import Path
import hashlib, json, subprocess, sys
R = Path.cwd()
G = R / 'artifacts/crow-rl-git-recovery-motor-contact-20261108'
sys.path.insert(0, str(R / 'tools'))
from crow_evidence_archive import tree, Objects, content
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/agent/american-crow-rl-20261107'], text=True).split()[0]
assert head == remote
entries = tree()
with Objects() as objects:
    raw = objects.read(entries[str((G / 'local-inventory.json').relative_to(R))][1])
    local = json.loads(raw)
    plan = json.loads(objects.read(entries[str((G / 'plan.json').relative_to(R))][1]))
    manifest_raw = objects.read(entries[plan['manifest']][1])
    assert hashlib.sha256(manifest_raw).hexdigest() == plan['manifest_sha256']
    m = json.loads(manifest_raw)
    sources = {f['path']: f for f in m['files']}
    verified = []
    for f in local['files']:
        archived = sources['build/crow-motor-contact-build-01/local-drafts/' + f['relative_path']]
        data = b''.join(content(archived, entries, objects))
        assert len(data) == f['bytes'] and hashlib.sha256(data).hexdigest() == f['sha256']
        verified.append(f['path'])
report = dict(archive_commit=head, remote_commit=remote, manifest_sha256=hashlib.sha256(raw).hexdigest(),
              verified_files=verified, source_bytes=sum(f['bytes'] for f in local['files']))
print(json.dumps(report))
