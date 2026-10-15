from pathlib import Path
import hashlib
import json
import shutil

B = Path('/Users/home/cardboard-implementation-20261014')
R = Path('/Users/home/numi-cardboard-20261006')
S = B / 'snapshots/source-coupon-cache-admission-139'
OLD = B / 'snapshots/source-repeated-cauchy-fallback-137'
O = B / 'analysis/source139-root-integration-preflight-001'
assert not O.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = json.loads((S / 'frozen-source.json').read_text())['source_sha256']
old = json.loads((OLD / 'frozen-source.json').read_text())['source_sha256']
different = {p for p, h in files.items() if sha(R / p) != h}
assert different == {'matter/src/runtime.mm', 'matter/src/metal/fem.metalinc'}
assert all(sha(R / p) == h for p, h in old.items())
build = R / 'build-cardboard-provenance-139/source-build.json'
assert json.loads(build.read_text())['success']
parity = B / 'analysis/native441-cache-reuse-history-parity-001/report.json'
assert json.loads(parity.read_text())['verdict'] == 'PASS_for_exact_attempted_history_only'
retention = B / 'analysis/retained-source-snapshot139-001/manifest.json'
assert json.loads(retention.read_text())['snapshots'][0]['complete_disk_restoration_verified']
rows = [dict(path=p, before_sha256=sha(R / p), candidate_sha256=files[p]) for p in sorted(different)]
O.mkdir()
for p in sorted(different):
    assert sha(S / p) == files[p]
    shutil.copy2(S / p, R / p)
assert all(sha(R / p) == h for p, h in files.items())
(O / 'integration.json').write_text(json.dumps(dict(
    changed_files=rows, selected_source_files_verified=len(files),
    build_sha256=sha(build), native441_parity_sha256=sha(parity),
    source_retention_sha256=sha(retention),
    scope='Correct opt-in paper projection cache reuse admission for existing elastic starch and compiler IPC capability; original starch path and physical operator retained. Full attempted source139/cache-on history equals source133/cache-off native434 numerical/physical history. Both fail Newton48; complete coupon remains unqualified. Fix stale observer-cap labels without budget change.'
), indent=2) + '\n')
(O / 'integrate.py').write_bytes(Path(__file__).read_bytes())
print('root139', len(different), 'changed files;', len(files), 'source hashes verified')
