from pathlib import Path
import hashlib
import json

base = Path('/Users/home/cardboard-implementation-20261014')
left = base / 'solver/rationalized-paper-complete-coupon-cycle-303'
right = base / 'solver/rationalized-paper-control2-newton63-capture-304'
out = base / 'analysis/control2-operator-capture-parity-304-001'
out.mkdir(exist_ok=False)
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = []
for path in sorted((left / 'output').rglob('*')):
    if not path.is_file():
        continue
    rel = path.relative_to(left / 'output')
    other = right / 'output' / rel
    a, b = path.read_bytes(), other.read_bytes()
    row = dict(path=str(rel), bytes_a=len(a), bytes_b=len(b),
               sha256_a=sha(a), sha256_b=sha(b), byte_identical=a == b)
    rows.append(row)
physical = [r for r in rows if r['path'] != 'manifest.json']
assert all(r['byte_identical'] for r in physical), [r for r in physical if not r['byte_identical']]
ma, mb = [json.loads((p / 'output/manifest.json').read_text()) for p in (left, right)]
def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [r for k in sorted(a.keys() | b.keys())
                for r in differences(a.get(k), b.get(k), prefix + '/' + k)]
    if a != b:
        return [dict(path=prefix, a=a, b=b)]
    return []
diff = differences(ma, mb)
report = dict(schema='numi.cardboard.operator-capture-parity.v1',
              capture_off=str(left), capture_on=str(right),
              receipts=[json.loads((p / 'receipt.json').read_text()) for p in (left, right)],
              files=rows, manifest_differences=diff,
              production_payload_files=len(physical),
              production_payloads_byte_identical=True,
              complete_cycle=False, accepted_steps=2,
              qualification='Failed prefix only. Third loading step rejects in both runs; no complete loading/unloading/release history or convergence claim.')
(out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(out / 'audit.py').write_bytes(Path(__file__).read_bytes())
(out / 'README.md').write_text(
    '# Actual control2 capture parity\n\n'
    'The source091 capture-on304 and capture-off303 emitted byte-identical '
    'production payloads, observations, material state, accepted geometry, '
    'rejected diagnostic geometry and rejected conditioning arrays. Both '
    'accept two loading steps and reject the third. This is failed-prefix '
    'instrument parity; the complete coupon cycle remains unqualified. '
    'Manifest differences are listed explicitly in the report.\n')
print(json.dumps(dict(files=len(rows), physical_files=len(physical), manifest_differences=diff), indent=2))
