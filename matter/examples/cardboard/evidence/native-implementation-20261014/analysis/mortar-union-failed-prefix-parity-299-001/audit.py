from pathlib import Path
import hashlib, json, struct

base = Path('/Users/home/cardboard-implementation-20261014')
case = base / 'native/same-parent-mortar-original-four-arm-299'
root = case / 'diagnostic-state'
out = base / 'analysis/mortar-union-failed-prefix-parity-299-001'
out.mkdir(exist_ok=False)
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = []
rollback = []
for count in (1, 3):
    left, right = [root / f'face-count-{count}' / name for name in ('legacy', 'tiled')]
    names = sorted(p.name for p in left.iterdir() if p.is_file())
    assert names == sorted(p.name for p in right.iterdir() if p.is_file())
    for name in names:
        a, b = [(side / name).read_bytes() for side in (left, right)]
        row = dict(face_count=count, name=name, bytes_a=len(a), bytes_b=len(b),
                   sha256_a=sha(a), sha256_b=sha(b), byte_identical=a == b)
        if name.endswith('.bin'):
            ha, hb = struct.unpack_from('<IQQ', a), struct.unpack_from('<IQQ', b)
            assert ha[0] == hb[0] == 0
            row.update(source_fingerprint_a=f'{ha[1]:016x}', source_fingerprint_b=f'{hb[1]:016x}',
                       program_fingerprint_a=f'{ha[2]:016x}', program_fingerprint_b=f'{hb[2]:016x}',
                       bytes_outside_fingerprint_fields_identical=a[:4] + a[20:] == b[:4] + b[20:])
        rows.append(row)
    for layout, side in (('legacy', left), ('tiled', right)):
        attempts = [json.loads((side / f'transaction-{i:06d}.json').read_text()) for i in (0, 1)]
        assert attempts[0]['accepted'] is True and attempts[1]['accepted'] is False
        failure = attempts[1]
        a = failure['before']['accepted_state_vector_sha256']
        b = failure['after']['accepted_state_vector_sha256']
        assert a.keys() == b.keys()
        changed = [key for key in a if a[key] != b[key]]
        rollback.append(dict(face_count=count, layout=layout,
                             accepted_activation=True, rejected_closure=True,
                             accepted_state_vectors=len(a), changed_accepted_state_vectors=changed,
                             rejected_certificate_validity=failure['after']['solver_certificates'][0]['validity']))
assert all(row.get('bytes_outside_fingerprint_fields_identical', row['byte_identical']) for row in rows if row['name'].startswith('transaction-'))
assert all(not row['changed_accepted_state_vectors'] for row in rollback)
report = dict(schema='numi.cardboard.mortar-failed-prefix-parity.v1', run=str(case),
              receipt=json.loads((case / 'receipt.json').read_text()),
              interpretation='Accepted activation followed by strict closure rejection in every arm. Compare only emitted failed prefixes; no complete-sequence parity, contact qualification, calibration or physical assembly.',
              paired_files=rows, rollback=rollback)
(out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
(out / 'audit.py').write_bytes(Path(__file__).read_bytes())
(out / 'README.md').write_text('''# Mortar union failed-prefix audit299

All four arms accept bond activation and reject the following closure at the
unchanged nonlinear certificate. The transaction JSON files agree exactly
between legacy and tiled modes; raw snapshots are compared with fingerprint
fields reported separately. All emitted accepted-history vector hashes remain
unchanged across rejection. The failed certificate remains invalid.

This is failed-prefix parity and rollback evidence. No opening, shear,
completed closure sequence, full contact or physical assembly is qualified.
''')
print(json.dumps(dict(paired_files=len(rows), rollback_checks=rollback,
                      nonidentical=[dict(face_count=r['face_count'], name=r['name']) for r in rows if not r['byte_identical']])))
