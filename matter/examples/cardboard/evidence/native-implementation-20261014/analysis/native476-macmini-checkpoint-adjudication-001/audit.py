from pathlib import Path
import csv
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
C = B / 'solver/macmini-warm-cgls-control0-prefix-476'
O = B / 'analysis/native476-macmini-checkpoint-adjudication-001'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def props(p):
    d = {}
    for line in p.read_text().splitlines():
        k, sep, v = line.partition('=')
        assert sep and k not in d
        d[k] = v
    return d

assert not O.exists()
T = B / 'analysis/closed-macmini-transfer-inventories-031-001'
transfer = json.loads((T / 'report.json').read_text())['cases']['native476']
assert transfer['all_mirrored_bytes_match_remote'] and transfer['mirrored_files'] == 19
assert sha(T / 'native476-remote-inventory.json') == transfer['inventory_sha256']
inventory = json.loads((T / 'native476-remote-inventory.json').read_text())
files = {r['path']: r for r in inventory['files']}
for key, row in files.items():
    if key.startswith('inputs/'):
        continue
    p = C / key
    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], key
receipt = json.loads((C / 'receipt.json').read_text())
assert receipt['exit_code'] == 0 and not receipt['timed_out']
assert receipt['source_inputs_unchanged'] and not receipt['changed_inputs']
result = json.loads((C / 'output/result.json').read_text())
assert result['status'] == 'accepted_checkpoint_prefix_saved' and result['accepted_steps'] == 1
prefix = json.loads((C / 'output/coupon-checkpoint-prefix.json').read_text())
checkpoint = C / 'output' / prefix['checkpoint_file']
meta_path = C / 'output' / prefix['metadata_file']
assert sha(checkpoint) == prefix['checkpoint_sha256']
assert sha(meta_path) == prefix['metadata_sha256']
meta = props(meta_path)
assert meta['checkpoint_sha256'] == sha(checkpoint)
assert int(meta['checkpoint_bytes']) == checkpoint.stat().st_size
assert meta['accepted_control_index'] == '0' and meta['next_control_index'] == '1'
assert meta['environment_count'] == '2' and meta['both_environments_accepted'] == 'true'
assert meta['node_count'] == '882' and meta['tetrahedron_count'] == '1920'
assert meta['source_physics_fingerprint'] == '5247741070077684697'
assert meta['protocol_manifest_sha256'] == sha(C / 'output/manifest.json')
pm = checkpoint.with_suffix('.prefix')
assert meta['prefix_manifest_sha256'] == sha(pm)
prefix_files = []
for line in pm.read_text().splitlines():
    if not line.startswith('file='):
        continue
    key, size, digest, observation = line[5:].split('|')
    path = Path(key)
    assert not path.is_absolute() and '..' not in path.parts
    p = C / 'output' / path
    assert p.stat().st_size == int(size) and sha(p) == digest, key
    prefix_files.append(key)
assert len(prefix_files) == 7
rows = list(csv.DictReader((C / 'output/observations.csv').open()))
assert len(rows) == 2 and {r['environment'] for r in rows} == {'0', '1'}
for r in rows:
    assert r['step'] == '0' and r['status_code'] == '0' and r['step_accepted'] == '1'
    assert r['certificate_raw_accepted_flag'] == '1' and r['completed_microsteps'] == '1'
    assert 0 <= float(r['certificate_residual']) <= 5e-7
    assert float(r['min_J']) > 0
O.mkdir()
report = {'schema': 'numi.cardboard.native476-checkpoint-adjudication.v1',
    'verdict': 'PASS_accepted_control0_checkpoint_write_and_integrity',
    'host': 'macmini M4 Pro 24 GiB; distinct from original M4',
    'source173_frozen_sha256': '6d7e76fdb07702b0b0727fc6610ae979c4e0e161ad92f6888036b1dfef3b53b3',
    'source_build_sha256': '96396de7fa14197a6759583d8b7bb053c412e08f1051b947502c1cc38b185229',
    'checkpoint_sha256': sha(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
    'metadata_sha256': sha(meta_path), 'prefix_manifest_sha256': sha(pm),
    'prefix_files_verified': prefix_files, 'observations': rows,
    'remote_transfer_inventory_sha256': transfer['inventory_sha256'],
    'duplicate_input_files_omitted_locally': 813,
    'duplicate_input_bytes_preserved_remote': True,
    'full_protocol_complete': False, 'restore_and_continuation_qualified': False,
    'uninterrupted_trajectory_parity': False, 'numerical_convergence': False,
    'physical_box_qualification': False,
    'limitation': 'One tiny end-grip coupon control, no scoring or self-contact. Native477 restore/continuation is a separate active run, excluded here.'}
(O / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
(O / 'audit.py').write_bytes(Path(__file__).read_bytes())
(O / 'manifest.json').write_text(json.dumps({'files_sha256': {p.name: sha(p) for p in sorted(O.iterdir())}}, indent=2) + '\n')
print(json.dumps({'verdict': report['verdict'], 'checkpoint_sha256': sha(checkpoint),
    'prefix_files_verified': len(prefix_files), 'accepted_environments': len(rows)}))
