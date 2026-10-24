"""Check the delivered videos against the complete native scene evidence."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

ROOT = Path('/Users/home/Documents/Codex/2026-10-21/i-wa')
W = ROOT / 'work/plastic_bag_franka_contact_finish_20261024'
OUT = ROOT / 'outputs/plastic_bag_franka_contact_finish_20261024'

def read(name):
    return json.loads((OUT / name).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

status = read('STATUS.json')
assert status['status'] == 'complete' and all(status['numerical_checks'].values())
checks = read('grasp_task_checks.json')
assert all(checks.values()), checks
grip = read('grip_alignment.json')
assert grip['all_forces_at_visible_stock_jaws']
assert grip['no_grip_force_during_first_drop']
assert read('robot_film_full.json')['intersection_state_count'] == 0
binding = read('physical_grip_binding.json')
native_sha = sha(OUT / 'trajectory.json')
assert binding['trajectory_sha256'] == native_sha
videos = {}
for record_name, filename in [
    ('video_record.json', 'air_drop_corner_grip_pickup_drop_8s_normal.mp4'),
    ('side_video_record.json', 'stock_grip_side_8s_normal.mp4'),
]:
    record = read(record_name)
    assert record['trajectory_sha256'] == native_sha
    assert record['video_sha256'] == sha(OUT / filename)
    assert record['decode_pass'] and record['end_native_state_included']
    assert record['probe']['streams'][0]['nb_frames'] == '241'
    videos[filename] = record['video_sha256']

checkpoint = OUT / 'combined/latest_checkpoint.bin'
data = checkpoint.read_bytes()
header = struct.unpack_from('<QQQdddQ', data)
assert header[0] == 0x4e554d4942473031 and header[1] == 1920
assert hashlib.sha256(data).hexdigest() == binding['checkpoint_sha256']
offset = 56
buffers = []
for _ in range(header[2]):
    size = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    buffers.append(data[offset:offset + size])
    offset += size
assert offset == len(data)
# Checkpoint order is the makeSharedBuffer order in the bound native host.
nodes = np.frombuffer(buffers[14], dtype='<f4').reshape(1081, 24)
coins = np.frombuffer(buffers[20], dtype='<f4').reshape(10, 44)
assert np.isfinite(nodes).all() and np.isfinite(coins).all()
assert (nodes[:, 3] > 0).all() and (coins[:, 3] > 0).all()
assert np.max(np.abs(nodes[:, 3].astype(float) * nodes[:, 7] - 1)) < 2e-7
assert np.max(np.abs(coins[:, 3].astype(float) * coins[:, 7] - 1)) < 2e-7

identity = {}
for line in (OUT / 'combined_identity_sha256.txt').read_text().splitlines():
    digest, name = line.split(None, 1)
    identity[name.strip()] = digest
for name in ['source/PlasticBag.metal', 'source/pair_mpr.metalinc',
             'source/plastic_bag_coins_scene.mm', 'source/franka_grip_gpu.h',
             'source/plastic_bag_gpu.h', 'grip.csv', 'robot.json']:
    assert sha(W / name) == identity[name], name

receipt = {
    'duration_s': 8, 'native_hz': 240, 'native_states': 1921,
    'trajectory_sha256': native_sha,
    'checkpoint_sha256': binding['checkpoint_sha256'],
    'film_nodes': len(nodes), 'dynamic_film_nodes': int((nodes[:, 3] > 0).sum()),
    'fixed_film_nodes': int((nodes[:, 3] <= 0).sum()),
    'film_mass_kg_from_gpu': float(nodes[:, 7].astype(float).sum()),
    'coin_mass_kg_each_from_gpu': coins[:, 7].astype(float).tolist(),
    'grasp_task_checks': checks, 'numerical_checks': status['numerical_checks'],
    'video_sha256': videos,
    'source_and_actuator_inputs_match_native_launch_manifest': True,
    'limits': [
        'Robot joint positions are prescribed; motor dynamics are not simulated.',
        'Robot housing clearance is audited at saved states, not continuous body contact.',
        'Film and pad material parameters are assumptions, not physical calibration.',
    ],
}
(OUT / 'delivery_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
