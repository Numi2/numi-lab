#!/usr/bin/env python3
"""Plot measured native release trajectories; never synthesize a folded shape."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('plastic', type=Path)
p.add_argument('elastic', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
if a.output.exists() or a.output.with_suffix('.json').exists():
    raise FileExistsError(a.output)
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
bindings = {}
for path, label, color in [(a.plastic, 'Plastic paper', '#9b6330'), (a.elastic, 'Elastic control', '#287d98')]:
    result = json.loads((path/'result.json').read_text())
    if result['status'] != 'completed':
        raise ValueError(f'{path}: incomplete trajectory')
    source = path/'observations.csv'
    raw = source.read_bytes()
    bindings[str(source)] = hashlib.sha256(raw).hexdigest()
    all_rows = list(csv.DictReader(raw.decode().splitlines()))
    if not all(int(r['step_accepted']) == 1 and int(r['status_code']) == 0 for r in all_rows):
        raise ValueError(f'{path}: rejected observation')
    rows = [r for r in all_rows if r['arm'] == 'bent']
    release = [r for r in rows if r['phase'] == 'release']
    if not release or any(int(r['right_grip_constrained']) for r in release):
        raise ValueError(f'{path}: missing native release')
    time = [1e3*float(r['time_s']) for r in rows]
    axes[0].plot(time, [float(r['measured_right_grip_angle_deg']) for r in rows], label=label, color=color)
    axes[1].plot(time, [1e3*float(r['max_free_speed_m_s']) for r in rows], label=label, color=color)
    dt = float(rows[1]['time_s'])-float(rows[0]['time_s'])
    onset = 1e3*(float(release[0]['time_s'])-dt)
    for ax in axes:
        ax.axvline(onset, color=color, alpha=.35, linestyle='--')
axes[0].set_ylabel('Measured right-end angle (degrees)')
axes[1].set_ylabel('Maximum free-node speed (mm/s)')
axes[1].set_xlabel('Native elapsed time (ms)')
axes[0].set_title('Released cardboard coupon — left grip remains fixed\nTransient motion; no settled-crease or physical-validation claim')
axes[0].legend()
for ax in axes:
    ax.grid(alpha=.2)
fig.savefig(a.output, dpi=160)
a.output.with_suffix('.json').write_text(json.dumps({'schema':'numi.cardboard.release-plot.v1','inputs':bindings,'release_marker':'time of last constrained accepted state','physical_validation':False,'image_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}, indent=2)+'\n')
