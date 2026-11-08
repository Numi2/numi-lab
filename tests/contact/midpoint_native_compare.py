#!/usr/bin/env python3
"""Fixed-mesh force/impulse gates and matched-timestep FP64 comparison."""
import json
import math
import sys

native_path, cpu_path, reference_path = sys.argv[1:]
with open(native_path) as f:
    a = json.load(f)
with open(cpu_path) as f:
    c = json.load(f)
with open(reference_path) as f:
    r = json.load(f)
assert a['arm'] == 'ipc_area_midpoint'
assert a['h'] == c['h'] and a['divisions'] == 2
assert a['area_ipc_k_N_per_m5'] > 0
assert len(c['force_bins_N']) == len(r['force_bins_N']) == 10
assert all(math.isfinite(v) for v in c['force_bins_N'] + r['force_bins_N'])
bins = [0.0] * 10
for row in a['traces']:
    idx = min(9, int((row['step'] + .5) * a['h'] / .001))
    bins[idx] += row['force'] * a['h'] / .001
impulse = sum(row['accepted_normal_impulse'] for row in a['traces'])
def compare(x):
    return {
        'max_binned_force_relative_to_peak': max(abs(u-v) for u,v in zip(bins, x['force_bins_N'])) / max(x['force_bins_N']),
        'integrated_impulse_relative_error': abs(impulse - x['impulse_Ns']) / abs(x['impulse_Ns']),
    }
assert len(a['final_nodes']) == len(c['final_nodes'])
position = max(abs(u[j] - v[j]) for u,v in zip(a['final_nodes'],c['final_nodes']) for j in range(3))
velocity = max(abs(u[j] - v[j]) for u,v in zip(a['final_nodes'],c['final_nodes']) for j in range(3,6))
matched = compare(c)
reference = compare(r)
gates = {
    'completed_all_steps': a['accepted_steps'] == round(.01/a['h']) and a['rejected_steps'] == 0,
    'clearance': a['minimum_gap'] > 1e-7,
    'force_reference_0p25_percent': reference['max_binned_force_relative_to_peak'] <= .0025,
    'impulse_reference_0p25_percent': reference['integrated_impulse_relative_error'] <= .0025,
    'contact_law_0p01_percent': a['law_max_relative_error'] <= 1e-4,
    'local_momentum': a['independent_max_local_scaled_residual'] <= .005,
    'kinematics_m': a['position_velocity_max_defect_m'] <= 1e-11,
    'matched_cpu_node_position_m': position <= 1e-8,
    'matched_cpu_node_velocity_mps': velocity <= 1e-5,
    'replay_and_restore': a['restore_replay'] and a['continuation_replay'] and a['forced_rollback_exact'] and a['identity_mismatch_rejected'],
    'nonzero_contact': a['maximum_accepted_multiplier_impulse_Ns'] > 0,
}
result = {'native_h':a['h'], 'matched_cpu':matched, 'independent_refined':reference,
          'max_node_position_difference_m':position,'max_node_velocity_difference_mps':velocity,
          'native_contact_work_J':a['contact_work_J'],'cpu_contact_work_J':c['contact_work_J'],
          'gates':gates}
print(json.dumps(result, indent=2, sort_keys=True))
sys.exit(0 if all(gates.values()) else 1)
