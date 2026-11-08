#!/usr/bin/env python3
"""One borrowed-command-buffer transaction, including publication and rollback."""
import json
import subprocess
import sys

exe, shader = sys.argv[1:3]

def run(args, expect_success):
    proc = subprocess.run([exe, shader, *args], text=True, capture_output=True)
    if (proc.returncode == 0) != expect_success:
        raise AssertionError((proc.returncode, proc.stdout, proc.stderr))
    try:
        x = json.loads(proc.stdout)
    except ValueError as error:
        raise AssertionError(proc.stderr) from error
    assert x['restore_replay'] and x['continuation_replay']
    assert x['forced_rollback_exact'] and x['identity_mismatch_rejected']
    assert x['rejected_state_unchanged']
    return x

common = ['0.0000625', '1', '2']
noncontact = run(common + ['ipc_area_midpoint', '0.002', 'physical', 'contact', 'reference'], True)
contact = run(common + ['ipc_area_midpoint', '0.00009', 'physical', 'contact', 'reference'], True)
control = run(common + ['ipc_area', '0.00009', 'physical', 'contact', 'reference'], True)
for x in (noncontact, contact, control):
    assert x['accepted_steps'] == 1 and x['rejected_steps'] == 0
    assert x['position_velocity_max_defect_m'] < 1e-11
    assert x['independent_max_local_scaled_residual'] < .005
    assert x['momentum_max_absolute_error_Ns'] < 1e-7
    assert x['law_max_relative_error'] < 1e-4
assert noncontact['traces'][0]['force'] == 0
assert contact['traces'][0]['force'] > 0 and contact['maximum_accepted_multiplier_impulse_Ns'] > 0
assert abs(contact['contact_work_J'] - contact['contact_impulse_midvelocity_work_J']) < 1e-11
assert control['source_fingerprint'] == contact['source_fingerprint']
assert control['device_fingerprint'] != contact['device_fingerprint']
assert control['area_ipc_k_N_per_m5'] == contact['area_ipc_k_N_per_m5']

# The midpoint remains above the plane; the reflected endpoint crosses it.
crossing = run(common + ['ipc_area_midpoint', '0.00008', 'physical', 'contact', 'reference', '1', '-1.5'], False)
assert crossing['accepted_steps'] == 0 and crossing['traces'][0]['status'] == 6
assert crossing['traces'][0]['status_values'][0] == 2

# A looser diagnostic tolerance lets the reflected endpoint reach element
# validation; it is not used in the physical-accuracy fixture.
inversion = run(common + ['ipc_area_midpoint', '1.0', 'physical', 'contact', 'reference', '1e16', '0', '24000', '10000'], False)
assert inversion['accepted_steps'] == 0 and inversion['traces'][0]['status'] == 5
assert inversion['traces'][0]['status_values'][0] < 0

# Unsupported coordinate and convergence policies must fail at initialization.
for bad in (common + ['ipc_area_midpoint', '0.00009', 'physical', 'contact', 'world'],
            common + ['ipc_area_midpoint', '0.00009', 'legacy', 'contact', 'reference']):
    proc = subprocess.run([exe, shader, *bad], text=True, capture_output=True)
    assert proc.returncode != 0 and 'initialize:' in proc.stderr
print('native midpoint: freefall, contact, replay, rollback, crossing and inversion passed')
