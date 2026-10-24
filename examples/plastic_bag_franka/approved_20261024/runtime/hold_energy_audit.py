"""Measure stationary-gripper work and energy in the saved native ledger."""
import csv
import json
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open()))
start = next(r for r in rows if int(r['step']) == 1248)
hold = [r for r in rows if int(r['step']) > 1248]
end = hold[-1]
work = sum(float(r['signed_actuator_work_J']) for r in hold)
change = float(end['system_energy_J'])-float(start['system_energy_J'])
print(json.dumps({
    'start_step': 1248, 'end_step': int(end['step']),
    'start_time_s': 5.2, 'end_time_s': float(end['time_s']),
    'accepted_hold_intervals': len(hold),
    'signed_prescribed_gripper_work_J': work,
    'system_energy_change_J': change,
    'net_work_energy_balance_J': change-work,
    'largest_accepted_interval_balance_J': max(float(r['frame_energy_balance_J']) for r in hold),
    'all_interval_balances_within_tolerance': all(float(r['frame_energy_balance_J']) <= float(r['local_tolerance_J'])+1e-12 for r in hold),
    'method': 'Native ledger includes translational and rotational kinetic energy, gravity, membrane and bending energy, and compressed gripper-pad energy. The prescribed gripper is stationary after state1248; no velocity or energy rescaling is applied.',
}, indent=2))
