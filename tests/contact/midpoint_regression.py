#!/usr/bin/env python3
"""Fixed-mesh CPU diagnostic gate; not a production or mesh qualification."""
import json, pathlib, subprocess, sys
exe=sys.argv[1]
reference=json.loads((pathlib.Path(__file__).parents[2]/'docs/evidence/contact-area-reference-20261107/fp64-fem-h0.5us.json').read_text())
x=json.loads(subprocess.run([exe,'0.000015625','0.01'],capture_output=True,text=True,check=True).stdout)
force=max(abs(a-b) for a,b in zip(x['force_bins_N'],reference['force_bins_N']))/max(reference['force_bins_N'])
impulse=abs(x['impulse_Ns']-reference['impulse_Ns'])/abs(reference['impulse_Ns'])
assert force<=.0025 and impulse<=.0025,(force,impulse)
assert x['minimum_gap_m']>1e-7 and x['forced_rejection_left_state_unchanged']
assert x['max_total_energy_defect_J']<1e-8 and abs(x['contact_work_plus_potential_change_J'])<1e-8
assert x['max_step_contact_work_plus_potential_change_J']<1e-8
print('CPU midpoint fixed-mesh force/impulse:',force,impulse,'clearance:',x['minimum_gap_m'])
