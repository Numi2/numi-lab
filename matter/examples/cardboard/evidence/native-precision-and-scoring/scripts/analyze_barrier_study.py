"""Summarize retained native barrier trials; never infer success from partial rows."""
import csv, hashlib, json, math
from pathlib import Path
root = Path(__file__).resolve().parent
names = ['barrier50-scale1-001', 'barrier50-scale2-001', 'barrier50-scale4-001', 'barrier20-tight-001']
report = {'schema':'numi.cardboard.barrier-sensitivity.v1', 'physical_validation':False,
          'interpretation':'Numerical conditioning sensitivity, not calibrated scoring. Rejected rows excluded from force/gap extrema.',
          'runs': {}}
for name in names:
    run = root/name
    if not (root/'receipts'/name/'receipt.json').exists():
        report['runs'][name] = {'status':'not_completed'}
        continue
    manifest = json.loads((run/'manifest.json').read_text())
    result = json.loads((run/'result.json').read_text())
    receipt = json.loads((root/'receipts'/name/'receipt.json').read_text())
    rows = list(csv.DictReader((run/'observations.csv').open()))
    tools = list(csv.DictReader((run/'tool-observations.csv').open()))
    accepted = {int(r['step']):r for r in rows if r['environment']=='0' and r['status_code']=='0' and r['step_accepted']=='1'}
    accepted_tools = [r for r in tools if r['environment']=='0' and int(r['step']) in accepted and r['step_accepted']=='1']
    report['runs'][name] = {
       'status':result['status'], 'accepted_steps':result['accepted_steps'],
       'scale':manifest['tooling']['prescribed_tool_barrier_stiffness_scale'],
       'residual_tolerance':manifest['solver']['relative_residual_tolerance'],
       'inputs_unchanged':receipt['inputs_unchanged'], 'wall_seconds':receipt['wall_seconds'],
       'world_fingerprint':str(result['compiled_world_fingerprint']),
       'minimum_accepted_sampled_punch_gap_m': min((float(r['min_end_punch_node_gap_m']) for r in accepted_tools),default=None),
       'peak_accepted_punch_force_abs_N':max((abs(float(r['punch_force_z_N'])) for r in accepted_tools),default=None),
       'maximum_accepted_free_force_imbalance_l2_N':max((float(r['free_force_imbalance_l2_N']) for r in accepted.values()),default=None),
       'rejected_row_diagnostics':[{k:r[k] for k in ['step','status_code','certificate_residual','free_force_imbalance_l2_N']} for r in rows if r['environment']=='0' and r['step_accepted']!='1'],
       'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'manifest.json',run/'result.json',run/'observations.csv',run/'tool-observations.csv']}}
# Only complete equal-protocol trajectories can supply stiffness endpoint differences.
a,b = (report['runs'][n] for n in names[1:3])
if a['status']=='completed' and b['status']=='completed':
    for key in ['peak_accepted_punch_force_abs_N','minimum_accepted_sampled_punch_gap_m']:
        report.setdefault('factor2_vs_factor4',{})[key] = {'factor2':a[key], 'factor4':b[key], 'relative_difference':abs(a[key]-b[key])/max(abs(a[key]),abs(b[key]),1e-30)}
report['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out=root/'barrier-sensitivity-comparison.json'
if out.exists():raise SystemExit('refusing to overwrite retained report')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
