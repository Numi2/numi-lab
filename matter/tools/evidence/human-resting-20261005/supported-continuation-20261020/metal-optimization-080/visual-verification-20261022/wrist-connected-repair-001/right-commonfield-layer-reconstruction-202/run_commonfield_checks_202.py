from pathlib import Path
import json,subprocess,sys,hashlib,time
B=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/right-commonfield-layer-reconstruction-202');attempt=sys.argv[1];O=B/attempt;py='/Users/n/numi-human-prep-venv-20261005/bin/python';start=time.monotonic();commands=[]
for name in ['extract_right_interfaces_202.py','audit_right_interfaces_202.py','audit_right_bone_clearance_202.py']:
 command=[py,str(B/name),attempt];commands.append(command)
 with (O/(name+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise SystemExit(f'{name} failed: {r.returncode}')
 if name=='audit_right_interfaces_202.py':
  a=json.loads((O/'interface-exact-audit-202.json').read_text())
  if any(r['count'] for r in a['self'].values()) or any(r['intersection_pairs'] for r in a['cross'].values()):raise SystemExit('exact layer crossing gate failed')
prep=json.loads((O/'interface-preflight-report-202.json').read_text());bone=json.loads((O/'interface-bone-clearance-202.json').read_text());refine=json.loads((O/'field-refinement-report-202.json').read_text());passed=bool(prep['preparation_pass'] and bone['pass'] and refine['volume_calibration_pass'])
files=[p for p in O.iterdir() if p.is_file()];report={'source_geometry_pass':passed,'physical_admission':False,'accepted_native_steps':0,'host':'numi.local','device':'Apple M6 Mac18,5','frame':'actual007 registered Float32 world','commands':commands,'wall_seconds':time.monotonic()-start,'anatomy_assembly':'mixed-source reference adult','files':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files},'limitations':['single captured pose','construction volume targets, not measured anatomical thickness','skin/fat cell emission, quality and conservative mechanical ancestry required','native contact and accepted-state qualification pending']};(O/'source-geometry-receipt-202.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'source_geometry_pass':passed,'wall_seconds':report['wall_seconds']}),flush=True)
if not passed:raise SystemExit('source geometry gate failed')
