from pathlib import Path
import json,subprocess,time,os,hashlib
H=Path('/Users/numi/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170');D=H.parent
PY='/Users/n/numi-human-prep-venv-20261005/bin/python';P=H/'connected-clearance-239-run001';P.mkdir(exist_ok=False)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def run(cmd,log):
 st=time.monotonic()
 with Path(log).open('x')as f:r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,timeout=3600)
 write(str(log)+'.execution.json',dict(command=list(map(str,cmd)),returncode=r.returncode,wall_seconds=time.monotonic()-st))
 assert r.returncode==0,(log,r.returncode)
# Single heavy-work owner; wait for the prior bounded continuation to finish.
previous=H/'joint-clearance-continuation-234-run001'
while True:
 try:os.kill(85495,0);time.sleep(3)
 except ProcessLookupError:break
assert (previous/'completed.json').exists(),'Prior continuation failed; do not run from ambiguous state'
state=previous/'selected-result.json'
runner=H/'run_connected_clearance_238.py'
solver=D/'D/connected-local-witness-proposal-230/propose_joint_clearance_witnesses_230.py'
sh=hashlib.sha256(solver.read_bytes()).hexdigest();assert sh=='b942a9dff53f865a592db50c36369fb5014f0df990203c88dc945a7aeb9c3731'
full=P/'full-gradient'
run([PY,runner,'--state',state,'--out',full,'--solver',solver,'--solver-sha256',sh,'--rounds',1,'--alpha',1.0,'--max-edge-gradient',.2,'--learned-root',previous],P/'full.log')
if (full/'selected-result.json').exists():state=full/'selected-result.json'
# Preserve any failed full-step witnesses; the established path keeps the same gates.
solver=D/'D/connected-local-witness-proposal-222/propose_joint_clearance_witnesses_222.py'
half=P/'continuation'
run([PY,runner,'--state',state,'--out',half,'--solver',solver,'--solver-sha256','13171cd5559ea3644aaea7f98bf37bbbd4298168434cdffdc98aa729dde08564','--rounds',12,'--alpha',.5,'--learned-root',previous,'--learned-root',full],P/'continuation.log')
if (half/'selected-result.json').exists():state=half/'selected-result.json'
write(P/'completed.json',dict(state=str(state),native_admission=False,accepted_physical_steps=0))
print(str(state),flush=True)
