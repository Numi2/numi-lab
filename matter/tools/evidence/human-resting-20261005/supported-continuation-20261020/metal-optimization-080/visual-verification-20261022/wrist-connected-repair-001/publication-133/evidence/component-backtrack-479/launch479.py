from pathlib import Path
import json,time,subprocess
D=Path("/Users/numi/Work/numi-human-connected-surface-069");done=D/"D/component-fit-477/run-001-launch/execution.json";t=time.monotonic()
while not done.exists():
 if time.monotonic()-t>800:raise TimeoutError("fit477 incomplete")
 time.sleep(2)
assert json.loads(done.read_text())["returncode"]==0
r=json.loads((D/"barycentric-connected-clearance-170/component-fit-477/run-001/report.json").read_text())
if r["status"]!="CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED":raise SystemExit("fit477 produced no construction proposal; exact audit not run")
args=["/Users/n/numi-human-prep-venv-20261005/bin/python",str(D/"D/checked-proposal-459/run_checked_proposal_459.py"),str(D/"D/outward-plane-backtrack-461/run-001/backtrack-373/selected-result.json"),str(D/"barycentric-connected-clearance-170/component-fit-477/run-001"),str(D/"D/connected-continuation-prep476/run-001"),str(D/"D/fixed-bone-guard-refresh476/run-001"),str(D/"D/component-backtrack-479/run-001")]
raise SystemExit(subprocess.run(args).returncode)
