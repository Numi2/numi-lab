from pathlib import Path
import json,time,subprocess
D=Path("/Users/numi/Work/numi-human-connected-surface-069");done=D/"D/outward-plane-fit-460/run-002-launch/execution.json";t=time.monotonic()
while not done.exists():
 if time.monotonic()-t>800:raise TimeoutError("fit460 incomplete")
 time.sleep(2)
assert json.loads(done.read_text())["returncode"]==0
args=["/Users/n/numi-human-prep-venv-20261005/bin/python",str(D/"D/checked-proposal-459/run_checked_proposal_459.py"),str(D/"D/constraint-project-backtrack-456/run-001/backtrack-373/selected-result.json"),str(D/"barycentric-connected-clearance-170/outward-plane-fit-460/run-002"),str(D/"D/connected-continuation-prep-457/run-001"),str(D/"D/fixed-bone-guard-refresh-457/run-001"),str(D/"D/outward-plane-backtrack-461/run-001")]
raise SystemExit(subprocess.run(args).returncode)
