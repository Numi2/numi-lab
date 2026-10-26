from pathlib import Path
import time,json,subprocess
D=Path("/Users/numi/Work/numi-human-connected-surface-069");done=D/"D/reseeded-fit-434/run-001-launch/execution.json"
t=time.monotonic()
while not done.exists():
 if time.monotonic()-t>700:raise TimeoutError("fit434 not completed")
 time.sleep(2)
assert json.loads(done.read_text())["returncode"]==0
args=["/Users/n/numi-human-prep-venv-20261005/bin/python",str(D/"D/checked-proposal-436/run_checked_proposal_436.py"),str(D/"D/postfit-backtrack-427/run-001/backtrack-373/selected-result.json"),str(D/"barycentric-connected-clearance-170/reseeded-fit-434/run-001"),str(D/"D/connected-continuation-prep-433/run-001"),str(D/"D/seed-preserving-backtrack-436/run-001")]
r=subprocess.run(args);raise SystemExit(r.returncode)
