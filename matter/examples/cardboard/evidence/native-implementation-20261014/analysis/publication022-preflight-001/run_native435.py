from pathlib import Path
import json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
assert json.loads((B/'analysis/native434-disabled-history-parity-001/report.json').read_text())['all_numerical_trace_fields_exact']
assert json.loads((B/'analysis/native-base-cauchy-kernel-result-433/receipt.json').read_text())['exit_code']==0
assert json.loads((B/'analysis/native433-cauchy-kernel-adjudication-001/report.json').read_text())['checks']['all_passed']
cmd=json.loads((B/'native-435-command.json').read_text())
with (B/'native-435-runner.log').open('x') as out:q=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
print('native435',q.returncode,flush=True)
raise SystemExit(q.returncode)
