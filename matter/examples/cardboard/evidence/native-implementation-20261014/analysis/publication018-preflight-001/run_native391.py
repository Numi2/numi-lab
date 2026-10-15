from pathlib import Path
import json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
cmd=json.loads((B/'native-391-command.json').read_text())
with (B/'native-391-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
r=json.loads((Path(cmd[cmd.index('--case')+1])/'receipt.json').read_text())
assert r['execution_started'] and not r['timed_out'] and r['source_inputs_unchanged'] and not r['termination_errors']
print('native391',code,flush=True)
