from pathlib import Path
import json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
cmd=json.loads((B/'native-407-command.json').read_text())
with (B/'native-407-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
print('native407',code,flush=True)
