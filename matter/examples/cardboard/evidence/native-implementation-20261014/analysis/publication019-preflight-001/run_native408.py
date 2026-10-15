from pathlib import Path
import json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
cmd=json.loads((B/'native-408-command.json').read_text())
with (B/'native-408-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
print('native408',code,flush=True)
