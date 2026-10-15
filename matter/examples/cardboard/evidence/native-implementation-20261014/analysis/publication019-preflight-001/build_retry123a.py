from pathlib import Path
import json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
cmd=json.loads((B/'build-command-123a.json').read_text())
with (B/'build123a-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
print('build123a',code,flush=True)
raise SystemExit(code)
