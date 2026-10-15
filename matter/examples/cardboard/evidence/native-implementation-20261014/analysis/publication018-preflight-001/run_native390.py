from pathlib import Path
import json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006')
cmd=json.loads((B/'native-390-command.json').read_text())
assert json.loads((R/'build-cardboard-provenance-118/source-build.json').read_text())['success']
with (B/'native-390-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
print('native390',code,flush=True);raise SystemExit(code)
