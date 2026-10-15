from pathlib import Path
import json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
cmd=json.loads((B/'native-434-command.json').read_text())
with (B/'native-434-runner.log').open('x') as out:q=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
print('native434',q.returncode,flush=True)
raise SystemExit(q.returncode)
