from pathlib import Path
import json,subprocess,hashlib
B=Path('/Users/home/cardboard-implementation-20261014')
rows=json.loads((B/'analysis/native401-manifest-alias-bindings-001/bindings.json').read_text())['files']
for r in rows:assert hashlib.sha256(Path(r['original']).read_bytes()).hexdigest()==r['sha256']
cmd=json.loads((B/'native-401-command.json').read_text())
with (B/'native-401-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for r in rows:assert hashlib.sha256(Path(r['original']).read_bytes()).hexdigest()==r['sha256']
print('native401',code,flush=True)
