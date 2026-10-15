from pathlib import Path
import hashlib,json,subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
j=json.loads((B/'analysis/cache-reuse-native423-inputs-001/fixture-bindings.json').read_text())
for p,h in j['files'].items():assert sha(p)==h,p
cmd=json.loads((B/'native-423-command.json').read_text())
with (B/'native-423-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in j['files'].items():assert sha(p)==h,p
print('native423',code,flush=True)
