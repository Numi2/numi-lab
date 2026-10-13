from pathlib import Path
import json,subprocess
r=Path('/Users/home/cardboard-followup-20261013')
m=json.loads((r/'operator-fd/local-accuracy-build035.json').read_text())
with (r/'build-local-projection-035.log').open('w') as log:
 for cmd in m['commands']:subprocess.run(cmd,stdout=log,stderr=log,check=True)
print('alternate tighter-local-projection035 Metal image built')
