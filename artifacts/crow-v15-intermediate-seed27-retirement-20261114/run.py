import json,shlex,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent;v=json.loads((R/'crow-v15-intermediate-seed27-github-verification-20261114/verification.json').read_text());where=sys.argv[1];retire=len(sys.argv)>2;root=R if where=='local' else Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');spec=[{'path':str(root/x['path']),'bytes':x['bytes'],'sha256':x['sha256']} for x in v['files']];args=[json.dumps(spec,separators=(',',':'))]
if retire:args.append((A/(where+'-preflight.json')).read_text())
if where=='local':cmd=['python3',str(A/'check.py'),*args];inp=None
else:cmd=['ssh','-4','-o','HostName=192.168.1.141','macmini','python3 - '+ ' '.join(shlex.quote(x) for x in args)];inp=(A/'check.py').read_bytes()
r=subprocess.run(cmd,input=inp,stdout=subprocess.PIPE,check=True);(A/(where+('-receipt.json' if retire else '-preflight.json'))).write_bytes(r.stdout)
