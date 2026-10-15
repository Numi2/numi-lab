from pathlib import Path
import json,hashlib,subprocess
B=Path('/Users/home/cardboard-implementation-20261014');O=B/'analysis/native401-manifest-alias-bindings-001';O.mkdir(exist_ok=False);(O/'aliases').mkdir()
cmd=json.loads((B/'native-397-command.json').read_text());rows=[]
for i in range(len(cmd)-1):
 if cmd[i]=='--input-file':
  p=Path(cmd[i+1]);alias=O/'aliases'/f'source392-input-{len(rows):02d}-{p.name}';q=subprocess.run(['/bin/cp','-c',str(p),str(alias)],capture_output=True,text=True);assert q.returncode==0
  h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(alias.read_bytes()).hexdigest()==h;rows.append(dict(original=str(p),alias=str(alias),sha256=h,bytes=p.stat().st_size));cmd[i+1]=str(alias)
cmd=[x.replace('source392-tet858-point-397','source392-tet858-point-401') for x in cmd]
(B/'native-401-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
(O/'bindings.json').write_text(json.dumps(dict(schema='numi.cardboard.external_input_aliases.v1',cause='Native397 input seal rejects duplicate manifest.json basenames before execution. Original397 failure preserved. Fresh401 uses uniquely named exact-byte aliases; native host still hashes every original input against its fixed bindings.',physical_execution397=False,files=rows),indent=2)+'\n')
(O/'prepare.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in O.rglob('*') if p.is_file()},indent=2)+'\n')
(B/'run_native401.py').write_text('from pathlib import Path\nimport json,subprocess,hashlib\nB=Path(\'/Users/home/cardboard-implementation-20261014\')\nrows=json.loads((B/\'analysis/native401-manifest-alias-bindings-001/bindings.json\').read_text())[\'files\']\nfor r in rows:assert hashlib.sha256(Path(r[\'original\']).read_bytes()).hexdigest()==r[\'sha256\']\ncmd=json.loads((B/\'native-401-command.json\').read_text())\nwith (B/\'native-401-runner.log\').open(\'x\') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)\nfor r in rows:assert hashlib.sha256(Path(r[\'original\']).read_bytes()).hexdigest()==r[\'sha256\']\nprint(\'native401\',code,flush=True)\n')
