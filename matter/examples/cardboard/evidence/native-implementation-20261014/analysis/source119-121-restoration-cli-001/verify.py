from pathlib import Path
import hashlib,json,subprocess,tempfile
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');A=B/'analysis/retained-source-snapshots119-121-001';O=B/'analysis/source119-121-restoration-cli-001';O.mkdir(exist_ok=False)
text=(B/'analysis/retained-source-snapshot118-001/restore.py').read_text().replace('choices=(118,)','choices=(119,120,121)');(A/'restore.py').write_text(text)
(A/'README.md').write_text('Exact source119, source120 and source121 restoration from fixed Git base7ebc64a666c4bd5e1ff61b3684655979ab80dc60 and hash-checked gzip blobs.\n\nRun `python3 restore.py --repository REPO --number 121 --output NEW_DIRECTORY`. The output must not exist. Each snapshot retains its actual build receipt hash; restoration does not reclassify any failed physical run.\n')
rows=[]
for number in (119,120,121):
 with tempfile.TemporaryDirectory(prefix=f'cardboard-restore{number}-',dir=B) as tmp:
  dest=Path(tmp)/'restored';cmd=['/Users/home/.pyenv/versions/3.10.12/bin/python3',str(A/'restore.py'),'--repository',str(R),'--number',str(number),'--output',str(dest)];q=subprocess.run(cmd,capture_output=True,text=True);(O/f'output{number}.log').write_text(q.stdout+q.stderr);assert q.returncode==0,q.stderr;rows.append(json.loads(q.stdout))
(O/'report.json').write_text(json.dumps(dict(status='PASS',restorer_sha256=hashlib.sha256((A/'restore.py').read_bytes()).hexdigest(),restored_snapshots=rows),indent=2)+'\n');(O/'verify.py').write_bytes(Path(__file__).read_bytes());(O/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()},indent=2)+'\n');print('PASS exact CLI restorations119/120/121')
