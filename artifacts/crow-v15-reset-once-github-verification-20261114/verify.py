from pathlib import Path
import hashlib,json,subprocess,urllib.request,zlib
p=Path('artifacts/crow-v15-reset-once-ablation-20261114/turn-20261126-rebound-resetonce/run.stderr.gz'); expected=hashlib.sha256(p.read_bytes()).hexdigest(); n_expected=p.stat().st_size; token=subprocess.check_output(['gh','auth','token'],text=True).strip(); commit='2b347715'; url='https://api.github.com/repos/Numi2/numi-lab/contents/'+p.as_posix()+'?ref='+commit;req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
with urllib.request.urlopen(req,timeout=180) as f:
 while True:
  b=f.read(1<<20)
  if not b:break
  n+=len(b);h.update(b);z.decompress(b)
assert n==n_expected and h.hexdigest()==expected and z.eof and not z.unused_data
Path('artifacts/crow-v15-reset-once-github-verification-20261114/verification.json').write_text(json.dumps({'commit':commit,'bytes':n,'sha256':expected,'gzip_eof':True},indent=2)+'\n');print('independently verified',n,expected)
