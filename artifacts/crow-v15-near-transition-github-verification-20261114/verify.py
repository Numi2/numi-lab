from pathlib import Path
import gzip,hashlib,json,subprocess,urllib.request,zlib
A=Path('artifacts/crow-v15-near-wing-transition-20261113');commit='699d8643'
token=subprocess.check_output(['gh','auth','token'],text=True).strip();out=[]
for name in ['push-20261126-rebound-near','hold-20261126-rebound-near']:
 p=A/name/'run.stderr.gz';local=hashlib.sha256(p.read_bytes()).hexdigest();url='https://api.github.com/repos/Numi2/numi-lab/contents/'+str(p)+'?ref='+commit
 req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
 with urllib.request.urlopen(req,timeout=180) as f:
  while b:=f.read(1<<20):n+=len(b);h.update(b);z.decompress(b)
 assert n==p.stat().st_size and h.hexdigest()==local and z.eof and not z.unused_data
 out.append({'path':str(p),'bytes':n,'sha256':h.hexdigest(),'gzip_eof':True})
Path('artifacts/crow-v15-near-transition-github-verification-20261114/verification.json').write_text(json.dumps({'commit':commit,'files':out},indent=2)+'\n');print(json.dumps(out,indent=2))
