from pathlib import Path
import hashlib,json,subprocess,sys,urllib.request,zlib
A=Path('artifacts/crow-v15-tucked-wing-phase-20261114');commit=sys.argv[1];token=subprocess.check_output(['gh','auth','token'],text=True).strip();out={}
for arm in ('push','hold'):
 p=A/(arm+'-20261126-rebound-nonleg')/'run.stderr.gz';local=hashlib.sha256();size=0;zl=zlib.decompressobj(31)
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):local.update(b);size+=len(b);zl.decompress(b)
 assert zl.eof and not zl.unused_data
 req=urllib.request.Request('https://api.github.com/repos/Numi2/numi-lab/contents/'+p.as_posix()+'?ref='+commit,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});remote=hashlib.sha256();n=0;zr=zlib.decompressobj(31)
 with urllib.request.urlopen(req,timeout=180) as f:
  for b in iter(lambda:f.read(1<<20),b''):remote.update(b);n+=len(b);zr.decompress(b)
 assert n==size and remote.hexdigest()==local.hexdigest() and zr.eof and not zr.unused_data
 out[arm]={'path':str(p),'bytes':n,'sha256':remote.hexdigest(),'gzip_eof':True}
 print(arm,n,remote.hexdigest(),flush=True)
(A/'github-verification.json').write_text(json.dumps({'source_commit':commit,'arms':out},indent=2)+'\n')
