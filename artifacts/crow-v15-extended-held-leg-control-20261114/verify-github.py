from pathlib import Path
import hashlib,json,subprocess,sys,urllib.request,zlib
A=Path('artifacts/crow-v15-extended-held-leg-control-20261114');O=Path('artifacts/crow-v15-release-gated-wing-20261114');commit=sys.argv[1];token=subprocess.check_output(['gh','auth','token'],text=True).strip();out=[]
for p in (O/'push-20261126-rebound-gated/run.stderr.gz',O/'hold-20261126-rebound-gated/run.stderr.gz',A/'hold-20261126-rebound-gated/run.stderr.gz'):
 h=hashlib.sha256();n=0;z=zlib.decompressobj(31)
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b);n+=len(b);z.decompress(b)
 assert z.eof and not z.unused_data
 req=urllib.request.Request('https://api.github.com/repos/Numi2/numi-lab/contents/'+p.as_posix()+'?ref='+commit,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});rh=hashlib.sha256();rn=0;rz=zlib.decompressobj(31)
 with urllib.request.urlopen(req,timeout=180) as f:
  for b in iter(lambda:f.read(1<<20),b''):rh.update(b);rn+=len(b);rz.decompress(b)
 assert rn==n and rh.hexdigest()==h.hexdigest() and rz.eof and not rz.unused_data
 out.append({'path':str(p),'bytes':rn,'sha256':rh.hexdigest(),'gzip_eof':True});print(p,rn,rh.hexdigest(),flush=True)
(A/'github-verification.json').write_text(json.dumps({'source_commit':commit,'files':out},indent=2)+'\n')
