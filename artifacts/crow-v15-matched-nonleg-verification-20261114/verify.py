from pathlib import Path
import hashlib,json,subprocess,urllib.request,zlib
A=Path('artifacts/crow-v15-matched-nonleg-ablation-20261114');V=Path('artifacts/crow-v15-matched-nonleg-verification-20261114');commit='ad43c4ab';m=json.loads((A/'trace-manifest.json').read_text());token=subprocess.check_output(['gh','auth','token'],text=True).strip();out={}
for arm in ('push','hold'):
 path=A/(arm+'-20261126-rebound-nonleg')/'run.stderr.gz';url='https://api.github.com/repos/Numi2/numi-lab/contents/'+path.as_posix()+'?ref='+commit;req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
 with urllib.request.urlopen(req,timeout=180) as f:
  for b in iter(lambda:f.read(1048576),b''):n+=len(b);h.update(b);z.decompress(b)
 assert n==m[arm]['bytes'] and h.hexdigest()==m[arm]['sha256'] and z.eof and not z.unused_data
 out[arm]={'bytes':n,'sha256':h.hexdigest(),'gzip_eof':True,'source_commit':commit}
(V/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
