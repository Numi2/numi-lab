from pathlib import Path
import hashlib,json,subprocess,urllib.request,zlib
root=Path(__file__).resolve().parent.parent;commit='f379b617';token=subprocess.check_output(['gh','auth','token'],text=True).strip();results=[]
for arm in ('push','hold'):
 p=Path('crow-v15-ramped-wing-transition-20261114')/(arm+'-20261126-rebound-phase')/'run.stderr.gz';local=root/p;lh=hashlib.sha256();
 with local.open('rb') as lf:
  for b in iter(lambda:lf.read(1048576),b''):lh.update(b)
 expected=lh.hexdigest();size=local.stat().st_size;url='https://api.github.com/repos/Numi2/numi-lab/contents/artifacts/'+str(p)+'?ref='+commit;req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
 with urllib.request.urlopen(req,timeout=180) as f:
  while True:
   b=f.read(1048576)
   if not b:break
   n+=len(b);h.update(b);z.decompress(b)
 assert n==size and h.hexdigest()==expected and z.eof and not z.unused_data
 results.append({'arm':arm,'path':str(p),'bytes':n,'sha256':expected,'gzip_eof':True})
(Path(__file__).resolve().parent/'verification.json').write_text(json.dumps({'source_commit':commit,'files':results},indent=2)+'\n');print('verified',[(v['arm'],v['bytes']) for v in results])
