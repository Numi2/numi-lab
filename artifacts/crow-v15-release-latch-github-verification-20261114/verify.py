from pathlib import Path
import hashlib,json,subprocess,urllib.request,zlib
R=Path(__file__).resolve().parent.parent;d=Path('crow-v15-flight-release-latch-20261114/turn-20261126-rebound-latch');m=json.loads((R/d/'stderr-parts.json').read_text());commit=__import__('sys').argv[1];token=subprocess.check_output(['gh','auth','token'],text=True).strip();h=hashlib.sha256();z=zlib.decompressobj(31);n=0;parts=[]
for x in m['parts']:
 url='https://api.github.com/repos/Numi2/numi-lab/contents/artifacts/'+str(d/x['file'])+'?ref='+commit;req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});ph=hashlib.sha256();pn=0
 with urllib.request.urlopen(req,timeout=180) as f:
  while True:
   b=f.read(1<<20)
   if not b:break
   n+=len(b);pn+=len(b);h.update(b);ph.update(b);z.decompress(b)
 assert pn==x['bytes'] and ph.hexdigest()==x['sha256'];parts.append({'file':x['file'],'bytes':pn,'sha256':ph.hexdigest()})
assert n==m['bytes'] and h.hexdigest()==m['sha256'] and z.eof and not z.unused_data
(Path(__file__).resolve().parent/'verification.json').write_text(json.dumps({'source_commit':commit,'bytes':n,'sha256':h.hexdigest(),'gzip_eof':True,'parts':parts},indent=2)+'\n');print('independently verified',n,h.hexdigest())
