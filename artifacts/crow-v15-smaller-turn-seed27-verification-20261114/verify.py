from pathlib import Path
import hashlib,json,subprocess,sys,urllib.request,zlib
A=Path('artifacts/crow-v15-turn-zero-twist-reduced-command-20261114');V=Path('artifacts/crow-v15-smaller-turn-seed27-verification-20261114');commit=sys.argv[1];arm='turn';d=Path('turn-20261127-rebound-smallerturn27');m=json.loads((A/d/'stderr-parts.json').read_text());token=subprocess.check_output(['gh','auth','token'],text=True).strip();h=hashlib.sha256();z=zlib.decompressobj(31);n=0;parts=[]
for x in m['parts']:
 url='https://api.github.com/repos/Numi2/numi-lab/contents/'+(A/d/x['file']).as_posix()+'?ref='+commit;req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});ph=hashlib.sha256();pn=0
 with urllib.request.urlopen(req,timeout=180) as f:
  while True:
   b=f.read(1<<20)
   if not b:break
   n+=len(b);pn+=len(b);h.update(b);ph.update(b);z.decompress(b)
 assert pn==x['bytes'] and ph.hexdigest()==x['sha256'];parts.append({'file':x['file'],'bytes':pn,'sha256':ph.hexdigest()})
assert n==m['bytes'] and h.hexdigest()==m['sha256'] and z.eof and not z.unused_data
(V/(arm+'-verification.json')).write_text(json.dumps({'source_commit':commit,'bytes':n,'sha256':h.hexdigest(),'gzip_eof':True,'parts':parts},indent=2)+'\n');print('independently verified',arm,n,h.hexdigest())
