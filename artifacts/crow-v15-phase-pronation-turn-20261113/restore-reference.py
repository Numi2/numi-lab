from pathlib import Path
import hashlib,json,subprocess,urllib.request,zlib
A=Path(__file__).resolve().parent;P=A.parent/'crow-v15-mirrored-leg-wing-longflight-20261113'/'signed-turn-20261126-rebound-longflight';m=json.loads((P/'stderr-parts.json').read_text());commit='6bc6ead4d24ea0ba84f0b502c09e83206f43078f';token=subprocess.check_output(['gh','auth','token'],text=True).strip();dest=Path('/tmp/crow-leg-longflight-reference-20261126.gz');h=hashlib.sha256();n=0;z=zlib.decompressobj(31);receipt=[]
with dest.open('xb') as o:
 for part in m['parts']:
  path='artifacts/crow-v15-mirrored-leg-wing-longflight-20261113/signed-turn-20261126-rebound-longflight/'+part['file'];url='https://api.github.com/repos/Numi2/numi-lab/contents/'+path+'?ref='+commit;req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});ph=hashlib.sha256();pn=0
  with urllib.request.urlopen(req,timeout=180) as f:
   while True:
    b=f.read(1048576)
    if not b:break
    o.write(b);h.update(b);ph.update(b);n+=len(b);pn+=len(b);z.decompress(b)
  assert pn==part['bytes'] and ph.hexdigest()==part['sha256'];receipt.append({'file':part['file'],'bytes':pn,'sha256':ph.hexdigest()})
assert n==m['bytes'] and h.hexdigest()==m['sha256'] and z.eof and not z.unused_data
(A/'reference-verification.json').write_text(json.dumps({'commit':commit,'sha256':h.hexdigest(),'bytes':n,'gzip_eof':z.eof,'parts':receipt},indent=2)+'\n');print('verified baseline',n,h.hexdigest())
