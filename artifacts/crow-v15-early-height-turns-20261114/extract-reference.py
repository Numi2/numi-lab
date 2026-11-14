from pathlib import Path
import gzip,hashlib,json,subprocess,urllib.request,zlib
p=Path('artifacts/crow-v15-early-height-turns-20261114');manifest=json.loads(subprocess.check_output(['git','show','HEAD:artifacts/crow-v15-early-height-bias-20261114/altitude-20261126-rebound-earlyheight/stderr-parts.json'],text=True));token=subprocess.check_output(['gh','auth','token'],text=True).strip();h=hashlib.sha256();z=zlib.decompressobj(31);buf=b'';count=0;seen=set();n=0
with (p/'reference-states.json.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:
  out.write(b'[')
  for part in manifest['parts']:
   path='artifacts/crow-v15-early-height-bias-20261114/altitude-20261126-rebound-earlyheight/'+part['file'];url='https://api.github.com/repos/Numi2/numi-lab/contents/'+path+'?ref=01a2cb8b';req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});ph=hashlib.sha256();pn=0
   with urllib.request.urlopen(req,timeout=180) as f:
    for b in iter(lambda:f.read(1048576),b''):
     n+=len(b);pn+=len(b);h.update(b);ph.update(b);buf+=z.decompress(b)
     lines=buf.split(b'\n');buf=lines.pop()
     for line in lines:
      if line.startswith(b'CROW_SUPPORT_SAMPLE '):
       payload=line.split(b' ',1)[1];x=json.loads(payload);key=(x['lane'],x['step']);assert key not in seen;seen.add(key)
       if count:out.write(b',')
       out.write(payload);count+=1
   assert pn==part['bytes'] and ph.hexdigest()==part['sha256']
  assert z.eof and not z.unused_data and n==manifest['bytes'] and h.hexdigest()==manifest['sha256'] and count==8*1801
  out.write(b']')
(p/'reference-verification.json').write_text(json.dumps({'source_commit':'01a2cb8b','source_sha256':h.hexdigest(),'source_bytes':n,'gzip_eof':True,'unique_state_records':count,'reference_sha256':hashlib.sha256((p/'reference-states.json.gz').read_bytes()).hexdigest()},indent=2)+'\n');print('verified and extracted',count,'states')
