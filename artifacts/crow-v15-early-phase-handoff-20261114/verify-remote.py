from pathlib import Path
import hashlib,json,subprocess,zlib,urllib.request
root=Path('artifacts/crow-v15-early-phase-handoff-20261114/push-20261126-rebound-early')
token=subprocess.check_output(['gh','auth','token'],text=True).strip(); m=json.loads((root/'split-manifest.json').read_text()); whole=hashlib.sha256(); gz=zlib.decompressobj(16+zlib.MAX_WBITS); total=0; reports=[]
for part in m['parts']:
 endpoint='repos/Numi2/numi-lab/contents/'+str(root/part['name'])+'?ref=9c17b4e5'
 req=urllib.request.Request('https://api.github.com/'+endpoint,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'})
 d=hashlib.sha256(); n=0
 with urllib.request.urlopen(req,timeout=180) as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):
   d.update(chunk);whole.update(chunk);n+=len(chunk);total+=len(chunk);gz.decompress(chunk)
 assert n==part['bytes'] and d.hexdigest()==part['sha256'], (part['name'],n,d.hexdigest())
 reports.append({'name':part['name'],'bytes':n,'sha256':d.hexdigest()})
assert total==m['bytes'] and whole.hexdigest()==m['sha256'] and gz.eof and not gz.unused_data
out={'commit':'9c17b4e5','total_bytes':total,'aggregate_sha256':whole.hexdigest(),'gzip_eof':gz.eof,'parts':reports,'method':'streamed GitHub contents API, independently of local git objects'}
Path('artifacts/crow-v15-early-phase-handoff-20261114/remote-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
