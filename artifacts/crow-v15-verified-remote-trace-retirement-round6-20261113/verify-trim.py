import hashlib,json,os,pathlib,subprocess,urllib.request,zlib
root=pathlib.Path(open('/tmp/crow-continuation-path').read().strip())/'repo';d=pathlib.Path('artifacts/crow-v15-mirrored-height-trim-20261113/signed-turn-20261126-rebound-trim'); manifest=json.loads((root/d/'stderr-parts.json').read_text());commit='6f8487d9beb56b68874385b259c3c195c2240eda';token=subprocess.check_output(['gh','auth','token'],text=True).strip();agg=hashlib.sha256();z=zlib.decompressobj(31);parts=[];total=0
for part in manifest['parts']:
 url='https://api.github.com/repos/Numi2/numi-lab/contents/'+str(d/part['file'])+'?ref='+commit
 req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'})
 h=hashlib.sha256();n=0
 with urllib.request.urlopen(req,timeout=90) as f:
  while True:
   b=f.read(1024*1024)
   if not b:break
   n+=len(b);h.update(b);agg.update(b);z.decompress(b)
 assert n==part['bytes'] and h.hexdigest()==part['sha256'],part['file'];total+=n;parts.append({'file':part['file'],'bytes':n,'sha256':h.hexdigest()})
assert total==manifest['bytes'] and agg.hexdigest()==manifest['sha256'] and z.eof and not z.unused_data
out={'commit':commit,'source':'independent authenticated GitHub raw content','manifest':str(d/'stderr-parts.json'),'bytes':total,'sha256':agg.hexdigest(),'gzip_eof_verified':True,'parts':parts};pathlib.Path('/tmp/crow-trim-github-verification.json').write_text(json.dumps(out,indent=2)+'\n');print('verified',total,'bytes; gzip EOF',z.eof)
