import json,subprocess,urllib.request,gzip,hashlib,io,concurrent.futures
p='/tmp/crow-retirement-new-manifest.json';entries=json.load(open(p));commit=open('/tmp/crow-continuation-commit').read().strip();token=subprocess.check_output(['gh','auth','token'],text=True).strip()
def verify(item):
 url='https://api.github.com/repos/Numi2/numi-lab/contents/'+item['archive']+'?ref='+commit
 req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-recovery-verification'})
 with urllib.request.urlopen(req,timeout=60) as response:
  h=hashlib.sha256();n=0
  with gzip.GzipFile(fileobj=response) as z:
   while True:
    b=z.read(1<<20)
    if not b:break
    h.update(b);n+=len(b)
 item['restored_sha256']=h.hexdigest();item['restored_bytes']=n;item['verified_commit']=commit
 if (n,h.hexdigest())!=(item['bytes'],item['sha256']):raise ValueError(item['archive'])
 return item
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:out=list(pool.map(verify,entries))
json.dump(out,open('/tmp/crow-retirement-new-verified.json','w'),indent=2);print('Verified independent GitHub gzip restoration for',len(out),'files, bytes',sum(x['bytes'] for x in out),'commit',commit)
