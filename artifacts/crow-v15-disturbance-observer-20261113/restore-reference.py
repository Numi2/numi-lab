from pathlib import Path
import gzip,hashlib,json,subprocess,urllib.request,zlib
A=Path('artifacts/crow-v15-disturbance-observer-20261113');m=json.loads((A/'straight-reference-manifest.json').read_text());commit='b59a4408c56509059c876154ba86eabf5a044a64';root='artifacts/crow-v15-equal-launch-onset-20261113/straight-20261127-rebound-equallong/';token=subprocess.check_output(['gh','auth','token'],text=True).strip();stream=zlib.decompressobj(31);total=hashlib.sha256();n=0;carry=b'';states=[]
for part in m['parts']:
 h=hashlib.sha256();size=0;url='https://api.github.com/repos/Numi2/numi-lab/contents/'+root+part['file']+'?ref='+commit;req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-state-restore'})
 with urllib.request.urlopen(req,timeout=180) as f:
  while True:
   b=f.read(1<<20)
   if not b:break
   n+=len(b);size+=len(b);h.update(b);total.update(b);pieces=(carry+stream.decompress(b)).split(b'\n');carry=pieces.pop()
   for line in pieces:
    if line.startswith(b'CROW_SUPPORT_SAMPLE '):states.append(json.loads(line.split(b' ',1)[1]))
 assert size==part['bytes'] and h.hexdigest()==part['sha256']
assert n==m['bytes'] and total.hexdigest()==m['sha256'] and stream.eof and not stream.unused_data and not carry
with (A/'straight-reference-states.json.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(json.dumps(states,separators=(',',':')).encode())
print(json.dumps({'commit':commit,'source_sha256':total.hexdigest(),'bytes':n,'states':len(states),'gzip_eof':stream.eof}))
