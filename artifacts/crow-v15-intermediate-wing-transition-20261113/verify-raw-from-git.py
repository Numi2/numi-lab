from pathlib import Path
import hashlib,json,subprocess,urllib.request,zlib
A=Path('artifacts/crow-v15-intermediate-wing-transition-20261113');commit='c1019e75f8020455c77d14dbf1aaeaa1d19a2b7d';token=subprocess.check_output(['gh','auth','token'],text=True).strip();out=[]
expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (A/'SHA256SUMS-results').read_text().splitlines()}
for arm in ('push','hold'):
 p=A/(arm+'-20261126-rebound-intermediate/run.stderr.gz');url='https://api.github.com/repos/Numi2/numi-lab/contents/'+str(p)+'?ref='+commit;req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
 with urllib.request.urlopen(req,timeout=180) as f:
  while True:
   b=f.read(1<<20)
   if not b:break
   n+=len(b);h.update(b);z.decompress(b)
 assert h.hexdigest()==expected[str(p)] and z.eof and not z.unused_data
 out.append({'path':str(p),'bytes':n,'sha256':h.hexdigest(),'gzip_eof':True,'source_commit':commit})
(A/'raw-git-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
