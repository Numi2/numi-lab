import hashlib,json,pathlib,subprocess,urllib.request,zlib
p=pathlib.Path(open('/tmp/crow-continuation-path').read().strip())/'repo';d=pathlib.Path('artifacts/crow-v15-toe-regrip-fk-balance-20261113');commit='9fdd47455745792e7563902eedae14cf0f265efc';token=subprocess.check_output(['gh','auth','token'],text=True).strip();out=[]
for name in ('trace.log.gz','substeps.log.gz','probe.gz'):
 expected=hashlib.sha256((p/d/name).read_bytes()).hexdigest();u='https://api.github.com/repos/Numi2/numi-lab/contents/'+str(d/name)+'?ref='+commit;r=urllib.request.Request(u,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
 with urllib.request.urlopen(r,timeout=90) as f:
  while True:
   b=f.read(1048576)
   if not b:break
   n+=len(b);h.update(b);z.decompress(b)
 assert h.hexdigest()==expected and z.eof and not z.unused_data;out.append({'name':name,'sha256':expected,'bytes':n,'gzip_eof_verified':True})
pathlib.Path('/tmp/crow-toe-fk-github-verification.json').write_text(json.dumps({'commit':commit,'source':'independent GitHub raw','files':out},indent=2)+'\n');print('verified',len(out),'files')
