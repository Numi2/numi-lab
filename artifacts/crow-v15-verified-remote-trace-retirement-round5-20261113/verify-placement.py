import hashlib,json,pathlib,subprocess,urllib.request,zlib
root=pathlib.Path(open('/tmp/crow-continuation-path').read().strip())/'repo';base=pathlib.Path('artifacts/crow-v15-mirrored-crouch-placement-20261113'); commit='f4834e6c68cd9fd4374089158927d35f4e60188d';token=subprocess.check_output(['gh','auth','token'],text=True).strip();sums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/base/'SHA256SUMS').read_text().splitlines()};rows=[]
for arm in ('push','hold'):
 rel=arm+'-wing-20261126-rebound-placement/run.stderr.gz';req=urllib.request.Request('https://api.github.com/repos/Numi2/numi-lab/contents/'+str(base/rel)+'?ref='+commit,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
 with urllib.request.urlopen(req,timeout=90) as f:
  while True:
   b=f.read(1024*1024)
   if not b:break
   n+=len(b);h.update(b);z.decompress(b)
 assert h.hexdigest()==sums[rel] and z.eof and not z.unused_data;rows.append({'path':str(base/rel),'bytes':n,'sha256':h.hexdigest(),'gzip_eof_verified':True})
pathlib.Path('/tmp/crow-placement-github-verification.json').write_text(json.dumps({'commit':commit,'source':'independent authenticated GitHub raw content','files':rows},indent=2)+'\n');print('verified',len(rows),'ankle traces,',sum(x['bytes'] for x in rows),'bytes')
