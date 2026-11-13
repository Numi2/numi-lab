import hashlib,json,pathlib,subprocess,urllib.request,zlib
root=pathlib.Path(open('/tmp/crow-continuation-path').read().strip())/'repo';base=pathlib.Path('artifacts/crow-v15-mirrored-leg-wing-transition-20261113');rel='signed-turn-20261126-rebound-transition/run.stderr.gz';commit='77e428b73ec2ef2818f3de7761e7575ffaead187';token=subprocess.check_output(['gh','auth','token'],text=True).strip();sums={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/base/'SHA256SUMS').read_text().splitlines()};req=urllib.request.Request('https://api.github.com/repos/Numi2/numi-lab/contents/'+str(base/rel)+'?ref='+commit,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
with urllib.request.urlopen(req,timeout=90) as f:
 while True:
  b=f.read(1024*1024)
  if not b:break
  n+=len(b);h.update(b);z.decompress(b)
assert h.hexdigest()==sums[rel] and z.eof and not z.unused_data;out={'commit':commit,'source':'independent authenticated GitHub raw content','files':[{'path':str(base/rel),'bytes':n,'sha256':h.hexdigest(),'gzip_eof_verified':True}]};pathlib.Path('/tmp/crow-transition-github-verification.json').write_text(json.dumps(out,indent=2)+'\n');print('verified transition',n,'bytes')
