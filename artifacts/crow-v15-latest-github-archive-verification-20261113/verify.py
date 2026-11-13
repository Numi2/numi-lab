from pathlib import Path
import hashlib,json,os,subprocess,urllib.request,zlib
R=Path(__file__).resolve().parents[2];A=Path(__file__).resolve().parent
commit='c080c124b4fdcc218693c772567549c36d978cfd'
paths=[]
for stem in ('crow-v15-delayed-wing-leg-push-20261113','crow-v15-wing-transition-timing-20261113'):
 for arm in ('push','hold'):
  for name in ('run.stderr.gz','replay.json.gz'):
   paths.append(Path('artifacts')/stem/(arm+'-20261126-rebound-delay')/name)
token=subprocess.check_output(['gh','auth','token'],text=True).strip();rows=[]
for rel in paths:
 p=R/rel;assert p.is_file() and not p.is_symlink();st=p.stat();assert st.st_nlink==1
 expected=hashlib.sha256(p.read_bytes()).hexdigest();url='https://api.github.com/repos/Numi2/numi-lab/contents/'+str(rel)+'?ref='+commit
 req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'})
 h=hashlib.sha256();n=0;z=zlib.decompressobj(31)
 with urllib.request.urlopen(req,timeout=180) as f:
  while True:
   b=f.read(1<<20)
   if not b:break
   h.update(b);n+=len(b);z.decompress(b)
 assert (h.hexdigest(),n,z.eof,not z.unused_data)==(expected,st.st_size,True,True)
 rows.append({'path':str(rel),'bytes':n,'sha256':expected,'gzip_eof':True})
 print('verified',rel,n,flush=True)
(A/'verification.json').write_text(json.dumps({'source_commit':commit,'files':rows,'total_bytes':sum(r['bytes'] for r in rows)},indent=2)+'\n')
