from pathlib import Path
import hashlib,json,subprocess,urllib.request,zlib
A=Path(__file__).resolve().parent;prefix='artifacts/crow-v15-stance-pitch-authority-20261114/';names=('prebuild-probe.gz','prebuild-probe.sha256');token=subprocess.check_output(['gh','auth','token'],text=True).strip();rows=[]
for name in names:
 u='https://api.github.com/repos/Numi2/numi-lab/contents/'+prefix+name+'?ref=4af0f4ed';r=urllib.request.Request(u,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'})
 with urllib.request.urlopen(r,timeout=120) as f:b=f.read()
 if name.endswith('.gz'):
  z=zlib.decompressobj(31);z.decompress(b);assert z.eof and not z.unused_data
 rows.append({'path':'/Users/n/Developer/numi-crow-rl-20261107/'+prefix+name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(A/'misplaced-verification.json').write_text(json.dumps({'commit':'4af0f4ed','files':rows},indent=2)+'\n');print('independently retrieved',len(rows),'misplaced files')
