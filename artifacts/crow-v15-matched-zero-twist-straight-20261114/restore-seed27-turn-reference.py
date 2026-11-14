from pathlib import Path
import gzip,hashlib,json,subprocess,urllib.request,zlib
A=Path('artifacts/crow-v15-matched-zero-twist-straight-20261114');D=Path('artifacts/crow-v15-turn-zero-twist-reduced-command-20261114/turn-20261127-rebound-smallerturn27');V=Path('artifacts/crow-v15-smaller-turn-seed27-verification-20261114/turn-verification.json');m=json.loads((D/'stderr-parts.json').read_text());v=json.loads(V.read_text());tok=subprocess.check_output(['gh','auth','token'],text=True).strip();z=zlib.decompressobj(31);h=hashlib.sha256();n=0;buf=b'';states=[];keys=set()
def parse(line):
 if line.startswith(b'CROW_SUPPORT_SAMPLE '):
  x=json.loads(line.split(b' ',1)[1]);k=(x['lane'],x['step']);assert k not in keys;keys.add(k);states.append(x)
for part in m['parts']:
 url='https://api.github.com/repos/Numi2/numi-lab/contents/'+(D/part['file']).as_posix()+'?ref='+v['source_commit'];req=urllib.request.Request(url,headers={'Authorization':'Bearer '+tok,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});ph=hashlib.sha256();size=0
 with urllib.request.urlopen(req,timeout=180) as f:
  while True:
   b=f.read(1<<20)
   if not b:break
   h.update(b);ph.update(b);n+=len(b);size+=len(b);buf+=z.decompress(b);lines=buf.split(b'\n');buf=lines.pop()
   for line in lines:parse(line)
 assert size==part['bytes'] and ph.hexdigest()==part['sha256']
assert n==m['bytes'] and h.hexdigest()==m['sha256'] and z.eof and not z.unused_data
if buf:parse(buf)
assert all((e,t) in keys for e in (6,7) for t in range(1,1801))
with (A/'seed27-turn-reference.json.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:out.write(json.dumps(states,separators=(',',':')).encode())
out={'source_commit':v['source_commit'],'source_sha256':h.hexdigest(),'state_count':len(states),'candidate_complete':True,'reference_sha256':hashlib.sha256((A/'seed27-turn-reference.json.gz').read_bytes()).hexdigest()};(A/'seed27-turn-reference-manifest.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
