from pathlib import Path
import gzip,hashlib,json,subprocess,urllib.request,zlib
A=Path('artifacts/crow-v15-corrected-reset-flight-20261114');p=Path('artifacts/crow-v15-reset-once-ablation-20261114/turn-20261126-rebound-resetonce/run.stderr.gz');v=json.loads(Path('artifacts/crow-v15-reset-once-github-verification-20261114/verification.json').read_text());tok=subprocess.check_output(['gh','auth','token'],text=True).strip();url='https://api.github.com/repos/Numi2/numi-lab/contents/'+p.as_posix()+'?ref='+v['commit'];req=urllib.request.Request(url,headers={'Authorization':'Bearer '+tok,'Accept':'application/vnd.github.raw+json','User-Agent':'crow-evidence-verifier'});h=hashlib.sha256();z=zlib.decompressobj(31);n=0;buf=b'';states=[];keys=set()
def parse(line):
 if line.startswith(b'CROW_SUPPORT_SAMPLE '):
  x=json.loads(line.split(b' ',1)[1]);k=x['lane'],x['step'];assert k not in keys;keys.add(k);states.append(x)
with urllib.request.urlopen(req,timeout=180) as f:
 while True:
  b=f.read(1<<20)
  if not b:break
  h.update(b);n+=len(b);buf+=z.decompress(b);lines=buf.split(b'\n');buf=lines.pop()
  for line in lines:parse(line)
assert n==v['bytes'] and h.hexdigest()==v['sha256'] and z.eof and not z.unused_data
if buf:parse(buf)
assert all((e,t) in keys for e in range(8) for t in range(1,501))
with (A/'reset-reference.json.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as out:out.write(json.dumps(states,separators=(',',':')).encode())
out={'source_commit':v['commit'],'source_sha256':h.hexdigest(),'state_count':len(states),'all_lanes_complete_500':True,'reference_sha256':hashlib.sha256((A/'reset-reference.json.gz').read_bytes()).hexdigest()};(A/'reset-reference-manifest.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
