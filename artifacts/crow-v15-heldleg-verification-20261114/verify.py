from pathlib import Path
import hashlib,json,subprocess,urllib.request,zlib
p=Path("artifacts/crow-v15-reset-corrected-leg-ablation-20261114/hold-20261126-rebound-fixed500/run.stderr.gz");m=json.loads((p.parent/"raw-manifest.json").read_text());commit="7ff8af38";token=subprocess.check_output(["gh","auth","token"],text=True).strip();url="https://api.github.com/repos/Numi2/numi-lab/contents/"+p.as_posix()+"?ref="+commit;req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token,"Accept":"application/vnd.github.raw+json","User-Agent":"crow-evidence-verifier"});h=hashlib.sha256();z=zlib.decompressobj(31);n=0
with urllib.request.urlopen(req,timeout=180) as f:
 while True:
  b=f.read(1<<20)
  if not b:break
  n+=len(b);h.update(b);z.decompress(b)
assert n==m["bytes"] and h.hexdigest()==m["sha256"] and z.eof and not z.unused_data
out={"commit":commit,"sha256":h.hexdigest(),"bytes":n,"gzip_eof":True};Path("artifacts/crow-v15-heldleg-verification-20261114/verification.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
