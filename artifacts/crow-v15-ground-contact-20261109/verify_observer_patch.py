from pathlib import Path
import hashlib,json,subprocess,tempfile
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for prefix,meta in [('observer','build.json'),('late-observer','late-observer-build.json')]:
 with tempfile.TemporaryDirectory(prefix='crow-observer-verify-',dir=B) as t:
  out=Path(t)/'c_api.cpp';patch=A/(prefix+'.patch');subprocess.run(['patch','--batch','-o',str(out),str(R/'src/c_api.cpp'),str(patch)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);expected=json.loads((A/meta).read_text());key='observer_source_sha256' if prefix=='observer' else 'source_sha256';assert h(out)==expected[key],(prefix,h(out),expected[key]);print(prefix,'source restored and matches',h(out))
