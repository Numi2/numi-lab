from pathlib import Path
import hashlib,json,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107')
a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
old=(a/'MetalWorld-reset-observer.mm').read_text()
n1='crowTraceEnabled && crowTraceStep >= 335 && crowTraceStep <= 2400'
n2='crowTraceStep>=348 && crowTraceStep<=2400 && ((c.bodyA<35u && c.bodyB==35u)'
assert old.count(n1)==1 and old.count(n2)==1
new=old.replace(n1,'crowTraceEnabled && crowTraceStep >= 1 && crowTraceStep <= 2400').replace(n2,'crowTraceStep>=1 && crowTraceStep<=2400 && ((c.bodyA<35u && c.bodyB==35u)')
src=a/'MetalWorld-full-contact-observer.mm';src.write_text(new)
previous=json.loads((a/'reset-observer-build.json').read_text());cmd=previous['compile']
obj=b/'MetalWorld-full-contact-observer.mm.o'
cmd[cmd.index('-c')+1]=str(src);cmd[cmd.index('-o')+1]=str(obj)
subprocess.run(cmd,cwd=r/'build',check=True)
out=b/'full-contact-observer';out.mkdir(exist_ok=False);lib=out/'libmetalrobo.dylib'
link=previous['link'];link[link.index(str(b/'MetalWorld-reset-observer.mm.o'))]=str(obj);link[link.index('-o')+1]=str(lib)
subprocess.run(link,cwd=r/'build',check=True)
(a/'full-contact-observer-build.json').write_text(json.dumps({'compile':cmd,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,obj,lib)}},indent=2)+'\n')
print('built full-window diagnostic observer')
