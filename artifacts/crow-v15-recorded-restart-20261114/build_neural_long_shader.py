from pathlib import Path
import hashlib,json,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
s=(a/'restart-round-neural-balance-masked.metal').read_text()
old='''        : avianCurriculumBand == 9u ? 1600u
        : program.schedule.x;
    const bool timeout'''
new='''        : avianCurriculumBand == 9u ? 1600u
        : max(program.schedule.x, 5000u);
    const bool timeout'''
assert s.count(old)==1
s=s.replace(old,new)
src=a/'restart-round-neural-balance-long.metal';src.write_text(s)
build=json.loads((a/'shader-build.json').read_text());cc=build['compile'];cc[cc.index('-c')+1]=str(src);air=b/'restart-round-neural-balance-long.air';lib=b/'restart-round-neural-balance-long.metallib';cc[cc.index('-o')+1]=str(air);subprocess.run(cc,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(air);link[link.index('-o')+1]=str(lib);subprocess.run(link,check=True)
(a/'neural-long-shader-build.json').write_text(json.dumps({'compile':cc,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,air,lib)}},indent=2)+'\n')
