"""Create an isolated diagnostic host that can initialize one recorded lane.

This reconstructs q/v and static scene poses, not missing task/resident state.
"""
from pathlib import Path
import hashlib,json,shlex,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107')
a=r/'artifacts/crow-v15-recorded-restart-20261114'; b=r/'build/crow-v15-recorded-restart-20261114'
a.mkdir(exist_ok=False);b.mkdir(exist_ok=False)
old=(r/'build/crow-v15-course-collision-restored-20261114/c_api.cpp').read_text()
needle='    handle.result = {};\n    handle.statusCodes.clear();'
assert old.count(needle)==1
block='''    if (const char* restart=std::getenv("CROW_RECORDED_STATE_RESTART")) {
        if (environmentCount!=1 || nq!=41 || nv!=40 || sceneBodyCount!=6)
            throw std::logic_error("recorded restart requires one v15 lane and six scene bodies");
        std::ifstream input(restart);
        if (!input) throw std::runtime_error("cannot open recorded restart");
        for (auto& x:handle.resetQ) if (!(input>>x) || !std::isfinite(x)) throw std::runtime_error("invalid recorded q");
        for (auto& x:handle.resetV) if (!(input>>x) || !std::isfinite(x)) throw std::runtime_error("invalid recorded v");
        for (auto& s:handle.resetSceneBodies) {
            float px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz;
            if (!(input>>px>>py>>pz>>qx>>qy>>qz>>qw>>vx>>vy>>vz>>wx>>wy>>wz)) throw std::runtime_error("invalid recorded scene");
            for(float x:{px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz}) if (!std::isfinite(x)) throw std::runtime_error("nonfinite recorded scene");
            s.position.x=px; s.position.y=py; s.position.z=pz;
            s.orientation.x=qx; s.orientation.y=qy; s.orientation.z=qz; s.orientation.w=qw;
            s.linearVelocityAndInverseMass.x=vx; s.linearVelocityAndInverseMass.y=vy; s.linearVelocityAndInverseMass.z=vz;
            s.angularVelocity.x=wx; s.angularVelocity.y=wy; s.angularVelocity.z=wz;
        }
        std::fprintf(stderr,"CROW_RECORDED_STATE_RESTART loaded 1 lane, q=41 v=40 scene=6; controller state reset\\n");
    }
'''
source='#include <fstream>\n'+old.replace(needle,block+needle)
src=b/'c_api.cpp';src.write_text(source)
patch=subprocess.run(['diff','-u','--label','collision-enabled-c_api.cpp','--label','recorded-restart-c_api.cpp','-',str(src)],input=old,text=True,stdout=subprocess.PIPE)
(a/'c_api.patch').write_text(patch.stdout)
commands=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(r/'build'),'-t','commands'],text=True).splitlines()
args=shlex.split(next(x for x in commands if ' -c '+str(r/'src/c_api.cpp') in x));obj=b/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args[1:1]=['-I'+str(r/'build/crow-v15-wide-contact-isolation-20261111/include'),'-I'+str(r/'src/metal'),'-I'+str(r/'artifacts/crow-clearance-observer-20261109'),'-I'+str(r/'artifacts/crow-v15-two-segment-frozen-flight-contact-map-20261111')]
subprocess.run(args,cwd=r/'build',check=True)
link=json.loads((r/'artifacts/crow-v15-descent-observer-extension-20261114/build-commands.json').read_text())['link']; oldobj=str(r/'build/crow-v15-mirrored-pronation-20261113/c_api.cpp.o');assert oldobj in link
link[link.index(oldobj)]=str(obj);link[link.index('-o')+1]=str(b/'libmetalrobo.dylib');subprocess.run(link,cwd=r/'build',check=True)
(a/'build.json').write_text(json.dumps({'compile':args,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,obj,b/'libmetalrobo.dylib')}},indent=2)+'\n')
print('built reconstructed-state diagnostic host')
