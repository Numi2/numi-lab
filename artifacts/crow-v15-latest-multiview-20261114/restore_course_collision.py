"""Build a separate, instrumented host with course collision masks restored.

Leaves the recorded diagnostic host and old evidence untouched. A smoke test
and independent contact/penetration experiments are still required.
"""
from pathlib import Path
import gzip, hashlib, json, shlex, subprocess

r = Path('/Users/n/Developer/numi-crow-rl-20261107')
a = r/'artifacts/crow-v15-course-collision-restored-20261114'
b = r/'build/crow-v15-course-collision-restored-20261114'
a.mkdir(exist_ok=False)
b.mkdir(exist_ok=False)
original = gzip.decompress((r/'artifacts/crow-v15-mirrored-pronation-20261113/c_api.cpp.gz').read_bytes()).decode()
start = original.index('        // Diagnostic only: preserve the ground and disable exactly the five course colliders.')
end = original.index('        manifest.sensors.id =', start)
replacement = '''        // Collision-on candidate: keep the authored masks and expose them in the run log.
        if (activeNavigation) {
            unsigned found=0;
            for (auto& object:manifest.scene.objects) for (std::size_t i=0;i<object.mechanics.shapeNames.size();++i) {
                const auto& name=object.mechanics.shapeNames[i]; const auto& shape=object.mechanics.shapes.at(i);
                if (name.starts_with("crow_course_")) {
                    if (shape.collisionMask == 0u) throw std::logic_error("course collider unexpectedly disabled");
                    ++found;
                    std::fprintf(stderr,"CROW_SCENE_MASK_RESTORED %s %u\\n",name.c_str(),unsigned(shape.collisionMask));
                }
            }
            if (found!=5u) throw std::logic_error("expected five enabled course shapes");
        }
'''
source = original[:start] + replacement + original[end:]
src = b/'c_api.cpp'; src.write_text(source)
(a/'c_api.patch').write_text(subprocess.run(['diff','-u','--label','prior-c_api.cpp','--label','collision-restored-c_api.cpp','-',str(src)],input=original,text=True,stdout=subprocess.PIPE).stdout)
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(r/'build'),'-t','commands'],text=True).splitlines()
args=shlex.split(next(x for x in lines if ' -c '+str(r/'src/c_api.cpp') in x))
obj=b/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]: args[args.index(k)+1]=v
args[-1]=str(src)
args[1:1]=['-I'+str(r/'build/crow-v15-wide-contact-isolation-20261111/include'),'-I'+str(r/'src/metal'),'-I'+str(r/'artifacts/crow-clearance-observer-20261109'),'-I'+str(r/'artifacts/crow-v15-two-segment-frozen-flight-contact-map-20261111')]
subprocess.run(args,cwd=r/'build',check=True)
link=json.loads((r/'artifacts/crow-v15-descent-observer-extension-20261114/build-commands.json').read_text())['link']
old=str(r/'build/crow-v15-mirrored-pronation-20261113/c_api.cpp.o')
assert old in link
link[link.index(old)]=str(obj)
link[link.index('-o')+1]=str(b/'libmetalrobo.dylib')
subprocess.run(link,cwd=r/'build',check=True)
paths=[src,obj,b/'libmetalrobo.dylib']
(a/'build.json').write_text(json.dumps({'compile':args,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
print('Built collision-restored diagnostic candidate. No rollout or contact qualification yet.')
