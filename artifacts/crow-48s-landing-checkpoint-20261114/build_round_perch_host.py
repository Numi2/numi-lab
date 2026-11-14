from pathlib import Path
import hashlib,json,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
old=(b/'c_api.cpp').read_text()
needle='''                    shape.dimensions.x=0.025f;
                    shape.contactRestAndBoundingRadius.z=std::sqrt(shape.dimensions.x*shape.dimensions.x+shape.dimensions.y*shape.dimensions.y+shape.dimensions.z*shape.dimensions.z);'''
assert old.count(needle)==1
replacement='''                    // Research fixture: actual rigid capsule collision, not a box render.
                    shape.shapeType=MR_SHAPE_CAPSULE;
                    shape.dimensions={0.025f,0.695f,0.0f,0.0f};
                    shape.contactRestAndBoundingRadius.z=0.72f;
                    std::fprintf(stderr,"CROW_ROUND_PERCH type=%u radius=%.9g half_length=%.9g group=%u mask=%u\\n",unsigned(shape.shapeType),shape.dimensions.x,shape.dimensions.y,unsigned(shape.collisionGroup),unsigned(shape.collisionMask));'''
src=a/'c_api-round-perch.cpp';src.write_text(old.replace(needle,replacement))
build=json.loads((a/'build.json').read_text());cmd=build['compile'];obj=b/'c_api-round-perch.o'
cmd[cmd.index('-o')+1]=str(obj);cmd[cmd.index('-MF')+1]=str(obj)+'.d';cmd[cmd.index('-MT')+1]=str(obj);cmd[-1]=str(src)
subprocess.run(cmd,cwd=r/'build',check=True)
link=json.loads((a/'full-contact-observer-build.json').read_text())['link'];link[link.index(str(b/'c_api.cpp.o'))]=str(obj)
out=b/'round-perch-observer';out.mkdir(exist_ok=False);lib=out/'libmetalrobo.dylib';link[link.index('-o')+1]=str(lib);subprocess.run(link,cwd=r/'build',check=True)
(a/'round-perch-host-build.json').write_text(json.dumps({'compile':cmd,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,obj,lib)}},indent=2)+'\n');print('built round-perch host')
