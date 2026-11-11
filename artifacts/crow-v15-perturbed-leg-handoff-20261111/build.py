from pathlib import Path
import difflib,hashlib,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-perturbed-leg-handoff-20261111';B=R/'build/crow-v15-perturbed-leg-handoff-20261111';old=(R/'build/crow-v15-walk-substep-damping-comparison-20261111/zero.cpp').read_text();needle='''    const std::size_t sceneBodyCount =
        handle.defaultSceneBodies.size();''';assert old.count(needle)==1
insert='''    // Development-only bounded reset perturbations; lane zero is an exact control.
    for (std::size_t e=1; e<environmentCount && e<8u; ++e) {
        const std::size_t qb=e*nq, vb=e*nv;
        float roll=0.f,pitch=0.f,yaw=0.f;
        if(e==1u) handle.resetV[vb]+=0.05f;
        if(e==2u) handle.resetV[vb]-=0.05f;
        if(e==3u) roll=0.03f;
        if(e==4u) roll=-0.03f;
        if(e==5u) pitch=0.03f;
        if(e==6u) pitch=-0.03f;
        if(e==7u) {yaw=0.02f;handle.resetV[vb+1]+=0.03f;}
        const float cr=std::cos(roll*0.5f),sr=std::sin(roll*0.5f),cp=std::cos(pitch*0.5f),sp=std::sin(pitch*0.5f),cy=std::cos(yaw*0.5f),sy=std::sin(yaw*0.5f);
        const float dx=sr*cp*cy-cr*sp*sy,dy=cr*sp*cy+sr*cp*sy,dz=cr*cp*sy-sr*sp*cy,dw=cr*cp*cy+sr*sp*sy;
        const float x=handle.resetQ[qb+3],y=handle.resetQ[qb+4],z=handle.resetQ[qb+5],w=handle.resetQ[qb+6];
        handle.resetQ[qb+3]=dw*x+dx*w+dy*z-dz*y;
        handle.resetQ[qb+4]=dw*y-dx*z+dy*w+dz*x;
        handle.resetQ[qb+5]=dw*z+dx*y-dy*x+dz*w;
        handle.resetQ[qb+6]=dw*w-dx*x-dy*y-dz*z;
    }

'''+needle
s=old.replace(needle,insert);src=B/'c_api.cpp';src.write_text(s);(A/'reset.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/zero.cpp',tofile='b/perturbed.cpp')));lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=B/'c_api.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-bounded-launch-commands-20261111/MetalWorld.mm.o');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True);print('library_sha256',hashlib.sha256(lib.read_bytes()).hexdigest())
