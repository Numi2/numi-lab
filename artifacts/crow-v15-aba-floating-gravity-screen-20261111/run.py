from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-aba-floating-gravity-screen-20261111';B=R/'build/crow-v15-aba-floating-gravity-screen-20261111';P=R/'build/crow-v15-aba-40-body-isolation-20261111';base=(R/'src/metal/ArticulatedABA.metal').read_text();s=base
old='''        float3 externalForce =
            massScale * body.massAndInverseMass.x *
            world.gravityAndTimestep.xyz;''';new='''        float3 externalForce = articulation.rootType == MR_ROOT_FLOATING
            ? float3(0.0f)
            : massScale * body.massAndInverseMass.x * world.gravityAndTimestep.xyz;''';assert s.count(old)==1;s=s.replace(old,new)
old='''        candidateAcceleration[0] = rootLinearAcceleration.x;
        candidateAcceleration[1] = rootLinearAcceleration.y;
        candidateAcceleration[2] = rootLinearAcceleration.z;''';new='''        candidateAcceleration[0] = rootLinearAcceleration.x + world.gravityAndTimestep.x;
        candidateAcceleration[1] = rootLinearAcceleration.y + world.gravityAndTimestep.y;
        candidateAcceleration[2] = rootLinearAcceleration.z + world.gravityAndTimestep.z;''';assert s.count(old)==1;s=s.replace(old,new)
(B/'aba.metal').write_text(s);(A/'aba.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/ArticulatedABA.metal',tofile='b/floating-gravity.metal')));air=B/'aba.air';lib=B/'aba.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(P/'include'),'-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'aba.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',str(air),'-o',str(lib)],check=True)
for n in ('diagnostic','control'):
 s=(R/('artifacts/crow-v15-aba-varied-math-control-20261111/'+n+'.cpp')).read_text();old=str(P/'aba.metallib');assert s.count(old)==1;src=A/(n+'.cpp');src.write_text(s.replace(old,str(lib)));exe=B/n;subprocess.run(['xcrun','clang++','-std=c++20','-O2','-I'+str(P/'include'),'-Iinclude',str(src),'-L'+str(P),'-lmetalrobo','-Wl,-rpath,'+str(P),'-o',str(exe)],check=True)
 with (A/(n+'.json')).open('wb') as out,(A/(n+'.stderr')).open('wb') as err:r=subprocess.run([str(exe)],stdout=out,stderr=err)
 print(n,'exit',r.returncode,'metallib_sha256',hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
