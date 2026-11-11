from pathlib import Path
import difflib,hashlib,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-v1-contact-balance-landing-20261111';B=R/'build/crow-v15-v1-contact-balance-landing-20261111';base=(R/'build/crow-v15-v1-rear-first-narrow-approach-20261111/teacher.metal').read_text();s=base
old='float errorOffset = 0.0f;\n            if (landing) {';new='float errorOffset = 0.0f;\n            float landingBalanceError = 0.0f;\n            if (landing) {';assert s.count(old)==1;s=s.replace(old,new)
old='const float xError = landingTarget.x-pos.x;';new='const float xError = landingTarget.x-pos.x;\n                landingBalanceError = perch.x - 0.012f - pos.x;';assert s.count(old)==1;s=s.replace(old,new)
old='state.commandAndPhase.xyz=float3(0.15f,0.0f,0.0f);';new='state.commandAndPhase.xyz=float3(0.15f,landingBalanceError,0.0f);';assert s.count(old)==1;s=s.replace(old,new)
old='''        const float target = clamp(
            targetCandidate,''';new='''        if (fixedFlightController && state.navigation.z>=5.0f && state.threatTeacher.w>0.5f &&
            (binding.indices.x==7u || binding.indices.x==9u || binding.indices.x==10u || binding.indices.x==12u)) {
            const bool right=binding.indices.x>=10u;
            const float load=right ? state.nutrition.y : state.nutrition.x;
            if (load>0.05f) {
                const float positionCorrection=clamp(state.commandAndPhase.y-0.2f*state.commandExtension.z,-0.08f,0.08f);
                const float pitchCorrection=clamp(-0.25f*(state.threatGeometry.z+0.15f*state.threatGeometry.w),-0.08f,0.08f);
                const bool hip=binding.indices.x==7u || binding.indices.x==10u;
                targetCandidate += hip ? positionCorrection+pitchCorrection : -pitchCorrection;
            }
        }
        const float target = clamp(
            targetCandidate,''';assert s.count(old)==1;s=s.replace(old,new)
(B/'teacher.metal').write_text(s);(A/'teacher.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/rear-first-narrow.metal',tofile='b/contact-balance.metal')));air=B/'teacher.air';lib=B/'teacher.metallib';subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(B/'teacher.metal'),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print('shader_sha256',hashlib.sha256(lib.read_bytes()).hexdigest())
