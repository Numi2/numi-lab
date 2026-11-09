from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109';base=(B/'walk-handoff.metal').read_text();marker='''        controllerParameters[environment].z =
            float(actionDelay) * dispatch.timing.x;
''';assert base.count(marker)==1
inject='''        // Experimental start location adjustment; dynamics own all subsequent motion.
        if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u && curriculum == 10u)
            resetQ[qBase + program.root.z] -= 0.052f;
''';s=base.replace(marker,inject+marker);(A/'offset-handoff.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/walk-handoff.metal',tofile='b/offset-handoff.metal')));src=B/'offset-handoff.metal';air=B/'offset-handoff.air';lib=B/'offset-handoff.metallib';src.write_text(s)
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True)
subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'offset-handoff-build.json').write_text(json.dumps({'shader_sha256':h(lib),'patch_sha256':h(A/'offset-handoff.patch'),'plan_sha256':h(A/'offset-handoff-plan.md'),'base_shader':'walk-handoff'},indent=2)+'\n')
s=(A/'full_handoff.py').read_text();start=s.index('# Instrumentation test');end=s.index("for course,seed in [('reserved-final-i'",start);s=s[:start]+s[end:];s=s.replace("lib=B/'walk-handoff.metallib';assert h(lib)==json.loads((A/'handoff-reserved-final-i-bindings.json').read_text())['shader_sha256']","lib=B/'offset-handoff.metallib';assert h(lib)==json.loads((A/'offset-handoff-build.json').read_text())['shader_sha256'];base[base.index('--metallib')+1]=str(lib)");s=s.replace("h(A/'full-handoff-plan.md')","h(A/'offset-handoff-plan.md')");s=s.replace("name='full-handoff-'+course","name='offset-handoff-'+course");(A/'offset_handoff.py').write_text(s)
