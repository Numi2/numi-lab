from pathlib import Path
import difflib,subprocess,hashlib
R=Path.cwd();A=R/'artifacts/crow-v15-sustained-gait-20261110';B=R/'build/crow-v15-sustained-gait-20261110';base=(R/'build/crow-v15-gait-no-wrench-refine-20261110/stance014.metal').read_text();s=base
for old,new,count in [
 ('seconds >= 2.0f && seconds < 4.0f','seconds >= 2.0f && seconds < 12.0f',2),
 ('seconds >= 4.3f && seconds < 5.4f','seconds >= 12.3f && seconds < 13.4f',1),
 ('seconds<4.96f','seconds<12.96f',1),('smoothstep(4.3f,4.55f,seconds)','smoothstep(12.3f,12.55f,seconds)',1),('smoothstep(4.96f,5.00f,seconds)','smoothstep(12.96f,13.00f,seconds)',1),
 ('(action == 13u || action == 14u) && float(state.episode.x)*dispatch.timing.x < 5.4f','(action < 7u || action == 13u || action == 14u) && float(state.episode.x)*dispatch.timing.x < 13.4f',1),
 ('float(state.episode.x)*dispatch.timing.x < 5.4f','float(state.episode.x)*dispatch.timing.x < 13.4f',1),
 ('seconds>=4.96f && seconds<5.4f','seconds>=12.96f && seconds<13.4f',1),('seconds>=5.4f) packed','seconds>=13.4f) packed',1)]:
 assert s.count(old)==count,(old,s.count(old));s=s.replace(old,new)
old='''        if (kind == MR_TASK_ACTUATOR_BODY_WRENCH &&
            (worldDispatch.flags & MR_METAL_WORLD_HAS_BODY_WRENCHES) != 0u &&
            binding.actuator.y < bodyCount && binding.actuator.z < 6u) {
''';new=old+'''            const MRTaskStateGPU grounded = taskStates[environment];
            if ((program.schedule.w & MR_TASK_PROGRAM_AVIAN_CROW_HIERARCHICAL_NAVIGATION) != 0u &&
                grounded.episode.z == 10u && (action == 13u || action == 14u) &&
                float(grounded.episode.x)*dispatch.timing.x < 13.4f &&
                (grounded.status.w & 4096u)==0u) continue;
''';assert s.count(old)==1;s=s.replace(old,new)
for name in ('push-trigger','control-trigger'):
 candidate=s
 if name=='control-trigger':
  old=': bilateral ? mix(crouched,extended,smoothstep(12.96f,13.00f,seconds)) : crouched;';assert candidate.count(old)==1;candidate=candidate.replace(old,': crouched;')
 src=B/(name+'.metal');src.write_text(candidate);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/stance014.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);print(name,hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
