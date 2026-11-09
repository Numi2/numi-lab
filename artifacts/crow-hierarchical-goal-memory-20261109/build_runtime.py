from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-hierarchical-goal-memory-20261109';B=R/'build/crow-hierarchical-goal-memory-20261109';B.mkdir(parents=True,exist_ok=True)
old='american_crow_active_head_navigation_development_v1';new='american_crow_v15_goal_memory_navigation_development_v1'
base=(R/'src/core/TaskProgram.cpp').read_text()
token='pack.id == "'+old+'"'
assert base.count(token)==5
s=base.replace(token,'('+token+' || pack.id == "'+new+'")')
needle='''    if (threatGroup != MR_INVALID_INDEX) {'''
assert s.count(needle)==1
s=s.replace(needle,'''    if (pack.id == "'''+new+'''") staged->header.schedule.w |= (1u << 28u);
'''+needle)
task=B/'TaskProgram.cpp';task.write_text(s)
(A/'task-program.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/core/TaskProgram.cpp',tofile='b/src/core/TaskProgram.cpp')))
base2=(R/'build/crow-bounded-support-observer-20261109/c_api.cpp').read_text()
needle2='''        if (activeNavigation) manifest.task.maximumEpisodeSteps = 3000u;'''
assert base2.count(needle2)==1
s2=base2.replace(needle2,needle2+'''
        if (activeNavigation) manifest.task.id = "'''+new+'''";''')
api=B/'c_api.cpp';api.write_text(s2)
(A/'api.patch').write_text(''.join(difflib.unified_diff(base2.splitlines(True),s2.splitlines(True),fromfile='a/extended-observer-c_api.cpp',tofile='b/hierarchy-c_api.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
objects={}
for original,source in [('src/core/TaskProgram.cpp',task),('src/c_api.cpp',api)]:
 line=next(x for x in lines if ' -c '+str(R/original) in x)
 args=shlex.split(line);obj=B/(Path(original).name+'.o')
 for key,val in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(key)+1]=val
 args[-1]=str(source);subprocess.run(args,cwd=R/'build',check=True);objects[original]=obj
link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
args=shlex.split(link[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib)
for original,obj in objects.items():args[args.index('CMakeFiles/metalrobo.dir/'+original+'.o')]=str(obj)
subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'build.json').write_text(json.dumps({'task_id':new,'schedule_bit':28,'task_program_sha256':h(task),'api_sha256':h(api),'library_sha256':h(lib),'frozen_controller_sha256':h(R/'src/metal/CrowFlightControllerV1.metal')},indent=2)+'\n')
