from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-reserved-layout-runtime-20261109';B=R/'build/crow-reserved-layout-runtime-20261109'
base=(R/'build/crow-clearance-observer-20261109/c_api.cpp').read_text()
production=(R/'src/c_api.cpp').read_text()
def region(text):
 a=text.index('void applyCrowNavigationCourseSplit(')
 b=text.index('\nvoid validateTaskRolloutConfiguration(',a)
 return a,b,text[a:b]
a,b,_=region(base);_,_,updated=region(production);s=base[:a]+updated+base[b:]
old='''    if (config.birdflow_navigation_course >
        MR_BIRDFLOW_NAVIGATION_COURSE_DEVELOPMENT_REFERENCE) {'''
new='''    if (config.birdflow_navigation_course >
        MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_D) {'''
assert s.count(old)==1;s=s.replace(old,new)
src=B/'c_api.cpp';src.write_text(s)
(A/'runtime.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/clearance-c_api.cpp',tofile='b/reserved-layout-c_api.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x)
args=shlex.split(line);obj=B/'c_api.cpp.o'
for key,val in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(key)+1]=val
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
args=shlex.split(link[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib)
args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj)
args[args.index('CMakeFiles/metalrobo.dir/src/core/TaskProgram.cpp.o')]=str(R/'build/crow-hierarchical-goal-memory-20261109/TaskProgram.cpp.o')
subprocess.run(args,cwd=R/'build',check=True)
(A/'build.json').write_text(json.dumps({'api_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'committed_course_source_sha256':hashlib.sha256((R/'src/c_api.cpp').read_bytes()).hexdigest(),'scope':'diagnostic runtime with goal-memory task, opt-in clearance, reserved layouts; not a production integration'},indent=2)+'\n')
