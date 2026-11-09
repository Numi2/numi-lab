from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-third-native-observer-20261109';B=R/'build/crow-third-native-observer-20261109'
base=(R/'src/c_api.cpp').read_text()
obs=(R/'build/crow-clearance-observer-20261109/c_api.cpp').read_text()
s=base.replace('#include "metalrobo/RetinalObserver.hpp"','#include "box_distance.hpp"\n#include "metalrobo/RetinalObserver.hpp"',1)
a=obs.index('struct CrowClearanceRecord {');b=obs.index('struct MRTaskRolloutHandle {',a)
record=obs[a:b]
s=s.replace('struct MRTaskRolloutHandle {',record+'struct MRTaskRolloutHandle {\n    std::vector<CrowClearanceRecord> crowClearance;',1)
old='handle->stepConfig.captureContactEvidence = false;'
new='handle->stepConfig.captureContactEvidence = std::getenv("CROW_CONTACT_TRACE") != nullptr;'
assert s.count(old)==1;s=s.replace(old,new)
a=obs.index('        if (std::getenv("CROW_CLEARANCE_TRACE") != nullptr &&')
b=obs.index('        if (handle->visualRuntime && handle->visualRuntime->retinal) {',a)
clearance=obs[a:b]
needle='        if (handle->visualRuntime && handle->visualRuntime->retinal) {'
assert s.count(needle)==1;s=s.replace(needle,clearance+needle,1)
a=obs.index('        if (std::getenv("CROW_CONTACT_TRACE") != nullptr &&',b)
b=obs.index('        handle->pendingLifeWorld.clear();',a)
tracing=obs[a:b]
tracing=tracing.replace('array("applied_filtered_action", published.teacherActions,','array("policy_action", published.policyActions,')
needle='        handle->pendingLifeWorld.clear();'
assert s.count(needle)==1;s=s.replace(needle,tracing+needle,1)
src=B/'c_api.cpp';src.write_text(s)
(A/'observer.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/src/c_api.cpp',tofile='b/production-observer-c_api.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines()
line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x)
args=shlex.split(line);obj=B/'c_api.cpp.o'
for key,val in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(key)+1]=val
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x)
args=shlex.split(link[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib)
args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj)
subprocess.run(args,cwd=R/'build',check=True)
(A/'build.json').write_text(json.dumps({'api_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'library_sha256':hashlib.sha256(lib.read_bytes()).hexdigest(),'production_api_sha256':hashlib.sha256((R/'src/c_api.cpp').read_bytes()).hexdigest(),'scope':'opt-in read-only contact, support and sampled box-clearance evidence for production task'},indent=2)+'\n')
