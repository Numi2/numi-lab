from pathlib import Path
import json,shutil,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-refined-regional-graph-240-003';s=q/'snapshots/source-refined-regional-graph-240-004';d=q/'analysis/source240-refined-graph-cmake-004';b=q/'build-source240-refined-graph-cmake-004';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());s.mkdir();d.mkdir()
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
n='matter/CMakeLists.txt';p=s/n;t=p.read_text();anchor='add_executable(numi-matter-regional-graph-completed-tail-check'
assert t.count(anchor)==1
block='''add_executable(numi-matter-cardboard-refined-graph-layout-check
    tools/cardboard_refined_graph_layout_check.cpp)
target_include_directories(numi-matter-cardboard-refined-graph-layout-check PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(numi-matter-cardboard-refined-graph-layout-check PRIVATE
    -Wall -Wextra -Wpedantic -Werror)
if(BUILD_TESTING)
    add_test(NAME matter.compiler.cardboard_refined_graph_layout
        COMMAND numi-matter-cardboard-refined-graph-layout-check)
    set_tests_properties(matter.compiler.cardboard_refined_graph_layout PROPERTIES
        TIMEOUT 120 LABELS "matter;cardboard;solver;cpu;mesh;arena")
endif()
'''
t=t.replace(anchor,block+anchor);p.write_text(t)
f['source_sha256'][n]=sha(p)
if n in cfg['files']:cfg['files'][n]=sha(p)
f['source240_cmake_integration']={'new_target':'numi-matter-cardboard-refined-graph-layout-check','test':'matter.compiler.cardboard_refined_graph_layout','runtime_and_shader_bytes_equal_to_source240003':True}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2','--target','numi-matter-cardboard-refined-graph-layout-check']
commands=[]
with (d/'build.log').open('xb') as out:p=subprocess.run(a,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env)
commands.append({'argv':a,'exit_code':p.returncode})
if p.returncode==0:
 a=['ctest','--test-dir',str(b),'-R','^matter.compiler.cardboard_refined_graph_layout$','--output-on-failure'];p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);(d/'ctest.stdout').write_bytes(p.stdout);(d/'ctest.stderr').write_bytes(p.stderr);commands.append({'argv':a,'exit_code':p.returncode})
r={'schema':'numi.cardboard.source240-cmake-integration.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'commands':commands,'all_pass':len(commands)==2 and all(x['exit_code']==0 for x in commands),'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(int(not r['all_pass']))
