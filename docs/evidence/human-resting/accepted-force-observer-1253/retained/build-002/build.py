from pathlib import Path
import hashlib,json,shlex,subprocess,shutil,time
SOURCE=Path('/Users/n/numi-human-accepted-force-observer-1253')
BUILD=Path('/Users/n/numi-human-accepted-force-build-1253-002')
PREVIOUS=Path('/Users/n/numi-human-q-integration-publish-build-002')
RUNTIME=Path('/Users/n/numi-human-contoured-bed-build-1196')
RUNTIME_SOURCE='/Users/n/numi-human-contoured-bed-source-1196'
OUT=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(SOURCE/'apps/numilab_human_myosim_visual_probe.mm')=='f089fdff77f0b47371f2d9553004ef1a81a3b248ab5ca3c3c4aa3f4508117e7a'
assert not BUILD.exists()
shutil.copytree(PREVIOUS,BUILD,symlinks=True,ignore=shutil.ignore_patterns('*.o','numi-human-native'))
entry=next(x for x in json.loads((RUNTIME/'compile_commands.json').read_text()) if x.get('output','').startswith('CMakeFiles/numi-human-native.dir/'))
compile_args=[a.replace(RUNTIME_SOURCE,str(SOURCE)) for a in shlex.split(entry['command'])]
compile_args=[('-O2' if a=='-O3' else a) for a in compile_args]
obj=BUILD/'numilab_human_myosim_visual_probe.mm.o'
compile_args[compile_args.index('-o')+1]=str(obj)
link_args=shlex.split((RUNTIME/'CMakeFiles/numi-human-native.dir/link.txt').read_text())
link_args=[('-O2' if a=='-O3' else str(obj) if a.endswith('.mm.o') else a) for a in link_args]
link_args[link_args.index('-o')+1]=str(BUILD/'bin/numi-human-native')
link_args=[str(RUNTIME/a) if a=='lib/libmetalrobo.dylib' else a for a in link_args]
pins={str(p):sha(p) for p in [Path(__file__),SOURCE/'apps/numilab_human_myosim_visual_probe.mm',SOURCE/'apps/NumiHumanAcceptedQAuditWindow.hpp',SOURCE/'include/metalrobo/numi_human_accepted_force_observer.hpp',SOURCE/'tests/numi_human_accepted_force_observer_test.cpp',SOURCE/'CMakeLists.txt',RUNTIME/'compile_commands.json',RUNTIME/'CMakeFiles/numi-human-native.dir/link.txt',RUNTIME/'lib/libmetalrobo.dylib']}
records=[]
for name,args in [('compile',compile_args),('link',link_args)]:
 start=time.monotonic()
 with (OUT/(name+'.log')).open('x') as f:r=subprocess.run(args,cwd=BUILD,stdout=f,stderr=subprocess.STDOUT)
 records.append({'phase':name,'argv':args,'returncode':r.returncode,'wall_seconds':time.monotonic()-start})
 if r.returncode:break
report={'source':str(SOURCE),'source_head':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),'commands':records,'input_sha256':pins,'inputs_unchanged':all(sha(Path(p))==h for p,h in pins.items()),'outputs':{str(p):sha(p) for p in [obj,BUILD/'bin/numi-human-native'] if p.is_file()},'native_execution':False}
with (OUT/'build-report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'commands':[{'phase':x['phase'],'returncode':x['returncode'],'wall_seconds':x['wall_seconds']} for x in records],'outputs':report['outputs'],'inputs_unchanged':report['inputs_unchanged']}),flush=True)
raise SystemExit(records[-1]['returncode'])
