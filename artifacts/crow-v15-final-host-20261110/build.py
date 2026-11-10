from pathlib import Path
import json,hashlib,subprocess,shlex,difflib,re
R=Path.cwd();A=R/'artifacts/crow-v15-final-host-20261110';B=R/'build/crow-v15-final-host-20261110';base=(R/'build/crow-v15-long-perch-host-20261110/c_api.cpp').read_text();s=base
needle='std::array<std::array<mr_float4, 5u>, 11u> positions';assert s.count(needle)==1;s=s.replace(needle,'std::array<std::array<mr_float4, 5u>, 13u> positions')
reservation=json.loads((R/'artifacts/crow-v15-final-layout-reservation-20261110/reserved-layouts.json').read_text());layouts=reservation['layouts'];assert [x['name'] for x in layouts]==['K','L']
new=[]
for x in layouts:
 assert len(x['positions_m'])==5 and all(len(v)==3 for v in x['positions_m'])
 new.append('        {{{'+('},\n          {'.join(', '.join(f'{v:.6f}f' for v in row)+', 1.0f' for row in x['positions_m']))+'}}},\n')
needle='    }};\n    const auto sceneBodies = handle.world.sceneBodyIndices();';assert s.count(needle)==1;s=s.replace(needle,''.join(new)+needle)
needle='MR_BIRDFLOW_NAVIGATION_COURSE_RESERVED_FINAL_J) {';assert s.count(needle)==1;s=s.replace(needle,'13u) {');(B/'c_api.cpp').write_text(s)
# Archive the reproducible generator; layout bytes are already frozen separately.
(A/'host-source.sha256').write_text(hashlib.sha256(s.encode()).hexdigest()+'  generated-c_api.cpp\n')
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();compile=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);link=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(compile);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(B/'c_api.cpp');args.insert(1,'-I'+str(R/'artifacts/crow-v15-wing-inertia-prototype-20261110'));args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
args=shlex.split(link[len(': && '):-len(' && :')]);args[args.index('-o')+1]=str(B/'libmetalrobo.dylib');args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(R/'build/crow-v15-substep-foot-sensor-20261110/MetalWorld.mm.o');subprocess.run(args,cwd=R/'build',check=True)
binding=(R/'bindings/swift/MetalRoboTaskRollout.swift').read_text();needle='    case reservedFinalJ = 11\n';assert binding.count(needle)==1;binding=binding.replace(needle,needle+'    case reservedFinalK = 12\n    case reservedFinalL = 13\n');(B/'MetalRoboTaskRollout.swift').write_text(binding)
swift=(R/'apps/task_rollout.swift').read_text();needle='                case "reserved-final-j":\n                    birdFlowNavigationCourse = .reservedFinalJ\n';assert swift.count(needle)==1;swift=swift.replace(needle,needle+'                case "reserved-final-k":\n                    birdFlowNavigationCourse = .reservedFinalK\n                case "reserved-final-l":\n                    birdFlowNavigationCourse = .reservedFinalL\n')
pattern=r'(\s*)options\.birdFlowNavigationCourse == \.reservedFinalI\n\1\? "reserved-final-i"';
def prepend(m):
 indent=m.group(1);return indent+'options.birdFlowNavigationCourse == .reservedFinalK\n'+indent+'? "reserved-final-k"\n'+indent+': options.birdFlowNavigationCourse == .reservedFinalL\n'+indent+'? "reserved-final-l"\n'+indent+': options.birdFlowNavigationCourse == .reservedFinalI\n'+indent+'? "reserved-final-i"'
swift,n=re.subn(pattern,prepend,swift);assert n==3,n
swift=swift.replace('options.birdFlowNavigationCourse == .reservedFinalI ||\n                               options.birdFlowNavigationCourse == .reservedFinalJ','options.birdFlowNavigationCourse == .reservedFinalK ||\n                               options.birdFlowNavigationCourse == .reservedFinalL').replace('options.birdFlowNavigationCourse == .reservedFinalI ||\n                       options.birdFlowNavigationCourse == .reservedFinalJ','options.birdFlowNavigationCourse == .reservedFinalK ||\n                       options.birdFlowNavigationCourse == .reservedFinalL')
(B/'task_rollout.swift').write_text(swift)
(A/'swift.patch').write_text(''.join(difflib.unified_diff((R/'apps/task_rollout.swift').read_text().splitlines(True),swift.splitlines(True),fromfile='a/task_rollout.swift',tofile='b/task_rollout.swift'))+''.join(difflib.unified_diff((R/'bindings/swift/MetalRoboTaskRollout.swift').read_text().splitlines(True),binding.splitlines(True),fromfile='a/MetalRoboTaskRollout.swift',tofile='b/MetalRoboTaskRollout.swift')))
args=['/usr/bin/swiftc','-O','-swift-version','6','-I',str(R/'include/metalrobo'),'-I',str(R/'include'),'-I',str(R/'matter/include'),str(B/'task_rollout.swift'),str(R/'apps/CrowEpisodeEvidence.swift'),str(R/'apps/RunInspector.swift'),str(B/'MetalRoboTaskRollout.swift'),'-L',str(B),'-lmetalrobo','-Xlinker','-rpath','-Xlinker',str(B),'-Xlinker','-framework','-Xlinker','AppKit','-o',str(B/'metalrobo_task_rollout')];subprocess.run(args,check=True)
for name in ['libmetalrobo.dylib','metalrobo_task_rollout']:
 p=B/name;print(name,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
