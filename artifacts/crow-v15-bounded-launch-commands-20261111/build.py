from pathlib import Path
import difflib,hashlib,json,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-bounded-launch-commands-20261111';B=R/'build/crow-v15-bounded-launch-commands-20261111';P=R/'build/crow-v15-walk-substep-damping-comparison-20261111';old=(R/'build/crow-v15-repaired-toe-takeoff-control-20261111/MetalWorld.mm').read_text();needle='crowTraceStep <= 355';assert old.count(needle)==1;s=old.replace(needle,'crowTraceStep <= 505');src=B/'MetalWorld.mm';src.write_text(s);(A/'observer.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/launch.mm',tofile='b/extended.mm')))
lines=subprocess.check_output(['/opt/homebrew/bin/ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=B/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'src/metal'));args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(obj);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(P/'zero.o');subprocess.run(args,cwd=R/'build',check=True);print('library',hashlib.sha256(lib.read_bytes()).hexdigest(),flush=True)
base=(R/'build/crow-v15-contact-flight-handoff-20261111/handoff.metal').read_text();needle='''        const float4 oracleCommand = privilegedCommand ? state.nutrition : actorCommand;''';assert base.count(needle)==1;rows=[]
for speed in ('040','070'):
 for height in ('065','085','105'):
  name='v'+speed+'-h'+height;repl='''        float4 oracleCommand = privilegedCommand ? state.nutrition : actorCommand;
        if (!privilegedCommand && (state.status.w & 8192u) != 0u) {
            const float launchSeconds = float(state.episode.x)*dispatch.timing.x;
            const float blend = smoothstep(7.0f,8.0f,launchSeconds);
            oracleCommand = mix(float4(0.'''+speed+'''f,0.0f,0.0f,'''+('1.05f' if height=='105' else '0.'+height+'f')+'''),oracleCommand,blend);
        }''';s=base.replace(needle,repl);src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/handoff.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');shader=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(shader)],check=True);rows.append({'name':name,'sha256':hashlib.sha256(shader.read_bytes()).hexdigest()});print(name,flush=True)
(A/'shaders.json').write_text(json.dumps(rows,indent=2)+'\n')
