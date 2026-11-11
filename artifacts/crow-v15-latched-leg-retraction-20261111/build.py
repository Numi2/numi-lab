from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-latched-leg-retraction-20261111';B=R/'build/crow-v15-latched-leg-retraction-20261111';base=(R/'build/crow-v15-toe-launch-control-20261110/push.metal').read_text();needle='''                    : bilateral ? mix(crouched,extended,smoothstep(4.96f,5.00f,seconds)) : crouched;
            }''';assert base.count(needle)==1;status='''            state.status.w=packed;''';assert base.count(status)==1;out=[]
for height in ('166','170','174'):
 for ramp in ('fast','base'):
  name='h'+height+'-'+ramp;repl=needle.replace('5.00f,seconds','4.98f,seconds') if ramp=='fast' else needle;repl=repl.replace('\n            }','\n                if ((state.status.w & 4096u) != 0u) navigationTeacherRequested = crouched;\n            }');s=base.replace(needle,repl);latch='''            const uint launchRetractLatch = seconds >= 5.00f && seconds < 5.40f &&
                (((state.status.w & 4096u) != 0u) || state.airReturnTracking.y > 0.'''+height+'''f) ? 4096u : 0u;
            state.status.w=packed | launchRetractLatch;''';s=s.replace(status,latch);src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/push.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);out.append({'name':name,'sha256':hashlib.sha256(lib.read_bytes()).hexdigest()});print(name,flush=True)
(A/'shaders.json').write_text(json.dumps(out,indent=2)+'\n')
