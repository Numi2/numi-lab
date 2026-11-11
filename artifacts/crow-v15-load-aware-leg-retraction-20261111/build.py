from pathlib import Path
import difflib,hashlib,subprocess,json
R=Path.cwd();A=R/'artifacts/crow-v15-load-aware-leg-retraction-20261111';B=R/'build/crow-v15-load-aware-leg-retraction-20261111';src0=(R/'build/crow-v15-toe-launch-control-20261110/push.metal').read_text();needle='''                    : bilateral ? mix(crouched,extended,smoothstep(4.96f,5.00f,seconds)) : crouched;
            }''';assert src0.count(needle)==1;out=[]
for height in ('166','170','174'):
 for kind in ('crouch','moderate'):
  target='crouched' if kind=='crouch' else '(joint==0u ? -0.45f : joint==1u ? 0.55f : -0.30f)';name='h'+height+'-'+kind
  repl=needle.replace('\n            }','\n                if (seconds >= 5.00f && state.airReturnTracking.y > 0.'+height+'f) navigationTeacherRequested = '+target+';\n            }');s=src0.replace(needle,repl);src=B/(name+'.metal');src.write_text(s);(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(src0.splitlines(True),s.splitlines(True),fromfile='a/push.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');lib=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(lib)],check=True);out.append({'name':name,'shader_sha256':hashlib.sha256(lib.read_bytes()).hexdigest()});print(name,flush=True)
(A/'shaders.json').write_text(json.dumps(out,indent=2)+'\n')
