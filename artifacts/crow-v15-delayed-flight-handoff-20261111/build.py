from pathlib import Path
import difflib,hashlib,json,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-delayed-flight-handoff-20261111';B=R/'build/crow-v15-delayed-flight-handoff-20261111';base=(R/'build/crow-v15-propulsive-leg-push-20261111/hn50-an30.metal').read_text();needle='const float extended=joint==0u ? -0.50f : joint==1u ? -0.70f : -0.30f;';assert base.count(needle)==1
old='''            const uint flightReleaseLatch = seconds >= 5.00f && seconds < 8.00f &&
                (((state.status.w & 8192u) != 0u) ||
                 (launchRetractLatch != 0u && state.nutrition.x <= 0.05f && state.nutrition.y <= 0.05f)) ? 8192u : 0u;
            state.status.w=packed | launchRetractLatch | flightReleaseLatch;''';new='''            const bool lowLoad = seconds >= 5.00f && seconds < 8.00f && launchRetractLatch != 0u && state.nutrition.x <= 0.05f && state.nutrition.y <= 0.05f;
            const uint lowLoad1 = lowLoad ? 16384u : 0u;
            const uint lowLoad2 = lowLoad && (state.status.w & 16384u) != 0u ? 32768u : 0u;
            const uint flightReleaseLatch = seconds >= 5.00f && seconds < 8.00f && (((state.status.w & 8192u) != 0u) || (lowLoad && (state.status.w & 32768u) != 0u)) ? 8192u : 0u;
            state.status.w=packed | launchRetractLatch | lowLoad1 | lowLoad2 | flightReleaseLatch;''';assert base.count(old)==1;base=base.replace(old,new);old2='if (!privilegedCommand && (state.status.w & 8192u) != 0u) {';new2='if (!privilegedCommand && (state.status.w & (8192u | 16384u)) != 0u) {';assert base.count(old2)==1;base=base.replace(old2,new2)
rows=[]
for name,h,a in [('hn50-ap18','-0.50','0.18'),('hn50-ap30','-0.50','0.30'),('hn50-ap45','-0.50','0.45'),('hn70-ap45','-0.70','0.45'),('hn85-ap45','-0.85','0.45'),('hn100-ap45','-1.00','0.45')]:
 s=base.replace(needle,'const float extended=joint==0u ? '+h+'f : joint==1u ? -0.70f : '+a+'f;');src=B/(name+'.metal');src.write_text(s);original=(R/'build/crow-v15-propulsive-leg-push-20261111/hn50-an30.metal').read_text();(A/(name+'.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/immediate-handoff.metal',tofile='b/'+name+'.metal')));air=B/(name+'.air');shader=B/(name+'.metallib');subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(R/'include'),'-I',str(R/'src/metal'),'-c',str(src),'-o',str(air)],check=True);subprocess.run(['xcrun','-sdk','macosx','metallib',*[str(p) for p in sorted((R/'build/shaders').glob('MetalRobo*.air')) if p.name!='MetalRoboLocomotionTask.air'],str(air),'-o',str(shader)],check=True);rows.append({'name':name,'sha256':hashlib.sha256(shader.read_bytes()).hexdigest()});print(name,flush=True)
(A/'shaders.json').write_text(json.dumps(rows,indent=2)+'\n')
