"""Compile an isolated shader that restores saved q/v/scene after first reset.

The task/controller episode state remains newly initialized.
"""
from pathlib import Path
import hashlib,json,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114';f=r/'artifacts/crow-v15-flight-feedback-freeze-20261110';w=r/'build/crow-v15-wide-contact-isolation-20261111';orig=r/'artifacts/crow-v15-uniform-wing-handoff-20261114/push.metal'
snapshot=json.loads(Path('/tmp/crow-48s-multiview-20261114/recorded-state.json').read_text());frame=snapshot['last_three_recorded_frames'][-1];q=frame['q'];v=frame['v'];scene=frame['body_states'][35*13:41*13];assert (len(q),len(v),len(scene))==(41,40,78)
def fmt(x):
    s=format(x,'.10g');return (s if '.' in s or 'e' in s else s+'.0')+'f'
def arr(name,x):return 'constant float '+name+'['+str(len(x))+'] = {'+', '.join(map(fmt,x))+'};\n'
constants=arr('crowRestartQ',q)+arr('crowRestartV',v)+arr('crowRestartScene',scene)
source=orig.read_text();needle='        device float* firstActor =\n';assert source.count(needle)==1
inject='''        // Diagnostic reconstruction: task/controller history is freshly initialized.
        if (dispatch.counts.x == 1u && dispatch.counts.z == 41u && dispatch.counts.w == 40u && dispatch.strides.w == 6u) {
            for (uint i=0; i<41u; ++i) resetQ[qBase+i]=crowRestartQ[i];
            for (uint i=0; i<40u; ++i) resetV[vBase+i]=crowRestartV[i];
            for (uint i=0; i<6u; ++i) {
                const uint j=i*13u; device MRBodyStateGPU& s=resetScene[sceneBase+i];
                s.position.xyz=float3(crowRestartScene[j],crowRestartScene[j+1u],crowRestartScene[j+2u]);
                s.orientation=float4(crowRestartScene[j+3u],crowRestartScene[j+4u],crowRestartScene[j+5u],crowRestartScene[j+6u]);
                s.linearVelocityAndInverseMass.xyz=float3(crowRestartScene[j+7u],crowRestartScene[j+8u],crowRestartScene[j+9u]);
                s.angularVelocity.xyz=float3(crowRestartScene[j+10u],crowRestartScene[j+11u],crowRestartScene[j+12u]);
            }
        }
'''
new=constants+source.replace(needle,inject+needle);out=a/'restart.metal';out.write_text(new)
compile=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(f),'-I',str(w/'include'),'-I',str(r/'include'),'-I',str(r/'src/metal'),'-c',str(out),'-o',str(b/'restart.air')];subprocess.run(compile,check=True)
replaced={p.name:p for p in (w/'shaders').glob('*.air')};world=r/'build/crow-v15-uniform-wing-handoff-20261114/World.air';airs=[str(world if p.name=='MetalRoboWorld.air' else replaced.get(p.name,p)) for p in sorted((r/'build/shaders').glob('MetalRobo*.air')) if p.name not in ('MetalRoboLocomotionTask.air','MetalRoboFlappingWingDynamics.air')]
link=['xcrun','-sdk','macosx','metallib',*airs,str(r/'build/crow-v15-energy-consistent-aero-20261113/aero.air'),str(b/'restart.air'),'-o',str(b/'restart.metallib')];subprocess.run(link,check=True)
(a/'shader-build.json').write_text(json.dumps({'compile':compile,'link':link,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (orig,out,b/'restart.air',b/'restart.metallib')}},indent=2)+'\n');print('built reconstructed reset shader')
