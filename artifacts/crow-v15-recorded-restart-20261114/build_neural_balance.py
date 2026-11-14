from pathlib import Path
import hashlib,json,subprocess
r=Path('/Users/n/Developer/numi-crow-rl-20261107');a=r/'artifacts/crow-v15-recorded-restart-20261114';b=r/'build/crow-v15-recorded-restart-20261114'
s=(a/'restart-round-aligned-valid-toe.metal').read_text()
old='if (dispatch.counts.x == 1u && dispatch.counts.z == 41u && dispatch.counts.w == 40u && dispatch.strides.w == 6u) {'
assert s.count(old)==1
s=s.replace(old,'if (dispatch.counts.z == 41u && dispatch.counts.w == 40u && dispatch.strides.w == 6u) {')
needle='resetScene[sceneBase+5u].position.xyz=float3(17.770f,0.46f,0.23f);'
assert s.count(needle)==1
s=s.replace(needle,needle+'''\n            // Independently randomized recorded-state neighborhood, keyed by seed, lane and episode.
            const float dx=(randomUnit(dispatch,environment,episode,0u,4991u)-0.5f)*0.004f;
            const float dy=(randomUnit(dispatch,environment,episode,0u,4992u)-0.5f)*0.004f;
            const float dv=(randomUnit(dispatch,environment,episode,0u,4993u)-0.5f)*0.04f;
            resetQ[qBase+0u]+=dx; resetQ[qBase+1u]+=dy; resetV[vBase+0u]+=dv;''')
old='if (dispatch.sampling.w==0u && fixedFlightController && state.navigation.z>=5.0f) {'
assert s.count(old)==1
s=s.replace(old,'if (fixedFlightController && state.navigation.z>=5.0f) {')
needle='''        const float target = clamp(
            targetCandidate,
            binding.parameters.y,'''
assert s.count(needle)==1
s=s.replace(needle,'''        // Trainable neural residual; actor slots 0..3 control hips/ankles only on support.
        if (fixedFlightController && state.navigation.z>=5.0f && state.episode.x>=2450u &&
            (state.status.w & 63u)>=12u && ((state.status.w>>6u)&63u)>=12u &&
            (action==7u || action==10u || action==9u || action==12u)) {
            const uint neuralSlot=action==7u ? 0u : action==10u ? 1u : action==9u ? 2u : 3u;
            targetCandidate+=0.08f*clamp(actionStream[actionBase+neuralSlot],-1.0f,1.0f);
        }
'''+needle)
needle='''    taskStates[environment] = state;

    const uint transitionIndex ='''
assert s.count(needle)==1
s=s.replace(needle,'''    // Explicit diagnostic curriculum reward, independent physical checks gate promotion.
    if (state.episode.z==10u && state.navigation.z>=5.0f) {
        const bool bilateral=(state.status.w & 63u)>=12u && ((state.status.w>>6u)&63u)>=12u;
        const float support=bilateral ? 1.0f : 0.0f;
        reward=dispatch.timing.x*(2.0f*support - 2.0f*min(tilt*tilt,4.0f)
              -0.4f*min(dot(rootLinearVelocity,rootLinearVelocity),4.0f)
              -0.05f*min(dot(baseAngular,baseAngular),25.0f));
        if(done && !timeout) reward-=2.0f;
    }
    taskStates[environment] = state;

    const uint transitionIndex =''')
src=a/'restart-round-neural-balance.metal';src.write_text(s)
build=json.loads((a/'shader-build.json').read_text());cmd=build['compile'];cmd[cmd.index('-c')+1]=str(src)
air=b/'restart-round-neural-balance.air';lib=b/'restart-round-neural-balance.metallib';cmd[cmd.index('-o')+1]=str(air);subprocess.run(cmd,check=True)
link=build['link'];link[link.index(str(b/'restart.air'))]=str(air);link[link.index('-o')+1]=str(lib);subprocess.run(link,check=True)
host=json.loads((a/'build.json').read_text())['link'];host[host.index(str(b/'c_api.cpp.o'))]=str(b/'c_api-round-perch.o')
out=b/'round-neural-host';out.mkdir(exist_ok=True);dylib=out/'libmetalrobo.dylib';host[host.index('-o')+1]=str(dylib);subprocess.run(host,cwd=r/'build',check=True)
(a/'neural-balance-build.json').write_text(json.dumps({'compile':cmd,'link':link,'host_link':host,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (src,air,lib,dylib)}},indent=2)+'\n');print('neural balance candidate built')
