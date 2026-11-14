from pathlib import Path
import hashlib
base=Path('/private/tmp/crow-release-gated-baseline.metal').read_text(); old=Path('/private/tmp/crow-common-role6.metal').read_text()
start=old.index('        // Development-only physical joint targets, onset varied by paired lane.')
start=old.index('            const float seconds=',start)
end=old.index('        // Candidate lanes: wing targets remain at trim',start)
body=old[start:end]; assert body.endswith('        }\n'); body=body[:-10]
assert 'const uint arm=6u;' in body and 'targetCandidate=defaultQ[binding.indices.z];' in body
marker='        // Diagnostic ablation: match the tail across push and hold arms.'
assert base.count(marker)==1
prefix='''        // Development-only: transition from release-gated launch to the prior role-6 wing law.
        if (fixedFlightController && state.episode.z==10u && action<=5u &&
            float(state.episode.x)*dispatch.timing.x>=10.02f &&
            float(state.episode.x)*dispatch.timing.x<36.02f && (state.status.w & 8192u)!=0u) {
            const float t=float(state.episode.x)*dispatch.timing.x;
            const float onset=7.50f;
            const float launchPhase=kTwoPi*6.4f*(t-onset);
            const float launchRamp=smoothstep(onset,onset+0.04f,t);
            const float amplitudes[4]={.70f,.75f,.80f,.85f};
            float launchTarget=defaultQ[binding.indices.z];
            if (action<=1u) launchTarget+=launchRamp*mix(.55f,amplitudes[environment%4u],smoothstep(7.50f,7.58f,t))*sin(launchPhase);
            else if (action<=3u) launchTarget+=launchRamp*.12f;
            else launchTarget+=launchRamp*.06f*sin(launchPhase+3.14159265359f);
'''
suffix='''            targetCandidate=mix(launchTarget,targetCandidate,smoothstep(10.02f,10.52f,t));
        }
'''
out=base.replace(marker,prefix+body+suffix+marker)
Path('artifacts/crow-v15-role6-wing-handoff-20261114/push.metal').write_text(out)
print('base',hashlib.sha256(base.encode()).hexdigest(),'prior-role6',hashlib.sha256(old.encode()).hexdigest(),'candidate',hashlib.sha256(out.encode()).hexdigest())
