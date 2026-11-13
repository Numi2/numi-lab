from pathlib import Path
import gzip,hashlib,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-launch-ground-contact-audit-20261113';B=R/'build/crow-v15-launch-ground-contact-audit-20261113';A.mkdir(exist_ok=False);B.mkdir(exist_ok=False);s=(R/'build/crow-v15-leg-work-observer-20261113/MetalWorld.mm').read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
needle='const auto n=std::min<NSUInteger>(statuses[lane].requiredConstraints,crowTraceCapacity);'
replacement='''const auto n=std::min<NSUInteger>(statuses[lane].requiredConstraints,crowTraceCapacity);
                            if (crowTraceStep>=348 && crowTraceStep<=375) {
                                std::fprintf(stderr,"CROW_KIN_STATE %llu %lu %lu %lu %lu",crowTraceStep,(unsigned long)sub,(unsigned long)lane,(unsigned long)crowTraceQStride,(unsigned long)crowTraceVStride);
                                for (NSUInteger j=0;j<crowTraceQStride;++j) std::fprintf(stderr," %.9g",startQ[lane*crowTraceQStride+j]);
                                for (NSUInteger j=0;j<crowTraceVStride;++j) std::fprintf(stderr," %.9g",startV[lane*crowTraceVStride+j]);
                                std::fprintf(stderr,"\\n");
                            }'''
replace(needle,replacement)
needle='for(NSUInteger i=0;i<n;++i){const auto& c=contacts[lane*crowTraceCapacity+i];'
replacement='''for(NSUInteger i=0;i<n;++i){const auto& c=contacts[lane*crowTraceCapacity+i];
                                if (crowTraceStep>=348 && crowTraceStep<=375 && ((c.bodyA<35u && c.bodyB==35u)||(c.bodyB<35u && c.bodyA==35u)) && c.impulses.x>0.f) {
                                    std::fprintf(stderr,"CROW_GROUND_CONTACT %llu %lu %lu %u %u %llu %llu",crowTraceStep,(unsigned long)sub,(unsigned long)lane,c.bodyA,c.bodyB,(unsigned long long)c.pairKey,(unsigned long long)c.featureKey);
                                    std::fprintf(stderr," %.9g %.9g %.9g %.9g",c.pointAndSeparation.x,c.pointAndSeparation.y,c.pointAndSeparation.z,c.pointAndSeparation.w);
                                    std::fprintf(stderr," %.9g %.9g %.9g",c.normal.x,c.normal.y,c.normal.z);
                                    std::fprintf(stderr," %.9g %.9g %.9g",c.tangent.x,c.tangent.y,c.tangent.z);
                                    std::fprintf(stderr," %.9g %.9g %.9g %.9g %.9g %.9g\\n",c.friction.x,c.friction.y,c.impulses.x,c.impulses.y,c.impulses.z,c.targetVelocityAndPreSolveNormal.w);
                                }'''
replace(needle,replacement);(B/'MetalWorld.mm').write_text(s)
run=(R/'artifacts/crow-v15-inrange-knee-push-20261113/run.py').read_text();pairs=[("A=R/'artifacts/crow-v15-inrange-knee-push-20261113'","A=R/'artifacts/crow-v15-launch-ground-contact-audit-20261113'"),("B=R/'build/crow-v15-inrange-knee-push-20261113'","B=R/'build/crow-v15-launch-ground-contact-audit-20261113'"),("L=R/'build/crow-v15-leg-work-observer-20261113'","L=R/'build/crow-v15-launch-ground-contact-audit-20261113'"),("str(B/'inrange.metallib')","str(R/'build/crow-v15-inrange-knee-push-20261113/inrange.metallib')")]
for a,b in pairs:assert run.count(a)==1;(run:=run.replace(a,b))
(A/'run.py').write_text(run)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'observer-source.json').write_text(json.dumps({'parent':h(R/'build/crow-v15-leg-work-observer-20261113/MetalWorld.mm'),'candidate':h(B/'MetalWorld.mm')},indent=2)+'\n')
(A/'PLAN.md').write_text('''# Wing-suppressed launch ground-contact audit: pre-run plan

The in-range 160 ms knee candidate was physically valid at the target boundary but failed the prespecified lift and impulse screens on six correlated used-seed lanes. Prior endpoint FK showed the similar out-of-range attempt's foot origins moving, but origin motion is not contact slip. Determine whether the retained model's foot-ground contacts approach friction limits and whether their point-velocity estimate is consistent with the solver's recorded pre-solve normal velocity. This is a diagnostic reproduction, not a change to physics.

Rebuild the previously parity-verified leg-work observer with logging only: during steps 348–375 retain each substep's pre-state q/v and every positive-impulse crow-ground contact's body IDs, pair/feature IDs, world point, contact basis, friction coefficients, normal/tangential impulses, and solver-reported pre-solve normal velocity. Keep v15 body, in-range shader, used seed 20261127, 8 lanes, 400 steps, 16 substeps, and all controller actions unchanged. No held-out seed or layout is consumed. Compare all endpoint state/action/contact and existing substep aggregate records against the already retained in-range run; reject interpretation on mismatch, missing samples, overflows, nonfinite values or observer error.

Use native v15 forward kinematics on recorded pre-state q/v to estimate each body contact-point velocity v_COM + omega cross (point - COM), projecting into the recorded contact basis. Validate endpoint body poses and q against the retained lane-5 replay, and compare both sign conventions for predicted normal contact velocity against the solver's pre-solve value; report residuals and sample counts before interpreting estimated tangential motion. Friction utilization is the magnitude of the two tangential impulses divided by static coefficient times normal impulse, restricted to positive finite normal impulses; it is an observer diagnostic and may depend on solver convention. Report time/lane/body distributions and sample selection. Do not call a near-limit ratio a proven slip or contact velocity when timing, frame, point or sign agreement is poor. Keep all failed records and numerical boundary checks. No biological calibration, launch qualification, flight, navigation or landing conclusion follows from this audit.
''')
print('Prepared logging-only contact observer:',h(B/'MetalWorld.mm'))
