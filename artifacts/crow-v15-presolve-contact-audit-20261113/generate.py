from pathlib import Path
import gzip,hashlib,json
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-presolve-contact-audit-20261113';B=R/'build/crow-v15-presolve-contact-audit-20261113';A.mkdir(exist_ok=False);B.mkdir(exist_ok=False);s=(R/'build/crow-v15-launch-ground-contact-audit-20261113/MetalWorld.mm').read_text()
def replace(a,b):
 global s
 assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b)
needle='bool encodeContactSubstep(\n    detail::MetalWorldContextState& context,'
helper='''bool encodeCrowPreContactCapture(detail::MetalWorldContextState& context, id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> destination) {
    if (destination == nil) return true;
    id<MTLBuffer> source = context.buffers[kCandidateV];
    if (source == nil || source.length != destination.length) return false;
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (blit == nil) return false;
    [blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:source.length];
    [blit endEncoding];
    return true;
}

bool encodeContactSubstep(
    detail::MetalWorldContextState& context,'''
replace(needle,helper)
start=s.index('bool encodeContactSubstep(');end=s.index('    MRMetalWorldPassGPU solverPass',start);part=s[start:end];needle='const std::size_t pairFlagThreadCount\n) {';assert part.count(needle)==1;part=part.replace(needle,'const std::size_t pairFlagThreadCount,\n    id<MTLBuffer> crowPreContactCopy\n) {');s=s[:start]+part+s[end:]
start=s.index('bool encodeContactSubstep(');end=s.index('\n}',start)+2;part=s[start:end];needle='''        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.evaluateIRPipeline,
            @"MetalWorld ConstraintIR evaluation",''';assert part.count(needle)==1;part=part.replace(needle,'''        !encodeCrowPreContactCapture(context, commandBuffer, crowPreContactCopy) ||
        !encodeContactThreadKernel(
            context,
            commandBuffer,
            context.evaluateIRPipeline,
            @"MetalWorld ConstraintIR evaluation",''');s=s[:start]+part+s[end:]
needle='''                    const bool encodedABA =
                        (
                            !hasBodyWrenches ||''';replacement='''                    id<MTLBuffer> crowPreContactCopy = nil;
                    if (crowTraceWindow && controlStep == 0u) {
                        if (useHybridContact || !contactMode) throw std::runtime_error("pre-contact observer requires non-hybrid contact");
                        id<MTLBuffer> sourceCandidateV = selectedState->buffers[kCandidateV];
                        crowPreContactCopy = [selectedState->device newBufferWithLength:sourceCandidateV.length options:MTLResourceStorageModeShared];
                        if (crowPreContactCopy == nil) throw std::bad_alloc();
                    }
                    const bool encodedABA =
                        (
                            !hasBodyWrenches ||''';replace(needle,replacement)
start=s.index(': contactMode\n                        ? encodeContactSubstep(');end=s.index('\n                        : encodeCommit(',start);part=s[start:end];needle='''                                  1u
                              )
                          )''';assert part.count(needle)==1;part=part.replace(needle,'''                                  1u
                              ),
                              crowPreContactCopy
                          )''');s=s[:start]+part+s[end:]
replace('[crowTraceBuffers addObject:startVCopy];','[crowTraceBuffers addObject:startVCopy];\n                        [crowTraceBuffers addObject:crowPreContactCopy];')
old='''for (NSUInteger sub=0; sub<crowTraceBuffers.count/8; ++sub) {
                        id<MTLBuffer> cb=crowTraceBuffers[8*sub], sb=crowTraceBuffers[8*sub+1], vb=crowTraceBuffers[8*sub+2], qb=crowTraceBuffers[8*sub+3], tb=crowTraceBuffers[8*sub+4], eb=crowTraceBuffers[8*sub+5], q0b=crowTraceBuffers[8*sub+6], v0b=crowTraceBuffers[8*sub+7];'''
new='''for (NSUInteger sub=0; sub<crowTraceBuffers.count/9; ++sub) {
                        id<MTLBuffer> cb=crowTraceBuffers[9*sub], sb=crowTraceBuffers[9*sub+1], vb=crowTraceBuffers[9*sub+2], qb=crowTraceBuffers[9*sub+3], tb=crowTraceBuffers[9*sub+4], eb=crowTraceBuffers[9*sub+5], q0b=crowTraceBuffers[9*sub+6], v0b=crowTraceBuffers[9*sub+7], vpreb=crowTraceBuffers[9*sub+8];
                        const auto* preContactV=static_cast<const float*>(vpreb.contents);''';replace(old,new)
needle='''                            if (crowTraceStep>=348 && crowTraceStep<=375) {
                                std::fprintf(stderr,"CROW_KIN_STATE'''
replacement='''                            if (crowTraceStep>=348 && crowTraceStep<=375) {
                                std::fprintf(stderr,"CROW_PRESOLVE_V %llu %lu %lu %lu",crowTraceStep,(unsigned long)sub,(unsigned long)lane,(unsigned long)crowTraceVStride);
                                for (NSUInteger j=0;j<crowTraceVStride;++j) std::fprintf(stderr," %.9g",preContactV[lane*crowTraceVStride+j]);
                                std::fprintf(stderr,"\\n");
                                std::fprintf(stderr,"CROW_KIN_STATE''';replace(needle,replacement)
(B/'MetalWorld.mm').write_text(s)
run=(R/'artifacts/crow-v15-launch-ground-contact-audit-20261113/run.py').read_text();pairs=[("A=R/'artifacts/crow-v15-launch-ground-contact-audit-20261113'","A=R/'artifacts/crow-v15-presolve-contact-audit-20261113'"),("B=R/'build/crow-v15-launch-ground-contact-audit-20261113'","B=R/'build/crow-v15-presolve-contact-audit-20261113'"),("L=R/'build/crow-v15-launch-ground-contact-audit-20261113'","L=R/'build/crow-v15-presolve-contact-audit-20261113'")]
for a,b in pairs:assert run.count(a)==1;run=run.replace(a,b)
(A/'run.py').write_text(run)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(A/'observer-source.json').write_text(json.dumps({'parent':h(R/'build/crow-v15-launch-ground-contact-audit-20261113/MetalWorld.mm'),'candidate':h(B/'MetalWorld.mm')},indent=2)+'\n')
(A/'PLAN.md').write_text('''# Predicted pre-contact velocity observer: pre-run diagnostic plan

The previous contact audit reproduced the in-range trial's recorded state and impulse fields. Native FK evaluated at the beginning of the physics substep matched endpoint replay positions and velocities, but its contact-normal-speed estimate disagreed substantially with the solver's pre-solve record (best-sign RMSE 0.0293 m/s versus solver RMS 0.0445 m/s). Do not interpret the old tangential estimate as slip.

The source dispatch sequence evaluates contact rows using `kCandidateV`, after unconstrained dynamics prediction and before contact solve. Add a logging-only Metal blit of that candidate velocity immediately before the ConstraintIR evaluation kernel for the non-hybrid contact path. Record it alongside the original substep-start q/v and existing contact records for steps 348–375. Reject this observer if the run takes the hybrid or noncontact path, if the new copy is absent/nonfinite, if q/state/contact fields differ from the retained in-range and prior observer trials, if any episode terminates early, or if contact overflow/error appears.

Use native v15 FK at substep-start q and the copied pre-constraint candidate velocity to estimate each ground contact point's body velocity and both signs of its normal component. Compare to the stored solver pre-solve normal speed. Prespecified timing screen: the better-sign RMSE must be at most 0.005 m/s and at most 10% of solver normal RMS before treating the tangential contact-point estimate as validated even for this model and window. If the screen fails, report raw impulses and mismatch, not a sliding claim. If it passes, report per-lane time/distribution of estimated tangential speed and dynamic-friction-limit utilization for positive normal impulse contacts, without treating correlated contacts as independent. Dynamic coefficient and modeled force are research-model settings, not measured crow biomechanics. This is a used-seed observation-only replay, not takeoff, flight or mission qualification.
''')
print('Prepared pre-constraint velocity observer:',h(B/'MetalWorld.mm'))
