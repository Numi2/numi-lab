from pathlib import Path
import hashlib
B=Path('/Users/home/cardboard-implementation-20261014');D=B/'analysis/native-base-cauchy-fallback-133';P=D/'source/matter/src/runtime.mm';s=P.read_text();(D/'runtime-before-parent-host-integration.mm').write_text(s)
def one(a,b):
 global s
 assert s.count(a)==1,(a[:90],s.count(a));s=s.replace(a,b)
one('            "nm_fgmres_preserve_true_rhs",','            "nm_fgmres_capture_base_cauchy_rhs",\n            "nm_fgmres_prepare_base_cauchy_action",\n            "nm_fgmres_build_base_cauchy_direction",\n            "nm_fgmres_preserve_true_rhs",')
one('''                        multiplied(static_cast<std::size_t>(environments) *
                            world.dispatch.femNodeCount),''','''                        candidate->fgmresLayout.unknownCount,''')
one('''                    candidate->device, multiplied(environments),''','''                    candidate->device, environments,''') if s.count('                    candidate->device, multiplied(environments),')==1 else None
# The trace allocation is a per-environment vector, not multiplied twice.
one('''                candidate->femBaseCauchyTrace = privateScratch<nm_float4>(
                    candidate->device, multiplied(environments),''','''                candidate->femBaseCauchyTrace = privateScratch<nm_float4>(
                    candidate->device, environments,''') if 'candidate->device, multiplied(environments),' in s[s.index('candidate->femBaseCauchyTrace ='):s.index('candidate->femBaseCauchyTrace =')+200] else None
# Validate the physical opt-in only on its selected control step; other steps
# retain the ordinary production path.
one('''                    if (!femBaseCauchyFallbackConfigValid) {
                        // The parse result is retained only for diagnostics.
                    } else if (!state.femBaseCauchyScratchEnabled ||''','''                    if (request.controlStep != femBaseCauchyFallbackRoot) {
                        // This bounded experimental selector changes one
                        // exact control/Newton state only.
                    } else if (!state.femBaseCauchyScratchEnabled ||''')
# Private sentinel 3 is pending fallback, not an accepted/noop trial.
s=s.replace('metrics[environment].w > 0.5f ? "true" : "false"','(metrics[environment].w > 0.5f && metrics[environment].w < 2.5f) ? "true" : "false"')
s=s.replace('metrics[environment].w > 1.5f ? "true" : "false"','(metrics[environment].w > 1.5f && metrics[environment].w < 2.5f) ? "true" : "false"')
s=s.replace('''const bool selectedAccepted =
                            metrics[environment].w > 0.5f;''','''const bool selectedAccepted =
                            metrics[environment].w > 0.5f &&
                            metrics[environment].w < 2.5f;''')
s=s.replace('''const bool certifiedNoop = metrics[environment].w > 1.5f;''','''const bool certifiedNoop = metrics[environment].w > 1.5f &&
                            metrics[environment].w < 2.5f;''')
# Preserve the base residual before any FGMRES restart overwrites it.
marker='''            if (state.enableVascularDense45) {
                const std::uint32_t restartCycle = 0u;'''
insert='''            const bool baseCauchyNewtonSelected =
                femBaseCauchyFallbackSelectedRoot &&
                nonlinearIteration == femBaseCauchyFallbackIteration;
            if (baseCauchyNewtonSelected) {
                dispatchThreads("nm_fgmres_capture_base_cauchy_rhs",
                    state.fgmresLayout.unknownCount, [&] {
                    setDispatch();
                    [encoder setBuffer:state.femResidual offset:0u atIndex:1u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                    [encoder setBuffer:state.femBaseCauchyRightHandSide
                                 offset:0u atIndex:3u];
                });
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }

'''
one(marker,insert+marker)
# Clone exact production bindings from the already audited true-residual path.
start=s.index('''                dispatchThreads("nm_fem_apply_operator_elements",''',s.index('''const std::uint32_t forceCompletedCacheRebuild'''))
# Prefer the final true-residual copy, which explicitly uses the readonly flag.
starts=[];at=0
while True:
 at=s.find('''                dispatchThreads("nm_fem_apply_operator_elements",''',at)
 if at<0:break
 end=s.index('''                });''',at)+len('''                });''')
 block=s[at:end]
 if '&trueResidualOperatorMicro' in block and 'state.mixedSolverValue.executionBudgets.w == 1u ? 1u : 0u' in block:starts.append((at,end,block))
 at=end
assert starts
op=starts[-1][2]
op=op.replace('trueResidualOperatorMicro','cauchyOperatorMicro').replace('state.femSolution offset:0u atIndex:10u','state.femArmijoDirection offset:0u atIndex:10u').replace('state.femTrueResidualElementOperator','state.femArmijoElementForces').replace('state.femTrueResidualStatuses','state.statuses')
a='''                    const std::uint32_t useCache =
                        state.mixedSolverValue.executionBudgets.w == 1u ? 1u : 0u;'''
assert a in op;op=op.replace(a,'''                    const std::uint32_t useCache = cacheMode;''')
# Matching native gather has a full 2*E*N output; never bind an E*N scratch.
start=s.index('''                dispatchThreads("nm_fgmres_gather_nodes",''',starts[-1][1]);end=s.index('''                });''',start)+len('''                });''');gather=s[start:end];assert '&trueResidualOperatorMicro' in gather
gather=gather.replace('trueResidualOperatorMicro','cauchyOperatorMicro').replace('state.femTrueResidualElementOperator','state.femArmijoElementForces').replace('state.femSolution offset:0u atIndex:10u','state.femArmijoDirection offset:0u atIndex:10u').replace('state.femTrueResidualOperatorValue','cauchyOutput')
helper='''            const auto encodeBaseCauchyDirection = [&]() {
                NMMicrostepGPU cauchyOperatorMicro = micro;
                cauchyOperatorMicro.flags |= NM_MICROSTEP_FGMRES_OPERATOR |
                    NM_MICROSTEP_FGMRES_TRUE_RESIDUAL;
                const auto action = [&](const std::uint32_t cacheMode,
                                         id<MTLBuffer> cauchyOutput) {
                    const std::uint32_t useGradient = cacheMode == 4u ? 1u : 0u;
                    dispatchThreads("nm_fgmres_prepare_base_cauchy_action",
                        femNodeTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:state.femArmijoMetrics offset:0u atIndex:1u];
                        [encoder setBuffer:state.femBaseCauchyRightHandSide offset:0u atIndex:2u];
                        [encoder setBuffer:state.femArmijoResidual offset:0u atIndex:3u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                        [encoder setBuffer:state.femArmijoDirection offset:0u atIndex:5u];
                        [encoder setBytes:&useGradient length:sizeof(useGradient) atIndex:6u];
                    });
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    // Both applications bypass the already-completed linear
                    // latch read-only; the original Newton certificate is
                    // not relabeled as a certificate for this new direction.
OPERATOR
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
GATHER
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                };
                action(3u, state.femArmijoResidual);
                action(4u, state.femOperatorValue);
                dispatchThreads("nm_fgmres_build_base_cauchy_direction",
                    environments, [&] {
                    setDispatch();
                    [encoder setBuffer:state.mixedSolver offset:0u atIndex:1u];
                    [encoder setBuffer:state.femArmijoMetrics offset:0u atIndex:2u];
                    [encoder setBuffer:state.femCandidate offset:0u atIndex:3u];
                    [encoder setBuffer:state.femArmijoResidual offset:0u atIndex:4u];
                    [encoder setBuffer:state.femOperatorValue offset:0u atIndex:5u];
                    [encoder setBuffer:state.femSolution offset:0u atIndex:6u];
                    [encoder setBuffer:state.femLineSearch offset:0u atIndex:7u];
                    [encoder setBuffer:state.environmentLineSearch offset:0u atIndex:8u];
                    [encoder setBuffer:state.femBaseCauchyTrace offset:0u atIndex:9u];
                    [encoder setBuffer:state.fgmresStates offset:0u atIndex:10u];
                    [encoder setBuffer:state.statuses offset:0u atIndex:11u];
                    [encoder setBuffer:state.femArmijoStatuses offset:0u atIndex:12u];
                    [encoder setBytes:&nonlinearIteration length:sizeof(nonlinearIteration) atIndex:13u];
                    [encoder setBuffer:state.femBaseCauchyRightHandSide offset:0u atIndex:14u];
                });
                [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
            };
'''.replace('\nOPERATOR\n','\n'+op+'\n').replace('\nGATHER\n','\n'+gather+'\n')
marker='''            if (femResidualArmijoEnabled &&
                !femCapsuleResidualArmijoEnabled &&
                !femCapsuleBoxResidualArmijoEnabled) {'''
one(marker,helper+marker)
# At most one fallback pass, with the same 24 actual trial updates. No host
# decision is inferred from a status or accepted prefix.
one('''                encodeFEMResidualArmijoBaseStatus();
                for (std::uint32_t trialIndex = 0u;''','''                encodeFEMResidualArmijoBaseStatus();
                const std::uint32_t directionPassCount =
                    baseCauchyNewtonSelected ? 2u : 1u;
                for (std::uint32_t directionPass = 0u;
                     directionPass < directionPassCount; ++directionPass) {
                if (directionPass == 1u) encodeBaseCauchyDirection();
                const std::uint32_t fallbackAllowed =
                    baseCauchyNewtonSelected && directionPass == 0u ? 1u : 0u;
                for (std::uint32_t trialIndex = 0u;''')
# Every call site binds the new argument, zero for all other contact paths.
needle='''                        [encoder setBuffer:state.environmentLineSearch offset:0u atIndex:10u];
                    });'''
# Scope replacements specifically inside the two decide dispatch blocks.
at=0;calls=0
while True:
 at=s.find('"nm_fem_armijo_decide", environments',at)
 if at<0:break
 end=s.index('                    });',at)+len('                    });');block=s[at:end]
 assert needle in block
 binding='''                        [encoder setBytes:&fallbackAllowed length:sizeof(fallbackAllowed) atIndex:11u];'''
 if calls!=0:binding='''                        const std::uint32_t fallbackAllowed = 0u;\n'''+binding
 block=block.replace(needle,needle.replace('\n                    });','\n'+binding+'\n                    });'))
 s=s[:at]+block+s[end:];at+=len(block);calls+=1
assert calls==2,calls
one('''                if (!encodeFEMResidualArmijoTrace()) {
                    ownership->preDynamicsOpen = false;
                    return diagnostics;
                }
            } else {''','''                if (!encodeFEMResidualArmijoTrace()) {
                    ownership->preDynamicsOpen = false;
                    return diagnostics;
                }
                } // Primary Newton and optional same-history Cauchy pass.
            } else {''')
P.write_text(s);(D/'runtime-host-integration.py').write_bytes(Path(__file__).read_bytes());print('runtime',hashlib.sha256(s.encode()).hexdigest())
