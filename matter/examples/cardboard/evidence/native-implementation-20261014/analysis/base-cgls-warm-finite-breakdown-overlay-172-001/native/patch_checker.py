from pathlib import Path
root = Path('/Users/home/cardboard-implementation-20261014/analysis/base-cgls-warm-finite-breakdown-overlay-172-001/candidate/matter')
checker = root / 'tools/cardboard_base_cgls_warm_start_check.mm'
s = checker.read_text()
old = '''auto pseed=pipeline(d,lib,"nm_fgmres_base_cgls_seed_residual"),padd=pipeline(d,lib,"nm_fgmres_base_cgls_add_initial_solution"),pinstall=pipeline(d,lib,"nm_fgmres_install_base_cgls_direction"),pclear=pipeline(d,lib,"nm_warm_check_clear"),papply=pipeline(d,lib,"nm_warm_check_apply_A");'''
new = '''auto pseed=pipeline(d,lib,"nm_fgmres_base_cgls_seed_residual"),padd=pipeline(d,lib,"nm_fgmres_base_cgls_add_initial_solution"),pvalidateRows=pipeline(d,lib,"nm_fgmres_base_cgls_validate_warm_breakdown_rows"),pmarkBreakdown=pipeline(d,lib,"nm_fgmres_base_cgls_mark_warm_breakdown_decline"),pinstall=pipeline(d,lib,"nm_fgmres_install_base_cgls_direction"),pclear=pipeline(d,lib,"nm_warm_check_clear"),papply=pipeline(d,lib,"nm_warm_check_apply_A");'''
if old not in s: raise SystemExit('pipeline list not found')
s = s.replace(old,new,1)
old = '''run(d,q,padd,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,selectedB,4);bind(e,x0B,5);bind(e,candidateB,6);bind(e,errorsB,7);});'''
new = '''run(d,q,padd,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,selectedB,4);bind(e,x0B,5);bind(e,candidateB,6);bind(e,errorsB,7);bind(e,metricsB,8);});'''
if old not in s: raise SystemExit('normal add initial dispatch not found')
s = s.replace(old,new,1)
needle = '''    std::cout<<"warm-start Metal kernel checker complete; checks="<<checks<<"; metal_kernel_checks_executed=1; physical_transactions=0\\n";return 0;
'''
block = r'''    // A denominator-only breakdown may decline an explicitly enabled warm
    // refinement only after every relevant input and CGLS scratch row is finite.
    std::vector<F4> breakdownMetrics{{.5f,0,0,3}}, finiteScalars{{1,1,1,1}};
    std::vector<U4> breakdownControl{{5u,3u,64u,1u}};
    std::vector<U> breakdownErrors{32u};
    std::vector<F4> breakdownScaleRows(E*N*3u,F4{0,0,0,0});
    for(U n : {0u,2u}) {
        breakdownScaleRows[n*3u+0u]=F4{1,0,0,1};
        breakdownScaleRows[n*3u+1u]=F4{0,1,0,0};
        breakdownScaleRows[n*3u+2u]=F4{0,0,1,0};
    }
    std::vector<F4> breakdownScalePhysical(UCount,F4{0,0,0,0});
    std::vector<F4> breakdownCorrection=correction;
    auto breakdownMetricsB=makeBuffer(d,breakdownMetrics,"finite breakdown metrics");
    auto breakdownScalarsB=makeBuffer(d,finiteScalars,"finite breakdown scalars");
    auto breakdownControlB=makeBuffer(d,breakdownControl,"finite breakdown control");
    auto breakdownErrorsB=makeBuffer(d,breakdownErrors,"finite breakdown errors");
    auto breakdownScaleRowsB=makeBuffer(d,breakdownScaleRows,"finite breakdown scale rows");
    auto breakdownScalePhysicalB=makeBuffer(d,breakdownScalePhysical,"finite breakdown scale vector");
    auto breakdownX B=makeBuffer(d,breakdownCorrection,"finite breakdown x");
'''
# Replace a deliberate token placeholder after insertion for readability below.
if needle not in s: raise SystemExit('checker end insertion point missing')
# Build C++ block without accidental shell substitutions.
block = block.replace('breakdownX B','breakdownXB') + r'''    auto breakdownRB=makeBuffer(d,rhs,"finite breakdown r");
    auto breakdownSB=makeBuffer(d,rhs,"finite breakdown s");
    auto breakdownPB=makeBuffer(d,rhs,"finite breakdown p");
    auto breakdownQB=makeBuffer(d,rhs,"finite breakdown q");
    auto breakdownCandidateB=makeBuffer(d,breakdownCorrection,"finite breakdown candidate");
    auto breakdownSolutionB=makeBuffer(d,x0,"finite breakdown solution x0");
    auto breakdownAxB=makeBuffer(d,std::vector<F4>(UCount,F4{0,0,0,0}),"finite breakdown action");
    std::vector<NMFGMRESStateGPU> breakdownState(1); breakdownState[0].diagnostics={0,0,1,0};
    std::vector<NMMatterStatusGPU> breakdownStatus(1), breakdownTrial(1);
    auto breakdownStateB=makeBuffer(d,breakdownState,"finite breakdown FGMRES state");
    auto breakdownStatusB=makeBuffer(d,breakdownStatus,"finite breakdown status");
    auto breakdownTrialB=makeBuffer(d,breakdownTrial,"finite breakdown trial status");
    auto breakdownLineB=makeBuffer(d,lines0,"finite breakdown line search");
    auto breakdownEnvLineB=makeBuffer(d,envLines0,"finite breakdown environment line");
    auto breakdownTraceB=makeBuffer(d,std::vector<F4>(1,F4{0,0,0,0}),"finite breakdown trace");
    run(d,q,pvalidateRows,UCount,[&](id<MTLComputeCommandEncoder>e){
        bind(e,dispatchB,0); bind(e,breakdownMetricsB,1); bind(e,nodesB,2);
        bind(e,breakdownSolutionB,3); bind(e,rhsB,4); bind(e,residualB,5);
        bind(e,maskB,6); bind(e,envB,7); bind(e,selectedB,8);
        bind(e,breakdownXB,9); bind(e,breakdownRB,10); bind(e,breakdownSB,11);
        bind(e,breakdownPB,12); bind(e,breakdownQB,13);
        bind(e,breakdownScaleRowsB,14); bind(e,breakdownScalePhysicalB,15);
        bind(e,breakdownControlB,16); bind(e,breakdownErrorsB,17);
    });
    run(d,q,pmarkBreakdown,E,[&](id<MTLComputeCommandEncoder>e){
        bind(e,dispatchB,0); bind(e,breakdownMetricsB,1); bind(e,breakdownScalarsB,2);
        bind(e,breakdownControlB,3); bind(e,breakdownErrorsB,4);
        bind(e,breakdownStateB,5); bind(e,breakdownStatusB,6);
    });
    require(static_cast<const F4*>(breakdownMetricsB.contents)[0].w==5.0f &&
        static_cast<const U*>(breakdownErrorsB.contents)[0]==32u,
        "finite exact BadDenominator did not arm warm decline");
    run(d,q,padd,UCount,[&](id<MTLComputeCommandEncoder>e){
        bytes(e,UCount,0); bytes(e,E,1); bind(e,maskB,2); bind(e,envB,3);
        bind(e,selectedB,4); bind(e,breakdownSolutionB,5);
        bind(e,breakdownCandidateB,6); bind(e,breakdownErrorsB,7);
        bind(e,breakdownMetricsB,8);
    });
    require(std::memcmp(breakdownCandidateB.contents,breakdownCorrection.data(),
        UCount*sizeof(F4))==0,"finite breakdown changed private candidate during x0 add");
    U warmMode=1;
    install(d,q,pinstall,dispatchB,layoutB,breakdownMetricsB,nodesB,rhsB,
        breakdownCandidateB,breakdownAxB,breakdownControlB,breakdownErrorsB,
        breakdownSolutionB,breakdownLineB,breakdownEnvLineB,breakdownTraceB,
        breakdownStateB,breakdownStatusB,breakdownTrialB,iter,residualB,warmMode);
    require(static_cast<const NMMatterStatusGPU*>(breakdownStatusB.contents)[0].code==0,
        "finite warm BadDenominator decline failed production status");
    require(static_cast<const NMFGMRESStateGPU*>(breakdownStateB.contents)[0].diagnostics.z==1.0f,
        "finite warm BadDenominator decline cleared original certificate");
    require(std::memcmp(breakdownSolutionB.contents,x0.data(),UCount*sizeof(F4))==0,
        "finite warm BadDenominator decline changed x0");
    require(std::memcmp(breakdownLineB.contents,lines0.data(),sizeof(F4))==0 &&
        std::memcmp(breakdownEnvLineB.contents,envLines0.data(),sizeof(float))==0,
        "finite warm BadDenominator decline changed Armijo state");
    require(static_cast<const F4*>(breakdownMetricsB.contents)[0].w==0.0f &&
        static_cast<const F4*>(breakdownTraceB.contents)[0].w==3.0f,
        "finite warm BadDenominator decline marker/trace mismatch");

    // Nonfinite scratch invalidates the exact error signature; no decline marker
    // may be set and the normal installer must remain fail-closed.
    std::vector<F4> nonfiniteP=rhs; nonfiniteP[0].x=std::numeric_limits<float>::infinity();
    auto nonfinitePB=makeBuffer(d,nonfiniteP,"nonfinite breakdown p");
    std::memcpy(breakdownMetricsB.contents,breakdownMetrics.data(),sizeof(F4));
    std::memcpy(breakdownErrorsB.contents,breakdownErrors.data(),sizeof(U));
    std::memcpy(breakdownControlB.contents,breakdownControl.data(),sizeof(U4));
    run(d,q,pvalidateRows,UCount,[&](id<MTLComputeCommandEncoder>e){
        bind(e,dispatchB,0); bind(e,breakdownMetricsB,1); bind(e,nodesB,2);
        bind(e,breakdownSolutionB,3); bind(e,rhsB,4); bind(e,residualB,5);
        bind(e,maskB,6); bind(e,envB,7); bind(e,selectedB,8);
        bind(e,breakdownXB,9); bind(e,breakdownRB,10); bind(e,breakdownSB,11);
        bind(e,nonfinitePB,12); bind(e,breakdownQB,13);
        bind(e,breakdownScaleRowsB,14); bind(e,breakdownScalePhysicalB,15);
        bind(e,breakdownControlB,16); bind(e,breakdownErrorsB,17);
    });
    run(d,q,pmarkBreakdown,E,[&](id<MTLComputeCommandEncoder>e){
        bind(e,dispatchB,0); bind(e,breakdownMetricsB,1); bind(e,breakdownScalarsB,2);
        bind(e,breakdownControlB,3); bind(e,breakdownErrorsB,4);
        bind(e,breakdownStateB,5); bind(e,breakdownStatusB,6);
    });
    require(static_cast<const F4*>(breakdownMetricsB.contents)[0].w==3.0f &&
        static_cast<const U*>(breakdownErrorsB.contents)[0]==36u,
        "nonfinite CGLS scratch was classified as a safe decline");
    std::memcpy(breakdownErrorsB.contents,breakdownErrors.data(),sizeof(U));
    *static_cast<U*>(breakdownErrorsB.contents)=36u;
    std::memcpy(breakdownMetricsB.contents,breakdownMetrics.data(),sizeof(F4));
    install(d,q,pinstall,dispatchB,layoutB,breakdownMetricsB,nodesB,rhsB,
        breakdownCandidateB,breakdownAxB,breakdownControlB,breakdownErrorsB,
        breakdownSolutionB,breakdownLineB,breakdownEnvLineB,breakdownTraceB,
        breakdownStateB,breakdownStatusB,breakdownTrialB,iter,residualB,warmMode);
    require(static_cast<const NMMatterStatusGPU*>(breakdownStatusB.contents)[0].code!=0,
        "nonfinite/mixed-error warm breakdown was declined as safe");

    // The legacy alternative-direction policy still rejects a CGLS breakdown.
    std::memcpy(breakdownMetricsB.contents,breakdownMetrics.data(),sizeof(F4));
    std::memcpy(breakdownErrorsB.contents,breakdownErrors.data(),sizeof(U));
    std::memcpy(breakdownControlB.contents,breakdownControl.data(),sizeof(U4));
    std::memcpy(breakdownStatusB.contents,breakdownStatus.data(),sizeof(NMMatterStatusGPU));
    std::memcpy(breakdownStateB.contents,breakdownState.data(),sizeof(NMFGMRESStateGPU));
    std::memcpy(breakdownSolutionB.contents,x0.data(),UCount*sizeof(F4));
    U legacyMode=0;
    install(d,q,pinstall,dispatchB,layoutB,breakdownMetricsB,nodesB,rhsB,
        breakdownCandidateB,breakdownAxB,breakdownControlB,breakdownErrorsB,
        breakdownSolutionB,breakdownLineB,breakdownEnvLineB,breakdownTraceB,
        breakdownStateB,breakdownStatusB,breakdownTrialB,iter,residualB,legacyMode);
    require(static_cast<const NMMatterStatusGPU*>(breakdownStatusB.contents)[0].code!=0,
        "legacy CGLS breakdown was incorrectly declined");
    require(std::memcmp(breakdownSolutionB.contents,x0.data(),UCount*sizeof(F4))==0 &&
        static_cast<const NMFGMRESStateGPU*>(breakdownStateB.contents)[0].diagnostics.z==1.0f,
        "legacy CGLS breakdown changed x0 or its certificate");

    std::cout<<"warm-start Metal kernel checker complete; checks="<<checks<<"; metal_kernel_checks_executed=1; physical_transactions=0\\n";return 0;
'''
s = s.replace(needle,block,1)
# Need standard numeric limits include for deterministic nonfinite injection.
if '#include <limits>' not in s:
    s = s.replace('#include <iostream>\n', '#include <iostream>\n#include <limits>\n',1)
checker.write_text(s)

runtime = root / 'src/runtime.mm'
s = runtime.read_text()
old = '''                            line << ",\\\"refinement_installed\\\":"
                                 << (values[e].w > 1.5f ? "true" : "false");
'''
new = '''                            line << ",\\\"refinement_installed\\\":"
                                 << (values[e].w == 2.0f ? "true" : "false");
                            if (values[e].w == 3.0f)
                                line << ",\\\"refinement_decline_reason\\\":\\\"finite_bad_denominator_breakdown\\\",\\\"linear_residual_is_baseline\\\":true";
'''
if old not in s: raise SystemExit('runtime trace boolean output not found')
s = s.replace(old,new,1)
runtime.write_text(s)
print('checker and trace metadata patched')
