from pathlib import Path
root = Path('/Users/home/cardboard-implementation-20261014/analysis/base-cgls-warm-finite-breakdown-overlay-172-001/candidate')
base = Path('/Users/home/cardboard-implementation-20261014/snapshots/source-warm-cgls-refinement-169')

runtime = root / 'matter/src/runtime.mm'
s = runtime.read_text()
needle = '''                    dispatchThreads("nm_fgmres_base_cgls_finalize", environments, [&] {
                        [encoder setBytes:&cglsEnvironments length:sizeof(cglsEnvironments) atIndex:0u];
                        [encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:1u];
                        [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:2u];
                    }); barrier();
                    if (femBaseCGLSRightScaleEnabled)
'''
replacement = '''                    dispatchThreads("nm_fgmres_base_cgls_finalize", environments, [&] {
                        [encoder setBytes:&cglsEnvironments length:sizeof(cglsEnvironments) atIndex:0u];
                        [encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:1u];
                        [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:2u];
                    }); barrier();
                    if (warmStartRefinementThisNewton) {
                        // finalize maps nonzero error bits to InvalidInput. Only
                        // an exact BadDenominator with finite private CGLS state
                        // may decline this optional refinement and preserve x0.
                        dispatchThreads(
                            "nm_fgmres_base_cgls_validate_warm_breakdown_rows",
                            cglsUnknowns, [&] {
                            setDispatch();
                            [encoder setBuffer:state.femArmijoMetrics offset:0u atIndex:1u];
                            [encoder setBuffer:state.femCandidate offset:0u atIndex:2u];
                            [encoder setBuffer:state.femSolution offset:0u atIndex:3u];
                            [encoder setBuffer:state.femBaseCauchyRightHandSide offset:0u atIndex:4u];
                            [encoder setBuffer:state.femBaseCGLSB offset:0u atIndex:5u];
                            [encoder setBuffer:state.femBaseCGLSMask offset:0u atIndex:6u];
                            [encoder setBuffer:state.femBaseCGLSEnvironment offset:0u atIndex:7u];
                            [encoder setBuffer:state.femBaseCGLSSelected offset:0u atIndex:8u];
                            [encoder setBuffer:state.femBaseCGLSX offset:0u atIndex:9u];
                            [encoder setBuffer:state.femBaseCGLSR offset:0u atIndex:10u];
                            [encoder setBuffer:state.femBaseCGLSS offset:0u atIndex:11u];
                            [encoder setBuffer:state.femBaseCGLSP offset:0u atIndex:12u];
                            [encoder setBuffer:state.femBaseCGLSQ offset:0u atIndex:13u];
                            [encoder setBuffer:state.femBaseCGLSRightScaleRows offset:0u atIndex:14u];
                            [encoder setBuffer:state.femBaseCGLSRightScalePhysical offset:0u atIndex:15u];
                            [encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:16u];
                            [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:17u];
                        }); barrier();
                        dispatchThreads(
                            "nm_fgmres_base_cgls_mark_warm_breakdown_decline",
                            environments, [&] {
                            setDispatch();
                            [encoder setBuffer:state.femArmijoMetrics offset:0u atIndex:1u];
                            [encoder setBuffer:state.femBaseCGLSScalars offset:0u atIndex:2u];
                            [encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:3u];
                            [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:4u];
                            [encoder setBuffer:state.fgmresStates offset:0u atIndex:5u];
                            [encoder setBuffer:state.statuses offset:0u atIndex:6u];
                        }); barrier();
                    }
                    if (femBaseCGLSRightScaleEnabled)
'''
if needle not in s:
    raise SystemExit('runtime insertion point not found')
runtime.write_text(s.replace(needle, replacement, 1))

metal = root / 'matter/src/metal/base_cgls_runtime.metalinc'
s = metal.read_text()
needle = '''// Installing x is separate from its construction: the old linear certificate
// certifies delta only. No CGLS status is a production convergence certificate.
kernel void nm_fgmres_install_base_cgls_direction(
'''
insert = '''// A denominator-only CGLS breakdown is a normal decline of the opt-in warm
// refinement only when every private row and scalar is still finite. The row
// scan is parallel; invalid data adds a distinct error bit so the final
// one-thread-per-environment classifier cannot mark the decline safe.
kernel void nm_fgmres_base_cgls_validate_warm_breakdown_rows(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    device float4* metrics [[buffer(1)]],
    device const NMFEMNodeStateGPU* candidate [[buffer(2)]],
    device const float4* initialSolution [[buffer(3)]],
    device const float4* baseRHS [[buffer(4)]],
    device const float4* baselineResidual [[buffer(5)]],
    device const float4* rowMask [[buffer(6)]],
    device const uint* rowEnvironment [[buffer(7)]],
    device const uint* selected [[buffer(8)]],
    device const float4* x [[buffer(9)]],
    device const float4* r [[buffer(10)]],
    device const float4* s [[buffer(11)]],
    device const float4* p [[buffer(12)]],
    device const float4* q [[buffer(13)]],
    device const float4* rightScaleRows [[buffer(14)]],
    device const float4* rightScalePhysical [[buffer(15)]],
    device const uint4* control [[buffer(16)]],
    device atomic_uint* errors [[buffer(17)]],
    uint index [[thread_position_in_grid]]) {
    const uint environments = dispatch.environmentCount;
    const uint nodes = dispatch.femNodeCount;
    const uint mechanicalRows = environments * nodes;
    const uint unknownCount = 2u * mechanicalRows;
    if (index >= unknownCount || environments == 0u || nodes == 0u) return;
    const bool fieldRow = index >= mechanicalRows;
    const uint local = fieldRow ? index - mechanicalRows : index;
    const uint environment = local / nodes;
    if (environment >= environments || metrics[environment].w != 3.0f ||
        selected[environment] != 1u || control[environment].x != 5u ||
        control[environment].w != 1u ||
        atomic_load_explicit(errors + environment, memory_order_relaxed) !=
            kBaseCGLSErrorBadDenominator) return;
    const uint expected = environment;
    if (rowEnvironment[index] != expected) {
        baseCGLSMarkError(errors, environment, kBaseCGLSErrorBadShape);
        return;
    }
    const float4 mask = rowMask[index];
    const float4 rhs = baseRHS[index];
    const float4 residual = baselineResidual[index];
    const float4 xi = x[index], ri = r[index], si = s[index];
    const float4 pi = p[index], qi = q[index];
    const float4 physical = rightScalePhysical[index];
    bool finiteRows = baseCGLSFinite4(mask) && baseCGLSFinite4(rhs) &&
        baseCGLSFinite4(residual) && baseCGLSFinite4(xi) &&
        baseCGLSFinite4(ri) && baseCGLSFinite4(si) &&
        baseCGLSFinite4(pi) && baseCGLSFinite4(qi) &&
        baseCGLSFinite4(physical) && mask.w == 0.0f && rhs.w == 0.0f &&
        residual.w == 0.0f && xi.w == 0.0f && ri.w == 0.0f &&
        si.w == 0.0f && pi.w == 0.0f && qi.w == 0.0f && physical.w == 0.0f;
    if (!finiteRows) {
        baseCGLSMarkError(errors, environment, kBaseCGLSErrorNonfinite);
        return;
    }
    uint rowError = 0u;
    if (!baseCGLSMaskComponent(mask.x) || !baseCGLSMaskComponent(mask.y) ||
        !baseCGLSMaskComponent(mask.z)) rowError |= kBaseCGLSErrorBadMask;
    if (fieldRow) {
        if (!baseCGLSZero4(mask) || !baseCGLSZero4(rhs) ||
            !baseCGLSZero4(residual) || !baseCGLSZero4(xi) ||
            !baseCGLSZero4(ri) || !baseCGLSZero4(si) ||
            !baseCGLSZero4(pi) || !baseCGLSZero4(qi) ||
            !baseCGLSZero4(physical))
            rowError |= kBaseCGLSErrorNonzeroConstrainedRow;
    } else {
        const NMFEMNodeStateGPU node = candidate[index];
        const bool finiteNode = baseCGLSFinite4(node.positionAndMass) &&
            baseCGLSFinite4(node.velocityAndInverseMass) &&
            baseCGLSFinite4(node.restAndFixed) &&
            baseCGLSFinite4(node.deltaVelocity) &&
            baseCGLSFinite4(node.referenceDisplacementAndMode);
        if (!finiteNode) rowError |= kBaseCGLSErrorNonfinite;
        if (any((mask.xyz == float3(0.0f)) &
                ((rhs.xyz != float3(0.0f)) |
                 (residual.xyz != float3(0.0f)) |
                 (xi.xyz != float3(0.0f)) | (ri.xyz != float3(0.0f)) |
                 (si.xyz != float3(0.0f)) | (pi.xyz != float3(0.0f)) |
                 (qi.xyz != float3(0.0f)) |
                 (physical.xyz != float3(0.0f)))))
            rowError |= kBaseCGLSErrorNonzeroConstrainedRow;
        const uint block = index * 3u;
        const float4 row0 = rightScaleRows[block];
        const float4 row1 = rightScaleRows[block + 1u];
        const float4 row2 = rightScaleRows[block + 2u];
        if (!baseCGLSFinite4(row0) || !baseCGLSFinite4(row1) ||
            !baseCGLSFinite4(row2)) rowError |= kBaseCGLSErrorNonfinite;
        if (any(mask.xyz == float3(1.0f))) {
            if (row0.w != 1.0f || row1.w != 0.0f || row2.w != 0.0f)
                rowError |= kBaseCGLSErrorInvalidRightScale;
        } else if (any(row0 != float4(0.0f)) ||
                   any(row1 != float4(0.0f)) ||
                   any(row2 != float4(0.0f))) {
            rowError |= kBaseCGLSErrorInvalidRightScale;
        }
    }
    if (rowError != 0u) baseCGLSMarkError(errors, environment, rowError);
}

// The finalizer converts any CGLS error to status 5. This exact signature is
// the only one which can decline the optional warm solve: status 5, selected,
// exact BadDenominator only, original certificate still present, and all scans
// above successful. Errors or status changes never authorize this path.
kernel void nm_fgmres_base_cgls_mark_warm_breakdown_decline(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    device float4* metrics [[buffer(1)]],
    device const float4* scalars [[buffer(2)]],
    device const uint4* control [[buffer(3)]],
    device const atomic_uint* errors [[buffer(4)]],
    device const NMFGMRESStateGPU* states [[buffer(5)]],
    device const NMMatterStatusGPU* statuses [[buffer(6)]],
    uint environment [[thread_position_in_grid]]) {
    if (environment >= dispatch.environmentCount ||
        metrics[environment].w != 3.0f) return;
    const uint4 c = control[environment];
    const float4 scalar = scalars[environment];
    if (c.x == 5u && c.w == 1u && c.y <= c.z &&
        (c.z == 32u || c.z == 64u) &&
        atomic_load_explicit(errors + environment, memory_order_relaxed) ==
            kBaseCGLSErrorBadDenominator && baseCGLSFinite4(scalar) &&
        isfinite(metrics[environment].x) && metrics[environment].x > 0.0f &&
        states[environment].diagnostics.z > 0.5f &&
        environmentHealthy(statuses, environment)) {
        metrics[environment].w = 5.0f;
    }
}

// Installing x is separate from its construction: the old linear certificate
// certifies delta only. No CGLS status is a production convergence certificate.
kernel void nm_fgmres_install_base_cgls_direction(
'''
if needle not in s:
    raise SystemExit('Metal kernel insertion point not found')
s = s.replace(needle, insert, 1)
# Add metrics input to add_initial_solution and skip mutation for safe decline.
needle = '''    device float4* candidateSolution [[buffer(6)]],
    device atomic_uint* errors [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
'''
replacement = '''    device float4* candidateSolution [[buffer(6)]],
    device atomic_uint* errors [[buffer(7)]],
    device const float4* metrics [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
'''
if needle not in s:
    raise SystemExit('add initial signature not found')
s = s.replace(needle, replacement, 1)
needle = '''    const uint selectedValue = selected[environment];
    if (selectedValue > 1u) {
        baseCGLSMarkError(errors, environment, kBaseCGLSErrorBadSelection);
'''
replacement = '''    // Preserve the unfinished private correction on the exact warm-only
    // finite-breakdown decline; x0 remains the production solution.
    if (metrics[environment].w == 5.0f) return;
    const uint selectedValue = selected[environment];
    if (selectedValue > 1u) {
        baseCGLSMarkError(errors, environment, kBaseCGLSErrorBadSelection);
'''
if needle not in s:
    raise SystemExit('add initial selected block not found')
s = s.replace(needle, replacement, 1)
# Add warm decline handling before ordinary unselected marker handling.
needle = '''    if (environment >= dispatch.environmentCount) return;
    float4 metric = metrics[environment];
    if (metric.w != 3.0f) {
        trace[environment] = float4(0.0f);
        return;
    }
    const uint4 c = control[environment];
'''
replacement = '''    if (environment >= dispatch.environmentCount) return;
    float4 metric = metrics[environment];
    const bool refinementMode = requireResidualImprovement == 1u;
    if (metric.w == 5.0f) {
        const uint4 breakdownControl = control[environment];
        const uint breakdownErrors = atomic_load_explicit(
            errors + environment, memory_order_relaxed);
        bool finiteDecline = refinementMode &&
            breakdownControl.x == 5u && breakdownControl.w == 1u &&
            breakdownControl.y <= breakdownControl.z &&
            breakdownErrors == kBaseCGLSErrorBadDenominator &&
            environmentHealthy(statuses, environment) &&
            states[environment].diagnostics.z > 0.5f;
        const uint base = environment * dispatch.femNodeCount;
        const uint fieldBase = dispatch.environmentCount * dispatch.femNodeCount;
        float initialResidual2 = 0.0f, initialCorrection = 0.0f;
        for (uint node = 0u; node < dispatch.femNodeCount; ++node) {
            const uint global = base + node;
            const uint field = fieldBase + global;
            const float4 initial = solution[global];
            const float4 b = baseRHS[global];
            const float4 a = ax[global];
            const float4 residual = baselineResidual[global];
            const float4 partial = x[global];
            finiteDecline = finiteDecline && all(isfinite(initial)) &&
                all(isfinite(b)) && all(isfinite(a)) &&
                all(isfinite(residual)) && all(isfinite(partial)) &&
                initial.w == 0.0f && b.w == 0.0f && a.w == 0.0f &&
                residual.w == 0.0f && partial.w == 0.0f &&
                all(isfinite(baseRHS[field])) &&
                all(isfinite(baselineResidual[field])) &&
                all(isfinite(x[field])) && all(isfinite(ax[field])) &&
                all(baseRHS[field] == float4(0.0f)) &&
                all(baselineResidual[field] == float4(0.0f)) &&
                all(x[field] == float4(0.0f)) &&
                all(ax[field] == float4(0.0f));
            const NMFEMNodeStateGPU n = candidate[global];
            const bool free = !femStaticConstraint(n.restAndFixed.w) &&
                isfinite(n.positionAndMass.w) && n.positionAndMass.w > 0.0f;
            if (free) {
                fgmresCompensatedAdd(initialResidual2, initialCorrection,
                    dot(residual.xyz, residual.xyz));
            } else {
                finiteDecline = finiteDecline && all(initial == float4(0.0f)) &&
                    all(b == float4(0.0f)) && all(residual == float4(0.0f)) &&
                    all(partial == float4(0.0f));
            }
        }
        finiteDecline = finiteDecline && isfinite(initialResidual2) &&
            initialResidual2 >= 0.0f;
        if (finiteDecline) {
            trace[environment] = float4(0.0f, sqrt(initialResidual2), 0.0f, 3.0f);
            metric.w = 0.0f;
            metrics[environment] = metric;
            return;
        }
        // A later nonfinite operator action or a new error bit invalidates the
        // pre-action classification. Restore the ordinary fail-closed path.
        metric.w = 3.0f;
        metrics[environment] = metric;
    }
    if (metric.w != 3.0f) {
        trace[environment] = float4(0.0f);
        return;
    }
    const uint4 c = control[environment];
'''
if needle not in s:
    raise SystemExit('installer marker insertion not found')
s = s.replace(needle, replacement, 1)
# Remove duplicate refinementMode declaration later.
needle = '''    float baselineResidual2 = 0.0f, brc = 0.0f;
    const bool refinementMode = requireResidualImprovement == 1u;
    for (uint node = 0u; node < dispatch.femNodeCount; ++node) {
'''
replacement = '''    float baselineResidual2 = 0.0f, brc = 0.0f;
    for (uint node = 0u; node < dispatch.femNodeCount; ++node) {
'''
if needle not in s:
    raise SystemExit('duplicate refinementMode not found')
s = s.replace(needle, replacement, 1)
metal.write_text(s)

# Update checker tool copies: add pipeline/load, slot 8 binding, tests below.
checker = root / 'matter/tools/cardboard_base_cgls_warm_start_check.mm'
s = checker.read_text()
old = '''auto pseed=pipeline(d,lib,"nm_fgmres_base_cgls_seed_residual"),padd=pipeline(d,lib,"nm_fgmres_base_cgls_add_initial_solution"),pinstall=pipeline(d,lib,"nm_fgmres_install_base_cgls_direction"),pclear=pipeline(d,lib,"nm_warm_check_clear"),papply=pipeline(d,lib,"nm_warm_check_apply_A");'''
new = '''auto pseed=pipeline(d,lib,"nm_fgmres_base_cgls_seed_residual"),padd=pipeline(d,lib,"nm_fgmres_base_cgls_add_initial_solution"),pvalidateRows=pipeline(d,lib,"nm_fgmres_base_cgls_validate_warm_breakdown_rows"),pmarkBreakdown=pipeline(d,lib,"nm_fgmres_base_cgls_mark_warm_breakdown_decline"),pinstall=pipeline(d,lib,"nm_fgmres_install_base_cgls_direction"),pclear=pipeline(d,lib,"nm_warm_check_clear"),papply=pipeline(d,lib,"nm_warm_check_apply_A");'''
if old not in s: raise SystemExit('checker pipeline list not found')
s = s.replace(old,new,1)
# Add optional metrics parameter to C++ add-solution dispatch helper by extending local call only.
old = '''run(d,q,padd,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,selectedB,4);bind(e,x0B,5);bind(e,candidateB,6);bind(e,errorsB,7);});'''
new = '''run(d,q,padd,UCount,[&](id<MTLComputeCommandEncoder>e){bytes(e,UCount,0);bytes(e,E,1);bind(e,maskB,2);bind(e,envB,3);bind(e,selectedB,4);bind(e,x0B,5);bind(e,candidateB,6);bind(e,errorsB,7);bind(e,metricsB,8);});'''
if old not in s: raise SystemExit('checker add-solution dispatch not found')
s = s.replace(old,new,1)
checker.write_text(s)
print('candidate source patches written')
