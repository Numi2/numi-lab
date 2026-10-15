from pathlib import Path
import difflib
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
S = B / 'snapshots/source-coupon-cache-admission-139'
O = B / 'analysis/native-cgls-runtime-overlay-140-001'
P = 'matter/src/runtime.mm'
assert not O.exists()
original = (S / P).read_text()
text = original

def replace(old, new):
    global text
    assert text.count(old) == 1, old[:180]
    text = text.replace(old, new)

replace('    bool femBaseCauchyScratchEnabled = false;',
        '    bool femBaseCauchyScratchEnabled = false;\n'
        '    bool femBaseCGLSScratchEnabled = false;')
fields = ['X', 'R', 'S', 'P', 'Q', 'Mask', 'Environment', 'Selected',
          'Scalars', 'Control', 'Errors']
replace('    id<MTLBuffer> femBaseCauchyTrace = nil;',
        '    id<MTLBuffer> femBaseCauchyTrace = nil;\n' +
        ''.join(f'    id<MTLBuffer> femBaseCGLS{name} = nil;\n' for name in fields).rstrip())
replace('''        candidate->femBaseCauchyScratchEnabled =
            std::getenv("NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR") != nullptr;''',
'''        candidate->femBaseCauchyScratchEnabled =
            std::getenv("NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR") != nullptr;
        candidate->femBaseCGLSScratchEnabled =
            std::getenv("NM_FEM_BASE_CGLS_ITERATIONS") != nullptr &&
            candidate->femBaseCauchyScratchEnabled &&
            candidate->dispatch.environmentCount <= 2u &&
            candidate->dispatch.femNodeCount <= 4096u &&
            candidate->dispatch.tetrahedronCount <= 32768u;''')
allocation = '''            if (candidate->femBaseCGLSScratchEnabled) {
                // Five vectors plus a mask/map belong to this direction
                // experiment; they never alias accepted physical history.
'''
for name in ['X', 'R', 'S', 'P', 'Q', 'Mask']:
    allocation += f'''                candidate->femBaseCGLS{name} = privateScratch<nm_float4>(
                    candidate->device, candidate->fgmresLayout.unknownCount,
                    valid, candidate->residentBytes);
'''
allocation += '''                candidate->femBaseCGLSEnvironment = privateScratch<std::uint32_t>(
                    candidate->device, candidate->fgmresLayout.unknownCount,
                    valid, candidate->residentBytes);
'''
for name, typ in [('Selected', 'std::uint32_t'), ('Scalars', 'nm_float4'),
                  ('Control', 'nm_uint4'), ('Errors', 'std::uint32_t')]:
    allocation += f'''                candidate->femBaseCGLS{name} = privateScratch<{typ}>(
                    candidate->device, environments, valid, candidate->residentBytes);
'''
allocation += '            }\n'
replace('''            if (world.dispatch.rigidGeneralizedCapacity != 0u) {
                candidate->femArmijoRigidCandidate''', allocation +
'''            if (world.dispatch.rigidGeneralizedCapacity != 0u) {
                candidate->femArmijoRigidCandidate''')
replace('''            femBaseCauchyFallbackRequested = baseCauchySelector != nullptr;
            if (femBaseCauchyFallbackRequested) {''',
'''            femBaseCauchyFallbackRequested = baseCauchySelector != nullptr;
            std::uint32_t femBaseCGLSIterations = 0u;
            if (const char* cglsIterations = std::getenv("NM_FEM_BASE_CGLS_ITERATIONS")) {
                if (!femBaseCauchyFallbackRequested ||
                    (std::strcmp(cglsIterations, "32") != 0 &&
                     std::strcmp(cglsIterations, "64") != 0) ||
                    !state.femBaseCGLSScratchEnabled ||
                    state.fgmresLayout.unknownCount !=
                        2u * state.dispatch.environmentCount * state.dispatch.femNodeCount) {
                    std::fprintf(stderr,
                        "fem_base_cgls_rejected={\\\"schema\\\":1,\\\"control_step\\\":%u,\\\"reason\\\":\\\"requires_selected_cached_pure_fem_fallback_and_exact_32_or_64_iterations\\\"}\\n",
                        request.controlStep);
                    diagnostics.message = "requested bounded base CGLS experiment is unsupported";
                    ownership->preDynamicsOpen = false;
                    return diagnostics;
                }
                femBaseCGLSIterations = std::strcmp(cglsIterations, "32") == 0 ? 32u : 64u;
            }
            if (femBaseCauchyFallbackRequested) {''')
replace('''                        femBaseCauchyFallbackSelectedRoot = true;
                        if (femBaseCauchyFallbackAllNewtonIterations) {''',
'''                        femBaseCauchyFallbackSelectedRoot = true;
                        if (femBaseCGLSIterations != 0u) {
                            std::fprintf(stderr,
                                "fem_base_cgls_enabled={\\\"schema\\\":1,\\\"root\\\":%u,\\\"iterations\\\":%u,\\\"operator_changed\\\":false,\\\"linear_solve_claim\\\":false}\\n",
                                femBaseCauchyFallbackRoot, femBaseCGLSIterations);
                        }
                        if (femBaseCauchyFallbackAllNewtonIterations) {''')
replace('''                const auto action = [&](const std::uint32_t cacheMode,
                                         id<MTLBuffer> cauchyOutput) {
                    const std::uint32_t useGradient = cacheMode == 4u ? 1u : 0u;''',
'''                const auto action = [&](const std::uint32_t cacheMode,
                                         id<MTLBuffer> cauchyInput,
                                         id<MTLBuffer> cauchyOutput) {
                    const std::uint32_t useGradient = 0u;''')
replace('''                        [encoder setBuffer:state.femBaseCauchyRightHandSide offset:0u atIndex:2u];
                        [encoder setBuffer:state.femArmijoResidual offset:0u atIndex:3u];''',
'''                        [encoder setBuffer:cauchyInput offset:0u atIndex:2u];
                        [encoder setBuffer:cauchyInput offset:0u atIndex:3u];''')
replace('''                action(3u, state.femArmijoResidual);
                action(4u, state.femOperatorValue);''',
        (B / 'cgls_runtime_encode.txt').read_text() +
'''                action(3u, state.femBaseCauchyRightHandSide, state.femArmijoResidual);
                action(4u, state.femArmijoResidual, state.femOperatorValue);''')
replace('''                const auto formatFloat = appendTraceFloat;
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    if (completed.status != MTLCommandBufferStatusCompleted) return;
                    const auto* values = static_cast<const nm_float4*>(snapshot.contents);
                    std::ostringstream line;
                    line << "fem_base_cauchy_direction''',
'''                const auto formatFloat = appendTraceFloat;
                const auto capturedCGLSIterations = femBaseCGLSIterations;
                [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                    if (completed.status != MTLCommandBufferStatusCompleted) return;
                    const auto* values = static_cast<const nm_float4*>(snapshot.contents);
                    std::ostringstream line;
                    if (capturedCGLSIterations != 0u) {
                        line << "fem_base_cgls_direction={\\\"schema\\\":1,\\\"root\\\":"
                             << capturedRoot << ",\\\"newton_iteration\\\":" << capturedIteration
                             << ",\\\"direction_kind\\\":\\\"truncated_unregularized_CGLS\\\",\\\"iteration_budget\\\":"
                             << capturedCGLSIterations
                             << ",\\\"original_newton_certificate_applies_to_direction\\\":false,\\\"environments\\\":[";
                    } else line << "fem_base_cauchy_direction''')
replace('''                        line << "{\\\"environment\\\":" << e << ",\\\"gradient_l2\\\":";
                        formatFloat(line, values[e].x);
                        line << ",\\\"gradient_action_l2\\\":"; formatFloat(line, values[e].y);
                        line << ",\\\"tau\\\":"; formatFloat(line, values[e].z);''',
'''                        line << "{\\\"environment\\\":" << e;
                        line << (capturedCGLSIterations != 0u ? ",\\\"candidate_l2\\\":" : ",\\\"gradient_l2\\\":");
                        formatFloat(line, values[e].x);
                        line << (capturedCGLSIterations != 0u ? ",\\\"linear_residual_l2\\\":" : ",\\\"gradient_action_l2\\\":");
                        formatFloat(line, values[e].y);
                        line << (capturedCGLSIterations != 0u ? ",\\\"b_dot_Ap\\\":" : ",\\\"tau\\\":");
                        formatFloat(line, values[e].z);''')

candidate = O / 'candidate' / P
candidate.parent.mkdir(parents=True)
candidate.write_text(text)
(O / 'runtime.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(True), text.splitlines(True), fromfile='a/' + P, tofile='b/' + P)))
helper = (B / 'cgls_runtime_kernels.metalinc').read_bytes()
(O / 'runtime-adapters.metalinc').write_bytes(helper)
(O / 'prepare.py').write_bytes(Path(__file__).read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(O / 'preflight.json').write_text(json.dumps({
    'source': str(S), 'base_runtime_sha256': sha(S / P),
    'candidate_runtime_sha256': sha(candidate),
    'runtime_adapters_sha256': sha(O / 'runtime-adapters.metalinc'),
    'physical_steps': 0, 'status': 'AWAITING_KERNEL_INTEGRATION_AND_BUILD',
    'scope': 'Default-off bounded CGLS candidate generator; original operator, true Newton certificate, nonlinear Armijo and rollback gates are unchanged. New direction clears original convergence latch before trial.'
}, indent=2) + '\n')
print('prepared runtime140')
