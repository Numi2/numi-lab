from pathlib import Path
import difflib
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/native-cgls-runtime-overlay-140-001'
P = O / 'candidate/matter/src/runtime.mm'
K = O / 'runtime-adapters.metalinc'
before, kernel_before = P.read_text(), K.read_text()
text = before
def replace(old, new):
    global text
    assert text.count(old) == 1, old[:100]
    text = text.replace(old, new)

replace('    id<MTLBuffer> femBaseCGLSX = nil;',
        '    id<MTLBuffer> femBaseCGLSB = nil;\n    id<MTLBuffer> femBaseCGLSX = nil;')
replace('''                // Five vectors plus a mask/map belong to this direction
                // experiment; they never alias accepted physical history.''',
'''                // Six vectors plus a mask/map belong to this direction
                // experiment; they never alias accepted physical history.
                candidate->femBaseCGLSB = privateScratch<nm_float4>(
                    candidate->device, candidate->fgmresLayout.unknownCount,
                    valid, candidate->residentBytes);''')
replace('''                        state.femBaseCauchyRightHandSide, state.femBaseCGLSMask,
                        state.femBaseCGLSEnvironment, state.femBaseCGLSScalars,''',
'''                        state.femBaseCGLSB, state.femBaseCGLSMask,
                        state.femBaseCGLSEnvironment, state.femBaseCGLSScalars,''')
start = text.index('                if (femBaseCGLSIterations != 0u) {', text.index('const auto encodeBaseCauchyDirection ='))
end = text.index('                action(3u, state.femBaseCauchyRightHandSide, state.femArmijoResidual);', start)
part = text[start:end]
part = part.replace('''                    const std::uint32_t cglsUnknowns = state.fgmresLayout.unknownCount;''',
'''                    const std::uint32_t cglsUnknowns = state.fgmresLayout.unknownCount;
                    const std::uint32_t cglsEnvironments = state.dispatch.environmentCount;''')
part = part.replace('[encoder setBytes:&environments length:sizeof(environments)',
                    '[encoder setBytes:&cglsEnvironments length:sizeof(cglsEnvironments)')
old = '[encoder setBuffer:state.femBaseCGLSSelected offset:0u atIndex:5u];'
assert part.count(old) == 1
part = part.replace(old, old + '''
                        [encoder setBuffer:state.femBaseCauchyRightHandSide offset:0u atIndex:6u];
                        [encoder setBuffer:state.femBaseCGLSB offset:0u atIndex:7u];''')
part = part.replace('action(3u, state.femBaseCauchyRightHandSide, state.femBaseCGLSS);',
                    'action(3u, state.femBaseCGLSB, state.femBaseCGLSS);')
part = part.replace('[encoder setBuffer:state.femBaseCauchyRightHandSide offset:0u atIndex:5u];',
                    '[encoder setBuffer:state.femBaseCGLSB offset:0u atIndex:5u];')
part = part.replace('[encoder setBuffer:state.femBaseCauchyRightHandSide offset:0u atIndex:4u];',
                    '[encoder setBuffer:state.femBaseCGLSB offset:0u atIndex:4u];')
text = text[:start] + part + text[end:]
P.write_text(text)

kernel = kernel_before
old = '''    device uint* selected [[buffer(5)]],
    uint global [[thread_position_in_grid]])'''
new = '''    device uint* selected [[buffer(5)]],
    device const float4* baseRHS [[buffer(6)]],
    device float4* preparedRHS [[buffer(7)]],
    uint global [[thread_position_in_grid]])'''
assert kernel.count(old) == 1
kernel = kernel.replace(old, new)
old = '''    float4 value = float4(0.0f);
    if (global < count) {'''
new = '''    const bool active = metrics[environment].w == 3.0f;
    preparedRHS[global] = active ? baseRHS[global] : float4(0.0f);
    float4 value = float4(0.0f);
    if (global < count) {'''
assert kernel.count(old) == 1
kernel = kernel.replace(old, new).replace('if (free) value = float4(', 'if (active && free) value = float4(')
K.write_text(kernel)
patch = ''.join(difflib.unified_diff(before.splitlines(True), text.splitlines(True),
    fromfile='a/matter/src/runtime.mm', tofile='b/matter/src/runtime.mm'))
patch += ''.join(difflib.unified_diff(kernel_before.splitlines(True), kernel.splitlines(True),
    fromfile='a/runtime-adapters.metalinc', tofile='b/runtime-adapters.metalinc'))
(O / 'quiet-environment-fixes.patch').write_text(patch)
(O / 'fix_quiet.py').write_bytes(Path(__file__).read_bytes())
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
(O / 'quiet-fix-preflight.json').write_text(json.dumps({
    'runtime_before_sha256': sha(before), 'runtime_after_sha256': sha(text),
    'adapter_before_sha256': sha(kernel_before), 'adapter_after_sha256': sha(kernel),
    'scratch_maximum_bytes': 16384 * (7*16 + 4) + 2*(16+16+4+4),
    'scope': 'Prepare private b and mask zero for nonselected environments; selected b retains all input field/w bits for fail-closed validation. Original base RHS, statuses and corrections are untouched.',
    'physical_steps': 0, 'status': 'AWAITING_INTEGRATED_BUILD_AND_NATIVE_CHECKS'
}, indent=2) + '\n')
print('quiet layout140 fixed')
