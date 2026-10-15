from pathlib import Path
import difflib
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/native-cgls-runtime-overlay-140-001'
P = O / 'candidate/matter/src/runtime.mm'
before = P.read_text()
text = before

def replace(old, new):
    global text
    assert text.count(old) == 1, old[:160]
    text = text.replace(old, new)

replace('''            candidate->dispatch.environmentCount <= 2u &&
            candidate->dispatch.femNodeCount <= 4096u &&
            candidate->dispatch.tetrahedronCount <= 32768u;''',
'''            candidate->dispatch.environmentCount >= 1u &&
            candidate->dispatch.environmentCount <= 2u &&
            candidate->dispatch.femNodeCount >= 1u &&
            candidate->dispatch.femNodeCount <= 4096u &&
            candidate->dispatch.tetrahedronCount <= 32768u &&
            candidate->fgmresLayout.unknownCount == 2u *
                candidate->dispatch.environmentCount * candidate->dispatch.femNodeCount;''')
replace('''            "nm_fgmres_build_base_cauchy_direction",
            "nm_fgmres_preserve_true_rhs",''',
'''            "nm_fgmres_build_base_cauchy_direction",
            "nm_fgmres_prepare_base_cgls_layout",
            "nm_fgmres_base_cgls_initialize_state",
            "nm_fgmres_base_cgls_initialize_vectors",
            "nm_fgmres_base_cgls_measure_initial",
            "nm_fgmres_base_cgls_form_alpha",
            "nm_fgmres_base_cgls_update_xr",
            "nm_fgmres_base_cgls_complete_iteration",
            "nm_fgmres_base_cgls_form_beta",
            "nm_fgmres_base_cgls_update_p",
            "nm_fgmres_base_cgls_finalize",
            "nm_fgmres_install_base_cgls_direction",
            "nm_fgmres_preserve_true_rhs",''')

def block_replace(name, old, new):
    global text
    start = text.index('dispatch' + ('Groups32' if name in ['measure_initial', 'form_alpha', 'form_beta'] else 'Threads') + '("nm_fgmres_base_cgls_' + name + '"')
    end = text.index('}); barrier();', start)
    body = text[start:end]
    assert body.count(old) == 1, (name, old)
    text = text[:start] + body.replace(old, new) + text[end:]

block_replace('measure_initial',
'''[encoder setBuffer:state.femBaseCGLSR offset:0u atIndex:4u];
                        [encoder setBuffer:state.femBaseCGLSS offset:0u atIndex:5u];
                        [encoder setBuffer:state.femBaseCGLSScalars offset:0u atIndex:6u];
                        [encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:7u];
                        [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:8u];''',
'''[encoder setBuffer:state.femBaseCauchyRightHandSide offset:0u atIndex:4u];
                        [encoder setBuffer:state.femBaseCGLSR offset:0u atIndex:5u];
                        [encoder setBuffer:state.femBaseCGLSS offset:0u atIndex:6u];
                        [encoder setBuffer:state.femBaseCGLSScalars offset:0u atIndex:7u];
                        [encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:8u];
                        [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:9u];''')
for name, anchor, slot in [('update_xr', 'R', 10), ('update_p', 'P', 8)]:
    old = f'[encoder setBuffer:state.femBaseCGLS{anchor} offset:0u atIndex:{slot-1}u];'
    block_replace(name, old, old + f'\n                            [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:{slot}u];')
block_replace('complete_iteration',
    '[encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:1u];',
    '[encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:1u];\n'
    '                            [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:2u];')
block_replace('finalize',
'''[encoder setBuffer:state.femBaseCGLSScalars offset:0u atIndex:1u];
                        [encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:2u];
                        [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:3u];''',
'''[encoder setBuffer:state.femBaseCGLSControl offset:0u atIndex:1u];
                        [encoder setBuffer:state.femBaseCGLSErrors offset:0u atIndex:2u];''')

P.write_text(text)
(O / 'abi-and-allocation-fixes.patch').write_text(''.join(difflib.unified_diff(
    before.splitlines(True), text.splitlines(True), fromfile='a/matter/src/runtime.mm', tofile='b/matter/src/runtime.mm')))
(O / 'fix_abi.py').write_bytes(Path(__file__).read_bytes())
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
(O / 'abi-fix-preflight.json').write_text(json.dumps({
    'before_sha256': sha(before), 'after_sha256': sha(text),
    'uncompiled_draft_abi_failures_preserved_in_prior_patch': True,
    'creation_scratch_bound_bytes_upper': 16384 * (6*16 + 4) + 2*(16+16+4+4),
    'physical_steps': 0,
    'status': 'AWAITING_INTEGRATED_BUILD_AND_NATIVE_CHECKS'
}, indent=2) + '\n')
print('fixed ABI140')
