from pathlib import Path
import difflib
import hashlib
import json

B = Path('/Users/home/cardboard-implementation-20261014')
O = B / 'analysis/native-cgls-runtime-overlay-140-001'
P = O / 'candidate/matter/src/runtime.mm'
before = P.read_text()
text = before
old = '"b_dot_Ap'
# Match escaped JSON source; metadata reports the final generated x, not a Krylov p.
assert text.count('b_dot_Ap') == 1
text = text.replace('b_dot_Ap', 'b_dot_Ax')
old = 'directionPass == 1u ? "base_cauchy" : "newton"'
new = 'directionPass == 1u ? (femBaseCGLSIterations != 0u ? "base_cgls" : "base_cauchy") : "newton"'
assert text.count(old) == 1
text = text.replace(old, new)
old = '                        } else std::fprintf(stderr,\n                            "fem_base_cauchy_fallback_enabled='
new = '                        } else if (femBaseCGLSIterations == 0u) std::fprintf(stderr,\n                            "fem_base_cauchy_fallback_enabled='
assert text.count(old) == 1
text = text.replace(old, new)
P.write_text(text)
(O / 'final-metadata.patch').write_text(''.join(difflib.unified_diff(
    before.splitlines(True), text.splitlines(True), fromfile='a/matter/src/runtime.mm', tofile='b/matter/src/runtime.mm')))
(O / 'finalize.py').write_bytes(Path(__file__).read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(O / 'final-runtime-binding.json').write_text(json.dumps(dict(
    candidate_runtime_sha256=sha(P), runtime_adapters_sha256=sha(O / 'runtime-adapters.metalinc'),
    physical_steps=0, status='AWAITING_SOURCE141_BUILD_AND_NATIVE_CHECKS',
    prior_preflight_scope='Each preflight records its own historical draft stage; final-runtime-binding identifies current bytes. Failed draft ABI and quiet-layout assumptions are preserved in their sequential patches.',
    ordered_generators=['prepare.py', 'extend_capture.py', 'fix_abi.py', 'fix_quiet.py', 'finalize.py'],
    ordered_patches=['runtime.patch', 'capture.patch', 'abi-and-allocation-fixes.patch', 'quiet-environment-fixes.patch', 'final-metadata.patch'],
    current_new_scratch_bound_bytes=1900624,
    old_base_rhs_trace_additional_bound_bytes=262176
), indent=2) + '\n')
print('final draft140', sha(P))
