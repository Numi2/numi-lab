"""Apply the reviewed candidate edit; removed before native-branch promotion."""
import hashlib
from pathlib import Path


def once(source, before, after):
    assert source.count(before) == 1, (before, source.count(before))
    return source.replace(before, after, 1)


p = Path('matter/src/metal/fgmres.metalinc')
raw = p.read_bytes()
assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == '1a604502a3f8876f23c59467d978b7a95db9db0b'
s = raw.decode()
s = once(s, 'namespace numi_matter_metal {', '#include "numi/matter/support_feasibility.hpp"\n\nnamespace numi_matter_metal {')
s = once(s, '    device const float* rigidCandidate [[buffer(18)]],\n    uint lane', '    device const float* rigidCandidate [[buffer(18)]],\n    device const float4* supportHistories [[buffer(19)]],\n    uint lane')
s = once(s, '    float maximumCandidateSpeedSquared = 0.0f;\n    const bool active', '    float maximumCandidateSpeedSquared = 0.0f;\n    uint invalidSupportImpulse = 0u;\n    const bool active')
before = ('        for (uint local = lane; local < layout.supportContactCount;\n'
          '             local += 32u)\n'
          '            fgmresCompensatedAdd(\n'
          '                norm2, normCorrection,\n'
          '                dot(residual[supportBase + local], residual[supportBase + local]));')
after = ('        for (uint local = lane; local < layout.supportContactCount;\n'
         '             local += 32u) {\n'
         '            fgmresCompensatedAdd(\n'
         '                norm2, normCorrection,\n'
         '                dot(residual[supportBase + local], residual[supportBase + local]));\n'
         '            const float normalImpulse = supportHistories[\n'
         '                environment * layout.supportContactCount + local].w;\n'
         '            invalidSupportImpulse |=\n'
         '                numi_matter_contact::normalImpulseFeasible(normalImpulse) ? 0u : 1u;\n'
         '        }')
s = once(s, before, after)
s = once(s, '    maximumCandidateSpeedSquared = simd_max(maximumCandidateSpeedSquared);\n', '    maximumCandidateSpeedSquared = simd_max(maximumCandidateSpeedSquared);\n    const bool supportFeasible = simd_max(invalidSupportImpulse) == 0u;\n')
s = once(s, '    const bool nonlinearConverged = restartCycle == 0u &&\n        (correctionFinite || contactFreeStaticRest) &&\n        norm <= nonlinearThreshold;', '    // Residual convergence cannot bypass the unchanged support feasibility gate.\n    // Continue the coupled correction rather than clamping a dual variable.\n    const bool nonlinearConverged = restartCycle == 0u &&\n        (correctionFinite || contactFreeStaticRest) && supportFeasible &&\n        norm <= nonlinearThreshold;')
s = once(s, '    bool invalidSupportImpulse = false;\n    if (lane == 0u) {', '    bool invalidSupportImpulse = false;\n    uint invalidSupportRow = NM_INVALID_INDEX;\n    float invalidNormalImpulse = 0.0f;\n    if (lane == 0u) {')
s = once(s, '            if (!isfinite(normalImpulse) || normalImpulse < -1.0e-7f) {\n                invalidSupportImpulse = true;\n                break;\n            }', '            if (!numi_matter_contact::normalImpulseFeasible(normalImpulse)) {\n                invalidSupportImpulse = true;\n                invalidSupportRow = row;\n                invalidNormalImpulse = normalImpulse;\n                break;\n            }')
s = once(s, '            NM_INVALID_INDEX, NM_INVALID_INDEX, float4(-1.0f, 0.0f, supportNorm, rigidNorm));', '            NM_INVALID_INDEX, invalidSupportRow,\n            float4(-1.0f, invalidNormalImpulse, supportNorm, rigidNorm));')
p.write_text(s)
p = Path('matter/src/runtime.mm')
s = p.read_text()
s = once(s, '                [encoder setBuffer:state.coupledGeneralizedCandidate\n                             offset:0u atIndex:18u];\n            });\n            dispatchThreads("nm_fgmres_build_preconditioner"', '                [encoder setBuffer:state.coupledGeneralizedCandidate\n                             offset:0u atIndex:18u];\n                [encoder setBuffer:state.humanSupportHistoriesCandidate\n                             offset:0u atIndex:19u];\n            });\n            dispatchThreads("nm_fgmres_build_preconditioner"')
p.write_text(s)
p = Path('matter/CMakeLists.txt')
s = p.read_text()
p.write_text(once(s, '            include/numi/matter/shared.h\n', '            include/numi/matter/shared.h\n            include/numi/matter/support_feasibility.hpp\n'))
Path('support-evidence/edit-applied').write_text('candidate only; not root-104 qualification\n')
