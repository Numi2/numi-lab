"""One-shot, hash-checked integration of the full-equality limit response.

Only the five listed source files may change. Retire after verification.
"""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
hashes = {
    'include/metalrobo/numi_human_stand_gpu.h': ('bdf97db2cb3acb5b9d60ce6626448b3e00e02cfe9d42ed84f17dc716e77f52dd', 'e617e7f8806a666a530d6a8494b1045e9621a2b54358ac712746295a89d67ee1'),
    'src/metal/MetalArticulatedOperator.mm': ('4de21a03f67d8102d4b863c993802468f63659a3c827db24318fd4c8c009bd7a', 'b36e57f3b928309bbc9c3dde52223052c8196a0df0979180cc6b826d03d504cd'),
    'src/metal/NumiHumanStand.metal': ('4e5155b8bfcb836f0d3ddf06ccde75b270ed6cef3fffb727cc3c10ed32967f7f', '91c2e186dde892b8f9fdd925d8133bf6497be745a6ac579a37bd23eb256efdf9'),
    'tests/numi_human_bilateral_metal_test.mm': ('82327dacb812012c09d5d05ed94a4eda774ce1b92bf61d1bc2a1b40cd05c61fb', 'c75148c3e1746d0837629f0bd3a6a16a27199015bf24ceca9b875c5016712683'),
    'tools/check_human_constraint_owner.py': ('2a38d22481180295ee0baaa1c205bfb683d314b4c166d7d4ad5d33a914ece146', 'a435c0999eb3982fef3387878ee77aa0d7b84707ae489d6fac42dd1c09896df9'),
}
for name, (before, _) in hashes.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == before, name

p = root/'src/metal/NumiHumanStand.metal'
s = p.read_text()
s = s.replace('const uint responseStride = responseColumns + equalityCount * (equalityCount + 3u);', 'const uint responseStride = responseColumns + equalityCount * (equalityCount + 3u) +\n        nv * equalityCount;')
s = s.replace('device float* equalityRhs = equalityPivots + equalityCount;', 'device float* equalityRhs = equalityPivots + equalityCount;\n    // Equality multiplier corrections for each conditioned limit response.\n    device float* limitEqualityCorrections = equalityRhs + equalityCount;')
s = s.replace('    uint limitEqualityIndices[MR_NUMI_HUMAN_STAND_MAX_DOFS];\n', '')
start = s.index('        // Strong equality/limit pairs use the same mass-response Schur block.')
end = s.index('        ++limitCount;', start)
s = s[:start] + '''        // Eliminate ALL bilateral rows from this limit's response. A pair
        // correction can satisfy one equality while violating another. These
        // columns use the already factored E M_eff^-1 E^T; no new global
        // solve or artificial compliance is introduced.
        device float* equalityCorrection = limitEqualityCorrections +
            limitCount * equalityCount;
        for (uint ei = 0u; ei < equalityCount; ++ei)
            equalityCorrection[ei] = 0.0f;
        if (equalityCount != 0u) {
            const float rawDiagonal = response[dof];
            // workspace is free after solveFactor; preserve the raw column
            // for unresolved/rank-dependent directions without dropping rows.
            for (uint index = 0u; index < nv; ++index)
                workspace[index] = response[index];
            for (uint refinement = 0u; refinement < 2u; ++refinement) {
                for (uint ei = 0u; ei < equalityCount; ++ei) {
                    device const MRNumiHumanJointEqualityGPU& eq = jointEqualities[ei];
                    float target = 0.0f, derivative = 0.0f, error = 0.0f;
                    if (!evaluateJointEquality(eq, qState, qBase, nq, nv,
                                               target, derivative, error)) {
                        fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, ei);
                        return;
                    }
                    float residual = response[eq.indices.y];
                    if (eq.indices.w != MR_INVALID_INDEX)
                        residual = fma(-derivative, response[eq.indices.w], residual);
                    equalityRhs[ei] = residual;
                }
                if (!mrNumiHumanBilateralSolve(equalityFactor, equalityScale,
                        equalityPivots, equalityRhs, equalityCount)) {
                    fail(status, MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED, dof);
                    return;
                }
                for (uint ei = 0u; ei < equalityCount; ++ei)
                    equalityCorrection[ei] -= equalityRhs[ei];
                for (uint index = 0u; index < nv; ++index) {
                    float correction = 0.0f;
                    for (uint ei = 0u; ei < equalityCount; ++ei) {
                        device const float* er = responseScratch + responseBase +
                            (3u * dispatch.supportContactCount + ei) * nv;
                        correction = fma(equalityRhs[ei], er[index], correction);
                    }
                    response[index] -= correction;
                    if (!isfinite(response[index])) {
                        fail(status, MR_NUMI_HUMAN_STAND_NONFINITE_RESULT, dof);
                        return;
                    }
                }
            }
            // Cancellation in a direction already fixed by E must not be
            // inverted as a new independent limit. Retain the original scalar
            // coupled update for unresolved directions; do not manufacture
            // response with a diagonal floor, omit the row, or loosen gates.
            if (!(response[dof] > 1.0e-6f * rawDiagonal)) {
                for (uint index = 0u; index < nv; ++index)
                    response[index] = workspace[index];
                for (uint ei = 0u; ei < equalityCount; ++ei)
                    equalityCorrection[ei] = 0.0f;
            }
        }
''' + s[end:]
start = s.index('            float nextImpulse = mrNumiHumanProjectIntervalImpulse(')
end = s.index('            const float impulse = nextImpulse - limitAccumulatedImpulses[limit];', start)
s = s[:start] + '''            const float nextImpulse = mrNumiHumanProjectIntervalImpulse(
                limitAccumulatedImpulses[limit], candidateV[dof],
                lowerVelocity, upperVelocity, effectiveMass);
''' + s[end:]
start = s.index('            limitAccumulatedImpulses[limit] = nextImpulse;')
end = s.index('\n        }\n    }', start)
s = s[:start] + '''            limitAccumulatedImpulses[limit] = nextImpulse;
            // Preserve ownership of the compensating bilateral reactions,
            // including negative increments when a limit is released.
            device const float* equalityCorrection = limitEqualityCorrections +
                limit * equalityCount;
            for (uint ei = 0u; ei < equalityCount; ++ei)
                equalityLambdas[ei] = fma(impulse, equalityCorrection[ei], equalityLambdas[ei]);
            for (uint index = 0u; index < nv; ++index)
                candidateV[index] = fma(impulse, response[index], candidateV[index]);''' + s[end:]
s = s.replace('Final equality evidence includes paired limit corrections.', 'Final equality evidence includes full-block limit corrections.')
p.write_text(s)
p = root/'src/metal/MetalArticulatedOperator.mm'
s = p.read_text()
old = '''            // Per-environment bilateral Schur factor, diagonal scaling,
            // pivot indices and correction RHS. Never alias the next arena.
            !checkedAdd(input.stand.jointEqualities.size(), 3u,
                        bilateralScratchElements) ||'''
new = '''            // Per-environment bilateral Schur factor, scaling, pivot/RHS
            // vectors and one equality correction per possible limit column.
            // Include every term before applying the environment stride.
            !checkedAdd(input.stand.jointEqualities.size(), 3u,
                        bilateralScratchElements) ||
            !checkedAdd(bilateralScratchElements, articulation.nv,
                        bilateralScratchElements) ||'''
assert s.count(old) == 1
p.write_text(s.replace(old, new))
p = root/'include/metalrobo/numi_human_stand_gpu.h'
p.write_text(p.read_text().replace('#define MR_NUMI_HUMAN_STAND_ABI_VERSION 10u', '#define MR_NUMI_HUMAN_STAND_ABI_VERSION 11u'))
p = root/'tests/numi_human_bilateral_metal_test.mm'
s = p.read_text()
s = s.replace('bool reverse,bool loaded) {', 'bool reverse,bool loaded,unsigned limitedDof) {')
s = s.replace('envs*((neq+nv)*nv+neq*(neq+3))*sizeof(float)', 'envs*((neq+nv)*nv+neq*(neq+3+nv))*sizeof(float)')
s = s.replace('    const std::array<float,4> inertia=', '''    if (limitedDof != MR_INVALID_INDEX) {
        // Finite-range stop, not a structural lock. A limit on the shared
        // independent coordinate couples to both equality rows. Test a
        // dependent coordinate as well; both environments share the model.
        dofs[limitedDof].flags |= MR_DOF_FLAG_POSITION_LIMIT;
        dofs[limitedDof].limits = {-0.25f, 0.0f, 0.0f, 0.0f};
    }
    const std::array<float,4> inertia=''', 1)
s = s.replace('    d->groundPointAndTimestep={0,0,0,h};', '''    if (limitedDof != MR_INVALID_INDEX)
        d->flags |= MR_NUMI_HUMAN_STAND_ENABLE_CONTACT;
    d->groundPointAndTimestep={0,0,0,h};''', 1)
s = s.replace('        expected[e][2]=c1*expected[e][1];', '''        if (limitedDof != MR_INVALID_INDEX) {
            const double multiplier = limitedDof == 6u ? 1.0 : limitedDof == 7u ? c1 : c2;
            if (multiplier * expected[e][1] > 0.0) {
                // With the independent joint stopped, angular momentum fixes
                // the floating root. This oracle does not use a Schur solve.
                expected[e][0] = r0 / a;
                expected[e][1] = 0.0;
            }
        }
        expected[e][2]=c1*expected[e][1];''', 1)
s = s.replace('<<" got="<<v[e*nv+5+j]', '<<" limitedDof="<<limitedDof<<" got="<<v[e*nv+5+j]')
s = s.replace('        require(status[e].factorAndAssistance.z==0', '''        if (limitedDof != MR_INVALID_INDEX) {
            require(v[e*nv+limitedDof] <= 2e-6, "published finite-stop velocity violated"); ++checks;
            require(status[e].jointEqualityDiagnostics.y < 2e-6,
                    "limit correction invalidated a bilateral row before projection"); ++checks;
        }
        require(status[e].factorAndAssistance.z==0''', 1)
s = s.replace('                checks+=exercise(device,pipeline,queue,h,sweeps,reverse,loaded);', '''                for(unsigned limitedDof:{MR_INVALID_INDEX,6u,7u,8u})
                    checks+=exercise(device,pipeline,queue,h,sweeps,reverse,loaded,limitedDof);''')
p.write_text(s)
p = root/'tools/check_human_constraint_owner.py'
s = p.read_text().replace('"mrNumiHumanProjectIntervalImpulse(", "mrNumiHumanProjectEqualityLimitBlock("', '"mrNumiHumanProjectIntervalImpulse(", "mrNumiHumanBilateralSolve("')
s = s.replace('Final equality evidence includes paired limit corrections.', 'Final equality evidence includes full-block limit corrections.')
s = s.replace('kernel.index("mrNumiHumanProjectEqualityLimitBlock(")', 'kernel.index("equalityLambdas[ei] = fma(impulse,")')
s = s.replace('require("preloadedGeneralizedForce[dof] =" not in runner,', 'require("limitEqualityIndices" not in kernel and "limitEqualityCorrections" in kernel,\n        "limit solve regressed to an isolated equality pair")\nrequire("nv * equalityCount" in kernel,\n        "conditioned limit multipliers are missing from the response stride")\nrequire("preloadedGeneralizedForce[dof] =" not in runner,')
p.write_text(s)
for name, (_, after) in hashes.items():
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual == after, (name, actual, after)
print('Five exact conditioned-limit source files verified')
