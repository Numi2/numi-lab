"""One-shot, source-checked integration of the existing Human equality block."""
from pathlib import Path
import hashlib

root=Path(__file__).resolve().parents[1]
p=root/'src/metal/NumiHumanStand.metal';s=p.read_text()
s=s.replace('#include "metalrobo/numi_human_friction.h"','#include "metalrobo/numi_human_friction.h"\n#include "metalrobo/numi_human_bilateral.h"',1)
old='''    const uint responseBase = environment *
        (dispatch.supportContactCount * 3u + dispatch.jointEqualityCount + nv) * nv;'''
new='''    const uint equalityCount = dispatch.jointEqualityCount;
    const uint responseColumns = (dispatch.supportContactCount * 3u + equalityCount + nv) * nv;
    const uint responseStride = responseColumns + equalityCount * (equalityCount + 3u);
    const uint responseBase = environment * responseStride;
    device float* equalityFactor = responseScratch + responseBase + responseColumns;
    device float* equalityScale = equalityFactor + equalityCount * equalityCount;
    device float* equalityPivots = equalityScale + equalityCount;
    device float* equalityRhs = equalityPivots + equalityCount;'''
assert s.count(old)==1;s=s.replace(old,new)
start=s.index('        }\n\n        for (uint iteration = 0u;',s.index('    // Equality rows use'))
end=s.index('\n    if ((dispatch.flags & MR_NUMI_HUMAN_STAND_ENABLE_CONTACT)',start)
s=s[:start]+'''        // Solve all bilateral rows together, not as scalar Gauss-Seidel
        // updates which can undo each other. Keep the actual nonsymmetric
        // FP32 contractions instead of silently adding diagonal compliance.
        for (uint row=0u; row<equalityCount; ++row) {
            device const MRNumiHumanJointEqualityGPU& equality=jointEqualities[row];
            float target=0.0f, derivative=0.0f, error=0.0f;
            if (!evaluateJointEquality(equality,qState,qBase,nq,nv,target,derivative,error)) {
                fail(status,MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,row);
                return;
            }
            for (uint column=0u; column<equalityCount; ++column) {
                device const float* response=responseScratch+responseBase+
                    (3u*dispatch.supportContactCount+column)*nv;
                float value=response[equality.indices.y];
                if (equality.indices.w!=MR_INVALID_INDEX)
                    value=fma(-derivative,response[equality.indices.w],value);
                equalityFactor[row*equalityCount+column]=value;
            }
        }
        if (!mrNumiHumanBilateralFactor(equalityFactor,equalityScale,equalityPivots,equalityCount)) {
            fail(status,MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,MR_INVALID_INDEX);
            return;
        }
        }

        // Recompute the actual velocity residual for a refinement correction.
        // This is still the same E M_eff^-1 E^T block and impulse ownership.
        for (uint refinement=0u; refinement<2u; ++refinement) {
            for (uint row=0u; row<equalityCount; ++row) {
                device const MRNumiHumanJointEqualityGPU& equality=jointEqualities[row];
                float target=0.0f, derivative=0.0f, error=0.0f;
                if (!evaluateJointEquality(equality,qState,qBase,nq,nv,target,derivative,error)) {
                    fail(status,MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,row);
                    return;
                }
                float velocity=candidateV[equality.indices.y];
                if (equality.indices.w!=MR_INVALID_INDEX)
                    velocity=fma(-derivative,candidateV[equality.indices.w],velocity);
                equalityRhs[row]=clamp(-0.2f*error/timestep,-4.0f,4.0f)-velocity;
            }
            if (!mrNumiHumanBilateralSolve(equalityFactor,equalityScale,equalityPivots,equalityRhs,equalityCount)) {
                fail(status,MR_NUMI_HUMAN_STAND_JOINT_EQUALITY_FAILED,MR_INVALID_INDEX);
                return;
            }
            for (uint row=0u; row<equalityCount; ++row)
                equalityLambdas[row]+=equalityRhs[row];
            for (uint dof=0u; dof<nv; ++dof) {
                float correction=0.0f;
                for (uint row=0u; row<equalityCount; ++row) {
                    device const float* response=responseScratch+responseBase+
                        (3u*dispatch.supportContactCount+row)*nv;
                    correction=fma(equalityRhs[row],response[dof],correction);
                }
                candidateV[dof]+=correction;
            }
        }
    }
''' + s[end:]
p.write_text(s)
p=root/'src/metal/MetalArticulatedOperator.mm';s=p.read_text()
old='        std::size_t responsePerEnvironment = 0u;';assert s.count(old)==1
s=s.replace(old,old+'\n        std::size_t bilateralScratchElements = 0u;')
old='''            !checkedMultiply(input.environmentCount, responsePerEnvironment,
                             layout.standResponseElements))'''
new='''            // Per-environment bilateral Schur factor, diagonal scaling,
            // pivot indices and correction RHS. Never alias the next arena.
            !checkedAdd(input.stand.jointEqualities.size(), 3u,
                        bilateralScratchElements) ||
            !checkedMultiply(input.stand.jointEqualities.size(),
                             bilateralScratchElements, bilateralScratchElements) ||
            !checkedAdd(responsePerEnvironment, bilateralScratchElements,
                        responsePerEnvironment) ||
            !checkedMultiply(input.environmentCount, responsePerEnvironment,
                             layout.standResponseElements))'''
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
p=root/'include/metalrobo/numi_human_stand_gpu.h';s=p.read_text()
assert s.count('STAND_ABI_VERSION 8u')==1
p.write_text(s.replace('STAND_ABI_VERSION 8u','STAND_ABI_VERSION 9u'))
p=root/'CMakeLists.txt';s=p.read_text()
needle='        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/numi_human_friction.h"'
assert s.count(needle)==1
s=s.replace(needle,needle+'\n        "${CMAKE_CURRENT_SOURCE_DIR}/include/metalrobo/numi_human_bilateral.h"')
s+='''\nif(BUILD_TESTING)
    add_executable(metalrobo_numi_human_bilateral_test tests/numi_human_bilateral_test.cpp)
    target_include_directories(metalrobo_numi_human_bilateral_test PRIVATE include)
    target_compile_features(metalrobo_numi_human_bilateral_test PRIVATE cxx_std_20)
    add_test(NAME numi_human_bilateral COMMAND metalrobo_numi_human_bilateral_test)
endif()
''';p.write_text(s)
expected={
    'CMakeLists.txt':'e205ef85e86a3786462fde428d72743e0f943acda5ab6c70b492b8ca133c34f2',
    'include/metalrobo/numi_human_stand_gpu.h':'86a1d801dac6f398fc97aaaa6e3d2ce8d8c2930d927fa962008ee2048d92e1e5',
    'src/metal/MetalArticulatedOperator.mm':'910be77c1039acf364793ccac11f2a63a64016d7adafade436c82480d9979b5e',
    'src/metal/NumiHumanStand.metal':'eaf92e2e81e86ddabe4145c8b02637b8ba149d69170f19f389026defdabb6d7c',
    'include/metalrobo/numi_human_bilateral.h':'d5e1e97d70f1eae3f92464ec02660240e2759369918d6ad73ea7eb343697ba91',
    'tests/numi_human_bilateral_test.cpp':'b2484580af1fa2ca0bf496eefada525e9d32b1a7ad21749a6e9bae282c4daba4',
    'tests/numi_human_bilateral_metal_test.mm':'4f54a6e89196765f4826a703b4807e2b7c7508f947f264de4846484fc146df6e',
}
for name,digest in expected.items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual==digest,(name,actual,digest)
print('Exact bilateral source and test hashes verified')
