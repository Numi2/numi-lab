from pathlib import Path
import shutil
R=Path("/Users/n/numi-human-aortic-contact-opt-20261019");E=R/"matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001";D=E/"complete-contact-capacity-139";D.mkdir(exist_ok=False);O=D/"source-overlay"
shutil.copytree(E/"sparse-contact-arithmetic-138/source-overlay",O,ignore=shutil.ignore_patterns("__pycache__"))
for n in ["matter/include/numi/matter/numi_human_shared.h","matter/include/numi/matter/fem_body_triangle_contact_neighborhood_validation.hpp"]:
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/n,p)
p=O/"matter/include/numi/matter/numi_human_shared.h";s=p.read_text();s=s.replace("#define NM_NUMI_HUMAN_FEM_BODY_CONTACT_NEIGHBORHOOD_MOTION_ENVELOPE 1u","#define NM_NUMI_HUMAN_FEM_BODY_CONTACT_NEIGHBORHOOD_MOTION_ENVELOPE 1u\n// Bounded complete banks: the integrated source preflight requires up to 869.\n#define NM_NUMI_HUMAN_FEM_BODY_CONTACT_MAXIMUM_NEIGHBORHOOD_TRIANGLES 1024u");p.write_text(s)
p=O/"matter/include/numi/matter/fem_body_triangle_contact_neighborhood_validation.hpp";s=p.read_text().replace("kMaximumTrianglesPerSample = 128u;","kMaximumTrianglesPerSample =\n    NM_NUMI_HUMAN_FEM_BODY_CONTACT_MAXIMUM_NEIGHBORHOOD_TRIANGLES;");p.write_text(s)
p=O/"matter/src/metal/numi_human.metalinc";s=p.read_text().replace("range.triangleCount > 128u", "range.triangleCount >\n            NM_NUMI_HUMAN_FEM_BODY_CONTACT_MAXIMUM_NEIGHBORHOOD_TRIANGLES");p.write_text(s)
p=O/"tests/numi_human_fem_body_triangle_contact/fem_body_triangle_contact_math_test.cpp";s=p.read_text().replace("0.00007f,overflowInventory.size()","0.001f,overflowInventory.size()").replace("overflowReport.candidateCount==129u","overflowReport.candidateCount==kMaximumTrianglesPerSample+1u")
needle="    const auto missingInventory=appendCompleteNeighborhood("
assert needle in s
s=s.replace(needle,"""    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> maximumOutput;
    NMNumiHumanFEMBodyContactNeighborhoodRangeGPU maximumRange{};
    const auto maximumReport=appendCompleteNeighborhood(
        12u,{0.003f,0.003f,0,0},0.001f,kMaximumTrianglesPerSample,
        std::span(overflowInventory.data(),kMaximumTrianglesPerSample),
        maximumOutput,maximumRange);
    require(maximumReport.complete() &&
            maximumReport.candidateCount==kMaximumTrianglesPerSample &&
            maximumRange.triangleCount==kMaximumTrianglesPerSample &&
            maximumOutput.size()==kMaximumTrianglesPerSample,
            "complete maximum-capacity bank is admitted without truncation");
"""+needle);p.write_text(s)
p=O/"tests/numi_human_fem_body_triangle_contact/fem_body_triangle_contact_coupon.mm";s=p.read_text().replace("#include <cstdlib>","#include <cstdlib>\n#include <vector>")
needle="        // A genuinely empty optional triangle table"
assert needle in s
s=s.replace(needle,"""        // The final candidate wins a complete maximum-size bank. This catches
        // CPU/Metal capacity disagreement and accidental prefix truncation.
        constexpr nm_u32 bankLimit=
            NM_NUMI_HUMAN_FEM_BODY_CONTACT_MAXIMUM_NEIGHBORHOOD_TRIANGLES;
        std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
            maximumCandidates(bankLimit+1u,ownerCandidates[0]);
        for (nm_u32 i=0u;i<maximumCandidates.size();++i) {
            maximumCandidates[i].triangleStableId=1000u+i;
            maximumCandidates[i].boneStableId=1u;
        }
        maximumCandidates[bankLimit-1u].bodyLocalVertex0.z=-0.04f;
        maximumCandidates[bankLimit-1u].bodyLocalVertex1.z=-0.04f;
        maximumCandidates[bankLimit-1u].bodyLocalVertex2.z=-0.04f;
        NMNumiHumanTendonFEMLoadDispatchGPU maximumDispatch=dispatch;
        maximumDispatch.femBodyContactSampleCount=1u;
        maximumDispatch.femBodyContactCandidateTriangleCount=bankLimit;
        NMNumiHumanFEMBodyContactNeighborhoodRangeGPU maximumRange{
            0u,bankLimit,0.05f,1u,{0.4f,0.4f,-0.05f,0.01f}};
        std::array<nm_float4,2> maximumOutput{};
        auto maximumDispatchBuffer=dispatchBuffer(&maximumDispatch,sizeof(maximumDispatch));
        auto maximumRangeBuffer=dispatchBuffer(&maximumRange,sizeof(maximumRange));
        auto maximumCandidateBuffer=dispatchBuffer(maximumCandidates.data(),
            maximumCandidates.size()*sizeof(maximumCandidates[0]));
        auto maximumOutputBuffer=dispatchBuffer(maximumOutput.data(),sizeof(maximumOutput));
        for (nm_u32 trial=0u;trial<2u;++trial) {
            if (trial==1u) {
                maximumDispatch.femBodyContactCandidateTriangleCount=bankLimit+1u;
                maximumRange.triangleCount=bankLimit+1u;
                std::memcpy(maximumDispatchBuffer.contents,&maximumDispatch,sizeof(maximumDispatch));
                std::memcpy(maximumRangeBuffer.contents,&maximumRange,sizeof(maximumRange));
            }
            auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:selectorPipeline];
            [encoder setBuffer:maximumDispatchBuffer offset:0 atIndex:0];
            [encoder setBuffer:ownerSampleBuffer offset:0 atIndex:1];
            [encoder setBuffer:nodeBuffer offset:0 atIndex:2];
            [encoder setBuffer:poseBuffer offset:0 atIndex:3];
            [encoder setBuffer:maximumOutputBuffer offset:0 atIndex:4];
            [encoder setBuffer:maximumRangeBuffer offset:0 atIndex:5];
            [encoder setBuffer:maximumCandidateBuffer offset:0 atIndex:6];
            [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            require(command.status==MTLCommandBufferStatusCompleted,"maximum bank query completes");
            const auto* out=static_cast<const nm_float4*>(maximumOutputBuffer.contents);
            if (trial==0u)
                require(std::isfinite(out[0].x) && std::abs(out[1].z+0.04f)<1.0e-6f &&
                    std::abs(out[0].z-2.499975f)<2.0e-3f,
                    "last triangle of complete maximum bank wins the GPU query");
            else require(std::isnan(out[0].x)&&std::isnan(out[1].x),
                    "GPU bank capacity overflow fails closed");
        }

"""+needle).replace("#include <vector>","#include <vector>\n#include <cstring>");p.write_text(s)
print(D)
