#include "NumiHumanFEMOuterGroundAlignment.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

int main() {
    using namespace metalrobo::human::coarse_tissue;
    const std::vector<float> seed{
        0.25f,0.1f,-0.2f,0.3f,0.0f,0.0f,0.0f,1.0f,0.75f};
    const std::vector<FEMOuterPointF32> sourceOuter{
        {{0.1f,0.2f,5.686e-6f}},
        {{0.3f,0.2f,2.0e-5f}},
        {{0.1f,0.4f,1.5e-5f}},
        {{0.3f,0.4f,3.0e-5f}}};
    unsigned evaluationCount=0u;
    auto evaluate=[&](std::span<const float> q,std::vector<FEMOuterPointF32>& output,
                      std::string& error) {
        ++evaluationCount;
        if(q.size()!=seed.size()) {
            error="unexpected q size";
            return false;
        }
        output=sourceOuter;
        const std::array<double,3> delta{
            double(q[1])-double(seed[1]),
            double(q[2])-double(seed[2]),
            double(q[3])-double(seed[3])};
        for(auto& point:output)for(unsigned axis=0;axis<3u;++axis)
            point[axis]=static_cast<float>(double(point[axis])+delta[axis]);
        return true;
    };

    FEMOuterGroundAlignment result;
    std::string error;
    assert(makeWholeBodyFEMGroundAlignedSupport(seed,1u,{0.0,0.0,0.0},
        {0.0,0.0,1.0},sourceOuter.size(),32u,evaluate,result,error));
    assert(error.empty());
    assert(evaluationCount>=2u&&evaluationCount<=18u);
    assert(result.outerNodeCount==4u);
    assert(result.legacyNHCNTWitnessCount==32u);
    assert(std::abs(result.initialMinimumGapM-5.686e-6)<1.0e-10);
    assert(result.alignedMinimumGapM>=1.0e-8&&result.alignedMinimumGapM<=1.0e-6);
    assert(result.maximumCommonTranslationResidualM<=1.0e-6);
    assert(std::abs(result.rootTranslationDeltaM[0])<1.0e-12);
    assert(std::abs(result.rootTranslationDeltaM[1])<1.0e-12);
    assert(std::abs(result.rootTranslationDeltaM[2]+5.686e-6)<1.0e-7);
    assert(result.q.size()==seed.size());
    for(std::size_t i=0;i<seed.size();++i)
        if(i!=3u)assert(result.q[i]==seed[i]);
    assert(result.q[3]<seed[3]);

    FEMOuterGroundAlignment unchanged;
    unchanged.outerNodeCount=999u;
    unchanged.q={9.0f};
    auto wrongCount=[&](std::span<const float>,std::vector<FEMOuterPointF32>& output,
                        std::string&) {
        output.assign(2u,FEMOuterPointF32{});
        return true;
    };
    assert(!makeWholeBodyFEMGroundAlignedSupport(seed,1u,{0.0,0.0,0.0},
        {0.0,0.0,1.0},sourceOuter.size(),32u,wrongCount,unchanged,error));
    assert(error.find("expected 4")!=std::string::npos);
    assert(unchanged.outerNodeCount==999u&&unchanged.q==std::vector<float>{9.0f});

    const std::vector<float> truncatedQ{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    assert(!makeWholeBodyFEMGroundAlignedSupport(truncatedQ,0u,{0.0,0.0,0.0},
        {0.0,0.0,1.0},sourceOuter.size(),32u,evaluate,unchanged,error));
    assert(error.find("floating root pose")!=std::string::npos);

    auto stationary=[&](std::span<const float>,std::vector<FEMOuterPointF32>& output,
                        std::string&) {
        output=sourceOuter;
        return true;
    };
    assert(!makeWholeBodyFEMGroundAlignedSupport(seed,1u,{0.0,0.0,0.0},
        {0.0,0.0,1.0},sourceOuter.size(),32u,stationary,unchanged,error));
    assert(error.find("1um first-touch bound")!=std::string::npos||
           error.find("common translation")!=std::string::npos);

    // A callback that remains just below the contact floor must never be
    // admitted merely because it fits the older +/-1 um placement tolerance.
    auto immovablePenetration=[&](std::span<const float>,
            std::vector<FEMOuterPointF32>& output,std::string&) {
        output=sourceOuter;output[0][2]=-5.777e-9f;return true;
    };
    assert(!makeWholeBodyFEMGroundAlignedSupport(seed,1u,{0.0,0.0,0.0},
        {0.0,0.0,1.0},sourceOuter.size(),32u,immovablePenetration,unchanged,error));
    assert(unchanged.outerNodeCount==999u&&unchanged.q==std::vector<float>{9.0f});

    auto nonfinite=[&](std::span<const float>,std::vector<FEMOuterPointF32>& output,
                       std::string&) {
        output=sourceOuter;
        output[0][2]=std::numeric_limits<float>::quiet_NaN();
        return true;
    };
    assert(!makeWholeBodyFEMGroundAlignedSupport(seed,1u,{0.0,0.0,0.0},
        {0.0,0.0,1.0},sourceOuter.size(),32u,nonfinite,unchanged,error));
    assert(error.find("nonfinite Float32 point")!=std::string::npos);

    std::cout<<"whole-body FEM outer ground alignment check passed\n";
    return 0;
}

