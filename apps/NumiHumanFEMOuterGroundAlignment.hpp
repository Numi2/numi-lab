#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

using FEMOuterPointF32=std::array<float,3>;

struct FEMOuterGroundAlignment {
    std::vector<float> q;
    std::uint32_t outerNodeCount=0u;
    std::uint32_t legacyNHCNTWitnessCount=0u;
    double initialMinimumGapM=0.0;
    double alignedMinimumGapM=0.0;
    std::uint32_t outwardRoundingSteps=0u;
    double seedTranslationMeters=0.0;
    double maximumCommonTranslationResidualM=0.0;
    std::array<double,3> rootTranslationDeltaM{};
};

// Align from the exact Float32 outer FEM rows while retaining the legacy
// support-witness count as provenance. The evaluator must return the same
// authored outer rows for a supplied Float32 q and must use its normal owner
// path (FK, registered mechanical region, source deltas, common placement,
// Float32 emission). The evaluator is run both before and after translation.
template<class EvaluateOuterF32>
[[nodiscard]] inline bool makeWholeBodyFEMGroundAlignedSupport(
        const std::span<const float> seedQ,
        const std::uint32_t rootQOffset,
        const std::array<double,3>& groundPoint,
        const std::array<double,3>& groundNormalInput,
        const std::size_t expectedOuterCount,
        const std::uint32_t legacyNHCNTWitnessCount,
        EvaluateOuterF32&& evaluateOuterF32,
        FEMOuterGroundAlignment& result,
        std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    if(seedQ.empty()||std::uint64_t(rootQOffset)+7u>seedQ.size())
        return fail("FEM outer ground alignment requires a complete floating root pose");
    if(expectedOuterCount==0u||expectedOuterCount>std::numeric_limits<std::uint32_t>::max())
        return fail("FEM outer ground alignment requires a bounded nonempty outer-node domain");
    if(!std::all_of(groundPoint.begin(),groundPoint.end(),[](double x){return std::isfinite(x);})||
       !std::all_of(groundNormalInput.begin(),groundNormalInput.end(),[](double x){return std::isfinite(x);}))
        return fail("FEM outer ground plane has nonfinite coordinates");
    const double normalLength=std::sqrt(groundNormalInput[0]*groundNormalInput[0]+
        groundNormalInput[1]*groundNormalInput[1]+groundNormalInput[2]*groundNormalInput[2]);
    if(!std::isfinite(normalLength)||normalLength<=1.0e-8||
       std::abs(normalLength-1.0)>1.0e-5)
        return fail("FEM outer ground alignment requires the authored unit bed normal");
    std::array<double,3> normal{};
    for(unsigned axis=0;axis<3u;++axis)normal[axis]=groundNormalInput[axis]/normalLength;
    for(const float value:seedQ)
        if(!std::isfinite(value))
            return fail("FEM outer ground alignment seed q contains a nonfinite value");

    const auto evaluate=[&](const std::vector<float>& q,std::vector<FEMOuterPointF32>& points,
                            std::string_view phase)->bool {
        std::string callbackError;
        if(!evaluateOuterF32(std::span<const float>(q),points,callbackError))
            return fail("FEM outer "+std::string(phase)+" evaluator failed: "+callbackError);
        if(points.size()!=expectedOuterCount)
            return fail("FEM outer "+std::string(phase)+" evaluator returned "+
                std::to_string(points.size())+" rows; expected "+
                std::to_string(expectedOuterCount));
        for(std::size_t row=0;row<points.size();++row)
            for(const float coordinate:points[row])
                if(!std::isfinite(coordinate))
                    return fail("FEM outer "+std::string(phase)+
                        " evaluator returned a nonfinite Float32 point at row "+
                        std::to_string(row));
        return true;
    };
    const auto minimumGap=[&](const std::vector<FEMOuterPointF32>& points) {
        double value=std::numeric_limits<double>::infinity();
        for(const auto& point:points) {
            const double gap=(double(point[0])-groundPoint[0])*normal[0]+
                (double(point[1])-groundPoint[1])*normal[1]+
                (double(point[2])-groundPoint[2])*normal[2];
            value=std::min(value,gap);
        }
        return value;
    };

    std::vector<float> seed(seedQ.begin(),seedQ.end());
    std::vector<FEMOuterPointF32> outerAtSeed;
    if(!evaluate(seed,outerAtSeed,"seed"))return false;
    const double initialGap=minimumGap(outerAtSeed);
    if(!std::isfinite(initialGap)||initialGap<-1.0e-4||initialGap>0.25)
        return fail("FEM outer minimum ground gap is outside the existing bounded placement range: "+
            std::to_string(initialGap));
    // The contact owner requires positive numerical clearance. Place the
    // source 10 nm above the plane, then round the root outward if its
    // Float32 evaluation falls below that margin. This is initialization
    // geometry only: gravity and FEM contact determine all later reactions.
    constexpr double firstTouchMarginM=1.0e-8;
    std::vector<float> aligned=seed;
    FEMOuterGroundAlignment candidate;
    candidate.outerNodeCount=static_cast<std::uint32_t>(expectedOuterCount);
    candidate.legacyNHCNTWitnessCount=legacyNHCNTWitnessCount;
    candidate.initialMinimumGapM=initialGap;
    candidate.seedTranslationMeters=initialGap;
    for(unsigned axis=0;axis<3u;++axis) {
        const auto index=static_cast<std::size_t>(rootQOffset)+axis;
        const double coordinate=double(seed[index])-normal[axis]*(initialGap-firstTouchMarginM);
        if(!std::isfinite(coordinate)||!std::isfinite(static_cast<float>(coordinate)))
            return fail("FEM outer ground alignment produced a nonfinite floating-root translation");
        aligned[index]=static_cast<float>(coordinate);
        candidate.rootTranslationDeltaM[axis]=double(aligned[index])-double(seed[index]);
    }
    for(std::size_t q=0;q<aligned.size();++q)
        if((q<rootQOffset||q>=static_cast<std::size_t>(rootQOffset)+3u)&&aligned[q]!=seed[q])
            return fail("FEM outer ground alignment changed a non-translation coordinate");

    std::vector<FEMOuterPointF32> outerAfter;
    if(!evaluate(aligned,outerAfter,"aligned"))return false;
    double finalGap=minimumGap(outerAfter);
    const unsigned outwardAxis=static_cast<unsigned>(std::distance(normal.begin(),
        std::max_element(normal.begin(),normal.end(),[](double a,double b){
            return std::abs(a)<std::abs(b); })));
    while(std::isfinite(finalGap)&&finalGap<firstTouchMarginM&&
          candidate.outwardRoundingSteps<16u) {
        const auto index=static_cast<std::size_t>(rootQOffset)+outwardAxis;
        aligned[index]=std::nextafter(aligned[index],
            normal[outwardAxis]>0.0?std::numeric_limits<float>::infinity():
                -std::numeric_limits<float>::infinity());
        if(!std::isfinite(aligned[index]))
            return fail("FEM outer outward-rounded root coordinate is nonfinite");
        ++candidate.outwardRoundingSteps;
        if(!evaluate(aligned,outerAfter,"outward-rounded"))return false;
        finalGap=minimumGap(outerAfter);
    }
    if(!std::isfinite(finalGap)||finalGap<firstTouchMarginM||finalGap>1.0e-6)
        return fail("aligned FEM outer minimum ground gap exceeds the existing 1um first-touch bound: "+
            std::to_string(finalGap));
    for(unsigned axis=0;axis<3u;++axis)
        candidate.rootTranslationDeltaM[axis]=
            double(aligned[static_cast<std::size_t>(rootQOffset)+axis])-
            double(seed[static_cast<std::size_t>(rootQOffset)+axis]);
    double maximumResidual=0.0;
    for(std::size_t row=0;row<outerAtSeed.size();++row)for(unsigned axis=0;axis<3u;++axis) {
        const double actual=double(outerAfter[row][axis])-double(outerAtSeed[row][axis]);
        const double residual=std::abs(actual-candidate.rootTranslationDeltaM[axis]);
        if(!std::isfinite(residual))
            return fail("FEM outer common-placement residual is nonfinite");
        maximumResidual=std::max(maximumResidual,residual);
    }
    if(maximumResidual>1.0e-6)
        return fail("FEM outer rows did not follow the root-only common translation within 1um: "+
            std::to_string(maximumResidual));
    candidate.alignedMinimumGapM=finalGap;
    candidate.maximumCommonTranslationResidualM=maximumResidual;
    candidate.q=std::move(aligned);
    result=std::move(candidate);
    error.clear();
    return true;
}

} // namespace metalrobo::human::coarse_tissue

