#include "support/fem_body_triangle_contact_math.hpp"
#include "numi/matter/fem_body_triangle_contact_neighborhood_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <limits>

using namespace numi::matter::fem_body_triangle_contact;

namespace {
void require(bool value, std::string_view message) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %.*s\n",
                     int(message.size()), message.data());
        std::exit(1);
    }
}
double maxAbs(Vec3 a) { return std::max({std::abs(a.x),std::abs(a.y),std::abs(a.z)}); }
double component(Vec3 a, std::size_t i) {
    return i==0 ? a.x : (i==1 ? a.y : a.z);
}
double maxTangentError(const Evaluation& e, Vec3 p,
                       Vec3 bodyOrigin,
                       const std::array<Vec3,3>& tri,
                       double area, double k, double strain) {
    constexpr double h=2.0e-7;
    double maximum=0.0;
    for (std::size_t j=0;j<3;++j) {
        Vec3 plus=p, minus=p;
        if (j==0) { plus.x+=h; minus.x-=h; }
        if (j==1) { plus.y+=h; minus.y-=h; }
        if (j==2) { plus.z+=h; minus.z-=h; }
        const auto fp=evaluate(plus,bodyOrigin,tri,area,k,strain).slaveForce;
        const auto fm=evaluate(minus,bodyOrigin,tri,area,k,strain).slaveForce;
        for (std::size_t i=0;i<3;++i) {
            const double fd=-(component(fp,i)-component(fm,i))/(2.0*h);
            maximum=std::max(maximum,
                std::abs(fd-e.positiveStiffness[i][j])/
                std::max({1.0,std::abs(fd),std::abs(e.positiveStiffness[i][j])}));
        }
    }
    return maximum;
}
}

int main() {
    constexpr double area=0.25, k=1000.0, strain=1.0e-6;
    const std::array<Vec3,3> tri{{{0,0,0},{2,0,0},{0,2,0}}};
    const Vec3 bodyOrigin{0.1,-0.2,0.3};
    const auto sample=makeSample(7,12,tri[0],tri[1],tri[2],
                                 float(area),float(k),float(strain));
    require(sizeof(sample)==64, "triangle packing preserves 64-byte sample ABI");
    require(sample.flags==(NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
                           NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT),
            "explicit triangle flag combination");
    require(validTriangleContactSample(sample),
            "valid explicit triangle sample passes host validation");
    auto illegalFlags=sample;
    illegalFlags.flags |= NM_NUMI_HUMAN_FEM_BODY_CONTACT_ELASTIC_ATTACHMENT;
    require(!validTriangleContactSample(illegalFlags),
            "illegal combined contact modes fail before upload");
    auto zeroArea=sample;
    zeroArea.bodyLocalPointAndArea.w=0.0f;
    require(!validTriangleContactSample(zeroArea),
            "zero slave tributary area fails before upload");
    auto degenerate=sample;
    degenerate.bodyLocalNormalAndReferenceSeparation.x=0.0f;
    degenerate.bodyLocalNormalAndReferenceSeparation.y=0.0f;
    degenerate.bodyLocalNormalAndReferenceSeparation.z=0.0f;
    require(!validTriangleContactSample(degenerate),
            "degenerate master triangle fails before upload");
    auto negativeArea=sample;
    negativeArea.bodyLocalPointAndArea.w=-0.25f;
    require(!validTriangleContactSample(negativeArea),
            "negative slave tributary area fails before upload");
    auto nonFinite=sample;
    nonFinite.stiffnessAndNormalStrainPerPressure.y=
        std::numeric_limits<float>::quiet_NaN();
    require(!validTriangleContactSample(nonFinite),
            "non-finite packed coordinate fails before upload");
    auto invalidStiffness=sample;
    invalidStiffness.bodyLocalNormalAndReferenceSeparation.w=-1.0f;
    require(!validTriangleContactSample(invalidStiffness),
            "non-positive stiffness fails before upload");
    auto invalidReserved=sample;
    invalidReserved.reserved0=1u;
    require(!validTriangleContactSample(invalidReserved),
            "nonzero reserved field fails before upload");
    auto overflowProduct=sample;
    overflowProduct.bodyLocalPointAndArea.w=1.0e30f;
    overflowProduct.bodyLocalNormalAndReferenceSeparation.w=1.0e30f;
    require(!validTriangleContactSample(overflowProduct),
            "overflowing stiffness-area product fails before upload");
    require(sample.bodyLocalPointAndArea.w==0.25f,
            "slave quadrature area is explicit and differs from master area");

    using namespace numi::matter::fem_body_contact_neighborhood;
    require(sizeof(NMNumiHumanFEMBodyContactNeighborhoodRangeGPU)==32u,
            "ABI-15 neighborhood range includes its aligned motion envelope");
    const std::array<NMNumiHumanFEMBodyContactSampleGPU,2> neighborhoodSamples{
        sample,sample};
    require(validNeighborhoodSpans(neighborhoodSamples,{},{}),
            "empty optional spans preserve legacy contact mode");
    std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,2> candidates{};
    for (std::uint32_t i=0;i<candidates.size();++i) {
        auto& t=candidates[i];
        t.bodyLocalVertex0={0,0,float(i)*0.01f,0};
        t.bodyLocalVertex1={2,0,float(i)*0.01f,0};
        t.bodyLocalVertex2={0,2,float(i)*0.01f,0};
        t.authoredOutwardNormal={0,0,1,0};
        t.bodyIndex=sample.bodyIndex;
        t.boneStableId=40u+i;
        t.triangleStableId=80u+i;
    }
    const std::array<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU,2> ranges{{
        {0u,2u,0.02f,0u,{}},{2u,0u,0.0f,0u,{}}}};
    require(validNeighborhoodSpans(neighborhoodSamples,ranges,candidates),
            "contiguous bounded neighborhood plus legacy sample validates");
    auto malformedLegacyRange=ranges;
    malformedLegacyRange[0].restPointAndMotionRadius.x=1.0e-3f;
    require(!validNeighborhoodSpans(neighborhoodSamples,malformedLegacyRange,candidates),
            "legacy ranges reject nonzero ABI-15 envelope lanes");
    auto unknownEnvelopeFlag=ranges;
    unknownEnvelopeFlag[0].reserved0=2u;
    require(!validNeighborhoodSpans(neighborhoodSamples,unknownEnvelopeFlag,candidates),
            "unknown neighborhood range modes fail closed");
    auto fullSamples=neighborhoodSamples;
    fullSamples[0].flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE |
        NM_NUMI_HUMAN_FEM_BODY_CONTACT_ELASTIC_ATTACHMENT;
    const std::array<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU,2>
        fullRanges{{{0u,0u,0.0f,0u,{}},{0u,2u,0.02f,0u,{}}}};
    require(validNeighborhoodSpans(fullSamples,fullRanges,candidates),
            "filtered mixed samples keep a valid full source table");
    const std::vector<bool> filtered{true,false};
    std::vector<NMNumiHumanFEMBodyContactSampleGPU> solverSamples;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> solverRanges;
    require(appendUnselectedSolverRows(fullSamples,fullRanges,filtered,
                solverSamples,solverRanges) && solverSamples.size()==1u &&
            solverRanges.size()==1u && solverRanges[0].firstTriangle==0u &&
            solverRanges[0].triangleCount==2u &&
            solverRanges[0].maximumDistanceMeters==0.02f &&
            solverSamples[0].bodyIndex==fullSamples[1].bodyIndex,
            "filtered solver samples retain their original full-range alignment");
    std::vector<NMNumiHumanFEMBodyContactSampleGPU> legacySolverSamples;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> noSolverRanges;
    const std::vector<bool> noneSelected{false,false};
    require(appendUnselectedSolverRows(neighborhoodSamples,{},noneSelected,
                legacySolverSamples,noSolverRanges) &&
            legacySolverSamples.size()==2u && noSolverRanges.empty(),
            "empty optional spans preserve filtered legacy sample alignment");
    std::vector<NMNumiHumanFEMBodyContactSampleGPU> rejectedSolverSamples;
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> rejectedSolverRanges;
    require(!appendUnselectedSolverRows(neighborhoodSamples,
                std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>(ranges.data(),1u),
                filtered,rejectedSolverSamples,rejectedSolverRanges),
            "filtered solver projection rejects mismatched full-range indexing");
    auto duplicate=candidates;
    duplicate[1].boneStableId=duplicate[0].boneStableId;
    duplicate[1].triangleStableId=duplicate[0].triangleStableId;
    require(!validNeighborhoodSpans(neighborhoodSamples,ranges,duplicate),
            "duplicate stable triangle identity is rejected");
    auto wrongBody=candidates;
    wrongBody[1].bodyIndex=sample.bodyIndex+1u;
    require(!validNeighborhoodSpans(neighborhoodSamples,ranges,wrongBody),
            "cross-body candidate is rejected");
    auto wrongNormal=candidates;
    wrongNormal[1].authoredOutwardNormal={0,0,-1,0};
    require(!validNeighborhoodSpans(neighborhoodSamples,ranges,wrongNormal),
            "candidate winding must agree with authored outward normal");
    auto farBound=ranges;
    farBound[0].maximumDistanceMeters=0.0501f;
    require(!validNeighborhoodSpans(neighborhoodSamples,farBound,candidates),
            "unbounded neighborhood radius is rejected");
    auto discontinuous=ranges;
    discontinuous[1].firstTriangle=1u;
    require(!validNeighborhoodSpans(neighborhoodSamples,discontinuous,candidates),
            "noncontiguous flattened ranges are rejected");
    auto overCapacity=ranges;
    overCapacity[0].triangleCount=129u;
    require(!validNeighborhoodSpans(neighborhoodSamples,overCapacity,candidates),
            "per-sample triangle capacity is enforced");

    const auto registeredTriangle=[](
        std::uint32_t body,std::uint32_t bone,std::uint32_t face,
        nm_float4 a,nm_float4 b,nm_float4 c) {
        NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU triangle{};
        triangle.bodyLocalVertex0=a;
        triangle.bodyLocalVertex1=b;
        triangle.bodyLocalVertex2=c;
        const auto ab=nm_float4{b.x-a.x,b.y-a.y,b.z-a.z,0.0f};
        const auto ac=nm_float4{c.x-a.x,c.y-a.y,c.z-a.z,0.0f};
        auto normal=cross3(ab,ac);
        const double length=std::sqrt(dot3(normal,normal));
        normal.x=static_cast<float>(double(normal.x)/length);
        normal.y=static_cast<float>(double(normal.y)/length);
        normal.z=static_cast<float>(double(normal.z)/length);
        triangle.authoredOutwardNormal=normal;
        triangle.bodyIndex=body;
        triangle.boneStableId=bone;
        triangle.triangleStableId=face;
        return triangle;
    };
    const auto a0=nm_float4{0,0,0,0};
    const auto a1=nm_float4{0.01f,0,0,0};
    const auto a2=nm_float4{0.01f,0.01f,0,0};
    const auto b0=nm_float4{0,0,0,0};
    const auto b1=nm_float4{0.01f,0.01f,0,0};
    const auto b2=nm_float4{0,0.01f,0,0};
    const auto far0=nm_float4{0,0,0.1f,0};
    const auto far1=nm_float4{0.01f,0,0.1f,0};
    const auto far2=nm_float4{0.01f,0.01f,0.1f,0};
    const std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,4>
        seamInventory{{
            registeredTriangle(12u,10u,10u,a0,a1,a2),
            registeredTriangle(12u,10u,20u,b0,b1,b2),
            // A duplicated registration seam remains a separate stable face.
            registeredTriangle(12u,9u,30u,a0,a1,a2),
            registeredTriangle(12u,10u,99u,far0,far1,far2)}};
    const nm_float4 restPoint{0.008f,0.002f,0.0f,0.0f};
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
        completeCandidates;
    NMNumiHumanFEMBodyContactNeighborhoodRangeGPU completeRange{};
    const auto completeReport=appendCompleteNeighborhood(
        12u,restPoint,0.009f,seamInventory.size(),seamInventory,
        completeCandidates,completeRange);
    require(completeReport.complete() &&
            completeReport.candidateCount==3u &&
            completeCandidates.size()==3u &&
            completeRange.firstTriangle==0u &&
            completeRange.triangleCount==3u &&
            completeRange.maximumDistanceMeters>0.009f &&
            completeRange.maximumDistanceMeters<0.010f &&
            completeRange.reserved0==kRangeFlagMotionEnvelope &&
            completeRange.restPointAndMotionRadius.x==restPoint.x &&
            completeRange.restPointAndMotionRadius.y==restPoint.y &&
            completeRange.restPointAndMotionRadius.z==restPoint.z &&
            completeRange.restPointAndMotionRadius.w==0.009f,
            "producer emits a restpoint/radius guard and retains seam faces");
    require(withinMotionEnvelope(completeRange,
                {restPoint.x+0.009f,restPoint.y,restPoint.z,0.0f}) &&
            !withinMotionEnvelope(completeRange,
                {restPoint.x+0.0095f,restPoint.y,restPoint.z,0.0f}),
            "CPU envelope gate accepts its boundary and rejects tangential escape");
    const std::array<NMNumiHumanFEMBodyContactSampleGPU,1>
        buildSamples{sample};
    auto malformedEnvelope=completeRange;
    malformedEnvelope.restPointAndMotionRadius.w=-0.001f;
    require(!validMotionEnvelope(malformedEnvelope),
            "negative motion radius is rejected");
    malformedEnvelope=completeRange;
    malformedEnvelope.restPointAndMotionRadius.x=
        std::numeric_limits<float>::quiet_NaN();
    require(!validMotionEnvelope(malformedEnvelope),
            "non-finite motion restpoint is rejected");
    malformedEnvelope=completeRange;
    malformedEnvelope.restPointAndMotionRadius.w=
        completeRange.maximumDistanceMeters+0.001f;
    require(!validMotionEnvelope(malformedEnvelope),
            "motion radius exceeding the certified distance is rejected");
    const std::array<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU,1>
        malformedEnvelopeSpan{{malformedEnvelope}};
    require(!validNeighborhoodSpans(buildSamples,malformedEnvelopeSpan,
                completeCandidates),
            "the Source-span validator rejects malformed motion-envelope metadata");
    require(validNeighborhoodSpans(buildSamples,
                std::span<const NMNumiHumanFEMBodyContactNeighborhoodRangeGPU>(
                    &completeRange,1u),completeCandidates),
            "producer output satisfies the existing Source-span validator");
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
        appendedCandidates{completeCandidates[0]};
    NMNumiHumanFEMBodyContactNeighborhoodRangeGPU appendedRange{};
    const auto appendedReport=appendCompleteNeighborhood(
        12u,restPoint,0.009f,seamInventory.size(),seamInventory,
        appendedCandidates,appendedRange);
    const std::array<NMNumiHumanFEMBodyContactSampleGPU,2> appendedSamples{
        sample,sample};
    const std::array<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU,2>
        appendedRanges{{{0u,1u,0.01f,0u,{}},appendedRange}};
    require(appendedReport.complete() && appendedRange.firstTriangle==1u &&
            appendedRange.triangleCount==3u &&
            validNeighborhoodSpans(appendedSamples,appendedRanges,
                appendedCandidates),
            "successive sample banks append with exact full-Source range offsets");

    const auto exhaustiveWinner=[](Vec3 query,auto triangles) {
        std::pair<std::uint32_t,std::uint32_t> winner{
            std::numeric_limits<std::uint32_t>::max(),
            std::numeric_limits<std::uint32_t>::max()};
        double best=std::numeric_limits<double>::infinity();
        for (const auto& triangle:triangles) {
            const std::array<Vec3,3> vertices{{
                read(triangle.bodyLocalVertex0),
                read(triangle.bodyLocalVertex1),
                read(triangle.bodyLocalVertex2)}};
            const auto closest=closestPointOnTriangle(
                query,vertices[0],vertices[1],vertices[2]);
            const double distance=norm2(sub(query,closest.point));
            const auto identity=std::pair{
                triangle.boneStableId,triangle.triangleStableId};
            if (distance<best || (distance==best && identity<winner)) {
                best=distance;
                winner=identity;
            }
        }
        return winner;
    };
    const auto restVec=read(restPoint);
    const std::array<Vec3,4> envelopeQueries{{
        restVec,{0.002,0.008,0.0},
        {double(a2.x)*0.5,double(a2.y)*0.5,0.0},{0.0,0.0,0.0}}};
    for (std::size_t i=0;i<envelopeQueries.size();++i) {
        require(norm(sub(envelopeQueries[i],restVec))<=0.009000001,
                "switch query remains inside declared sample motion envelope");
        const auto allWinner=exhaustiveWinner(envelopeQueries[i],seamInventory);
        const auto bankWinner=exhaustiveWinner(envelopeQueries[i],completeCandidates);
        require(allWinner==bankWinner,
                "bounded bank winner equals exhaustive winner through face-edge-vertex switches");
        if (i==0u)
            require(allWinner==std::pair<std::uint32_t,std::uint32_t>{9u,30u},
                    "duplicate registration seam keeps deterministic stable ownership");
        if (i==1u)
            require(allWinner==std::pair<std::uint32_t,std::uint32_t>{10u,20u},
                    "closest face switches across a shared edge");
        if (i==2u) {
            const std::array<Vec3,3> vertices{{
                read(completeCandidates[0].bodyLocalVertex0),
                read(completeCandidates[0].bodyLocalVertex1),
                read(completeCandidates[0].bodyLocalVertex2)}};
            const auto edgeClosest=closestPointOnTriangle(envelopeQueries[i],
                vertices[0],vertices[1],vertices[2]);
            require(norm2(sub(envelopeQueries[i],edgeClosest.point))<1.0e-24 &&
                    envelopeQueries[i].x==double(a2.x)*0.5 &&
                    envelopeQueries[i].y==double(a2.y)*0.5,
                    "shared-edge query lies on its exact registered finite boundary");
        }
        if (i==3u) {
            const std::array<Vec3,3> vertices{{
                read(completeCandidates[0].bodyLocalVertex0),
                read(completeCandidates[0].bodyLocalVertex1),
                read(completeCandidates[0].bodyLocalVertex2)}};
            const auto cornerClosest=closestPointOnTriangle(envelopeQueries[i],
                vertices[0],vertices[1],vertices[2]);
            require(norm2(sub(envelopeQueries[i],cornerClosest.point))<1.0e-24 &&
                    cornerClosest.point.x==0.0 && cornerClosest.point.y==0.0,
                    "corner query lies on its exact registered finite vertex");
        }
    }
    auto permutedInventory=seamInventory;
    std::reverse(permutedInventory.begin(),permutedInventory.end());
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
        permutedCandidates;
    NMNumiHumanFEMBodyContactNeighborhoodRangeGPU permutedRange{};
    const auto permutedReport=appendCompleteNeighborhood(
        12u,restPoint,0.009f,permutedInventory.size(),permutedInventory,
        permutedCandidates,permutedRange);
    require(permutedReport.complete() &&
            permutedRange.triangleCount==completeRange.triangleCount,
            "input triangle ordering does not change neighborhood size");
    for (std::size_t i=0;i<completeCandidates.size();++i)
        require(completeCandidates[i].boneStableId==
                    permutedCandidates[i].boneStableId &&
                completeCandidates[i].triangleStableId==
                    permutedCandidates[i].triangleStableId &&
                completeCandidates[i].bodyLocalVertex0.x==
                    permutedCandidates[i].bodyLocalVertex0.x &&
                completeCandidates[i].bodyLocalVertex0.y==
                    permutedCandidates[i].bodyLocalVertex0.y &&
                completeCandidates[i].bodyLocalVertex0.z==
                    permutedCandidates[i].bodyLocalVertex0.z,
                "candidate order is deterministic under source permutation");

    const std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,2>
        radiusBoundaryInventory{{
            registeredTriangle(12u,2u,20u,
                {0,0,0,0},{0.01f,0,0,0},{0,0.01f,0,0}),
            registeredTriangle(12u,1u,10u,
                {0,0,0.02f,0},{0.01f,0,0.02f,0},
                {0,0.01f,0.02f,0})}};
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
        radiusBoundaryCandidates;
    NMNumiHumanFEMBodyContactNeighborhoodRangeGPU radiusBoundaryRange{};
    const auto radiusBoundaryReport=appendCompleteNeighborhood(
        12u,{0.003f,0.003f,0,0},0.01f,
        radiusBoundaryInventory.size(),radiusBoundaryInventory,
        radiusBoundaryCandidates,radiusBoundaryRange);
    require(radiusBoundaryReport.complete() &&
            radiusBoundaryReport.candidateCount==2u &&
            radiusBoundaryCandidates.size()==2u,
            "inclusive d_min plus two-radii boundary preserves a switching face");
    const Vec3 halfway{0.003,0.003,0.01};
    require(exhaustiveWinner(halfway,radiusBoundaryInventory)==
                exhaustiveWinner(halfway,radiusBoundaryCandidates) &&
            exhaustiveWinner(halfway,radiusBoundaryCandidates)==
                std::pair<std::uint32_t,std::uint32_t>{1u,10u},
            "exact radius-boundary tie follows deterministic stable face ordering");

    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
        overflowInventory;
    overflowInventory.reserve(kMaximumTrianglesPerSample+1u);
    for (std::uint32_t i=0;i<kMaximumTrianglesPerSample+1u;++i) {
        const float z=float(i)*1.0e-6f;
        overflowInventory.push_back(registeredTriangle(12u,1u,1000u+i,
            {0,0,z,0},{0.01f,0,z,0},{0,0.01f,z,0}));
    }
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>
        overflowOutput;
    auto unchangedFailureRange=
        NMNumiHumanFEMBodyContactNeighborhoodRangeGPU{7u,8u,0.02f,9u,{}};
    const auto overflowReport=appendCompleteNeighborhood(
        12u,{0.003f,0.003f,0,0},0.001f,overflowInventory.size(),
        overflowInventory,overflowOutput,unchangedFailureRange);
    require(overflowReport.failure==
                NeighborhoodBuildFailure::candidateCapacityExceeded &&
            overflowReport.candidateCount==kMaximumTrianglesPerSample+1u && overflowOutput.empty() &&
            unchangedFailureRange.firstTriangle==7u &&
            unchangedFailureRange.triangleCount==8u &&
            unchangedFailureRange.maximumDistanceMeters==0.02f &&
            unchangedFailureRange.reserved0==9u &&
            unchangedFailureRange.restPointAndMotionRadius.x==0.0f,
            "overflow fails closed without truncating or partially mutating Source data");
    std::vector<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU> maximumOutput;
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
    const auto missingInventory=appendCompleteNeighborhood(
        12u,restPoint,0.009f,0u,
        std::span<const NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU>{},
        overflowOutput,unchangedFailureRange);
    require(missingInventory.failure==
                NeighborhoodBuildFailure::missingTopology,
            "an absent registered surface is rejected rather than treated as an empty bank");
    const auto countMismatch=appendCompleteNeighborhood(
        12u,restPoint,0.009f,seamInventory.size()+1u,seamInventory,
        overflowOutput,unchangedFailureRange);
    require(countMismatch.failure==
                NeighborhoodBuildFailure::inventoryCountMismatch,
            "incomplete registered topology is rejected by its declared inventory count");
    auto ambiguousInventory=seamInventory;
    ambiguousInventory[1].triangleStableId=
        ambiguousInventory[0].triangleStableId;
    ambiguousInventory[1].boneStableId=
        ambiguousInventory[0].boneStableId;
    const auto ambiguousReport=appendCompleteNeighborhood(
        12u,restPoint,0.009f,ambiguousInventory.size(),ambiguousInventory,
        overflowOutput,unchangedFailureRange);
    require(ambiguousReport.failure==
                NeighborhoodBuildFailure::ambiguousIdentity,
            "duplicate stable face identity is rejected as ambiguous topology");
    auto wrongOwnerInventory=seamInventory;
    wrongOwnerInventory[0].bodyIndex=13u;
    const auto wrongOwnerReport=appendCompleteNeighborhood(
        12u,restPoint,0.009f,wrongOwnerInventory.size(),wrongOwnerInventory,
        overflowOutput,unchangedFailureRange);
    require(wrongOwnerReport.failure==
                NeighborhoodBuildFailure::invalidTriangle,
            "mixed articulated-body coordinates are rejected, not merged");
    auto nonFiniteInventory=seamInventory;
    nonFiniteInventory[0].bodyLocalVertex0.x=
        std::numeric_limits<float>::quiet_NaN();
    const auto nonFiniteReport=appendCompleteNeighborhood(
        12u,restPoint,0.009f,nonFiniteInventory.size(),nonFiniteInventory,
        overflowOutput,unchangedFailureRange);
    require(nonFiniteReport.failure==
                NeighborhoodBuildFailure::invalidTriangle,
            "non-finite registered source faces fail the complete inventory");
    const auto radiusReport=appendCompleteNeighborhood(
        12u,restPoint,0.051f,seamInventory.size(),seamInventory,
        overflowOutput,unchangedFailureRange);
    require(radiusReport.failure==
                NeighborhoodBuildFailure::runtimeDistanceLimitExceeded,
            "envelopes beyond the runtime distance guard are not silently clipped");
    require(isUnitQuaternion({0.0,0.0,0.0,1.0}) &&
            !isUnitQuaternion({0.0,0.0,0.0,2.0}),
            "pose quaternion validation");
    const auto decoded=decodeTriangle(sample);
    require(decoded[0].x==tri[0].x && decoded[1].x==tri[1].x &&
            decoded[2].y==tri[2].y,
            "packed triangle round-trips exactly");

    const auto outside=evaluate({0.4,0.4,0.3},bodyOrigin,tri,area,k,strain);
    require(outside.valid && !outside.active && outside.closure==0.0 &&
            maxAbs(outside.slaveForce)==0.0,
            "outside the zero-gap surface produces exactly zero force");

    const Vec3 facePoint{0.4,0.4,-0.05};
    const auto face=evaluate(facePoint,bodyOrigin,tri,area,k,strain);
    require(face.valid && face.active && face.feature==Feature::face &&
            face.normal.z>0.999999 && face.slaveForce.z>0.0,
            "interior contact uses outward oriented face normal");
    require(maxTangentError(face,facePoint,bodyOrigin,tri,area,k,strain)<2.0e-5,
            "face force tangent matches finite differences");

    const Vec3 edgePoint{0.7,-0.03,-0.04};
    const auto edge=evaluate(edgePoint,bodyOrigin,tri,area,k,strain);
    require(edge.valid && edge.active && edge.feature==Feature::edge &&
            std::abs(edge.closestPoint.y)<1.0e-12 &&
            std::abs(edge.closestPoint.x-0.7)<1.0e-12 &&
            edge.normal.z>0.0,
            "closest edge point and outward normal");
    require(maxTangentError(edge,edgePoint,bodyOrigin,tri,area,k,strain)<3.0e-5,
            "edge curvature tangent matches finite differences");

    const Vec3 vertexPoint{-0.03,-0.04,-0.02};
    const auto vertex=evaluate(vertexPoint,bodyOrigin,tri,area,k,strain);
    require(vertex.valid && vertex.active && vertex.feature==Feature::vertex &&
            maxAbs(vertex.closestPoint)==0.0 && vertex.normal.z>0.0,
            "closest vertex point and outward normal");
    require(maxTangentError(vertex,vertexPoint,bodyOrigin,tri,area,k,strain)<4.0e-5,
            "vertex curvature tangent matches finite differences");

    const auto backside=evaluate({0.4,0.4,-0.02},bodyOrigin,tri,area,k,strain);
    require(backside.valid && backside.active && backside.normal.z>0.999999 &&
            backside.slaveForce.z>0.0,
            "penetration cannot flip force behind authored triangle winding");

    const Vec3 forceResidual=add(face.slaveForce,face.bodyForce);
    const Vec3 momentResidual=add(cross(facePoint,face.slaveForce),
        add(cross(bodyOrigin,face.bodyForce),face.bodyTorque));
    require(maxAbs(forceResidual)<1.0e-10 &&
            maxAbs(momentResidual)<1.0e-10,
            "equal/opposite force and wrench close about world origin");

    std::printf("{\"status\":\"PASS\",\"abi_bytes\":%zu,"
                "\"outside_force\":%.9g,\"face_force\":%.9g,"
                "\"edge_force\":%.9g,\"vertex_force\":%.9g,"
                "\"tangent_rel_error\":[%.3g,%.3g,%.3g],"
                "\"action_reaction_residual\":%.3g,"
                "\"moment_residual\":%.3g}\n",
                sizeof(sample),maxAbs(outside.slaveForce),
                maxAbs(face.slaveForce),maxAbs(edge.slaveForce),
                maxAbs(vertex.slaveForce),
                maxTangentError(face,facePoint,bodyOrigin,tri,area,k,strain),
                maxTangentError(edge,edgePoint,bodyOrigin,tri,area,k,strain),
                maxTangentError(vertex,vertexPoint,bodyOrigin,tri,area,k,strain),
                maxAbs(forceResidual),maxAbs(momentResidual));
    return 0;
}
