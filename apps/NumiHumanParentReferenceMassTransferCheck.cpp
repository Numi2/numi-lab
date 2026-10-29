#include "NumiHumanParentReferenceMassTransfer.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

int main() {
    const std::vector<std::array<double,3>> points{
        {0,0,0},{1,0,0},{0,1,0},{0,0,1},{.25,.25,.25}};
    const std::array<std::array<std::uint32_t,4>,1> parentTets{{{{0,1,2,3}}}};
    const std::array<std::uint32_t,1> parentMaterial{0},parentDonor{17};
    const double parentVolume=parentReferenceTetVolumeM3(parentTets[0],points);
    assert(std::abs(parentVolume-1.0/6.0)<1.0e-15);
    const std::array<double,1> parentVolumes{parentVolume},parentMass{parentVolume*1000.0};
    const std::array<double,2> densityLower{900.0,850.0},densityUpper{1100.0,950.0};
    std::string error;

    std::vector<std::uint32_t> offsets,parents;
    std::vector<double> measures;
    std::array<std::array<std::uint32_t,4>,1> sameTet{{{{0,1,2,3}}}};
    std::array<std::uint32_t,1> sameMaterial{0};
    assert(deriveIdentityParentMeasures(points,parentTets,parentMaterial,sameTet,sameMaterial,
        offsets,parents,measures,error));
    assert((offsets==std::vector<std::uint32_t>{0,1}));
    assert((parents==std::vector<std::uint32_t>{0}));
    assert(std::abs(measures[0]-parentVolume)<1.0e-15);

    // Split the parent at its barycentric center. Each child has one quarter of
    // the parent reference measure, independent of any future current pose.
    const std::array<std::array<std::uint32_t,4>,4> splitTets{{
        {{1,2,3,4}},{{0,2,3,4}},{{0,1,3,4}},{{0,1,2,4}}
    }};
    const std::array<std::uint32_t,4> splitMaterials{0,0,0,0},splitParents{0,0,0,0};
    const std::array<std::uint32_t,5> nodeParent{4294967295u,4294967295u,4294967295u,4294967295u,0};
    const std::array<std::array<double,4>,5> nodeBary{{
        {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{.25,.25,.25,.25}
    }};
    assert(deriveSingleParentRefinementMeasures(points,parentTets,parentMaterial,
        splitTets,splitMaterials,splitParents,nodeParent,nodeBary,offsets,parents,measures,error));
    assert((offsets==std::vector<std::uint32_t>{0,1,2,3,4}));
    for(std::size_t i=0;i<4;++i)assert(parents[i]==0u&&std::abs(measures[i]-parentVolume*.25)<1.0e-15);

    const std::array<double,4> childVolumes{parentVolume*.25,parentVolume*.25,parentVolume*.25,parentVolume*.25};
    ParentReferenceMassTransferInput input{
        parentVolumes,parentMass,parentMaterial,parentDonor,childVolumes,splitMaterials,
        offsets,parents,measures,densityLower,densityUpper};
    ParentReferenceMassTransferResult result;
    assert(assembleParentReferenceMassTransfer(input,result,error));
    assert(result.childParentVolumeFractions.size()==4u);
    for(std::size_t c=0;c<4;++c){
        assert(std::abs(result.childParentVolumeFractions[c]-1.0)<1.0e-15);
        assert(std::abs(result.childMassKg[c]-parentMass[0]*.25)<1.0e-14);
        assert(std::abs(result.childImpliedDensityKgM3[c]-1000.0)<1.0e-10);
    }
    assert(std::abs(result.parentMeasureCoveredM3[0]-parentVolume)<1.0e-15);
    assert(std::abs(result.parentMassTransferredKg[0]-parentMass[0])<1.0e-14);
    assert(std::abs(result.childMassByMaterialOriginDonorKg[{0,17}]-parentMass[0])<1.0e-14);

    // A single global midpoint on an edge shared by two parent tets carries
    // one source-parent row, but its pullback barycentrics are valid in both
    // adjacent cells. The helper remaps it through the common reference point.
    const std::vector<std::array<double,3>> adjacentPoints{
        {0,0,0},{1,0,0},{0,1,0},{0,0,1},{-1,-1,-1},{.5,.5,0}};
    const std::array<std::array<std::uint32_t,4>,2> adjacentParents{{
        {{0,1,2,3}},{{1,2,3,4}}
    }};
    const std::array<std::uint32_t,2> adjacentMaterials{0,0};
    const std::array<std::array<std::uint32_t,4>,3> adjacentChildren{{
        {{0,1,2,3}},{{1,5,3,4}},{{5,2,3,4}}
    }};
    const std::array<std::uint32_t,3> adjacentChildMaterials{0,0,0},adjacentChildParents{0,1,1};
    const std::array<std::uint32_t,6> adjacentNodeParent{
        4294967295u,4294967295u,4294967295u,4294967295u,4294967295u,0};
    const std::array<std::array<double,4>,6> adjacentNodeBary{{
        {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,.5,.5,0}
    }};
    std::vector<std::uint32_t> adjacentOffsets,adjacentRows;
    std::vector<double> adjacentMeasures;
    assert(deriveSingleParentRefinementMeasures(adjacentPoints,adjacentParents,adjacentMaterials,
        adjacentChildren,adjacentChildMaterials,adjacentChildParents,adjacentNodeParent,
        adjacentNodeBary,adjacentOffsets,adjacentRows,adjacentMeasures,error));
    const double adjacentParent1=parentReferenceTetVolumeM3(adjacentParents[1],adjacentPoints);
    assert(std::abs(adjacentMeasures[0]-parentReferenceTetVolumeM3(adjacentParents[0],adjacentPoints))<1.0e-14);
    assert(std::abs(adjacentMeasures[1]+adjacentMeasures[2]-adjacentParent1)<1.0e-14);
    assert(std::abs(adjacentMeasures[1]-adjacentParent1*.5)<1.0e-14);
    auto falseParentRows=adjacentChildParents;falseParentRows[1]=0;
    assert(!deriveSingleParentRefinementMeasures(adjacentPoints,adjacentParents,adjacentMaterials,
        adjacentChildren,adjacentChildMaterials,falseParentRows,adjacentNodeParent,adjacentNodeBary,
        adjacentOffsets,adjacentRows,adjacentMeasures,error));

    // Mixed-parent refinement is supported only from explicit source-reference
    // overlap measures; donor mass remains attributed to each origin ledger.
    const std::array<double,2> mixedParentVolumes{parentVolume,parentVolume};
    const std::array<double,2> mixedParentMass{parentVolume*1000.0,parentVolume*900.0};
    const std::array<std::uint32_t,2> mixedParentMaterials{0,0},mixedParentDonors{17,18};
    const std::array<double,2> mixedChildVolumes{parentVolume,parentVolume};
    const std::array<std::uint32_t,2> mixedChildMaterials{0,0};
    const std::array<std::uint32_t,3> mixedOffsets{0,2,4};
    const std::array<std::uint32_t,4> mixedParents{0,1,0,1};
    const std::array<double,4> mixedMeasures{parentVolume*.5,parentVolume*.5,parentVolume*.5,parentVolume*.5};
    ParentReferenceMassTransferInput mixedParentInput{
        mixedParentVolumes,mixedParentMass,mixedParentMaterials,mixedParentDonors,
        mixedChildVolumes,mixedChildMaterials,mixedOffsets,mixedParents,mixedMeasures,
        densityLower,densityUpper};
    ParentReferenceMassTransferResult mixedResult;
    assert(assembleParentReferenceMassTransfer(mixedParentInput,mixedResult,error));
    for(std::size_t c=0;c<2;++c){
        assert(std::abs(mixedResult.childParentVolumeFractions[2*c]-.5)<1.0e-15);
        assert(std::abs(mixedResult.childParentVolumeFractions[2*c+1]-.5)<1.0e-15);
        assert(std::abs(mixedResult.childImpliedDensityKgM3[c]-950.0)<1.0e-10);
    }
    assert(std::abs(mixedResult.childMassByMaterialOriginDonorKg[{0,17}]-mixedParentMass[0])<1.0e-14);
    assert(std::abs(mixedResult.childMassByMaterialOriginDonorKg[{0,18}]-mixedParentMass[1])<1.0e-14);
    const std::array<double,2> tooSmallChildren{parentVolume*.7,parentVolume*.7};
    auto outOfBoundsInput=mixedParentInput;outOfBoundsInput.childGeometricReferenceVolumeM3=tooSmallChildren;
    assert(!assembleParentReferenceMassTransfer(outOfBoundsInput,mixedResult,error));
    assert(error.find("transferred child mass implies density outside material bounds")!=std::string::npos);

    auto badMeasures=measures;badMeasures[0]*=1.001;
    auto badInput=input;badInput.childParentReferenceMeasureM3=badMeasures;
    ParentReferenceMassTransferResult rejected;
    assert(!assembleParentReferenceMassTransfer(badInput,rejected,error));
    assert(error.find("does not close an authoritative parent column")!=std::string::npos);

    auto wrongMaterial=splitMaterials;wrongMaterial[0]=1;
    auto mixedInput=input;mixedInput.childMaterialIds=wrongMaterial;
    assert(!assembleParentReferenceMassTransfer(mixedInput,rejected,error));
    assert(error.find("cross-material ancestry")!=std::string::npos);

    auto outside=splitTets;
    auto missingParent=nodeParent;missingParent[4]=4294967295u;
    assert(!deriveSingleParentRefinementMeasures(points,parentTets,parentMaterial,
        outside,splitMaterials,splitParents,missingParent,nodeBary,offsets,parents,measures,error));
    assert(error.find("crosses parent tets or lacks source-parent node ancestry")!=std::string::npos);

    const std::array<double,1> tooHighMass{parentVolume*1200.0};
    auto highInput=input;highInput.parentMassKg=tooHighMass;
    assert(!assembleParentReferenceMassTransfer(highInput,rejected,error));
    assert(error.find("authoritative parent mass implies density outside material bounds")!=std::string::npos);


    // Operational-reference proof: a node/tet bijection defines a piecewise-
    // affine map; the returned parent measures preserve each original cell mass.
    const std::vector<std::array<double,3>> pullbackSourcePoints{
        {0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    const std::array<std::array<std::uint32_t,4>,1> pullbackSourceTets{{{{0,1,2,3}}}};
    const std::array<std::array<double,3>,4> pullbackCandidatePoints{{
        {0.31,0.18,-0.07},{1.33,0.18,-0.07},{0.31,1.19,-0.07},{0.31,0.18,0.96}
    }};
    // Candidate node rows are permuted; mapping remains a bijection and the
    // child tet's orientation is preserved after applying that node map.
    const std::array<std::array<double,3>,4> pullbackPermutedPoints{{
        pullbackCandidatePoints[1],pullbackCandidatePoints[0],
        pullbackCandidatePoints[2],pullbackCandidatePoints[3]
    }};
    const std::array<std::uint32_t,4> pullbackNodeMap{1,0,2,3};
    const std::array<std::array<std::uint32_t,4>,1> pullbackCandidateTets{{{{1,0,2,3}}}};
    const std::array<std::uint32_t,1> pullbackTetMap{0};
    const std::array<std::uint32_t,1> pullbackMaterial{0},pullbackDonor{17},
        pullbackRegion{4},pullbackStableFace{123};
    const double pullbackSourceVolume=parentReferenceTetVolumeM3(
        pullbackSourceTets[0],pullbackSourcePoints);
    const std::array<double,1> pullbackMass{pullbackSourceVolume*1000.0};
    const std::array<double,2> pullbackDensityLower{900.0,850.0},
        pullbackDensityUpper{1100.0,950.0};
    const std::string pin64(64,'a');
    SameTopologyReferencePullbackInput pullbackInput{
        .sourceReferencePoints=pullbackSourcePoints,
        .candidateReferencePoints=pullbackPermutedPoints,
        .candidateToSourceNodeRows=pullbackNodeMap,
        .sourceTetrahedra=pullbackSourceTets,
        .candidateTetrahedra=pullbackCandidateTets,
        .candidateToSourceTetRows=pullbackTetMap,
        .sourceMaterialIds=pullbackMaterial,
        .candidateMaterialIds=pullbackMaterial,
        .sourceGeometricDonorIds=pullbackDonor,
        .candidateGeometricDonorIds=pullbackDonor,
        .sourceRegionIds=pullbackRegion,
        .candidateRegionIds=pullbackRegion,
        .sourceStableFaceIds=pullbackStableFace,
        .candidateStableFaceIds=pullbackStableFace,
        .sourceCellMassKg=pullbackMass,
        .materialDensityLowerKgM3=pullbackDensityLower,
        .materialDensityUpperKgM3=pullbackDensityUpper,
        .sourceVolumeSHA256=pin64,
        .candidateVolumeSHA256=pin64,
        .parentMassPacketSHA256=pin64,
        .parentSourceContributionSHA256=pin64,
        .sourceBoundarySelfAuditSHA256=pin64,
        .candidateBoundarySelfAuditSHA256=pin64,
        .sourceBoundarySelfIntersectionPairs=0u,
        .candidateBoundarySelfIntersectionPairs=0u
    };
    SameTopologyReferencePullbackResult pullback;
    assert(verifySameTopologyReferencePullback(pullbackInput,pullback,error));
    assert((pullback.childParentAncestryOffsets==std::vector<std::uint32_t>{0,1}));
    assert((pullback.childParentAncestryTetrahedra==std::vector<std::uint32_t>{0}));
    assert((pullback.childParentAncestryVolumeFractions==std::vector<double>{1.0}));
    assert(std::abs(pullback.childAncestryMeasureVolumeM3[0]-pullbackSourceVolume)<1.0e-15);
    assert(std::abs(pullback.childMassKg[0]-pullbackMass[0])<1.0e-15);
    assert(pullback.childImpliedDensityKgM3[0]>=900.0&&
           pullback.childImpliedDensityKgM3[0]<=1100.0);
    assert(pullback.minimumJacobianDeterminant>1.0);
    assert(pullback.maximumJacobianDeterminant==pullback.minimumJacobianDeterminant);
    assert(pullback.maximumNodeDisplacementM>0.1);
    assert(pullback.maximumTetCentroidDisplacementM>0.1);
    assert(pullback.verificationSHA256.size()==64u);

    const std::array<double,1> pullbackParentVolumes{pullbackSourceVolume};
    const std::array<std::uint32_t,1> pullbackParentDonor{17};
    const std::array<double,1> pullbackChildVolumes{pullback.candidateReferenceVolumeM3[0]};
    ParentReferenceMassTransferInput pullbackMassInput{
        pullbackParentVolumes,pullbackMass,pullbackMaterial,pullbackParentDonor,
        pullbackChildVolumes,pullbackMaterial,pullback.childParentAncestryOffsets,
        pullback.childParentAncestryTetrahedra,pullback.childAncestryMeasureVolumeM3,
        pullbackDensityLower,pullbackDensityUpper};
    ParentReferenceMassTransferResult pullbackMassResult;
    assert(assembleParentReferenceMassTransfer(pullbackMassInput,pullbackMassResult,error));
    assert(pullbackMassResult.childMassKg[0]==pullbackMass[0]);
    assert(std::abs(pullbackMassResult.childImpliedDensityKgM3[0]-
                    pullback.childImpliedDensityKgM3[0])<1.0e-10);
    assert(pullbackMassResult.parentMassTransferredKg[0]==pullbackMass[0]);

    const auto expectPullbackReject=[&](SameTopologyReferencePullbackInput bad,
                                         const std::string_view expected){
        SameTopologyReferencePullbackResult unchanged;
        unchanged.verificationSHA256="preserve-on-failure";
        const bool accepted=verifySameTopologyReferencePullback(bad,unchanged,error);
        assert(!accepted);
        assert(error.find(expected)!=std::string::npos);
        assert(unchanged.verificationSHA256=="preserve-on-failure");
    };
    auto duplicateNodeMap=pullbackInput;
    const std::array<std::uint32_t,4> badNodeMap{1,1,2,3};
    duplicateNodeMap.candidateToSourceNodeRows=badNodeMap;
    expectPullbackReject(duplicateNodeMap,"not one-to-one");

    auto duplicateTetMap=pullbackInput;
    const std::array<std::uint32_t,1> badTetMap{1};
    duplicateTetMap.candidateToSourceTetRows=badTetMap;
    expectPullbackReject(duplicateTetMap,"out of range or not one-to-one");

    auto wrongPullbackMaterial=pullbackInput;
    const std::array<std::uint32_t,1> wrongMaterialLabel{1};
    wrongPullbackMaterial.candidateMaterialIds=wrongMaterialLabel;
    expectPullbackReject(wrongPullbackMaterial,"changes material");

    auto wrongPullbackDonor=pullbackInput;
    const std::array<std::uint32_t,1> wrongDonorLabel{18};
    wrongPullbackDonor.candidateGeometricDonorIds=wrongDonorLabel;
    expectPullbackReject(wrongPullbackDonor,"geometric donor");

    auto wrongPullbackRegion=pullbackInput;
    const std::array<std::uint32_t,1> wrongRegionLabel{5};
    wrongPullbackRegion.candidateRegionIds=wrongRegionLabel;
    expectPullbackReject(wrongPullbackRegion,"region");

    auto wrongPullbackFace=pullbackInput;
    const std::array<std::uint32_t,1> wrongFaceLabel{124};
    wrongPullbackFace.candidateStableFaceIds=wrongFaceLabel;
    expectPullbackReject(wrongPullbackFace,"stable source-face");

    auto nonPositiveCandidate=pullbackInput;
    const std::array<std::array<double,3>,4> reflectedPoints{{
        {-0.71,0.18,-0.07},{0.31,0.18,-0.07},
        {0.31,1.19,-0.07},{0.31,0.18,0.96}
    }};
    nonPositiveCandidate.candidateReferencePoints=reflectedPoints;
    expectPullbackReject(nonPositiveCandidate,"nonpositive oriented reference cell");

    auto nonZeroSelf=pullbackInput;
    nonZeroSelf.candidateBoundarySelfIntersectionPairs=1u;
    expectPullbackReject(nonZeroSelf,"zero source and candidate");

    auto missingPin=pullbackInput;
    missingPin.candidateVolumeSHA256="missing";
    expectPullbackReject(missingPin,"requires pinned source");

    const std::array<std::array<double,3>,4> compressedPoints{{
        {0.31+0.8*1.02,0.18,-0.07},{0.31,0.18,-0.07},
        {0.31,0.18+0.8*1.01,-0.07},{0.31,0.18,-0.07+0.8*1.03}
    }};
    auto densityReject=pullbackInput;
    densityReject.candidateReferencePoints=compressedPoints;
    SameTopologyReferencePullbackResult densityRejected;
    assert(!verifySameTopologyReferencePullback(densityReject,densityRejected,error));
    assert(error.find("implies density outside actual bounds")!=std::string::npos);
    auto massDensityReject=pullbackMassInput;
    const std::array<double,1> compressedVolume{
        signedReferenceTetVolumeM3(pullbackCandidateTets[0],compressedPoints)};
    massDensityReject.childGeometricReferenceVolumeM3=compressedVolume;
    ParentReferenceMassTransferResult rejectedMassTransfer;
    assert(!assembleParentReferenceMassTransfer(massDensityReject,rejectedMassTransfer,error));


    auto orientationPermutationReject=pullbackInput;
    const std::array<std::array<std::uint32_t,4>,1> oddCandidateTet{{{{0,1,2,3}}}};
    orientationPermutationReject.candidateTetrahedra=oddCandidateTet;
    expectPullbackReject(orientationPermutationReject,"same oriented source simplex");

    auto sourceSelfNonZero=pullbackInput;
    sourceSelfNonZero.sourceBoundarySelfIntersectionPairs=1u;
    expectPullbackReject(sourceSelfNonZero,"zero source and candidate");

    auto changedButValidGeometry=pullbackInput;
    const std::array<std::array<double,3>,4> changedCandidatePoints{{
        {pullbackCandidatePoints[1][0]+1.0e-6,pullbackCandidatePoints[1][1],pullbackCandidatePoints[1][2]},
        {pullbackCandidatePoints[0][0]+1.0e-6,pullbackCandidatePoints[0][1],pullbackCandidatePoints[0][2]},
        {pullbackCandidatePoints[2][0]+1.0e-6,pullbackCandidatePoints[2][1],pullbackCandidatePoints[2][2]},
        {pullbackCandidatePoints[3][0]+1.0e-6,pullbackCandidatePoints[3][1],pullbackCandidatePoints[3][2]}
    }};
    changedButValidGeometry.candidateReferencePoints=changedCandidatePoints;
    SameTopologyReferencePullbackResult changedProof;
    assert(verifySameTopologyReferencePullback(changedButValidGeometry,changedProof,error));
    assert(changedProof.verificationSHA256!=pullback.verificationSHA256);
    auto changedSelfPin=pullbackInput;
    const std::string secondPin(64,'b');
    changedSelfPin.candidateBoundarySelfAuditSHA256=secondPin;
    SameTopologyReferencePullbackResult changedPinProof;
    assert(verifySameTopologyReferencePullback(changedSelfPin,changedPinProof,error));
    assert(changedPinProof.verificationSHA256!=pullback.verificationSHA256);

    std::cout<<"identity, single-parent split, same-topology pullback, mass/density closure, and fail-closed cases passed\n";
}
