#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

static ConservativeSpatialMassTransportInput makeInput() {
    static const std::array<SpatialMassTransportPoint, 2> parentCentroids{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0},
        SpatialMassTransportPoint{0.0, 0.0, 0.0}}};
    static const std::array<double, 2> parentVolume{5.0e-4, 5.0e-4};
    static const std::array<double, 2> parentMass{0.5, 0.5};
    static const std::array<std::uint32_t, 2> parentMaterial{15, 15};
    static const std::array<std::uint32_t, 2> parentDonor{1, 2};
    static const std::array<std::uint32_t, 2> parentRegion{10, 20};
    static const std::array<std::uint32_t, 3> sourceOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> sourceDonors{1, 2};
    static const std::array<double, 2> sourceMass{0.5, 0.5};
    static const std::array<SpatialMassTransportPoint, 1> childCentroid{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0}}};
    static const std::array<double, 1> childVolume{1.0e-3};
    static const std::array<std::uint32_t, 1> childMaterial{15};
    static const std::array<std::uint32_t, 1> childDonor{1};
    static const std::array<std::uint32_t, 1> childRegion{10};
    static const std::array<double, 16> densityLower = [] {
        std::array<double, 16> x{}; x.fill(900.0); return x; }();
    static const std::array<double, 16> densityUpper = [] {
        std::array<double, 16> x{}; x.fill(1100.0); return x; }();
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld = parentCentroids;
    in.parentReferenceVolumeM3 = parentVolume;
    in.parentMassKg = parentMass;
    in.parentMaterialIndices = parentMaterial;
    in.parentDonorBodyIndices = parentDonor;
    in.parentAnatomicalRegionIndices = parentRegion;
    in.parentSourceDonorOffsets = sourceOffsets;
    in.parentSourceDonorBodyIndices = sourceDonors;
    in.parentSourceDonorMassKg = sourceMass;
    in.childCentroidsWorld = childCentroid;
    in.childReferenceVolumeM3 = childVolume;
    in.childMaterialIndices = childMaterial;
    in.childDonorBodyIndices = childDonor;
    in.childAnatomicalRegionIndices = childRegion;
    in.materialDensityLowerKgM3 = densityLower;
    in.materialDensityUpperKgM3 = densityUpper;
    static const std::string parentHash(64, 'a'), sourceHash(64, 'b');
    static const std::string childHash(64, 'c'), regionHash(64, 'd'), frameHash(64, 'e');
    in.parentMassPacketSHA256 = parentHash;
    in.parentSourceContributionSHA256 = sourceHash;
    in.childMeshSHA256 = childHash;
    in.regionLabelSHA256 = regionHash;
    in.coordinateFrameSHA256 = frameHash;
    in.maximumTransportDistanceM = 0.1;
    in.maximumAllowedEdges = 100;
    in.maximumAugmentations = 100;
    return in;
}

static ConservativeSpatialChildSemanticSupport makeSupport() {
    static const std::array<std::uint32_t, 2> offsets{0, 2};
    static const std::array<std::uint32_t, 2> slotGroups{0, 1};
    static const std::array<std::uint32_t, 2> materials{15, 15};
    static const std::array<std::uint32_t, 2> donors{1, 2};
    static const std::array<std::uint32_t, 2> regions{10, 20};
    static const std::array<std::uint64_t, 2> identities{0x8000000000000010ull, 0x8000000000000020ull};
    static const std::array<double, 2> fractions{0.5, 0.5};
    static const std::array<std::uint32_t, 2> parentFaces{501, 601};
    static const std::array<std::uint32_t, 3> parentTetOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> parentTets{0, 1};
    static const std::array<std::uint32_t, 3> stableFaceOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> stableFaces{501, 601};
    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets = offsets;
    support.slotGroupIndices = slotGroups;
    support.volumeFractions = fractions;
    support.groupMaterialIndices = materials;
    support.donorBodyIndices = donors;
    support.anatomicalRegionIndices = regions;
    support.derivedGroupIdentityIDs = identities;
    support.parentStableSourceFaceIdentities = parentFaces;
    support.sourceParentTetOffsets = parentTetOffsets;
    support.sourceParentTetRows = parentTets;
    support.sourceStableFaceOffsets = stableFaceOffsets;
    support.sourceStableFaceRows = stableFaces;
    static const std::string proofHash(64, 'f');
    support.typedSupportSHA256 = proofHash;
    return support;
}

static bool rejected(ConservativeSpatialMassTransportInput in,
                     ConservativeSpatialChildSemanticSupport support,
                     const char* name) {
    ConservativeSpatialMassTransportResult out;
    std::string error;
    if (assembleConservativeSpatialMassTransportWithSemanticSupport(in, support, out, error)) {
        std::cerr << "unexpected acceptance: " << name << "\n";
        return false;
    }
    return true;
}

int main() {
    auto in = makeInput();
    auto support = makeSupport();
    ConservativeSpatialMassTransportResult result;
    std::string error;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(
            in, support, result, error)) {
        std::cerr << "valid 50/50 case failed: " << error << "\n";
        return 1;
    }
    if (result.childMassKg.size() != 1 || result.childMassKg[0] != 1.0 ||
        result.childImpliedDensityKgM3.size() != 1 ||
        result.childImpliedDensityKgM3[0] != 1000.0 ||
        result.childParentRows.size() != 2 || result.groupClosure.size() != 2 ||
        result.childSourceDonorBodyIndices.size() != 2 ||
        result.childSourceDonorMassKg[0] != 0.5 ||
        result.childSourceDonorMassKg[1] != 0.5 ||
        result.childSemanticGroupOffsets != std::vector<std::uint32_t>({0, 2}) ||
        result.childSemanticGroupIndices != std::vector<std::uint32_t>({0, 1}) ||
        result.parentMassTransferredKg != std::vector<double>({0.5, 0.5})) {
        std::cerr << "50/50 aggregate did not conserve one real-cell mass and two source groups\n";
        return 2;
    }
    if (!verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
            in, support, result, error)) {
        std::cerr << "valid 50/50 receipt failed replay: " << error << "\n";
        return 3;
    }
    auto tampered = result;
    tampered.childParentVolumeFractions[0] += 0.01;
    if (verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
            in, support, tampered, error)) {
        std::cerr << "tampered aggregate flow was accepted\n";
        return 4;
    }

    auto badSum = support;
    static const std::array<double, 2> fractionsBadSum{0.5, 0.25};
    badSum.volumeFractions = fractionsBadSum;
    if (!rejected(in, badSum, "fraction sum")) return 5;

    auto unequalPreference = support;
    static const std::array<double, 2> fractionsDensity{0.9, 0.1};
    unequalPreference.volumeFractions = fractionsDensity;
    ConservativeSpatialMassTransportResult unequalPreferenceResult;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(
            in, unequalPreference, unequalPreferenceResult, error) ||
        unequalPreferenceResult.childMassKg != std::vector<double>({1.0}) ||
        unequalPreferenceResult.childImpliedDensityKgM3 != std::vector<double>({1000.0}) ||
        unequalPreferenceResult.groupClosure.size() != 2u ||
        unequalPreferenceResult.groupClosure[0].childMassKg != 0.5 ||
        unequalPreferenceResult.groupClosure[1].childMassKg != 0.5 ||
        !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
            in, unequalPreference, unequalPreferenceResult, error)) {
        std::cerr << "P1 preference changed a shared real-child capacity or source-group conservation: "
                  << error << "\n";
        return 6;
    }
    // Nearest-one seeds deliberately produce an infeasible subgraph. A
    // residual min-cut witness must add only crossing candidates and recover
    // a valid shared-capacity flow without raising the graph budget.
    {
        static const std::array<SpatialMassTransportPoint,3> parentPoints{{
            SpatialMassTransportPoint{0.0,0.0,0.0},
            SpatialMassTransportPoint{0.9,0.0,0.0},
            SpatialMassTransportPoint{1.0,0.0,0.0}}};
        static const std::array<double,3> parentVolumes{2.5e-3,2.5e-4,2.5e-4};
        static const std::array<double,3> parentMasses{2.5,0.25,0.25};
        static const std::array<std::uint32_t,3> parentMaterials{15,15,15};
        static const std::array<std::uint32_t,3> parentDonors{1,1,1};
        static const std::array<std::uint32_t,3> parentRegions{10,10,10};
        static const std::array<std::uint32_t,4> donorOffsets{0,1,2,3};
        static const std::array<std::uint32_t,3> donorIds{1,1,1};
        static const std::array<double,3> donorMasses{2.5,0.25,0.25};
        static const std::array<SpatialMassTransportPoint,3> childPoints{{
            SpatialMassTransportPoint{0.1,0.0,0.0},
            SpatialMassTransportPoint{0.2,0.0,0.0},
            SpatialMassTransportPoint{1.1,0.0,0.0}}};
        static const std::array<double,3> childVolumes{1.0e-3,1.0e-3,1.0e-3};
        static const std::array<std::uint32_t,3> childMaterials{15,15,15};
        static const std::array<std::uint32_t,3> childDonorIds{1,1,1};
        static const std::array<std::uint32_t,3> childRegionIds{10,10,10};
        static const std::array<double,16> exactLower=[] {std::array<double,16>x{};x.fill(1000.0);return x;}();
        static const std::array<double,16> exactUpper=[] {std::array<double,16>x{};x.fill(1000.0);return x;}();
        static const std::array<std::uint32_t,4> supportOffsets{0,1,2,3};
        static const std::array<std::uint32_t,3> supportGroups{0,0,0};
        static const std::array<double,3> supportFractions{1.0,1.0,1.0};
        static const std::array<std::uint32_t,1> groupMaterial{15};
        static const std::array<std::uint32_t,1> groupDonor{1};
        static const std::array<std::uint32_t,1> groupRegion{10};
        static const std::array<std::uint64_t,1> groupIdentity{0x8000000000000010ull};
        static const std::array<std::uint32_t,3> parentStable{501,502,503};
        static const std::array<std::uint32_t,2> groupParentOffsets{0,3};
        static const std::array<std::uint32_t,3> groupParentRows{0,1,2};
        static const std::array<std::uint32_t,2> groupFaceOffsets{0,3};
        static const std::array<std::uint32_t,3> groupFaces{501,502,503};
        static const std::string h1(64,'1'),h2(64,'2'),h3(64,'3'),h4(64,'4'),h5(64,'5');
        ConservativeSpatialMassTransportInput sparseInput;
        sparseInput.parentCentroidsWorld=parentPoints;
        sparseInput.parentReferenceVolumeM3=parentVolumes;
        sparseInput.parentMassKg=parentMasses;
        sparseInput.parentMaterialIndices=parentMaterials;
        sparseInput.parentDonorBodyIndices=parentDonors;
        sparseInput.parentAnatomicalRegionIndices=parentRegions;
        sparseInput.parentSourceDonorOffsets=donorOffsets;
        sparseInput.parentSourceDonorBodyIndices=donorIds;
        sparseInput.parentSourceDonorMassKg=donorMasses;
        sparseInput.childCentroidsWorld=childPoints;
        sparseInput.childReferenceVolumeM3=childVolumes;
        sparseInput.childMaterialIndices=childMaterials;
        sparseInput.childDonorBodyIndices=childDonorIds;
        sparseInput.childAnatomicalRegionIndices=childRegionIds;
        sparseInput.materialDensityLowerKgM3=exactLower;
        sparseInput.materialDensityUpperKgM3=exactUpper;
        sparseInput.parentMassPacketSHA256=h1;
        sparseInput.parentSourceContributionSHA256=h2;
        sparseInput.childMeshSHA256=h3;
        sparseInput.regionLabelSHA256=h4;
        sparseInput.coordinateFrameSHA256=h5;
        sparseInput.maximumTransportDistanceM=2.0;
        sparseInput.maximumAllowedEdges=100;
        sparseInput.maximumAugmentations=100;
        ConservativeSpatialChildSemanticSupport sparseSupport;
        sparseSupport.childOffsets=supportOffsets;
        sparseSupport.slotGroupIndices=supportGroups;
        sparseSupport.volumeFractions=supportFractions;
        sparseSupport.groupMaterialIndices=groupMaterial;
        sparseSupport.donorBodyIndices=groupDonor;
        sparseSupport.anatomicalRegionIndices=groupRegion;
        sparseSupport.derivedGroupIdentityIDs=groupIdentity;
        sparseSupport.parentStableSourceFaceIdentities=parentStable;
        sparseSupport.sourceParentTetOffsets=groupParentOffsets;
        sparseSupport.sourceParentTetRows=groupParentRows;
        sparseSupport.sourceStableFaceOffsets=groupFaceOffsets;
        sparseSupport.sourceStableFaceRows=groupFaces;
        sparseSupport.typedSupportSHA256=h5;
        ConservativeSpatialMassTransportResult expanded;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
                sparseInput,sparseSupport,expanded,error)||
           expanded.semanticCandidateEdgeCount!=9u||
           expanded.childMassKg!=std::vector<double>({1.0,1.0,1.0})||
           !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                sparseInput,sparseSupport,expanded,error)){
            std::cerr<<"cut-guided sparse expansion failed: "<<error<<"\n";return 17;
        }
        auto cappedInput=sparseInput;
        cappedInput.maximumAllowedEdges=4u;
        ConservativeSpatialMassTransportResult capped;
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(
                cappedInput,sparseSupport,capped,error)||
           error.find("INCONCLUSIVE shared-capacity sparse seed graph exceeds pinned edge budget") == std::string::npos||
           error.find("INFEASIBLE") != std::string::npos){
            std::cerr<<"sparse edge-budget failure was not classified as inconclusive: "<<error<<"\n";return 18;
        }
    }

    // The region adjacency CSR is necessary but does not invent a donor/region
    // tuple: a parent may flow only when its exact tuple is in the child's P1
    // tuple allowlist. This is the current native material/donor/region gate.
    {
        auto adjacentInput=in;
        auto unlistedGroup=support;
        static const std::array<std::uint32_t,2> oneRegionOffset{0,1};
        static const std::array<std::uint32_t,1> oneP1Group{0};
        static const std::array<double,1> oneP1Score{1.0};
        static const std::array<std::uint32_t,2> allowedOffsets{0,2};
        static const std::array<std::uint32_t,2> allowedRegions{10,20};
        static const std::string adjacencyHash(64,'9');
        unlistedGroup.childOffsets=oneRegionOffset;
        unlistedGroup.slotGroupIndices=oneP1Group;
        unlistedGroup.volumeFractions=oneP1Score;
        unlistedGroup.cellAdmissibleRegionOffsets=allowedOffsets;
        unlistedGroup.cellAdmissibleRegionIds=allowedRegions;
        unlistedGroup.admissibilitySourceSupportSHA256=adjacencyHash;
        unlistedGroup.admissibilityMaximumSourceRegionHops=1u;
        ConservativeSpatialMassTransportResult rejectedUnlisted;
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(
                adjacentInput,unlistedGroup,rejectedUnlisted,error)||
           error.find("same-material/donor/region exact-group-supported child") == std::string::npos||
           error.find("INFEASIBLE") == std::string::npos){
            std::cerr<<"one-hop region alone invented an unsupported donor/region tuple: "<<error<<"\n";return 19;
        }

        // Listing both exact tuples makes both source groups legal; the real
        // child still has one shared density interval and closes to 1 kg.
        auto listedGroups=support;
        listedGroups.cellAdmissibleRegionOffsets=allowedOffsets;
        listedGroups.cellAdmissibleRegionIds=allowedRegions;
        listedGroups.admissibilitySourceSupportSHA256=adjacencyHash;
        listedGroups.admissibilityMaximumSourceRegionHops=1u;
        ConservativeSpatialMassTransportResult adjacentResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
                adjacentInput,listedGroups,adjacentResult,error)||
           adjacentResult.childMassKg!=std::vector<double>({1.0})||
           adjacentResult.childImpliedDensityKgM3!=std::vector<double>({1000.0})||
           adjacentResult.groupClosure.size()!=2u||
           adjacentResult.groupClosure[0].childMassKg!=0.5||
           adjacentResult.groupClosure[1].childMassKg!=0.5||
           adjacentResult.childSemanticGroupOffsets!=std::vector<std::uint32_t>({0,2})||
           adjacentResult.childSemanticGroupIndices!=std::vector<std::uint32_t>({0,1})||
           !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                adjacentInput,listedGroups,adjacentResult,error)){
            std::cerr<<"exact-tuple one-hop support failed shared-capacity replay: "<<error<<"\n";return 20;
        }
        auto changedAdjacency=listedGroups;
        static const std::array<std::uint32_t,2> changedAllowedRegions{10,30};
        changedAdjacency.cellAdmissibleRegionIds=changedAllowedRegions;
        if(verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                adjacentInput,changedAdjacency,adjacentResult,error)){
            std::cerr<<"receipt accepted mutated one-hop support CSR\n";return 23;
        }
        auto missingP1=listedGroups;
        static const std::array<std::uint32_t,2> wrongAllowedRegions{20,30};
        missingP1.cellAdmissibleRegionIds=wrongAllowedRegions;
        if(!rejected(adjacentInput,missingP1,"adjacency missing original P1 region"))return 21;
        auto badHop=listedGroups;
        badHop.admissibilityMaximumSourceRegionHops=2u;
        if(!rejected(adjacentInput,badHop,"non-one-hop adjacency"))return 22;
        auto badOrder=listedGroups;
        static const std::array<std::uint32_t,2> unsortedAllowed{20,10};
        badOrder.cellAdmissibleRegionIds=unsortedAllowed;
        if(!rejected(adjacentInput,badOrder,"unsorted adjacency row"))return 24;
    }

    // A source-projection-backed exact tuple may extend a child beyond its
    // P1 preference slots. The extension is an allowlist only: it keeps one
    // shared real-child density interval and records the realized tuple flow.
    {
        auto projected=support;
        static const std::array<std::uint32_t,2> p1OnlyOffsets{0,1};
        static const std::array<std::uint32_t,1> p1OnlyGroup{0};
        static const std::array<double,1> p1OnlyScore{1.0};
        static const std::array<std::uint32_t,2> allowedGroupOffsets{0,2};
        static const std::array<std::uint32_t,2> allowedGroupIds{0,1};
        static const std::array<std::uint32_t,2> allowedRegionOffsets{0,2};
        static const std::array<std::uint32_t,2> allowedRegionIds{10,20};
        static const std::string adjacencyHash(64,'9');
        static const std::string projectionHash(64,'8');
        projected.childOffsets=p1OnlyOffsets;
        projected.slotGroupIndices=p1OnlyGroup;
        projected.volumeFractions=p1OnlyScore;
        projected.childAdmissibleGroupOffsets=allowedGroupOffsets;
        projected.childAdmissibleGroupIndices=allowedGroupIds;
        projected.admissibleGroupEvidenceSHA256=projectionHash;
        projected.cellAdmissibleRegionOffsets=allowedRegionOffsets;
        projected.cellAdmissibleRegionIds=allowedRegionIds;
        projected.admissibilitySourceSupportSHA256=adjacencyHash;
        projected.admissibilityMaximumSourceRegionHops=1u;
        ConservativeSpatialMassTransportResult projectedResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
                in,projected,projectedResult,error)||
           projectedResult.childMassKg!=std::vector<double>({1.0})||
           projectedResult.childImpliedDensityKgM3!=std::vector<double>({1000.0})||
           projectedResult.groupClosure.size()!=2u||
           projectedResult.groupClosure[0].childMassKg!=0.5||
           projectedResult.groupClosure[1].childMassKg!=0.5||
           projectedResult.childSemanticGroupOffsets!=std::vector<std::uint32_t>({0,2})||
           projectedResult.childSemanticGroupIndices!=std::vector<std::uint32_t>({0,1})||
           !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                in,projected,projectedResult,error)){
            std::cerr<<"authenticated exact-group extension failed shared-capacity replay: "<<error<<"\n";
            return 26;
        }
        auto missingProjectionProof=projected;
        missingProjectionProof.admissibleGroupEvidenceSHA256={};
        if(!rejected(in,missingProjectionProof,"exact-group extension without evidence"))return 27;
        auto missingExactTuple=projected;
        missingExactTuple.childAdmissibleGroupOffsets=p1OnlyOffsets;
        missingExactTuple.childAdmissibleGroupIndices=p1OnlyGroup;
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(
                in,missingExactTuple,projectedResult,error)||
           error.find("exact-group-supported child")==std::string::npos){
            std::cerr<<"P1 region/preference alone admitted the extra exact tuple: "<<error<<"\n";
            return 28;
        }
        auto outsideOneHop=projected;
        static const std::array<std::uint32_t,2> primaryRegionOffsets{0,1};
        static const std::array<std::uint32_t,1> primaryRegionIds{10};
        outsideOneHop.cellAdmissibleRegionOffsets=primaryRegionOffsets;
        outsideOneHop.cellAdmissibleRegionIds=primaryRegionIds;
        if(!rejected(in,outsideOneHop,"exact tuple outside authenticated one-hop region CSR"))return 29;
        auto tamperedProjection=projected;
        static const std::array<std::uint32_t,2> tamperedAllowed{0,0};
        tamperedProjection.childAdmissibleGroupIndices=tamperedAllowed;
        if(verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                in,tamperedProjection,projectedResult,error)){
            std::cerr<<"replay accepted mutated exact-group allowlist\n";return 30;
        }
    }

    // Same material and region are insufficient if the registered geometric
    // donor differs. The scalar child label remains the primary P1 tuple;
    // another exact tuple must be explicitly carried in its slot CSR.
    {
        auto sameRegionInput=in;
        static const std::array<std::uint32_t,2> parentRegionsSame{10,10};
        static const std::array<std::uint32_t,2> oneChildGroupOffsets{0,1};
        static const std::array<std::uint32_t,1> oneChildGroup{0};
        static const std::array<double,1> oneChildScore{1.0};
        static const std::array<std::uint32_t,2> sameRegionGroups{10,10};
        sameRegionInput.parentAnatomicalRegionIndices=parentRegionsSame;
        auto donorRestricted=support;
        donorRestricted.childOffsets=oneChildGroupOffsets;
        donorRestricted.slotGroupIndices=oneChildGroup;
        donorRestricted.volumeFractions=oneChildScore;
        donorRestricted.anatomicalRegionIndices=sameRegionGroups;
        ConservativeSpatialMassTransportResult shouldReject;
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(
                sameRegionInput,donorRestricted,shouldReject,error)||
           error.find("same-material/donor/region exact-group-supported child") == std::string::npos){
            std::cerr<<"same-region cross-donor source was accepted: "<<error<<"\n";return 25;
        }
    }

    // Tiny positive mass must survive: the production lower-capacity floor can
    // be far below the legacy 1e-18 kg augmentation epsilon.
    {
        static const std::array<SpatialMassTransportPoint,2> tinyParentPoints{{
            SpatialMassTransportPoint{0.0,0.0,0.0},SpatialMassTransportPoint{0.0,0.0,0.0}}};
        static const std::array<double,2> tinyParentVolumes{5.0e-26,5.0e-26};
        static const std::array<double,2> tinyParentMasses{5.0e-23,5.0e-23};
        static const std::array<std::uint32_t,2> tinyMaterials{15,15};
        static const std::array<std::uint32_t,2> tinyDonors{1,2};
        static const std::array<std::uint32_t,2> tinyRegions{10,20};
        static const std::array<std::uint32_t,3> tinyDonorOffsets{0,1,2};
        static const std::array<double,2> tinyDonorMasses{5.0e-23,5.0e-23};
        static const std::array<SpatialMassTransportPoint,1> tinyChildPoints{{
            SpatialMassTransportPoint{0.0,0.0,0.0}}};
        static const std::array<double,1> tinyChildVolumes{1.0e-25};
        static const std::array<std::uint32_t,1> tinyChildMaterial{15};
        static const std::array<std::uint32_t,1> tinyChildDonor{1};
        static const std::array<std::uint32_t,1> tinyChildRegion{10};
        static const std::array<double,16> tinyLower=[] {std::array<double,16>x{};x.fill(900.0);return x;}();
        static const std::array<double,16> tinyUpper=[] {std::array<double,16>x{};x.fill(1100.0);return x;}();
        static const std::string th1(64,'a'),th2(64,'b'),th3(64,'c'),th4(64,'d'),th5(64,'e');
        auto tiny=makeInput();
        tiny.parentCentroidsWorld=tinyParentPoints;tiny.parentReferenceVolumeM3=tinyParentVolumes;
        tiny.parentMassKg=tinyParentMasses;tiny.parentMaterialIndices=tinyMaterials;
        tiny.parentDonorBodyIndices=tinyDonors;tiny.parentAnatomicalRegionIndices=tinyRegions;
        tiny.parentSourceDonorOffsets=tinyDonorOffsets;tiny.parentSourceDonorBodyIndices=tinyDonors;
        tiny.parentSourceDonorMassKg=tinyDonorMasses;tiny.childCentroidsWorld=tinyChildPoints;
        tiny.childReferenceVolumeM3=tinyChildVolumes;tiny.childMaterialIndices=tinyChildMaterial;
        tiny.childDonorBodyIndices=tinyChildDonor;tiny.childAnatomicalRegionIndices=tinyChildRegion;
        tiny.materialDensityLowerKgM3=tinyLower;tiny.materialDensityUpperKgM3=tinyUpper;
        tiny.parentMassPacketSHA256=th1;tiny.parentSourceContributionSHA256=th2;
        tiny.childMeshSHA256=th3;tiny.regionLabelSHA256=th4;tiny.coordinateFrameSHA256=th5;
        tiny.maximumTransportDistanceM=0.1;tiny.maximumAllowedEdges=10;tiny.maximumAugmentations=100;
        ConservativeSpatialMassTransportResult tinyResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
                tiny,support,tinyResult,error)||tinyResult.childMassKg.size()!=1u||
           tinyResult.childMassKg[0]<=0.0||
           std::abs(tinyResult.childMassKg[0]-1.0e-22)>1.0e-36||
           std::abs(tinyResult.childImpliedDensityKgM3[0]-1000.0)>1.0e-10||
           !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                tiny,support,tinyResult,error)){
            std::cerr<<"sub-epsilon positive child mass was lost: "<<error<<"\n";return 24;
        }
    }

    auto overDensity = in;
    static const std::array<double,1> tooSmallChildVolume{8.0e-4};
    overDensity.childReferenceVolumeM3=tooSmallChildVolume;
    if (!rejected(overDensity, support, "real child density capacity")) return 16;

    auto missingCSR = support;
    static const std::array<std::uint32_t, 0> emptyOffsets{};
    missingCSR.sourceStableFaceOffsets = emptyOffsets;
    missingCSR.sourceStableFaceRows = emptyOffsets;
    if (!rejected(in, missingCSR, "missing stable-face CSR")) return 7;

    auto wrongDonor = support;
    static const std::array<std::uint32_t, 2> wrongDonors{1, 3};
    wrongDonor.donorBodyIndices = wrongDonors;
    if (!rejected(in, wrongDonor, "unregistered donor group")) return 8;

    auto wrongParent = support;
    static const std::array<std::uint32_t, 2> wrongParents{0, 0};
    wrongParent.sourceParentTetRows = wrongParents;
    if (!rejected(in, wrongParent, "wrong group parent-tet CSR")) return 9;

    auto wrongFace = support;
    static const std::array<std::uint32_t, 2> wrongFaces{501, 999};
    wrongFace.sourceStableFaceRows = wrongFaces;
    if (!rejected(in, wrongFace, "incomplete stable-face union")) return 10;

    auto outOfRangeParent = support;
    static const std::array<std::uint32_t, 2> badParentRows{0, 99};
    outOfRangeParent.sourceParentTetRows = badParentRows;
    if (!rejected(in, outOfRangeParent, "out-of-range source parent row")) return 14;

    auto outOfRangeGroup = support;
    static const std::array<std::uint32_t, 2> badGroupRows{0, 99};
    outOfRangeGroup.slotGroupIndices = badGroupRows;
    if (!rejected(in, outOfRangeGroup, "out-of-range semantic group row")) return 15;

    auto tooFar = in;
    static const std::array<SpatialMassTransportPoint, 2> farParent{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0},
        SpatialMassTransportPoint{0.2, 0.0, 0.0}}};
    tooFar.parentCentroidsWorld = farParent;
    if (!rejected(tooFar, support, "distance bound")) return 11;

    auto missingSupport = support;
    static const std::array<std::uint32_t, 0> noChildren{};
    missingSupport.childOffsets = noChildren;
    if (!rejected(in, missingSupport, "missing child CSR")) return 12;

    // The no-support legacy entrypoint remains usable and verifier-compatible.
    static const std::array<SpatialMassTransportPoint, 1> oneParentCentroid{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0}}};
    static const std::array<double, 1> oneParentVolume{1.0e-3};
    static const std::array<double, 1> oneParentMass{1.0};
    static const std::array<std::uint32_t, 1> oneMaterial{15};
    static const std::array<std::uint32_t, 1> oneDonor{1};
    static const std::array<std::uint32_t, 1> oneRegion{10};
    static const std::array<std::uint32_t, 2> oneSourceOffsets{0, 1};
    static const std::array<double, 1> oneSourceMass{1.0};
    auto legacy = in;
    legacy.parentCentroidsWorld = oneParentCentroid;
    legacy.parentReferenceVolumeM3 = oneParentVolume;
    legacy.parentMassKg = oneParentMass;
    legacy.parentMaterialIndices = oneMaterial;
    legacy.parentDonorBodyIndices = oneDonor;
    legacy.parentAnatomicalRegionIndices = oneRegion;
    legacy.parentSourceDonorOffsets = oneSourceOffsets;
    legacy.parentSourceDonorBodyIndices = oneDonor;
    legacy.parentSourceDonorMassKg = oneSourceMass;
    ConservativeSpatialMassTransportResult legacyResult;
    if (!assembleConservativeSpatialMassTransport(legacy, legacyResult, error) ||
        !verifyRecordedConservativeSpatialMassTransport(legacy, legacyResult, error) ||
        legacyResult.childMassKg.size() != 1 || legacyResult.childMassKg[0] != 1.0) {
        std::cerr << "legacy no-support transport changed: " << error << "\n";
        return 13;
    }
    std::cout << "shared real-child semantic capacity controls PASS\n";
    return 0;
}
