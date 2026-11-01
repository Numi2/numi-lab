#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include <array>
#include <iostream>
#include <string>

using namespace metalrobo::human::coarse_tissue;

static ConservativeSpatialChildSemanticSupportData role(
        std::uint32_t material,std::uint32_t donor,std::uint32_t region,
        std::uint32_t parentRow,std::uint32_t stableFace,
        std::string supportHash,std::string typedHash) {
    ConservativeSpatialChildSemanticSupportData value;
    value.childOffsets={0u,1u};
    value.slotGroupIndices={0u};
    value.volumeFractions={1.0};
    value.groupMaterialIndices={material};
    value.donorBodyIndices={donor};
    value.anatomicalRegionIndices={region};
    value.derivedGroupIdentityIDs={1u};
    value.sourceParentTetOffsets={0u,1u};
    value.sourceParentTetRows={parentRow};
    value.sourceStableFaceOffsets={0u,1u};
    value.sourceStableFaceRows={stableFace};
    value.parentStableSourceFaceIdentities={stableFace};
    value.cellAdmissibleRegionOffsets={0u,1u};
    value.cellAdmissibleRegionIds={region};
    value.admissibilitySourceSupportSHA256=std::move(supportHash);
    value.admissibilityMaximumSourceRegionHops=1u;
    value.typedSupportSHA256=std::move(typedHash);
    return value;
}

static PhysicalSkinParentMassPacket parentPacket() {
    PhysicalSkinParentMassPacket value;
    value.tetrahedra={{{0u,1u,2u,3u}},{{4u,5u,6u,7u}}};
    value.materialIndices={15u,16u};
    value.donorBodyIndices={7u,8u};
    value.anatomicalRegionIndices={128u,129u};
    value.stableSourceFaceIdentities={42u,43u};
    value.parentMassPacketSHA256=std::string(64,'c');
    return value;
}

int main() {
    auto skin=role(15u,7u,128u,0u,42u,std::string(64,'a'),std::string(64,'d'));
    auto fat=role(16u,8u,129u,1u,43u,std::string(64,'b'),std::string(64,'e'));
    const std::array<std::uint32_t,1> skinRows{0u},fatRows{1u};
    const std::array<std::uint32_t,2> childMaterials{15u,16u};
    auto parent=parentPacket();
    ConservativeSpatialChildSemanticSupportData merged;
    std::string error;
    if(!mergeCurrentPairedP1SemanticSupport(skin,fat,skinRows,fatRows,
            childMaterials,parent,merged,error)) {
        std::cerr<<"distinct role-local P1 base hashes should merge: "<<error<<"\n";
        return 1;
    }
    spatial_mass_transport_detail::SHA256Builder expectedHash;
    expectedHash.string("numi.human.current-paired-source-region-support.v1");
    expectedHash.string(skin.admissibilitySourceSupportSHA256);
    expectedHash.string(fat.admissibilitySourceSupportSHA256);
    if(merged.admissibilitySourceSupportSHA256!=expectedHash.finish()||
       merged.admissibilityMaximumSourceRegionHops!=1u||
       merged.childOffsets!=std::vector<std::uint32_t>({0u,1u,2u})||
       merged.slotGroupIndices.size()!=2u||
       merged.groupMaterialIndices.size()!=2u||
       merged.groupMaterialIndices[merged.slotGroupIndices[0]]!=15u||
       merged.groupMaterialIndices[merged.slotGroupIndices[1]]!=16u) {
        std::cerr<<"paired support lost role-local hashes, hop policy, or child/group row mapping\n";
        return 2;
    }
    auto badFat=fat;badFat.admissibilityMaximumSourceRegionHops=2u;
    ConservativeSpatialChildSemanticSupportData rejected;
    if(mergeCurrentPairedP1SemanticSupport(skin,badFat,skinRows,fatRows,
            childMaterials,parent,rejected,error)) {
        std::cerr<<"paired support accepted a non-one-hop role\n";
        return 3;
    }
    badFat=fat;badFat.admissibilitySourceSupportSHA256.clear();
    if(mergeCurrentPairedP1SemanticSupport(skin,badFat,skinRows,fatRows,
            childMaterials,parent,rejected,error)) {
        std::cerr<<"paired support accepted an unpinned role-local source-support hash\n";
        return 4;
    }
    std::cout<<"PASS paired current P1 merge preserves distinct role base hashes and one-hop controls\n";
    return 0;
}
