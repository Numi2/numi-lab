#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main unpaired_skin_fat_volume_original_main
#include "NumiHumanUnpairedSkinFatVolumeCheck.cpp"
#undef main
#pragma clang diagnostic pop

#include "NumiHumanPhysicalSkinReplacement.hpp"

#include <cassert>
#include <iostream>
#include <map>
#include <numeric>

namespace {

V3 exactFloatPoint(const V3& point) {
    return {double(static_cast<float>(point[0])),double(static_cast<float>(point[1])),
            double(static_cast<float>(point[2]))};
}

bool prepareFixtureBundle(Fixture& fixture,PhysicalSkinChildMassMappingBundle& bundle,
                          std::string& error) {
    const auto pointRows=[&](const std::vector<V3>& input) {
        std::vector<V3> output;output.reserve(input.size());
        for(const auto& point:input)output.push_back(exactFloatPoint(point));
        return output;
    };
    bundle.outerSkin=fixture.surface;
    bundle.outerSkin.worldPoints=pointRows(fixture.surface.worldPoints);
    bundle.outerSkin.donorLocalPoints=pointRows(fixture.surface.donorLocalPoints);
    bundle.boundaryMap=fixture.boundary;
    const auto layerSurface=[&](const std::vector<std::uint32_t>& volumeRows,
                                const char* role) {
        RegisteredSurface surface=fixture.surface;
        surface.name=std::string("registered ")+role+" fixture boundary";
        surface.worldPoints.clear();surface.donorLocalPoints.clear();
        surface.donorBodyIndices.clear();surface.worldToDonorLocalLinear.clear();
        surface.sourceVertexIndices.clear();surface.sourceDomainIndices.clear();
        surface.sourceEmbeddingFaceIndices.clear();surface.sourceEmbeddingBarycentric.clear();
        surface.sourceEmbeddingResidualWorldM.clear();
        for(const auto node:volumeRows) {
            surface.worldPoints.push_back(exactFloatPoint(fixture.current[node]));
            surface.donorLocalPoints.push_back(exactFloatPoint(fixture.localPoints[node]));
            surface.donorBodyIndices.push_back(0u);
            surface.worldToDonorLocalLinear.push_back(kIdentity);
        }
        return surface;
    };
    bundle.independentSkinInner=layerSurface(fixture.boundary.skinInner,"I");
    bundle.independentFatDeep=layerSurface(fixture.boundary.fatDeep,"D");

    auto& parent=bundle.parent;
    parent.currentWorldPoints=pointRows(fixture.parentPoints);
    parent.referenceWorldPoints=parent.currentWorldPoints;
    parent.tetrahedra=fixture.parentTets;
    parent.materialIndices=fixture.parentMaterial;
    parent.donorBodyIndices=fixture.parentDonor;
    parent.anatomicalRegionIndices=fixture.parentRegion;
    parent.stableSourceFaceIdentities=fixture.parentStable;
    parent.referenceTetVolumeM3.resize(parent.tetrahedra.size());
    parent.tetrahedronMassKg.resize(parent.tetrahedra.size());
    parent.compiledLumpedNodeMassKg.assign(parent.referenceWorldPoints.size(),0.0);
    parent.compiledGlobalNodeIndices.resize(parent.referenceWorldPoints.size());
    parent.nodeFrameOwnerBodyIndices.assign(parent.referenceWorldPoints.size(),0u);
    parent.nodeFrameOwnerLocalPoints.assign(parent.referenceWorldPoints.size(),V3{});
    std::iota(parent.compiledGlobalNodeIndices.begin(),parent.compiledGlobalNodeIndices.end(),100u);
    for(std::size_t node=0;node<parent.referenceWorldPoints.size();++node)
        parent.nodeFrameOwnerLocalPoints[node]=exactFloatPoint(fixture.localPoints[node]);
    std::map<std::pair<std::uint32_t,std::uint32_t>,double> sourceMass;
    for(std::size_t cell=0;cell<parent.tetrahedra.size();++cell) {
        const auto tet=parent.tetrahedra[cell];
        const double volume=tetVolume(parent.referenceWorldPoints[tet[0]],
            parent.referenceWorldPoints[tet[1]],parent.referenceWorldPoints[tet[2]],
            parent.referenceWorldPoints[tet[3]]);
        if(!std::isfinite(volume)||volume<=0.0){error="fixture parent has a nonpositive F32 cell";return false;}
        parent.referenceTetVolumeM3[cell]=volume;
        const auto material=parent.materialIndices[cell];
        const double density=material==0u?1000.0:900.0;
        const double mass=volume*density;
        parent.tetrahedronMassKg[cell]=mass;parent.totalMassKg+=mass;
        for(const auto node:tet) {
            sourceMass[{node,material}]+=mass*0.25;
            parent.compiledLumpedNodeMassKg[node]+=mass*0.25;
        }
    }
    for(const auto& [key,mass]:sourceMass) {
        const auto [node,material]=key;
        parent.sourceMassContributions.push_back({
            node,material,0u,mass,parent.nodeFrameOwnerLocalPoints[node]});
    }
    parent.parentMassPacketSHA256=std::string(64u,'1');
    parent.parentSourceContributionSHA256=std::string(64u,'2');
    parent.parentTopologySHA256=std::string(64u,'3');
    parent.parentLabelSHA256=std::string(64u,'4');

    bundle.currentWorldPoints=pointRows(fixture.current);
    bundle.referenceWorldPoints=pointRows(fixture.reference);
    bundle.childTetrahedra=fixture.children;
    bundle.childMaterialIndices=fixture.childMaterial;
    bundle.childDonorBodyIndices=fixture.childDonor;
    bundle.childAnatomicalRegionIndices=fixture.childRegion;
    bundle.childDominantParentTetrahedra=fixture.dominant;
    bundle.childParentSurfaceFaces=fixture.parentSurface;
    bundle.childStableSourceFaceIdentities=fixture.childStable;
    bundle.childRegisteredLayerFacetSurfaceFaces=fixture.facetRows;
    bundle.childRegisteredLayerFacetSourceFaceIdentities=fixture.facetStable;
    bundle.childAuthoredHeadSidewallFacetFlags=fixture.sidewallFlags;
    bundle.childAuthoredHeadSidewallStableFaceIdentities=fixture.sidewallStable;
    bundle.volumeNodeDonorBodyIndices=fixture.nodeDonor;
    bundle.volumeNodeDonorLocalPoints=pointRows(fixture.localPoints);
    bundle.volumeNodeDonorOriginWorld=pointRows(fixture.origins);
    bundle.volumeNodeDonorLocalToWorldLinear=fixture.nodeRotations;
    bundle.registeredBodyLocalToWorldLinear=fixture.bodyRotations;
    bundle.registeredBodyOriginWorld=pointRows(fixture.bodyOrigins);
    bundle.detailedSourceVertexCount=fixture.topology.detailedSourceVertexCount;
    bundle.detailedSourceFaceCount=fixture.topology.detailedSourceFaceCount;
    bundle.volumeNodeSourceVertexIndices=fixture.sourceVertex;
    bundle.volumeNodeSourceFaceIndices=fixture.sourceFace;
    bundle.volumeNodeSourceBarycentric=fixture.sourceBary;
    bundle.retainedOriginalNodeCount=fixture.topology.retainedOriginalNodeCount;
    bundle.volumeNodeParentTetrahedra=fixture.nodeParent;
    bundle.volumeNodeParentBarycentric=fixture.nodeParentBary;
    bundle.volumeNodeParentResidualWorldM.assign(bundle.referenceWorldPoints.size(),V3{});
    for(std::size_t node=0;node<bundle.volumeNodeParentTetrahedra.size();++node) {
        const auto parentRow=bundle.volumeNodeParentTetrahedra[node];
        if(parentRow==NM_INVALID_INDEX)continue;
        const auto tet=parent.tetrahedra[parentRow];
        for(unsigned axis=0;axis<3u;++axis) {
            double reconstructed=0.0;
            for(unsigned corner=0;corner<4u;++corner)
                reconstructed+=bundle.volumeNodeParentBarycentric[node][corner]*
                    parent.referenceWorldPoints[tet[corner]][axis];
            bundle.volumeNodeParentResidualWorldM[node][axis]=
                bundle.referenceWorldPoints[node][axis]-reconstructed;
        }
    }
    bundle.referenceSourceIdentity=fixture.topology.referenceSourceIdentity;
    bundle.materialSourceIdentity=fixture.topology.materialSourceIdentity;
    bundle.maximumSpatialMassTransportDistanceM=1.0;
    bundle.childMeshSHA256=std::string(64u,'5');
    bundle.regionLabelSHA256=std::string(64u,'6');
    bundle.coordinateFrameSHA256=std::string(64u,'7');
    const std::array<double,2> densityLower{900.0,850.0};
    const std::array<double,2> densityUpper{1100.0,950.0};
    return preparePhysicalSkinChildMassMapping(bundle,densityLower,densityUpper,
                                               1.0,8u,error);
}

} // namespace

int main() {
    using namespace metalrobo::human::coarse_tissue;

    Fixture fixture;
    auto world=worldFor();
    numi::matter::ObjectSource prefix;
    prefix.name="prefix FEM node";
    prefix.representation=Representation::fem;
    prefix.materialIndex=0u;
    prefix.femNodes.push_back({9.0,8.0,7.0});
    world.objects.push_back(prefix);

    ExplicitUnpairedVolumeReport oldReport;
    std::string error;
    PreparedOrganicFEM oldSkin,oldFat;
    assert(emit(fixture,world,oldReport,error,&oldSkin,&oldFat));
    world.objects.back().femReferenceDisplacementGradient=true;
    oldSkin.layerThicknessMinM=0.002;
    oldSkin.layerThicknessMaxM=0.003;
    oldSkin.layerThicknessProvenance="synthetic accepted skin range";
    oldSkin.layerThicknessBySurfaceVertexM.assign(fixture.surface.worldPoints.size(),0.0025);
    oldFat.layerThicknessMinM=0.001;
    oldFat.layerThicknessMaxM=0.002;
    oldFat.layerThicknessProvenance="synthetic accepted fat range";
    oldFat.layerThicknessBySurfaceVertexM.assign(fixture.surface.worldPoints.size(),0.0015);
    assert(world.objects.size()==2u);
    assert(oldSkin.firstGlobalFEMNode==1u&&oldFat.firstGlobalFEMNode==1u);
    const auto originalObject=world.objects.back();
    const auto originalPrefix=world.objects.front();

    // Retain the original raw-topology transaction as a strict regression
    // control; the prepared bundle test below exercises the new route wrapper.
    auto legacyRawWorld=world;
    PreparedOrganicFEM legacyRawSkin,legacyRawFat;
    ExplicitUnpairedVolumeReport legacyRawReport;
    RegisteredSkinFatReplacementAudit legacyRawAudit;
    assert(replaceLastRegisteredSkinFatUnpairedVolume(
        legacyRawWorld,1u,fixture.surface,fixture.boundary,fixture.topology,
        0u,"skin","test skin material",oldSkin.massDebitKg,
        1u,"fat","test fat material",oldFat.massDebitKg,
        1u,"unpaired test shell",oldSkin,oldFat,&oldReport,
        legacyRawSkin,legacyRawFat,legacyRawReport,legacyRawAudit,error));
    assert(legacyRawWorld.objects.size()==2u);
    assert(legacyRawWorld.objects.front().femNodes==originalPrefix.femNodes);
    assert(legacyRawWorld.objects.back().femNodes==originalObject.femNodes);
    assert(legacyRawWorld.objects.back().femReferenceNodes==originalObject.femReferenceNodes);
    assert(legacyRawWorld.objects.back().femMaterialIndices==originalObject.femMaterialIndices);
    assert(legacyRawWorld.objects.back().tetrahedra.size()==originalObject.tetrahedra.size());
    assert(legacyRawReport.volumeNodeCount==oldReport.volumeNodeCount);
    assert(legacyRawReport.tetrahedronCount==oldReport.tetrahedronCount);
    assert(legacyRawReport.skinChildMassKg==oldReport.skinChildMassKg);
    assert(legacyRawReport.fatChildMassKg==oldReport.fatChildMassKg);
    assert(legacyRawAudit.sourceDonorDeltas.size()==1u);
    assert(legacyRawAudit.sourceDonorDeltas.front().sourceMassDeltaKg==0.0);
    for(double value:legacyRawAudit.sourceDonorDeltas.front().sourceFirstMomentDeltaKgM)
        assert(value==0.0);

    PhysicalSkinChildMassMappingBundle identityBundle;
    if(!prepareFixtureBundle(fixture,identityBundle,error)) {
        std::cerr<<"positive prepared identity bundle rejected: "<<error<<'\n';return 2;
    }

    PreparedOrganicFEM newSkin,newFat;
    newSkin.sourceName="sentinel skin";
    newFat.sourceName="sentinel fat";
    ExplicitUnpairedVolumeReport newReport;
    newReport.volumeNodeCount=999u;
    RegisteredSkinFatReplacementAudit audit;
    audit.replacedObjectIndex=777u;

    if(!replaceLastPreparedRegisteredSkinFatUnpairedVolume(
        world,1u,identityBundle,
        0u,"skin","test skin material",oldSkin.massDebitKg,
        1u,"fat","test fat material",oldFat.massDebitKg,
        1u,"unpaired test shell",oldSkin,oldFat,&oldReport,
        newSkin,newFat,newReport,audit,error)) {
        std::cerr<<"prepared identity replacement rejected: "<<error<<'\n';return 3;
    }
    assert(error.empty());
    assert(world.objects.size()==2u);
    assert(world.objects.front().name==originalPrefix.name);
    assert(world.objects.front().femNodes==originalPrefix.femNodes);
    const auto& replaced=world.objects.back();
    assert(replaced.name==originalObject.name);
    assert(replaced.femNodes.size()==originalObject.femNodes.size());
    assert(replaced.femReferenceNodes.size()==originalObject.femReferenceNodes.size());
    for(std::size_t node=0;node<replaced.femNodes.size();++node) {
        assert(norm(sub(replaced.femNodes[node],originalObject.femNodes[node]))<1.0e-7);
        assert(norm(sub(replaced.femReferenceNodes[node],originalObject.femReferenceNodes[node]))<1.0e-7);
    }
    assert(replaced.femMaterialIndices==originalObject.femMaterialIndices);
    assert(replaced.femReferenceDisplacementGradient);
    assert(replaced.tetrahedra.size()==originalObject.tetrahedra.size());
    for(std::size_t cell=0;cell<replaced.tetrahedra.size();++cell)
        assert(replaced.tetrahedra[cell].nodes==originalObject.tetrahedra[cell].nodes);
    assert(newSkin.firstGlobalFEMNode==1u&&newFat.firstGlobalFEMNode==1u);
    assert(newReport.volumeNodeCount==oldReport.volumeNodeCount);
    assert(newReport.tetrahedronCount==oldReport.tetrahedronCount);
    assert(newReport.usesPerChildMassLedger);
    assert(std::abs(newReport.skinChildMassKg-oldSkin.massDebitKg)<
           1.0e-6*std::max(oldSkin.massDebitKg,1.0e-12));
    assert(std::abs(newReport.fatChildMassKg-oldFat.massDebitKg)<
           1.0e-6*std::max(oldFat.massDebitKg,1.0e-12));
    assert(audit.replacedObjectIndex==1u&&audit.firstGlobalFEMNode==1u);
    assert(audit.inheritedSkinThicknessMetadata&&audit.inheritedFatThicknessMetadata);
    assert(newSkin.layerThicknessMinM==oldSkin.layerThicknessMinM);
    assert(newSkin.layerThicknessMaxM==oldSkin.layerThicknessMaxM);
    assert(newFat.layerThicknessMinM==oldFat.layerThicknessMinM);
    assert(newFat.layerThicknessMaxM==oldFat.layerThicknessMaxM);
    assert(newSkin.layerThicknessBySurfaceVertexM==oldSkin.layerThicknessBySurfaceVertexM);
    assert(newFat.layerThicknessBySurfaceVertexM==oldFat.layerThicknessBySurfaceVertexM);
    assert(newSkin.layerThicknessProvenance.find("not remeasured")!=std::string::npos);
    assert(audit.oldReferenceSourceIdentity==audit.newReferenceSourceIdentity);
    assert(audit.oldMaterialSourceIdentity==audit.newMaterialSourceIdentity);
    assert(audit.sourceDonorDeltas.size()==1u);
    assert(audit.sourceDonorDeltas.front().donorBodyIndex==0u);
    assert(std::abs(audit.sourceDonorDeltas.front().sourceMassDeltaKg)<
           1.0e-6*std::max(audit.sourceDonorDeltas.front().oldSourceMassKg,1.0e-12));
    for(double value:audit.sourceDonorDeltas.front().sourceFirstMomentDeltaKgM)
        assert(std::abs(value)<1.0e-7);

    auto staleAttachmentBundle=identityBundle;
    staleAttachmentBundle.stableDeepAttachments.push_back(
        {1u,0u,0u,0u,fixture.boundary.fatDeep.front()});
    auto staleAttachmentWorld=world;
    PreparedOrganicFEM staleAttachmentSkin,staleAttachmentFat;
    ExplicitUnpairedVolumeReport staleAttachmentReport;
    RegisteredSkinFatReplacementAudit staleAttachmentAudit;
    const auto staleAttachmentObject=staleAttachmentWorld.objects.back();
    assert(!replaceLastPreparedRegisteredSkinFatUnpairedVolume(
        staleAttachmentWorld,1u,staleAttachmentBundle,
        0u,"skin","test skin material",newSkin.massDebitKg,
        1u,"fat","test fat material",newFat.massDebitKg,
        1u,"unpaired test shell",newSkin,newFat,&newReport,
        staleAttachmentSkin,staleAttachmentFat,staleAttachmentReport,
        staleAttachmentAudit,error));
    assert(error.find("path receipt")!=std::string::npos);
    assert(staleAttachmentWorld.objects.back().femNodes==staleAttachmentObject.femNodes);

    auto staleCellLabelBundle=identityBundle;
    ++staleCellLabelBundle.childAnatomicalRegionIndices.front();
    auto staleCellLabelWorld=world;
    const auto staleCellLabelObject=staleCellLabelWorld.objects.back();
    PreparedOrganicFEM staleCellLabelSkin,staleCellLabelFat;
    ExplicitUnpairedVolumeReport staleCellLabelReport;
    RegisteredSkinFatReplacementAudit staleCellLabelAudit;
    assert(!replaceLastPreparedRegisteredSkinFatUnpairedVolume(
        staleCellLabelWorld,1u,staleCellLabelBundle,
        0u,"skin","test skin material",newSkin.massDebitKg,
        1u,"fat","test fat material",newFat.massDebitKg,
        1u,"unpaired test shell",newSkin,newFat,&newReport,
        staleCellLabelSkin,staleCellLabelFat,staleCellLabelReport,
        staleCellLabelAudit,error));
    assert(error.find("path receipt")!=std::string::npos);
    assert(staleCellLabelWorld.objects.back().femNodes==staleCellLabelObject.femNodes);

    const auto unchangedWorld=world;
    auto rebasedWorld=unchangedWorld;
    auto rebasedTopology=fixture.topology;
    rebasedTopology.referenceSourceIdentity={11u,12u,13u,14u};
    rebasedTopology.materialSourceIdentity={15u,16u,17u,18u};
    PreparedOrganicFEM rebasedSkin,rebasedFat;
    ExplicitUnpairedVolumeReport rebasedReport;
    RegisteredSkinFatReplacementAudit rebasedAudit;
    assert(replaceLastRegisteredSkinFatUnpairedVolume(
        rebasedWorld,1u,fixture.surface,fixture.boundary,rebasedTopology,
        0u,"skin","test skin material",oldSkin.massDebitKg,
        1u,"fat","test fat material",oldFat.massDebitKg,
        1u,"unpaired test shell",oldSkin,oldFat,&oldReport,
        rebasedSkin,rebasedFat,rebasedReport,rebasedAudit,error));
    assert(rebasedAudit.oldReferenceSourceIdentity==originalObject.femReferenceSourceIdentity);
    assert(rebasedAudit.newReferenceSourceIdentity==rebasedTopology.referenceSourceIdentity);
    assert(rebasedAudit.oldMaterialSourceIdentity==originalObject.femMaterialSourceIdentity);
    assert(rebasedAudit.newMaterialSourceIdentity==rebasedTopology.materialSourceIdentity);
    assert(rebasedWorld.objects.back().femReferenceSourceIdentity==rebasedTopology.referenceSourceIdentity);
    assert(rebasedWorld.objects.back().femMaterialSourceIdentity==rebasedTopology.materialSourceIdentity);

    PreparedOrganicFEM sentinelSkin,sentinelFat;
    sentinelSkin.sourceName="unchanged-skin";
    sentinelFat.sourceName="unchanged-fat";
    ExplicitUnpairedVolumeReport sentinelReport;
    sentinelReport.volumeNodeCount=321u;
    RegisteredSkinFatReplacementAudit sentinelAudit;
    sentinelAudit.replacedObjectIndex=123u;
    assert(!replaceLastRegisteredSkinFatUnpairedVolume(
        world,0u,fixture.surface,fixture.boundary,fixture.topology,
        0u,"skin","test skin material",newSkin.massDebitKg,
        1u,"fat","test fat material",newFat.massDebitKg,
        1u,"unpaired test shell",newSkin,newFat,&newReport,
        sentinelSkin,sentinelFat,sentinelReport,sentinelAudit,error));
    assert(error.find("final source object")!=std::string::npos);
    assert(world.objects.size()==unchangedWorld.objects.size());
    assert(world.objects.back().femNodes==unchangedWorld.objects.back().femNodes);
    assert(sentinelSkin.sourceName=="unchanged-skin");
    assert(sentinelFat.sourceName=="unchanged-fat");
    assert(sentinelReport.volumeNodeCount==321u);
    assert(sentinelAudit.replacedObjectIndex==123u);

    auto nonFEMWorld=world;
    nonFEMWorld.objects.back().representation=Representation::mpm;
    assert(!replaceLastRegisteredSkinFatUnpairedVolume(
        nonFEMWorld,1u,fixture.surface,fixture.boundary,fixture.topology,
        0u,"skin","test skin material",newSkin.massDebitKg,
        1u,"fat","test fat material",newFat.massDebitKg,
        1u,"unpaired test shell",newSkin,newFat,&newReport,
        sentinelSkin,sentinelFat,sentinelReport,sentinelAudit,error));
    assert(error.find("not an FEM")!=std::string::npos);

    auto badDebitWorld=world;
    const auto beforeBadDebit=badDebitWorld.objects.back().femNodes;
    assert(!replaceLastRegisteredSkinFatUnpairedVolume(
        badDebitWorld,1u,fixture.surface,fixture.boundary,fixture.topology,
        0u,"skin","test skin material",newSkin.massDebitKg*1.01,
        1u,"fat","test fat material",newFat.massDebitKg,
        1u,"unpaired test shell",newSkin,newFat,&newReport,
        sentinelSkin,sentinelFat,sentinelReport,sentinelAudit,error));
    assert(error.find("debit")!=std::string::npos);
    assert(badDebitWorld.objects.back().femNodes==beforeBadDebit);

    auto failedBuilderWorld=world;
    auto invalidTopology=fixture.topology;
    invalidTopology.currentWorldPoints={};
    const auto beforeFailedBuilder=failedBuilderWorld.objects.back().femNodes;
    assert(!replaceLastRegisteredSkinFatUnpairedVolume(
        failedBuilderWorld,1u,fixture.surface,fixture.boundary,invalidTopology,
        0u,"skin","test skin material",newSkin.massDebitKg,
        1u,"fat","test fat material",newFat.massDebitKg,
        1u,"unpaired test shell",newSkin,newFat,&newReport,
        sentinelSkin,sentinelFat,sentinelReport,sentinelAudit,error));
    assert(error.find("builder")!=std::string::npos);
    assert(failedBuilderWorld.objects.back().femNodes==beforeFailedBuilder);

    std::cout<<"transactional registered skin/fat replacement check passed\n";
    return 0;
}

