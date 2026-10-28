#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main unpaired_skin_fat_volume_original_main
#include "NumiHumanUnpairedSkinFatVolumeCheck.cpp"
#undef main
#pragma clang diagnostic pop

#include "NumiHumanPhysicalSkinReplacement.hpp"

#include <cassert>
#include <iostream>

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

    PreparedOrganicFEM newSkin,newFat;
    newSkin.sourceName="sentinel skin";
    newFat.sourceName="sentinel fat";
    ExplicitUnpairedVolumeReport newReport;
    newReport.volumeNodeCount=999u;
    RegisteredSkinFatReplacementAudit audit;
    audit.replacedObjectIndex=777u;

    assert(replaceLastRegisteredSkinFatUnpairedVolume(
        world,1u,fixture.surface,fixture.boundary,fixture.topology,
        0u,"skin","test skin material",oldSkin.massDebitKg,
        1u,"fat","test fat material",oldFat.massDebitKg,
        1u,"unpaired test shell",oldSkin,oldFat,&oldReport,
        newSkin,newFat,newReport,audit,error));
    assert(error.empty());
    assert(world.objects.size()==2u);
    assert(world.objects.front().name==originalPrefix.name);
    assert(world.objects.front().femNodes==originalPrefix.femNodes);
    const auto& replaced=world.objects.back();
    assert(replaced.name==originalObject.name);
    assert(replaced.femNodes==originalObject.femNodes);
    assert(replaced.femReferenceNodes==originalObject.femReferenceNodes);
    assert(replaced.femMaterialIndices==originalObject.femMaterialIndices);
    assert(replaced.tetrahedra.size()==originalObject.tetrahedra.size());
    for(std::size_t cell=0;cell<replaced.tetrahedra.size();++cell)
        assert(replaced.tetrahedra[cell].nodes==originalObject.tetrahedra[cell].nodes);
    assert(newSkin.firstGlobalFEMNode==1u&&newFat.firstGlobalFEMNode==1u);
    assert(newReport.volumeNodeCount==oldReport.volumeNodeCount);
    assert(newReport.tetrahedronCount==oldReport.tetrahedronCount);
    assert(std::abs(newReport.skinChildMassKg-oldReport.skinChildMassKg)<1.0e-12);
    assert(std::abs(newReport.fatChildMassKg-oldReport.fatChildMassKg)<1.0e-12);
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
    assert(std::abs(audit.sourceDonorDeltas.front().sourceMassDeltaKg)<1.0e-12);
    for(double value:audit.sourceDonorDeltas.front().sourceFirstMomentDeltaKgM)
        assert(std::abs(value)<1.0e-12);

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

