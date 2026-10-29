#include "NumiHumanPhysicalSkinAuthoringRoleConsumer.hpp"

#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

namespace {
struct Fixture {
    PhysicalSkinChildMassMappingBundle bundle;
    PhysicalSkinRegisteredSurfaceCSVSet registered;
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    std::vector<PhysicalSkinRegisteredBodyPose> poses;
    std::array<PhysicalSkinAuthoringFullWeightsCSV,3> weightFiles;
    std::filesystem::path directory;

    Fixture() {
        directory=std::filesystem::temp_directory_path()/"numi-authoring-role-consumer-check";
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
        constexpr std::size_t roleRows=3u, volumeNodes=9u, bodyCount=157u;
        constexpr std::uint32_t parentSemanticDonor=150u;
        constexpr std::array<PhysicalSkinSourceDomainRole,3> roles{
            PhysicalSkinSourceDomainRole::outerSurface,
            PhysicalSkinSourceDomainRole::innerSurface,
            PhysicalSkinSourceDomainRole::deepSurface};
        constexpr std::array<const char*,3> names{"derived-O-fixture","derived-I-fixture","derived-D-fixture"};
        constexpr std::array<unsigned,3> bindingLanes{7u,8u,9u};
        constexpr std::array<std::uint32_t,3> rowOffsets{0u,3u,6u};
        constexpr std::array<double,3> zValues{2.0,1.0,0.0};

        bundle.boundaryMap.volumeNodeCount=static_cast<std::uint32_t>(volumeNodes);
        bundle.boundaryMap.outer={0u,1u,2u};
        bundle.boundaryMap.skinInner={3u,4u,5u};
        bundle.boundaryMap.fatDeep={6u,7u,8u};
        bundle.currentWorldPoints.resize(volumeNodes);
        bundle.referenceWorldPoints.resize(volumeNodes);
        bundle.volumeNodeDonorBodyIndices.resize(volumeNodes);
        bundle.volumeNodeSourceDomainIndices.resize(volumeNodes);
        bundle.volumeNodeSourceVertexIndices.resize(volumeNodes);
        bundle.volumeNodeSourceFaceIndices.assign(volumeNodes,NM_INVALID_INDEX);
        bundle.volumeNodeParentTetrahedra.assign(volumeNodes,0u);
        bundle.parent.tetrahedra={{{0u,1u,2u,3u}}};
        bundle.parent.donorBodyIndices={parentSemanticDonor};
        bundle.parent.materialIndices={15u};
        bundle.parent.anatomicalRegionIndices={42u};
        bundle.parent.stableSourceFaceIdentities={98586u};
        sidecars.currentParentRows.resize(volumeNodes);

        poses.resize(bodyCount);
        registered.framesByDonor.resize(bodyCount);
        for(std::size_t lane=0;lane<kPhysicalSkinAuthoringFullWeightBindingCount;++lane) {
            const std::size_t body=10u+lane;
            auto& pose=poses[body];
            pose.bodyIndex=static_cast<std::uint32_t>(body);
            pose.binding=static_cast<std::uint32_t>(lane);
            pose.available=true;
            pose.currentOrigin={0.1*double(body),-0.03*double(body),0.02*double(body)};
            pose.currentLocalToWorld={0.0,-1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0};
            pose.restOrigin=pose.currentOrigin;
            pose.restLocalToWorld=pose.currentLocalToWorld;
            auto& frame=registered.framesByDonor[body];
            frame.available=true;
            frame.binding=static_cast<std::uint32_t>(lane);
            frame.currentOrigin=pose.currentOrigin;
            frame.currentR=pose.currentLocalToWorld;
            frame.restOrigin=pose.restOrigin;
            frame.restR=pose.restLocalToWorld;
        }

        for(unsigned layer=0u;layer<3u;++layer) {
            const std::string domainName=names[layer];
            const std::string domainSHA(64u,char('a'+layer));
            const auto domainIndex=static_cast<std::uint32_t>(bundle.sourceDomains.size());
            bundle.sourceDomains.push_back({domainName,roles[layer],domainSHA,roleRows,1u});
            auto& surface=registered.surfaces[layer];
            surface.name=std::string(1,"OID"[layer]);
            surface.provider="fixture";
            surface.sourceMember="derived role";
            surface.sourceLayer=layer;
            surface.faces={{0u,1u,2u}};
            surface.sourceDomainIndices.assign(roleRows,domainIndex);
            surface.sourceVertexIndices={0u,1u,2u};
            surface.sourceEmbeddingFaceIndices.assign(roleRows,NM_INVALID_INDEX);
            surface.sourceEmbeddingBarycentric.resize(roleRows);
            surface.sourceEmbeddingResidualWorldM.resize(roleRows);
            surface.donorBodyIndices.resize(roleRows);
            surface.worldToDonorLocalLinear.resize(roleRows);
            registered.nodeSource[layer].resize(roleRows);
            for(unsigned row=0u;row<roleRows;++row) {
                const auto volumeNode=rowOffsets[layer]+row;
                const auto body=static_cast<std::uint32_t>(10u+bindingLanes[layer]);
                const auto& pose=poses[body];
                const V3 world{0.01*double(row),0.02*double(row),zValues[layer]+0.01*double(row)};
                surface.worldPoints.push_back(world);
                surface.donorLocalPoints.push_back(
                    physical_skin_authoring_role_consumer_detail::expectedDonorLocal(pose,world));
                surface.donorBodyIndices[row]=body;
                surface.worldToDonorLocalLinear[row]=
                    physical_skin_authoring_role_consumer_detail::expectedWorldToDonorLocal(pose);
                bundle.currentWorldPoints[volumeNode]=world;
                bundle.referenceWorldPoints[volumeNode]=world;
                bundle.volumeNodeDonorBodyIndices[volumeNode]=body;
                bundle.volumeNodeSourceDomainIndices[volumeNode]=domainIndex;
                bundle.volumeNodeSourceVertexIndices[volumeNode]=row;
                auto& source=registered.nodeSource[layer][row];
                source.domain=domainName;
                source.vertex=row;
                source.face=-1;
                auto& parent=sidecars.currentParentRows[volumeNode];
                parent.available=true;
                parent.tet=0u;
                parent.bary={0.25,0.25,0.25,0.25};
            }
            const auto path=directory/(domainName+"-weights.csv");
            weightFiles[layer].path=path;
            std::ofstream csv(path,std::ios::binary);
            const auto header=physical_skin_authoring_role_consumer_detail::weightHeader();
            for(std::size_t column=0;column<header.size();++column)csv<<(column?",":"")<<header[column];
            csv<<'\n';
            for(unsigned row=0u;row<roleRows;++row) {
                csv<<row;
                for(std::size_t lane=0;lane<kPhysicalSkinAuthoringFullWeightBindingCount;++lane)
                    csv<<','<<(lane==bindingLanes[layer]?"1":"0");
                csv<<'\n';
            }
            csv.close();
            std::string digest,error;
            assert(physical_skin_child_mass_detail::sha256File(path,digest,error));
            weightFiles[layer].expectedSHA256=digest;
        }
        bundle.outerSkin=registered.surfaces[0];
        bundle.independentSkinInner=registered.surfaces[1];
        bundle.independentFatDeep=registered.surfaces[2];
        assert(bundle.sourceDomains.size()==3u);
        assert(bundle.volumeNodeSourceFaceIndices.size()==volumeNodes);
    }

    ~Fixture() { std::filesystem::remove_all(directory); }
};

bool prepare(Fixture& f,PhysicalSkinAuthoringRoleConsumerState& state,
             PhysicalSkinAuthoringRoleConsumerReport& report,std::string& error) {
    return preparePhysicalSkinAuthoringRoleConsumerState(f.bundle,f.registered,f.sidecars,
        f.poses,f.weightFiles,state,report,error);
}


PhysicalSkinAuthoringDetailBridgeInput bridgeFixture() {
    PhysicalSkinAuthoringDetailBridgeInput in;
    in.detailDomain={"derived-detail-O",PhysicalSkinSourceDomainRole::outerSurface,
        std::string(64u,'a'),3u,1u};
    in.physicalBoundaryDomain={"reduced-physical-O",PhysicalSkinSourceDomainRole::outerSurface,
        std::string(64u,'b'),3u,1u};
    in.detailReferenceWorldPoints={{0.2,0.2,-0.005},{0.6,0.2,-0.005},{0.2,0.6,-0.005}};
    in.detailSourceVertexRows={0u,1u,2u};
    in.detailGeometricDonorBodyIndices={41u,41u,41u};
    in.detailMaterialIndices={15u,15u,15u};
    in.detailAnatomicalRegionIndices={7u,7u,7u};
    in.detailFrameOwnerBodyIndices={77u,77u,77u};
    in.detailBoundaryProjectionDonorBodyIndices={41u,41u,41u};
    in.detailRegionScopedDonorFallback={0u,0u,0u};
    in.detailParentTetRows={0u,0u,0u};
    in.detailStableSourceFaceIDs={9001u,9001u,9001u};
    in.detailFaces={{0u,1u,2u}};
    in.physicalBoundaryReferenceWorldPoints={{0,0,0},{1,0,0},{0,1,0}};
    in.physicalBoundarySourceVertexRows={0u,1u,2u};
    in.physicalBoundaryGeometricDonorBodyIndices={41u,41u,41u};
    in.physicalBoundaryFrameOwnerBodyIndices={88u,88u,88u};
    in.physicalBoundaryVolumeNodeRows={0u,1u,2u};
    in.physicalBoundaryFaces={{0u,1u,2u}};
    in.physicalBoundaryStableFaceIDs={9001u};
    in.physicalBoundaryFaceMaterialIndices={15u};
    in.physicalBoundaryFaceGeometricDonorBodyIndices={41u};
    in.physicalBoundaryFaceAnatomicalRegionIndices={7u};
    in.physicalBoundaryFaceFrameOwnerBodyIndices={89u};
    in.physicalBoundaryFaceParentTetRows={0u};
    in.detailRepresentativeBoundaryVertexRows={0u,0u,0u};
    in.detailExpectedBoundaryFaceRows={0u,0u,0u};
    in.detailExpectedBoundaryFaceBarycentrics={{{0.6,0.2,0.2}},{{0.2,0.6,0.2}},{{0.2,0.2,0.6}}};
    in.mechanicalReferenceWorldPoints={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    in.mechanicalTetNodeRows={{{0u,1u,2u,3u}}};
    in.mechanicalTetMaterialIndices={15u};
    in.mechanicalTetGeometricDonorBodyIndices={41u};
    in.mechanicalTetAnatomicalRegionIndices={7u};
    in.detailMechanicalTetRows={0u,0u,0u};
    in.maximumBoundaryProjectionDistanceM=0.01;
    in.maximumTetExtrapolationDistanceM=0.01;
    in.maximumAbsoluteEffectiveWeight=1.5;
    in.maximumReferenceReconstructionErrorM=1.0e-6;
    return in;
}

bool nearPoint(const V3& a,const V3& b,double tolerance=2.0e-6) {
    return physical_skin_authoring_detail_bridge_detail::distance(a,b)<=tolerance;
}

void testAuthoringDetailBridge() {
    auto in=bridgeFixture();
    PhysicalSkinAuthoringDetailBridgeState state;
    PhysicalSkinAuthoringDetailBridgeReport report;
    std::string error;
    assert(preparePhysicalSkinAuthoringDetailBridge(in,state,report,error));
    assert(!state.physicalAdmission&&!report.physicalAdmission);
    assert(!state.provenDirectIdentity&&report.sourceNamespaceIdentitySuppressed);
    assert(report.surfaceExactAncestryRows==0u&&report.surfaceProjectedRows==3u);
    assert(report.tetraEmbeddedRows==3u&&report.directIdentityRows==0u);
    assert(report.maximumBoundaryProjectionDistanceM<=in.maximumBoundaryProjectionDistanceM);
    assert(state.detailGeometricDonorBodyIndices==in.detailGeometricDonorBodyIndices);
    assert(state.detailFrameOwnerBodyIndices==in.detailFrameOwnerBodyIndices);
    assert(state.selectedPhysicalBoundaryFaceDonorBodyIndices==
           std::vector<std::uint32_t>({41u,41u,41u}));
    assert(state.selectedPhysicalBoundaryFaceFrameOwnerBodyIndices==
           std::vector<std::uint32_t>({89u,89u,89u}));
    for(const auto& row:state.tetraRows){
        double sum=0.0;for(float w:row.weight)sum+=w;
        assert(std::abs(sum-1.0)<2.0e-6);
        assert(row.weight[3]<0.0f); // nonzero outward surface offset is retained in tet coordinates
    }

    // Zero displacement and translation preserve the detailed offset.
    std::vector<V3> result;
    assert(evaluatePhysicalSkinAuthoringDetailBridgeForCheck(
        state,in.mechanicalReferenceWorldPoints,result,error));
    assert(result.size()==in.detailReferenceWorldPoints.size());
    for(std::size_t i=0;i<result.size();++i)assert(nearPoint(result[i],in.detailReferenceWorldPoints[i]));
    const V3 translation{0.31,-0.17,0.23};
    auto translated=in.mechanicalReferenceWorldPoints;
    for(auto& p:translated)p=add(p,translation);
    assert(evaluatePhysicalSkinAuthoringDetailBridgeForCheck(state,translated,result,error));
    for(std::size_t i=0;i<result.size();++i)
        assert(nearPoint(result[i],add(in.detailReferenceWorldPoints[i],translation)));

    // A nonzero projection residual must rotate with a rigidly rotated tetra.
    auto rotated=in.mechanicalReferenceWorldPoints;
    for(auto& p:rotated){const V3 q=p; p={-q[1]+0.4,q[0]-0.2,q[2]+0.1};}
    assert(evaluatePhysicalSkinAuthoringDetailBridgeForCheck(state,rotated,result,error));
    for(std::size_t i=0;i<result.size();++i){const auto p=in.detailReferenceWorldPoints[i];
        assert(nearPoint(result[i],{-p[1]+0.4,p[0]-0.2,p[2]+0.1}));}

    // General affine patch motion is reproduced by the same four accepted nodes.
    auto affine=in.mechanicalReferenceWorldPoints;
    for(auto& p:affine){const V3 q=p;
        p={1.2*q[0]+0.1*q[1]+0.05*q[2]+0.03,
           -0.04*q[0]+0.8*q[1]+0.12*q[2]-0.02,
           0.03*q[0]+0.07*q[1]+1.1*q[2]+0.11};}
    assert(evaluatePhysicalSkinAuthoringDetailBridgeForCheck(state,affine,result,error));
    for(std::size_t i=0;i<result.size();++i){const auto p=in.detailReferenceWorldPoints[i];
        assert(nearPoint(result[i],{1.2*p[0]+0.1*p[1]+0.05*p[2]+0.03,
            -0.04*p[0]+0.8*p[1]+0.12*p[2]-0.02,
            0.03*p[0]+0.07*p[1]+1.1*p[2]+0.11}));}

    // Same numeric source IDs do not enable direct source ancestry across two
    // pinned namespaces; the only accepted rows are the bounded projections.
    assert(state.faceTransferRows.size()==3u);
    for(const auto& row:state.faceTransferRows)
        assert(row.restDistance>0.0f);
    auto sameNamespaceDifferentGeometry=in;
    sameNamespaceDifferentGeometry.physicalBoundaryDomain=
        sameNamespaceDifferentGeometry.detailDomain;
    PhysicalSkinAuthoringDetailBridgeState collisionState;
    PhysicalSkinAuthoringDetailBridgeReport collisionReport;
    assert(preparePhysicalSkinAuthoringDetailBridge(
        sameNamespaceDifferentGeometry,collisionState,collisionReport,error));
    assert(!collisionState.provenDirectIdentity&&
           collisionReport.surfaceExactAncestryRows==0u&&
           collisionReport.sourceNamespaceIdentitySuppressed);

    // A truly identical complete domain may use direct one-hot rows.
    auto direct=bridgeFixture();
    direct.detailDomain=direct.physicalBoundaryDomain;
    direct.detailReferenceWorldPoints=direct.physicalBoundaryReferenceWorldPoints;
    direct.detailFaces=direct.physicalBoundaryFaces;
    direct.detailExpectedBoundaryFaceBarycentrics={{{1,0,0}},{{0,1,0}},{{0,0,1}}};
    direct.detailRepresentativeBoundaryVertexRows={0u,1u,2u};
    direct.detailGeometricDonorBodyIndices=direct.physicalBoundaryGeometricDonorBodyIndices;
    direct.detailSourceVertexRows=direct.physicalBoundarySourceVertexRows;
    direct.mechanicalReferenceWorldPoints[0]=direct.detailReferenceWorldPoints[0];
    direct.mechanicalReferenceWorldPoints[1]=direct.detailReferenceWorldPoints[1];
    direct.mechanicalReferenceWorldPoints[2]=direct.detailReferenceWorldPoints[2];
    PhysicalSkinAuthoringDetailBridgeState directState;
    PhysicalSkinAuthoringDetailBridgeReport directReport;
    assert(preparePhysicalSkinAuthoringDetailBridge(direct,directState,directReport,error));
    assert(directState.provenDirectIdentity&&directReport.surfaceExactAncestryRows==3u);
    assert(directReport.directIdentityRows==3u&&directReport.tetraEmbeddedRows==0u);
    for(std::size_t i=0;i<directState.tetraRows.size();++i){
        assert(directState.tetraRows[i].node[0]==i&&directState.tetraRows[i].node[1]==i&&
               directState.tetraRows[i].node[2]==i&&directState.tetraRows[i].node[3]==i);
        assert(directState.tetraRows[i].weight[0]==1.0f&&
               directState.tetraRows[i].weight[1]==0.0f&&
               directState.tetraRows[i].weight[2]==0.0f&&
               directState.tetraRows[i].weight[3]==0.0f);
    }

    // A mapped donor difference is admitted only through exact original-face
    // identity and a bounded face-adjacent path in the pinned parent topology.
    auto regionScoped=in;
    regionScoped.detailGeometricDonorBodyIndices={42u,42u,42u};
    regionScoped.detailBoundaryProjectionDonorBodyIndices={41u,41u,41u};
    regionScoped.detailRegionScopedDonorFallback={1u,1u,1u};
    regionScoped.sourceParentNodeCount=5u;
    regionScoped.sourceParentTetNodeRows={{{0u,1u,2u,3u}},{{1u,2u,3u,4u}}};
    regionScoped.sourceParentTetMaterialIndices={15u,15u};
    regionScoped.sourceParentTetAnatomicalRegionIndices={7u,7u};
    regionScoped.sourceParentTetStableSourceFaceIDs={9001u,9001u};
    regionScoped.physicalBoundaryFaceParentTetRows={1u};
    PhysicalSkinAuthoringDetailBridgeState regionScopedState;
    PhysicalSkinAuthoringDetailBridgeReport regionScopedReport;
    assert(preparePhysicalSkinAuthoringDetailBridge(
        regionScoped,regionScopedState,regionScopedReport,error));
    assert(regionScopedReport.regionScopedDonorFallbackRows==3u&&
           regionScopedReport.maximumSourceParentTetHops==1u);
    assert(regionScopedState.sourceParentTetPathRows.size()==3u&&
           regionScopedState.sourceParentTetPathRows[0]==std::vector<std::uint32_t>({0u,1u}));
    assert(regionScopedState.detailGeometricDonorBodyIndices==
           std::vector<std::uint32_t>({42u,42u,42u}));
    assert(regionScopedState.selectedPhysicalBoundaryFaceDonorBodyIndices==
           std::vector<std::uint32_t>({41u,41u,41u}));
    assert(regionScopedState.detailFrameOwnerBodyIndices==
           std::vector<std::uint32_t>({77u,77u,77u}));
    assert(regionScopedState.selectedPhysicalBoundaryFaceFrameOwnerBodyIndices==
           std::vector<std::uint32_t>({89u,89u,89u}));
    auto unsupportedCrosswalk=regionScoped;
    unsupportedCrosswalk.detailBoundaryProjectionDonorBodyIndices[0]=43u;
    assert(!preparePhysicalSkinAuthoringDetailBridge(
        unsupportedCrosswalk,regionScopedState,regionScopedReport,error));
    assert(regionScopedReport.regionScopedDonorFallbackRows==3u&&
           regionScopedState.selectedPhysicalBoundaryFaceDonorBodyIndices[0]==41u);
    auto wrongRegion=regionScoped;
    wrongRegion.detailAnatomicalRegionIndices[0]=8u;
    assert(!preparePhysicalSkinAuthoringDetailBridge(wrongRegion,regionScopedState,regionScopedReport,error));
    auto wrongAncestry=regionScoped;
    wrongAncestry.detailStableSourceFaceIDs[0]=9002u;
    assert(!preparePhysicalSkinAuthoringDetailBridge(wrongAncestry,regionScopedState,regionScopedReport,error));
    auto disconnectedPatch=regionScoped;
    disconnectedPatch.sourceParentNodeCount=6u;
    disconnectedPatch.sourceParentTetNodeRows[1]={0u,1u,4u,5u};
    assert(!preparePhysicalSkinAuthoringDetailBridge(disconnectedPatch,regionScopedState,regionScopedReport,error));

    // Every failed preparation leaves both the prior prepared state and report
    // byte fields untouched.
    const auto stateRows=state.tetraRows;
    const auto priorReport=report;
    auto badFace=in;badFace.detailExpectedBoundaryFaceRows[0]=9u;
    assert(!preparePhysicalSkinAuthoringDetailBridge(badFace,state,report,error));
    assert(state.tetraRows.size()==stateRows.size()&&state.tetraRows[0].node[0]==stateRows[0].node[0]&&
           state.tetraRows[0].weight[3]==stateRows[0].weight[3]);
    assert(report.detailRows==priorReport.detailRows&&
           report.maximumBoundaryProjectionDistanceM==priorReport.maximumBoundaryProjectionDistanceM);
    auto badLabels=in;badLabels.mechanicalTetGeometricDonorBodyIndices[0]=42u;
    assert(!preparePhysicalSkinAuthoringDetailBridge(badLabels,state,report,error));
    assert(state.tetraRows[0].weight[3]==stateRows[0].weight[3]&&report.detailRows==priorReport.detailRows);
    auto tooFar=in;tooFar.maximumTetExtrapolationDistanceM=0.001;
    assert(!preparePhysicalSkinAuthoringDetailBridge(tooFar,state,report,error));
    assert(state.tetraRows[0].weight[3]==stateRows[0].weight[3]&&report.detailRows==priorReport.detailRows);
    auto missingProjectionBound=in;missingProjectionBound.maximumBoundaryProjectionDistanceM=0.0;
    assert(!preparePhysicalSkinAuthoringDetailBridge(missingProjectionBound,state,report,error));
    assert(state.tetraRows[0].weight[3]==stateRows[0].weight[3]&&report.detailRows==priorReport.detailRows);
    auto singular=in;singular.mechanicalReferenceWorldPoints[3]=singular.mechanicalReferenceWorldPoints[0];
    assert(!preparePhysicalSkinAuthoringDetailBridge(singular,state,report,error));
    assert(state.tetraRows[0].weight[3]==stateRows[0].weight[3]&&report.detailRows==priorReport.detailRows);

    const auto priorOutput=result;
    auto invalidAccepted=in.mechanicalReferenceWorldPoints;
    invalidAccepted[0][0]=std::numeric_limits<double>::quiet_NaN();
    assert(!evaluatePhysicalSkinAuthoringDetailBridgeForCheck(state,invalidAccepted,result,error));
    assert(result==priorOutput);
}

} // namespace

int main() {
    Fixture f;
    PhysicalSkinAuthoringRoleConsumerState state;
    PhysicalSkinAuthoringRoleConsumerReport report;
    std::string error;
    if(!prepare(f,state,report,error)){std::cerr<<error<<std::endl;return 1;}
    assert(!state.physicalAdmission&&!report.physicalAdmission);
    assert(state.bindingBodyByLane.size()==86u&&report.bindingCount==86u);
    PhysicalSkinAuthoringOuterRoleRenderDomain outerRender;
    if(!preparePhysicalSkinAuthoringOuterRoleRenderDomain(
            f.bundle,state,outerRender,error)) {
        std::cerr<<"direct O role render map rejected: "<<error<<std::endl;return 2;
    }
    const auto& outerRole=state.roles[0];
    assert(!outerRender.physicalAdmission);
    assert(outerRender.sourceDomain==outerRole.sourceDomain);
    assert(outerRender.worldPoints==outerRole.currentWorldPoints);
    assert(outerRender.faces==outerRole.roleLocalFaces);
    assert(outerRender.authoringVertexRows==outerRole.authoringVertexRows);
    assert(outerRender.volumeNodeRows==outerRole.volumeNodeRows);
    assert(outerRender.transferRows.size()==outerRole.volumeNodeRows.size());
    for(std::size_t row=0;row<outerRender.transferRows.size();++row) {
        const auto& transfer=outerRender.transferRows[row];
        assert(transfer.coarseVertex[0]==row&&transfer.coarseVertex[1]==row&&
               transfer.coarseVertex[2]==row);
        assert(transfer.donorBodyIndex==outerRole.frameOwnerBodyIndices[row]);
        assert(transfer.weight[0]==1.0f&&transfer.weight[1]==0.0f&&
               transfer.weight[2]==0.0f&&transfer.restDistance==0.0f);
    }
    auto wrongOuterRows=state;
    wrongOuterRows.roles[0].volumeNodeRows[0]=wrongOuterRows.roles[0].volumeNodeRows[1];
    PhysicalSkinAuthoringOuterRoleRenderDomain untouchedOuter=outerRender;
    assert(!preparePhysicalSkinAuthoringOuterRoleRenderDomain(
        f.bundle,wrongOuterRows,untouchedOuter,error));
    assert(untouchedOuter.volumeNodeRows==outerRender.volumeNodeRows);
    for(unsigned layer=0u;layer<3u;++layer) {
        assert(report.rows[layer]==3u&&report.faces[layer]==1u);
        assert(report.frameOwnerParentDonorDifferences[layer]==3u);
        assert(state.roles[layer].authoringVertexRows==std::vector<std::uint32_t>({0u,1u,2u}));
        assert(state.roles[layer].frameOwnerBodyIndices[0]!=state.roles[layer].parentGeometricDonorBodyIndices[0]);
        assert(state.roles[layer].parentGeometricDonorBodyIndices[0]==150u);
        assert(state.roles[layer].dominantBindingIndices[0]==7u+layer);
        assert(report.maximumWeightNormalizationError[layer]==0.0);
        assert(report.maximumWorldLocalReconstructionErrorM[layer]<1.0e-12);
    }

    // Failed preparation is transactional: caller state and report remain untouched.
    const auto priorState=state;
    const auto priorReport=report;
    auto badOwner=f.bundle;
    badOwner.volumeNodeDonorBodyIndices[0]=150u;
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(badOwner,f.registered,f.sidecars,
        f.poses,f.weightFiles,state,report,error));
    assert(state.roles[0].authoringVertexRows==priorState.roles[0].authoringVertexRows);
    assert(state.bindingBodyByLane==priorState.bindingBodyByLane);
    assert(report.rows==priorReport.rows&&report.weightFileSHA256==priorReport.weightFileSHA256);

    auto duplicateLane=f.registered;
    f.poses[96].available=true;
    f.poses[96].binding=7u;
    duplicateLane.framesByDonor[96].available=true;
    duplicateLane.framesByDonor[96].binding=7u;
    duplicateLane.framesByDonor[96].currentOrigin=f.poses[96].currentOrigin;
    duplicateLane.framesByDonor[96].currentR=f.poses[96].currentLocalToWorld;
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(f.bundle,duplicateLane,f.sidecars,
        f.poses,f.weightFiles,state,report,error));

    auto wrongNamespace=f.bundle;
    wrongNamespace.sourceDomains[1].role=PhysicalSkinSourceDomainRole::outerSurface;
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(wrongNamespace,f.registered,f.sidecars,
        f.poses,f.weightFiles,state,report,error));

    auto swappedRows=f.bundle;
    auto swappedRegistration=f.registered;
    std::swap(swappedRows.outerSkin.sourceVertexIndices[0],swappedRows.outerSkin.sourceVertexIndices[1]);
    std::swap(swappedRows.volumeNodeSourceVertexIndices[0],swappedRows.volumeNodeSourceVertexIndices[1]);
    std::swap(swappedRegistration.nodeSource[0][0].vertex,swappedRegistration.nodeSource[0][1].vertex);
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(swappedRows,swappedRegistration,f.sidecars,
        f.poses,f.weightFiles,state,report,error));

    auto duplicateRows=f.bundle;
    auto duplicateRegistration=f.registered;
    duplicateRows.outerSkin.sourceVertexIndices[1]=0u;
    duplicateRows.volumeNodeSourceVertexIndices[1]=0u;
    duplicateRegistration.nodeSource[0][1].vertex=0;
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(duplicateRows,duplicateRegistration,f.sidecars,
        f.poses,f.weightFiles,state,report,error));

    auto wrongCounts=f.bundle;
    wrongCounts.sourceDomains[0].vertexCount=4u;
    wrongCounts.sourceDomains[0].faceCount=2u;
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(wrongCounts,f.registered,f.sidecars,
        f.poses,f.weightFiles,state,report,error));

    auto sourceEmbedding=f.registered;
    sourceEmbedding.nodeSource[0][0].bary[0]=1.0e-15;
    sourceEmbedding.nodeSource[0][0].residual[1]=1.0e-9;
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(f.bundle,sourceEmbedding,f.sidecars,
        f.poses,f.weightFiles,state,report,error));

    auto surfaceEmbedding=f.bundle;
    surfaceEmbedding.outerSkin.sourceEmbeddingResidualWorldM[0][0]=1.0e-9;
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(surfaceEmbedding,f.registered,f.sidecars,
        f.poses,f.weightFiles,state,report,error));

    auto wrongLocalBundle=f.bundle;
    wrongLocalBundle.independentFatDeep.donorLocalPoints[1][0]+=1.0e-3;
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(wrongLocalBundle,f.registered,f.sidecars,
        f.poses,f.weightFiles,state,report,error));

    auto changed=f.weightFiles;
    changed[1].expectedSHA256=std::string(64u,'0');
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(f.bundle,f.registered,f.sidecars,
        f.poses,changed,state,report,error));

    const auto missingPath=f.directory/"missing-row.csv";
    {
        std::ofstream csv(missingPath,std::ios::binary);
        const auto header=physical_skin_authoring_role_consumer_detail::weightHeader();
        for(std::size_t column=0;column<header.size();++column)csv<<(column?",":"")<<header[column];
        csv<<'\n';
        for(unsigned row=0u;row<2u;++row) {
            csv<<row;
            for(std::size_t lane=0;lane<kPhysicalSkinAuthoringFullWeightBindingCount;++lane)
                csv<<','<<(lane==7u?"1":"0");
            csv<<'\n';
        }
    }
    auto missingRows=f.weightFiles;
    missingRows[0].path=missingPath;
    assert(physical_skin_child_mass_detail::sha256File(missingPath,missingRows[0].expectedSHA256,error));
    assert(!preparePhysicalSkinAuthoringRoleConsumerState(f.bundle,f.registered,f.sidecars,
        f.poses,missingRows,state,report,error));
    assert(state.roles[0].authoringVertexRows==priorState.roles[0].authoringVertexRows);
    assert(report.rows==priorReport.rows&&report.weightFileSHA256==priorReport.weightFileSHA256);

    testAuthoringDetailBridge();
    std::cout<<"authoring_role_consumer=pass rows=3/3/3 bindings=86 parent_geometric_donor=150 frame_owners=distinct role_consumer_transactional_failures=11 detail_bridge_failures=10 detail_bridge=pass identity_namespaces=separate face_bary=source_connected tet_transfer=affine_objective rigid_rotation=pass physical_admission=false\n";
}
