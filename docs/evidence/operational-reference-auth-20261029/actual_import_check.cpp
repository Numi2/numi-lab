#include "NumiHumanParentReferenceMassTransfer.hpp"
#include "NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp"
#include "NumiHumanPhysicalSkinAuthoringRoleConsumer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
namespace fs=std::filesystem;
using Point=std::array<double,3>;
using Tet=std::array<std::uint32_t,4>;
static constexpr const char* P="/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209";
static constexpr const char* PARENT="/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-physical-parent-mass-330/native-source-identity-control-003";
static constexpr const char* SOURCE="/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/reduced-physical-boundary-400/attempt-002-parent-ancestry-walk";
static constexpr const char* SIDECARS="/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/operational-reference-map-514/run-002/sidecars";
static constexpr const char* RUN="/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/operational-reference-map-514/run-005-owner-authenticated-import";
static constexpr const char* CANDIDATE="/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/connected-layer-column-fit-514/run-001/native-mass-516/volume";
static constexpr const char* CANDIDATE_NPZ_SHA="a05bab5c739f2e0c7beae3e1e30afcec5a05274befadd399fcba6a238bf92a3e";
static constexpr const char* SOURCE_NPZ_SHA="2f547b21c6be0f3e21a021d80dad6c63c3a0e6af31d5ebe1780ca1b32c237a54";
static constexpr const char* PARENT_PACKET_SHA="c7823c42d64a2154a34c87cc0d4d202f0d5256f79217e74f5f62bcf1d5b31679";
static constexpr const char* SOURCE_CONTRIB_SHA="8e2424901318f25caca1540aeca412ca779fc014f68f49fe0cb6c17fb5aa0c1f";
static constexpr const char* SOURCE_SELF_SHA="4f33f1779864b188a127e7a3df8ac68c58f7ba134695d1fd3d3613052f0ae34c";
static constexpr const char* CANDIDATE_SELF_SHA="a2f983b20231f242853618151d858c10918eb40cd3125f59d899e80f91ad6e7a";

bool need(bool ok,const std::string& message){if(!ok)std::cerr<<"FAIL "<<message<<'\n';return ok;}
bool pin(const fs::path& path,std::string_view expected,std::string& error){
    std::string actual;
    if(!physical_skin_child_mass_detail::sha256File(path,actual,error))return false;
    if(actual!=expected){error="hash pin mismatch for "+path.string()+" got="+actual;return false;}
    return true;
}
std::string readText(const fs::path& path){std::ifstream in(path);std::ostringstream out;out<<in.rdbuf();return out.str();}
bool readPoseRows(const fs::path& path,std::vector<PhysicalSkinRegisteredBodyPose>& poses,
                  std::vector<PhysicalSkinDonorFrameRow>& pins,std::string& error){
    std::ifstream in(path);std::string line;
    if(!std::getline(in,line))return false;
    const auto header=physical_skin_child_mass_detail::splitCSV(line);
    if(header.size()!=26u||header[0]!="body_index"||header[1]!="binding")return false;
    poses.assign(157u,{});pins.assign(157u,{});std::vector<bool> seen(157u,false);std::size_t rows=0u;
    while(std::getline(in,line)){
        const auto f=physical_skin_child_mass_detail::splitCSV(line);
        std::uint32_t body=0u;std::int64_t binding=-1;
        if(f.size()!=26u||!physical_skin_child_mass_detail::parseNumber(f[0],body)||body>=157u||seen[body]||
           !physical_skin_child_mass_detail::parseNumber(f[1],binding)||binding< -1||binding>=86){error="bad body-pose identity";return false;}
        seen[body]=true;auto& p=poses[body];p.available=true;p.bodyIndex=body;p.arithmetic=PhysicalSkinPoseArithmetic::nativeFloat32;
        p.binding=binding<0?NM_INVALID_INDEX:static_cast<std::uint32_t>(binding);
        for(unsigned a=0;a<3u;++a)
            if(!physical_skin_child_mass_detail::parseNumber(f[2u+a],p.currentOrigin[a])||
               !physical_skin_child_mass_detail::parseNumber(f[14u+a],p.restOrigin[a])){error="invalid pose origin";return false;}
        for(unsigned a=0;a<9u;++a)
            if(!physical_skin_child_mass_detail::parseNumber(f[5u+a],p.currentLocalToWorld[a])||
               !physical_skin_child_mass_detail::parseNumber(f[17u+a],p.restLocalToWorld[a])){error="invalid pose matrix";return false;}
        if(binding>=0){auto& q=pins[body];q.available=true;q.binding=p.binding;q.arithmetic=PhysicalSkinPoseArithmetic::nativeFloat32;
            q.currentOrigin=p.currentOrigin;q.restOrigin=p.restOrigin;q.currentR=p.currentLocalToWorld;q.restR=p.restLocalToWorld;}
        ++rows;
    }
    if(!in.eof()||rows!=157u||std::find(seen.begin(),seen.end(),false)!=seen.end()){error="incomplete full native pose table";return false;}
    return true;
}
bool readRequests(const fs::path& path,std::vector<PhysicalSkinStableDeepAttachmentRequest>& requests,std::string& error){
    std::ifstream in(path);std::string line;
    if(!std::getline(in,line)||physical_skin_child_mass_detail::splitCSV(line)!=
       std::vector<std::string>{"stable_attachment_id","donor_body","stable_source_vertex","source_domain"}){error="attachment request header";return false;}
    while(std::getline(in,line)){
        auto f=physical_skin_child_mass_detail::splitCSV(line);PhysicalSkinStableDeepAttachmentRequest r;
        if(f.size()!=4u||!physical_skin_child_mass_detail::parseNumber(f[0],r.stableAttachmentID)||
           !physical_skin_child_mass_detail::parseNumber(f[1],r.donorBody)||
           !physical_skin_child_mass_detail::parseNumber(f[2],r.stableSourceVertex)||f[3]!="D37_coarse_parent"){
            error="invalid stable attachment request";return false;}
        r.sourceDomain=f[3];requests.push_back(std::move(r));
    }
    return in.eof()&&requests.size()==14u;
}

int main(){
    std::string error;
    const fs::path parentMass=fs::path(PARENT)/"compiled-parent-cell-mass.csv";
    const fs::path parentLabels=fs::path(PARENT)/"compiled-parent-cell-mass.csv.labels.csv";
    const fs::path parentNodes=fs::path(PARENT)/"compiled-parent-cell-mass.csv.nodes.csv";
    const fs::path parentTets=fs::path(PARENT)/"compiled-parent-cell-mass.csv.tets.csv";
    const fs::path parentSource=fs::path(PARENT)/"compiled-parent-cell-mass.csv.source.csv";
    const fs::path sourceNode=fs::path(SOURCE)/"parent-reduced-boundary.node";
    const fs::path sourceEle=fs::path(SOURCE)/"parent-reduced-boundary.ele";
    const fs::path sourceLabels=fs::path(SOURCE)/"parent-reduced-boundary.labels.csv";
    const fs::path sourceNPZ=fs::path(SOURCE)/"candidate.npz";
    const fs::path candidateNPZ=fs::path(P)/"connected-layer-column-fit-514/run-001/candidate.npz";
    const fs::path side=SIDECARS;
    const auto vdir=side/"volume", rdir=side/"registered", roledir=side/"roles";
    const fs::path sourceSelf=fs::path(SOURCE)/"exact-self-521/receipt.json";
    const fs::path candidateSelf=fs::path(P)/"connected-layer-column-fit-514/run-001/exact-self-517/receipt.json";
    const fs::path bodyPoseJSON=fs::path(PARENT)/"registered-skin-poses-100.json";
    const fs::path poseRows=fs::path(RUN)/"native-body-poses.csv";
    const fs::path attachmentRequests=fs::path(P)/"native-registered-volume-parent-identity-physical-control-363/inputs/deep-attachment-requests.csv";

    const std::array<std::pair<fs::path,std::string_view>,13> pinsToCheck{{
        {parentMass,PARENT_PACKET_SHA},{parentLabels,"99d643f71a735ee49223cdcd7b863d2d0978b8cc490a138bad4a63012438a457"},
        {parentNodes,"7044051353d173b94143a4027ba7dcc468b221512b3f41d5819d27025b36d044"},
        {parentTets,"82e20067454748a9eca91bc6f06ab343677c051a139797ae051c64022bd87992"},
        {parentSource,SOURCE_CONTRIB_SHA},{sourceNPZ,SOURCE_NPZ_SHA},{candidateNPZ,CANDIDATE_NPZ_SHA},
        {sourceSelf,SOURCE_SELF_SHA},{candidateSelf,CANDIDATE_SELF_SHA},
        {bodyPoseJSON,"0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a"},
        {attachmentRequests,""},{vdir/"combined-node-map.csv","59dc5716224050686a6d2df5537f011a94558b7e25a9bfff30a85e27c1c3640f"},
        {vdir/"mesh.node","b8d1a5fd7b450808dabcc7cdc3450e4ddeb043c3e57e62e3ed49b840589a2016"}
    }};
    for(const auto& [path,digest]:pinsToCheck){
        if(digest.empty())continue;
        if(!pin(path,digest,error)){std::cerr<<"FAIL input pin: "<<error<<'\n';return 1;}
    }
    const auto sourceReceiptText=readText(sourceSelf),candidateReceiptText=readText(candidateSelf);
    if(sourceReceiptText.find("\"candidate_sha256\": \""+std::string(SOURCE_NPZ_SHA)+"\"")==std::string::npos||
       candidateReceiptText.find("\"candidate_sha256\": \""+std::string(CANDIDATE_NPZ_SHA)+"\"")==std::string::npos||
       sourceReceiptText.find("\"self_intersection_pairs\": 0")==std::string::npos||
       candidateReceiptText.find("\"self_intersection_pairs\": 0")==std::string::npos){
        std::cerr<<"FAIL exact-self receipt content does not match pinned assets and zero-pair audits\n";return 2;
    }
    PhysicalSkinChildMassMappingBundle parentBundle;
    if(!loadCompiledSkinFatParentMassPacketCSV(parentMass,parentBundle.parent,error)||
       !loadCompiledSkinFatParentLabelsCSV(parentLabels,parentBundle.parent,error)){
        std::cerr<<"FAIL immutable parent load: "<<error<<'\n';return 3;
    }
    if(parentBundle.parent.parentMassPacketSHA256!=PARENT_PACKET_SHA||
       parentBundle.parent.parentSourceContributionSHA256!=SOURCE_CONTRIB_SHA){
        std::cerr<<"FAIL parent 330 packet/source contribution digest mismatch\n";return 4;
    }
    PhysicalSkinChildMassMappingBundle sourceGeometry;sourceGeometry.parent=parentBundle.parent;
    PhysicalSkinChildTetGenImportReport sourceTetgen;
    if(!loadPhysicalSkinChildTetGenReference(sourceNode,sourceEle,sourceLabels,
            "0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a",157u,0.000259,
            sourceGeometry,sourceTetgen,error)){
        std::cerr<<"FAIL source002 TetGen load: "<<error<<'\n';return 5;
    }
    PhysicalSkinChildMassMappingBundle candidateGeometry;candidateGeometry.parent=parentBundle.parent;
    PhysicalSkinChildTetGenImportReport candidateTetgen;
    if(!loadPhysicalSkinChildTetGenReference(vdir/"mesh.node",vdir/"mesh.ele",vdir/"child.labels.csv",
            "0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a",157u,0.000259,
            candidateGeometry,candidateTetgen,error)){
        std::cerr<<"FAIL candidate514 TetGen load: "<<error<<'\n';return 6;
    }
    const auto& parent=parentBundle.parent;
    if(sourceGeometry.referenceWorldPoints.size()!=parent.referenceWorldPoints.size()||
       sourceGeometry.childTetrahedra!=parent.tetrahedra||candidateGeometry.childTetrahedra!=parent.tetrahedra||
       sourceGeometry.childMaterialIndices!=parent.materialIndices||candidateGeometry.childMaterialIndices!=parent.materialIndices||
       sourceGeometry.childDonorBodyIndices!=parent.donorBodyIndices||candidateGeometry.childDonorBodyIndices!=parent.donorBodyIndices||
       sourceGeometry.childAnatomicalRegionIndices!=parent.anatomicalRegionIndices||candidateGeometry.childAnatomicalRegionIndices!=parent.anatomicalRegionIndices||
       sourceGeometry.childStableSourceFaceIdentities!=parent.stableSourceFaceIdentities||candidateGeometry.childStableSourceFaceIdentities!=parent.stableSourceFaceIdentities){
        std::cerr<<"FAIL source002/514 topology or material/donor/region/stable-face identity differs from immutable 330\n";return 7;
    }
    double source330To002Max=0.0;
    for(std::size_t node=0;node<parent.referenceWorldPoints.size();++node){
        Point f32parent{};
        for(unsigned axis=0;axis<3u;++axis)f32parent[axis]=double(float(parent.referenceWorldPoints[node][axis]));
        source330To002Max=std::max(source330To002Max,
            norm(sub(sourceGeometry.referenceWorldPoints[node],f32parent)));
    }
    if(source330To002Max>0.0004){std::cerr<<"FAIL source330->source002 registration exceeds 0.4mm: "<<source330To002Max<<'\n';return 8;}
    std::vector<std::uint32_t> nodeMap(parent.referenceWorldPoints.size()),tetMap(parent.tetrahedra.size());
    std::iota(nodeMap.begin(),nodeMap.end(),0u);std::iota(tetMap.begin(),tetMap.end(),0u);
    std::vector<double> lower(17u,0.0),upper(17u,0.0);lower[15]=900.0;upper[15]=1100.0;lower[16]=850.0;upper[16]=950.0;
    SameTopologyReferencePullbackInput proofInput{
        .sourceReferencePoints=sourceGeometry.referenceWorldPoints,
        .candidateReferencePoints=candidateGeometry.referenceWorldPoints,
        .candidateToSourceNodeRows=nodeMap,
        .sourceTetrahedra=sourceGeometry.childTetrahedra,
        .candidateTetrahedra=candidateGeometry.childTetrahedra,
        .candidateToSourceTetRows=tetMap,
        .sourceMaterialIds=sourceGeometry.childMaterialIndices,
        .candidateMaterialIds=candidateGeometry.childMaterialIndices,
        .sourceGeometricDonorIds=sourceGeometry.childDonorBodyIndices,
        .candidateGeometricDonorIds=candidateGeometry.childDonorBodyIndices,
        .sourceRegionIds=sourceGeometry.childAnatomicalRegionIndices,
        .candidateRegionIds=candidateGeometry.childAnatomicalRegionIndices,
        .sourceStableFaceIds=sourceGeometry.childStableSourceFaceIdentities,
        .candidateStableFaceIds=candidateGeometry.childStableSourceFaceIdentities,
        .sourceCellMassKg=parent.tetrahedronMassKg,
        .materialDensityLowerKgM3=lower,.materialDensityUpperKgM3=upper,
        .sourceVolumeSHA256=SOURCE_NPZ_SHA,.candidateVolumeSHA256=CANDIDATE_NPZ_SHA,
        .parentMassPacketSHA256=PARENT_PACKET_SHA,.parentSourceContributionSHA256=SOURCE_CONTRIB_SHA,
        .sourceBoundarySelfAuditSHA256=SOURCE_SELF_SHA,.candidateBoundarySelfAuditSHA256=CANDIDATE_SELF_SHA,
        .sourceBoundarySelfIntersectionPairs=0u,.candidateBoundarySelfIntersectionPairs=0u};
    SameTopologyReferencePullbackResult pullback;
    if(!verifySameTopologyReferencePullback(proofInput,pullback,error)){
        std::cerr<<"FAIL exact operational reference pullback proof: "<<error<<'\n';return 9;
    }
    if(pullback.verificationSHA256!="c5659beb43f184902ca06f21a8d2cad5ff7920ecf88b34e83caa0f00bed2f1cb"){
        std::cerr<<"FAIL pullback proof digest differs from frozen actual 520 receipt: "<<pullback.verificationSHA256<<'\n';return 10;
    }
    auto rejectProofMutation=[&](SameTopologyReferencePullbackInput bad,const char* name){
        SameTopologyReferencePullbackResult rejected;std::string negativeError;
        if(verifySameTopologyReferencePullback(bad,rejected,negativeError)){
            std::cerr<<"FAIL pullback negative accepted: "<<name<<'\n';return false;
        }
        std::cout<<"negative_"<<name<<"=REJECTED ";
        return true;
    };
    const auto originalNodeMap0=nodeMap[0];nodeMap[0]=1u;
    const bool wrongNodeRejected=rejectProofMutation(proofInput,"wrong_node");nodeMap[0]=originalNodeMap0;
    if(!wrongNodeRejected)return 18;
    const auto originalStableFace0=candidateGeometry.childStableSourceFaceIdentities[0];
    candidateGeometry.childStableSourceFaceIdentities[0]^=1u;
    const bool wrongFaceRejected=rejectProofMutation(proofInput,"wrong_stable_face");
    candidateGeometry.childStableSourceFaceIdentities[0]=originalStableFace0;
    if(!wrongFaceRejected)return 19;
    const auto originalSourceMass0=parentBundle.parent.tetrahedronMassKg[0];parentBundle.parent.tetrahedronMassKg[0]*=2.0;
    const bool wrongMassRejected=rejectProofMutation(proofInput,"wrong_immutable_source_mass");
    parentBundle.parent.tetrahedronMassKg[0]=originalSourceMass0;
    if(!wrongMassRejected)return 20;
    std::map<std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>,double> sourceMassByGroup,candidateMassByGroup;
    for(std::size_t cell=0u;cell<parent.tetrahedronMassKg.size();++cell){
        const auto key=std::tuple{parent.materialIndices[cell],parent.donorBodyIndices[cell],parent.anatomicalRegionIndices[cell]};
        sourceMassByGroup[key]+=parent.tetrahedronMassKg[cell];
        candidateMassByGroup[key]+=pullback.childMassKg[cell];
    }
    double sourceDonorGroupMaxClosureKg=0.0;
    for(const auto& [key,mass]:sourceMassByGroup)
        sourceDonorGroupMaxClosureKg=std::max(sourceDonorGroupMaxClosureKg,std::abs(mass-candidateMassByGroup[key]));
    if(sourceDonorGroupMaxClosureKg>1.0e-12){std::cerr<<"FAIL actual source-donor group closure "<<sourceDonorGroupMaxClosureKg<<'\n';return 21;}
    std::vector<PhysicalSkinRegisteredBodyPose> nativePoses;
    std::vector<PhysicalSkinDonorFrameRow> nativePins;
    if(!readPoseRows(fs::path(RUN)/"native-body-poses.csv",nativePoses,nativePins,error)){
        std::cerr<<"FAIL captured native poses: "<<error<<'\n';return 11;
    }
    PhysicalSkinRegisteredSurfaceCSVFiles surfaceFiles;
    const std::array<std::string,3> roleNames{"O","I","D"};
    const std::array<std::string,3> roleDomains{"derived_connected_refinement433_O","D37_coarse_parent","D37_coarse_parent"};
    const std::array<std::string,3> roleAssetHashes{
        "10c42bce48081f740c90dcf047d23ae2c4bb26761c984236391d40350e8c969b",
        "da6a8148b5fd3fde939285bac68af289d3af8a5bcf787039829eb24bf4195242",
        "ddbc84bdd54e92c49ccba78484d08b3a4f1d91ce427e8338113a91b8138fc781"};
    std::array<PhysicalSkinSourceDomainRegistration,3> domains{};
    for(unsigned layer=0u;layer<3u;++layer){
        auto& file=surfaceFiles.layers[layer];const auto role=roleNames[layer];
        file.nodes=rdir/("registered-"+role+".csv");
        file.faces=rdir/("registered-"+role+"-faces.csv");
        file.nodesSHA256=layer==0?"ed6a87d1228f4648b825d787348e7a08b74eb085b1ef0befb6883299bfff283b":
            (layer==1?"530df1441ca5a15e883d279ba4ba215e122d5da88d4b8dba26a5c2a7051b6e81":"822773fb703dab8f09cfcd2885873784acc8652c7f17e81cd8ae165207ad34fc");
        file.facesSHA256=layer==0?"0a41d9287b3ae14a754a1d1f273f03c18715e718bb2a3f292057a1de4acd1181":
            (layer==1?"c35204767f2bee819de809761e8f561f9aa16c4cddd6eb4f749b8d25cc0d4d02":"e4b4d099b41daffe4173592d168e38389828222219e75f9a8b32bc42ffb08b48");
        file.name="registered "+role+" surface";file.sourceMember=file.nodes.string();
        file.stableIdentifier=layer==0?150001u:(layer==1?150002u:160001u);
        domains[layer]={roleDomains[layer],static_cast<PhysicalSkinSourceDomainRole>(layer+1u),
            roleAssetHashes[layer],2411u,4777u};
    }
    surfaceFiles.nodeRegistration=rdir/"node-parent-registration.csv";
    surfaceFiles.nodeRegistrationSHA256="d652f9dbc8fad4b693aab589a930feaeca1a7c0b786ff6cf0fbacb031a97ab1b";
    surfaceFiles.donorFrames=rdir/"donor-frames.csv";
    surfaceFiles.donorFramesSHA256="b1704fb7b846254583f4b6a08c394ed3cad1603554244078706ca4b3811ac674";
    surfaceFiles.outerFaceAncestry=rdir/"outer-face-ancestry.csv";
    surfaceFiles.outerFaceAncestrySHA256="b694cf38bfa962a5878992b299f82f757e3d31e4a8a6ddb5838a1d324e0cb92f";
    surfaceFiles.donorCount=157u;surfaceFiles.d37InnerFaceCount=4777u;surfaceFiles.d37DeepFaceCount=4777u;
    surfaceFiles.maxParentRegistrationDistanceM=0.0004;surfaceFiles.parentRegistrationIsHistoricalProvenanceOnly=true;
    surfaceFiles.nativeFloat32FramesByDonor=nativePins;
    PhysicalSkinRegisteredSurfaceCSVSet registered;PhysicalSkinRegisteredSurfaceCSVReport surfaceReport;
    PhysicalSkinSurfaceParentRegistrationReport parentReport;
    if(!loadPhysicalSkinRegisteredSurfaceCSVSetForParent(surfaceFiles,parent,registered,surfaceReport,parentReport,error)){
        std::cerr<<"FAIL actual pinned O/I/D source registration: "<<error<<'\n';return 12;
    }
    const std::array<std::size_t,5> columns{};(void)columns;
    PhysicalSkinRegisteredVolumeAssetFiles files;
    files.tetgenNode=vdir/"mesh.node";files.tetgenElements=vdir/"mesh.ele";files.childLabels=vdir/"child.labels.csv";
    files.combinedNodeMapCSV=vdir/"combined-node-map.csv";files.facetAncestryCSV=vdir/"facet-ancestry.csv";
    files.expectedSHA256={"b8d1a5fd7b450808dabcc7cdc3450e4ddeb043c3e57e62e3ed49b840589a2016",
        "49671889c6711fedd8dd5727bd264e9d652b3c8ef48e15783b6be7a44c32c347",
        "fc5c45635613c6280a8301646ba7c95daab1422dfdb22b8f25e0f45acc81b07a",
        "59dc5716224050686a6d2df5537f011a94558b7e25a9bfff30a85e27c1c3640f",
        "c0f02267baa4ed6a42bb464fc3d0172f0adab98a59f92a14cfb5f3edbdb0d30f"};
    files.registrationDomain=PhysicalSkinRegisteredVolumeRegistrationDomain::candidateOperationalReference;
    PhysicalSkinOperationalReferencePullbackAssets pullbackAssets;
    pullbackAssets.sourceParentPacketCSV=parentMass;
    pullbackAssets.sourceTetgenNode=fs::path(SOURCE)/"parent-reduced-boundary.node";
    pullbackAssets.sourceTetgenElements=fs::path(SOURCE)/"parent-reduced-boundary.ele";
    pullbackAssets.sourceLabels=fs::path(SOURCE)/"parent-reduced-boundary.labels.csv";
    pullbackAssets.sourceTetGenSHA256={"d1238fc477b290c78970d5a7bbf01e7465a5e85aa00fe7414ee3b95503cc1b50",
        "49671889c6711fedd8dd5727bd264e9d652b3c8ef48e15783b6be7a44c32c347",
        "c4f751327f3b47e24e892f397d726105e2c8be0846e7b3b07f99b2e202e5ae5c"};
    pullbackAssets.sourceVolumeAsset=fs::path(SOURCE)/"candidate.npz";
    pullbackAssets.sourceVolumeSHA256=SOURCE_NPZ_SHA;
    pullbackAssets.sourceProducerReceipt=fs::path(SOURCE)/"reconstructed-receipt.json";
    pullbackAssets.sourceProducerReceiptSHA256="70faa10f6952d85e5d7b788cddca4896fc329126c82102215489a97ffc7c685f";
    pullbackAssets.sourceSelfAuditReceipt=fs::path(SOURCE)/"exact-self-521/receipt.json";
    pullbackAssets.sourceSelfAuditReceiptSHA256=SOURCE_SELF_SHA;
    pullbackAssets.candidateVolumeAsset=fs::path(P)/"connected-layer-column-fit-514/run-001/candidate.npz";
    pullbackAssets.candidateVolumeSHA256=CANDIDATE_NPZ_SHA;
    pullbackAssets.candidateProducerReceipt=fs::path(SIDECARS)/"receipt.json";
    std::string candidateProducerSHA;
    if(!physical_skin_child_mass_detail::sha256File(pullbackAssets.candidateProducerReceipt,candidateProducerSHA,error)){
        std::cerr<<"FAIL candidate sidecar receipt hash: "<<error<<'\n';return 24;
    }
    pullbackAssets.candidateProducerReceiptSHA256=candidateProducerSHA;
    pullbackAssets.candidateSelfAuditReceipt=fs::path(P)/"connected-layer-column-fit-514/run-001/exact-self-517/receipt.json";
    pullbackAssets.candidateSelfAuditReceiptSHA256=CANDIDATE_SELF_SHA;
    files.operationalReferencePullback=std::move(pullbackAssets);
    {
        auto badFiles=files;
        badFiles.operationalReferencePullback->candidateProducerReceiptSHA256=std::string(64u,'0');
        PhysicalSkinChildMassMappingBundle unchanged;unchanged.parent=parent;
        PhysicalSkinRegisteredVolumeImportSidecars unchangedSidecars;
        unchangedSidecars.sourceDomains={"sentinel"};
        PhysicalSkinRegisteredVolumeFileImportReport unchangedReport;unchangedReport.nodeMap.fileRows=6789u;
        std::vector<PhysicalSkinStableDeepAttachmentRequest> noAttachments;
        const bool accepted=importPhysicalSkinRegisteredVolumeFiles(badFiles,
            "0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a",157u,0.000259,0.0004,
            "",0u,0u,registered,noAttachments,nativePoses,unchanged,unchangedSidecars,unchangedReport,error,
            std::span<const PhysicalSkinSourceDomainRegistration>(domains));
        if(accepted||error.find("hash differs") == std::string::npos||
           !unchanged.childTetrahedra.empty()||
           unchangedSidecars.sourceDomains!=std::vector<std::string>{"sentinel"}||
           unchangedReport.nodeMap.fileRows!=6789u){
            std::cerr<<"FAIL tampered producer receipt was not rejected atomically: "<<error<<'\n';return 23;
        }
        std::cout<<"negative_tampered_candidate_receipt=REJECTED atomic_outputs=PASS\n";
    }
    std::vector<PhysicalSkinStableDeepAttachmentRequest> attachments;
    if(!readRequests(attachmentRequests,attachments,error)){std::cerr<<"FAIL exact D attachment source requests: "<<error<<'\n';return 13;}
    PhysicalSkinChildMassMappingBundle imported;imported.parent=parent;
    PhysicalSkinRegisteredVolumeImportSidecars importSidecars;
    PhysicalSkinRegisteredVolumeFileImportReport importReport;
    if(!importPhysicalSkinRegisteredVolumeFiles(files,
            "0c6e10d411604cb317eb5a9da234da568d683e6cc0276ac184f80fd29067b80a",157u,0.000259,0.0004,
            "",0u,0u,registered,attachments,nativePoses,imported,importSidecars,importReport,error,
            std::span<const PhysicalSkinSourceDomainRegistration>(domains))){
        std::cerr<<"FAIL actual 514 registered-volume file import: "<<error<<'\n';return 14;
    }
    if(importReport.physicalAdmission||
       !importReport.registeredVolume.separatedOperationalReference||
       importReport.registeredVolume.volumeNodeCount!=7233u||
       importReport.registeredVolume.sourceParentNodeIdentityRows!=7233u||
       importReport.registeredVolume.maximumOperationalReferenceDistanceM!=0.0){
        std::cerr<<"FAIL import result counts/claim/domain"<<'\n';return 15;
    }
    std::array<PhysicalSkinAuthoringFullWeightsCSV,3> weights{};
    for(unsigned layer=0u;layer<3u;++layer){
        weights[layer].path=roledir/(roleNames[layer]+"-full86.csv");
        weights[layer].expectedSHA256="3cc383ecd0678b5777bc7db00ce104309a09389c0975cfa6c252f14a5e8690a0";
    }
    PhysicalSkinAuthoringRoleConsumerState roleState;PhysicalSkinAuthoringRoleConsumerReport roleReport;
    if(!preparePhysicalSkinAuthoringRoleConsumerState(imported,registered,importSidecars,nativePoses,weights,
            roleState,roleReport,error)){
        std::cerr<<"FAIL actual full86 O/I/D consumer: "<<error<<'\n';return 16;
    }
    PhysicalSkinAuthoringOuterRoleRenderDomain outer;
    if(!preparePhysicalSkinAuthoringOuterRoleRenderDomain(imported,roleState,outer,error)){
        std::cerr<<"FAIL actual typed O direct role adapter: "<<error<<'\n';return 17;
    }
    std::cout<<"actual_cpu_only_chain_and_import=PASS source330_to_source002_max_m="<<source330To002Max
      <<" source002_to_candidate514_max_m="<<pullback.maximumNodeDisplacementM
      <<" operational_candidate_self_map_max_m="<<importReport.registeredVolume.maximumOperationalReferenceDistanceM
      <<" historical_candidate_parent_registration_max_m="<<importReport.registeredVolume.maximumHistoricalSourceParentDistanceM
      <<" source330_parent_mass_kg="<<parent.totalMassKg
      <<" source_donor_group_max_closure_kg="<<std::setprecision(17)<<sourceDonorGroupMaxClosureKg
      <<" volume_nodes="<<importReport.registeredVolume.volumeNodeCount
      <<" role_nodes="<<roleReport.rows[0]<<","<<roleReport.rows[1]<<","<<roleReport.rows[2]
      <<" role_faces="<<roleReport.faces[0]<<","<<roleReport.faces[1]<<","<<roleReport.faces[2]
      <<" shared_I="<<importReport.nodeMap.sharedInterfacePairs
      <<" O_direct_rows="<<outer.worldPoints.size()<<" D_attachments="<<importSidecars.deepAttachments.size()
      <<" geometry_bone_pairs=183 physical_admission=0 runtime=0 accepted_steps=0 GPU=0\n";
    return 0;
}
