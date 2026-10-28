#include "NumiHumanPhysicalSkinChildIntegration.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace metalrobo::human::coarse_tissue;

static bool writeNodes(const std::filesystem::path& path, const std::vector<V3>& points,
                       int alteredNode=-1) {
    std::ofstream out(path); out << std::setprecision(17) << points.size() << " 3 0 0\n";
    for (std::size_t i=0;i<points.size();++i) {
        V3 p{}; for (unsigned a=0;a<3u;++a) p[a]=static_cast<double>(static_cast<float>(points[i][a]));
        if (static_cast<int>(i)==alteredNode) p[0]+=0.02;
        out << (i+1u) << ' ' << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
    }
    return bool(out);
}
static bool writeElements(const std::filesystem::path& path,
                          const std::array<std::uint32_t,4>& nodes,
                          std::uint32_t material) {
    std::ofstream out(path); out << "1 4 1\n1";
    for (auto n:nodes) out << ' ' << (n+1u);
    out << ' ' << material << '\n'; return bool(out);
}
static bool writeLabels(const std::filesystem::path& path, std::uint32_t material,
                        std::uint32_t donor, std::uint32_t region, std::uint32_t face) {
    std::ofstream out(path);
    out << "child_tet_row,material_index,geometric_donor_body,anatomical_region_id,stable_source_face_id\n"
        << "0," << material << ',' << donor << ',' << region << ',' << face << '\n';
    return bool(out);
}
static PhysicalSkinParentMassPacket makeParent(const std::vector<V3>& ref,
                                                const std::vector<V3>& current) {
    PhysicalSkinParentMassPacket p;
    p.referenceWorldPoints=ref; p.currentWorldPoints=current;
    for(std::size_t i=0;i<ref.size();++i) {
        p.compiledGlobalNodeIndices.push_back(static_cast<std::uint32_t>(i)); p.nodeFrameOwnerBodyIndices.push_back(0u);
        p.nodeFrameOwnerLocalPoints.push_back(current[i]);
    }
    p.tetrahedra.push_back({0u,1u,2u,3u}); p.referenceTetVolumeM3.push_back(1.0/6.0);
    p.tetrahedronMassKg.push_back(0.1); p.materialIndices.push_back(15u);
    p.donorBodyIndices.push_back(0u); p.anatomicalRegionIndices.push_back(3u);
    p.stableSourceFaceIdentities.push_back(901u);
    p.parentMassPacketSHA256=std::string(64u,'a');
    p.parentSourceContributionSHA256=std::string(64u,'b');
    p.parentTopologySHA256=std::string(64u,'c'); p.parentLabelSHA256=std::string(64u,'d');
    return p;
}
static ExplicitUnpairedVolumeTopology makeExpected(
        const PhysicalSkinParentMassPacket& p, const std::vector<std::uint32_t>& parentFace,
        const std::vector<std::uint32_t>& facetRows, const std::vector<std::uint32_t>& facetStable,
        const std::vector<std::uint8_t>& wallFlags, const std::vector<std::uint32_t>& wallStable,
        const std::vector<std::uint32_t>& sourceVertex, const std::vector<std::uint32_t>& sourceFace,
        const std::vector<std::array<double,3>>& sourceBary, const std::vector<std::uint32_t>& nodeParent,
        const std::vector<std::array<double,4>>& nodeParentBary, const std::vector<V3>& nodeResidual,
        const std::vector<std::array<double,9>>& nodeLinear, const std::vector<V3>& nodeOrigin,
        const std::vector<std::array<double,9>>& bodyLinear, const std::vector<V3>& bodyOrigin) {
    ExplicitUnpairedVolumeTopology t;
    t.currentWorldPoints=p.currentWorldPoints; t.referenceWorldPoints=p.referenceWorldPoints;
    t.childTetrahedra=p.tetrahedra; t.childMaterialIndices=p.materialIndices;
    t.childDonorBodyIndices=p.donorBodyIndices; t.childParentSurfaceFaces=parentFace;
    t.childStableSourceFaceIdentities=p.stableSourceFaceIdentities;
    t.childRegisteredLayerFacetSurfaceFaces=facetRows;
    t.childRegisteredLayerFacetSourceFaceIdentities=facetStable;
    t.childAuthoredHeadSidewallFacetFlags=wallFlags;
    t.childAuthoredHeadSidewallStableFaceIdentities=wallStable;
    t.volumeNodeDonorBodyIndices=p.nodeFrameOwnerBodyIndices;
    t.volumeNodeDonorLocalPoints=p.nodeFrameOwnerLocalPoints;
    t.volumeNodeDonorLocalToWorldLinear=nodeLinear; t.volumeNodeDonorOriginWorld=nodeOrigin;
    t.registeredBodyLocalToWorldLinear=bodyLinear; t.registeredBodyOriginWorld=bodyOrigin;
    t.detailedSourceVertexCount=p.referenceWorldPoints.size(); t.detailedSourceFaceCount=1u;
    t.volumeNodeSourceVertexIndices=sourceVertex; t.volumeNodeSourceFaceIndices=sourceFace;
    t.volumeNodeSourceBarycentric=sourceBary; t.retainedOriginalNodeCount=p.referenceWorldPoints.size();
    t.volumeNodeParentTetrahedra=nodeParent; t.volumeNodeParentBarycentric=nodeParentBary;
    t.volumeNodeParentResidualWorldM=nodeResidual;
    t.referenceSourceIdentity={1u,2u,3u,4u}; t.materialSourceIdentity={5u,6u,7u,8u};
    return t;
}
static PhysicalSkinChildMassMappingBundle makeBundle(
        const PhysicalSkinParentMassPacket& parent, const ExplicitUnpairedVolumeTopology& expected,
        const std::vector<std::uint32_t>& parentFace, const std::vector<std::uint32_t>& facetRows,
        const std::vector<std::uint32_t>& facetStable, const std::vector<std::uint8_t>& wallFlags,
        const std::vector<std::uint32_t>& wallStable, const std::vector<std::uint32_t>& sourceVertex,
        const std::vector<std::uint32_t>& sourceFace, const std::vector<std::array<double,3>>& sourceBary,
        const std::vector<std::uint32_t>& nodeParent, const std::vector<std::array<double,4>>& nodeParentBary,
        const std::vector<V3>& nodeResidual, const std::vector<std::array<double,9>>& nodeLinear,
        const std::vector<V3>& nodeOrigin, const std::vector<std::array<double,9>>& bodyLinear,
        const std::vector<V3>& bodyOrigin) {
    PhysicalSkinChildMassMappingBundle b; b.parent=parent;
    b.childParentSurfaceFaces=parentFace; b.childRegisteredLayerFacetSurfaceFaces=facetRows;
    b.childRegisteredLayerFacetSourceFaceIdentities=facetStable;
    b.childAuthoredHeadSidewallFacetFlags=wallFlags; b.childAuthoredHeadSidewallStableFaceIdentities=wallStable;
    b.volumeNodeDonorBodyIndices=parent.nodeFrameOwnerBodyIndices;
    b.volumeNodeDonorLocalPoints=parent.nodeFrameOwnerLocalPoints;
    b.volumeNodeDonorLocalToWorldLinear=nodeLinear; b.volumeNodeDonorOriginWorld=nodeOrigin;
    b.registeredBodyLocalToWorldLinear=bodyLinear; b.registeredBodyOriginWorld=bodyOrigin;
    b.detailedSourceVertexCount=parent.referenceWorldPoints.size(); b.detailedSourceFaceCount=1u;
    b.volumeNodeSourceVertexIndices=sourceVertex; b.volumeNodeSourceFaceIndices=sourceFace;
    b.volumeNodeSourceBarycentric=sourceBary; b.retainedOriginalNodeCount=parent.referenceWorldPoints.size();
    b.volumeNodeParentTetrahedra=nodeParent; b.volumeNodeParentBarycentric=nodeParentBary;
    b.volumeNodeParentResidualWorldM=nodeResidual;
    b.referenceSourceIdentity=expected.referenceSourceIdentity; b.materialSourceIdentity=expected.materialSourceIdentity;
    b.maximumSpatialMassTransportDistanceM=1.0e-6;
    return b;
}
static bool parseAndValidate(
        const PhysicalSkinChildIdentityImportFiles& files,
        const PhysicalSkinParentMassPacket& parent, const ExplicitUnpairedVolumeTopology& expected,
        const std::vector<std::uint32_t>& parentFace,const std::vector<std::uint32_t>& facetRows,
        const std::vector<std::uint32_t>& facetStable,const std::vector<std::uint8_t>& wallFlags,
        const std::vector<std::uint32_t>& wallStable,const std::vector<std::uint32_t>& sourceVertex,
        const std::vector<std::uint32_t>& sourceFace,const std::vector<std::array<double,3>>& sourceBary,
        const std::vector<std::uint32_t>& nodeParent,const std::vector<std::array<double,4>>& nodeParentBary,
        const std::vector<V3>& nodeResidual,const std::vector<std::array<double,9>>& nodeLinear,
        const std::vector<V3>& nodeOrigin,const std::vector<std::array<double,9>>& bodyLinear,
        const std::vector<V3>& bodyOrigin,const std::string& expectedFailure,std::string& failure) {
    auto bundle=makeBundle(parent,expected,parentFace,facetRows,facetStable,wallFlags,wallStable,
        sourceVertex,sourceFace,sourceBary,nodeParent,nodeParentBary,nodeResidual,nodeLinear,nodeOrigin,bodyLinear,bodyOrigin);
    const auto frame=physical_skin_child_integration_detail::identityCoordinateFrameSHA256(parent,expected);
    PhysicalSkinChildTetGenImportReport tetReport; std::string error;
    if(!loadPhysicalSkinChildTetGenReference(files.childNodeFile,files.childElementFile,
            files.childLabelsCSV,frame,1u,1.0e-6,bundle,tetReport,error)) {
        failure=error; return expectedFailure=="import";
    }
    PhysicalSkinChildIdentityImportReport report; report.coordinateFrameSHA256=frame;
    report.importedTetGenMeshSHA256=bundle.childMeshSHA256;
    const auto importedReference=bundle.referenceWorldPoints;
    if(validatePhysicalSkinChildIdentityImport(bundle,expected,report,error)) {
        if(!expectedFailure.empty()) { failure="unexpected accept"; return false; }
        if(bundle.currentWorldPoints!=parent.currentWorldPoints||
           bundle.referenceWorldPoints!=parent.referenceWorldPoints||
           importedReference==bundle.referenceWorldPoints||
           bundle.coordinateFrameSHA256!=frame||
           bundle.childMeshSHA256!=report.exactLiveChildMeshSHA256) {
            failure="exact F64 live current/reference or frame fingerprint was not retained"; return false;
        }
        return true;
    }
    failure=error;
    return !expectedFailure.empty()&&error.find(expectedFailure)!=std::string::npos;
}

int main(int argc,char** argv) {
    if(argc!=6) { std::cerr << "usage: check <parent-mass> <parent-labels> <actual-node> <actual-ele> <actual-labels>" << std::endl; return 2; }
    const std::filesystem::path parentMass=argv[1],parentLabels=argv[2];
    const std::filesystem::path actualNode=argv[3],actualEle=argv[4],actualLabels=argv[5];
    std::string error;
    PhysicalSkinParentMassPacket actualParent;
    if(!loadCompiledSkinFatParentMassPacketCSV(parentMass,actualParent,error)||
       !loadCompiledSkinFatParentLabelsCSV(parentLabels,actualParent,error)) {
        std::cerr << "actual-parent input rejected: " << error << std::endl; return 3;
    }
    PhysicalSkinChildMassMappingBundle actual; actual.parent=actualParent;
    actual.outerSkin.name="preserved-outer-sidecar"; actual.volumeNodeDonorBodyIndices={77u};
    PhysicalSkinChildTetGenImportReport actualReport;
    const std::string priorFrame="cb8ccc78499a65cdf3404114f531cf379e1884a98003a622d2775a7d15639c60";
    if(!loadPhysicalSkinChildTetGenReference(actualNode,actualEle,actualLabels,priorFrame,
            157u,1.0e-6,actual,actualReport,error)||actualReport.nodeCount!=7233u||
       actualReport.cellCount!=28632u||actual.outerSkin.name!="preserved-outer-sidecar"||
       actual.volumeNodeDonorBodyIndices!=std::vector<std::uint32_t>{77u}) {
        std::cerr << "actual-parent TetGen import failed: " << error << std::endl; return 4;
    }
    std::cout << "actual_parent_import=PASS nodes=" << actualReport.nodeCount
              << " cells=" << actualReport.cellCount
              << " node_sha256=" << actualReport.nodeFileSHA256
              << " ele_sha256=" << actualReport.elementFileSHA256
              << " labels_sha256=" << actualReport.labelFileSHA256
              << " preloaded_sidecars_preserved=1 physical_admission=0" << std::endl;

    const auto dir=std::filesystem::temp_directory_path()/"numi-child-import-synthetic";
    std::filesystem::create_directories(dir);
    const std::vector<V3> ref{{0.10000000000000003,0.2,0.3},{1.1,0.2,0.3},
                              {0.1,1.2,0.3},{0.1,0.2,1.3},{0.1,0.2,2.3}};
    std::vector<V3> current=ref; for(auto& p:current)p[0]+=1.0e-11;
    std::vector<V3> capturedCurrent=current;
    for(auto& point:capturedCurrent)for(auto& coordinate:point)
        coordinate=static_cast<double>(static_cast<float>(coordinate));
    const auto parent=makeParent(ref,capturedCurrent);
    const std::vector<std::uint32_t> parentFace{7u};
    const std::vector<std::uint32_t> facetRows(4u,NM_INVALID_INDEX),facetStable(4u,NM_INVALID_INDEX);
    const std::vector<std::uint8_t> wallFlags(4u,0u);
    const std::vector<std::uint32_t> wallStable(4u,NM_INVALID_INDEX);
    std::vector<std::uint32_t> sourceVertex{0u,1u,2u,3u,4u};
    std::vector<std::uint32_t> sourceFace(5u,NM_INVALID_INDEX),nodeParent(5u,NM_INVALID_INDEX);
    std::vector<std::array<double,3>> sourceBary(5u);
    std::vector<std::array<double,4>> nodeParentBary(5u);
    std::vector<V3> nodeResidual(5u),nodeOrigin(5u);
    const std::array<double,9> identity{1,0,0,0,1,0,0,0,1};
    std::vector<std::array<double,9>> nodeLinear(5u,identity),bodyLinear{identity};
    std::vector<V3> bodyOrigin(1u);
    auto expected=makeExpected(parent,parentFace,facetRows,facetStable,wallFlags,wallStable,
        sourceVertex,sourceFace,sourceBary,nodeParent,nodeParentBary,nodeResidual,nodeLinear,nodeOrigin,
        bodyLinear,bodyOrigin);
    expected.currentWorldPoints=current;
    const auto node=dir/"child.node",ele=dir/"child.ele",labels=dir/"child.labels.csv";
    PhysicalSkinChildIdentityImportFiles files{parentMass,parentLabels,node,ele,labels};
    if(!writeNodes(node,ref)||!writeElements(ele,{0u,1u,2u,3u},15u)||!writeLabels(labels,15u,0u,3u,901u)) return 5;
    auto bundle=makeBundle(parent,expected,parentFace,facetRows,facetStable,wallFlags,wallStable,
        sourceVertex,sourceFace,sourceBary,nodeParent,nodeParentBary,nodeResidual,nodeLinear,nodeOrigin,
        bodyLinear,bodyOrigin);
    const auto frame=physical_skin_child_integration_detail::identityCoordinateFrameSHA256(parent,expected);
    PhysicalSkinChildTetGenImportReport parsed;
    if(!loadPhysicalSkinChildTetGenReference(node,ele,labels,frame,1u,1.0e-6,bundle,parsed,error)) {
        std::cerr << "synthetic child import failed: " << error << std::endl; return 6;
    }
    PhysicalSkinChildIdentityImportReport syntheticReport;
    syntheticReport.coordinateFrameSHA256=frame;
    syntheticReport.importedTetGenMeshSHA256=bundle.childMeshSHA256;
    const auto roundedReference=bundle.referenceWorldPoints;
    if(!validatePhysicalSkinChildIdentityImport(bundle,expected,syntheticReport,error)||
       syntheticReport.capturedParentCurrentExactF32Rows!=current.size()||
       syntheticReport.capturedParentReferenceExactF64Rows!=ref.size()||
       bundle.currentWorldPoints!=current||bundle.referenceWorldPoints!=ref||
       roundedReference==bundle.referenceWorldPoints||bundle.coordinateFrameSHA256!=frame||
       !physical_skin_child_mass_detail::validSHA256(bundle.childMeshSHA256)) {
        std::cerr << "exact F64 identity retention failed: " << error << std::endl; return 7;
    }
    std::cout << "synthetic_exact_frame=PASS current_reference_distinct=1 F32_input_retained_as_evidence=1" << std::endl;
    std::cout << "capture_representation=PASS current_exact_f32_rows="
              << syntheticReport.capturedParentCurrentExactF32Rows
              << " reference_exact_f64_rows=" << syntheticReport.capturedParentReferenceExactF64Rows
              << " live_f64_retained=1" << std::endl;

    if(!writeNodes(node,ref,4)||!writeElements(ele,{0u,1u,2u,3u},15u)||!writeLabels(labels,15u,0u,3u,901u)) return 8;
    std::string failure;
    if(!parseAndValidate(files,parent,expected,parentFace,facetRows,facetStable,wallFlags,wallStable,
        sourceVertex,sourceFace,sourceBary,nodeParent,nodeParentBary,nodeResidual,nodeLinear,nodeOrigin,
        bodyLinear,bodyOrigin,"IEEE-F32 emission",failure)) { std::cerr<<"node negative: "<<failure<<std::endl; return 9; }
    if(!writeNodes(node,ref)||!writeElements(ele,{0u,1u,2u,4u},15u)||!writeLabels(labels,15u,0u,3u,901u)) return 10;
    failure.clear();
    if(!parseAndValidate(files,parent,expected,parentFace,facetRows,facetStable,wallFlags,wallStable,
        sourceVertex,sourceFace,sourceBary,nodeParent,nodeParentBary,nodeResidual,nodeLinear,nodeOrigin,
        bodyLinear,bodyOrigin,"cells or explicit labels",failure)) { std::cerr<<"tet negative: "<<failure<<std::endl; return 11; }
    if(!writeNodes(node,ref)||!writeElements(ele,{0u,1u,2u,3u},16u)||!writeLabels(labels,16u,0u,3u,901u)) return 12;
    failure.clear();
    if(!parseAndValidate(files,parent,expected,parentFace,facetRows,facetStable,wallFlags,wallStable,
        sourceVertex,sourceFace,sourceBary,nodeParent,nodeParentBary,nodeResidual,nodeLinear,nodeOrigin,
        bodyLinear,bodyOrigin,"cells or explicit labels",failure)) { std::cerr<<"material negative: "<<failure<<std::endl; return 13; }
    if(!writeNodes(node,ref)||!writeElements(ele,{0u,1u,2u,3u},15u)||!writeLabels(labels,15u,0u,4u,901u)) return 14;
    failure.clear();
    if(!parseAndValidate(files,parent,expected,parentFace,facetRows,facetStable,wallFlags,wallStable,
        sourceVertex,sourceFace,sourceBary,nodeParent,nodeParentBary,nodeResidual,nodeLinear,nodeOrigin,
        bodyLinear,bodyOrigin,"captured parent topology and labels",failure)) { std::cerr<<"label negative: "<<failure<<std::endl; return 15; }
    PhysicalSkinChildIdentityImportFiles incomplete;
    PhysicalSkinChildMassMappingBundle empty; PhysicalSkinChildIdentityImportReport emptyReport;
    if(loadPhysicalSkinChildIdentityImport(incomplete,1u,expected,empty,emptyReport,error)||
       error.find("explicit parent mass/label")==std::string::npos) {
        std::cerr << "partial-path negative was not rejected" << std::endl; return 16;
    }
    std::cout << "negative_controls=PASS node_mismatch=tet_mismatch=material_mismatch=label_mismatch=partial_paths=REJECTED" << std::endl;
    std::filesystem::remove_all(dir);
    return 0;
}
