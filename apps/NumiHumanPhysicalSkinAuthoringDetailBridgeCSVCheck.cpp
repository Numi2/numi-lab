#include "NumiHumanPhysicalSkinAuthoringDetailBridgeCSV.hpp"

#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

namespace {
using Row=std::vector<std::string>;

std::string csvRow(const Row& fields) {
    std::string line;
    for(std::size_t i=0;i<fields.size();++i) {
        if(i)line.push_back(',');
        const auto& f=fields[i];
        const bool quote=f.find_first_of(",\"\r\n")!=std::string::npos;
        if(quote)line.push_back('"');
        for(const char c:f) {
            if(c=='"')line.push_back('"');
            line.push_back(c);
        }
        if(quote)line.push_back('"');
    }
    return line;
}
void writeCSV(const std::filesystem::path& path,const Row& header,
              const std::vector<Row>& rows,const bool crlf=false) {
    std::ofstream out(path,std::ios::binary);
    const std::string eol=crlf
        ?std::string(1u,static_cast<char>(13))+static_cast<char>(10)
        :std::string(1u,static_cast<char>(10));
    out<<csvRow(header)<<eol;
    for(const auto& row:rows)out<<csvRow(row)<<eol;
    out.close();
    assert(out.good());
}
void mutateFirstCSVRow(const std::filesystem::path& path,const std::size_t column,
                       const std::string& value) {
    std::ifstream in(path,std::ios::binary);
    std::string line;assert(std::getline(in,line));
    if(!line.empty()&&line.back()==static_cast<char>(13))line.pop_back();
    Row header;assert(physical_skin_authoring_detail_bridge_csv_detail::parseLine(line,header));
    std::vector<Row> rows;
    while(std::getline(in,line)) {
        if(!line.empty()&&line.back()==static_cast<char>(13))line.pop_back();
        Row row;assert(physical_skin_authoring_detail_bridge_csv_detail::parseLine(line,row));
        rows.push_back(std::move(row));
    }
    assert(!rows.empty()&&column<rows.front().size());
    rows.front()[column]=value;
    writeCSV(path,header,rows,true);
}
void mutateAllCSVRows(const std::filesystem::path& path,
                      const std::vector<std::pair<std::size_t,std::string>>& replacements) {
    std::ifstream in(path,std::ios::binary);
    std::string line;assert(std::getline(in,line));
    if(!line.empty()&&line.back()==static_cast<char>(13))line.pop_back();
    Row header;assert(physical_skin_authoring_detail_bridge_csv_detail::parseLine(line,header));
    std::vector<Row> rows;
    while(std::getline(in,line)) {
        if(!line.empty()&&line.back()==static_cast<char>(13))line.pop_back();
        Row row;assert(physical_skin_authoring_detail_bridge_csv_detail::parseLine(line,row));
        for(const auto& [column,value]:replacements) {
            assert(column<row.size());row[column]=value;
        }
        rows.push_back(std::move(row));
    }
    assert(!rows.empty());writeCSV(path,header,rows,true);
}
std::string fileSHA(const std::filesystem::path& path) {
    std::string sha,error;
    assert(physical_skin_child_mass_detail::sha256File(path,sha,error));
    return sha;
}
struct FaceAllowList {
    std::vector<std::uint8_t> allowed;
};
bool allowListedFace(const void* opaque,std::size_t row,std::size_t face) {
    const auto& list=*static_cast<const FaceAllowList*>(opaque);
    return row==0u&&face<list.allowed.size()&&list.allowed[face]!=0u;
}

void testTransferHonorsPerDetailFaceEligibility() {
    using Transfer=numi::human::AcceptedFEMSkinTransfer;
    const std::vector<Transfer::Point> detail{{0.2,0.2,-0.0003}};
    const std::vector<std::uint32_t> detailSource{0x80000000u};
    const std::vector<std::uint32_t> detailDonor{66u};
    const std::vector<std::uint8_t> used{1u};
    const std::vector<Transfer::Point> coarse{
        {0,0,0},{1,0,0},{0,1,0},{1,0,-0.0004},{0,1,-0.0004}};
    const std::vector<std::uint32_t> coarseSource(5u,NM_INVALID_INDEX);
    const std::vector<std::uint32_t> coarseDonor(5u,66u);
    const std::vector<Transfer::Face> faces{{0u,1u,2u},{0u,3u,4u}};
    const std::vector<std::uint32_t> representatives{0u},stableIDs{9002u,9100u};
    Transfer transfer;Transfer::BuildReport report;std::string error;
    assert(transfer.build(detail,detailSource,detailDonor,used,coarse,coarseSource,
        coarseDonor,faces,&report,&error,0.02,&representatives,nullptr,nullptr,nullptr,1u,
        &stableIDs,nullptr));
    assert(transfer.selectedFaceIndex(0u)==1u);
    FaceAllowList onlyLegitimate{{1u,0u}};
    assert(transfer.build(detail,detailSource,detailDonor,used,coarse,coarseSource,
        coarseDonor,faces,&report,&error,0.02,&representatives,nullptr,nullptr,nullptr,1u,
        &stableIDs,nullptr,allowListedFace,&onlyLegitimate));
    assert(transfer.selectedFaceIndex(0u)==0u);
    assert(std::abs(transfer.rows()[0].restDistance-0.0003f)<1.0e-7f);
    FaceAllowList noEligible{{0u,0u}};
    assert(!transfer.build(detail,detailSource,detailDonor,used,coarse,coarseSource,
        coarseDonor,faces,&report,&error,0.02,&representatives,nullptr,nullptr,nullptr,1u,
        &stableIDs,nullptr,allowListedFace,&noEligible));
    assert(error.find("typed per-detail face eligibility rejected 1 rows")!=std::string::npos);
}

template<class T> std::string n(T value) { return std::to_string(value); }
std::string d(double value) { return std::to_string(value); }

struct Fixture {
    std::filesystem::path root;
    PhysicalSkinChildMassMappingBundle bundle;
    PhysicalSkinAuthoringRoleSurfaceRows boundary;
    PhysicalSkinAuthoringDetailBridgeCSVFiles files;

    Fixture() {
        root=std::filesystem::temp_directory_path()/"numi-detail-bridge-csv-check";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        bundle.parent.referenceWorldPoints={{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,-1}};
        bundle.parent.currentWorldPoints=bundle.parent.referenceWorldPoints;
        bundle.referenceWorldPoints=bundle.parent.referenceWorldPoints;
        bundle.currentWorldPoints=bundle.parent.currentWorldPoints;
        bundle.parent.tetrahedra={{{0u,1u,2u,3u}},{{0u,1u,2u,4u}}};
        bundle.parent.materialIndices={15u,15u};
        bundle.parent.donorBodyIndices={41u,41u};
        bundle.parent.anatomicalRegionIndices={7u,7u};
        bundle.parent.stableSourceFaceIdentities={9001u,9001u};
        bundle.childTetrahedra=bundle.parent.tetrahedra;
        bundle.childMaterialIndices=bundle.parent.materialIndices;
        bundle.childDonorBodyIndices=bundle.parent.donorBodyIndices;
        bundle.childAnatomicalRegionIndices=bundle.parent.anatomicalRegionIndices;
        bundle.childDominantParentTetrahedra={0u,1u};

        boundary.sourceDomain={"reduced-physical-O",PhysicalSkinSourceDomainRole::outerSurface,
            std::string(64u,'b'),3u,1u};
        boundary.authoringVertexRows={0u,1u,2u};
        boundary.volumeNodeRows={0u,1u,2u};
        boundary.currentWorldPoints={{0,0,0},{1,0,0},{0,1,0}};
        boundary.frameOwnerBodyIndices={88u,88u,88u};
        boundary.parentGeometricDonorBodyIndices={41u,41u,41u};
        boundary.roleLocalFaces={{0u,1u,2u}};

        files.detailSourceAsset.path=root/"candidate.npz";
        {
            std::ofstream asset(files.detailSourceAsset.path,std::ios::binary);
            asset<<"pinned-detail-source-fixture";
        }
        std::string sourceSHA,error;
        assert(physical_skin_child_mass_detail::sha256File(
            files.detailSourceAsset.path,sourceSHA,error));
        files.detailSourceAsset.expectedSHA256=sourceSHA;
        files.detailDomain={"derived_registered_skin_authoring",
            PhysicalSkinSourceDomainRole::outerSurface,sourceSHA,3u,1u};
        files.detailNodes.path=root/"registered-O.csv";
        files.detailFaces.path=root/"registered-O-faces.csv";
        files.detailFull86Weights.path=root/"parent-tet-full-weights-O.csv";
        files.detailFull86Owners.path=root/"parent-tet-full-weight-owners-O.csv";
        files.fineToBoundaryMap.path=root/"fine-O-to-coarse-boundary-canonical.csv";
        files.historicalFineToBoundaryMap.path=root/"fine-O-to-coarse-boundary-history.csv";
        files.canonicalTetGenNode.path=root/"mesh.node";
        files.coarseBoundaryFaces.path=root/"coarse-boundary-faces.csv";
        emitFiles();
    }

    void emitFiles() {
        const Row nodeHeader{
            "vertex","donor","wx","wy","wz","nx","ny","nz","lx","ly","lz",
            "m00","m01","m02","m10","m11","m12","m20","m21","m22",
            "source_domain","source_mapping_kind","source_vertex_row","source_face_row",
            "source_bary0","source_bary1","source_bary2","source_residual_x_m",
            "source_residual_y_m","source_residual_z_m","source_registration_method",
            "frame_owner_method","parent_registration_kind","parent_tet_row",
            "parent_bary0","parent_bary1","parent_bary2","parent_bary3",
            "parent_residual_x_m","parent_residual_y_m","parent_residual_z_m",
            "parent_registration_distance_m","parent_registration_tie_count",
            "parent_tied_tet_rows_json","parent_material_index",
            "parent_geometric_donor_body","parent_anatomical_region_id",
            "parent_stable_source_face_id"};
        const std::array<V3,3> points{{{0.2,0.2,-0.005},{0.6,0.2,-0.005},{0.2,0.6,-0.005}}};
        const std::array<std::array<double,3>,3> bary{{{{0.6,0.2,0.2}},{{0.2,0.6,0.2}},{{0.2,0.2,0.6}}}};
        std::vector<Row> nodes,weights,owners,maps;
        for(std::uint32_t i=0;i<3u;++i) {
            Row row(48u);
            row[0]=n(i);row[1]="66";
            for(unsigned k=0;k<3u;++k)row[2u+k]=d(points[i][k]);
            row[5]="0";row[6]="0";row[7]="1";
            for(unsigned k=8;k<20u;++k)row[k]="0";
            row[20]=files.detailDomain.domain;row[21]="vertex";
            row[22]=n(i);row[23]="-1";row[24]=row[25]=row[26]="0";
            row[27]=row[28]=row[29]="0";row[30]="candidate_surface_npz_direct_vertex_identity";
            row[31]="parent_tet_full86_dominant_binding_current_pose";
            row[32]="inferred_current_parent_tet_same_material_donor_region";
            row[33]="0";row[34]="0.6";row[35]="0.2";row[36]="0.2";row[37]="0";
            row[38]=row[39]=row[40]="0";row[41]="0.005";row[42]="0";
            row[42]="2";row[43]="[0,1]";row[44]="15";row[45]="41";row[46]="7";row[47]="9001";
            nodes.push_back(row);
            Row weight(87u,"0");weight[0]=n(i);weight[1u+7u]="1";weights.push_back(weight);
            owners.push_back({n(i),"7","77","41","0","0.6","0.2","0.2","0","7"});

            Row map(45u);
            map[0]=n(i);map[1]="O";map[2]="0";map[3]="0";map[4]="0";map[5]="1";
            map[6]="1";map[7]="[1,0]";map[8]="0";map[9]="1";map[10]="2";
            map[11]="15";map[12]="41";map[13]="7";map[14]="9001";
            map[15]="15";map[16]="41";map[17]="7";map[18]="9001";
            for(unsigned k=0;k<3u;++k)map[19u+k]=d(bary[i][k]);
            map[22]="0";map[23]="0";map[24]="-0.005";map[25]="0.005";
            map[26]="primary_registered_parent_tet_boundary_stable_face";
            map[27]="15";map[28]="41";map[29]="7";map[30]="9001";
            map[31]="inferred_current_parent_tet_same_material_donor_region";
            map[32]="0";map[33]="[0,1]";map[34]="0.005";map[35]="2";
            map[36]="parent_tet_full86_dominant_binding_current_pose";
            map[37]=files.detailDomain.domain;map[38]="vertex";map[39]=n(i);map[40]="-1";
            map[41]=map[42]=map[43]="0";map[44]="[0]";
            maps.push_back(map);
        }
        writeCSV(files.detailNodes.path,nodeHeader,nodes);
        writeCSV(files.detailFaces.path,{"face","v0","v1","v2","face_source_layer",
            "face_source_body","face_source_bone","face_source_face_id","face_source_original_id",
            "face_source_identity0","face_source_identity1"},
            {{"0","0","1","2","0","41","-1","9001","9001","0","0"}});
        Row weightHeader{"vertex"};for(unsigned i=0;i<86u;++i)weightHeader.push_back("w"+n(i));
        writeCSV(files.detailFull86Weights.path,weightHeader,weights);
        writeCSV(files.detailFull86Owners.path,{"vertex","dominant_binding_index",
            "dominant_binding_body","parent_geometric_donor_body","parent_tet_row",
            "parent_bary0","parent_bary1","parent_bary2","parent_bary3",
            "parent_anatomical_region_id"},owners);
        const Row historicalMapHeader{
            "fine_vertex_row","role","coarse_face_row","coarse_boundary_primary_parent_tet_row",
            "mapped_boundary_side_parent_tet_row","boundary_walk_start_parent_tet_row",
            "boundary_walk_hop_count","boundary_walk_parent_tet_rows_json","volume_node0",
            "volume_node1","volume_node2","coarse_primary_material_index",
            "coarse_primary_geometric_donor_body","coarse_primary_region_id",
            "coarse_primary_stable_source_face_id","mapped_side_material_index",
            "mapped_side_geometric_donor_body","mapped_side_region_id",
            "mapped_side_stable_source_face_id","bary0","bary1","bary2","residual_x_m",
            "residual_y_m","residual_z_m","projection_distance_m","mapping_kind",
            "fine_parent_material_index","fine_parent_geometric_donor_body","fine_parent_region_id",
            "fine_parent_stable_source_face_id","fine_parent_registration_kind",
            "fine_parent_tet_row","fine_parent_tied_tet_rows_json",
            "fine_parent_registration_distance_m","fine_parent_registration_tie_count",
            "fine_frame_owner_body","fine_source_domain","fine_source_mapping_kind",
            "fine_source_vertex_row","fine_source_face_row","fine_source_bary0",
            "fine_source_bary1","fine_source_bary2","fine_incident_detail_face_rows_json"};
        writeCSV(files.historicalFineToBoundaryMap.path,historicalMapHeader,maps,true);
        {
            std::ofstream node(files.canonicalTetGenNode.path,std::ios::binary);
            node<<"5 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n5 0 0 -1\n";
        }
        std::string historicalSHA,nodeSHA,error;
        assert(physical_skin_child_mass_detail::sha256File(
            files.historicalFineToBoundaryMap.path,historicalSHA,error));
        assert(physical_skin_child_mass_detail::sha256File(
            files.canonicalTetGenNode.path,nodeSHA,error));
        files.historicalFineToBoundaryMap.expectedSHA256=historicalSHA;
        files.canonicalTetGenNode.expectedSHA256=nodeSHA;
        Row canonicalHeader=historicalMapHeader;
        canonicalHeader.insert(canonicalHeader.end(),{
            "historical_producer_map_sha256","canonical_tetgen_node_sha256",
            "canonical_bary0","canonical_bary1","canonical_bary2",
            "canonical_residual_x_m","canonical_residual_y_m","canonical_residual_z_m",
            "canonical_projection_distance_m"});
        auto canonicalRows=maps;
        for(auto& row:canonicalRows) {
            row.push_back(historicalSHA);row.push_back(nodeSHA);
            row.push_back(row[19]);row.push_back(row[20]);row.push_back(row[21]);
            row.push_back("0");row.push_back("0");row.push_back("-0.005");row.push_back("0.005");
        }
        writeCSV(files.fineToBoundaryMap.path,canonicalHeader,canonicalRows,true);
        writeCSV(files.coarseBoundaryFaces.path,{"role","face_row","parent_tet_row",
            "volume_node0","volume_node1","volume_node2","material_index",
            "geometric_donor_body","anatomical_region_id","stable_source_face_id",
            "opposite_parent_tet_row","opposite_material_index",
            "opposite_geometric_donor_body","opposite_anatomical_region_id",
            "opposite_stable_source_face_id"},
            {{"O","0","0","0","1","2","15","41","7","9001","-1","-1","-1","-1","-1"}},true);
        assert(physical_skin_child_mass_detail::sha256File(files.detailNodes.path,files.detailNodes.expectedSHA256,error));
        assert(physical_skin_child_mass_detail::sha256File(files.detailFaces.path,files.detailFaces.expectedSHA256,error));
        assert(physical_skin_child_mass_detail::sha256File(files.detailFull86Weights.path,files.detailFull86Weights.expectedSHA256,error));
        assert(physical_skin_child_mass_detail::sha256File(files.detailFull86Owners.path,files.detailFull86Owners.expectedSHA256,error));
        assert(physical_skin_child_mass_detail::sha256File(files.fineToBoundaryMap.path,files.fineToBoundaryMap.expectedSHA256,error));
        assert(physical_skin_child_mass_detail::sha256File(files.coarseBoundaryFaces.path,files.coarseBoundaryFaces.expectedSHA256,error));
    }
};

void writeRefinedCoarseFace(Fixture& f,const std::uint32_t child,const std::uint32_t parent) {
    writeCSV(f.files.coarseBoundaryFaces.path,
        {"role","face_row","child_tet_row","parent_tet_row","volume_node0","volume_node1",
         "volume_node2","material_index","geometric_donor_body","anatomical_region_id",
         "stable_source_face_id","opposite_parent_tet_row","opposite_material_index",
         "opposite_geometric_donor_body","opposite_anatomical_region_id",
         "opposite_stable_source_face_id"},
        {{"O","0",n(child),n(parent),"0","1","2","15","41","7","9001",
          "-1","-1","-1","-1","-1"}},true);
    std::string sha,error;
    assert(physical_skin_child_mass_detail::sha256File(
        f.files.coarseBoundaryFaces.path,sha,error));
    f.files.coarseBoundaryFaces.expectedSHA256=sha;
}

void authenticateFixtureRefinementRoute(PhysicalSkinChildMassMappingBundle& bundle) {
    bundle.conformingLineageMassRouteRequired=true;
    bundle.conformingLineageMassTransportUsed=true;
    bundle.usesSeparatedOperationalReference=true;
    bundle.operationalReferenceDomainSHA256=std::string(64u,'b');
    auto& route=bundle.conformingLineageMassRoute;
    route.authenticated=true;
    route.refinedSourceCellCount=3u;
    route.childRefinedSourceRows={0u,1u,2u};
    route.childImmutableParentRows={1u,0u,1u};
    for(auto* value:{&route.parentMassPacketSHA256,&route.parentSourceContributionSHA256,
            &route.sourceVolumeSHA256,&route.refinedVolumeSHA256,&route.candidateVolumeSHA256,
            &route.refinementVerificationSHA256,&route.sourceDonorLedgerVerificationSHA256,
            &route.pullbackVerificationSHA256,&route.chainVerificationSHA256})
        *value=std::string(64u,'a');
    route.candidateOperationalReferenceDomainSHA256=bundle.operationalReferenceDomainSHA256;
    route.candidateCompiledDataSHA256=
        physical_skin_child_mass_mapping_detail::operationalChildCompiledDataSHA256(bundle);
    route.verificationSHA256=
        physical_skin_child_mass_mapping_detail::verifiedConformingMassRouteSHA256(route);
}

void testRefinedReorderedChildBoundaryUsesAuthenticatedParentRows() {
    Fixture f;
    // Reordered child domain with a real bisection of parent row 1 along
    // edge (0,4). Child row 1 remains unchanged parent row 0 for the O face.
    // The two child tets partition parent row 1 at the midpoint node 5.
    const auto midpoint=f.bundle.parent.referenceWorldPoints[0];
    const auto endpoint=f.bundle.parent.referenceWorldPoints[4];
    f.bundle.referenceWorldPoints.push_back({
        0.5*(midpoint[0]+endpoint[0]),
        0.5*(midpoint[1]+endpoint[1]),
        0.5*(midpoint[2]+endpoint[2])});
    f.bundle.currentWorldPoints=f.bundle.referenceWorldPoints;
    f.bundle.childTetrahedra={{{0u,1u,2u,5u}},
                              f.bundle.parent.tetrahedra[0],
                              {{5u,1u,2u,4u}}};
    f.bundle.childMaterialIndices={15u,15u,15u};
    f.bundle.childDonorBodyIndices={41u,41u,41u};
    f.bundle.childAnatomicalRegionIndices={7u,7u,7u};
    f.bundle.childStableSourceFaceIdentities={9001u,9001u,9001u};
    f.bundle.childDominantParentTetrahedra={1u,0u,1u};
    const auto signedSixVolume=[](const std::vector<V3>& points,
                                  const std::array<std::uint32_t,4>& tet) {
        const auto a=points[tet[0]],b=points[tet[1]],c=points[tet[2]],d=points[tet[3]];
        const V3 ab{b[0]-a[0],b[1]-a[1],b[2]-a[2]};
        const V3 ac{c[0]-a[0],c[1]-a[1],c[2]-a[2]};
        const V3 ad{d[0]-a[0],d[1]-a[1],d[2]-a[2]};
        return ab[0]*(ac[1]*ad[2]-ac[2]*ad[1])-
               ab[1]*(ac[0]*ad[2]-ac[2]*ad[0])+
               ab[2]*(ac[0]*ad[1]-ac[1]*ad[0]);
    };
    const auto parentSixVolume=signedSixVolume(
        f.bundle.parent.referenceWorldPoints,f.bundle.parent.tetrahedra[1]);
    const auto childSixVolume=signedSixVolume(
        f.bundle.referenceWorldPoints,f.bundle.childTetrahedra[0])+
        signedSixVolume(f.bundle.referenceWorldPoints,f.bundle.childTetrahedra[2]);
    assert(std::abs(childSixVolume-parentSixVolume)<1.0e-15);
    authenticateFixtureRefinementRoute(f.bundle);
    f.files.allowPrecanonicalExportSeed=true;
    writeRefinedCoarseFace(f,1u,0u);

    PhysicalSkinAuthoringDetailBridgeInput input;
    PhysicalSkinAuthoringDetailBridgeCSVReport report;
    std::string error;
    assert(loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,report,error));
    assert(input.physicalBoundaryFaceParentTetRows==std::vector<std::uint32_t>({0u}));
    assert(input.detailMechanicalTetRows==std::vector<std::uint32_t>({1u,1u,1u}));
    assert(input.detailMechanicalTetRows[0]!=input.physicalBoundaryFaceParentTetRows[0]);

    // Canonical map parsing is separate from seed-map parsing. A source-only
    // caller may load a refined child route only with the dedicated opt-in.
    const auto canonicalPath=f.root/"canonical-refined-map.csv";
    PhysicalSkinAuthoringDetailBridgeCanonicalExportReport exportReport;
    assert(exportPhysicalSkinAuthoringDetailBridgeCanonicalMap(
        f.files,input,report,canonicalPath,exportReport,error));
    assert(!exportReport.physicalAdmission);
    auto canonicalFiles=f.files;
    canonicalFiles.allowPrecanonicalExportSeed=false;
    canonicalFiles.fineToBoundaryMap.path=canonicalPath;
    canonicalFiles.fineToBoundaryMap.expectedSHA256=exportReport.outputSHA256;
    canonicalFiles.allowAuthenticatedRefinedChildRouteForSourceOnlyPreflight=true;
    PhysicalSkinAuthoringDetailBridgeInput canonicalInput;
    PhysicalSkinAuthoringDetailBridgeCSVReport canonicalReport;
    assert(loadPhysicalSkinAuthoringDetailBridgeCSV(
        canonicalFiles,f.bundle,f.boundary,canonicalInput,canonicalReport,error));
    assert(canonicalInput.detailMechanicalTetRows==std::vector<std::uint32_t>({1u,1u,1u}));
    assert(canonicalInput.physicalBoundaryFaceParentTetRows==std::vector<std::uint32_t>({0u}));
    assert(!canonicalReport.physicalAdmission);

    // The same refined canonical map is rejected without the source-only
    // authorization, preserving the default identity-only runtime contract.
    canonicalFiles.allowAuthenticatedRefinedChildRouteForSourceOnlyPreflight=false;
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        canonicalFiles,f.bundle,f.boundary,canonicalInput,canonicalReport,error));
    assert(error.find("runtime detail bridge requires exact child/parent cell identity")!=std::string::npos);

    // A child face row that claims the wrong authenticated parent must fail.
    writeRefinedCoarseFace(f,1u,1u);
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,report,error));
    assert(error.find("child-to-parent ancestry")!=std::string::npos);

    // The operational runtime path remains identity-only, even with a valid
    // canonical-export route and explicit child/parent columns.
    f.files.allowPrecanonicalExportSeed=false;
    writeRefinedCoarseFace(f,1u,0u);
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,report,error));
    assert(error.find("runtime detail bridge requires exact child/parent cell identity")!=std::string::npos);
}

void rewritePinnedMapPair(Fixture& f,
        const std::vector<std::pair<std::size_t,std::string>>& replacements) {
    mutateAllCSVRows(f.files.historicalFineToBoundaryMap.path,replacements);
    f.files.historicalFineToBoundaryMap.expectedSHA256=
        fileSHA(f.files.historicalFineToBoundaryMap.path);
    auto canonicalReplacements=replacements;
    canonicalReplacements.emplace_back(45u,f.files.historicalFineToBoundaryMap.expectedSHA256);
    mutateAllCSVRows(f.files.fineToBoundaryMap.path,canonicalReplacements);
    f.files.fineToBoundaryMap.expectedSHA256=fileSHA(f.files.fineToBoundaryMap.path);
}

void prepareStableFaceWalkFixture(Fixture& f) {
    f.bundle.parent.donorBodyIndices={66u,65u};
    f.bundle.parent.stableSourceFaceIdentities={9001u,9002u};
    f.bundle.childDonorBodyIndices={66u,65u};
    f.boundary.parentGeometricDonorBodyIndices={65u,65u,65u};
    f.emitFiles();
    mutateAllCSVRows(f.files.detailNodes.path,{{45u,"66"}});
    f.files.detailNodes.expectedSHA256=fileSHA(f.files.detailNodes.path);
    mutateAllCSVRows(f.files.detailFull86Owners.path,{{3u,"66"}});
    f.files.detailFull86Owners.expectedSHA256=fileSHA(f.files.detailFull86Owners.path);
    writeCSV(f.files.coarseBoundaryFaces.path,
        {"role","face_row","parent_tet_row","volume_node0","volume_node1","volume_node2",
         "material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id",
         "opposite_parent_tet_row","opposite_material_index","opposite_geometric_donor_body",
         "opposite_anatomical_region_id","opposite_stable_source_face_id"},
        {{"O","0","1","0","1","2","15","65","7","9002",
          "-1","-1","-1","-1","-1"}},true);
    f.files.coarseBoundaryFaces.expectedSHA256=fileSHA(f.files.coarseBoundaryFaces.path);
    rewritePinnedMapPair(f,{{3u,"1"},{4u,"1"},{5u,"0"},{6u,"1"},{7u,"[0,1]"},
        {11u,"15"},{12u,"65"},{13u,"7"},{14u,"9002"},{15u,"15"},{16u,"65"},{17u,"7"},
        {18u,"9002"},{26u,"bounded_same_material_region_parent_tet_walk_to_stable_boundary_face"},
        {27u,"15"},{28u,"66"},{29u,"7"},{30u,"9001"},{32u,"0"},{33u,"[0,1]"},{35u,"2"}});
}

void testCSVBridgeAllowsOnlyAuthenticatedStableFaceWalk() {
    Fixture f;f.files.allowPrecanonicalExportSeed=true;
    prepareStableFaceWalkFixture(f);
    PhysicalSkinAuthoringDetailBridgeInput input;
    PhysicalSkinAuthoringDetailBridgeCSVReport report;
    std::string error;
    const bool loaded=loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,report,error);
    if(!loaded)std::cerr<<"stable_face_walk_error="<<error<<"\n";
    assert(loaded);
    assert(input.detailStableSourceFaceIDs==std::vector<std::uint32_t>({9001u,9001u,9001u}));
    assert(input.physicalBoundaryStableFaceIDs==std::vector<std::uint32_t>({9002u}));
    assert(input.detailGeometricDonorBodyIndices==std::vector<std::uint32_t>({66u,66u,66u}));
    assert(input.detailBoundaryProjectionDonorBodyIndices==std::vector<std::uint32_t>({65u,65u,65u}));
    assert(input.detailRequiresParentTetPath==std::vector<std::uint8_t>({1u,1u,1u}));
    assert(!report.physicalAdmission);

    // A changed fine-source stable ID cannot be laundered through the walk.
    prepareStableFaceWalkFixture(f);
    rewritePinnedMapPair(f,{{30u,"9999"}});
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,report,error));
    assert(error.find("fine source, frame-owner, or parent labels")!=std::string::npos);

    // The start and target rows must belong to the authenticated source mesh.
    prepareStableFaceWalkFixture(f);
    f.bundle.parent.stableSourceFaceIdentities[0]=9999u;
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,report,error));
    assert(error.find("registered detail ancestry labels differ from the pinned parent cell")!=std::string::npos);

    // A path whose endpoints are valid but whose consecutive cells are not
    // adjacent is not a bounded walk.
    prepareStableFaceWalkFixture(f);
    rewritePinnedMapPair(f,{{6u,"2"},{7u,"[0,0,1]"}});
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,report,error));
    assert(error.find("nonadjacent consecutive source cells")!=std::string::npos);

    // A wrong target stable ID remains rejected by the authenticated coarse
    // parent/child source route.
    prepareStableFaceWalkFixture(f);
    writeCSV(f.files.coarseBoundaryFaces.path,
        {"role","face_row","parent_tet_row","volume_node0","volume_node1","volume_node2",
         "material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id",
         "opposite_parent_tet_row","opposite_material_index","opposite_geometric_donor_body",
         "opposite_anatomical_region_id","opposite_stable_source_face_id"},
        {{"O","0","1","0","1","2","15","65","7","9001",
          "-1","-1","-1","-1","-1"}},true);
    f.files.coarseBoundaryFaces.expectedSHA256=fileSHA(f.files.coarseBoundaryFaces.path);
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,report,error));
    assert(error.find("child-to-parent ancestry or labels")!=std::string::npos);
}

void testCanonicalEligibilityReinfersStableFace() {
    PhysicalSkinAuthoringDetailBridgeInput input;
    input.detailReferenceWorldPoints={{0.2,0.2,0.0}};
    input.detailMaterialIndices={15u};
    input.detailAnatomicalRegionIndices={7u};
    input.detailParentTetRows={0u};
    input.detailRegisteredSourceDonorBodyIndices={66u};
    input.detailGeometricDonorBodyIndices={65u};
    input.sourceParentTetNodeRows={{{0u,1u,2u,3u}},{{0u,1u,2u,4u}},{{5u,6u,7u,8u}}};
    input.sourceParentTetMaterialIndices={15u,15u,16u};
    input.sourceParentTetAnatomicalRegionIndices={7u,7u,8u};
    input.sourceParentTetStableSourceFaceIDs={9001u,9002u,9100u};
    input.physicalBoundaryFaceParentTetRows={1u,2u};
    input.physicalBoundaryFaceMaterialIndices={15u,16u};
    input.physicalBoundaryFaceAnatomicalRegionIndices={7u,8u};
    input.physicalBoundaryFaceFrameOwnerBodyIndices={77u,88u};
    input.maximumSourceParentTetHops=8u;
    std::vector<std::vector<std::uint32_t>> adjacency;
    std::string error;
    assert(physical_skin_authoring_detail_bridge_detail::buildParentTetFaceAdjacency(
        input.sourceParentTetNodeRows,adjacency,error));
    PhysicalSkinAuthoringDetailFaceEligibilityContext context{&input,&adjacency};
    // The changed stable face is a valid re-inference: exact material/region
    // and a bounded source-tet path are preserved while frame and geometry
    // donors remain distinct labels.
    assert(physicalSkinAuthoringDetailFaceEligible(
        context,0u,0u));
    // A closer unrelated region/material remains outside the eligible domain.
    assert(!physicalSkinAuthoringDetailFaceEligible(
        context,0u,1u));
}

void testPinnedCSVBridge() {
    Fixture f;
    f.files.allowPrecanonicalExportSeed=true;
    PhysicalSkinAuthoringDetailBridgeInput input;
    PhysicalSkinAuthoringDetailBridgeCSVReport csvReport;
    std::string error;
    const bool loaded=loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,input,csvReport,error);
    if(!loaded)std::cerr<<"detail_bridge_csv_error="<<error<<"\n";
    assert(loaded);
    assert(input.detailReferenceWorldPoints.size()==3u&&input.detailFaces.size()==1u);
    assert(input.detailReferenceWorldNormals.size()==3u);
    assert(input.detailFullBodyWeights86.size()==3u&&
           input.detailFullBodyWeights86[1][7]==1.0&&
           input.detailDominantBindingIndices==std::vector<std::uint32_t>({7u,7u,7u}));
    assert(input.detailFullBodyWeights86.size()==3u&&
           input.detailFullBodyWeights86[1][7]==1.0&&
           input.detailDominantBindingIndices==std::vector<std::uint32_t>({7u,7u,7u}));
    assert(input.detailFrameOwnerBodyIndices==std::vector<std::uint32_t>({77u,77u,77u}));
    assert(input.detailGeometricDonorBodyIndices==std::vector<std::uint32_t>({41u,41u,41u}));
    assert(input.detailRegisteredSourceDonorBodyIndices==
           std::vector<std::uint32_t>({66u,66u,66u}));
    assert(input.detailBoundaryProjectionDonorBodyIndices==std::vector<std::uint32_t>({41u,41u,41u}));
    assert(input.detailRequiresParentTetPath==std::vector<std::uint8_t>({1u,1u,1u}));
    assert(input.detailExpectedBoundaryFaceRows==std::vector<std::uint32_t>({0u,0u,0u}));
    assert(input.detailMechanicalTetRows==std::vector<std::uint32_t>({0u,0u,0u}));
    assert(csvReport.detailRows==3u&&csvReport.detailFaces==1u);
    assert(csvReport.maximumRecomputedProjectionDistanceM<=0.005001);
    assert(csvReport.inputSHA256[0]==f.files.detailNodes.expectedSHA256);
    assert(csvReport.historicalMapSHA256==f.files.historicalFineToBoundaryMap.expectedSHA256);
    assert(csvReport.canonicalTetGenNodeSHA256==f.files.canonicalTetGenNode.expectedSHA256);
    assert(!csvReport.physicalAdmission);
    PhysicalSkinAuthoringDetailBridgeState bridge;
    PhysicalSkinAuthoringDetailBridgeReport bridgeReport;
    assert(preparePhysicalSkinAuthoringDetailBridge(input,bridge,bridgeReport,error));
    assert(bridge.tetraRows.size()==3u&&!bridge.physicalAdmission);

    // A same-donor ancestry walk is distinct from a donor remap and must still
    // be recorded as a path requirement by the native bridge helper.
    input.detailRequiresParentTetPath={1u,1u,1u};
    input.sourceParentNodeCount=5u;
    input.sourceParentTetNodeRows={{{0u,1u,2u,3u}},{{1u,2u,3u,4u}}};
    input.sourceParentTetMaterialIndices={15u,15u};
    input.sourceParentTetAnatomicalRegionIndices={7u,7u};
    input.sourceParentTetStableSourceFaceIDs={9001u,9001u};
    input.physicalBoundaryFaceParentTetRows={0u};
    assert(preparePhysicalSkinAuthoringDetailBridge(input,bridge,bridgeReport,error));
    assert(bridgeReport.sourceParentTetWalkRows==3u&&bridgeReport.regionScopedDonorFallbackRows==0u);
    assert(bridge.sourceParentTetHopCounts==std::vector<std::uint32_t>({0u,0u,0u}));

    const auto priorPoints=input.detailReferenceWorldPoints;
    const auto priorRows=bridge.tetraRows;
    const auto refreshMapPin=[&] {
        assert(physical_skin_child_mass_detail::sha256File(
            f.files.fineToBoundaryMap.path,f.files.fineToBoundaryMap.expectedSHA256,error));
    };
    mutateFirstCSVRow(f.files.fineToBoundaryMap.path,46u,std::string(64u,'0'));
    refreshMapPin();
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(f.files,f.bundle,f.boundary,input,csvReport,error));
    assert(input.detailReferenceWorldPoints==priorPoints);
    f.emitFiles();
    mutateFirstCSVRow(f.files.fineToBoundaryMap.path,50u,"0.01");
    refreshMapPin();
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(f.files,f.bundle,f.boundary,input,csvReport,error));
    assert(input.detailReferenceWorldPoints==priorPoints);
    f.emitFiles();
    mutateFirstCSVRow(f.files.fineToBoundaryMap.path,22u,"0.0001");
    refreshMapPin();
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(f.files,f.bundle,f.boundary,input,csvReport,error));
    assert(input.detailReferenceWorldPoints==priorPoints);
    f.emitFiles();
    f.files.detailNodes.expectedSHA256=std::string(64u,'0');
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(f.files,f.bundle,f.boundary,input,csvReport,error));
    assert(input.detailReferenceWorldPoints==priorPoints);
    assert(bridge.tetraRows.size()==priorRows.size()&&
           bridge.tetraRows[0].node[0]==priorRows[0].node[0]&&
           bridge.tetraRows[0].weight[0]==priorRows[0].weight[0]);
}


void testCanonicalNativeFaceExport() {
    Fixture f;
    f.files.allowPrecanonicalExportSeed=true;
    PhysicalSkinAuthoringDetailBridgeInput seedInput;
    PhysicalSkinAuthoringDetailBridgeCSVReport seedReport;
    std::string error;
    assert(loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,seedInput,seedReport,error));
    // The historical path begins at a tied registration tet (1), while the
    // native selector must emit a canonical path from the primary parent (0).
    assert(seedInput.detailMechanicalTetRows[0]==0u);
    auto badSeedFiles=f.files;
    mutateFirstCSVRow(badSeedFiles.fineToBoundaryMap.path,5u,"2");
    assert(physical_skin_child_mass_detail::sha256File(
        badSeedFiles.fineToBoundaryMap.path,badSeedFiles.fineToBoundaryMap.expectedSHA256,error));
    PhysicalSkinAuthoringDetailBridgeInput rejectedSeed=seedInput;
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        badSeedFiles,f.bundle,f.boundary,rejectedSeed,seedReport,error));
    assert(rejectedSeed.detailReferenceWorldPoints==seedInput.detailReferenceWorldPoints);
    f.emitFiles();
    assert(loadPhysicalSkinAuthoringDetailBridgeCSV(
        f.files,f.bundle,f.boundary,seedInput,seedReport,error));
    const auto canonicalPath=f.root/"canonical-native-face-map.csv";
    PhysicalSkinAuthoringDetailBridgeCanonicalExportReport exportReport;
    assert(exportPhysicalSkinAuthoringDetailBridgeCanonicalMap(
        f.files,seedInput,seedReport,canonicalPath,exportReport,error));
    assert(exportReport.detailRows==3u&&exportReport.changedFaceRows==0u);
    assert(exportReport.canonicalTetGenNodeSHA256==f.files.canonicalTetGenNode.expectedSHA256);
    assert(!exportReport.physicalAdmission);
    std::string seedBytes,canonicalBytes;
    {
        std::ifstream in(f.files.fineToBoundaryMap.path,std::ios::binary);
        seedBytes.assign(std::istreambuf_iterator<char>(in),{});
    }
    {
        std::ifstream in(canonicalPath,std::ios::binary);
        canonicalBytes.assign(std::istreambuf_iterator<char>(in),{});
    }
    std::istringstream seedStream(seedBytes),canonicalStream(canonicalBytes);
    std::string seedLine,canonicalLine;
    assert(std::getline(seedStream,seedLine)&&std::getline(canonicalStream,canonicalLine));
    while(std::getline(seedStream,seedLine)) {
        assert(std::getline(canonicalStream,canonicalLine));
        Row seedRow,canonicalRow;
        assert(physical_skin_authoring_detail_bridge_csv_detail::parseLine(seedLine,seedRow));
        assert(physical_skin_authoring_detail_bridge_csv_detail::parseLine(canonicalLine,canonicalRow));
        assert(seedRow.size()==54u&&canonicalRow.size()==75u);
        assert(seedRow[5]=="1"&&seedRow[7]=="[1,0]");
        assert(canonicalRow[65]=="0"&&canonicalRow[67]=="[0]");
        assert(std::equal(seedRow.begin(),seedRow.begin()+45,canonicalRow.begin()));
    }
    assert(!std::getline(canonicalStream,canonicalLine));

    auto strictFiles=f.files;
    strictFiles.allowPrecanonicalExportSeed=false;
    strictFiles.fineToBoundaryMap.path=canonicalPath;
    strictFiles.fineToBoundaryMap.expectedSHA256=exportReport.outputSHA256;
    PhysicalSkinAuthoringDetailBridgeInput strictInput;
    PhysicalSkinAuthoringDetailBridgeCSVReport strictReport;
    assert(loadPhysicalSkinAuthoringDetailBridgeCSV(
        strictFiles,f.bundle,f.boundary,strictInput,strictReport,error));
    assert(strictReport.nativeCanonicalSelectionRows==3u);
    assert(strictReport.nativeCanonicalFaceChangeRows==0u);
    PhysicalSkinAuthoringDetailBridgeState strictState;
    PhysicalSkinAuthoringDetailBridgeReport strictBridgeReport;
    assert(preparePhysicalSkinAuthoringDetailBridge(
        strictInput,strictState,strictBridgeReport,error));
    assert(!strictState.physicalAdmission&&strictState.tetraRows.size()==3u);

    auto tamperedFiles=strictFiles;
    tamperedFiles.fineToBoundaryMap.path=f.root/"tampered-canonical-map.csv";
    std::filesystem::copy_file(canonicalPath,tamperedFiles.fineToBoundaryMap.path);
    mutateFirstCSVRow(tamperedFiles.fineToBoundaryMap.path,54u,"999999");
    assert(physical_skin_child_mass_detail::sha256File(
        tamperedFiles.fineToBoundaryMap.path,tamperedFiles.fineToBoundaryMap.expectedSHA256,error));
    PhysicalSkinAuthoringDetailBridgeInput rejected=strictInput;
    const auto preservedPoints=rejected.detailReferenceWorldPoints;
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        tamperedFiles,f.bundle,f.boundary,rejected,strictReport,error));
    assert(rejected.detailReferenceWorldPoints==preservedPoints);

    tamperedFiles=strictFiles;
    tamperedFiles.fineToBoundaryMap.path=f.root/"tampered-canonical-start.csv";
    std::filesystem::copy_file(canonicalPath,tamperedFiles.fineToBoundaryMap.path);
    mutateFirstCSVRow(tamperedFiles.fineToBoundaryMap.path,65u,"1");
    assert(physical_skin_child_mass_detail::sha256File(
        tamperedFiles.fineToBoundaryMap.path,tamperedFiles.fineToBoundaryMap.expectedSHA256,error));
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        tamperedFiles,f.bundle,f.boundary,rejected,strictReport,error));
    assert(rejected.detailReferenceWorldPoints==preservedPoints);

    tamperedFiles=strictFiles;
    tamperedFiles.fineToBoundaryMap.expectedSHA256=std::string(64u,'0');
    assert(!loadPhysicalSkinAuthoringDetailBridgeCSV(
        tamperedFiles,f.bundle,f.boundary,rejected,strictReport,error));
}
}

int main() {
    testPinnedCSVBridge();
    testRefinedReorderedChildBoundaryUsesAuthenticatedParentRows();
    testCanonicalNativeFaceExport();
    testCanonicalEligibilityReinfersStableFace();
    testCSVBridgeAllowsOnlyAuthenticatedStableFaceWalk();
    testTransferHonorsPerDetailFaceEligibility();
    std::cout<<"authoring_detail_bridge_csv=pass typed_O_nodes=3 full86=86 map_rows=3 face_reordered=validated frame_owner=separate same_donor_parent_walk=pass residual_recomputed=pass physical_admission=false\n";
}
