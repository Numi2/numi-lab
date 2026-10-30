#include "NumiHumanP1SupportedReferenceAnchorImport.hpp"
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
namespace detail = p1_supported_reference_anchor_import_detail;

static std::string sha(const std::filesystem::path& path) {
    std::string result,error;assert(physical_skin_child_mass_detail::sha256File(path,result,error));return result;
}
static std::string jsonEscape(const std::string& value) {
    std::string out="\"";for(char c:value){if(c=='\\'||c=='\"')out.push_back('\\');out.push_back(c);}out.push_back('\"');return out;
}
static std::string csvEscape(const std::string& value) {
    if(value.find_first_of(",\"\r\n")==std::string::npos)return value;
    std::string out="\"";for(char c:value){if(c=='\"')out.push_back('\"');out.push_back(c);}out.push_back('\"');return out;
}
static void writeText(const std::filesystem::path& path,const std::string& value) {
    std::ofstream f(path,std::ios::binary|std::ios::trunc);assert(f);f<<value;f.close();assert(f);
}
struct Fixture {
    std::filesystem::path root,childNode,childElements,childLabels,parentMass,parentNodes,parentSource,parentTets,
        weights,skinSupport,fatSupport,projection,csr,parentMap,receipt;
    PhysicalSkinChildMassMappingBundle bundle;
    PhysicalSkinFractionalSemanticSupportImportReport supportReport;
    PhysicalSkinP1SupportedReferenceAnchorAssets assets;
    std::array<std::string,3> childHashes;
    Fixture() {
        root=std::filesystem::temp_directory_path()/"numi-p1-anchor-import-723";
        std::filesystem::remove_all(root);std::filesystem::create_directories(root);
        childNode=root/"child.node";childElements=root/"child.ele";childLabels=root/"child.labels.csv";
        parentMass=root/"parent.csv";parentNodes=root/"parent.nodes.csv";parentSource=root/"parent.source.csv";
        parentTets=root/"parent.tets.csv";weights=root/"parent.weights.csv";skinSupport=root/"skin.support";
        fatSupport=root/"fat.support";projection=root/"projection.csv";csr=root/"anchor.npz";
        parentMap=root/"parent-map.csv";receipt=root/"receipt.json";
        for(const auto& path:{childNode,childElements,parentMass,parentNodes,parentSource,parentTets,weights,skinSupport,fatSupport})
            writeText(path,"fixture-input\n");
        writeText(childLabels,"child labels exact bytes\n");writeText(csr,"authenticated csr bytes\n");
        writeText(parentMap,"child_tet_row,parent_tet_row\n0,0\n");
        assets.anchorCSR_SHA256=sha(csr);assets.parentRowMapSHA256=sha(parentMap);
        bundle.parent.currentWorldPoints={{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}}};
        bundle.parent.referenceWorldPoints=bundle.parent.currentWorldPoints;
        bundle.parent.tetrahedra={{{0u,1u,2u,3u}}};
        bundle.parent.materialIndices={15u};bundle.parent.donorBodyIndices={3u};
        bundle.parent.anatomicalRegionIndices={7u};bundle.parent.stableSourceFaceIdentities={99u};
        bundle.parent.parentMassPacketSHA256=sha(parentMass);
        bundle.parent.parentSourceContributionSHA256=sha(parentSource);
        std::array<std::string,4> topology{sha(parentMass),sha(parentNodes),sha(parentSource),sha(parentTets)};
        assert(physical_skin_child_mass_detail::hashJoinedFiles(topology,bundle.parent.parentTopologySHA256));
        bundle.childTetrahedra={{{0u,1u,2u,3u}}};bundle.referenceWorldPoints=bundle.parent.currentWorldPoints;
        bundle.childMaterialIndices={15u};bundle.childDonorBodyIndices={3u};
        bundle.childAnatomicalRegionIndices={7u};bundle.childStableSourceFaceIdentities={99u};
        bundle.childMeshSHA256=std::string(64,'a');bundle.regionLabelSHA256=std::string(64,'b');
        auto& semantic=bundle.fractionalSemanticSupport;
        semantic.groupMaterialIndices={15u};semantic.donorBodyIndices={3u};semantic.anatomicalRegionIndices={7u};
        semantic.sourceParentTetOffsets={0u,1u};semantic.sourceParentTetRows={0u};
        semantic.cellAdmissibleRegionOffsets={0u,1u};semantic.cellAdmissibleRegionIds={7u};
        semantic.admissibilityMaximumSourceRegionHops=1u;
        supportReport.verified=true;supportReport.binarySHA256={sha(skinSupport),sha(fatSupport)};
        childHashes={sha(childNode),sha(childElements),sha(childLabels)};
        assets.projectionCSV=projection;assets.receiptJSON=receipt;assets.anchorCSRNPZ=csr;assets.parentRowMapCSV=parentMap;
        assets.pinnedInputs={
            {"child_node",childNode,childHashes[0]},{"child_elements",childElements,childHashes[1]},
            {"parent_mass",parentMass,sha(parentMass)},{"parent_nodes",parentNodes,sha(parentNodes)},
            {"parent_source",parentSource,sha(parentSource)},{"parent_tets",parentTets,sha(parentTets)},
            {"parent_full86_weights",weights,sha(weights)},{"skin_p1_support",skinSupport,sha(skinSupport)},
            {"fat_p1_support",fatSupport,sha(fatSupport)}};
        writeProjection(0.0);
        writeReceipt(true);
    }
    void writeProjection(double queryX) {
        const std::vector<std::string> header{
            "child_tet_row","carrier_child_node_row","carrier_query_x_m","carrier_query_y_m","carrier_query_z_m",
            "selected_parent_tet_row","selected_parent_barycentric0","selected_parent_barycentric1",
            "selected_parent_barycentric2","selected_parent_barycentric3","registered_world_x_m",
            "registered_world_y_m","registered_world_z_m","residual_x_m","residual_y_m","residual_z_m",
            "anchor_distance_m","material_index","geometric_donor_body","anatomical_region_id",
            "stable_source_face_id","primary_group_id","carrier_node_group_id","primary_score_numerator",
            "primary_score_denominator","exact_resolution_used","exact_min_squared_distance_ratio",
            "tie_parent_rows_json","tie_semantic_tuples_json"};
        const std::vector<std::string> row{
            "0","0",std::to_string(queryX),"0","0","0","1","0","0","0","0","0","0",
            "0","0","0","0","15","3","7","99","0","0","1","1","1","1/1","[0]","[[15,3,7,99]]"};
        std::string text;
        for(std::size_t i=0;i<header.size();++i){if(i)text.push_back(',');text+=csvEscape(header[i]);}
        text.push_back('\n');
        for(std::size_t i=0;i<row.size();++i){if(i)text.push_back(',');text+=csvEscape(row[i]);}
        text.push_back('\n');writeText(projection,text);assets.projectionSHA256=sha(projection);
    }
    void writeReceipt(bool full) {
        std::string j="{\n"
            "  \"schema\":\"numi.native-domain-217.p1-supported-reference-anchor.v1\",\n"
            "  \"status\":\"P1_SUPPORTED_REFERENCE_ANCHOR_LABELS_EXPORTED_NOT_PHYSICAL_ADMISSION\",\n"
            "  \"native_steps\":0,\n  \"physical_admission\":false,\n"
            "  \"mass_transport\":\"NOT_RUN\",\n"
            "  \"full_child_domain_export\":"+std::string(full?"true":"false")+",\n"
            "  \"full_child_domain_exported_rows\":1,\n"
            "  \"counts\":{\"child_tetrahedra\":1,\"exported_child_rows\":1},\n"
            "  \"inputs\":{";
        for(std::size_t i=0;i<assets.pinnedInputs.size();++i){
            const auto& pin=assets.pinnedInputs[i];if(i)j+=",";
            j+="\n    "+jsonEscape(pin.receiptKey)+":{\"path\":"+jsonEscape(pin.path.string())+
                ",\"sha256\":"+jsonEscape(pin.sha256)+"}";
        }
        j+="\n  },\n  \"outputs\":{"
            "\"p1_anchor_projection_csv\":{\"path\":"+jsonEscape(projection.string())+
                ",\"sha256\":"+jsonEscape(assets.projectionSHA256)+",\"row_count\":1,\"full_domain\":true},"
            "\"p1_anchor_csr\":{\"path\":"+jsonEscape(csr.string())+
                ",\"sha256\":"+jsonEscape(assets.anchorCSR_SHA256)+"},"
            "\"labels\":{\"path\":"+jsonEscape(childLabels.string())+
                ",\"sha256\":"+jsonEscape(childHashes[2])+"},"
            "\"parent_row_map\":{\"path\":"+jsonEscape(parentMap.string())+
                ",\"sha256\":"+jsonEscape(assets.parentRowMapSHA256)+"}"
            "}\n}\n";
        writeText(receipt,j);assets.receiptSHA256=sha(receipt);
    }
};
int main() {
    Fixture f;PhysicalSkinP1SupportedReferenceAnchorImportReport report;std::string error;
    assert(loadPhysicalSkinP1SupportedReferenceAnchor(f.assets,f.bundle,f.supportReport,
        f.childHashes,f.childLabels,f.childHashes[2],report,error));
    assert(report.verified&&report.fullChildDomain&&!report.physicalAdmission&&report.nativeSteps==0u);
    assert(report.childRows==1u&&report.verifiedProjectionRows==1u);
    std::cout<<"PASS full-domain P1 anchor import replays the nodal projection, source parent, tuple and support pins\n";

    f.writeReceipt(false);error.clear();PhysicalSkinP1SupportedReferenceAnchorImportReport rejected;
    assert(!loadPhysicalSkinP1SupportedReferenceAnchor(f.assets,f.bundle,f.supportReport,
        f.childHashes,f.childLabels,f.childHashes[2],rejected,error));
    assert(error.find("exact complete child domain")!=std::string::npos);
    std::cout<<"PASS P1 anchor importer rejects a diagnostic/non-full export\n";

    f.writeProjection(0.25);f.writeReceipt(true);error.clear();
    assert(!loadPhysicalSkinP1SupportedReferenceAnchor(f.assets,f.bundle,f.supportReport,
        f.childHashes,f.childLabels,f.childHashes[2],rejected,error));
    assert(error.find("query point differs")!=std::string::npos);
    std::cout<<"PASS P1 anchor importer rejects a rehashed projection that no longer matches the exact child node\n";
}
