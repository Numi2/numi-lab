#include "NumiHumanSourceCarrierQuadratureImport.hpp"
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
namespace detail = source_carrier_quadrature_import_detail;

template<class T> static void writeOne(std::ofstream& f,const T& value) {
    f.write(reinterpret_cast<const char*>(&value),sizeof(T));
}
template<class T> static void writeVector(std::ofstream& f,const std::vector<T>& values) {
    if(!values.empty())f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
}
template<class T,std::size_t N> static void writeVector(std::ofstream& f,const std::array<T,N>& values) {
    f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T));
}
static std::string sha(const std::filesystem::path& p) {
    std::string result,error;assert(physical_skin_child_mass_detail::sha256File(p,result,error));return result;
}
static void writeBinary(const std::filesystem::path& p,const std::vector<std::uint32_t>& sourceRows) {
    std::ofstream f(p,std::ios::binary|std::ios::trunc);assert(f);
    f.write("FSSCARR01",9);
    const std::uint64_t np=2,nc=1;
    writeOne(f,np);writeOne(f,nc);
    const std::vector<std::uint32_t> roleRows{0,1},offsets{0,0,1},children{1};
    writeVector(f,roleRows);
    const std::vector<double> weights{1.0},distances{0.0};
    const std::array<double,4> bary{0.25,0.25,0.25,0.25};
    writeVector(f,sourceRows);writeVector(f,offsets);writeVector(f,weights);writeVector(f,bary);
    writeVector(f,children);writeVector(f,distances);f.close();assert(f);
}
static void writeReceipt(const std::filesystem::path& p,const std::filesystem::path& bin,
                         const std::array<std::filesystem::path,5>& pins,
                         const std::string& supportSHA,const std::vector<std::uint32_t>& sourceRows) {
    const std::vector<std::uint32_t> roleRows{0,1},offsets{0,0,1},children{1};
    const std::vector<double> weights{1.0},bary{0.25,0.25,0.25,0.25},distances{0.0};
    std::string json="{\n"
        "  \"role\":\"Fat\",\n"
        "  \"schema\":\"numi.human.source-tetra-centroid-carrier-quadrature.v1\",\n"
        "  \"status\":\"PASS_SOURCE_ONLY_POSITIVE_FIRST_MOMENT_QUADRATURE\",\n"
        "  \"binary_sha256\":\""+sha(bin)+"\",\n"
        "  \"binary_layout\":\"FSSCARR01|u64 n_parent|u64 n_carrier|u32 source_parent_tet_rows[n_parent]|u32 global_source330_tet_rows[n_parent]|u32 parent_offsets[n_parent+1]|f64 weights[n_carrier]|f64 barycentric[n_carrier,4]|u32 witness_child_rows[n_carrier]|f64 witness_distances_m[n_carrier]\",\n"
        "  \"carrier_rule_degree\":1,\n  \"native_steps\":0,\n  \"physical_admission\":false,\n"
        "  \"maximum_transport_distance_m\":0.019353,\n"
        "  \"counts\":{\"source_parents\":2,\"carrierized_parents\":1,\"carriers\":1},\n"
        "  \"inputs\":{\n";
    const std::array<std::string,6> hashes{{sha(pins[0]),sha(pins[1]),sha(pins[2]),sha(pins[3]),sha(pins[4]),supportSHA}};
    for(std::size_t i=0;i<pins.size();++i)
        json+="    \""+pins[i].string()+"\":\""+hashes[i]+"\""+",\n";
    json+="    \"/fixture/support.bin\":\""+supportSHA+"\"\n"
        "  },\n  \"counts_end\":0,\n  \"outputs\":{\""+bin.string()+"\":{\"sha256\":\""+sha(bin)+"\"}},\n  \"carrier_array_sha256\":{\n"
        "    \"source_parent_tet_rows_sha256\":\""+detail::rawSHA(roleRows)+"\",\n"
        "    \"parent_offsets_sha256\":\""+detail::rawSHA(offsets)+"\",\n"
        "    \"source_global_tet_rows_sha256\":\""+detail::rawSHA(sourceRows)+"\",\n"
        "    \"weights_sha256\":\""+detail::rawSHA(weights)+"\",\n"
        "    \"barycentric_sha256\":\""+detail::rawSHA(bary)+"\",\n"
        "    \"witness_child_rows_sha256\":\""+detail::rawSHA(children)+"\",\n"
        "    \"witness_distances_sha256\":\""+detail::rawSHA(distances)+"\"\n"
        "  }\n}\n";
    std::ofstream f(p,std::ios::binary|std::ios::trunc);f<<json;f.close();assert(f);
}

static void writeBinaryOne(const std::filesystem::path& p,std::uint32_t globalSourceRow) {
    std::ofstream f(p,std::ios::binary|std::ios::trunc);assert(f);
    f.write("FSSCARR01",9);
    const std::uint64_t np=1,nc=1;writeOne(f,np);writeOne(f,nc);
    const std::vector<std::uint32_t> roleRows{0},globalRows{globalSourceRow},offsets{0,1},children{0};
    const std::vector<double> weights{1.0},distances{0.0};
    const std::array<double,4> bary{0.25,0.25,0.25,0.25};
    writeVector(f,roleRows);writeVector(f,globalRows);writeVector(f,offsets);writeVector(f,weights);
    writeVector(f,bary);writeVector(f,children);writeVector(f,distances);f.close();assert(f);
}
static void writePairedReceipt(const std::filesystem::path& receiptPath,
        const std::filesystem::path& binaryPath,const std::string& role,
        const std::array<std::filesystem::path,5>& sourcePaths,const std::string& supportSHA) {
    std::string json="{\n"
        "  \"role\":\""+role+"\",\n"
        "  \"schema\":\"numi.human.source-tetra-centroid-carrier-quadrature.v1\",\n"
        "  \"status\":\"PASS_SOURCE_ONLY_BATCH_POSITIVE_FIRST_MOMENT_QUADRATURE\",\n"
        "  \"binary_sha256\":\""+sha(binaryPath)+"\",\n"
        "  \"carrier_rule_degree\":1,\n  \"native_steps\":0,\n  \"physical_admission\":false,\n"
        "  \"maximum_transport_distance_m\":0.019353,\n"
        "  \"counts\":{\"source_parents\":1,\"carrierized_parents\":1,\"carriers\":1},\n"
        "  \"inputs\":{\n";
    for(std::size_t i=0;i<sourcePaths.size();++i)
        json+="    \""+sourcePaths[i].string()+"\":\""+sha(sourcePaths[i])+"\",\n";
    json+="    \"/fixture/"+role+"-support.bin\":\""+supportSHA+"\"\n"
        "  },\n"
        "  \"outputs\":{\""+binaryPath.string()+"\":{\"sha256\":\""+sha(binaryPath)+"\"}}\n"
        "}\n";
    std::ofstream f(receiptPath,std::ios::binary|std::ios::trunc);f<<json;f.close();assert(f);
}

int main() {
    const auto root=std::filesystem::temp_directory_path()/"numi-source-carrier-import-723";
    std::filesystem::remove_all(root);std::filesystem::create_directories(root);
    const std::array<std::filesystem::path,5> pinPaths{{root/"mass.csv",root/"tets.csv",root/"nodes.csv",root/"child.node",root/"child.ele"}};
    for(std::size_t i=0;i<pinPaths.size();++i){std::ofstream f(pinPaths[i]);f<<"source-pin-"<<i<<"\n";}
    const std::string supportSHA(64,static_cast<char>(0x66));
    const auto binary=root/"carriers.bin",receipt=root/"receipt.json";
    detail::Role asset;asset.materialIndex=16u;asset.binaryPath=binary;asset.receiptPath=receipt;
    asset.parentMassCSV=pinPaths[0];asset.sourceTetTableCSV=pinPaths[1];asset.sourceNodeTableCSV=pinPaths[2];
    asset.roleChildNode=pinPaths[3];asset.roleChildElements=pinPaths[4];
    asset.sourceTetTableSHA256=sha(pinPaths[1]);asset.sourceNodeTableSHA256=sha(pinPaths[2]);
    asset.roleChildNodeSHA256=sha(pinPaths[3]);asset.roleChildElementsSHA256=sha(pinPaths[4]);
    std::string error;detail::RolePayload payload;
    const std::vector<std::uint32_t> expectedParentRows{5u,6u},roleChildRows{10u,11u,12u};
    writeBinary(binary,expectedParentRows);asset.binarySHA256=sha(binary);
    writeReceipt(receipt,binary,pinPaths,supportSHA,expectedParentRows);asset.receiptSHA256=sha(receipt);
    assert(detail::readRole(asset,16u,expectedParentRows,8u,roleChildRows,
        sha(pinPaths[0]),supportSHA,payload,error));
    assert(payload.globalParentRows==expectedParentRows&&payload.globalChildRows==std::vector<std::uint32_t>{11u});

    const std::vector<std::uint32_t> outOfRangeRows{5u,8u};
    writeBinary(binary,outOfRangeRows);asset.binarySHA256=sha(binary);
    writeReceipt(receipt,binary,pinPaths,supportSHA,outOfRangeRows);asset.receiptSHA256=sha(receipt);
    detail::RolePayload rejected;error.clear();
    assert(!detail::readRole(asset,16u,outOfRangeRows,8u,roleChildRows,
        sha(pinPaths[0]),supportSHA,rejected,error));
    assert(error.find("role/global source-tet maps")!=std::string::npos);
    const auto massCSV=root/"paired-mass.csv",sourceTet=root/"paired-source.tets.csv",
        sourceNode=root/"paired-source.nodes.csv",childNode=root/"paired-child.node",
        childEle=root/"paired-child.ele",childLabels=root/"paired-child.labels.csv";
    for(const auto& path:{massCSV,sourceTet,sourceNode,childNode,childEle,childLabels}){
        std::ofstream f(path);f<<"paired fixture input\n";
    }
    const std::array<std::filesystem::path,5> pairedPins{{massCSV,sourceTet,sourceNode,childNode,childEle}};
    const std::string skinSupport(64,'1'),fatSupport(64,'2');
    const auto skinBin=root/"skin-carriers.bin",skinReceipt=root/"skin-carriers.json",
        fatBin=root/"fat-carriers.bin",fatReceipt=root/"fat-carriers.json";
    writeBinaryOne(skinBin,0u);writePairedReceipt(skinReceipt,skinBin,"Skin",pairedPins,skinSupport);
    writeBinaryOne(fatBin,1u);writePairedReceipt(fatReceipt,fatBin,"Fat",pairedPins,fatSupport);
    PhysicalSkinSourceCarrierQuadratureAssets pairedAssets;
    pairedAssets.roles[0].materialIndex=15u;pairedAssets.roles[1].materialIndex=16u;
    auto setRole=[&](PhysicalSkinSourceCarrierRoleAsset& a,const std::filesystem::path& bin,
            const std::filesystem::path& rec){
        a.binaryPath=bin;a.receiptPath=rec;a.binarySHA256=sha(bin);a.receiptSHA256=sha(rec);
        a.parentMassCSV=massCSV;a.sourceTetTableCSV=sourceTet;a.sourceNodeTableCSV=sourceNode;
        a.roleChildNode=childNode;a.roleChildElements=childEle;
        a.sourceTetTableSHA256=sha(sourceTet);a.sourceNodeTableSHA256=sha(sourceNode);
        a.roleChildNodeSHA256=sha(childNode);a.roleChildElementsSHA256=sha(childEle);
    };
    setRole(pairedAssets.roles[0],skinBin,skinReceipt);
    setRole(pairedAssets.roles[1],fatBin,fatReceipt);
    PhysicalSkinFractionalSemanticSupportAssets supportAssets;
    supportAssets.roles[0].materialIndex=15u;supportAssets.roles[0].binarySHA256=skinSupport;
    supportAssets.roles[1].materialIndex=16u;supportAssets.roles[1].binarySHA256=fatSupport;
    PhysicalSkinFractionalSemanticSupportImportReport supportReport;
    supportReport.verified=true;supportReport.childRows={1u,1u};
    PhysicalSkinChildMassMappingBundle bundle;
    bundle.parent.parentMassPacketSHA256=sha(massCSV);
    bundle.parent.materialIndices={15u,16u};
    bundle.parent.tetrahedra={{{0u,1u,2u,3u}},{{1u,2u,3u,4u}}};
    bundle.parent.referenceWorldPoints={{{0,0,0}},{{1,0,0}},{{0,1,0}},{{0,0,1}},{{1,1,1}}};
    bundle.parent.currentWorldPoints=bundle.parent.referenceWorldPoints;
    bundle.parent.donorBodyIndices={3u,4u};bundle.parent.anatomicalRegionIndices={7u,8u};
    bundle.parent.stableSourceFaceIdentities={99u,100u};
    bundle.childMaterialIndices={15u,16u};bundle.childDonorBodyIndices={3u,4u};
    bundle.childAnatomicalRegionIndices={7u,8u};bundle.childStableSourceFaceIdentities={99u,100u};
    bundle.childTetrahedra=bundle.parent.tetrahedra;
    bundle.referenceWorldPoints=bundle.parent.currentWorldPoints;
    auto& sem=bundle.fractionalSemanticSupport;
    sem.groupMaterialIndices={15u,16u};sem.donorBodyIndices={3u,4u};sem.anatomicalRegionIndices={7u,8u};
    sem.sourceParentTetOffsets={0u,1u,2u};sem.sourceParentTetRows={0u,1u};
    sem.cellAdmissibleRegionOffsets={0u,1u,2u};sem.cellAdmissibleRegionIds={7u,8u};
    sem.admissibilityMaximumSourceRegionHops=1u;
    std::array<std::string,3> childHashes{sha(childNode),sha(childEle),sha(childLabels)};
    PhysicalSkinSourceCarrierQuadratureData pairedOutput;
    PhysicalSkinSourceCarrierQuadratureImportReport pairedReport;error.clear();
    assert(loadPhysicalSkinSourceCarrierQuadrature090(pairedAssets,supportAssets,supportReport,bundle,
        childHashes,pairedOutput,pairedReport,error));
    assert((pairedOutput.globalSourceTetRows==std::vector<std::uint32_t>{0u,1u}));
    assert((pairedOutput.parentOffsets==std::vector<std::uint32_t>{0u,1u,2u}));
    assert((pairedOutput.witnessChildRows==std::vector<std::uint32_t>{0u,1u}));
    assert(pairedOutput.witnessChildRows[0]==0u&&pairedOutput.witnessChildRows[1]==1u);
    const std::array<std::uint32_t,4> skinTet{0u,1u,2u,3u},fatTet{1u,2u,3u,4u};
    assert(bundle.parent.tetrahedra[pairedOutput.globalSourceTetRows[0]]==skinTet);
    assert(bundle.parent.tetrahedra[pairedOutput.globalSourceTetRows[1]]==fatTet);
    assert((pairedReport.carrierizedParentRows==std::array<std::size_t,2>{1u,1u}));
    std::cout<<"PASS paired canonical FSSCARR01 maps: Fat-local witness 0 offsets after Skin, with exact source-tet and node-geometry maps\n";
}
