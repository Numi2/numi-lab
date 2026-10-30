#include "NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp"
#include <array>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
namespace {
using U32=std::uint32_t; using U64=std::uint64_t;
const std::string sha(char c){return std::string(64,c);}
template<class T> void put(std::ofstream& f,const T& value){f.write(reinterpret_cast<const char*>(&value),sizeof(T));}
template<class T> void putv(std::ofstream& f,const std::vector<T>& values){if(!values.empty())f.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(T)));}
void json(const std::filesystem::path& p,const std::string& value){std::ofstream f(p,std::ios::binary);assert(f);f<<value;f.close();assert(f);}
std::string hash(const std::filesystem::path& p){std::string h,e;assert(physical_skin_child_mass_detail::sha256File(p,h,e));return h;}
struct Files { std::filesystem::path bin,receipt,producer,verifier; std::string binSha,receiptSha,producerSha,verifierSha; };
Files makeRole(const std::filesystem::path& dir,const std::string& role,U32 material,
               const std::vector<U32>& parentFaces,const std::string& producerSha,const std::string& verifierSha,
               const std::string& childMeshSha,const std::string& labelSha,
               const std::string& massSha,const std::string& sourceMassSha) {
    Files x; x.bin=dir/(role+".support.bin");x.receipt=dir/(role+".receipt.json");
    const std::string typed=sha(role=="Skin"?'a':'d');
    const std::string source=sha(role=="Skin"?'b':'e');
    std::ofstream f(x.bin,std::ios::binary);assert(f);
    f.write("FSSUP090",8);
    const std::array<U64,6> counts{{3,3,1,3,3,3}};for(auto count:counts)put(f,count);
    putv(f,std::vector<U32>{0,1,2,3});                 // child offsets
    putv(f,std::vector<U32>{0,0,0});                   // group slots
    putv(f,std::vector<double>{1,1,1});                // fractions
    putv(f,std::vector<U32>{material});                 // group material
    putv(f,std::vector<U32>{145});                      // donor body
    putv(f,std::vector<U32>{42});                       // anatomical region
    putv(f,std::vector<U64>{material==15?15001u:16001u});
    putv(f,parentFaces);                                // parent stable face IDs
    putv(f,std::vector<U32>{0,3});                      // group parent offsets
    putv(f,std::vector<U32>{0,1,2});                    // local role parent rows
    putv(f,std::vector<U32>{0,3});                      // group face offsets
    putv(f,parentFaces);                                // full stable face union
    putv(f,std::vector<U32>{material,material,material});
    putv(f,std::vector<U32>{145,145,145});
    putv(f,std::vector<U32>{42,42,42});
    f.write(typed.data(),64);
    f.write("FSSADJ01",8);put(f,U64(3));put(f,U32(1));f.write(source.data(),64);
    putv(f,std::vector<U32>{0,1,2,3});
    putv(f,std::vector<U32>{42,42,42});
    f.close();assert(f);
    x.binSha=hash(x.bin);
    x.producerSha=producerSha;x.verifierSha=verifierSha;
    std::string r="{\"schema\":\"numi.human.fractional-semantic-support-role.v1\",\"status\":\"PASS_SOURCE_ONLY_FRACTIONAL_SEMANTIC_ROLE_SUPPORT\","
       "\"binary_sha256\":\""+x.binSha+"\",\"child_mesh_sha256\":\""+childMeshSha+"\",\"region_label_sha256\":\""+labelSha+"\","
       "\"parent_mass_packet_sha256\":\""+massSha+"\",\"parent_source_contribution_sha256\":\""+sourceMassSha+"\","
       "\"role_material_index\":"+std::to_string(material)+",\"admissibility_max_source_region_hops\":1,\"native_steps\":0,\"physical_admission\":false,"
       "\"support_sha256\":\""+typed+"\",\"admissibility_source_support_sha256\":\""+source+"\","
       "\"admissibility_producer_receipt_sha256\":\""+producerSha+"\",\"admissibility_verification_receipt_sha256\":\""+verifierSha+"\"}";
    json(x.receipt,r);x.receiptSha=hash(x.receipt);return x;
}
}
int main(){
    const auto root=std::filesystem::temp_directory_path()/"numi-fractional-support-import-control";
    std::filesystem::remove_all(root);std::filesystem::create_directories(root);
    const auto producer=root/"producer.json",verifier=root/"verifier.json";
    const auto producerText=std::string("{\"schema\":\"numi.one-hop.test.v1\",\"status\":\"SOURCE_ONLY_AUTHENTICATED_ONE_HOP_SAME_TISSUE_ADMISSIBILITY\",\"physical_admission\":false,\"native_steps\":0,\"roles\":[")+
       "{\"role\":\"Skin\",\"source_support_sha256\":\""+sha('b')+"\",\"material\":15},{\"role\":\"Fat\",\"source_support_sha256\":\""+sha('e')+"\",\"material\":16}]}";
    json(producer,producerText);const auto producerHash=hash(producer);
    const auto verifierText=std::string("{\"schema\":\"numi.one-hop.test-verification.v1\",\"status\":\"PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT\",\"physical_admission\":false,\"native_steps\":0,\"producer_receipt_sha256\":\"")+producerHash+"\",\"roles\":["
       "{\"role\":\"Skin\",\"cells\":3,\"verified_source_face_adjacency_edges\":1},{\"role\":\"Fat\",\"cells\":3,\"verified_source_face_adjacency_edges\":1}]}";
    json(verifier,verifierText);const auto verifierHash=hash(verifier);
    const std::string childMesh=sha('1'),labels=sha('2'),mass=sha('3'),sourceMass=sha('4');
    auto skin=makeRole(root,"Skin",15,{601,602,603},producerHash,verifierHash,childMesh,labels,mass,sourceMass);
    auto fat=makeRole(root,"Fat",16,{604,605,606},producerHash,verifierHash,childMesh,labels,mass,sourceMass);
    PhysicalSkinChildMassMappingBundle bundle;
    bundle.childMeshSHA256=childMesh;bundle.regionLabelSHA256=labels;
    auto& parent=bundle.parent;parent.parentMassPacketSHA256=mass;parent.parentSourceContributionSHA256=sourceMass;
    parent.materialIndices={15,15,15,16,16,16};parent.donorBodyIndices={145,145,145,145,145,145};
    parent.anatomicalRegionIndices={42,42,42,42,42,42};parent.stableSourceFaceIdentities={601,602,603,604,605,606};
    parent.tetrahedra.resize(6);bundle.childTetrahedra.resize(6);
    bundle.childMaterialIndices={15,15,15,16,16,16};bundle.childDonorBodyIndices={145,145,145,145,145,145};
    bundle.childAnatomicalRegionIndices={42,42,42,42,42,42};bundle.childStableSourceFaceIdentities={601,602,603,604,605,606};
    bundle.childDominantParentTetrahedra={0,1,2,3,4,5};
    PhysicalSkinFractionalSemanticSupportAssets assets;
    auto assign=[&](unsigned i,U32 mat,const Files& x){auto& a=assets.roles[i];a.materialIndex=mat;a.binaryPath=x.bin;a.binarySHA256=x.binSha;a.receiptPath=x.receipt;a.receiptSHA256=x.receiptSha;
      a.admissibilityProducerReceiptPath=producer;a.admissibilityProducerReceiptSHA256=producerHash;
      a.admissibilityVerificationReceiptPath=verifier;a.admissibilityVerificationReceiptSHA256=verifierHash;};
    assign(0,15,skin);assign(1,16,fat);
    ConservativeSpatialChildSemanticSupportData output;PhysicalSkinFractionalSemanticSupportImportReport report;std::string error;
    assert(loadPhysicalSkinFractionalSemanticSupport090(assets,bundle,output,report,error));
    assert(report.verified&&report.childRows[0]==3&&report.childRows[1]==3&&report.parentRows[0]==3&&report.parentRows[1]==3);
    assert(output.admissibilityMaximumSourceRegionHops==1&&output.childOffsets.size()==7&&output.slotGroupIndices.size()==6);
    auto incomplete=assets;incomplete.roles[1]=PhysicalSkinFractionalSemanticRoleSupportAsset{};incomplete.roles[1].materialIndex=16u;
    ConservativeSpatialChildSemanticSupportData missingPair;PhysicalSkinFractionalSemanticSupportImportReport missingPairReport;error.clear();
    assert(!loadPhysicalSkinFractionalSemanticSupport090(incomplete,bundle,missingPair,missingPairReport,error));
    assert(error.find("paths, domain, or SHA pins are incomplete")!=std::string::npos);
    auto bad=assets;const auto tampered=root/"tampered.bin";std::filesystem::copy_file(skin.bin,tampered);
    {std::fstream f(tampered,std::ios::in|std::ios::out|std::ios::binary);assert(f);f.seekp(0);f.put('X');}
    bad.roles[0].binaryPath=tampered;
    ConservativeSpatialChildSemanticSupportData rejected;PhysicalSkinFractionalSemanticSupportImportReport rejectedReport;error.clear();
    assert(!loadPhysicalSkinFractionalSemanticSupport090(bad,bundle,rejected,rejectedReport,error));
    assert(error.find("binary SHA mismatch")!=std::string::npos);
    std::filesystem::remove_all(root);
    std::cout<<"fractional_semantic_support_import=PASS paired_roles=2 one_hop=PASS missing_fat=REJECTED binary_pin_tamper=REJECTED physical_admission=NOT_CLAIMED runtime=0\\n";
}
