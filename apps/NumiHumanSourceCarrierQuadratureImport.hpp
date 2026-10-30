#pragma once

#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include "NumiHumanFractionalSemanticSupportImport.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinSourceCarrierRoleAsset {
    std::uint32_t materialIndex=NM_INVALID_INDEX;
    std::filesystem::path binaryPath,receiptPath,parentMassCSV,sourceTetTableCSV,sourceNodeTableCSV,
        roleChildNode,roleChildElements;
    std::string binarySHA256,receiptSHA256,sourceTetTableSHA256,sourceNodeTableSHA256,
        roleChildNodeSHA256,roleChildElementsSHA256;
};
struct PhysicalSkinSourceCarrierQuadratureAssets {
    std::array<PhysicalSkinSourceCarrierRoleAsset,2> roles{};
};
struct PhysicalSkinSourceCarrierQuadratureImportReport {
    std::array<std::string,2> binarySHA256{},receiptSHA256{},sourceTetTableSHA256{},sourceNodeTableSHA256{};
    std::array<std::string,3> currentChildFileSHA256{};
    std::array<std::size_t,2> parentRows{},carrierRows{},carrierizedParentRows{};
    std::string packetSHA256,verificationSHA256;
    bool verified=false,physicalAdmission=false;
    std::uint32_t nativeSteps=0;
};

namespace source_carrier_quadrature_import_detail {
using namespace fractional_semantic_support_import_detail;
using Role=PhysicalSkinSourceCarrierRoleAsset;
struct RolePayload {
    std::vector<std::uint32_t> parentOffsets,roleParentRows,sourceRows,witnessChildRows;
    std::vector<double> weights,witnessDistances;
    std::vector<std::array<double,4>> barycentric;
    std::vector<std::uint32_t> globalParentRows,globalChildRows;
    std::string binarySHA,receiptSHA,tetSHA,nodeSHA;
    std::size_t carrierizedParents=0;
};
struct CarrierRow { double weight=0.0; std::array<double,4> bary{}; std::uint32_t child=0; double distance=0.0; };

inline bool object(std::string_view json,std::string_view key,std::string& out) {
    const std::string token="\""+std::string(key)+"\"";
    auto p=json.find(token); if(p==json.npos||json.find(token,p+token.size())!=json.npos)return false;
    p=json.find(':',p+token.size());if(p==json.npos)return false;++p;
    while(p<json.size()&&std::isspace(static_cast<unsigned char>(json[p])))++p;
    if(p==json.size()||json[p]!='{')return false;
    const auto begin=p;unsigned depth=0;bool quoted=false,escape=false;
    for(;p<json.size();++p){const char c=json[p];if(quoted){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='\"')quoted=false;continue;}
        if(c=='\"'){quoted=true;continue;}if(c=='{')++depth;else if(c=='}'&&--depth==0){out.assign(json.substr(begin,p-begin+1));return true;}}
    return false;
}
inline bool objectString(std::string_view json,std::string_view key,std::string& value) {
    const std::string token="\""+std::string(key)+"\"";auto p=json.find(token);
    if(p==json.npos||json.find(token,p+token.size())!=json.npos)return false;
    p=json.find(':',p+token.size());if(p==json.npos)return false;++p;
    while(p<json.size()&&std::isspace(static_cast<unsigned char>(json[p])))++p;
    if(p==json.size()||json[p++]!='\"')return false;const auto e=json.find('\"',p);
    if(e==json.npos)return false;value.assign(json.substr(p,e-p));return true;
}
inline bool real(std::string_view json,std::string_view key,double& value) {
    const std::string token="\""+std::string(key)+"\"";auto p=json.find(token);
    if(p==json.npos||json.find(token,p+token.size())!=json.npos)return false;
    p=json.find(':',p+token.size());if(p==json.npos)return false;++p;
    while(p<json.size()&&std::isspace(static_cast<unsigned char>(json[p])))++p;
    const auto begin=p;while(p<json.size()&&(json[p]=='-'||json[p]=='+'||json[p]=='.'||json[p]=='e'||json[p]=='E'||(json[p]>='0'&&json[p]<='9')))++p;
    if(p==begin)return false;try{value=std::stod(std::string(json.substr(begin,p-begin)));}catch(...){return false;}
    return std::isfinite(value)&&(p==json.size()||json[p]==','||json[p]=='}'||std::isspace(static_cast<unsigned char>(json[p])));
}
template<class T> inline std::string rawSHA(const std::vector<T>& values) {
    spatial_mass_transport_detail::SHA256Builder h;
    if(!values.empty())h.bytes(values.data(),values.size()*sizeof(T));
    return h.finish();
}
inline bool inputPinHasSHA(std::string_view pins,std::string_view expected) {
    if(expected.empty())return false;
    const std::string quoted="\""+std::string(expected)+"\"";
    if(pins.find(quoted)!=pins.npos)return true;
    std::size_t p=0;
    while((p=pins.find("\"sha256\"",p))!=pins.npos){
        p=pins.find(':',p+8u);if(p==pins.npos)return false;++p;
        while(p<pins.size()&&std::isspace(static_cast<unsigned char>(pins[p])))++p;
        if(p>=pins.size()||pins[p++]!='\"')return false;const auto e=pins.find('\"',p);if(e==pins.npos)return false;
        if(pins.substr(p,e-p)==expected)return true;p=e+1u;
    }
    return false;
}
inline bool carrierWitnessHasSemanticSupport(const ConservativeSpatialChildSemanticSupportData& support,
        const PhysicalSkinChildMassMappingBundle& bundle,std::uint32_t parent,std::uint32_t child) {
    if(parent>=bundle.parent.materialIndices.size()||parent>=bundle.parent.donorBodyIndices.size()||
       parent>=bundle.parent.anatomicalRegionIndices.size()||child>=bundle.childMaterialIndices.size()||
       child>=bundle.childDonorBodyIndices.size()||child>=bundle.childAnatomicalRegionIndices.size()||
       support.admissibilityMaximumSourceRegionHops!=1u||
       support.cellAdmissibleRegionOffsets.size()!=bundle.childMaterialIndices.size()+1u)return false;
    const auto parentMaterial=bundle.parent.materialIndices[parent];
    const auto childMaterial=bundle.childMaterialIndices[child];
    const auto childDonor=bundle.childDonorBodyIndices[child];
    const auto childRegion=bundle.childAnatomicalRegionIndices[child];
    const auto parentRegion=bundle.parent.anatomicalRegionIndices[parent];
    if(parentMaterial!=childMaterial||child==std::numeric_limits<std::uint32_t>::max()||
       support.cellAdmissibleRegionOffsets[child]>support.cellAdmissibleRegionOffsets[child+1u]||
       support.cellAdmissibleRegionOffsets[child+1u]>support.cellAdmissibleRegionIds.size())return false;
    const auto rb=support.cellAdmissibleRegionOffsets[child],re=support.cellAdmissibleRegionOffsets[child+1u];
    if(!std::binary_search(support.cellAdmissibleRegionIds.begin()+rb,
                           support.cellAdmissibleRegionIds.begin()+re,parentRegion))return false;
    const auto groups=support.groupMaterialIndices.size();
    if(support.donorBodyIndices.size()!=groups||support.anatomicalRegionIndices.size()!=groups||
       support.sourceParentTetOffsets.size()!=groups+1u)return false;
    for(std::uint32_t g=0;g<groups;++g){
        if(support.groupMaterialIndices[g]!=parentMaterial||
           support.donorBodyIndices[g]!=childDonor||support.anatomicalRegionIndices[g]!=childRegion||
           support.sourceParentTetOffsets[g]>support.sourceParentTetOffsets[g+1u]||
           support.sourceParentTetOffsets[g+1u]>support.sourceParentTetRows.size())continue;
        const auto begin=support.sourceParentTetRows.begin()+support.sourceParentTetOffsets[g];
        const auto end=support.sourceParentTetRows.begin()+support.sourceParentTetOffsets[g+1u];
        if(std::find(begin,end,parent)!=end)return true;
    }
    return false;
}
inline bool finiteDistanceEqual(double a,double b) {
    return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=1.0e-9;
}
inline double carrierChildCentroidDistance(const PhysicalSkinChildMassMappingBundle& bundle,
        std::uint32_t parent,const std::array<double,4>& bary,std::uint32_t child) {
    if(parent>=bundle.parent.tetrahedra.size()||child>=bundle.childTetrahedra.size()||
       bundle.parent.currentWorldPoints.empty()||bundle.referenceWorldPoints.empty())return std::numeric_limits<double>::infinity();
    std::array<double,3> source{},target{};
    const auto& parentTet=bundle.parent.tetrahedra[parent];
    const auto& childTet=bundle.childTetrahedra[child];
    for(unsigned corner=0;corner<4u;++corner){
        if(parentTet[corner]>=bundle.parent.currentWorldPoints.size()||
           childTet[corner]>=bundle.referenceWorldPoints.size())return std::numeric_limits<double>::infinity();
        for(unsigned axis=0;axis<3u;++axis){
            source[axis]+=bary[corner]*bundle.parent.currentWorldPoints[parentTet[corner]][axis];
            target[axis]+=0.25*bundle.referenceWorldPoints[childTet[corner]][axis];
        }
    }
    double squared=0.0;for(unsigned axis=0;axis<3u;++axis){const double d=source[axis]-target[axis];squared+=d*d;}
    return std::sqrt(squared);
}
inline bool outputPinHasFileSHA(std::string_view outputs,const std::filesystem::path& path,
                                std::string_view expected) {
    const std::string quotedPath="\""+path.string()+"\"";
    const std::string quotedSHA="\""+std::string(expected)+"\"";
    return outputs.find(quotedPath)!=outputs.npos&&outputs.find(quotedSHA)!=outputs.npos;
}
inline bool verifyFilePin(const std::filesystem::path& path,const std::string& expected,std::string& error) {
    if(path.empty()||!validSHA(expected)){error="source-carrier input file path or SHA pin is absent";return false;}
    std::string actual;if(!physical_skin_child_mass_detail::sha256File(path,actual,error))return false;
    if(actual!=expected){error="source-carrier input file SHA mismatch: "+path.string();return false;}return true;
}
inline bool readRole(const Role& a,const std::uint32_t expectedMaterial,
                     const std::vector<std::uint32_t>& expectedParentRows,
                     const std::size_t currentParentCount,
                     const std::vector<std::uint32_t>& roleChildRows,
                     const std::string& parentMassSHA,const std::string& roleSupportSHA,
                     RolePayload& out,std::string& error) {
    const auto fail=[&](std::string m){error=std::move(m);return false;};
    if(a.materialIndex!=expectedMaterial||(expectedMaterial!=15u&&expectedMaterial!=16u)||
       a.binaryPath.empty()||a.receiptPath.empty()||!validSHA(a.binarySHA256)||!validSHA(a.receiptSHA256)||
       !validSHA(a.sourceTetTableSHA256)||!validSHA(a.sourceNodeTableSHA256)||
       !validSHA(a.roleChildNodeSHA256)||!validSHA(a.roleChildElementsSHA256))
        return fail("source-carrier role paths, material domain, or SHA pins are incomplete");
    if(std::endian::native!=std::endian::little)return fail("FSSCARR01 requires a little-endian host");
    std::string hash;if(!physical_skin_child_mass_detail::sha256File(a.binaryPath,hash,error)||hash!=a.binarySHA256)
        return fail("source-carrier binary SHA mismatch");
    std::string receipt;if(!pinnedJSON(a.receiptPath,a.receiptSHA256,receipt,error))return false;
    std::string role,schema,status,binarySHA,pins,inputs,outputs;
    std::uint64_t steps=0;double bound=0.0;
    if(!str(receipt,"role",role)||role!=(expectedMaterial==15u?"Skin":"Fat")||
       !str(receipt,"schema",schema)||schema!="numi.human.source-tetra-centroid-carrier-quadrature.v1"||
       !str(receipt,"status",status)||
       (status!="SOURCE_ONLY_VALIDATED_DEGREE1_CARRIER_RULE"&&
        status!="PASS_SOURCE_ONLY_POSITIVE_FIRST_MOMENT_QUADRATURE"&&
        status!="PASS_SOURCE_ONLY_BATCH_POSITIVE_FIRST_MOMENT_QUADRATURE")||
       !str(receipt,"binary_sha256",binarySHA)||binarySHA!=a.binarySHA256||
       !num(receipt,"native_steps",steps)||steps!=0u||!noAdmission(receipt)||
       !(real(receipt,"maximum_transport_distance_m",bound)||
         real(receipt,"strict_maximum_transport_distance_m",bound)))
        return fail("source-carrier receipt is not a pinned source-only degree-one quadrature");
    if(std::abs(bound-0.019353)>1.0e-15)return fail("source-carrier receipt changes the unchanged transport-distance bound");
    const std::string canonicalLayout=
        "FSSCARR01|u64 n_parent|u64 n_carrier|u32 source_parent_tet_rows[n_parent]|u32 global_source330_tet_rows[n_parent]|u32 parent_offsets[n_parent+1]|f64 weights[n_carrier]|f64 barycentric[n_carrier,4]|u32 witness_child_rows[n_carrier]|f64 witness_distances_m[n_carrier]";
    std::string layout;
    if(str(receipt,"binary_layout",layout)&&layout!=canonicalLayout)
        return fail("source-carrier receipt declares a noncanonical FSSCARR01 binary layout");
    const bool hasInputs=object(receipt,"inputs",inputs);
    const bool hasLegacyPins=object(receipt,"input_pins",pins);
    if(!hasInputs&&!hasLegacyPins)return fail("source-carrier receipt omits its complete input-pin map");
    const std::string_view pinMap=hasInputs?std::string_view(inputs):std::string_view(pins);
    const auto verifyInput=[&](const std::filesystem::path& path,const std::string& expected)->bool{
        if(path.empty()||!verifyFilePin(path,expected,error))return false;
        if(hasInputs){
            const std::string quotedPath="\""+path.string()+"\"";
            if(inputs.find(quotedPath)==inputs.npos)return false;
        }
        return inputPinHasSHA(pinMap,expected);
    };
    if(!verifyInput(a.parentMassCSV,parentMassSHA)||
       !verifyInput(a.sourceTetTableCSV,a.sourceTetTableSHA256)||
       !verifyInput(a.sourceNodeTableCSV,a.sourceNodeTableSHA256)||
       !verifyInput(a.roleChildNode,a.roleChildNodeSHA256)||
       !verifyInput(a.roleChildElements,a.roleChildElementsSHA256)||
       !inputPinHasSHA(pinMap,roleSupportSHA))
        return fail("source-carrier receipt input pin differs from the current source/support files");
    const bool hasOutputs=object(receipt,"outputs",outputs);
    if(hasOutputs){
        if(!outputPinHasFileSHA(outputs,a.binaryPath,a.binarySHA256))
            return fail("source-carrier outputs map does not pin the exact binary path and SHA");
    }else{
        std::string binaryPath;
        if(!str(receipt,"binary_path",binaryPath)||binaryPath!=a.binaryPath.string())
            return fail("source-carrier receipt does not name the exact binary output path");
    }
    std::uint64_t receiptParents=0,receiptCarriers=0,receiptCarrierized=0;
    std::string counts;
    const bool hasCounts=object(receipt,"counts",counts);
    if(hasCounts){
        if(!num(counts,"source_parents",receiptParents)||!num(counts,"carriers",receiptCarriers)||
           !num(counts,"carrierized_parents",receiptCarrierized))
            return fail("source-carrier receipt counts are incomplete");
    }
        std::error_code ec;const auto fileBytes=std::filesystem::file_size(a.binaryPath,ec);
    if(ec||fileBytes>32u*1024u*1024u)return fail("source-carrier binary is unreadable or exceeds the bounded format size");
    std::ifstream f(a.binaryPath,std::ios::binary);if(!f)return fail("cannot open source-carrier binary");
    char magic[9];std::uint64_t np=0,nc=0;
    if(!f.read(magic,9)||std::string(magic,9)!="FSSCARR01"||!one(f,np)||!one(f,nc))
        return fail("FSSCARR01 magic or two-count header is invalid");
    if(np!=expectedParentRows.size()||!np||np>1000000u||nc>100000u)
        return fail("source-carrier parent/carrier counts differ from the current role domain");
    RolePayload r;r.roleParentRows.resize(np);r.sourceRows.resize(np);r.parentOffsets.resize(np+1u);r.weights.resize(nc);
    r.barycentric.resize(nc);r.witnessChildRows.resize(nc);r.witnessDistances.resize(nc);
    if(!vector(f,r.roleParentRows,np)||!vector(f,r.sourceRows,np)||!vector(f,r.parentOffsets,np+1u)||!vector(f,r.weights,nc))
        return fail("source-carrier role/global parent maps or primary arrays are truncated");
    for(auto& bary:r.barycentric)for(auto& x:bary)if(!one(f,x))return fail("source-carrier barycentric array is truncated");
    if(!vector(f,r.witnessChildRows,nc)||!vector(f,r.witnessDistances,nc)||f.peek()!=EOF)
        return fail("source-carrier witness arrays or exact file end are invalid");
    const std::uint64_t expectedBytes=29u+12u*np+52u*nc;
    if(expectedBytes!=fileBytes||r.parentOffsets.front()!=0u||r.parentOffsets.back()!=nc)
        return fail("source-carrier binary byte count or CSR endpoints are invalid");
    for(std::size_t row=1u;row<r.parentOffsets.size();++row)
        if(r.parentOffsets[row]<r.parentOffsets[row-1u]||r.parentOffsets[row]>nc)
            return fail("source-carrier parent offsets are decreasing or out of range");
    std::size_t measuredCarrierizedParents=0;
    for(std::size_t row=0;row<np;++row)
        if(r.parentOffsets[row]!=r.parentOffsets[row+1u])++measuredCarrierizedParents;
    if((hasCounts&&(receiptParents!=np||receiptCarriers!=nc||receiptCarrierized!=measuredCarrierizedParents))||nc==0u||measuredCarrierizedParents==0u)
        return fail("source-carrier receipt counts disagree with its exact role map and CSR");
    std::set<std::uint32_t> uniqueParents;
    for(std::size_t local=0;local<np;++local){
        const auto currentRow=expectedParentRows[local];
        const auto globalSourceRow=r.sourceRows[local];
        if(r.roleParentRows[local]!=local||currentRow>=currentParentCount||
           globalSourceRow>=currentParentCount||!uniqueParents.insert(currentRow).second||
           globalSourceRow!=currentRow)
            return fail("source-carrier role/global source-tet maps do not exactly match the role-filtered current parent row order");
    }
    if(uniqueParents.size()!=expectedParentRows.size())return fail("source-carrier source-tet map omits a current role parent");
    for(std::size_t row=0;row<nc;++row){
        if(!std::isfinite(r.weights[row])||r.weights[row]<=0.0||r.witnessChildRows[row]>=roleChildRows.size()||
           !std::isfinite(r.witnessDistances[row])||r.witnessDistances[row]>bound)
            return fail("source-carrier weight or witness lies outside its authenticated role domain");
        double sum=0.0;for(auto x:r.barycentric[row]){if(!std::isfinite(x)||x<0.0||x>1.0)return fail("source-carrier barycentric coordinate is invalid");sum+=x;}
        if(std::abs(sum-1.0)>2.0e-12)return fail("source-carrier barycentric row does not sum to one");
    }
    std::string arrays;
    if(object(receipt,"carrier_array_sha256",arrays)){
        std::string expected;
        if(!objectString(arrays,"source_parent_tet_rows_sha256",expected)||rawSHA(r.roleParentRows)!=expected||
           !objectString(arrays,"parent_offsets_sha256",expected)||rawSHA(r.parentOffsets)!=expected||
           !objectString(arrays,"source_global_tet_rows_sha256",expected)||rawSHA(r.sourceRows)!=expected||
           !objectString(arrays,"weights_sha256",expected)||rawSHA(r.weights)!=expected||
           !objectString(arrays,"barycentric_sha256",expected))return fail("source-carrier optional array hashes are malformed");
        std::vector<double> flatBary;flatBary.reserve(nc*4u);for(const auto& row:r.barycentric)flatBary.insert(flatBary.end(),row.begin(),row.end());
        if(rawSHA(flatBary)!=expected||!objectString(arrays,"witness_child_rows_sha256",expected)||rawSHA(r.witnessChildRows)!=expected||
           !objectString(arrays,"witness_distances_sha256",expected)||rawSHA(r.witnessDistances)!=expected)
            return fail("source-carrier array bytes do not match the authenticated optional array hashes");
    }
    r.globalParentRows=expectedParentRows;r.globalChildRows.reserve(nc);
    for(const auto child:r.witnessChildRows)r.globalChildRows.push_back(roleChildRows[child]);
    r.carrierizedParents=measuredCarrierizedParents;r.binarySHA=a.binarySHA256;r.receiptSHA=a.receiptSHA256;
    r.tetSHA=a.sourceTetTableSHA256;r.nodeSHA=a.sourceNodeTableSHA256;out=std::move(r);error.clear();return true;
}
} // namespace source_carrier_quadrature_import_detail

[[nodiscard]] inline bool loadPhysicalSkinSourceCarrierQuadrature090(
        const PhysicalSkinSourceCarrierQuadratureAssets& assets,
        const PhysicalSkinFractionalSemanticSupportAssets& supportAssets,
        const PhysicalSkinFractionalSemanticSupportImportReport& supportReport,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const std::array<std::string,3>& currentChildFileSHA256,
        PhysicalSkinSourceCarrierQuadratureData& output,
        PhysicalSkinSourceCarrierQuadratureImportReport& report,std::string& error) {
    using namespace source_carrier_quadrature_import_detail;
    const auto fail=[&](std::string m){error=std::move(m);return false;};
    if(!supportReport.verified||!supportReport.childRows[0]||!supportReport.childRows[1]||
       !validSHA(bundle.parent.parentMassPacketSHA256)||bundle.parent.materialIndices.empty()||
       std::any_of(currentChildFileSHA256.begin(),currentChildFileSHA256.end(),[](const auto& h){return !validSHA(h);}))
        return fail("source-carrier import requires the verified paired support and immutable parent packet");
    std::array<std::vector<std::uint32_t>,2> parentRows,childRows;
    for(std::uint32_t row=0;row<bundle.parent.materialIndices.size();++row){
        const auto m=bundle.parent.materialIndices[row];if(m==15u)parentRows[0].push_back(row);else if(m==16u)parentRows[1].push_back(row);else return fail("source-carrier parent domain contains a material outside Skin/Fat");}
    for(std::uint32_t row=0;row<bundle.childMaterialIndices.size();++row){
        const auto m=bundle.childMaterialIndices[row];if(m==15u)childRows[0].push_back(row);else if(m==16u)childRows[1].push_back(row);else return fail("source-carrier child domain contains a material outside Skin/Fat");}
    std::array<RolePayload,2> payload;
    for(unsigned i=0;i<2u;++i){
        const auto expectedMaterial=i==0u?15u:16u;
        if(assets.roles[i].materialIndex!=expectedMaterial||supportAssets.roles[i].materialIndex!=expectedMaterial)
            return fail("source-carrier and semantic-support assets must be ordered Skin 15 then Fat 16");
        if(assets.roles[i].roleChildNodeSHA256!=currentChildFileSHA256[0]||
           assets.roles[i].roleChildElementsSHA256!=currentChildFileSHA256[1])
            return fail("source-carrier role packets are not pinned to the exact current combined child node/element inputs");
        if(!readRole(assets.roles[i],expectedMaterial,parentRows[i],bundle.parent.tetrahedra.size(),childRows[i],
                bundle.parent.parentMassPacketSHA256,supportAssets.roles[i].binarySHA256,payload[i],error))return false;
        if(payload[i].parentOffsets.size()!=parentRows[i].size()+1u||payload[i].globalParentRows.size()!=parentRows[i].size())
            return fail("source-carrier role arrays do not match the paired parent material subsequence");
    }
    const std::size_t np=bundle.parent.tetrahedra.size();
    std::vector<std::vector<CarrierRow>> byParent(np);
    std::vector<std::uint8_t> seen(np,0u);std::vector<std::uint32_t> sourceRows(np,NM_INVALID_INDEX);
    for(unsigned role=0;role<2u;++role){
        const auto& r=payload[role];
        for(std::size_t local=0;local<r.globalParentRows.size();++local){
            const auto global=r.globalParentRows[local];
            if(global>=np||seen[global]||bundle.parent.materialIndices[global]!=(role==0u?15u:16u))
                return fail("source-carrier row map is not a unique typed cover of the current parent domain");
            seen[global]=1u;sourceRows[global]=r.sourceRows[local];
            for(std::uint32_t j=r.parentOffsets[local];j<r.parentOffsets[local+1u];++j)
                byParent[global].push_back({r.weights[j],r.barycentric[j],r.globalChildRows[j],r.witnessDistances[j]});
        }
    }
    if(std::any_of(seen.begin(),seen.end(),[](auto v){return v!=1u;}))return fail("source-carrier row maps do not cover every current source parent");
    const auto& semantic=bundle.fractionalSemanticSupport;
    for(std::size_t parent=0;parent<np;++parent){
        const auto& carriers=byParent[parent];if(carriers.empty())continue;
        double weightSum=0.0;std::array<double,4> firstMoment{};
        for(const auto& carrier:carriers){
            weightSum+=carrier.weight;
            for(unsigned corner=0;corner<4u;++corner)firstMoment[corner]+=carrier.weight*carrier.bary[corner];
            if(!carrierWitnessHasSemanticSupport(semantic,bundle,static_cast<std::uint32_t>(parent),carrier.child))
                return fail("source-carrier witness lacks exact material, donor, one-hop region, and parent-pool support");
            const double distance=carrierChildCentroidDistance(bundle,static_cast<std::uint32_t>(parent),carrier.bary,carrier.child);
            if(!finiteDistanceEqual(distance,carrier.distance)||distance>0.019353)
                return fail("source-carrier witness distance does not replay from the exact parent carrier and current child tetrahedron centroid");
        }
        if(std::abs(weightSum-1.0)>2.0e-12||
           std::any_of(firstMoment.begin(),firstMoment.end(),[](double x){return !std::isfinite(x)||std::abs(x-0.25)>2.0e-12;}))
            return fail("source-carrier quadrature does not preserve the source tetrahedron's exact first moment");
    }
    PhysicalSkinSourceCarrierQuadratureData out;out.parentOffsets.reserve(np+1u);out.parentOffsets.push_back(0u);
    out.globalSourceTetRows=std::move(sourceRows);
    for(std::size_t parent=0;parent<np;++parent){
        for(const auto& row:byParent[parent]){out.carrierWeights.push_back(row.weight);out.carrierBarycentric.push_back(row.bary);
            out.witnessChildRows.push_back(row.child);out.witnessChildCentroidDistancesM.push_back(row.distance);}
        out.parentOffsets.push_back(static_cast<std::uint32_t>(out.carrierWeights.size()));
    }
    spatial_mass_transport_detail::SHA256Builder packetHash,verificationHash;
    packetHash.string("numi.human.source-carrier-paired-packet.v1");
    verificationHash.string("numi.human.source-carrier-paired-import.v1");
    for(const auto& sha:currentChildFileSHA256){packetHash.string(sha);verificationHash.string(sha);}
    for(unsigned i=0;i<2u;++i){packetHash.string(payload[i].binarySHA);packetHash.string(payload[i].receiptSHA);
        packetHash.string(payload[i].tetSHA);packetHash.string(payload[i].nodeSHA);
        verificationHash.string(payload[i].binarySHA);verificationHash.string(payload[i].receiptSHA);
        verificationHash.string(payload[i].tetSHA);verificationHash.string(payload[i].nodeSHA);
        verificationHash.u64(payload[i].globalParentRows.size());verificationHash.u64(payload[i].globalChildRows.size());}
    out.sourceCarrierPacketSHA256=packetHash.finish();out.sourceTetTableSHA256=payload[0].tetSHA;out.sourceNodeTableSHA256=payload[0].nodeSHA;
    if(payload[0].tetSHA!=payload[1].tetSHA||payload[0].nodeSHA!=payload[1].nodeSHA)
        return fail("Skin and Fat source-carrier packets do not bind the same immutable source330 node/tet tables");
    if(!out.carrierWeights.empty()){
        spatial_mass_transport_detail::SHA256Builder arrays;
        arrays.string(out.sourceCarrierPacketSHA256);
        for(auto x:out.parentOffsets)arrays.u32(x);for(auto x:out.globalSourceTetRows)arrays.u32(x);
        for(auto x:out.carrierWeights)arrays.real(x);for(const auto& row:out.carrierBarycentric)for(auto x:row)arrays.real(x);
        for(auto x:out.witnessChildRows)arrays.u32(x);for(auto x:out.witnessChildCentroidDistancesM)arrays.real(x);
        verificationHash.string(arrays.finish());
    }
    out.verificationSHA256=verificationHash.finish();out.importedAndVerified=true;
    PhysicalSkinSourceCarrierQuadratureImportReport measured;
    for(unsigned i=0;i<2u;++i){measured.binarySHA256[i]=payload[i].binarySHA;measured.receiptSHA256[i]=payload[i].receiptSHA;
        measured.sourceTetTableSHA256[i]=payload[i].tetSHA;measured.sourceNodeTableSHA256[i]=payload[i].nodeSHA;
        measured.parentRows[i]=payload[i].globalParentRows.size();measured.carrierRows[i]=payload[i].weights.size();
        measured.carrierizedParentRows[i]=payload[i].carrierizedParents;}
    measured.packetSHA256=out.sourceCarrierPacketSHA256;measured.verificationSHA256=out.verificationSHA256;
    measured.currentChildFileSHA256=currentChildFileSHA256;
    measured.verified=true;measured.physicalAdmission=false;measured.nativeSteps=0;
    output=std::move(out);report=std::move(measured);error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
