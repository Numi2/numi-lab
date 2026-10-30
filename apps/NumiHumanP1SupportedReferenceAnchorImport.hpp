#pragma once

#include "NumiHumanFractionalSemanticSupportImport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo::human::coarse_tissue {

struct PhysicalSkinP1AnchorInputFile {
    std::string receiptKey;
    std::filesystem::path path;
    std::string sha256;
};
struct PhysicalSkinP1SupportedReferenceAnchorAssets {
    std::filesystem::path projectionCSV,receiptJSON,anchorCSRNPZ,parentRowMapCSV;
    std::string projectionSHA256,receiptSHA256,anchorCSR_SHA256,parentRowMapSHA256;
    std::vector<PhysicalSkinP1AnchorInputFile> pinnedInputs;
};
struct PhysicalSkinP1SupportedReferenceAnchorImportReport {
    std::string receiptSHA256,projectionSHA256,anchorCSR_SHA256,parentRowMapSHA256,verificationSHA256;
    std::size_t childRows=0,verifiedProjectionRows=0;
    bool verified=false,fullChildDomain=false,physicalAdmission=false;
    std::uint32_t nativeSteps=0;
};

namespace p1_supported_reference_anchor_import_detail {
using namespace fractional_semantic_support_import_detail;

inline bool object(std::string_view json,std::string_view key,std::string& out) {
    const std::string token="\""+std::string(key)+"\"";
    auto p=json.find(token);if(p==json.npos||json.find(token,p+token.size())!=json.npos)return false;
    p=json.find(':',p+token.size());if(p==json.npos)return false;++p;
    while(p<json.size()&&std::isspace(static_cast<unsigned char>(json[p])))++p;
    if(p==json.size()||json[p]!='{')return false;const auto begin=p;unsigned depth=0;bool quoted=false,escape=false;
    for(;p<json.size();++p){const char c=json[p];if(quoted){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')quoted=false;continue;}
        if(c=='"'){quoted=true;continue;}if(c=='{')++depth;else if(c=='}'&&--depth==0){out.assign(json.substr(begin,p-begin+1));return true;}}
    return false;
}
inline bool numValue(const std::string& x,std::uint64_t& y) {
    if(x.empty())return false;std::size_t used=0;
    try{y=std::stoull(x,&used);}catch(...){return false;}return used==x.size();
}

inline bool boolValue(std::string_view json,std::string_view key,bool& value) {
    const std::string token="\""+std::string(key)+"\"";
    auto p=json.find(token);if(p==json.npos||json.find(token,p+token.size())!=json.npos)return false;
    p=json.find(':',p+token.size());if(p==json.npos)return false;++p;
    while(p<json.size()&&std::isspace(static_cast<unsigned char>(json[p])))++p;
    if(json.substr(p,4)=="true"){value=true;return p+4==json.size()||json[p+4]==','||json[p+4]=='}'||std::isspace(static_cast<unsigned char>(json[p+4]));}
    if(json.substr(p,5)=="false"){value=false;return p+5==json.size()||json[p+5]==','||json[p+5]=='}'||std::isspace(static_cast<unsigned char>(json[p+5]));}
    return false;
}
inline bool metaFile(std::string_view outputs,std::string_view name,const std::filesystem::path& path,
                    const std::string& sha,std::string& error) {
    std::string meta,gotPath,gotSHA;
    if(!object(outputs,name,meta)||!str(meta,"path",gotPath)||!str(meta,"sha256",gotSHA)||
       gotPath!=path.string()||gotSHA!=sha){error="P1 anchor receipt output path/hash differs for "+std::string(name);return false;}
    return true;
}
inline bool pinnedInput(std::string_view inputs,const PhysicalSkinP1AnchorInputFile& expected,
                        std::string& error) {
    std::string item,path,sha;
    if(expected.receiptKey.empty()||expected.path.empty()||!validSHA(expected.sha256)||
       !physical_skin_child_mass_detail::sha256File(expected.path,sha,error)||sha!=expected.sha256||
       !object(inputs,expected.receiptKey,item)||!str(item,"path",path)||!str(item,"sha256",sha)||
       path!=expected.path.string()||sha!=expected.sha256) {
        if(error.empty())error="P1 anchor receipt input path/hash differs for "+expected.receiptKey;
        return false;
    }
    return true;
}
inline bool close(double a,double b,double tolerance) {
    return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance;
}
inline bool parseCSVRecord(const std::string& line,std::vector<std::string>& fields) {
    fields.clear();std::string field;bool quoted=false;
    const std::size_t lineSize=!line.empty()&&line.back()=="\r"[0]?line.size()-1u:line.size();
    for(std::size_t i=0;i<lineSize;++i){
        const char c=line[i];
        if(quoted){
            if(c=='"'){
                if(i+1u<line.size()&&line[i+1u]=='"'){field.push_back('"');++i;}
                else quoted=false;
            }else field.push_back(c);
        }else if(c=='"'){
            if(!field.empty())return false;quoted=true;
        }else if(c==','){fields.push_back(std::move(field));field.clear();}
        else field.push_back(c);
    }
    if(quoted)return false;fields.push_back(std::move(field));return true;
}
inline bool parseU32(const std::string& x,std::uint32_t& y) {
    std::uint64_t n=0;if(!numValue(x,n)||n>UINT32_MAX)return false;y=static_cast<std::uint32_t>(n);return true;
}
inline bool doubleValue(const std::string& x,double& y) {
    if(x.empty())return false;std::size_t used=0;
    try{y=std::stod(x,&used);}catch(...){return false;}
    return used==x.size()&&std::isfinite(y);
}
inline bool groupContainsParent(const ConservativeSpatialChildSemanticSupportData& support,
        std::uint32_t group,std::uint32_t parent,const std::array<std::uint32_t,4>& tuple) {
    const auto n=support.groupMaterialIndices.size();
    if(group>=n||support.donorBodyIndices.size()!=n||support.anatomicalRegionIndices.size()!=n||
       support.sourceParentTetOffsets.size()!=n+1u||
       support.sourceParentTetOffsets[group]>support.sourceParentTetOffsets[group+1u]||
       support.sourceParentTetOffsets[group+1u]>support.sourceParentTetRows.size()||
       support.groupMaterialIndices[group]!=tuple[0]||support.donorBodyIndices[group]!=tuple[1]||
       support.anatomicalRegionIndices[group]!=tuple[2])return false;
    const auto begin=support.sourceParentTetOffsets[group],end=support.sourceParentTetOffsets[group+1u];
    return std::find(support.sourceParentTetRows.begin()+begin,support.sourceParentTetRows.begin()+end,parent)!=
        support.sourceParentTetRows.begin()+end;
}
inline bool jsonArrayWellFormed(const std::string& value) {
    if(value.size()<2u||value.front()!='['||value.back()!=']')return false;
    int depth=0;bool quoted=false,escape=false;
    for(char c:value){if(quoted){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')quoted=false;continue;}
        if(c=='"'){quoted=true;continue;}
        if(c=='[')++depth;else if(c==']'&&--depth<0)return false;
    }
    return !quoted&&depth==0;
}
inline bool containsJSONInteger(const std::string& array,std::uint32_t expected) {
    if(!jsonArrayWellFormed(array))return false;
    const auto needle=std::to_string(expected);std::size_t p=0;
    while((p=array.find(needle,p))!=std::string::npos){
        const auto before=p==0u?'[':array[p-1u];const auto after=p+needle.size()==array.size()?']':array[p+needle.size()];
        if((before=='['||before==','||std::isspace(static_cast<unsigned char>(before)))&&
           (after==']'||after==','||std::isspace(static_cast<unsigned char>(after)))&&
           !(p>0u&&before=='-' ))return true;
        p+=needle.size();
    }
    return false;
}
inline bool containsSemanticTuple(std::string value,const std::array<std::uint32_t,4>& tuple) {
    if(!jsonArrayWellFormed(value))return false;
    value.erase(std::remove_if(value.begin(),value.end(),[](unsigned char c){return std::isspace(c)!=0;}),value.end());
    const std::string needle="["+std::to_string(tuple[0])+","+std::to_string(tuple[1])+","+
        std::to_string(tuple[2])+","+std::to_string(tuple[3])+"]";
    return value.find(needle)!=std::string::npos;
}
inline bool supportedParentForTuple(const ConservativeSpatialChildSemanticSupportData& support,
        std::uint32_t parent,const std::array<std::uint32_t,4>& tuple) {
    const auto groups=support.groupMaterialIndices.size();
    for(std::uint32_t g=0;g<groups;++g)
        if(groupContainsParent(support,g,parent,tuple))return true;
    return false;
}
} // namespace p1_supported_reference_anchor_import_detail

[[nodiscard]] inline bool loadPhysicalSkinP1SupportedReferenceAnchor(
        const PhysicalSkinP1SupportedReferenceAnchorAssets& assets,
        const PhysicalSkinChildMassMappingBundle& bundle,
        const PhysicalSkinFractionalSemanticSupportImportReport& supportReport,
        const std::array<std::string,3>& currentChildFileSHA256,
        const std::filesystem::path& childLabelsPath,const std::string& childLabelsSHA256,
        PhysicalSkinP1SupportedReferenceAnchorImportReport& output,std::string& error) {
    using namespace p1_supported_reference_anchor_import_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const auto childCount=bundle.childTetrahedra.size();
    const auto parentCount=bundle.parent.tetrahedra.size();
    const auto& support=bundle.fractionalSemanticSupport;
    if(!supportReport.verified||childCount==0u||parentCount==0u||
       !validSHA(childLabelsSHA256)||!validSHA(bundle.parent.parentMassPacketSHA256)||
       !validSHA(bundle.parent.parentSourceContributionSHA256)||
       !validSHA(bundle.parent.parentTopologySHA256)||
       !validSHA(bundle.childMeshSHA256)||!validSHA(bundle.regionLabelSHA256)||
       currentChildFileSHA256[2]!=childLabelsSHA256||
       std::any_of(currentChildFileSHA256.begin(),currentChildFileSHA256.end(),[](const auto& h){return !validSHA(h);}))
        return fail("P1 anchor import requires the verified paired support and complete current parent/child domains");
    for(const auto& path:{assets.projectionCSV,assets.receiptJSON,assets.anchorCSRNPZ,assets.parentRowMapCSV})
        if(path.empty())return fail("P1 anchor projection, receipt, CSR, and parent-row map files are all required");
    for(const auto& hash:{assets.projectionSHA256,assets.receiptSHA256,assets.anchorCSR_SHA256,assets.parentRowMapSHA256})
        if(!validSHA(hash))return fail("P1 anchor output hash pins are incomplete");
    std::string receipt;
    if(!pinnedJSON(assets.receiptJSON,assets.receiptSHA256,receipt,error))return false;
    std::string schema,status,massStatus;
    std::uint64_t nativeSteps=0,childRows=0,exportedRows=0;
    if(!str(receipt,"schema",schema)||schema!="numi.native-domain-217.p1-supported-reference-anchor.v1"||
       !str(receipt,"status",status)||status!="P1_SUPPORTED_REFERENCE_ANCHOR_LABELS_EXPORTED_NOT_PHYSICAL_ADMISSION"||
       !num(receipt,"native_steps",nativeSteps)||nativeSteps!=0u||!noAdmission(receipt)||
       !str(receipt,"mass_transport",massStatus)||massStatus!="NOT_RUN"||
       !num(receipt,"full_child_domain_exported_rows",exportedRows)||exportedRows!=childCount)
        return fail("P1 anchor receipt is not a full source-only export or reports a physical/mass run");
    std::string counts,outputs,inputs,projectionMeta,csrMeta,labelMeta,parentMapMeta;
    if(!object(receipt,"counts",counts)||!num(counts,"child_tetrahedra",childRows)||childRows!=childCount||
       !num(counts,"exported_child_rows",exportedRows)||exportedRows!=childCount||
       !object(receipt,"outputs",outputs)||
       !object(outputs,"p1_anchor_projection_csv",projectionMeta)||
       !object(outputs,"p1_anchor_csr",csrMeta)||
       !object(outputs,"labels",labelMeta)||
       !object(outputs,"parent_row_map",parentMapMeta))
        return fail("P1 anchor receipt lacks complete full-domain output and row-count records");
    std::string projectionPath,projectionSHA,csrPath,csrSHA,labelPath,labelSHA,parentMapPath,parentMapSHA;
    std::uint64_t rowCount=0;bool fullDomain=false,fullChildDomain=false;
    if(!str(projectionMeta,"path",projectionPath)||projectionPath!=assets.projectionCSV.string()||
       !str(projectionMeta,"sha256",projectionSHA)||projectionSHA!=assets.projectionSHA256||
       !num(projectionMeta,"row_count",rowCount)||rowCount!=childCount||
       !boolValue(projectionMeta,"full_domain",fullDomain)||!fullDomain||
       !str(csrMeta,"path",csrPath)||csrPath!=assets.anchorCSRNPZ.string()||
       !str(csrMeta,"sha256",csrSHA)||csrSHA!=assets.anchorCSR_SHA256||
       !str(labelMeta,"path",labelPath)||labelPath!=childLabelsPath.string()||
       !str(labelMeta,"sha256",labelSHA)||labelSHA!=childLabelsSHA256||
       !str(parentMapMeta,"path",parentMapPath)||parentMapPath!=assets.parentRowMapCSV.string()||
       !str(parentMapMeta,"sha256",parentMapSHA)||parentMapSHA!=assets.parentRowMapSHA256||
       !boolValue(receipt,"full_child_domain_export",fullChildDomain)||!fullChildDomain)
        return fail("P1 anchor receipt output pins do not cover the exact complete child domain");
    if(!physical_skin_child_mass_detail::sha256File(assets.projectionCSV,projectionSHA,error)||
       projectionSHA!=assets.projectionSHA256||
       !physical_skin_child_mass_detail::sha256File(assets.anchorCSRNPZ,csrSHA,error)||
       csrSHA!=assets.anchorCSR_SHA256||
       !physical_skin_child_mass_detail::sha256File(assets.parentRowMapCSV,parentMapSHA,error)||
       parentMapSHA!=assets.parentRowMapSHA256||
       !physical_skin_child_mass_detail::sha256File(childLabelsPath,labelSHA,error)||labelSHA!=childLabelsSHA256)
        return fail("P1 anchor output bytes differ from the full receipt hashes");
    if(!object(receipt,"inputs",inputs))return fail("P1 anchor receipt omits its source input map");
    const std::set<std::string> requiredKeys{
        "child_node","child_elements","parent_mass","parent_nodes","parent_source","parent_tets",
        "parent_full86_weights","skin_p1_support","fat_p1_support"};
    std::set<std::string> seenKeys;std::array<std::string,4> parentHashes{};
    for(const auto& pin:assets.pinnedInputs){
        if(!seenKeys.insert(pin.receiptKey).second||!pinnedInput(inputs,pin,error))return false;
        if(pin.receiptKey=="child_node"&&pin.sha256!=currentChildFileSHA256[0])
            return fail("P1 source receipt is pinned to another child node file");
        if(pin.receiptKey=="child_elements"&&pin.sha256!=currentChildFileSHA256[1])
            return fail("P1 source receipt is pinned to another child element file");
        if(pin.receiptKey=="skin_p1_support"&&pin.sha256!=supportReport.binarySHA256[0])
            return fail("P1 source receipt is pinned to another Skin semantic-support binary");
        if(pin.receiptKey=="fat_p1_support"&&pin.sha256!=supportReport.binarySHA256[1])
            return fail("P1 source receipt is pinned to another Fat semantic-support binary");
        if(pin.receiptKey=="parent_mass"&&pin.sha256!=bundle.parent.parentMassPacketSHA256)
            return fail("P1 source receipt is pinned to another immutable parent mass packet");
        if(pin.receiptKey=="parent_source"&&pin.sha256!=bundle.parent.parentSourceContributionSHA256)
            return fail("P1 source receipt is pinned to another immutable parent donor ledger");
        if(pin.receiptKey=="parent_mass")parentHashes[0]=pin.sha256;
        else if(pin.receiptKey=="parent_nodes")parentHashes[1]=pin.sha256;
        else if(pin.receiptKey=="parent_source")parentHashes[2]=pin.sha256;
        else if(pin.receiptKey=="parent_tets")parentHashes[3]=pin.sha256;
    }
    for(const auto& key:requiredKeys)if(!seenKeys.contains(key))
        return fail("P1 anchor receipt inputs omit required current source pin "+key);
    std::string recomputedTopology;
    if(!physical_skin_child_mass_detail::hashJoinedFiles(parentHashes,recomputedTopology)||
       recomputedTopology!=bundle.parent.parentTopologySHA256)
        return fail("P1 source file pins do not reproduce the imported parent topology hash");

    const std::vector<std::string> expectedHeader{
        "child_tet_row","carrier_child_node_row","carrier_query_x_m","carrier_query_y_m","carrier_query_z_m",
        "selected_parent_tet_row","selected_parent_barycentric0","selected_parent_barycentric1",
        "selected_parent_barycentric2","selected_parent_barycentric3","registered_world_x_m",
        "registered_world_y_m","registered_world_z_m","residual_x_m","residual_y_m","residual_z_m",
        "anchor_distance_m","material_index","geometric_donor_body","anatomical_region_id",
        "stable_source_face_id","primary_group_id","carrier_node_group_id","primary_score_numerator",
        "primary_score_denominator","exact_resolution_used","exact_min_squared_distance_ratio",
        "tie_parent_rows_json","tie_semantic_tuples_json"};
    std::string beforeHash,afterHash;
    if(!physical_skin_child_mass_detail::sha256File(assets.projectionCSV,beforeHash,error)||
       beforeHash!=assets.projectionSHA256)return fail("P1 projection CSV hash changed before parsing");
    std::ifstream csv(assets.projectionCSV);if(!csv)return fail("cannot open full P1 anchor projection CSV");
    std::string line;std::vector<std::string> fields;
    if(!std::getline(csv,line)||!parseCSVRecord(line,fields)||fields!=expectedHeader)
        return fail("P1 projection CSV header differs from the authenticated full v1 schema");
    std::size_t row=0u;
    while(std::getline(csv,line)){
        if(row>=childCount||!parseCSVRecord(line,fields)||fields.size()!=expectedHeader.size())
            return fail("P1 projection CSV has an extra or malformed child row");
        std::uint32_t child=0,node=0,parent=0,material=0,donor=0,region=0,face=0,
            primaryGroup=0,carrierGroup=0,scoreNumerator=0,scoreDenominator=0,exact=0;
        if(!parseU32(fields[0],child)||child!=row||!parseU32(fields[1],node)||
           !parseU32(fields[5],parent)||!parseU32(fields[17],material)||
           !parseU32(fields[18],donor)||!parseU32(fields[19],region)||!parseU32(fields[20],face)||
           !parseU32(fields[21],primaryGroup)||!parseU32(fields[22],carrierGroup)||
           !parseU32(fields[23],scoreNumerator)||!parseU32(fields[24],scoreDenominator)||
           !parseU32(fields[25],exact)||exact>1u||scoreDenominator==0u||scoreNumerator>scoreDenominator||
           node>=bundle.referenceWorldPoints.size()||parent>=parentCount)
            return fail("P1 projection row has an invalid child, parent, node, or typed score identity");
        const auto childTet=bundle.childTetrahedra[child];
        if(std::find(childTet.begin(),childTet.end(),node)==childTet.end())
            return fail("P1 anchor carrier node is not one of the exact child tetrahedron vertices");
        std::array<double,3> query{},registered{},residual{};
        for(unsigned axis=0;axis<3u;++axis){
            if(!doubleValue(fields[2u+axis],query[axis])||
               !doubleValue(fields[10u+axis],registered[axis])||
               !doubleValue(fields[13u+axis],residual[axis]))
                return fail("P1 projection row has a nonfinite coordinate or residual");
            if(!close(query[axis],bundle.referenceWorldPoints[node][axis],1.0e-12))
                return fail("P1 projection query point differs from the authenticated child node coordinate");
        }
        std::array<double,4> bary{};double barySum=0.0;
        for(unsigned corner=0;corner<4u;++corner){
            if(!doubleValue(fields[6u+corner],bary[corner])||bary[corner]<-1.0e-12||bary[corner]>1.0+1.0e-12)
                return fail("P1 selected-parent barycentric coordinate lies outside the tetrahedron");
            barySum+=bary[corner];
        }
        if(!close(barySum,1.0,2.0e-12))return fail("P1 selected-parent barycentric coordinates do not sum to one");
        std::array<double,3> predicted{};
        const auto sourceTet=bundle.parent.tetrahedra[parent];
        for(unsigned corner=0;corner<4u;++corner){
            const auto sourceNode=sourceTet[corner];
            if(sourceNode>=bundle.parent.currentWorldPoints.size())
                return fail("P1 selected parent tet references a missing source node");
            for(unsigned axis=0;axis<3u;++axis)
                predicted[axis]+=bary[corner]*bundle.parent.currentWorldPoints[sourceNode][axis];
        }
        for(unsigned axis=0;axis<3u;++axis){
            if(!close(registered[axis],predicted[axis],2.0e-11)||
               !close(residual[axis],query[axis]-registered[axis],2.0e-11))
                return fail("P1 projection point/residual does not replay from the exact selected source parent tet");
        }
        double recordedDistance=0.0,actualDistance=0.0;
        if(!doubleValue(fields[16],recordedDistance)||recordedDistance<0.0)
            return fail("P1 anchor distance is invalid");
        for(unsigned axis=0;axis<3u;++axis)actualDistance+=residual[axis]*residual[axis];
        actualDistance=std::sqrt(actualDistance);
        if(!close(recordedDistance,actualDistance,2.0e-11)||recordedDistance>0.020)
            return fail("P1 anchor distance does not match its exact child-node residual or exceeds the explicit 20 mm construction bound");
        const std::array<std::uint32_t,4> tuple{
            bundle.childMaterialIndices[child],bundle.childDonorBodyIndices[child],
            bundle.childAnatomicalRegionIndices[child],bundle.childStableSourceFaceIdentities[child]};
        if(material!=tuple[0]||donor!=tuple[1]||region!=tuple[2]||face!=tuple[3]||
           bundle.parent.materialIndices[parent]!=material||
           bundle.parent.stableSourceFaceIdentities[parent]!=face)
            return fail("P1 scalar compatibility tuple differs from child labels or the selected source parent");
        if(primaryGroup!=carrierGroup||primaryGroup>=support.groupMaterialIndices.size()||
           carrierGroup>=support.groupMaterialIndices.size()||
           !groupContainsParent(support,primaryGroup,parent,tuple)||
           !groupContainsParent(support,carrierGroup,parent,tuple)||
           !supportedParentForTuple(support,parent,tuple))
            return fail("P1 selected parent is not present in the authenticated fractional support group");
        if(!containsJSONInteger(fields[27],parent)||!containsSemanticTuple(fields[28],tuple))
            return fail("P1 selected parent/semantic tuple is absent from the exact tie CSR projection");
        if(exact==1u&&fields[26].empty())
            return fail("P1 exact tie resolution omits its squared-distance ratio witness");
        ++row;
    }
    if(!csv.eof()||row!=childCount)return fail("P1 projection CSV does not cover every combined child tetrahedron exactly once");
    if(!physical_skin_child_mass_detail::sha256File(assets.projectionCSV,afterHash,error)||
       afterHash!=beforeHash)return fail("P1 projection CSV changed while being imported");
    spatial_mass_transport_detail::SHA256Builder verification;
    verification.string("numi.native-domain-217.p1-supported-reference-anchor-import.v1");
    verification.string(assets.receiptSHA256);verification.string(assets.projectionSHA256);
    verification.string(assets.anchorCSR_SHA256);verification.string(assets.parentRowMapSHA256);
    verification.string(bundle.parent.parentMassPacketSHA256);verification.string(bundle.parent.parentTopologySHA256);
    verification.string(bundle.parent.parentSourceContributionSHA256);
    for(const auto& h:currentChildFileSHA256)verification.string(h);
    verification.u64(childCount);
    PhysicalSkinP1SupportedReferenceAnchorImportReport measured;
    measured.receiptSHA256=assets.receiptSHA256;measured.projectionSHA256=assets.projectionSHA256;
    measured.anchorCSR_SHA256=assets.anchorCSR_SHA256;measured.parentRowMapSHA256=assets.parentRowMapSHA256;
    measured.verificationSHA256=verification.finish();measured.childRows=childCount;
    measured.verifiedProjectionRows=row;measured.verified=true;measured.fullChildDomain=true;
    measured.physicalAdmission=false;measured.nativeSteps=0;
    output=std::move(measured);error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
