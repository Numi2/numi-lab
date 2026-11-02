#pragma once

#include "NumiHumanCurrentFractionalRecordedFlowArtifacts.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace metalrobo::human::coarse_tissue {
struct CurrentFractionalRecordedFlowBinding {
    std::filesystem::path admissionReceiptPath,massTransportEnvelopeReceiptPath;
    std::string admissionReceiptSHA256,massTransportEnvelopeReceiptSHA256,transportReceiptSHA256,source330LabelSHA256;
    double transportDistanceLimitM=0.0;
    bool massTransportEnvelopeEnabled=false;
    std::array<PhysicalSkinRecordedCarrierAllocationCSVAssets,2> roleAssets{};
    std::array<PhysicalSkinRecordedCarrierAllocationCSVImportReport,2> roleReports{};
    std::array<std::string,2> roleP1SHA256{},roleCarrierPacketSHA256{};
    std::array<std::size_t,2> roleParentRows{},roleChildRows{};
    std::size_t candidateParentRows=0,candidateChildRows=0;
    bool environmentLoaded=false,admissionVerified=false,rolesVerified=false;
};
namespace current_fractional_recorded_flow_binding_detail {
using namespace fractional_semantic_support_import_detail;
using namespace source_carrier_quadrature_import_detail;
using namespace recorded_carrier_allocation_import_detail;
using namespace physical_skin_child_mass_detail;

inline bool env(const std::string& key,std::string& value,std::string& error) {
    const char* raw=std::getenv(key.c_str());
    if(!raw||raw[0]=='\0'){error="missing required recorded-flow environment variable "+key;return false;}
    value=raw;return true;
}
inline bool envSHA(const std::string& key,std::string& value,std::string& error) {
    if(!env(key,value,error))return false;
    if(!validSHA256(value)){error="malformed SHA-256 in "+key;return false;}
    return true;
}
inline bool envPath(const std::string& key,std::filesystem::path& value,std::string& error) {
    std::string raw;if(!env(key,raw,error))return false;value=raw;return true;
}
inline bool readSmallPinnedJSON(const std::filesystem::path& path,const std::string& expected,
                               std::string& text,std::string& error) {
    if(!verifyFilePin(path,expected,error))return false;
    std::error_code ec;const auto size=std::filesystem::file_size(path,ec);
    if(ec||size>32u*1024u*1024u){error="pinned mass-admission receipt is unreadable or oversized";return false;}
    std::ifstream f(path,std::ios::binary);if(!f){error="cannot read pinned mass-admission receipt";return false;}
    text.assign(std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>());
    if(f.bad()){error="failed while reading pinned mass-admission receipt";return false;}
    std::string after;if(!sha256File(path,after,error)||after!=expected){error="mass-admission receipt changed while being read";return false;}
    return true;
}
inline void skipSpace(std::string_view text,std::size_t& p) {
    while(p<text.size()&&std::isspace(static_cast<unsigned char>(text[p])))++p;
}
inline bool skipJSONString(std::string_view text,std::size_t& p) {
    if(p>=text.size()||text[p]!='"')return false;
    ++p;bool escape=false;
    for(;p<text.size();++p){const char c=text[p];if(escape){escape=false;continue;}if(c=='\\'){escape=true;continue;}if(c=='"'){++p;return true;}}
    return false;
}
inline bool skipJSONValue(std::string_view text,std::size_t& p) {
    skipSpace(text,p);if(p>=text.size())return false;
    if(text[p]=='"')return skipJSONString(text,p);
    if(text[p]=='{'||text[p]=='['){
        std::vector<char> stack;stack.push_back(text[p++]);bool quoted=false,escape=false;
        for(;p<text.size();++p){const char c=text[p];if(quoted){if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')quoted=false;continue;}
            if(c=='"'){quoted=true;continue;}
            if(c=='{'||c=='['){stack.push_back(c);continue;}
            if(c=='}'||c==']'){
                if(stack.empty()||((c=='}')!=(stack.back()=='{')))return false;
                stack.pop_back();if(stack.empty()){++p;return true;}
            }
        }
        return false;
    }
    const auto begin=p;while(p<text.size()&&text[p]!=','&&text[p]!='}'&&text[p]!=']'&&
        !std::isspace(static_cast<unsigned char>(text[p])))++p;
    return p>begin;
}
inline bool objectHasExactKeysImpl(std::string_view objectText,const std::set<std::string>& wanted) {
    std::size_t p=0;skipSpace(objectText,p);if(p>=objectText.size()||objectText[p++]!='{')return false;
    std::set<std::string> found;skipSpace(objectText,p);
    if(p<objectText.size()&&objectText[p]=='}')++p;
    else for(;;){
        skipSpace(objectText,p);if(p>=objectText.size()||objectText[p]!='"')return false;
        const auto keyBegin=++p;while(p<objectText.size()&&objectText[p]!='"'){
            if(objectText[p]=='\\')return false;++p;
        }
        if(p>=objectText.size())return false;
        const std::string key(objectText.substr(keyBegin,p-keyBegin));++p;skipSpace(objectText,p);
        if(p>=objectText.size()||objectText[p++]!=':')return false;
        if(!found.insert(key).second||!skipJSONValue(objectText,p))return false;
        skipSpace(objectText,p);if(p>=objectText.size())return false;
        if(objectText[p]=='}'){++p;break;}if(objectText[p++]!=',')return false;
    }
    skipSpace(objectText,p);return p==objectText.size()&&found==wanted;
}
inline bool objectHasExactKeys(std::string_view objectText,
                               std::initializer_list<std::string_view> expected) {
    std::set<std::string> wanted;for(const auto key:expected)wanted.emplace(key);
    return wanted.size()==expected.size()&&objectHasExactKeysImpl(objectText,wanted);
}
template<std::size_t N> inline bool objectHasExactKeys(
        std::string_view objectText,const std::array<std::string_view,N>& expected) {
    std::set<std::string> wanted;for(const auto key:expected)wanted.emplace(key);
    return wanted.size()==expected.size()&&objectHasExactKeysImpl(objectText,wanted);
}
inline bool topLevelRawValue(std::string_view json,std::string_view key,std::string& value) {
    std::size_t p=0;skipSpace(json,p);if(p>=json.size()||json[p++]!='{')return false;
    std::set<std::string> seen;skipSpace(json,p);
    if(p<json.size()&&json[p]=='}')return false;
    for(;;){
        skipSpace(json,p);if(p>=json.size()||json[p]!='"')return false;
        const auto beginKey=++p;while(p<json.size()&&json[p]!='"'){if(json[p]=='\\')return false;++p;}
        if(p>=json.size())return false;
        const std::string name(json.substr(beginKey,p-beginKey));++p;skipSpace(json,p);
        if(p>=json.size()||json[p++]!=':'||!seen.insert(name).second)return false;
        skipSpace(json,p);const auto beginValue=p;
        if(!skipJSONValue(json,p))return false;
        if(name==key){if(!value.empty())return false;value.assign(json.substr(beginValue,p-beginValue));}
        skipSpace(json,p);if(p>=json.size())return false;
        if(json[p]=='}'){++p;break;}if(json[p++]!=',')return false;
    }
    return !value.empty();
}
inline bool topLevelString(std::string_view json,std::string_view key,std::string& value) {
    std::string raw;if(!topLevelRawValue(json,key,raw)||raw.size()<2u||raw.front()!='"'||raw.back()!='"')return false;
    value.assign(raw.substr(1u,raw.size()-2u));return value.find('\"')==std::string::npos;
}
inline bool topLevelFiniteNumber(std::string_view json,std::string_view key,double& value) {
    std::string raw;if(!topLevelRawValue(json,key,raw)||raw.empty())return false;
    char* end=nullptr;const double parsed=std::strtod(raw.c_str(),&end);
    if(end==raw.c_str()||!end||*end!='\0'||!std::isfinite(parsed))return false;
    value=parsed;return true;
}
inline bool receiptNumber(std::string_view json,std::string_view key,double& value) {
    return source_carrier_quadrature_import_detail::real(json,key,value)&&std::isfinite(value);
}
inline bool requireFiniteNonnegative(std::string_view json,std::string_view key,double limit) {
    double value=0.0;return receiptNumber(json,key,value)&&value>=0.0&&value<=limit;
}
inline bool readPin(std::string_view map,const std::string& name,std::filesystem::path& path,
                    std::string& sha,std::string& error) {
    std::string pin,pathText;
    if(!object(map,name,pin)||!objectHasExactKeys(pin,{"path","sha256"})||
       !str(pin,"path",pathText)||!str(pin,"sha256",sha)||!validSHA256(sha)){
        error="admission receipt omits or malforms named path/SHA input pin "+name;return false;
    }
    path=pathText;return verifyFilePin(path,sha,error);
}
inline bool receiptBool(std::string_view json,std::string_view key,bool expected) {
    bool value=false;return boolean(json,key,value)&&value==expected;
}
inline bool parseFinitePositiveDistanceDecimal(const std::string& text,double& value) {
    if(text.empty()||std::any_of(text.begin(),text.end(),[](unsigned char c){return std::isspace(c);}))return false;
    char* end=nullptr;const double parsed=std::strtod(text.c_str(),&end);
    if(end==text.c_str()||!end||*end!='\0'||!std::isfinite(parsed)||parsed<=0.0)return false;
    value=parsed;return true;
}
// The Python envelope reader preserves declared bounds as exact decimal text.
// Accept that representation only when it equals the authenticated exact_decimal.
inline bool envelopeDeclaredDistance(std::string_view object,const std::string& exactDecimal,double& value) {
    if(topLevelFiniteNumber(object,"predeclared_transport_envelope_m",value))return true;
    std::string text;
    return topLevelString(object,"predeclared_transport_envelope_m",text)&&text==exactDecimal&&
        parseFinitePositiveDistanceDecimal(text,value);
}
inline bool validateMassTransportDistanceContract(const std::string& exactDecimal,
        const double maximumDeclaredSourceCentroidDistanceM,const double envelopeLimitM,
        const double configuredImportLimitM,const double massReceiptLimitM,
        const double actualMaximumUsedM,std::string& error) {
    double parsedDecimal=0.0;
    if(!parseFinitePositiveDistanceDecimal(exactDecimal,parsedDecimal)||
       !std::isfinite(maximumDeclaredSourceCentroidDistanceM)||maximumDeclaredSourceCentroidDistanceM<0.0||
       !std::isfinite(envelopeLimitM)||envelopeLimitM<=0.0||
       !std::isfinite(configuredImportLimitM)||configuredImportLimitM<=0.0||
       !std::isfinite(massReceiptLimitM)||massReceiptLimitM<=0.0||
       !std::isfinite(actualMaximumUsedM)||actualMaximumUsedM<0.0) {
        error="mass transport envelope has a nonfinite or nonpositive bound";return false;
    }
    const double derivedLimitM=std::max(200.0,std::ceil(maximumDeclaredSourceCentroidDistanceM*1.0e6)+1.0)/1.0e6;
    if(parsedDecimal!=envelopeLimitM||derivedLimitM!=envelopeLimitM||
       configuredImportLimitM!=envelopeLimitM||massReceiptLimitM!=envelopeLimitM) {
        error="geometry, stage import, and mass receipt transport bounds differ";return false;
    }
    if(actualMaximumUsedM>envelopeLimitM) {
        error="recorded mass transport exceeds the authenticated geometry envelope";return false;
    }
    error.clear();return true;
}
inline bool validateCurrentGeometryEnvelope(std::string_view candidate,
        const CurrentFractionalRecordedFlowBinding& binding,const double massReceiptLimitM,
        double& resolvedLimitM,std::string& error) {
    std::string envelope,receiptRef,receiptPathText,receiptSHA,schema,ruleID,exactDecimal,inputSHA;
    double maxDeclared=0.0,envelopeLimit=0.0;
    if(!object(candidate,"mass_transport_envelope",envelope)||
       !objectHasExactKeys(envelope,{"receipt","schema","rule_id",
          "maximum_declared_source_centroid_distance_m","predeclared_transport_envelope_m","exact_decimal","input_sha256"})||
       !object(envelope,"receipt",receiptRef)||!objectHasExactKeys(receiptRef,{"path","sha256"})||
       !str(receiptRef,"path",receiptPathText)||!str(receiptRef,"sha256",receiptSHA)||
       !validSHA256(receiptSHA)||receiptPathText!=binding.massTransportEnvelopeReceiptPath.string()||
       receiptSHA!=binding.massTransportEnvelopeReceiptSHA256||
       !str(envelope,"schema",schema)||schema!="numi.human.mass-transport-envelope.v1"||
       !str(envelope,"rule_id",ruleID)||ruleID!="ceil_max_declared_source_centroid_um_plus_1_floor_200"||
       !receiptNumber(envelope,"maximum_declared_source_centroid_distance_m",maxDeclared)||
       !str(envelope,"exact_decimal",exactDecimal)||
       !envelopeDeclaredDistance(envelope,exactDecimal,envelopeLimit)||!object(envelope,"input_sha256",inputSHA)) {
        error="unified admission omits or malforms the authenticated geometry envelope binding";return false;
    }
    std::string geometryReceipt;
    if(!readSmallPinnedJSON(binding.massTransportEnvelopeReceiptPath,
                            binding.massTransportEnvelopeReceiptSHA256,geometryReceipt,error))return false;
    std::string geometrySchema,geometryEnvelope,geometryInputs,geometryPhysicalAdmission,geometryNativeSteps;
    double geometryTopLevelLimit=0.0;
    if(!topLevelString(geometryReceipt,"schema",geometrySchema)||
       geometrySchema!="numi-human.registered-volume-parent-identity-source-control.v1") {
        error="pinned geometry receipt has an unexpected top-level schema";return false;
    }
    if(!topLevelRawValue(geometryReceipt,"physical_admission",geometryPhysicalAdmission)||
       geometryPhysicalAdmission!="false") {
        error="pinned geometry receipt claims physical admission or omits its source-only flag";return false;
    }
    if(!topLevelRawValue(geometryReceipt,"native_steps",geometryNativeSteps)||geometryNativeSteps!="0") {
        error="pinned geometry receipt claims native steps or omits its source-only count";return false;
    }
    if(!topLevelFiniteNumber(geometryReceipt,"mass_mapping_maximum_transport_distance_m",geometryTopLevelLimit)||
       !object(geometryReceipt,"mass_transport_envelope",geometryEnvelope)||
       !objectHasExactKeys(geometryEnvelope,{"exact_decimal","input_sha256",
          "maximum_declared_source_centroid_distance_m","predeclared_transport_envelope_m","rule_id","schema","status"})) {
        error="pinned geometry receipt omits the declared mass transport envelope or its exact schema";return false;
    }
    std::string geometryEnvelopeSchema,geometryStatus,geometryRule,geometryDecimal;
    double geometryMaxDeclared=0.0,geometryLimit=0.0;
    if(!str(geometryEnvelope,"schema",geometryEnvelopeSchema)||
       geometryEnvelopeSchema!="numi.human.mass-transport-envelope.v1"||
       !str(geometryEnvelope,"status",geometryStatus)||geometryStatus!="PREDECLARED_CURRENT_GEOMETRY_ENVELOPE"||
       !str(geometryEnvelope,"rule_id",geometryRule)||
       geometryRule!="ceil_max_declared_source_centroid_um_plus_1_floor_200"||
       !str(geometryEnvelope,"exact_decimal",geometryDecimal)||geometryDecimal!=exactDecimal||
       !receiptNumber(geometryEnvelope,"maximum_declared_source_centroid_distance_m",geometryMaxDeclared)||
       geometryMaxDeclared!=maxDeclared||
       !envelopeDeclaredDistance(geometryEnvelope,geometryDecimal,geometryLimit)||
       geometryLimit!=envelopeLimit||geometryTopLevelLimit!=envelopeLimit||
       !object(geometryEnvelope,"input_sha256",geometryInputs)) {
        error="pinned geometry envelope values differ from the unified mass receipt";return false;
    }
    constexpr std::array<std::string_view,13> inputNames{
      "candidate","child_element","child_labels","child_node","coordinate_frame","geometry_fat","geometry_skin",
      "source330_labels","source330_mass","source330_nodes","source330_source","source330_tets","state"};
    if(!objectHasExactKeys(inputSHA,inputNames)||!objectHasExactKeys(geometryInputs,inputNames)) {
        error="geometry envelope input digest map has an unexpected key set";return false;
    }
    constexpr std::array<std::pair<std::string_view,std::string_view>,13> liveInputEnv{{
      {"candidate","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_OPERATIONAL_CANDIDATE_VOLUME_SHA256"},
      {"child_element","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS_SHA256"},
      {"child_labels","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_CHILD_LABELS_SHA256"},
      {"child_node","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE_SHA256"},
      {"coordinate_frame","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COORDINATE_FRAME_SHA256"},
      {"geometry_fat","NUMI_H706_FRACTIONAL_GEOMETRY_FAT_SHA256"},
      {"geometry_skin","NUMI_H706_FRACTIONAL_GEOMETRY_SKIN_SHA256"},
      {"source330_labels","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_PARENT_LABELS_SHA256"},
      {"source330_mass","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_PARENT_PACKET_SHA256"},
      {"source330_nodes","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_PARENT_NODES_SHA256"},
      {"source330_source","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_PARENT_SOURCE_SHA256"},
      {"source330_tets","NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_PARENT_TETS_SHA256"},
      {"state","NUMI_H706_FRACTIONAL_STATE_SHA256"}}};
    for(const auto& [name,envKey]:liveInputEnv) {
        std::string candidateSHA,geometrySHA,liveSHA;
        if(!str(inputSHA,name,candidateSHA)||!str(geometryInputs,name,geometrySHA)||
           !validSHA256(candidateSHA)||candidateSHA!=geometrySHA||!envSHA(std::string(envKey),liveSHA,error)||
           liveSHA!=geometrySHA) {
            if(error.empty())error="geometry envelope input SHA differs from live import configuration: "+std::string(name);
            return false;
        }
    }
    std::string configuredText;double configuredLimit=0.0;
    if(!env("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M",
            configuredText,error)||!parseFinitePositiveDistanceDecimal(configuredText,configuredLimit)) {
        error="configured recorded-volume transport envelope is missing or malformed";return false;
    }
    if(!validateMassTransportDistanceContract(exactDecimal,maxDeclared,envelopeLimit,
            configuredLimit,massReceiptLimitM,0.0,error))return false;
    resolvedLimitM=envelopeLimit;return true;
}
inline bool candidateHasExpectedShape(std::string_view candidate,const bool envelopeEnabled) {
    if(envelopeEnabled)return objectHasExactKeys(candidate,{"current_state","skin_volume","fat_volume",
      "child_mesh_sha256","region_label_sha256","coordinate_frame_sha256","child_node_count",
      "child_tetrahedron_count","mass_transport_envelope"});
    return objectHasExactKeys(candidate,{"current_state","skin_volume","fat_volume","child_mesh_sha256",
      "region_label_sha256","coordinate_frame_sha256","child_node_count","child_tetrahedron_count"});
}
inline bool validateLegacyTransportDistance(const double receiptLimitM,const double actualMaximumUsedM,
        double& resolvedLimitM,std::string& error) {
    constexpr double legacyLimitM=0.019353;
    if(!std::isfinite(receiptLimitM)||receiptLimitM!=legacyLimitM||
       !std::isfinite(actualMaximumUsedM)||actualMaximumUsedM<0.0||actualMaximumUsedM>legacyLimitM) {
        error="legacy recorded-flow transport bound differs from the strict 0.019353 m contract";return false;
    }
    resolvedLimitM=legacyLimitM;error.clear();return true;
}
inline bool configureMassTransportEnvelopeFromEnvironment(CurrentFractionalRecordedFlowBinding& binding,
                                                              std::string& error) {
    constexpr const char* envelopePathKey=
        "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT";
    constexpr const char* envelopeSHAKey=
        "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT_SHA256";
    constexpr const char* legacyDistanceKey=
        "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M";
    const char* envelopePath=std::getenv(envelopePathKey);
    const char* envelopeSHA=std::getenv(envelopeSHAKey);
    if((envelopePath==nullptr)!=(envelopeSHA==nullptr)) {
        error="mass transport envelope receipt path and SHA must be provided together";return false;
    }
    binding.massTransportEnvelopeEnabled=envelopePath!=nullptr;
    if(binding.massTransportEnvelopeEnabled) {
        if(!envPath(envelopePathKey,binding.massTransportEnvelopeReceiptPath,error)||
           !envSHA(envelopeSHAKey,binding.massTransportEnvelopeReceiptSHA256,error))return false;
        return true;
    }
    const char* configuredDistance=std::getenv(legacyDistanceKey);
    if(configuredDistance&&std::string(configuredDistance)!="0.019353") {
        error="legacy recorded-flow mode rejects a nonlegacy transport envelope configuration";return false;
    }
    binding.massTransportEnvelopeReceiptPath.clear();
    binding.massTransportEnvelopeReceiptSHA256.clear();
    binding.transportDistanceLimitM=0.019353;
    error.clear();return true;
}
inline bool loadRoleEnvironment(const unsigned role,CurrentFractionalRecordedFlowBinding& binding,
                                std::string& error) {
    auto& assets=binding.roleAssets[role];assets.materialIndex=role==0u?15u:16u;
    const std::string prefix="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_"+
        std::string(role==0u?"SKIN":"FAT")+"_";
    if(!envPath(prefix+"RECEIPT_JSON",assets.receiptPath,error)||
       !envSHA(prefix+"RECEIPT_SHA256",assets.receiptSHA256,error))return false;
    constexpr std::array<std::string_view,6> stems{
        "CHILD_MASS","TRANSPORT_FLOW","SOURCE_CARRIER_FLOW",
        "CHILD_SOURCE_DONOR","SOURCE_PARENT","SOURCE_PARENT_DONOR"};
    for(std::size_t i=0;i<stems.size();++i){
        const std::string key=prefix+std::string(stems[i])+"_CSV";
        if(!envPath(key,assets.csvPaths[i],error)||
           !envSHA(key+"_SHA256",assets.csvSHA256[i],error))return false;
    }
    const std::string fractional="NUMI_H706_FRACTIONAL_",roleName=role==0u?"SKIN":"FAT";
    if(!envSHA(fractional+"P1_"+roleName+"_SHA256",binding.roleP1SHA256[role],error)||
       !envSHA(fractional+"CARRIERS_"+roleName+"_SHA256",
               binding.roleCarrierPacketSHA256[role],error))return false;
    return true;
}
// Match binary64 reduction roundoff without rounding the reported extrema or
// changing material bounds. This is much narrower than the per-cell source
// feasibility tolerance; physical mass and density are not modified.
inline bool observedDensityRangeWithinBounds(double lo,double hi,double minimum,double maximum){
    if(!std::isfinite(lo)||!std::isfinite(hi)||!std::isfinite(minimum)||!std::isfinite(maximum)||
       lo<=0.0||hi<lo||minimum>maximum)return false;
    const double error=64.0*std::numeric_limits<double>::epsilon()*
        std::max({hi,std::abs(minimum),std::abs(maximum)});
    return minimum>=lo-error&&maximum<=hi+error;
}
inline bool admission(CurrentFractionalRecordedFlowBinding& binding,
                      const ConservativeSpatialMassTransportInput& input,
                      const SourceCellCarrierQuadratureInput& carrier,std::string& error) {
    std::string receipt;
    if(!readSmallPinnedJSON(binding.admissionReceiptPath,binding.admissionReceiptSHA256,receipt,error))return false;
    std::string schema,status,allocation,solver,candidate,source330,inputs,outputs,replay,claims;
    std::string tolerances,semanticRows,donorRows,density;
    std::uint64_t nativeSteps=~std::uint64_t{0},parentRows=0,childTets=0;
    if(!topLevelString(receipt,"schema",schema)||schema!="numi.human.current-fractional-semantic-mass-mapping.v1"||
       !topLevelString(receipt,"status",status)||status!="PASS_SOURCE_ONLY_REPLAY"||
       !noAdmission(receipt)||!num(receipt,"native_steps",nativeSteps)||nativeSteps!=0u||
       !topLevelString(receipt,"allocation",allocation)||allocation!="NOT_NATIVE_EXECUTED"||
       !object(receipt,"solver",solver)||!topLevelRawValue(receipt,"candidate",candidate)||
       !object(receipt,"source330",source330)||!topLevelRawValue(receipt,"input_sha256",inputs)||
       !object(receipt,"output_sha256",outputs)||!object(receipt,"replay",replay)||
       !object(receipt,"claims",claims)||!object(receipt,"closure_tolerances",tolerances)||
       !object(receipt,"semantic_group_rows",semanticRows)||!object(receipt,"source_donor_rows",donorRows)||
       !object(receipt,"density",density)||
       !objectHasExactKeys(receipt,{"schema","status","physical_admission","native_steps","allocation","claims",
         "transport_distance_limit_m","closure_tolerances","input_sha256","output_sha256","candidate","source330",
         "solver","replay","semantic_group_rows","source_donor_rows","density"})||
       !candidateHasExpectedShape(candidate,binding.massTransportEnvelopeEnabled)||
       !objectHasExactKeys(source330,{"mass_packet","parent_source_contributions","labels","nodes","tets","source_mass_kg",
         "parent_count","source_donor_contribution_count"})||
       !objectHasExactKeys(solver,{"method","status","optimality_certified","transport_receipt_sha256","mapping_path_sha256",
         "elapsed_s","transport_edge_count","augmentation_count","scale_factor","postsolve_rescaling"})||
       !objectHasExactKeys(replay,{"status","result_sha256","max_parent_mass_closure_abs_kg",
         "max_parent_measure_closure_abs_m3","max_source_donor_closure_abs_kg","max_child_fraction_sum_error",
         "max_transport_distance_used_m","flow_array_equality"})||
       !objectHasExactKeys(tolerances,{"mass_absolute_kg","mass_relative","measure_absolute_m3",
         "carrier_barycentric_sum_absolute","carrier_first_moment_m"})||
       !objectHasExactKeys(semanticRows,{"count","csv","exact_tuple_membership_checked",
         "max_group_mass_closure_abs_kg","max_shared_child_capacity_violation_kg"})||
       !objectHasExactKeys(donorRows,{"count","csv","source_donor_closure_group_count","max_closure_abs_kg"})||
       !objectHasExactKeys(density,{"bounds_by_material","by_material","violation_count"})){
        error="unified admission receipt is not the exact source-only current fractional schema";return false;
    }
    constexpr std::array<std::string_view,22> inputNames{
      "current_state","candidate_skin_volume","candidate_fat_volume","child_skin_node","child_skin_elements",
      "child_skin_labels","child_fat_node","child_fat_elements","child_fat_labels","source330_mass","source330_labels",
      "source330_nodes","source330_tets","p1_skin_support","p1_fat_support","carrier_skin_packet","carrier_fat_packet",
      "projection_evidence","projection_report","source_centroid_slots","carrier_witness_skin","carrier_witness_fat"};
    if(!objectHasExactKeys(inputs,inputNames)||
       !objectHasExactKeys(outputs,{"flow","semantic_groups","source_donors","carrier_trace"})||
       !objectHasExactKeys(claims,{"mass_mapping_pass","mass_conservation_pass","all_material_density_bounds",
         "regional_capacity_pass","transport_distance_bounds"})){
        error="unified admission receipt has unexpected claim or named-input/output fields";return false;
    }
    std::string solverMethod,solverStatus,solverReceipt,mappingSHA,replayStatus;
    std::uint64_t transportEdges=0,augmentations=0;double distanceLimit=0.0,scale=0.0,elapsed=0.0;
    double authenticatedDistanceLimitM=0.0;
    if(!receiptNumber(receipt,"transport_distance_limit_m",distanceLimit))return false;
    if(binding.massTransportEnvelopeEnabled) {
        if(!validateCurrentGeometryEnvelope(candidate,binding,distanceLimit,authenticatedDistanceLimitM,error))return false;
    } else if(!validateLegacyTransportDistance(distanceLimit,0.0,authenticatedDistanceLimitM,error))return false;
    if(!str(solver,"method",solverMethod)||solverMethod!="deterministic_dinic_feasibility"||
       !str(solver,"transport_receipt_sha256",solverReceipt)||!validSHA256(solverReceipt)||
       !str(solver,"status",solverStatus)||solverStatus!="PASS_FEASIBLE_NOT_OPTIMAL"||
       !receiptBool(solver,"optimality_certified",false)||
       !str(solver,"mapping_path_sha256",mappingSHA)||!validSHA256(mappingSHA)||
       !num(solver,"transport_edge_count",transportEdges)||transportEdges==0u||
       !num(solver,"augmentation_count",augmentations)||augmentations==0u||
       !receiptNumber(solver,"elapsed_s",elapsed)||elapsed<0.0||
       !receiptNumber(solver,"scale_factor",scale)||scale!=1.0||
       !receiptBool(solver,"postsolve_rescaling",false)||
       distanceLimit!=authenticatedDistanceLimitM||
       !str(replay,"status",replayStatus)||replayStatus!="PASS_DETERMINISTIC_REPLAY"||
       !receiptBool(replay,"flow_array_equality",true)){
        error="unified admission solver/replay/distance claims are incomplete or changed";return false;
    }
    std::string replaySHA;
    if(!str(replay,"result_sha256",replaySHA)||replaySHA!=solverReceipt||
       !requireFiniteNonnegative(tolerances,"mass_absolute_kg",1.0e-13)||
       !requireFiniteNonnegative(tolerances,"mass_relative",1.0e-10)||
       !requireFiniteNonnegative(tolerances,"measure_absolute_m3",1.0e-15)||
       !requireFiniteNonnegative(tolerances,"carrier_barycentric_sum_absolute",2.0e-12)||
       !requireFiniteNonnegative(tolerances,"carrier_first_moment_m",1.0e-11)){
        error="unified admission replay hash or closure tolerance schema changed";return false;
    }
    double tolMassAbs=0,tolMassRel=0,tolMeasure=0,tolFraction=0,tolMoment=0;
    if(!receiptNumber(tolerances,"mass_absolute_kg",tolMassAbs)||tolMassAbs!=1.0e-13||
       !receiptNumber(tolerances,"mass_relative",tolMassRel)||tolMassRel!=1.0e-10||
       !receiptNumber(tolerances,"measure_absolute_m3",tolMeasure)||tolMeasure!=1.0e-15||
       !receiptNumber(tolerances,"carrier_barycentric_sum_absolute",tolFraction)||tolFraction!=2.0e-12||
       !receiptNumber(tolerances,"carrier_first_moment_m",tolMoment)||tolMoment!=1.0e-11){
        error="unified admission closure tolerances differ from the frozen 013/006 contract";return false;
    }
    for(const auto claim:{"mass_mapping_pass","mass_conservation_pass","all_material_density_bounds",
                          "regional_capacity_pass","transport_distance_bounds"}){
        bool value=false;
        if(!boolean(claims,claim,value)||!value){
            error="unified admission claim is absent or false: "+std::string(claim);return false;
        }
    }
    std::uint64_t groupCount=0,donorCount=0,donorClosureGroups=0;
    double parentMassClosure=0.0,parentMeasureClosure=0.0,replaySourceDonorClosure=0.0,
        donorRowMaxClosure=0.0,childFractionError=0.0,maxUsedDistance=0.0,
        groupMassClosure=0.0,sharedChildViolation=0.0;
    if(!num(semanticRows,"count",groupCount)||groupCount==0u||
       !receiptBool(semanticRows,"exact_tuple_membership_checked",true)||
       !receiptNumber(replay,"max_parent_mass_closure_abs_kg",parentMassClosure)||
       !receiptNumber(replay,"max_parent_measure_closure_abs_m3",parentMeasureClosure)||
       !receiptNumber(replay,"max_source_donor_closure_abs_kg",replaySourceDonorClosure)||
       !receiptNumber(replay,"max_child_fraction_sum_error",childFractionError)||
       !receiptNumber(replay,"max_transport_distance_used_m",maxUsedDistance)||
       !receiptNumber(semanticRows,"max_group_mass_closure_abs_kg",groupMassClosure)||
       !receiptNumber(semanticRows,"max_shared_child_capacity_violation_kg",sharedChildViolation)||
       !num(donorRows,"count",donorCount)||donorCount==0u||
       !num(donorRows,"source_donor_closure_group_count",donorClosureGroups)||donorClosureGroups==0u||
       !receiptNumber(donorRows,"max_closure_abs_kg",donorRowMaxClosure)){
        error="unified admission closure/group/donor metrics are missing or malformed";return false;
    }
    std::string candidateState,candidateSkin,candidateFat,meshSHA,regionSHA,frameSHA;
    std::uint64_t candidateNodes=0;
    if(!object(candidate,"current_state",candidateState)||!object(candidate,"skin_volume",candidateSkin)||
       !object(candidate,"fat_volume",candidateFat)||
       !str(candidate,"child_mesh_sha256",meshSHA)||meshSHA!=input.childMeshSHA256||
       !str(candidate,"region_label_sha256",regionSHA)||regionSHA!=input.regionLabelSHA256||
       !str(candidate,"coordinate_frame_sha256",frameSHA)||frameSHA!=input.coordinateFrameSHA256||
       !num(candidate,"child_node_count",candidateNodes)||candidateNodes==0u||
       !num(candidate,"child_tetrahedron_count",childTets)||childTets!=input.childMaterialIndices.size()){
        error="unified admission candidate does not bind the current imported child geometry/labels";return false;
    }
    std::string statePath,skinPath,fatPath,stateSHA,skinSHA,fatSHA;
    if(!objectHasExactKeys(candidateState,{"path","sha256"})||
       !objectHasExactKeys(candidateSkin,{"path","sha256"})||!objectHasExactKeys(candidateFat,{"path","sha256"})||
       !str(candidateState,"path",statePath)||!str(candidateState,"sha256",stateSHA)||!validSHA256(stateSHA)||
       !str(candidateSkin,"path",skinPath)||!str(candidateSkin,"sha256",skinSHA)||!validSHA256(skinSHA)||
       !str(candidateFat,"path",fatPath)||!str(candidateFat,"sha256",fatSHA)||!validSHA256(fatSHA)){
        error="unified admission current-state or volume path pin is malformed";return false;
    }
    std::string expectedStateSHA;
    if(!envSHA("NUMI_H706_FRACTIONAL_STATE_SHA256",expectedStateSHA,error)||stateSHA!=expectedStateSHA){
        error="unified admission state differs from the current registered-volume input";return false;
    }
    std::string sourceMass,sourceContributions,sourceLabels,sourceNodes,sourceTets;
    if(!object(source330,"mass_packet",sourceMass)||!object(source330,"parent_source_contributions",sourceContributions)||
       !object(source330,"labels",sourceLabels)||!object(source330,"nodes",sourceNodes)||!object(source330,"tets",sourceTets)){
        error="unified admission source330 input pins are incomplete";return false;
    }
    std::string massSHA,contribSHA,labelSHA,nodeSHA,tetSHA;
    std::string massPath,contribPath,labelPath,nodePath,tetPath;
    if(!objectHasExactKeys(sourceMass,{"path","sha256"})||
       !objectHasExactKeys(sourceContributions,{"path","sha256"})||
       !objectHasExactKeys(sourceLabels,{"path","sha256"})||
       !objectHasExactKeys(sourceNodes,{"path","sha256"})||!objectHasExactKeys(sourceTets,{"path","sha256"})||
       !str(sourceMass,"path",massPath)||!str(sourceMass,"sha256",massSHA)||
       !str(sourceContributions,"path",contribPath)||!str(sourceContributions,"sha256",contribSHA)||
       !str(sourceLabels,"path",labelPath)||!str(sourceLabels,"sha256",labelSHA)||
       !str(sourceNodes,"path",nodePath)||!str(sourceNodes,"sha256",nodeSHA)||
       !str(sourceTets,"path",tetPath)||!str(sourceTets,"sha256",tetSHA)||
       !validSHA256(massSHA)||!validSHA256(contribSHA)||!validSHA256(labelSHA)||
       !validSHA256(nodeSHA)||!validSHA256(tetSHA)||
       contribPath!=massPath+".source.csv"||
       massSHA!=input.parentMassPacketSHA256||contribSHA!=input.parentSourceContributionSHA256||
       labelSHA!=binding.source330LabelSHA256||nodeSHA!=carrier.sourceNodeTableSHA256||
       tetSHA!=carrier.sourceTetTableSHA256||
       !num(source330,"parent_count",parentRows)||parentRows!=input.parentMassKg.size()){
        error="unified admission source330 packet/label/node/tet pins differ from current owner input";return false;
    }
    double sourceMassKg=0.0;std::uint64_t sourceDonorContributionCount=0;
    if(!receiptNumber(source330,"source_mass_kg",sourceMassKg)||sourceMassKg<=0.0||
       !num(source330,"source_donor_contribution_count",sourceDonorContributionCount)||
       sourceDonorContributionCount==0u||
       !verifyFilePin(massPath,massSHA,error)||!verifyFilePin(contribPath,contribSHA,error)||
       !verifyFilePin(labelPath,labelSHA,error)||!verifyFilePin(nodePath,nodeSHA,error)||
       !verifyFilePin(tetPath,tetSHA,error)){
        error="unified admission source330 totals or file hashes are invalid";return false;
    }
    const double massClosureLimit=1.0e-13+1.0e-10*sourceMassKg;
    if(parentMassClosure>massClosureLimit||replaySourceDonorClosure>massClosureLimit||
       donorRowMaxClosure>massClosureLimit||groupMassClosure>massClosureLimit||
       sharedChildViolation>massClosureLimit||parentMeasureClosure>1.0e-15||
       childFractionError>2.0e-12||maxUsedDistance<0.0||maxUsedDistance>distanceLimit){
        error="unified admission replay closure or transport bound failed";return false;
    }

    constexpr std::array<std::string_view,22> names{
      "current_state","candidate_skin_volume","candidate_fat_volume",
      "child_skin_node","child_skin_elements","child_skin_labels",
      "child_fat_node","child_fat_elements","child_fat_labels",
      "source330_mass","source330_labels","source330_nodes","source330_tets",
      "p1_skin_support","p1_fat_support","carrier_skin_packet","carrier_fat_packet",
      "projection_evidence","projection_report","source_centroid_slots",
      "carrier_witness_skin","carrier_witness_fat"};
    std::map<std::string,std::pair<std::string,std::string>> pins;
    for(const auto keyView:names){
        const std::string key(keyView);std::filesystem::path path;std::string sha;
        if(!readPin(inputs,key,path,sha,error))return false;
        pins.emplace(key,std::make_pair(path.string(),sha));
    }
    if(pins["current_state"].first!=statePath||pins["current_state"].second!=stateSHA||
       pins["candidate_skin_volume"].first!=skinPath||pins["candidate_skin_volume"].second!=skinSHA||
       pins["candidate_fat_volume"].first!=fatPath||pins["candidate_fat_volume"].second!=fatSHA||
       pins["source330_mass"].first!=massPath||pins["source330_mass"].second!=massSHA||
       pins["source330_labels"].first!=labelPath||pins["source330_labels"].second!=labelSHA||
       pins["source330_nodes"].first!=nodePath||pins["source330_nodes"].second!=nodeSHA||
       pins["source330_tets"].first!=tetPath||pins["source330_tets"].second!=tetSHA||
       pins["p1_skin_support"].second!=binding.roleP1SHA256[0]||
       pins["p1_fat_support"].second!=binding.roleP1SHA256[1]||
       pins["carrier_skin_packet"].second!=binding.roleCarrierPacketSHA256[0]||
       pins["carrier_fat_packet"].second!=binding.roleCarrierPacketSHA256[1]){
        error="unified admission named inputs differ from loaded current/source/support/carrier pins";return false;
    }
    std::map<std::string,std::pair<std::string,std::string>> outputPins;
    for(const auto key:{"flow","semantic_groups","source_donors","carrier_trace"}){
        std::filesystem::path path;std::string sha;
        if(!readPin(outputs,key,path,sha,error))return false;
        outputPins.emplace(key,std::make_pair(path.string(),sha));
    }
    auto outputMatches=[&](std::string_view rowObject,const std::string& outputKey){
        std::string csv,path,sha;
        return object(rowObject,"csv",csv)&&objectHasExactKeys(csv,{"path","sha256"})&&
            str(csv,"path",path)&&str(csv,"sha256",sha)&&outputPins.contains(outputKey)&&
            outputPins.at(outputKey).first==path&&outputPins.at(outputKey).second==sha;
    };
    if(!outputMatches(semanticRows,"semantic_groups")||!outputMatches(donorRows,"source_donors")){
        error="unified admission semantic/donor tables do not match their pinned output files";return false;
    }
    std::string densityBounds,densityByMaterial,skinBounds,fatBounds,skinObserved,fatObserved;
    std::uint64_t densityViolations=~std::uint64_t{0};
    if(!object(density,"bounds_by_material",densityBounds)||!object(density,"by_material",densityByMaterial)||
       !objectHasExactKeys(densityBounds,{"15","16"})||!objectHasExactKeys(densityByMaterial,{"15","16"})||
       !num(density,"violation_count",densityViolations)||densityViolations!=0u||
       !object(densityBounds,"15",skinBounds)||!object(densityBounds,"16",fatBounds)||
       !object(densityByMaterial,"15",skinObserved)||!object(densityByMaterial,"16",fatObserved)||
       !objectHasExactKeys(skinBounds,{"lower_kgm3","upper_kgm3"})||
       !objectHasExactKeys(fatBounds,{"lower_kgm3","upper_kgm3"})||
       !objectHasExactKeys(skinObserved,{"min_kgm3","max_kgm3","within_bounds"})||
       !objectHasExactKeys(fatObserved,{"min_kgm3","max_kgm3","within_bounds"})){
        error="unified admission density schema or violation count is invalid";return false;
    }
    double skinLo=0,skinHi=0,fatLo=0,fatHi=0,skinMin=0,skinMax=0,fatMin=0,fatMax=0;
    if(!receiptNumber(skinBounds,"lower_kgm3",skinLo)||skinLo!=900.0||
       !receiptNumber(skinBounds,"upper_kgm3",skinHi)||skinHi!=1100.0||
       !receiptNumber(fatBounds,"lower_kgm3",fatLo)||fatLo!=850.0||
       !receiptNumber(fatBounds,"upper_kgm3",fatHi)||fatHi!=950.0||
       !receiptNumber(skinObserved,"min_kgm3",skinMin)||!receiptNumber(skinObserved,"max_kgm3",skinMax)||
       !receiptNumber(fatObserved,"min_kgm3",fatMin)||!receiptNumber(fatObserved,"max_kgm3",fatMax)||
       !receiptBool(skinObserved,"within_bounds",true)||!receiptBool(fatObserved,"within_bounds",true)||
       !observedDensityRangeWithinBounds(skinLo,skinHi,skinMin,skinMax)||
       !observedDensityRangeWithinBounds(fatLo,fatHi,fatMin,fatMax)){
        error="unified admission density ranges violate unchanged material bounds";return false;
    }
    std::string expectedSkinVolumeSHA,expectedFatVolumeSHA,expectedMassSHA;
    if(!envSHA("NUMI_H706_FRACTIONAL_VOLUME_SKIN_SHA256",expectedSkinVolumeSHA,error)||
       !envSHA("NUMI_H706_FRACTIONAL_VOLUME_FAT_SHA256",expectedFatVolumeSHA,error)||
       !envSHA("NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256",expectedMassSHA,error)){
        return false;
    }
    if(pins["candidate_skin_volume"].second!=expectedSkinVolumeSHA||
       pins["candidate_fat_volume"].second!=expectedFatVolumeSHA||
       massSHA!=expectedMassSHA){
        error="unified admission current volume or source mass differs from pinned H706 environment";return false;
    }
    for(unsigned role=0;role<2u;++role){
        const std::string prefix=role==0u?"child_skin_":"child_fat_";
        auto& expected=binding.roleAssets[role].expectedInputSHA256;
        expected={std::string(input.parentMassPacketSHA256),std::string(input.parentSourceContributionSHA256),
          std::string(input.childMeshSHA256),std::string(input.regionLabelSHA256),
          std::string(input.coordinateFrameSHA256),pins[prefix+"node"].second,
          pins[prefix+"elements"].second,pins[prefix+"labels"].second,
          binding.roleP1SHA256[role],binding.roleCarrierPacketSHA256[role]};
    }
    binding.transportReceiptSHA256=solverReceipt;
    binding.transportDistanceLimitM=authenticatedDistanceLimitM;
    binding.candidateParentRows=input.parentMassKg.size();
    binding.candidateChildRows=input.childMaterialIndices.size();
    binding.admissionVerified=true;return true;
}
inline bool loadEnvironment(CurrentFractionalRecordedFlowBinding& binding,std::string& error) {
    const std::string prefix="NUMI_H706_FRACTIONAL_RECORDED_FLOW_ADMISSION_RECEIPT_";
    if(!envPath(prefix+"JSON",binding.admissionReceiptPath,error)||
       !envSHA(prefix+"SHA256",binding.admissionReceiptSHA256,error))return false;
    if(!configureMassTransportEnvelopeFromEnvironment(binding,error))return false;
    for(unsigned role=0;role<2u;++role)if(!loadRoleEnvironment(role,binding,error))return false;
    binding.environmentLoaded=true;return true;
}
} // namespace current_fractional_recorded_flow_binding_detail

[[nodiscard]] inline bool loadCurrentFractionalRecordedFlowBindingFromEnvironment(
        CurrentFractionalRecordedFlowBinding& binding,std::string& error) {
    return current_fractional_recorded_flow_binding_detail::loadEnvironment(binding,error);
}
[[nodiscard]] inline bool loadCurrentFractionalRecordedRoleFlows(
        void* opaque,const ConservativeSpatialMassTransportInput& input,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        std::array<PhysicalSkinSourceCarrierRoleRecordedAllocationData,2>& roles,
        std::string& error) {
    using namespace current_fractional_recorded_flow_binding_detail;
    auto* binding=static_cast<CurrentFractionalRecordedFlowBinding*>(opaque);
    if(!binding||!binding->environmentLoaded){error="recorded-flow environment binding is not initialized";return false;}
    static_cast<void>(support);
    if(!admission(*binding,input,carrier,error))return false;
    for(unsigned role=0;role<2u;++role){
        const std::uint32_t material=role==0u?15u:16u;
        const auto parentCount=static_cast<std::size_t>(std::count(
            input.parentMaterialIndices.begin(),input.parentMaterialIndices.end(),material));
        const auto childCount=static_cast<std::size_t>(std::count(
            input.childMaterialIndices.begin(),input.childMaterialIndices.end(),material));
        PhysicalSkinRecordedCarrierAllocationCSVImportReport report;
        if(!loadPhysicalSkinRecordedCarrierAllocationCSVs(binding->roleAssets[role],material,
                parentCount,childCount,binding->transportDistanceLimitM,roles[role],report,error))return false;
        if(!report.verified||report.physicalAdmission||report.nativeSteps!=0u||
           !report.solverFreeFeasibility||report.deterministicOptimizerReplay||
           report.transportReceiptSHA256!=binding->transportReceiptSHA256||
           report.sourceCarrierPacketSHA256!=binding->roleCarrierPacketSHA256[role]){
            error="role-local recorded receipt disagrees with unified admission or claims physical/native status";return false;
        }
        binding->roleParentRows[role]=parentCount;binding->roleChildRows[role]=childCount;
        binding->roleReports[role]=std::move(report);
    }
    std::size_t parentTotal=0u,childTotal=0u;
    for(unsigned role=0;role<2u;++role){parentTotal+=binding->roleParentRows[role];childTotal+=binding->roleChildRows[role];}
    if(parentTotal!=binding->candidateParentRows||childTotal!=binding->candidateChildRows){
        error="paired role receipts do not partition current parent and child material domains";return false;
    }
    binding->rolesVerified=true;error.clear();return true;
}
} // namespace metalrobo::human::coarse_tissue
