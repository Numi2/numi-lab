#include "NumiHumanCurrentFractionalRecordedFlowBinding.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

using namespace metalrobo::human::coarse_tissue;
using namespace metalrobo::human::coarse_tissue::current_fractional_recorded_flow_binding_detail;

namespace {
constexpr std::array<std::pair<const char*,const char*>,13> kInputs{{
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

void set(const char* key,const std::string& value) {
    if(setenv(key,value.c_str(),1)!=0)std::abort();
}
std::string syntheticInputMap() {
    constexpr char hex[]="0123456789abcdef";
    std::ostringstream out;out<<'{';
    for(std::size_t i=0;i<kInputs.size();++i) {
        if(i)out<<',';
        out<<'"'<<kInputs[i].first<<"\":\""<<std::string(64u,hex[i%16u])<<'"';
    }
    out<<'}';return out.str();
}
std::string candidateJSON(const std::string& path,const std::string& sha,
                          const std::string& inputMap,
                          const std::string& exactDecimal,
                          const std::string& maxDeclared,
                          const std::string& cap) {
    std::ostringstream out;
    out<<"{\"mass_transport_envelope\":{\"receipt\":{\"path\":\""<<path
       <<"\",\"sha256\":\""<<sha<<"\"},\"schema\":\"numi.human.mass-transport-envelope.v1\","
       <<"\"rule_id\":\"ceil_max_declared_source_centroid_um_plus_1_floor_200\","
       <<"\"maximum_declared_source_centroid_distance_m\":"<<maxDeclared<<","
       <<"\"predeclared_transport_envelope_m\":"<<cap<<",\"exact_decimal\":\""<<exactDecimal
       <<"\",\"input_sha256\":"<<inputMap<<"}}";
    return out.str();
}
std::string syntheticGeometryReceipt(const std::string& inputMap,
                                     const std::string& exactDecimal,
                                     const std::string& maximumDeclared,
                                     const std::string& cap) {
    std::ostringstream out;
    out<<"{\"schema\":\"numi-human.registered-volume-parent-identity-source-control.v1\","
       <<"\"physical_admission\":false,\"native_steps\":0,"
       <<"\"mass_mapping_maximum_transport_distance_m\":"<<cap<<","
       <<"\"mass_transport_envelope\":{"
       <<"\"schema\":\"numi.human.mass-transport-envelope.v1\","
       <<"\"status\":\"PREDECLARED_CURRENT_GEOMETRY_ENVELOPE\","
       <<"\"rule_id\":\"ceil_max_declared_source_centroid_um_plus_1_floor_200\","
       <<"\"maximum_declared_source_centroid_distance_m\":"<<maximumDeclared<<","
       <<"\"predeclared_transport_envelope_m\":"<<cap<<","
       <<"\"exact_decimal\":\""<<exactDecimal<<"\","
       <<"\"input_sha256\":"<<inputMap<<"}}";
    return out.str();
}
bool writeReceipt(const std::filesystem::path& path,const std::string& json,
                  std::string& sha,std::string& error) {
    std::ofstream out(path,std::ios::binary);
    if(!out){error="cannot create synthetic envelope receipt";return false;}
    out<<json;out.close();
    if(!out){error="cannot write synthetic envelope receipt";return false;}
    return sha256File(path,sha,error);
}
std::string numberText(const double value) {
    std::ostringstream out;out<<std::setprecision(17)<<value;return out.str();
}
bool runOptionalPinnedReceipt(const std::filesystem::path& path,const std::string& sha,
                              std::string& error) {
    std::string geometry,envelope,inputs,exactDecimal;
    if(!readSmallPinnedJSON(path,sha,geometry,error)||
       !object(geometry,"mass_transport_envelope",envelope)||
       !object(envelope,"input_sha256",inputs)||
       !str(envelope,"exact_decimal",exactDecimal))return false;
    double maxDeclared=0.0,cap=0.0;
    if(!receiptNumber(envelope,"maximum_declared_source_centroid_distance_m",maxDeclared)||
       !receiptNumber(envelope,"predeclared_transport_envelope_m",cap))return false;
    for(const auto& [name,key]:kInputs) {
        std::string value;
        if(!str(inputs,name,value))return false;
        set(key,value);
    }
    const std::string pathText=path.string();
    const char* pathKey="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT";
    const char* shaKey="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT_SHA256";
    set(pathKey,pathText);set(shaKey,sha);
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M",exactDecimal);
    CurrentFractionalRecordedFlowBinding binding;
    binding.massTransportEnvelopeReceiptPath=path;
    binding.massTransportEnvelopeReceiptSHA256=sha;
    binding.massTransportEnvelopeEnabled=true;
    double resolved=0.0;
    return validateCurrentGeometryEnvelope(
        candidateJSON(pathText,sha,inputs,exactDecimal,numberText(maxDeclared),numberText(cap)),
        binding,cap,resolved,error)&&resolved==cap;
}
bool run(bool condition,const char* name,int& failures) {
    std::cout<<(condition?"PASS ":"FAIL ")<<name<<"\n";
    if(!condition)++failures;
    return condition;
}
}

int main(int argc,char** argv) {
    int failures=0;std::string error;
    const auto nonce=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto fixtureDir=std::filesystem::temp_directory_path()/
        ("numi-current-fractional-envelope-"+std::to_string(nonce));
    std::error_code ec;
    if(!std::filesystem::create_directories(fixtureDir,ec)||ec) {
        std::cerr<<"cannot create temporary fixture directory\n";return 2;
    }
    const std::string inputs=syntheticInputMap();
    const auto geometryFile=fixtureDir/"geometry-envelope.json";
    std::string geometrySHA;
    if(!writeReceipt(geometryFile,syntheticGeometryReceipt(
          inputs,"0.187444","0.1874423157229725","0.187444"),geometrySHA,error)) {
        std::cerr<<"cannot build synthetic geometry envelope: "<<error<<"\n";
        std::filesystem::remove_all(fixtureDir,ec);return 2;
    }
    const std::string geometryPath=geometryFile.string();
    for(const auto& [name,key]:kInputs) {
        std::string value;
        if(!str(inputs,name,value)){std::cerr<<"missing synthetic input "<<name<<"\n";return 2;}
        set(key,value);
    }
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT",geometryPath);
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT_SHA256",geometrySHA);
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M","0.187444");
    CurrentFractionalRecordedFlowBinding binding;
    binding.massTransportEnvelopeReceiptPath=geometryFile;
    binding.massTransportEnvelopeReceiptSHA256=geometrySHA;
    binding.massTransportEnvelopeEnabled=true;
    double resolved=0.0;
    const auto validCandidate=candidateJSON(geometryPath,geometrySHA,inputs,
        "0.187444","0.1874423157229725","0.187444");
    double serializedBound=0.0;
    run(envelopeDeclaredDistance(R"({"predeclared_transport_envelope_m":"0.187444"})","0.187444",serializedBound)&&serializedBound==0.187444,
        "exact decimal string envelope from Python reader",failures);
    run(!envelopeDeclaredDistance(R"({"predeclared_transport_envelope_m":"0.2"})","0.187444",serializedBound),
        "changed string envelope rejected",failures);
    run(!envelopeDeclaredDistance(R"({"predeclared_transport_envelope_m":"nan"})","nan",serializedBound),
        "nonfinite string envelope rejected",failures);
    const bool currentReceiptPass=validateCurrentGeometryEnvelope(validCandidate,binding,0.187444,resolved,error)&&resolved==0.187444;
    run(currentReceiptPass,"synthetic pinned geometry receipt resolves authenticated cap",failures);
    if(!currentReceiptPass)std::cerr<<"synthetic receipt failure: "<<error<<"\n";
    const auto alternateFile=fixtureDir/"alternate-envelope-receipt.json";
    const std::string alternatePath=alternateFile.string();
    const std::string alternateJSON=syntheticGeometryReceipt(inputs,"0.250001","0.25","0.250001");
    std::string alternateSHA;
    const bool alternateHash=writeReceipt(alternateFile,alternateJSON,alternateSHA,error);
    CurrentFractionalRecordedFlowBinding alternateBinding=binding;
    alternateBinding.massTransportEnvelopeReceiptPath=alternateFile;
    alternateBinding.massTransportEnvelopeReceiptSHA256=alternateSHA;
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT",alternatePath);
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT_SHA256",alternateSHA);
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M","0.250001");
    const bool alternateReceiptPass=alternateHash&&validateCurrentGeometryEnvelope(
        candidateJSON(alternatePath,alternateSHA,inputs,"0.250001","0.25","0.250001"),
        alternateBinding,0.250001,resolved,error)&&resolved==0.250001;
    run(alternateReceiptPass,"alternate SHA-pinned geometry envelope receipt and boundary pass",failures);
    if(!alternateReceiptPass)std::cerr<<"alternate receipt failure: "<<error<<"\n";
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT",geometryPath);
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT_SHA256",geometrySHA);
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M","0.187444");
    const std::string duplicateScopeFixture=
      "{\"input_sha256\":"+inputs+",\"candidate\":{"
      "\"mass_transport_envelope\":{\"input_sha256\":"+inputs+"}}}";
    std::string outerInputMap,candidateObject,nestedEnvelope,nestedInputMap;
    const bool rootInputScopePass=topLevelRawValue(duplicateScopeFixture,"input_sha256",outerInputMap)&&
      outerInputMap==inputs&&topLevelRawValue(duplicateScopeFixture,"candidate",candidateObject)&&
      object(candidateObject,"mass_transport_envelope",nestedEnvelope)&&
      object(nestedEnvelope,"input_sha256",nestedInputMap)&&nestedInputMap==inputs;
    run(rootInputScopePass,"root and nested 13-key input_sha256 maps are read at their own scopes",failures);
    const std::string duplicateRootInput=
      "{\"input_sha256\":{},\"input_sha256\":{},\"candidate\":{}}";
    run(!topLevelRawValue(duplicateRootInput,"input_sha256",outerInputMap),
        "duplicate top-level input_sha256 rejects",failures);
    const std::string legacyCandidate=
      "{\"current_state\":{},\"skin_volume\":{},\"fat_volume\":{},"
      "\"child_mesh_sha256\":\"x\",\"region_label_sha256\":\"x\","
      "\"coordinate_frame_sha256\":\"x\",\"child_node_count\":1,"
      "\"child_tetrahedron_count\":1}";
    run(candidateHasExpectedShape(legacyCandidate,false),"legacy candidate shape remains exact",failures);
    run(!candidateHasExpectedShape(legacyCandidate,true),"legacy candidate rejected in envelope mode",failures);
    run(!candidateHasExpectedShape(validCandidate,false),"envelope candidate rejected in legacy mode",failures);
    double legacyResolved=0.0;
    run(validateLegacyTransportDistance(0.019353,0.0,legacyResolved,error)&&legacyResolved==0.019353,
        "legacy strict 0.019353 m cap remains supported",failures);
    run(validWithinTransportDistance(0.019353,0.019353),
        "legacy CSV row at strict 0.019353 m cap passes",failures);
    run(!validWithinTransportDistance(0.0193531,0.019353),
        "legacy CSV row above strict 0.019353 m cap rejects",failures);
    run(!validateLegacyTransportDistance(0.019354,0.0,legacyResolved,error),
        "legacy mode rejects any other cap",failures);
    run(!validateLegacyTransportDistance(0.019353,0.019354,legacyResolved,error),
        "legacy mode rejects a recorded distance above its cap",failures);
    const char* envelopePathKey="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT";
    const char* envelopeSHAKey="NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MASS_TRANSPORT_ENVELOPE_RECEIPT_SHA256";
    unsetenv(envelopePathKey);unsetenv(envelopeSHAKey);unsetenv(
      "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M");
    CurrentFractionalRecordedFlowBinding legacyBinding;
    const bool legacyEnvPass=configureMassTransportEnvelopeFromEnvironment(legacyBinding,error)&&
       !legacyBinding.massTransportEnvelopeEnabled&&legacyBinding.transportDistanceLimitM==0.019353;
    run(legacyEnvPass,"both envelope vars absent selects strict legacy cap",failures);
    set(envelopePathKey,geometryPath);
    run(!configureMassTransportEnvelopeFromEnvironment(legacyBinding,error),
        "half-present envelope environment rejects",failures);
    set(envelopeSHAKey,geometrySHA);
    run(configureMassTransportEnvelopeFromEnvironment(legacyBinding,error)&&
        legacyBinding.massTransportEnvelopeEnabled&&legacyBinding.massTransportEnvelopeReceiptSHA256==geometrySHA,
        "both envelope vars present selects pinned dynamic mode",failures);
    unsetenv(envelopePathKey);unsetenv(envelopeSHAKey);
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M","0.187444");
    run(!configureMassTransportEnvelopeFromEnvironment(legacyBinding,error),
        "legacy mode rejects dynamic cap without receipt",failures);
    for(const auto& [name,key]:kInputs) {
        std::string value;str(inputs,name,value);set(key,value);
    }
    run(validateMassTransportDistanceContract("0.250001",0.25,0.250001,0.250001,0.250001,0.250001,error),
        "alternate authenticated envelope value and exact boundary pass",failures);
    run(!validateMassTransportDistanceContract("0.250001",0.25,0.250001,0.250002,0.250001,0.25,error),
        "stage import cap mismatch rejects",failures);
    run(!validateMassTransportDistanceContract("0.250001",0.25,0.250001,0.250001,0.250002,0.25,error),
        "mass receipt cap mismatch rejects",failures);
    run(!validateMassTransportDistanceContract("0.250001",0.25,0.250001,
        std::numeric_limits<double>::quiet_NaN(),0.250001,0.25,error),
        "NaN configured cap rejects",failures);
    run(!validateMassTransportDistanceContract("0.250001",0.25,
        std::numeric_limits<double>::infinity(),0.250001,0.250001,0.25,error),
        "infinite envelope cap rejects",failures);
    run(!validateMassTransportDistanceContract("0",0.25,0.250001,0.250001,0.250001,0.25,error),
        "nonpositive exact decimal rejects",failures);
    run(!validateMassTransportDistanceContract("0.250001",0.25,0.250001,0.250001,0.250001,0.250002,error),
        "used distance above cap rejects",failures);
    run(validWithinTransportDistance(0.187444,0.187444),
        "CSV row distance exactly at envelope passes",failures);
    run(!validWithinTransportDistance(0.1874441,0.187444),
        "CSV row distance above envelope rejects",failures);
    run(!validWithinTransportDistance(std::numeric_limits<double>::quiet_NaN(),0.187444),
        "CSV row NaN distance rejects",failures);
    run(!validWithinTransportDistance(0.0,0.0),
        "CSV row rejects nonpositive configured envelope",failures);
    std::string changedInputs=inputs;
    const auto first=changedInputs.find("\"candidate\"");
    const auto valueBegin=changedInputs.find('"',changedInputs.find(':',first)+1);
    const auto valueEnd=changedInputs.find('"',valueBegin+1);
    if(first==std::string::npos||valueBegin==std::string::npos||valueEnd==std::string::npos)return 2;
    changedInputs.replace(valueBegin+1,valueEnd-valueBegin-1,64,'f');
    run(!validateCurrentGeometryEnvelope(candidateJSON(geometryPath,geometrySHA,changedInputs,
        "0.187444","0.1874423157229725","0.187444"),binding,0.187444,resolved,error),
        "stale geometry input hash rejects",failures);
    set(kInputs[1].second,std::string(64,'0'));
    run(!validateCurrentGeometryEnvelope(validCandidate,binding,0.187444,resolved,error),
        "live stage input SHA mismatch rejects",failures);
    set(kInputs[1].second,[] (const std::string& json) {
        std::string geomEnv,inputMap,digest;
        object(json,"mass_transport_envelope",geomEnv);object(geomEnv,"input_sha256",inputMap);
        str(inputMap,"child_element",digest);return digest;
    }(syntheticGeometryReceipt(inputs,"0.187444","0.1874423157229725","0.187444")));
    set("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M","0.187443");
    run(!validateCurrentGeometryEnvelope(validCandidate,binding,0.187444,resolved,error),
        "runtime import configuration cap mismatch rejects",failures);
    if(argc==3) {
        std::string realError;
        const bool realReceiptPass=runOptionalPinnedReceipt(argv[1],argv[2],realError);
        run(realReceiptPass,"optional real pinned geometry envelope receipt integration",failures);
        if(!realReceiptPass)std::cerr<<"optional real receipt failure: "<<realError<<"\n";
    } else if(argc!=1) {
        std::cerr<<"usage: CurrentFractionalMassTransportEnvelopeCheck [receipt.json sha256]\n";
        std::filesystem::remove_all(fixtureDir,ec);return 2;
    }
    run(observedDensityRangeWithinBounds(900,1100,899.99999999999932,1100.0000000000005),
        "binary64 Skin extrema accepted without editing their values",failures);
    run(observedDensityRangeWithinBounds(850,950,849.99999999999977,950.00000000000034),
        "binary64 Fat extrema accepted without editing their values",failures);
    run(!observedDensityRangeWithinBounds(900,1100,899.99,1100),"real density deficit rejects",failures);
    run(!observedDensityRangeWithinBounds(900,1100,900,1100.01),"real density excess rejects",failures);
    run(!observedDensityRangeWithinBounds(900,1100,1000,999),"reversed density range rejects",failures);
    run(!observedDensityRangeWithinBounds(900,1100,std::numeric_limits<double>::quiet_NaN(),1000),"nonfinite density rejects",failures);
    std::cout<<"SUMMARY failures="<<failures<<"\n";
    std::filesystem::remove_all(fixtureDir,ec);
    return failures==0?0:1;
}
