#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include "numi/matter/matter.hpp"
#include "numi/matter/compiler.hpp"
#include "numi/matter/detail.hpp"
#include "numi/matter/shared.h"
#include <algorithm>
#include <numeric>
#include <array>
#include <bit>
#include <cmath>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {
using namespace numi::matter;
constexpr std::uint32_t kInputStride=47u, kOutputStride=65u, kStateCount=15u;
constexpr std::uint32_t kTraceStateCapacity=kStateCount+1u;
constexpr std::uint32_t kTraceIterationCapacity=32u;
constexpr std::uint32_t kTraceSummarySlots=73u;
constexpr std::uint32_t kTraceHeaderFailureSlot=7u;
static_assert(NM_MAX_MATERIAL_STATE>=kTraceStateCapacity);
constexpr std::uint32_t kTraceRecordSlots=6u+7u*kTraceStateCapacity+
    kTraceStateCapacity*kTraceStateCapacity;
constexpr std::uint32_t kTraceSlots=8u+kTraceIterationCapacity*kTraceRecordSlots;
static_assert(kTraceSlots==11976u);
constexpr char kCaptureSchema[]="numi.cardboard.fem-newton-capture.v1";
constexpr char kLinerSHA[]="a2720b90f2004f2eccf180063e7ecc4e0acdda93f700c4fd357914fc24d2255a";
constexpr char kMediumSHA[]="76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9";
constexpr char kLinerName[]="hajali2009_liner_hill_bilinear_fit";
constexpr char kMediumName[]="hajali2009_medium_hill_bilinear_fit";
#ifndef NUMI_CARDBOARD_BASE_SNAPSHOT_SHA256
#error "The checker must be bound to an exact frozen source manifest"
#endif
constexpr char kBaseSnapshotSHA[]=NUMI_CARDBOARD_BASE_SNAPSHOT_SHA256;

void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::string sha256(const std::filesystem::path& path){
    std::ifstream in(path,std::ios::binary);need(in.good(),"cannot read for SHA256: "+path.string());
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),{});
    need(bytes.size()<=static_cast<std::size_t>(UINT32_MAX),"SHA256 input exceeds CommonCrypto bound");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    need(CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),digest)!=nullptr,"SHA256 failed");
    static constexpr char hex[]="0123456789abcdef";std::string out;out.reserve(64u);
    for(unsigned char b:digest){out.push_back(hex[b>>4]);out.push_back(hex[b&15]);}return out;
}
NSDictionary* parseObject(const std::string& line,const std::string& label){
    NSData* data=[NSData dataWithBytes:line.data() length:line.size()];NSError* error=nil;
    id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    NSDictionary* result=[value isKindOfClass:[NSDictionary class]]?(NSDictionary*)value:nil;
    need(result!=nil,label+" is not a JSON object");return result;
}
id field(NSDictionary* d,const char* key){id v=[d objectForKey:[NSString stringWithUTF8String:key]];need(v!=nil,std::string("missing field ")+key);return v;}
double number(id v,const std::string& label){
    need([v isKindOfClass:[NSNumber class]],label+" is not numeric");double x=[(NSNumber*)v doubleValue];
    need(std::isfinite(x),label+" is nonfinite");return x;
}
std::string stringValue(id v,const std::string& label){
    need([v isKindOfClass:[NSString class]],label+" is not a string");return [(NSString*)v UTF8String];
}
template<std::size_t N> std::array<float,N> flatArray(id v,const std::string& label){
    need([v isKindOfClass:[NSArray class]],label+" is not an array");NSArray* a=(NSArray*)v;
    std::array<float,N> out{};std::size_t p=0;
    for(id row in a){
        if([row isKindOfClass:[NSArray class]]){
            for(id cell in (NSArray*)row){need(p<N,label+" too long");out[p++]=static_cast<float>(number(cell,label));}
        }else{need(p<N,label+" too long");out[p++]=static_cast<float>(number(row,label));}
    }
    need(p==N,label+" has wrong flattened length");return out;
}
std::vector<float> flatVector(id v,const std::size_t expected,const std::string& label){
    need([v isKindOfClass:[NSArray class]],label+" is not an array");NSArray* a=(NSArray*)v;
    need(a.count==expected,label+" has wrong length");std::vector<float> out;out.reserve(expected);
    for(id x in a)out.push_back(static_cast<float>(number(x,label)));return out;
}
struct Record{
    std::uint32_t env=0,tet=0,material=0;
    std::string materialName;
    std::array<float,kInputStride> input{};
    std::array<float,kStateCount> accepted{};
};
struct Capture{
    NSDictionary* header=nil;
    std::vector<Record> rows;
    std::vector<float> parameterRow;
};
Capture readCapture(const std::filesystem::path& path){
    std::ifstream in(path);need(in.good(),"cannot open capture JSONL");
    Capture result;std::string line;bool first=true;std::set<std::pair<unsigned,unsigned>> unique;
    std::array<std::size_t,2> materialCounts{};
    while(std::getline(in,line)){
        if(line.empty())continue;
        NSDictionary* d=parseObject(line,first?"capture header":"capture row");
        if(first){
            result.header=d;first=false;
            NSArray* mats=(NSArray*)field(d,"materials");
            need(mats.count>=2u,"capture header omits paper material bindings");
            for(std::size_t i=0;i<2u;++i){
                NSDictionary* md=(NSDictionary*)mats[i];
                need(stringValue(field(md,"name"),"header material name")== (i==0u?kLinerName:kMediumName),
                     "capture header paper material order differs");
                need(number(field(md,"index"),"header material index")==double(i)&&
                     number(field(md,"parameter_offset"),"header parameter offset")==double(28u*i),
                     "capture header material parameter binding differs");
            }
            need(stringValue(field(d,"schema"),"capture schema")==kCaptureSchema,"unexpected capture schema");
            need(number(field(d,"root"),"capture root")==0.0&&number(field(d,"newton_iteration"),"capture Newton") == 0.0,
                 "capture is not root0/Newton0");
            need(number(field(d,"environment_count"),"environment count")==2.0&&
                 number(field(d,"tetrahedron_count_per_environment"),"tet count")==1920.0,
                 "capture dimensions differ from pinned 2x1920 source");
            need(stringValue(field(d,"center_residual_source"),"residual source")=="pre_fgmres_private_copy",
                 "capture residual source is not the production pre-FGMRES copy");
            need(stringValue(field(d,"F_local_word_encoding"),"F encoding").find("high_float32_plus_low_float32")!=std::string::npos,
                 "capture does not bind compensated F input words");
            NSArray* envParams=(NSArray*)field(d,"parameters_by_environment");
            need(envParams.count==2u,"capture parameter environment count mismatch");
            auto p0=flatVector(envParams[0],59u,"env0 parameters");
            auto p1=flatVector(envParams[1],59u,"env1 parameters");
            for(std::size_t i=0;i<p0.size();++i)
                need(std::bit_cast<std::uint32_t>(p0[i])==std::bit_cast<std::uint32_t>(p1[i]),
                     "capture environments have differing material parameter words");
            result.parameterRow=std::move(p0);continue;
        }
        if(stringValue(field(d,"record"),"record type")!="tetrahedron")continue;
        const auto mat=static_cast<std::uint32_t>(number(field(d,"material_index"),"material index"));
        if(mat>1u)continue; // Glue rows are explicitly outside this paper projection diagnostic.
        Record row;
        row.env=static_cast<std::uint32_t>(number(field(d,"environment"),"environment"));
        row.tet=static_cast<std::uint32_t>(number(field(d,"tet_id"),"tet id"));
        row.material=mat;row.materialName=stringValue(field(d,"material_name"),"material name");
        need(row.env<2u&&row.tet<1920u,"paper row index outside capture shape");
        need(row.materialName==(mat==0u?kLinerName:kMediumName),"material index/name mismatch in capture");
        need(number(field(d,"material_state_count"),"paper state count")==15.0&&
             number(field(d,"material_parameter_count"),"paper parameter count")==28.0,
             "captured paper row is not the exact 15-state/28-parameter source-fit system");
        need(unique.emplace(row.env,row.tet).second,"duplicate env/tet paper row");
        ++materialCounts[row.material];
        NSDictionary* c=(NSDictionary*)field(d,"center");
        auto F=flatArray<9>(field(c,"F_local"),"F_local");
        auto Flo=flatArray<9>(field(c,"F_local_low"),"F_local_low");
        auto Fdot=flatArray<9>(field(c,"Fdot_local_per_s"),"Fdot_local_per_s");
        auto dF=flatArray<9>(field(c,"F_direction_local_per_s"),"F_direction_local_per_s");
        auto dFdot=flatArray<9>(field(c,"rate_direction_local_per_s2"),"rate_direction_local_per_s2");
        auto state=flatVector(field(c,"accepted_state"),15u,"accepted state");
        std::copy(F.begin(),F.end(),row.input.begin());
        std::copy(Flo.begin(),Flo.end(),row.input.begin()+9);
        std::copy(Fdot.begin(),Fdot.end(),row.input.begin()+18);
        std::copy(dF.begin(),dF.end(),row.input.begin()+27);
        std::copy(dFdot.begin(),dFdot.end(),row.input.begin()+36);
        NSArray* dtTemp=(NSArray*)field(c,"dt_seconds_bulk_modulus_kind_frame_active");
        need(dtTemp.count==4u,"capture dt/temp metadata shape mismatch");
        row.input[45]=static_cast<float>(number(dtTemp[0],"captured dt"));
        row.input[46]=static_cast<float>(number(dtTemp[1],"captured temperature"));
        need(row.input[45]>0.0f&&row.input[46]>0.0f,"nonpositive captured dt or temperature");
        std::copy(state.begin(),state.end(),row.accepted.begin());
        result.rows.push_back(std::move(row));
    }
    need(!first,"capture file is empty");
    need(result.rows.size()==3456u,"expected exactly 3456 paper rows (glue excluded)");
    need(unique.size()==3456u&&materialCounts[0]==2304u&&materialCounts[1]==1152u,
         "paper row identity/material distribution is incomplete");
    need(std::any_of(result.rows.begin(),result.rows.end(),[](const Record& r){return r.env==0u&&r.tet==1528u&&r.material==0u;}),
         "required first-failure liner tet env0/tet1528 is missing");
    return result;
}
NSDictionary* readJSONDocument(const std::filesystem::path& path) {
    NSData* data=[NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:path.string().c_str()]];
    need(data!=nil,"cannot read bound point input JSON");
    NSError* error=nil;
    id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    NSDictionary* result=[value isKindOfClass:[NSDictionary class]]?(NSDictionary*)value:nil;
    need(result!=nil,"bound point input is not a JSON object");
    return result;
}
std::vector<std::uint32_t> bitVector(id value,const std::size_t expected,const std::string& label) {
    need([value isKindOfClass:[NSArray class]],label+" is not an array");
    std::vector<std::uint32_t> out;out.reserve(expected);
    auto append=[&](auto&& self,id x)->void {
        if([x isKindOfClass:[NSArray class]]) {
            for(id child in (NSArray*)x) self(self,child);
            return;
        }
        const auto text=stringValue(x,label);
        need(text.size()==10u&&text.rfind("0x",0u)==0u,label+" must contain eight-digit 0x words");
        std::size_t used=0u;const auto word=std::stoul(text.substr(2u),&used,16);
        need(used==8u&&word<=UINT32_MAX,label+" contains invalid word");
        out.push_back(static_cast<std::uint32_t>(word));
    };
    append(append,value);
    need(out.size()==expected,label+" has wrong flattened length");
    return out;
}
void verifyFloatWords(const float* values,const std::vector<std::uint32_t>& words,const std::string& label) {
    need(words.size()>0u,label+" is empty");
    for(std::size_t i=0u;i<words.size();++i)
        need(std::bit_cast<std::uint32_t>(values[i])==words[i],
             label+" numeric/raw word mismatch at "+std::to_string(i));
}
Record readBoundPoint(const std::filesystem::path& path,NSDictionary** bindingOut) {
    NSDictionary* root=readJSONDocument(path);
    need(stringValue(field(root,"schema"),"point schema")==
         "numi.cardboard.source380-native-material-point-input.v2",
         "unexpected source380 point input schema");
    need(number(field(root,"case_count"),"case count")==1.0,
         "source380 point input must contain one case");
    NSDictionary* binding=(NSDictionary*)field(root,"run_binding");
    need(stringValue(field(binding,"run_name"),"run name")=="explicit32-certified-coupon-380",
         "point input is not bound to source380 run");
    need(stringValue(field(binding,"source_medium_asset_sha256"),"source medium SHA")==kMediumSHA,
         "point input medium source SHA differs from this diagnostic");
    NSArray* cases=(NSArray*)field(root,"cases");
    need([cases isKindOfClass:[NSArray class]]&&cases.count==1u,
         "source380 point input case list shape mismatch");
    NSDictionary* c=(NSDictionary*)cases[0];
    Record row;
    row.env=static_cast<std::uint32_t>(number(field(c,"environment"),"environment"));
    row.tet=static_cast<std::uint32_t>(number(field(c,"tet_id"),"tet id"));
    row.material=static_cast<std::uint32_t>(number(field(c,"material_index"),"material index"));
    row.materialName=stringValue(field(c,"material_name"),"material name");
    need(row.env==0u&&row.tet==674u&&row.material==1u&&row.materialName==kMediumName,
         "source380 point identity differs from exact env0/tet674 medium case");
    const auto f=flatArray<9>(field(c,"F_high"),"F_high");
    const auto flo=flatArray<9>(field(c,"F_low"),"F_low");
    const auto fdot=flatArray<9>(field(c,"Fdot"),"Fdot");
    const auto direction=flatArray<9>(field(c,"deformation_direction"),"deformation_direction");
    const auto rateDirection=flatArray<9>(field(c,"rate_direction"),"rate_direction");
    std::copy(f.begin(),f.end(),row.input.begin());
    std::copy(flo.begin(),flo.end(),row.input.begin()+9);
    std::copy(fdot.begin(),fdot.end(),row.input.begin()+18);
    std::copy(direction.begin(),direction.end(),row.input.begin()+27);
    std::copy(rateDirection.begin(),rateDirection.end(),row.input.begin()+36);
    row.input[45]=static_cast<float>(number(field(c,"dt_seconds"),"dt seconds"));
    row.input[46]=static_cast<float>(number(field(c,"temperature_K"),"temperature K"));
    need(row.input[45]>0.0f&&row.input[46]>0.0f,
         "point input timestep/temperature is invalid");
    const auto state=flatVector(field(c,"accepted_state"),kStateCount,"accepted state");
    std::copy(state.begin(),state.end(),row.accepted.begin());
    const auto packedBits=bitVector(field(c,"packed_input_f32_bits"),kInputStride,
                                    "packed input bits");
    verifyFloatWords(row.input.data(),packedBits,"packed input");
    const auto stateBits=bitVector(field(c,"accepted_state_f32_bits"),kStateCount,
                                   "accepted-state bits");
    verifyFloatWords(row.accepted.data(),stateBits,"accepted state");
    const auto highBits=bitVector(field(c,"F_high_bits"),9u,"F-high bits");
    const auto lowBits=bitVector(field(c,"F_low_bits"),9u,"F-low bits");
    verifyFloatWords(row.input.data(),highBits,"F-high");
    verifyFloatWords(row.input.data()+9u,lowBits,"F-low");
    if(bindingOut)*bindingOut=binding;
    return row;
}
struct SourceArtifactBinding {
    std::string candidateSHA,historySHA,meshSHA;
};
std::string expectedRunArtifact(NSDictionary* binding,const std::string& relativePath) {
    NSDictionary* artifacts=(NSDictionary*)field(binding,"artifact_sha256");
    return stringValue(field(artifacts,relativePath.c_str()),"source artifact SHA");
}
void verifyAcceptedHistory(const std::filesystem::path& historyPath,const Record& row) {
    NSDictionary* history=readJSONDocument(historyPath);
    need(number(field(history,"environment"),"history environment")==double(row.env),
         "accepted history environment differs from the bound point");
    NSArray* tetrahedra=(NSArray*)field(history,"tetrahedra");
    need(tetrahedra.count>row.tet,"accepted history omits the selected tetrahedron");
    NSDictionary* selected=(NSDictionary*)tetrahedra[row.tet];
    need(number(field(selected,"index"),"history tetrahedron index")==double(row.tet)&&
         number(field(selected,"material_index"),"history material index")==double(row.material),
         "accepted history selected tetrahedron/material identity differs");
    const auto state=flatVector(field(selected,"state"),kStateCount,"accepted history state");
    for(std::size_t i=0u;i<kStateCount;++i)
        need(std::bit_cast<std::uint32_t>(static_cast<float>(state[i]))==
             std::bit_cast<std::uint32_t>(row.accepted[i]),
             "point accepted state differs from source380 rollback history at component "+
             std::to_string(i));
}
SourceArtifactBinding verifySource380Artifacts(
    NSDictionary* inputBinding,const std::filesystem::path& candidatePath,
    const std::filesystem::path& historyPath,const std::filesystem::path& meshPath,
    const Record& row) {
    const std::string candidateRelative=
        "output/rejected-conditioning-000002/candidate_nodes.bin";
    const std::string historyRelative="output/material_state_000002_bent.json";
    const std::string meshRelative="output/mesh.json";
    need(std::filesystem::path(stringValue(field(inputBinding,"run_path"),"run path")+
             "/"+candidateRelative).lexically_normal()==candidatePath.lexically_normal(),
         "candidate-node path is not the source-bound run artifact");
    need(std::filesystem::path(stringValue(field(inputBinding,"run_path"),"run path")+
             "/"+historyRelative).lexically_normal()==historyPath.lexically_normal(),
         "accepted-history path is not the source-bound run artifact");
    need(std::filesystem::path(stringValue(field(inputBinding,"run_path"),"run path")+
             "/"+meshRelative).lexically_normal()==meshPath.lexically_normal(),
         "mesh path is not the source-bound run artifact");
    SourceArtifactBinding result{sha256(candidatePath),sha256(historyPath),sha256(meshPath)};
    need(result.candidateSHA==expectedRunArtifact(inputBinding,candidateRelative)&&
         result.historySHA==expectedRunArtifact(inputBinding,historyRelative)&&
         result.meshSHA==expectedRunArtifact(inputBinding,meshRelative),
         "candidate/history/mesh artifact bytes differ from the bound source380 run");
    NSDictionary* identity=(NSDictionary*)field(inputBinding,"source_candidate_identity");
    need(result.candidateSHA==stringValue(field(identity,"candidate_nodes_sha256"),"candidate hash")&&
         result.historySHA==stringValue(field(identity,"accepted_history_sha256"),"history hash")&&
         result.meshSHA==stringValue(field(identity,"mesh_sha256"),"mesh hash"),
         "source candidate identity hashes do not match the run artifacts");
    verifyAcceptedHistory(historyPath,row);
    return result;
}
void verifyBoundPackage(const CompiledWorld& world,const Record& row,
                        NSDictionary* inputBinding,const std::string& expectedFingerprint) {
    need(std::to_string(world.fingerprint)==expectedFingerprint,
         "compiled package world fingerprint differs from explicit source380 binding");
    need(stringValue(field(inputBinding,"compiled_world_fingerprint"),"bound package fingerprint")==expectedFingerprint,
         "point input fingerprint differs from CLI binding");
    NSDictionary* descriptor=(NSDictionary*)field(inputBinding,"material_descriptor_expectation");
    need(world.materials.size()>row.material,"compiled package lacks the selected material");
    need(world.fem.tetrahedra.size()>row.tet&&
         world.fem.tetrahedra[row.tet].identity.x==row.material,
         "compiled package tetrahedron/material identity differs from selected point");
    const auto& material=world.materials[row.material];
    const auto flags=static_cast<std::uint32_t>(number(field(descriptor,"flags"),"expected flags"));
    const auto stateCount=static_cast<std::uint32_t>(number(field(descriptor,"state_count"),"expected state count"));
    const auto parameterCount=static_cast<std::uint32_t>(number(field(descriptor,"parameter_count"),"expected parameter count"));
    const auto iterations=static_cast<std::uint32_t>(number(field(descriptor,"local_newton_iterations"),"expected local iterations"));
    need(material.flags==flags&&material.stateCount==stateCount&&
         material.parameterCount==parameterCount&&material.localNewtonIterations==iterations,
         "compiled source380 selected material descriptor differs from its run binding");
    const auto required=NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION|
        NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION;
    need((material.flags&required)==required&&material.stateCount==15u&&
         material.parameterCount==28u&&material.localNewtonIterations==32u,
         "compiled source380 material flags/state/local budget are not the expected explicit accurate 15-state mode");
    const auto parameters=flatVector(field(inputBinding,"point_parameters"),
                                     28u,"point parameter values");
    need(material.parameterOffset+material.parameterCount<=world.parameters.size(),
         "compiled source380 material parameter range is invalid");
    for(std::size_t i=0u;i<parameters.size();++i){
        const float actual=world.parameters[material.parameterOffset+i].valueAndBounds.x;
        const float expected=static_cast<float>(parameters[i]);
        need(std::bit_cast<std::uint32_t>(actual)==std::bit_cast<std::uint32_t>(expected),
             "compiled source380 parameter word differs at "+std::to_string(i));
    }
    const float expectedTemperature=static_cast<float>(number(field(descriptor,"temperature_K"),"expected temperature"));
    need(std::bit_cast<std::uint32_t>(material.bulk.y)==std::bit_cast<std::uint32_t>(expectedTemperature)&&
         std::bit_cast<std::uint32_t>(row.input[46])==std::bit_cast<std::uint32_t>(expectedTemperature),
         "point temperature does not match the exact compiled material bulk.y word");
}

CompiledWorld compileSources(const std::filesystem::path& linerPath,
                             const std::filesystem::path& mediumPath,
                             const bool accurate,
                             const bool strictStateCertification = false,
                             const std::optional<std::uint32_t> requestedIterations = std::nullopt){
    need(!strictStateCertification || (accurate && requestedIterations.has_value()),
         "strict state certification requires accurate policy and explicit bounded budget");
    auto l=parseMatterFile(linerPath.string()),m=parseMatterFile(mediumPath.string());
    need(l.succeeded()&&m.succeeded(),"paper source material parse failed");
    need(l.material.name==kLinerName&&m.material.name==kMediumName,"paper material source identity mismatch");
    need(l.material.internalState.size()==15u&&m.material.internalState.size()==15u&&
         l.material.parameters.size()==28u&&m.material.parameters.size()==28u,
         "source assets do not match captured 15-state/28-parameter bindings");
    for(auto* material:{&l.material,&m.material}){
        material->paperCompensatedArithmetic=true;
        material->genericCompensatedImplicitProjection=true;
        material->genericBilinear15AccurateProjection=accurate;
        material->genericBilinear15StrictStateCertification=strictStateCertification;
    }
    WorldSource source;source.materials={l.material,m.material};source.gravity={0.0,0.0,0.0};
    source.environmentCount=1u;source.frameTimestep=1.0e-4;
    ObjectSource object;object.name="readonly_bilinear15_capture_diagnostic";
    object.representation=Representation::fem;object.deformableContact=false;
    object.deformableSelfContact=false;object.mixedFEM=false;
    object.femNodes={{0,0,0},{.01,0,0},{0,.01,0},{0,0,.01}};
    object.tetrahedra={{0u,1u,2u,3u}};object.characteristicLength=.01;
    source.objects={object};
    CompileOptions options;options.maximumRateExponent=0u;options.emitSpecializedMetal=false;
    options.localMaterialNewtonIterations=32u;
    if(accurate&&requestedIterations.has_value())
        options.accurateBilinear15LocalNewtonIterations=requestedIterations;
    auto compiled=compileWorld(source,options);
    need(compiled.succeeded(),"two-material captured world failed CPU compilation");
    need(compiled.world.materials.size()==2u&&compiled.world.parameters.size()==56u,
         "compiled paper world has unexpected material/parameter layout");
    const std::uint32_t required=NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION;
    const auto expectedIterations=requestedIterations.value_or(accurate?16u:32u);
    for(std::size_t i=0;i<2u;++i){
        const auto& gpu=compiled.world.materials[i];
        need(gpu.stateCount==15u&&(gpu.flags&required)==required,
             "compiled paper material omitted state count or actual source flags");
        need(((gpu.flags&NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION)!=0u)==accurate,
             "compiled accurate projection selector differs from requested policy");
        need(((gpu.flags&NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION)!=0u)==strictStateCertification,
             "compiled strict state-certification selector differs from requested policy");
        need(gpu.localNewtonIterations==expectedIterations,
             "compiled local Newton budget differs from requested exact budget");
    }
    return compiled.world;
}
void verifyParameters(const CompiledWorld& world,const Capture& capture){
    need(world.parameters.size()==56u,"compiled paper parameter count changed");
    for(std::size_t i=0;i<56u;++i){
        const float compiled=world.parameters[i].valueAndBounds.x;
        need(std::bit_cast<std::uint32_t>(compiled)==std::bit_cast<std::uint32_t>(capture.parameterRow[i]),
             "compiled parameter word differs from capture header at "+std::to_string(i));
    }
}
template<class T> id<MTLBuffer> buffer(id<MTLDevice> d,const std::vector<T>& x){
    T zero{};const void* p=x.empty()?static_cast<const void*>(&zero):static_cast<const void*>(x.data());
    return [d newBufferWithBytes:p length:sizeof(T)*std::max<std::size_t>(1u,x.size()) options:MTLResourceStorageModeShared];
}
std::string escaped(const std::string& s){std::ostringstream o;for(unsigned char c:s){if(c=='\\'||c=='"')o<<'\\'<<char(c);else if(c=='\n')o<<"\\n";else if(c=='\r')o<<"\\r";else if(c<0x20)o<<' ';else o<<char(c);}return o.str();}
void numberJSON(std::ostream& o,double v){if(std::isfinite(v))o<<std::setprecision(9)<<v;else o<<"null";}
void floatArrayJSON(std::ostream& o,const float* v,std::size_t n){o<<'[';for(std::size_t i=0;i<n;++i){if(i)o<<',';numberJSON(o,v[i]);}o<<']';}
void bitsArrayJSON(std::ostream& o,const float* v,std::size_t n){o<<'[';for(std::size_t i=0;i<n;++i){if(i)o<<',';o<<'"'<<"0x"<<std::hex<<std::setw(8)<<std::setfill('0')<<std::bit_cast<std::uint32_t>(v[i])<<std::dec<<std::setfill(' ')<<'"';}o<<']';}
void inputRecordJSON(std::ostream& o,const Record& r){
    o<<"{\"environment\":"<<r.env<<",\"tet_id\":"<<r.tet<<",\"material_index\":"<<r.material
     <<",\"material_name\":\""<<r.materialName<<"\",\"input_f32_bits\":";
    bitsArrayJSON(o,r.input.data(),r.input.size());o<<",\"accepted_state_f32_bits\":";
    bitsArrayJSON(o,r.accepted.data(),r.accepted.size());o<<'}';
}
struct Metal {
    id<MTLDevice> device=nil;id<MTLCommandQueue> queue=nil;id<MTLLibrary> library=nil;
    id<MTLComputePipelineState> batch=nil,trace=nil;
    Metal(){
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");
        const std::string name=[[device name] UTF8String];
        need(name.find("Apple")!=std::string::npos&&name.find("Paravirtual")==std::string::npos,
             "physical Apple GPU required; no native execution on virtual device");
        queue=[device newCommandQueue];need(queue!=nil,"Metal queue unavailable");NSError* e=nil;
        library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&e];
        need(library!=nil,"paper diagnostic metallib unavailable");
        auto make=[&](NSString* n){id<MTLFunction> f=[library newFunctionWithName:n];need(f!=nil,"missing production diagnostic function");id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&e];need(p!=nil,"pipeline creation failed");return p;};
        batch=make(@"cardboard_bilinear15_capture_batch");trace=make(@"cardboard_bilinear15_capture_trace");
    }
    id<MTLBuffer> compileBuffer(const std::vector<NMMaterialGPU>& x){return buffer(device,x);}
    void dispatch(id<MTLComputePipelineState> pipeline,const std::vector<id<MTLBuffer>>& bs,
                  std::uint32_t count,const std::uint32_t scalarBuffer=UINT32_MAX){
        id<MTLCommandBuffer> cb=[queue commandBuffer];need(cb!=nil,"Metal command buffer unavailable");
        id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];need(enc!=nil,"Metal encoder unavailable");
        [enc setComputePipelineState:pipeline];for(NSUInteger i=0;i<bs.size();++i)[enc setBuffer:bs[i] offset:0 atIndex:i];
        if(scalarBuffer!=UINT32_MAX)[enc setBytes:&count length:sizeof(count) atIndex:scalarBuffer];
        const NSUInteger width=std::max<NSUInteger>(1u,std::min<NSUInteger>(pipeline.threadExecutionWidth,count));
        [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        need(cb.status==MTLCommandBufferStatusCompleted,cb.error?std::string([[cb.error description] UTF8String]):"Metal dispatch failed");
    }
};
std::vector<float> materialParameters(const CompiledWorld& w){std::vector<float> p;p.reserve(w.parameters.size());for(const auto& x:w.parameters)p.push_back(x.valueAndBounds.x);return p;}
std::vector<float> acceptedRows(const std::vector<Record>& rows){std::vector<float> x;x.reserve(rows.size()*kStateCount);for(const auto& r:rows)x.insert(x.end(),r.accepted.begin(),r.accepted.end());return x;}
std::vector<float> inputRows(const std::vector<Record>& rows){std::vector<float> x;x.reserve(rows.size()*kInputStride);for(const auto& r:rows)x.insert(x.end(),r.input.begin(),r.input.end());return x;}
std::vector<std::uint32_t> materialRows(const std::vector<Record>& rows){std::vector<std::uint32_t>x;x.reserve(rows.size());for(const auto&r:rows)x.push_back(r.material);return x;}
void writeBatch(const std::filesystem::path& path,const std::string& policy,const CompiledWorld& world,
                const std::vector<Record>& rows,const std::vector<float>& out,const std::string& captureHash,const bool pointInput=false){
    std::ofstream f(path);need(f.good(),"cannot write batch output");
    f<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.bilinear15-capture-diagnostic.v1\",\"policy\":\""<<policy
     <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA<<"\","<<(pointInput?"\"point_input_sha256\":\"":"\"capture_sha256\":\"")<<captureHash<<"\",\"record_count\":"<<rows.size()
     <<",\"compiled_local_iterations_liner\":"<<world.materials[0].localNewtonIterations
     <<",\"compiled_local_iterations_medium\":"<<world.materials[1].localNewtonIterations
     <<",\"physical_validation\":false,\"interpretation\":\"direct stress and tangent projection status comparison only; not a material calibration or full solver qualification\"}\n";
    std::size_t stressOK=0,tangentOK=0,projectOK=0;
    for(std::size_t i=0;i<rows.size();++i){const auto* p=out.data()+i*kOutputStride;stressOK+=p[0]==1.0f;tangentOK+=p[1]==1.0f;projectOK+=p[2]==1.0f;
        double diff2=0.0,ref2=0.0,maxDiff=0.0;for(unsigned j=0;j<9u;++j){double d=double(p[8+j])-double(p[17+j]);diff2+=d*d;ref2+=double(p[8+j])*double(p[8+j]);maxDiff=std::max(maxDiff,std::abs(d));}
        f<<"{\"record\":\"point\",\"environment\":"<<rows[i].env<<",\"tet_id\":"<<rows[i].tet
         <<",\"material_index\":"<<rows[i].material<<",\"material_name\":\""<<rows[i].materialName
         <<"\",\"stress_valid\":"<<(p[0]==1.0f?"true":"false")<<",\"tangent_valid\":"<<(p[1]==1.0f?"true":"false")
         <<",\"tangent_projection_converged\":"<<(p[2]==1.0f?"true":"false")<<",\"projection_residual\":";numberJSON(f,p[3]);
        f<<",\"projection_state_norm\":";numberJSON(f,p[4]);f<<",\"projection_iterations\":";numberJSON(f,p[5]);
        f<<",\"projection_flags\":";numberJSON(f,p[6]);f<<",\"projection_failure\":";numberJSON(f,p[7]);
        f<<",\"direct_stress_pa\":";floatArrayJSON(f,p+8,9);f<<",\"tangent_path_stress_pa\":";floatArrayJSON(f,p+17,9);
        f<<",\"tangent_pa_per_s\":";floatArrayJSON(f,p+26,9);f<<",\"projected_state_high\":";floatArrayJSON(f,p+35,15);
        f<<",\"projected_state_low\":";floatArrayJSON(f,p+50,15);f<<",\"direct_vs_tangent_stress_abs_l2\":";numberJSON(f,std::sqrt(diff2));
        f<<",\"direct_vs_tangent_stress_relative_l2\":";numberJSON(f,ref2>0?std::sqrt(diff2/ref2):NAN);
        f<<",\"direct_vs_tangent_stress_max_abs\":";numberJSON(f,maxDiff);f<<",\"output_f32_bits\":";
        bitsArrayJSON(f,p,kOutputStride);f<<"}\n";
    }
    f<<"{\"record\":\"end\",\"rows\":"<<rows.size()<<",\"direct_stress_valid_count\":"<<stressOK
     <<",\"tangent_valid_count\":"<<tangentOK<<",\"tangent_projection_converged_count\":"<<projectOK
     <<",\"status\":\"diagnostic_complete\",\"not_qualification\":true}\n";
}
void writeTrace(const std::filesystem::path& path,const std::string& policy,const CompiledWorld& world,
                const Record& row,const float* out,const std::string& captureHash,const bool pointInput=false){
    std::ofstream f(path);need(f.good(),"cannot write selected projection trace");
    f<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.bilinear15-selected-projection-trace.v1\",\"policy\":\""<<policy
     <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA<<"\","<<(pointInput?"\"point_input_sha256\":\"":"\"capture_sha256\":\"")<<captureHash<<"\",\"compiled_local_iterations\":"<<world.materials[row.material].localNewtonIterations
     <<",\"point\":";inputRecordJSON(f,row);f<<",\"summary\":{\"direct_project_valid\":"<<(out[0]==1?"true":"false")
     <<",\"direct_projection_residual\":";numberJSON(f,out[1]);f<<",\"direct_state_norm\":";numberJSON(f,out[2]);
    f<<",\"direct_iterations\":";numberJSON(f,out[3]);f<<",\"direct_flags\":";numberJSON(f,out[4]);f<<",\"direct_failure\":";numberJSON(f,out[5]);
    f<<",\"stress_valid\":"<<(out[6]==1?"true":"false")<<",\"tangent_valid\":"<<(out[7]==1?"true":"false")
     <<",\"tangent_projection_converged\":"<<(out[8]==1?"true":"false")<<",\"tangent_projection_residual\":";numberJSON(f,out[9]);
    f<<",\"tangent_iterations\":";numberJSON(f,out[11]);f<<",\"tangent_flags\":";numberJSON(f,out[12]);f<<",\"tangent_failure\":";numberJSON(f,out[13]);
    f<<",\"stress_direct\":";floatArrayJSON(f,out+14,9);f<<",\"stress_tangent_path\":";floatArrayJSON(f,out+23,9);
    f<<",\"tangent\":";floatArrayJSON(f,out+32,9);f<<",\"projected_state_high\":";floatArrayJSON(f,out+41,15);
    f<<",\"projected_state_low\":";floatArrayJSON(f,out+56,15);f<<"},\"summary_f32_bits\":";
    bitsArrayJSON(f,out,73u);
    const float traceStatus=out[kTraceSummarySlots+kTraceHeaderFailureSlot];
    const char* traceDisposition=traceStatus==-1.0f?"unsupported_by_trace_capacity":"trace_recorded";
    f<<",\"trace_status_code\":";numberJSON(f,traceStatus);
    f<<",\"trace_disposition\":\""<<traceDisposition<<"\",\"packing_sentinels\":[\"0x3e200000\",\"0x3ea00000\"],\"trace_layout\":{\"header_slots\":8,\"iteration_slots\":374,\"record_slot5\":\"scaled_correction_norm; -1 when the legacy solve path has no scaled metric\",\"max_iterations\":"<<kTraceIterationCapacity<<",\"slots\":"<<kTraceSlots<<"},\"trace_f32\":";
    floatArrayJSON(f,out+kTraceSummarySlots,kTraceSlots);f<<",\"trace_f32_bits\":";bitsArrayJSON(f,out+kTraceSummarySlots,kTraceSlots);f<<"}\n";
}
std::vector<float> runBatch(Metal& metal,const CompiledWorld& world,const std::string& policy,const std::vector<Record>& rows,
              const std::filesystem::path& outdir,const std::string& captureHash,const bool pointInput=false){
    auto params=materialParameters(world);auto acc=acceptedRows(rows);auto inputs=inputRows(rows);auto ids=materialRows(rows);
    std::vector<float> out(rows.size()*kOutputStride,0.0f);
    std::vector<id<MTLBuffer>> bs{buffer(metal.device,world.materials),buffer(metal.device,world.scalarPrograms),
       buffer(metal.device,world.instructions),buffer(metal.device,params),buffer(metal.device,acc),
       buffer(metal.device,inputs),buffer(metal.device,ids),buffer(metal.device,out)};
    need(std::all_of(bs.begin(),bs.end(),[](id<MTLBuffer> b){return b!=nil;}),"batch Metal buffer allocation failed");
    metal.dispatch(metal.batch,bs,static_cast<std::uint32_t>(rows.size()),8u);
    const auto* data=static_cast<const float*>(bs[7].contents);std::copy(data,data+out.size(),out.begin());
    writeBatch(outdir/("paper-points-"+policy+".jsonl"),policy,world,rows,out,captureHash,pointInput);
    return out;
}
std::vector<float> runTrace(Metal& metal,const CompiledWorld& world,const std::string& policy,const Record& row,
              const std::filesystem::path& outdir,const std::string& captureHash,const bool pointInput=false){
    auto params=materialParameters(world);std::vector<float> acc(row.accepted.begin(),row.accepted.end());
    std::vector<float> inputs(row.input.begin(),row.input.end());std::vector<float> out(kTraceSummarySlots+kTraceSlots,0.0f);
    std::uint32_t mat=row.material;
    std::vector<id<MTLBuffer>> bs{buffer(metal.device,world.materials),buffer(metal.device,world.scalarPrograms),
       buffer(metal.device,world.instructions),buffer(metal.device,params),buffer(metal.device,acc),
       buffer(metal.device,inputs),buffer(metal.device,out)};
    need(std::all_of(bs.begin(),bs.end(),[](id<MTLBuffer> b){return b!=nil;}),"trace Metal buffer allocation failed");
    id<MTLCommandBuffer> cb=[metal.queue commandBuffer];need(cb!=nil,"trace command buffer unavailable");
    id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];need(enc!=nil,"trace encoder unavailable");
    [enc setComputePipelineState:metal.trace];for(NSUInteger i=0;i<bs.size();++i)[enc setBuffer:bs[i] offset:0 atIndex:i];
    [enc setBytes:&mat length:sizeof(mat) atIndex:7];
    [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [enc endEncoding];[cb commit];[cb waitUntilCompleted];need(cb.status==MTLCommandBufferStatusCompleted,"trace Metal dispatch failed");
    const auto* data=static_cast<const float*>(bs[6].contents);
    need(std::bit_cast<std::uint32_t>(data[71])==0x3e200000u&&
         std::bit_cast<std::uint32_t>(data[72])==0x3ea00000u,
         "selected trace high/low state packing sentinel was overwritten");
    writeTrace(outdir/("tet"+std::to_string(row.tet)+"-env"+std::to_string(row.env)+"-"+policy+"-trace.json"),policy,world,row,data,captureHash,pointInput);
    return std::vector<float>(data,data+out.size());
}
// Reserve evidence exclusively before dispatch; failures retain partial evidence.
struct ExclusiveSweepOutput {
    int descriptor = -1;
    explicit ExclusiveSweepOutput(const std::filesystem::path& path) {
        descriptor=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
        need(descriptor>=0,"cannot exclusively create controlled sweep output: "+path.string());
    }
    ~ExclusiveSweepOutput(){if(descriptor>=0)::close(descriptor);}
    ExclusiveSweepOutput(const ExclusiveSweepOutput&)=delete;
    ExclusiveSweepOutput& operator=(const ExclusiveSweepOutput&)=delete;
    void writeAll(const std::string& text) {
        need(text.size()<=16u*1024u*1024u,"controlled sweep output exceeds bounded writer");
        std::size_t written=0u;
        while(written<text.size()) {
            const auto count=::write(descriptor,text.data()+written,text.size()-written);
            if(count<0&&errno==EINTR)continue;
            need(count>0,"controlled sweep write failed; partial evidence retained");
            written+=static_cast<std::size_t>(count);
        }
        const int saved=descriptor;descriptor=-1;
        need(::close(saved)==0,"controlled sweep close failed; evidence retained");
    }
};
// Instrument-only controlled directions around immutable captured centers.
// Output records identify authored perturbations separately from capture data.
struct DerivativeSweepRecord {
    Record row;
    const char* kind = "center_repeat";
    int basis = -1;
    int sign = 0;
    unsigned repeat = 0;
    double epsilon = 0.0;
};
std::vector<DerivativeSweepRecord> derivativeSweepRecords(const Capture& capture) {
    constexpr std::array<unsigned,8> selected = {567u,568u,569u,1527u,1528u,1529u,1634u,1638u};
    constexpr std::array<double,10> epsilons = {1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12};
    std::vector<DerivativeSweepRecord> records;
    for (const auto tet : selected) {
        const auto found = std::find_if(capture.rows.begin(),capture.rows.end(),[&](const Record& r) {
            return r.env == 0u && r.tet == tet;
        });
        need(found != capture.rows.end(),"controlled derivative sweep point missing");
        const Record center = *found;
        for (unsigned repeat=0u;repeat<3u;++repeat) {
            DerivativeSweepRecord record;record.row=center;record.repeat=repeat;
            records.push_back(record);
        }
        for (unsigned basis=0u;basis<9u;++basis) {
            Record directed=center;
            for (unsigned j=0u;j<9u;++j) {
                directed.input[27u+j]=j==basis?1.0f:0.0f;
                directed.input[36u+j]=j==basis?1.0f/center.input[45u]:0.0f;
            }
            DerivativeSweepRecord action;action.row=directed;
            action.kind="basis_action";action.basis=static_cast<int>(basis);
            records.push_back(action);
            for (const double epsilon : epsilons) for (const int sign : {-1,1}) {
                DerivativeSweepRecord sample=action;
                sample.kind="basis_perturbation";sample.sign=sign;sample.epsilon=epsilon;
                const double targetF=double(center.input[basis])+double(center.input[9u+basis])+
                    double(sign)*epsilon;
                const float high=static_cast<float>(targetF);
                sample.row.input[basis]=high;
                // Form the residual before adding the order-one high word.
                // This preserves sub-FP64-ULP perturbations when the high
                // word stays unchanged; every realized input is still logged.
                const double residualF=(double(center.input[basis])-double(high))+
                    (double(center.input[9u+basis])+double(sign)*epsilon);
                sample.row.input[9u+basis]=static_cast<float>(residualF);
                // Preserve the captured FEM kinematic F/Fdot perturbation
                // relationship; output retains every actually rounded word.
                const double targetRate=double(center.input[18u+basis])+
                    double(sign)*epsilon/double(center.input[45u]);
                sample.row.input[18u+basis]=static_cast<float>(targetRate);
                need(std::isfinite(sample.row.input[basis])&&
                     std::isfinite(sample.row.input[9u+basis])&&
                     std::isfinite(sample.row.input[18u+basis]),"nonfinite controlled perturbation");
                records.push_back(sample);
            }
        }
    }
    need(records.size()==8u*(3u+9u*(1u+2u*epsilons.size())),"controlled sweep cardinality mismatch");
    return records;
}
void runDerivativeSweep(Metal& metal,const CompiledWorld& world,const std::string& policy,
                        const Capture& capture,const std::filesystem::path& outdir,
                        const std::string& captureHash) {
    ExclusiveSweepOutput evidence(outdir/("paper-derivative-sweep-"+policy+".jsonl"));
    const auto samples=derivativeSweepRecords(capture);
    std::vector<Record> rows;rows.reserve(samples.size());
    for(const auto& sample:samples)rows.push_back(sample.row);
    auto params=materialParameters(world);auto accepted=acceptedRows(rows);
    auto inputs=inputRows(rows);auto ids=materialRows(rows);
    std::vector<float> output(rows.size()*kOutputStride,0.0f);
    std::vector<id<MTLBuffer>> buffers{buffer(metal.device,world.materials),
        buffer(metal.device,world.scalarPrograms),buffer(metal.device,world.instructions),
        buffer(metal.device,params),buffer(metal.device,accepted),buffer(metal.device,inputs),
        buffer(metal.device,ids),buffer(metal.device,output)};
    need(std::all_of(buffers.begin(),buffers.end(),[](id<MTLBuffer> b){return b!=nil;}),
         "controlled derivative sweep allocation failed");
    metal.dispatch(metal.batch,buffers,static_cast<std::uint32_t>(rows.size()),8u);
    const auto* result=static_cast<const float*>(buffers[7].contents);
    std::copy(result,result+output.size(),output.begin());
    std::ostringstream file;
    file<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.controlled-material-derivative-sweep.v1\","
        <<"\"policy\":\""<<policy<<"\",\"compiled_local_iterations_liner\":"<<world.materials[0].localNewtonIterations
        <<",\"compiled_local_iterations_medium\":"<<world.materials[1].localNewtonIterations
        <<",\"capture_sha256\":\""<<captureHash
        <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA
        <<"\",\"rows\":"<<samples.size()<<",\"selected_points\":8,\"basis_order\":\"row_major_F_11_12_13_21_22_23_31_32_33\","
        <<"\"epsilon_unit\":\"s\",\"basis_direction_unit\":\"per_s\","
        <<"\"rate_direction_rule\":\"Hdot=H/dt; actual rounded F and Fdot words recorded\","
        <<"\"center_repeat_meaning\":\"same-batch nondeterminism observation only; not an absolute arithmetic noise floor\","
        <<"\"original_capture_modified\":false,\"physical_steps\":0,\"physical_validation\":false}\n";
    for(std::size_t i=0u;i<samples.size();++i) {
        const auto& sample=samples[i];const auto* value=output.data()+i*kOutputStride;
        file<<"{\"record\":\"controlled_sample\",\"kind\":\""<<sample.kind
            <<"\",\"basis\":"<<sample.basis<<",\"sign\":"<<sample.sign
            <<",\"repeat\":"<<sample.repeat<<",\"epsilon_s\":"<<std::setprecision(17)<<sample.epsilon
            <<",\"point\":";inputRecordJSON(file,sample.row);
        file<<",\"output_f32_bits\":";bitsArrayJSON(file,value,kOutputStride);
        file<<",\"stress_valid\":"<<(value[0]==1.0f?"true":"false")
            <<",\"tangent_valid\":"<<(value[1]==1.0f?"true":"false")
            <<",\"projection_valid\":"<<(value[2]==1.0f?"true":"false")
            <<",\"projection_failure\":";numberJSON(file,value[7]);file<<"}\n";
    }
    file<<"{\"record\":\"end\",\"rows\":"<<samples.size()
        <<",\"status\":\"diagnostic_complete\",\"not_qualification\":true}\n";
    need(file.good(),"controlled derivative sweep formatting failed");
    evidence.writeAll(file.str());
}


void writePointRunManifest(const std::filesystem::path& path,const std::string& inputSHA,
                           const std::string& packageSHA,const std::string& mediumSHA,
                           const std::string& fingerprint,const CompiledWorld& world,
                           const SourceArtifactBinding& sourceArtifacts,
                           const Record& row,const std::vector<float>& trace,
                           const std::vector<float>& batch) {
    need(trace.size()==kTraceSummarySlots+kTraceSlots&&batch.size()==kOutputStride,
         "point trace/batch output sizes are inconsistent");
    const auto& material=world.materials[row.material];
    std::array<unsigned,5> mismatch{};
    const std::array<std::pair<unsigned,unsigned>,5> spans={{{14u,8u},{23u,17u},{32u,26u},{41u,35u},{56u,50u}}};
    const std::array<unsigned,5> lengths={9u,9u,9u,15u,15u};
    for(std::size_t span=0;span<spans.size();++span){
        for(unsigned i=0;i<lengths[span];++i){
            if(std::bit_cast<std::uint32_t>(trace[spans[span].first+i])!=
               std::bit_cast<std::uint32_t>(batch[spans[span].second+i]))++mismatch[span];
        }
    }
    std::ofstream f(path);need(f.good(),"cannot write source380 point run manifest");
    f<<"{\"schema\":\"numi.cardboard.source380-native-material-point-result.v1\",\"run_name\":\"explicit32-certified-coupon-380\",\"environment\":"<<row.env
     <<",\"tet_id\":"<<row.tet<<",\"material_index\":"<<row.material<<",\"material_name\":\""<<row.materialName
     <<"\",\"input_sha256\":\""<<inputSHA<<"\",\"compiled_package_sha256\":\""<<packageSHA
     <<"\",\"source_medium_asset_sha256\":\""<<mediumSHA<<"\",\"compiled_world_fingerprint\":\""<<fingerprint
     <<"\",\"source112_base_snapshot_sha256\":\""<<kBaseSnapshotSHA<<"\",\"observation_status\":\"native diagnostic dispatch completed; constitutive result reported without an expected pass/fail assertion\""
     <<",\"source380_artifacts\":{\"candidate_nodes_sha256\":\""<<sourceArtifacts.candidateSHA
     <<"\",\"accepted_history_sha256\":\""<<sourceArtifacts.historySHA
     <<"\",\"mesh_sha256\":\""<<sourceArtifacts.meshSHA
     <<"\",\"accepted_history_verified_against_point_state\":true,\"F_words_native_readback\":false}"
     <<",\"compiled_material\":{\"flags\":"<<material.flags<<",\"state_count\":"<<material.stateCount
     <<",\"parameter_count\":"<<material.parameterCount<<",\"local_newton_iterations\":"<<material.localNewtonIterations
     <<",\"parameter_offset\":"<<material.parameterOffset<<",\"bulk_temperature_word\":";numberJSON(f,material.bulk.y);
    f<<"},\"input_provenance\":{\"F_high_low_source\":\"CPU transcription from source380 candidate nodes and compiled tetrahedron; separately requested production-kernel readback remains pending\",\"F_words_native_readback\":false,\"accepted_history_source\":\"source380 post-rollback accepted-state export, independently hash- and word-checked\"}"
     <<",\"trace_capacity\":{\"state_slots\":"<<kTraceStateCapacity<<",\"iteration_slots\":"<<kTraceIterationCapacity<<",\"total_float_slots\":"<<kTraceSlots
     <<"},\"direct_trace_vs_trace_off_batch_bit_mismatches\":{\"stress_direct\":"<<mismatch[0]<<",\"stress_tangent_path\":"<<mismatch[1]
     <<",\"tangent\":"<<mismatch[2]<<",\"projected_state_high\":"<<mismatch[3]<<",\"projected_state_low\":"<<mismatch[4]
     <<"},\"trace_observed\":{\"projection_valid\":"<<(trace[0]==1.0f?"true":"false")<<",\"projection_iterations\":";numberJSON(f,trace[3]);
    f<<",\"projection_flags\":";numberJSON(f,trace[4]);f<<",\"projection_failure\":";numberJSON(f,trace[5]);
    f<<",\"projection_residual\":";numberJSON(f,trace[1]);f<<",\"projection_state_norm\":";numberJSON(f,trace[2]);
    f<<",\"stress_valid\":"<<(trace[6]==1.0f?"true":"false")<<",\"tangent_valid\":"<<(trace[7]==1.0f?"true":"false")
     <<",\"tangent_projection_valid\":"<<(trace[8]==1.0f?"true":"false")<<",\"tangent_failure\":";numberJSON(f,trace[13]);
    f<<",\"trace_status\":";numberJSON(f,trace[80]);f<<"},\"raw_input_f32_bits\":";bitsArrayJSON(f,row.input.data(),row.input.size());
    f<<",\"raw_accepted_state_f32_bits\":";bitsArrayJSON(f,row.accepted.data(),row.accepted.size());
    f<<",\"receipt_kind\":\"diagnostic_observation_only\",\"physical_steps\":0,\"physical_validation\":false,\"full_solver_qualification\":false}\n";
    need(f.good(),"source380 point result manifest write failed");
}
int runExactSource392Tet858Trace(const std::filesystem::path& pointPath,
                                  const std::filesystem::path& runPath,
                                  const std::filesystem::path& mediumPath,
                                  const std::filesystem::path& outdir) {
    constexpr std::string_view pointSHA="411d522ebd281eae2b75d99f8f7927d2df3521795b1a04dc4a325953aafcc4d8";
    constexpr std::string_view receiptSHA="bc1495a5a63e35a36182b9206350ff2f68480345ae1b205484215a1d53da7bcb";
    constexpr std::string_view runManifestSHA="fbfcf51db84a3710cc3bd96e016597728163da464974ea033554e82e587cd8eb";
    constexpr std::string_view packageSHA="cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c";
    constexpr std::string_view candidateSHA="89846e4efc91c1774b52d6b3732f633e78414a7d341b6cbd8b9b4966e4be0417";
    constexpr std::string_view conditioningSHA="ad51b0ceca073a4736ee128d0daf13eee82bf3a398919cc911a81fdff8399ced";
    constexpr std::string_view acceptedSHA="e32909672f5ef53992fd7c798419dcdcd257662daa0f0c6cc8d960a1dced5dfd";
    const auto output=runPath/"output";
    const auto receiptPath=runPath/"receipt.json";
    const auto runManifestPath=output/"manifest.json";
    const auto packagePath=output/"compiled.nmatterpack";
    const auto candidatePath=output/"rejected-conditioning-000001/candidate_nodes.bin";
    const auto conditioningPath=output/"rejected-conditioning-000001/manifest.json";
    const auto acceptedPath=output/"material_state_000001_bent.json";
    need(sha256(pointPath)==pointSHA,"native401 point output SHA changed");
    need(sha256(receiptPath)==receiptSHA&&sha256(runManifestPath)==runManifestSHA&&
         sha256(packagePath)==packageSHA&&sha256(candidatePath)==candidateSHA&&
         sha256(conditioningPath)==conditioningSHA&&sha256(acceptedPath)==acceptedSHA,
         "source392 source artifact hashes differ from the retained native401 case");
    need(sha256(mediumPath)==kMediumSHA,"source392 medium asset SHA changed");

    NSDictionary* point=readJSONDocument(pointPath);
    need(stringValue(field(point,"schema"),"source392 point schema")==
             "numi.cardboard.source392-tet-point.v1"&&
         number(field(point,"environment"),"source392 environment")==0.0&&
         number(field(point,"tet_id"),"source392 tet")==858.0&&
         number(field(point,"material_index"),"source392 material index")==1.0&&
         number(field(point,"control_step"),"source392 control step")==0.0,
         "source392 point identity differs from env0/tet858/material1/control0");
    need(stringValue(field(point,"receipt_sha256"),"point receipt binding")==receiptSHA&&
         stringValue(field(point,"run_manifest_sha256"),"point run manifest binding")==runManifestSHA&&
         stringValue(field(point,"package_sha256"),"point package binding")==packageSHA&&
         stringValue(field(point,"candidate_nodes_sha256"),"point candidate binding")==candidateSHA&&
         stringValue(field(point,"conditioning_manifest_sha256"),"point conditioning binding")==conditioningSHA&&
         stringValue(field(point,"accepted_state_sha256"),"point accepted history binding")==acceptedSHA&&
         stringValue(field(point,"medium_asset_sha256"),"point material asset binding")==kMediumSHA,
         "native401 point bindings differ from immutable source392 artifacts");
    need([(NSNumber*)field(point,"physics_fingerprint") unsignedLongLongValue]==8485105279886344921ull&&
         number(field(point,"projection_flags"),"source392 projection flags")==63.0&&
         number(field(point,"local_updates"),"source392 local updates")==32.0&&
         number(field(point,"terminal_failure"),"source392 projection failure")==7.0&&
         ![(NSNumber*)field(point,"projection_converged") boolValue],
         "native401 projection outcome/compiled world identity changed");

    NSDictionary* runManifest=readJSONDocument(runManifestPath);
    need([(NSNumber*)field(runManifest,"compiled_world_fingerprint") unsignedLongLongValue]==8485105279886344921ull,
         "source392 run manifest fingerprint differs from point diagnostic");
    NSDictionary* receipt=readJSONDocument(receiptPath);
    need([(NSNumber*)field(receipt,"execution_started") boolValue]&&
         [(NSNumber*)field(receipt,"source_inputs_unchanged") boolValue]&&
         number(field(receipt,"exit_code"),"native source392 point exit code")==2.0,
         "source392 coupon receipt is not the retained failed physical run");
    NSDictionary* condition=readJSONDocument(conditioningPath);
    need(number(field(condition,"control_step"),"conditioning step") == 0.0&&
         number(field(condition,"accepted_physical_state"),"conditioning accepted state")==0.0&&
         stringValue(field(condition,"source_physics_fingerprint"),"conditioning fingerprint")==
             "8485105279886344921",
         "source392 point candidate is not the exact rejected step-0 conditioning snapshot");
    NSDictionary* state=readJSONDocument(acceptedPath);
    need(number(field(state,"environment"),"accepted history environment")==0.0&&
         number(field(state,"status_code"),"accepted history status")==5.0,
         "source392 accepted history is not the status-5 rollback export");
    NSArray* stateRows=(NSArray*)field(state,"tetrahedra");
    need(stateRows.count==1920u,"source392 accepted history tet count mismatch");
    NSDictionary* selectedState=(NSDictionary*)stateRows[858u];
    need(number(field(selectedState,"index"),"accepted state tet index")==858.0&&
         number(field(selectedState,"material_index"),"accepted state material index")==1.0,
         "source392 accepted state row identity mismatch");
    const auto accepted=flatVector(field(selectedState,"state"),15u,"source392 accepted history state");
    need(std::all_of(accepted.begin(),accepted.end(),[](float value){return value==0.0f;}),
         "source392 accepted rollback history is not all zeros");

    const auto rawWords=bitVector(field(point,"raw_160_f32_words"),160u,"native401 raw output words");
    const auto FHigh=flatArray<9>(field(point,"F_high"),"native401 F_high");
    const auto FLow=flatArray<9>(field(point,"F_low"),"native401 F_low");
    const auto FDot=flatArray<9>(field(point,"Fdot_local"),"native401 Fdot_local");
    verifyFloatWords(FHigh.data(),std::vector<std::uint32_t>(rawWords.begin()+3u,rawWords.begin()+12u),"native401 F high");
    verifyFloatWords(FLow.data(),std::vector<std::uint32_t>(rawWords.begin()+12u,rawWords.begin()+21u),"native401 F low");
    verifyFloatWords(FDot.data(),std::vector<std::uint32_t>(rawWords.begin()+21u,rawWords.begin()+30u),"native401 Fdot");
    Record row;row.env=0u;row.tet=858u;row.material=1u;row.materialName=kMediumName;
    std::copy(FHigh.begin(),FHigh.end(),row.input.begin());
    std::copy(FLow.begin(),FLow.end(),row.input.begin()+9u);
    std::copy(FDot.begin(),FDot.end(),row.input.begin()+18u);
    // Match the source392 point instrument's explicit tangent direction.
    row.input[27u]=1.0f;
    const float dt=static_cast<float>(number(field(point,"dt_s"),"native401 dt"));
    const float temperature=static_cast<float>(number(field(point,"temperature_K"),"native401 temperature"));
    need(dt==9.999999747378752e-5f&&temperature==293.1499938964844f,
         "native401 timestep or temperature float word changed");
    row.input[36u]=1.0f/dt;row.input[45u]=dt;row.input[46u]=temperature;
    for(std::size_t i=0u;i<accepted.size();++i)row.accepted[i]=accepted[i];

    CompiledWorld world;std::string packageError;
    need(readPackage(packagePath,world,nullptr,&packageError),
         "source392 compiled package rejected: "+packageError);
    need(world.fingerprint==8485105279886344921ull&&world.dispatch.environmentCount==2u&&
         world.dispatch.femNodeCount==882u&&world.dispatch.tetrahedronCount==1920u&&
         world.materials.size()==3u&&world.materials[1].stateCount==15u&&
         world.materials[1].parameterCount==28u&&world.materials[1].flags==213u&&
         world.materials[1].localNewtonIterations==32u&&
         world.fem.tetrahedra.size()>858u&&world.fem.tetrahedra[858u].identity.x==1u&&
         world.fem.tetrahedra[858u].nodes.x==290u&&world.fem.tetrahedra[858u].nodes.y==293u&&
         world.fem.tetrahedra[858u].nodes.z==292u&&world.fem.tetrahedra[858u].nodes.w==587u,
         "source392 compiled package/material/tet descriptor differs from the exact point");
    const ParseResult parsed=parseMatterFile(mediumPath.string());
    need(parsed.succeeded()&&parsed.material.name==kMediumName&&
         parsed.material.parameters.size()==28u,"source392 medium source parse/binding failed");
    const auto& compiledMaterial=world.materials[1];
    for(std::size_t i=0u;i<28u;++i){
        need(compiledMaterial.parameterOffset+i<world.parameters.size(),"source392 parameter range invalid");
        const float compiled=world.parameters[compiledMaterial.parameterOffset+i].valueAndBounds.x;
        const float source=static_cast<float>(parsed.material.parameters[i].defaultValue);
        need(std::bit_cast<std::uint32_t>(compiled)==std::bit_cast<std::uint32_t>(source),
             "compiled source392 material parameter word differs at index "+std::to_string(i));
    }
    need(!std::filesystem::exists(outdir),"source392 trace output directory already exists");
    need(std::filesystem::create_directories(outdir),"could not exclusively create source392 trace output directory");
    const std::string inputSHA=std::string(pointSHA);
    Metal metal;
    const auto trace=runTrace(metal,world,"source392_tet858_exact401_trace",row,outdir,inputSHA,true);
    const auto batch=runBatch(metal,world,"source392_tet858_trace_off",std::vector<Record>{row},outdir,inputSHA,true);
    const std::array<std::pair<unsigned,unsigned>,5> spans={{{14u,8u},{23u,17u},{32u,26u},{41u,35u},{56u,50u}}};
    const std::array<unsigned,5> lengths={9u,9u,9u,15u,15u};
    std::array<unsigned,5> mismatches{};
    for(std::size_t span=0u;span<spans.size();++span)
        for(unsigned i=0u;i<lengths[span];++i)
            mismatches[span]+=std::bit_cast<std::uint32_t>(trace[spans[span].first+i])!=
                              std::bit_cast<std::uint32_t>(batch[spans[span].second+i]);
    const auto& outputPath=outdir/"source392-tet858-trace-run.json";
    std::ofstream result(outputPath);need(result.good(),"cannot create source392 trace-run manifest");
    result<<std::setprecision(17)
      <<"{\"schema\":\"numi.cardboard.source392-tet858-projection-trace.v1\","
      <<"\"input_coupon_source_snapshot\":\"source-scaled-material-merit-118\","
      <<"\"point401_sha256\":\""<<pointSHA<<"\",\"coupon_receipt_sha256\":\""<<receiptSHA
      <<"\",\"run_manifest_sha256\":\""<<runManifestSHA<<"\",\"compiled_package_sha256\":\""<<packageSHA
      <<"\",\"medium_asset_sha256\":\""<<kMediumSHA<<"\",\"candidate_nodes_sha256\":\""<<candidateSHA
      <<"\",\"accepted_state_sha256\":\""<<acceptedSHA<<"\",\"conditioning_manifest_sha256\":\""<<conditioningSHA
      <<"\",\"world_fingerprint\":8485105279886344921,\"environment\":0,\"tet_id\":858,\"material_index\":1,"
      <<"\"candidate_is_rejected_conditioning_snapshot\":true,\"first_failure_generation_bound\":false,"
      <<"\"input_f32_bits\":";bitsArrayJSON(result,row.input.data(),row.input.size());
    result<<",\"accepted_history_f32_bits\":";bitsArrayJSON(result,row.accepted.data(),row.accepted.size());
    result<<",\"native401_projection\":{\"converged\":false,\"iterations\":32,\"failure\":7,\"residual\":"
      <<number(field(point,"projection_residual"),"native401 projection residual")<<",\"state_norm\":"
      <<number(field(point,"projection_state_norm"),"native401 state norm")<<"},"
      <<"\"trace_projection\":{\"converged\":"<<(trace[0]==1.0f?"true":"false")<<",\"iterations\":"<<trace[3]
      <<",\"flags\":"<<trace[4]<<",\"failure\":"<<trace[5]<<",\"raw_residual\":"<<trace[1]
      <<",\"state_norm\":"<<trace[2]<<"},\"trace_off_bit_mismatches\":{"
      <<"\"direct_stress\":"<<mismatches[0]<<",\"tangent_path_stress\":"<<mismatches[1]
      <<",\"tangent\":"<<mismatches[2]<<",\"projected_state_high\":"<<mismatches[3]
      <<",\"projected_state_low\":"<<mismatches[4]<<"},\"physical_transactions\":0,"
      <<"\"physical_validation\":false,\"full_solver_qualification\":false}\n";
    result.flush();need(result.good(),"source392 trace-run manifest write failed");
    std::cout<<"{\"record\":\"source392_tet858_trace\",\"status\":\"diagnostic_dispatch_complete\","
      <<"\"point401_sha256\":\""<<pointSHA<<"\",\"package_sha256\":\""<<packageSHA
      <<"\",\"projection_valid\":"<<(trace[0]==1.0f?"true":"false")
      <<",\"iterations\":"<<trace[3]<<",\"failure\":"<<trace[5]
      <<",\"trace_off_total_mismatches\":"
      <<std::accumulate(mismatches.begin(),mismatches.end(),0u)
      <<",\"physical_transactions\":0}\n";
    return 0;
}

int runExactSource380Point(const std::string& expectedInputSHA,const std::string& expectedPackageSHA,
                           const std::string& expectedFingerprint,const std::filesystem::path& inputPath,
                           const std::filesystem::path& packagePath,const std::filesystem::path& mediumPath,
                           const std::filesystem::path& candidatePath,
                           const std::filesystem::path& historyPath,
                           const std::filesystem::path& meshPath,
                           const std::filesystem::path& outdir) {
    need(expectedInputSHA.size()==64u&&expectedPackageSHA.size()==64u,
         "explicit input/package SHA binding must be 64 hexadecimal characters");
    for(const auto* value:{&expectedInputSHA,&expectedPackageSHA})
        need(std::all_of(value->begin(),value->end(),[](unsigned char c){return std::isxdigit(c)!=0;}),
             "explicit SHA binding contains a non-hexadecimal character");
    const std::string inputSHA=sha256(inputPath),packageSHA=sha256(packagePath),assetSHA=sha256(mediumPath);
    need(inputSHA==expectedInputSHA,"source380 input SHA differs from CLI binding");
    need(packageSHA==expectedPackageSHA,"compiled source380 package SHA differs from CLI binding");
    need(assetSHA==kMediumSHA,"source380 medium source asset SHA differs from exact source binding");
    NSDictionary* inputBinding=nil;const Record row=readBoundPoint(inputPath,&inputBinding);
    need(stringValue(field(inputBinding,"compiled_package_sha256"),"package SHA in input") == packageSHA,
         "point input package SHA differs from exact package bytes");
    need(stringValue(field(inputBinding,"source_medium_asset_sha256"),"medium asset SHA in input") == assetSHA,
         "point input medium SHA differs from exact source asset bytes");
    need(stringValue(field(inputBinding,"compiled_world_fingerprint"),"world fingerprint in input") == expectedFingerprint,
         "point input world fingerprint differs from CLI binding");
    const SourceArtifactBinding sourceArtifacts=
        verifySource380Artifacts(inputBinding,candidatePath,historyPath,meshPath,row);
    CompiledWorld world;std::string error;
    need(readPackage(packagePath,world,nullptr,&error),"source380 compiled package rejected: "+error);
    verifyBoundPackage(world,row,inputBinding,expectedFingerprint);
    std::filesystem::create_directories(outdir);
    Metal metal;
    const std::string policy="source380_exact_compiled_budget32";
    const auto trace=runTrace(metal,world,policy,row,outdir,inputSHA,true);
    const auto batch=runBatch(metal,world,policy,std::vector<Record>{row},outdir,inputSHA,true);
    writePointRunManifest(outdir/"source380-point-run.json",inputSHA,packageSHA,assetSHA,
                          expectedFingerprint,world,sourceArtifacts,row,trace,batch);
    std::cout<<"{\"record\":\"source380_point_diagnostic\",\"status\":\"dispatch_complete_observation_emitted\",\"input_sha256\":\""
             <<inputSHA<<"\",\"package_sha256\":\""<<packageSHA<<"\",\"world_fingerprint\":\""<<expectedFingerprint
             <<"\",\"environment\":"<<row.env<<",\"tet_id\":"<<row.tet<<",\"local_newton_iterations\":"
             <<world.materials[row.material].localNewtonIterations<<",\"projection_valid\":"<<(trace[0]==1.0f?"true":"false")
             <<",\"projection_iterations\":";numberJSON(std::cout,trace[3]);std::cout<<",\"projection_failure\":";
    numberJSON(std::cout,trace[5]);std::cout<<",\"physical_steps\":0,\"physical_validation\":false}\n";
    return 0;
}


void verifySource547EndpointInput(const Record& center,const Record& endpoint,
                                  const int epsilonPower,const int sign) {
    constexpr std::size_t basis=2u;
    need(epsilonPower>=0&&(sign==1||sign==-1),
         "source547 endpoint power/sign is invalid");
    const double epsilon=std::ldexp(1.0,-epsilonPower);
    const double dt=static_cast<double>(center.input[45u]);
    need(std::isfinite(dt)&&dt>0.0,
         "source547 center dt is nonfinite or nonpositive");
    const double targetF=static_cast<double>(center.input[basis])+
        static_cast<double>(center.input[9u+basis])+
        static_cast<double>(sign)*epsilon;
    const float expectedHigh=static_cast<float>(targetF);
    const double residualF=
        (static_cast<double>(center.input[basis])-
         static_cast<double>(expectedHigh))+
        static_cast<double>(center.input[9u+basis])+
        static_cast<double>(sign)*epsilon;
    const float expectedLow=static_cast<float>(residualF);
    const float expectedRate=static_cast<float>(
        static_cast<double>(center.input[18u+basis])+
        static_cast<double>(sign)*epsilon/dt);
    auto bits=[](const float value) {
        return std::bit_cast<std::uint32_t>(value);
    };
    need(bits(endpoint.input[basis])==bits(expectedHigh)&&
         bits(endpoint.input[9u+basis])==bits(expectedLow)&&
         bits(endpoint.input[18u+basis])==bits(expectedRate),
         "source547 endpoint F-high/F-low/Fdot words do not match its declared basis, sign, and epsilon");
    for(std::size_t i=0u;i<kInputStride;++i) {
        if(i==basis||i==9u+basis||i==18u+basis) continue;
        need(bits(endpoint.input[i])==bits(center.input[i]),
             "source547 endpoint changed an undeclared packed input component");
    }
}
struct Bound547Input {
    Capture selected;
    std::string sourceCaptureSHA;
};
Bound547Input readBound547Input(const std::filesystem::path& path) {
    NSDictionary* root=readJSONDocument(path);
    need(stringValue(field(root,"schema"),"source547 batch schema")==
         "numi.cardboard.source-bound-native-batch.v1",
         "unexpected source547 input schema");
    const std::string sourceCaptureSHA=
        stringValue(field(root,"source_capture_sha256"),"source capture SHA");
    need(sourceCaptureSHA.size()==64u&&
         std::all_of(sourceCaptureSHA.begin(),sourceCaptureSHA.end(),
             [](unsigned char c){return std::isxdigit(c)!=0;}),
         "source capture SHA must be 64 hexadecimal characters");
    NSArray* parameterRows=(NSArray*)field(root,"parameters_by_material");
    need([parameterRows isKindOfClass:[NSArray class]]&&parameterRows.count==2u,
         "source547 parameter rows must bind liner and medium");
    Bound547Input out;
    out.sourceCaptureSHA=sourceCaptureSHA;
    for(std::size_t material=0;material<2u;++material) {
        const auto values=flatVector(parameterRows[material],28u,
            "source547 material parameters");
        out.selected.parameterRow.insert(out.selected.parameterRow.end(),
            values.begin(),values.end());
    }
    NSArray* rows=(NSArray*)field(root,"rows");
    need([rows isKindOfClass:[NSArray class]]&&rows.count==1472u,
         "source547 input must preserve all 1,472 selected native rows");
    struct ExpectedRow {
        std::size_t index;
        const char* label;
        int power;
        int sign;
    };
    constexpr std::array<ExpectedRow,10> expected{{
        {256u,"center",-1,0},{257u,"center_repeat",-1,0},
        {282u,"endpoint",22,1},{283u,"endpoint",22,-1},
        {284u,"endpoint",23,1},{285u,"endpoint",23,-1},
        {286u,"endpoint",24,1},{287u,"endpoint",24,-1},
        {288u,"endpoint",25,1},{289u,"endpoint",25,-1}
    }};
    std::array<float,kStateCount> acceptedCenter{};
    bool haveAccepted=false;
    for(const auto& expectedRow:expected) {
        NSDictionary* item=(NSDictionary*)rows[expectedRow.index];
        need(number(field(item,"environment"),"source547 environment")==0.0&&
             number(field(item,"tet_id"),"source547 tet id")==222.0&&
             number(field(item,"material_index"),"source547 material index")==0.0,
             "source547 selected row no longer binds env0/tet222/liner");
        need(stringValue(field(item,"label"),"source547 label")==expectedRow.label&&
             number(field(item,"basis_column"),"source547 basis column")==2.0&&
             number(field(item,"sign"),"source547 endpoint sign")==expectedRow.sign,
             "source547 selected row label/basis/sign changed");
        if(expectedRow.power>=0)
            need(number(field(item,"epsilon_power"),"source547 epsilon power")==
                 double(expectedRow.power),
                 "source547 selected endpoint epsilon power changed");
        Record row;
        row.env=0u;row.tet=222u;row.material=0u;row.materialName=kLinerName;
        const auto input=flatArray<kInputStride>(
            field(item,"packed_input_f32"),"source547 packed input");
        const auto accepted=flatArray<kStateCount>(
            field(item,"accepted_state_f32"),"source547 accepted history");
        row.input=input;row.accepted=accepted;
        if(!haveAccepted) { acceptedCenter=accepted;haveAccepted=true; }
        else for(std::size_t i=0;i<kStateCount;++i)
            need(std::bit_cast<std::uint32_t>(accepted[i])==
                 std::bit_cast<std::uint32_t>(acceptedCenter[i]),
                 "source547 rows do not share the exact accepted history");
        out.selected.rows.push_back(row);
    }
    need(out.selected.parameterRow.size()==56u,
         "source547 parameter binding has wrong total length");
    need(out.selected.rows.size()==expected.size(),
         "source547 selected row count changed");
    for(std::size_t i=2u;i<expected.size();++i)
        verifySource547EndpointInput(out.selected.rows[0],out.selected.rows[i],
                                     expected[i].power,expected[i].sign);
    return out;
}
void runSource547StrictStateCertification(
    const std::filesystem::path& inputPath,const std::string& expectedInputSHA,
    const std::filesystem::path& linerPath,const std::filesystem::path& mediumPath,
    const std::filesystem::path& outdir,const bool native) {
    need(expectedInputSHA.size()==64u&&
         std::all_of(expectedInputSHA.begin(),expectedInputSHA.end(),
             [](unsigned char c){return std::isxdigit(c)!=0;}),
         "expected source547 input SHA must be 64 hexadecimal characters");
    const std::string inputSHA=sha256(inputPath);
    need(inputSHA==expectedInputSHA,
         "source547 input SHA differs from explicit command binding");
    need(sha256(linerPath)==kLinerSHA&&sha256(mediumPath)==kMediumSHA,
         "source paper source asset SHA differs from the fixed material binding");
    const Bound547Input input=readBound547Input(inputPath);
    const auto baseline=compileSources(linerPath,mediumPath,true,false,32u);
    const auto strict=compileSources(linerPath,mediumPath,true,true,32u);
    verifyParameters(baseline,input.selected);
    verifyParameters(strict,input.selected);
    const std::uint32_t accurateFlag=
        NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION;
    const std::uint32_t strictFlag=
        NM_MATERIAL_GENERIC_BILINEAR15_STRICT_STATE_CERTIFICATION;
    for(std::size_t i=0;i<2u;++i) {
        need(baseline.materials[i].localNewtonIterations==32u&&
             (baseline.materials[i].flags&accurateFlag)!=0u&&
             (baseline.materials[i].flags&strictFlag)==0u,
             "source547 baseline is not accurate bilinear15 with explicit32 and no strict certificate");
        need(strict.materials[i].localNewtonIterations==32u&&
             (strict.materials[i].flags&accurateFlag)!=0u&&
             (strict.materials[i].flags&strictFlag)!=0u,
             "source547 strict build omitted policy or changed the local iteration budget");
    }
    need(baseline.fingerprint!=strict.fingerprint,
         "source547 strict policy did not change the compiled-world fingerprint");
    if(!native) {
        std::cout<<"{\"record\":\"source547_strict_state_cpu_admission\","
                 "\"status\":\"pass\",\"input_sha256\":\""<<inputSHA
                 <<"\",\"source_capture_sha256\":\""<<input.sourceCaptureSHA
                 <<"\",\"selected_rows\":10,\"baseline_iterations\":32,"
                 "\"strict_iterations\":32,\"strict_scaled_correction_tolerance\":2e-10,"
                 "\"physical_gpu_execution\":false}\n";
        return;
    }
    need(!std::filesystem::exists(outdir),
         "source547 output directory already exists; refusing evidence overwrite");
    need(std::filesystem::create_directories(outdir),
         "could not create source547 native diagnostic output directory");
    Metal metal;
    const auto baselineBatch=runBatch(
        metal,baseline,"accurate32",input.selected.rows,outdir,inputSHA,true);
    const auto strictBatch=runBatch(
        metal,strict,"strict_state_2e-10_budget32",input.selected.rows,outdir,inputSHA,true);
    const auto centerIt=std::find_if(input.selected.rows.begin(),input.selected.rows.end(),
        [](const Record& row) {
            return row.env==0u&&row.tet==222u&&row.material==0u&&
                row.input[27u]==0.0f&&row.input[28u]==0.0f&&
                row.input[29u]==1.0f&&row.input[30u]==0.0f;
        });
    const auto endpointIt=input.selected.rows.size()>4u
        ? input.selected.rows.begin()+4u : input.selected.rows.end();
    need(centerIt!=input.selected.rows.end(),"source547 tet222 center row missing");
    need(endpointIt!=input.selected.rows.end()&&endpointIt->tet==222u&&
         endpointIt->input[27u]==0.0f&&endpointIt->input[28u]==0.0f&&
         endpointIt->input[29u]==1.0f&&
         endpointIt->input[2u]!=centerIt->input[2u]&&
         endpointIt->input[20u]!=centerIt->input[20u],
         "source547 selected p23 endpoint row identity changed");
    const auto baselineCenterTrace=runTrace(
        metal,baseline,"accurate32_center",*centerIt,outdir,inputSHA,true);
    const auto strictCenterTrace=runTrace(
        metal,strict,"strict_state_2e-10_budget32_center",*centerIt,outdir,inputSHA,true);
    const auto baselineTrace=runTrace(
        metal,baseline,"accurate32_p23plus",*endpointIt,outdir,inputSHA,true);
    const auto strictTrace=runTrace(
        metal,strict,"strict_state_2e-10_budget32_p23plus",*endpointIt,outdir,inputSHA,true);
    const std::array<std::pair<unsigned,unsigned>,5> spans={{{14u,8u},{23u,17u},
        {32u,26u},{41u,35u},{56u,50u}}};
    const std::array<unsigned,5> lengths={9u,9u,9u,15u,15u};
    auto mismatchCounts=[&](const std::vector<float>& trace,
                            const std::vector<float>& batch) {
        std::array<unsigned,5> mismatch{};
        for(std::size_t span=0;span<spans.size();++span)
            for(unsigned j=0;j<lengths[span];++j)
                mismatch[span]+=std::bit_cast<std::uint32_t>(
                    trace[spans[span].first+j])!=
                    std::bit_cast<std::uint32_t>(
                    batch[spans[span].second+j]);
        return mismatch;
    };
    const auto baselineMismatch=mismatchCounts(
        baselineTrace,baselineBatch);
    const auto strictMismatch=mismatchCounts(strictTrace,strictBatch);
    const auto traceStatus=static_cast<int>(strictTrace[
        kTraceSummarySlots+kTraceHeaderFailureSlot]);
    std::ofstream f(outdir/"source547-strict-state-certification.json");
    need(f.good(),"could not write source547 strict-state summary");
    f<<"{\"schema\":\"numi.cardboard.source547-strict-state-certification.v1\","
     <<"\"input_sha256\":\""<<inputSHA<<"\","
     <<"\"source_capture_sha256\":\""<<input.sourceCaptureSHA<<"\","
     <<"\"liner_sha256\":\""<<sha256(linerPath)<<"\","
     <<"\"medium_sha256\":\""<<sha256(mediumPath)<<"\","
     <<"\"baseline_world_fingerprint\":"<<baseline.fingerprint<<","
     <<"\"strict_world_fingerprint\":"<<strict.fingerprint<<","
     <<"\"baseline_flags_liner\":"<<baseline.materials[0].flags<<","
     <<"\"strict_flags_liner\":"<<strict.materials[0].flags<<","
     <<"\"baseline_local_iterations\":32,\"strict_local_iterations\":32,"
     <<"\"baseline_scaled_correction_tolerance\":2e-8,"
     <<"\"strict_scaled_correction_tolerance\":2e-10,"
     <<"\"projection_residual_tolerance_unchanged\":2e-8,"
     <<"\"outer_residual_and_correction_gates_changed\":false,"
     <<"\"rows\":10,\"selected_point\":\"env0/tet222/liner/basis2\","
     <<"\"center_trace_iterations_baseline\":";
    numberJSON(f,baselineCenterTrace[3]);
    f<<",\"center_trace_iterations_strict\":";
    numberJSON(f,strictCenterTrace[3]);
    f<<",\"center_strict_project_valid\":"
     <<(strictCenterTrace[0]==1.0f?"true":"false")
     <<",\"p23plus_trace_iterations_baseline\":";
    numberJSON(f,baselineTrace[3]);
    f<<",\"p23plus_trace_iterations_strict\":";
    numberJSON(f,strictTrace[3]);
    f<<",\"p23plus_strict_project_valid\":"
     <<(strictTrace[0]==1.0f?"true":"false")
      <<",\"p23plus_strict_projection_residual\":";
    numberJSON(f,strictTrace[1]);
    f<<",\"p23plus_strict_projection_failure\":";
    numberJSON(f,strictTrace[5]);
    f<<",\"trace_status_code\":"<<traceStatus
      <<",\"trace_off_parity_mismatch_counts\":{"
      <<"\"baseline\":["<<baselineMismatch[0]<<","<<baselineMismatch[1]<<","
      <<baselineMismatch[2]<<","<<baselineMismatch[3]<<","<<baselineMismatch[4]<<"],"
      <<"\"strict\":["<<strictMismatch[0]<<","<<strictMismatch[1]<<","
      <<strictMismatch[2]<<","<<strictMismatch[3]<<","<<strictMismatch[4]<<"]},"
      <<"\"physical_transactions\":0,\"physical_qualification\":false,"
      <<"\"interpretation\":\"bounded native local-projection comparison only; strict policy enforces a smaller dimensionless Newton-correction certificate without changing the residual gate, iteration budget, equations, or outer criteria\"}\n";
    need(f.good(),"source547 strict-state summary write failed");
    std::cout<<"{\"record\":\"source547_strict_state_native_observation\","
             "\"status\":\"diagnostic_complete\",\"input_sha256\":\""<<inputSHA
             <<"\",\"source_capture_sha256\":\""<<input.sourceCaptureSHA
             <<"\",\"strict_projection_valid\":"<<(strictTrace[0]==1.0f?"true":"false")
             <<",\"strict_projection_iterations\":";numberJSON(std::cout,strictTrace[3]);
    std::cout<<",\"physical_transactions\":0,\"physical_qualification\":false}\n";
}

}

int main(int argc,char** argv){@autoreleasepool{try{
    const bool source547CPU=argc==6&&std::string(argv[1])=="--cpu-source547-strict-projection";
    const bool source547Native=argc==7&&std::string(argv[1])=="--native-source547-strict-projection";
    if(source547CPU||source547Native) {
        runSource547StrictStateCertification(
            argv[2],argv[3],argv[4],argv[5],
            source547Native?std::filesystem::path(argv[6]):std::filesystem::path{},
            source547Native);
        return 0;
    }
    const bool native392Trace=argc==6&&std::string(argv[1])=="--native-source392-tet858-trace";
    if(native392Trace)
        return runExactSource392Tet858Trace(argv[2],argv[3],argv[4],argv[5]);
    const bool nativeExactPoint=argc==15&&std::string(argv[1])=="--native-source380-point"&&
        std::string(argv[2])=="--expected-input-sha256"&&std::string(argv[4])=="--expected-package-sha256"&&
        std::string(argv[6])=="--expected-world-fingerprint";
    if(nativeExactPoint)
        return runExactSource380Point(argv[3],argv[5],argv[7],argv[8],argv[9],argv[10],
                                      argv[11],argv[12],argv[13],argv[14]);
    const bool cpu=argc==7&&std::string(argv[1])=="--cpu-admission"&&std::string(argv[2])=="--expected-capture-sha256";
    const bool nativeSelected=argc==9&&std::string(argv[1])=="--native"&&std::string(argv[2])=="--selected-only"&&std::string(argv[3])=="--expected-capture-sha256";
    const bool nativeAll=argc==9&&std::string(argv[1])=="--native"&&std::string(argv[2])=="--all-points"&&std::string(argv[3])=="--expected-capture-sha256";
    const bool nativeSweep=argc==9&&std::string(argv[1])=="--native"&&std::string(argv[2])=="--derivative-sweep"&&std::string(argv[3])=="--expected-capture-sha256";
    const bool native=nativeSelected||nativeAll||nativeSweep;
    need(source547CPU||source547Native||cpu||native||nativeExactPoint||native392Trace,"usage: capture-diagnostic --cpu-source547-strict-projection INPUT.json SHA LINER.nmatter MEDIUM.nmatter | --native-source547-strict-projection INPUT.json SHA LINER.nmatter MEDIUM.nmatter OUTPUT_DIR | --cpu-admission --expected-capture-sha256 SHA CAPTURE LINER.nmatter MEDIUM.nmatter | --native --selected-only|--all-points|--derivative-sweep --expected-capture-sha256 SHA CAPTURE LINER.nmatter MEDIUM.nmatter OUTPUT_DIR | --native-source380-point --expected-input-sha256 SHA --expected-package-sha256 SHA --expected-world-fingerprint DEC INPUT.json PACKAGE.nmatterpack MEDIUM.nmatter CANDIDATE_NODES.bin ACCEPTED_HISTORY.json MESH.json OUTPUT_DIR | --native-source392-tet858-trace POINT401.json RUN392_DIR MEDIUM.nmatter OUTPUT_DIR");
    const std::string expectedCaptureSHA=argv[cpu?3:4];
    need(expectedCaptureSHA.size()==64u&&std::all_of(expectedCaptureSHA.begin(),expectedCaptureSHA.end(),[](unsigned char c){return std::isxdigit(c)!=0;}),
         "expected capture SHA must be exactly 64 hexadecimal characters");
    const auto capturePath=std::filesystem::path(argv[cpu?4:5]);
    const auto linerPath=std::filesystem::path(argv[cpu?5:6]);const auto mediumPath=std::filesystem::path(argv[cpu?6:7]);
    const auto capSHA=sha256(capturePath),linerSHA=sha256(linerPath),mediumSHA=sha256(mediumPath);
    need(capSHA==expectedCaptureSHA,"capture SHA differs from explicit source-bound CLI binding");
    need(linerSHA==kLinerSHA&&mediumSHA==kMediumSHA,"source paper material SHA mismatch");
    const Capture capture=readCapture(capturePath);
    const CompiledWorld defaults=compileSources(linerPath,mediumPath,false);
    const CompiledWorld accurate=compileSources(linerPath,mediumPath,true);
    verifyParameters(defaults,capture);verifyParameters(accurate,capture);
    if(cpu){
        need(derivativeSweepRecords(capture).size()==960u,"controlled sweep CPU admission failed");
        std::cout<<"{\"record\":\"cpu_admission\",\"schema\":\"numi.cardboard.bilinear15-capture-diagnostic.v1\",\"status\":\"pass\",\"capture_sha256\":\""<<capSHA
         <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA
         <<"\",\"liner_sha256\":\""<<linerSHA<<"\",\"medium_sha256\":\""<<mediumSHA
         <<"\",\"paper_row_count\":"<<capture.rows.size()<<",\"selected_liner_tet1528\":true,\"default_iterations_liner\":"<<defaults.materials[0].localNewtonIterations
         <<",\"default_iterations_medium\":"<<defaults.materials[1].localNewtonIterations<<",\"accurate_iterations_liner\":"<<accurate.materials[0].localNewtonIterations
         <<",\"accurate_iterations_medium\":"<<accurate.materials[1].localNewtonIterations<<",\"physical_native_execution\":false}\n";return 0;
    }
    const std::filesystem::path outdir(argv[8]);std::filesystem::create_directories(outdir);
    Metal metal;
    if(nativeSweep){
        runDerivativeSweep(metal,defaults,"default",capture,outdir,capSHA);
        runDerivativeSweep(metal,accurate,"accurate",capture,outdir,capSHA);
        std::cout<<"{\"record\":\"end\",\"status\":\"controlled_derivative_diagnostic_complete\",\"physical_steps\":0,\"physical_validation\":false}\n";
        return 0;
    }
    std::cout<<"{\"record\":\"native_instrument\",\"device\":\""<<escaped([[metal.device name] UTF8String])
      <<"\",\"base_snapshot_manifest_sha256\":\""<<kBaseSnapshotSHA<<"\",\"capture_sha256\":\""<<capSHA
      <<"\",\"execution_scope\":\""<<(nativeAll?"selected_then_all_points":"selected_point_only")
      <<"\",\"paper_rows\":"<<capture.rows.size()<<",\"steps\":0,\"physical_validation\":false}\n";
    const auto it=std::find_if(capture.rows.begin(),capture.rows.end(),[](const Record&r){return r.env==0u&&r.tet==1528u&&r.material==0u;});
    need(it!=capture.rows.end(),"selected liner tet1528 disappeared");
    // Run the selected failure point first so a full 3,456-row replay cannot
    // delay the immediate native projection diagnosis.
    runTrace(metal,defaults,"default",*it,outdir,capSHA);
    runTrace(metal,accurate,"accurate",*it,outdir,capSHA);
    if(nativeAll){
        runBatch(metal,defaults,"default",capture.rows,outdir,capSHA);
        runBatch(metal,accurate,"accurate",capture.rows,outdir,capSHA);
    }
    std::cout<<"{\"record\":\"end\",\"status\":\"diagnostic_complete\",\"paper_rows_per_policy\":"<<capture.rows.size()
      <<",\"execution_scope\":\""<<(nativeAll?"selected_then_all_points":"selected_point_only")
      <<"\",\"selected_point\":\"env0/tet1528/liner\",\"physical_steps\":0,\"physical_validation\":false}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL source-bound bilinear15 capture diagnostic: "<<e.what()<<'\n';return 1;}}}
