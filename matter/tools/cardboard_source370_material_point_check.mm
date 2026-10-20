#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>

#include "numi/matter/matter.hpp"
#include "numi/matter/detail.hpp"
#include "numi/matter/shared.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef NUMI_CARDBOARD_PAPER_METALLIB
#error "This checker must bind its diagnostic metallib"
#endif
#ifndef NUMI_CARDBOARD_SOURCE107_MANIFEST_SHA256
#error "This checker must bind the source107 snapshot manifest"
#endif

namespace {
using namespace numi::matter;
constexpr std::uint32_t kTargetTet=674u;
constexpr std::uint32_t kTargetMaterial=1u;
constexpr char kMediumName[]="hajali2009_medium_hill_bilinear_fit";
constexpr char kPackageSHA[]="cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c";
constexpr char kCandidateNodesSHA[]="1ec2da4c997fac19986f5ad196c01dc653457297ae5da767f891d0bd05ea4b92";
constexpr char kConditioningManifestSHA[]="2caec04a94e9349d6cb9271a992dcbc9b831037c208cfde8c8c61caeff363b81";
constexpr char kAcceptedStateSHA[]="6f1b2031dc933cacec54142e9521e914d30a5f8f2889d6bd589e46878c3bfbbc";
constexpr char kRunManifestSHA[]="053a6ce46f8ab96a39413e0800fe89e1e27a2cf7d9d3b847d0b099528f1baba3";
constexpr char kMediumSHA[]="76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9";

void need(const bool ok,const std::string& message) {
    if(!ok) throw std::runtime_error(message);
}
std::string sha256(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    need(input.good(),"cannot read input for SHA-256: "+path.string());
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)),{});
    need(bytes.size()<=static_cast<std::size_t>(UINT32_MAX),"SHA-256 input exceeds CommonCrypto bound");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    need(CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),digest)!=nullptr,
         "CommonCrypto SHA-256 failed");
    static constexpr char hex[]="0123456789abcdef";
    std::string result;result.reserve(64u);
    for(const unsigned char b:digest){result.push_back(hex[b>>4]);result.push_back(hex[b&15u]);}
    return result;
}
id readJSON(const std::filesystem::path& path) {
    NSData* data=[NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:path.string().c_str()]];
    need(data!=nil,"cannot read JSON input: "+path.string());
    NSError* error=nil;
    id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    need(value!=nil,"cannot parse JSON input: "+path.string());
    return value;
}
NSDictionary* object(id value,const std::string& label) {
    NSDictionary* result=[value isKindOfClass:[NSDictionary class]]?(NSDictionary*)value:nil;
    need(result!=nil,label+" must be a JSON object");
    return result;
}
id field(NSDictionary* value,const char* key) {
    id result=[value objectForKey:[NSString stringWithUTF8String:key]];
    need(result!=nil,std::string("missing JSON field ")+key);
    return result;
}
double number(id value,const std::string& label) {
    need([value isKindOfClass:[NSNumber class]],label+" must be numeric");
    const double result=[(NSNumber*)value doubleValue];
    need(std::isfinite(result),label+" is nonfinite");
    return result;
}
std::uint64_t unsignedInteger(id value,const std::string& label) {
    need([value isKindOfClass:[NSNumber class]],label+" must be an integer");
    const NSNumber* numberValue=(NSNumber*)value;
    const double asDouble=[numberValue doubleValue];
    need(std::isfinite(asDouble)&&asDouble>=0.0&&std::floor(asDouble)==asDouble,
         label+" must be an unsigned integer");
    return [numberValue unsignedLongLongValue];
}
std::string stringValue(id value,const std::string& label) {
    need([value isKindOfClass:[NSString class]],label+" must be a string");
    return [(NSString*)value UTF8String];
}
template<std::size_t N>
std::array<float,N> floatArray(id value,const std::string& label) {
    need([value isKindOfClass:[NSArray class]],label+" must be an array");
    NSArray* values=(NSArray*)value;
    need(values.count==N,label+" has the wrong length");
    std::array<float,N> result{};
    for(std::size_t i=0u;i<N;++i)
        result[i]=static_cast<float>(number(values[i],label));
    return result;
}
template<class T>
std::vector<T> readRawVector(const std::filesystem::path& path,const std::size_t count) {
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    need(input.good(),"cannot open raw native input: "+path.string());
    const auto bytes=input.tellg();
    need(bytes>=0&&static_cast<std::uint64_t>(bytes)==count*sizeof(T),
         "raw native input has unexpected byte count: "+path.string());
    input.seekg(0);
    std::vector<T> result(count);
    input.read(reinterpret_cast<char*>(result.data()),static_cast<std::streamsize>(count*sizeof(T)));
    need(input.good(),"short raw native input read: "+path.string());
    return result;
}
struct alignas(16) LegacyNMFEMNodeStateGPUv45 {
    nm_float4 positionAndMass;
    nm_float4 velocityAndInverseMass;
    nm_float4 restAndFixed;
    nm_float4 deltaVelocity;
    nm_float4 referenceDisplacementAndMode;
};
static_assert(sizeof(LegacyNMFEMNodeStateGPUv45) == 80u);
static_assert(sizeof(NMFEMNodeStateGPU) == 96u);

std::vector<NMFEMNodeStateGPU> readLegacyV45NodeStates(
    const std::filesystem::path& path, const std::size_t count) {
    const auto legacy = readRawVector<LegacyNMFEMNodeStateGPUv45>(path, count);
    std::vector<NMFEMNodeStateGPU> current(count);
    for (std::size_t i = 0u; i < count; ++i) {
        auto& dst = current[i];
        const auto& src = legacy[i];
        dst.positionAndMass = src.positionAndMass;
        dst.velocityAndInverseMass = src.velocityAndInverseMass;
        dst.restAndFixed = src.restAndFixed;
        dst.deltaVelocity = src.deltaVelocity;
        dst.referenceDisplacementAndMode = src.referenceDisplacementAndMode;
        dst.referenceDisplacementLow = {0.0f, 0.0f, 0.0f, 0.0f};
    }
    return current;
}
template<class T>
id<MTLBuffer> makeBuffer(id<MTLDevice> device,const std::vector<T>& values) {
    need(!values.empty(),"refusing to create an empty Metal buffer");
    return [device newBufferWithBytes:values.data()
        length:sizeof(T)*values.size() options:MTLResourceStorageModeShared];
}
std::vector<float> compiledParameters(const CompiledWorld& world) {
    std::vector<float> result;result.reserve(world.parameters.size());
    for(const NMParameterRangeGPU& parameter:world.parameters)
        result.push_back(parameter.valueAndBounds.x);
    return result;
}
bool sameFloat4Bits(const nm_float4 a,const nm_float4 b) {
    return std::bit_cast<std::uint32_t>(a.x)==std::bit_cast<std::uint32_t>(b.x)&&
        std::bit_cast<std::uint32_t>(a.y)==std::bit_cast<std::uint32_t>(b.y)&&
        std::bit_cast<std::uint32_t>(a.z)==std::bit_cast<std::uint32_t>(b.z)&&
        std::bit_cast<std::uint32_t>(a.w)==std::bit_cast<std::uint32_t>(b.w);
}
CompiledWorld compileMaterial(const std::filesystem::path& path,
                              const std::uint32_t localIterations) {
    ParseResult parsed=parseMatterFile(path.string());
    need(parsed.succeeded(),"medium source material parse failed");
    MaterialProgram material=parsed.material;
    need(material.name==kMediumName&&material.internalState.size()==15u&&
         material.parameters.size()==28u,"medium asset is not the captured 15-state/28-parameter source");
    material.paperCompensatedArithmetic=true;
    material.genericCompensatedImplicitProjection=true;
    material.genericBilinear15AccurateProjection=true;
    WorldSource source;
    source.materials={material};source.gravity={0.0,0.0,0.0};
    source.environmentCount=1u;source.frameTimestep=1.0e-4;
    ObjectSource objectSource;
    objectSource.name="source370_exact_rejected_material_point";
    objectSource.representation=Representation::fem;
    objectSource.deformableContact=false;objectSource.deformableSelfContact=false;
    objectSource.mixedFEM=false;
    objectSource.femNodes={{0.0,0.0,0.0},{0.01,0.0,0.0},
                           {0.0,0.01,0.0},{0.0,0.0,0.01}};
    objectSource.tetrahedra={{0u,1u,2u,3u}};
    objectSource.characteristicLength=0.01;
    source.objects={objectSource};
    CompileOptions options;
    options.maximumRateExponent=0u;options.emitSpecializedMetal=false;
    options.localMaterialNewtonIterations=32u;
    options.accurateBilinear15LocalNewtonIterations=localIterations;
    CompileResult compiled=compileWorld(source,options);
    const std::string diagnostic=compiled.diagnostics.empty()?"no compiler diagnostic":
        compiled.diagnostics.front().message;
    need(compiled.succeeded(),"source material point compile failed: "+diagnostic);
    need(compiled.world.materials.size()==1u&&compiled.world.parameters.size()==28u&&
         compiled.world.materials[0].stateCount==15u&&
         compiled.world.materials[0].localNewtonIterations==localIterations,
         "compiled material descriptor did not preserve the requested 15-state local budget");
    const NMMaterialGPU& gpu=compiled.world.materials[0];
    const std::uint32_t required=NM_MATERIAL_PAPER_COMPENSATED_ARITHMETIC|
        NM_MATERIAL_GENERIC_COMPENSATED_IMPLICIT_PROJECTION|
        NM_MATERIAL_GENERIC_BILINEAR15_ACCURATE_PROJECTION;
    need((gpu.flags&required)==required,"compiled material omitted an exact source370 policy flag");
    return std::move(compiled.world);
}
void verifyRunIdentity(const std::filesystem::path& run,
                       const std::filesystem::path& medium,
                       CompiledWorld& package,
                       std::vector<NMFEMNodeStateGPU>& candidateNodes,
                       std::array<float,15>& acceptedState,
                       NMTetrahedronGPU& tetrahedron,
                       NMContinuumObjectGPU& objectValue) {
    static_assert(sizeof(LegacyNMFEMNodeStateGPUv45)==80u);
    static_assert(sizeof(NMFEMNodeStateGPU)==96u);
    const auto output=run/"output";
    const auto packagePath=output/"compiled.nmatterpack";
    const auto candidatePath=output/"rejected-conditioning-000002/candidate_nodes.bin";
    const auto conditioningManifestPath=output/"rejected-conditioning-000002/manifest.json";
    const auto statePath=output/"material_state_000002_bent.json";
    const auto manifestPath=output/"manifest.json";
    need(sha256(packagePath)==kPackageSHA,"370 compiled package hash mismatch");
    need(sha256(candidatePath)==kCandidateNodesSHA,"370 rejected candidate node buffer hash mismatch");
    need(sha256(conditioningManifestPath)==kConditioningManifestSHA,
         "370 rejected-candidate conditioning manifest hash mismatch");
    need(sha256(statePath)==kAcceptedStateSHA,"370 accepted-state export hash mismatch");
    need(sha256(manifestPath)==kRunManifestSHA,"370 run manifest hash mismatch");
    need(sha256(medium)==kMediumSHA,"medium source asset hash mismatch");
    std::string packageError;
    need(readPackage(packagePath,package,nullptr,&packageError),
         "370 compiled package could not be read/validated: "+packageError);
    need(package.fingerprint==8485105279886344921ull,
         "370 package fingerprint differs from the retained run manifest");
    NSDictionary* manifest=object(readJSON(manifestPath),"run manifest");
    need(unsignedInteger(field(manifest,"compiled_world_fingerprint"),"compiled world fingerprint")==
         package.fingerprint,"run manifest/package fingerprint mismatch");
    NSDictionary* solver=object(field(manifest,"solver"),"run solver settings");
    need(std::abs(number(field(solver,"dt_s"),"run timestep")-1.0e-4)<=1.0e-15,
         "370 source point timestep differs from the exact run manifest");
    need(unsignedInteger(field(solver,"local_material_newton_iterations"),
                         "requested local material iteration budget")==32u,
         "370 run manifest no longer records the requested local budget of 32");
    need(unsignedInteger(field(solver,"maximum_rate_exponent"),"maximum rate exponent")==0u,
         "370 run manifest does not use the captured zero-rate-exponent material inputs");
    need([field(solver,"paper_compensated_arithmetic") isKindOfClass:[NSNumber class]]&&
         [(NSNumber*)field(solver,"paper_compensated_arithmetic") boolValue],
         "370 run manifest does not enable compensated source-paper arithmetic");
    NSDictionary* sourcePolicy=object(field(manifest,"compensated_source_paper_projection"),
                                      "source paper projection policy");
    need([field(sourcePolicy,"enabled") isKindOfClass:[NSNumber class]]&&
         [(NSNumber*)field(sourcePolicy,"enabled") boolValue]&&
         [field(sourcePolicy,"accurate_bilinear15_projection") isKindOfClass:[NSNumber class]]&&
         [(NSNumber*)field(sourcePolicy,"accurate_bilinear15_projection") boolValue],
         "370 run manifest does not select the accurate source-15 policy");
    NSDictionary* effectiveFlags=object(field(sourcePolicy,"effective_compiled_flags"),
                                        "effective compiled material flags");
    need(unsignedInteger(field(effectiveFlags,"medium_flags"),"medium flags")==213u&&
         unsignedInteger(field(effectiveFlags,"medium_local_newton_iterations"),
                         "medium compiled local iteration budget")==32u,
         "370 manifest's actual compiled medium policy differs from the package");
    NSDictionary* conditioningManifest=object(readJSON(conditioningManifestPath),
                                              "rejected-candidate conditioning manifest");
    need(unsignedInteger(field(conditioningManifest,"environment_count"),
                         "conditioning environment count")==2u&&
         unsignedInteger(field(conditioningManifest,"fem_node_count"),
                         "conditioning FEM node count")==882u&&
         stringValue(field(conditioningManifest,"source_physics_fingerprint"),
                     "conditioning physics fingerprint")==
             std::to_string(package.fingerprint),
         "370 candidate conditioning manifest does not bind the compiled two-environment source");
    need(number(field(conditioningManifest,"accepted_physical_state"),
                "conditioning accepted-state flag")==0.0,
         "370 candidate conditioning record is unexpectedly marked as accepted physics");
    NSDictionary* mesh=object(field(manifest,"mesh"),"run mesh descriptor");
    need(unsignedInteger(field(mesh,"nodes"),"nodes per environment")==882u&&
         unsignedInteger(field(mesh,"tetrahedra"),"tetrahedra per environment")==1920u,
         "370 run manifest mesh dimensions differ from the retained point inputs");
    need(package.dispatch.environmentCount==2u&&package.dispatch.femNodeCount==882u&&
         package.dispatch.tetrahedronCount==1920u&&
         package.materials.size()==3u&&package.objects.size()==1u,
         "370 compiled world dimensions differ from the sealed coupon input");
    need(package.materials[kTargetMaterial].stateCount==15u&&
         package.materials[kTargetMaterial].parameterCount==28u&&
         package.materials[kTargetMaterial].localNewtonIterations==32u&&
         package.materials[kTargetMaterial].flags==213u,
         "370 package does not contain the recorded accurate medium descriptor");
    candidateNodes=readLegacyV45NodeStates(candidatePath,
        static_cast<std::size_t>(package.dispatch.environmentCount)*
        package.dispatch.femNodeCount);
    NSDictionary* stateDocument=object(readJSON(statePath),"accepted-state export");
    need(number(field(stateDocument,"environment"),"state environment")==0.0&&
         number(field(stateDocument,"status_code"),"state status")==5.0,
         "370 state export is not the rolled-back env0 rejected transaction");
    NSArray* tetrahedra=(NSArray*)field(stateDocument,"tetrahedra");
    need(tetrahedra.count==package.dispatch.tetrahedronCount,
         "370 accepted-state tetrahedron count differs from package");
    NSDictionary* row=object(tetrahedra[kTargetTet],"target accepted material state");
    need(number(field(row,"index"),"state tet index")==double(kTargetTet)&&
         number(field(row,"material_index"),"state material index")==double(kTargetMaterial),
         "370 accepted material state row identity mismatch");
    acceptedState=floatArray<15>(field(row,"state"),"accepted material state");
    tetrahedron=package.fem.tetrahedra.at(kTargetTet);
    need(tetrahedron.identity.x==kTargetMaterial&&tetrahedron.identity.y<package.objects.size()&&
         tetrahedron.nodes.x==228u&&tetrahedron.nodes.y==524u&&
         tetrahedron.nodes.z==522u&&tetrahedron.nodes.w==525u,
         "370 compiled target tetrahedron does not match its retained identity/connectivity");
    need((tetrahedron.identity.w&NM_TETRAHEDRON_REFERENCE_DISPLACEMENT_GRADIENT)!=0u,
         "370 target tetrahedron is not on the compiled reference-displacement deformation path");
    objectValue=package.objects[tetrahedron.identity.y];
    need((objectValue.flags&NM_OBJECT_FEM_MATERIAL_FRAME)!=0u,
         "370 target object omitted the captured material frame policy");
}
void verifyParameterBinding(const CompiledWorld& sourceWorld,
                            const CompiledWorld& package,
                            const std::uint32_t expectedSourceBudget) {
    need(expectedSourceBudget==16u||expectedSourceBudget==32u,
         "source point comparison budget is outside the declared 16/32 diagnostic set");
    need(sourceWorld.parameters.size()==28u,"source point compile parameter count changed");
    const auto offset=package.materials[kTargetMaterial].parameterOffset;
    need(offset+28u<=package.parameters.size(),"package medium parameter slice is out of range");
    for(std::uint32_t i=0u;i<28u;++i) {
        const float source=sourceWorld.parameters[i].valueAndBounds.x;
        const float captured=package.parameters[offset+i].valueAndBounds.x;
        need(std::bit_cast<std::uint32_t>(source)==std::bit_cast<std::uint32_t>(captured),
             "source asset parameter bits differ from the exact370 package at index "+std::to_string(i));
    }
    const auto& captured=package.materials[kTargetMaterial];
    const auto& source=sourceWorld.materials[0];
    need(source.localNewtonIterations==expectedSourceBudget,
         "private source compile did not preserve its explicitly requested local budget");
    need(captured.localNewtonIterations==32u,
         "immutable source-380 descriptor no longer records its actual 32-update budget");
    need(captured.stateCount==source.stateCount&&captured.parameterCount==source.parameterCount&&
         captured.flags==source.flags&&captured.constitutiveKind==source.constitutiveKind&&
         captured.projectionKind==source.projectionKind&&
         captured.stateTransferMask==source.stateTransferMask&&
         sameFloat4Bits(captured.bulk,source.bulk)&&
         sameFloat4Bits(captured.interfaceResponse,source.interfaceResponse)&&
         sameFloat4Bits(captured.inelastic,source.inelastic)&&
         sameFloat4Bits(captured.validity,source.validity),
         "source-only compiled material policy differs from the exact370 package descriptor");
    need(captured.stateInitialOffset+captured.stateCount<=package.stateInitials.size()&&
         source.stateInitialOffset+source.stateCount<=sourceWorld.stateInitials.size(),
         "source or captured material initial-state slice is outside its package");
    for(std::uint32_t i=0u;i<source.stateCount;++i)
        need(std::bit_cast<std::uint32_t>(
                 sourceWorld.stateInitials[source.stateInitialOffset+i])==
             std::bit_cast<std::uint32_t>(
                 package.stateInitials[captured.stateInitialOffset+i]),
             "source material initial-state word differs from exact370 package at index "+
             std::to_string(i));
}
struct NativeRun {
    id<MTLDevice> device=nil;
    id<MTLCommandQueue> queue=nil;
    id<MTLComputePipelineState> pipeline=nil;
    NativeRun() {
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");
        const std::string name=[[device name] UTF8String];
        need(name.find("Apple")!=std::string::npos&&name.find("Paravirtual")==std::string::npos,
             "physical Apple GPU required; no native point evaluation was dispatched");
        queue=[device newCommandQueue];need(queue!=nil,"Metal command queue unavailable");
        NSError* error=nil;
        id<MTLLibrary> library=[device newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&error];
        need(library!=nil,"source-bound paper diagnostic metallib is unavailable");
        id<MTLFunction> function=[library newFunctionWithName:
            @"cardboard_replay_source15_candidate_point"];
        need(function!=nil,"source370 native point kernel is missing from the metallib");
        pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        need(pipeline!=nil,"could not create source370 point replay pipeline");
        std::cout<<"{\"record\":\"device\",\"device_name\":\""<<name
                 <<"\",\"physical_steps\":0}\n";
    }
    std::vector<float> evaluate(const CompiledWorld& sourceWorld,
                                const NMContinuumObjectGPU& objectValue,
                                const NMTetrahedronGPU& tetrahedron,
                                const std::vector<NMFEMNodeStateGPU>& candidateNodes,
                                const std::array<float,15>& acceptedState,
                                const float dt) {
        const auto parameters=compiledParameters(sourceWorld);
        const std::vector<NMContinuumObjectGPU> objects{objectValue};
        const std::vector<NMTetrahedronGPU> tetrahedra{tetrahedron};
        const std::vector<NMMaterialGPU> materials{sourceWorld.materials.at(0)};
        const std::vector<float> accepted(acceptedState.begin(),acceptedState.end());
        const std::vector<float> config{dt,materials[0].bulk.y};
        std::vector<float> output(160u,0.0f);
        std::array<id<MTLBuffer>,10> buffers{
            makeBuffer(device,objects),makeBuffer(device,tetrahedra),
            makeBuffer(device,materials),makeBuffer(device,sourceWorld.scalarPrograms),
            makeBuffer(device,sourceWorld.instructions),makeBuffer(device,parameters),
            makeBuffer(device,candidateNodes),makeBuffer(device,accepted),
            makeBuffer(device,output),makeBuffer(device,config)};
        need(std::all_of(buffers.begin(),buffers.end(),[](id<MTLBuffer> b){return b!=nil;}),
             "source370 native replay buffer allocation failed");
        id<MTLCommandBuffer> command=[queue commandBuffer];
        need(command!=nil,"source370 native replay command buffer unavailable");
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        need(encoder!=nil,"source370 native replay encoder unavailable");
        [encoder setComputePipelineState:pipeline];
        for(NSUInteger i=0u;i<buffers.size();++i)
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1u,1u,1u)
            threadsPerThreadgroup:MTLSizeMake(1u,1u,1u)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,
             command.error?std::string([[command.error description] UTF8String]):
                 "source370 Metal point replay failed");
        const auto* values=static_cast<const float*>(buffers[8].contents);
        need(values[0]==1.0f,"source370 native point kernel did not complete");
        return std::vector<float>(values,values+160u);
    }
};
void writeBits(std::ostream& output,const float* values,const std::size_t count) {
    output<<'[';
    for(std::size_t i=0u;i<count;++i) {
        if(i)output<<',';
        output<<"\"0x"<<std::hex<<std::setw(8)<<std::setfill('0')
              <<std::bit_cast<std::uint32_t>(values[i])<<std::dec<<std::setfill(' ')<<'\"';
    }
    output<<']';
}
void writeArray(std::ostream& output,const float* values,const std::size_t count) {
    output<<'['<<std::setprecision(9);
    for(std::size_t i=0u;i<count;++i) {
        if(i)output<<',';
        if(std::isfinite(values[i]))output<<values[i];else output<<"null";
    }
    output<<']';
}
void writeResult(const std::filesystem::path& path,const std::string& policy,
                 const std::uint32_t budget,const std::vector<float>& x,
                 const float referenceTemperature,
                 const std::uint64_t fingerprint) {
    std::ofstream out(path,std::ios::out|std::ios::trunc);
    need(out.good(),"cannot create native point replay output");
    out<<std::setprecision(17)
       <<"{\"record\":\"point\",\"policy\":\""<<policy
       <<"\",\"compiled_local_iterations\":"<<budget
       <<",\"immutable_run_package_medium_iterations\":32"
       <<",\"diagnostic_descriptor_is_private_recompile\":true"
       <<",\"diagnostic_descriptor_source\":\"same source material asset and parameter words; not the immutable 380 compiled world\""
       <<",\"reference_temperature_K\":"<<referenceTemperature
       <<",\"source_material_flags\":213,\"source_world_fingerprint\":"<<fingerprint
       <<",\"direct_stress_valid\":"<<(x[2]==1.0f?"true":"false")
       <<",\"determinant\":"<<x[1]
       <<",\"tangent_stress_valid\":"<<(x[48]==1.0f?"true":"false")
       <<",\"projection_flags\":"<<x[49]
       <<",\"projection_converged\":"<<(x[55]==1.0f?"true":"false")
       <<",\"projection_residual\":"<<x[50]
       <<",\"projection_state_norm\":"<<x[51]
       <<",\"local_updates\":"<<x[52]
       <<",\"terminal_failure\":"<<x[53]
       <<",\"material_point_stress_world\":";writeArray(out,x.data()+39u,9u);
    out<<",\"tangent_path_stress_world\":";writeArray(out,x.data()+129u,9u);
    out<<",\"material_tangent_world_for_declared_unit_direction\":";writeArray(out,x.data()+147u,9u);
    out<<",\"F_high\":";writeArray(out,x.data()+3u,9u);
    out<<",\"F_low\":";writeArray(out,x.data()+12u,9u);
    out<<",\"Fdot_local\":";writeArray(out,x.data()+21u,9u);
    out<<",\"projected_state_high\":";writeArray(out,x.data()+56u,15u);
    out<<",\"projected_state_low\":";writeArray(out,x.data()+88u,15u);
    out<<",\"raw_160_float32_words\":";writeBits(out,x.data(),x.size());
    out<<",\"direction_definition\":\"row-major F direction has only (0,0)=1; Fdot direction is that entry divided by captured dt; diagnostic direction is not a physical solver correction\",\"physical_steps\":0,\"physical_validation\":false}\n";
    out.flush();need(out.good(),"failed writing native point replay output");
}
int run(const bool native,const std::filesystem::path& runPath,
        const std::filesystem::path& mediumPath,const std::filesystem::path& outputPath) {
    static_assert(sizeof(LegacyNMFEMNodeStateGPUv45)==80u);
    static_assert(sizeof(NMFEMNodeStateGPU)==96u);
    CompiledWorld package;
    std::vector<NMFEMNodeStateGPU> candidateNodes;
    std::array<float,15> acceptedState{};
    NMTetrahedronGPU tetrahedron{};
    NMContinuumObjectGPU objectValue{};
    verifyRunIdentity(runPath,mediumPath,package,candidateNodes,acceptedState,
                      tetrahedron,objectValue);
    const CompiledWorld budget16=compileMaterial(mediumPath,16u);
    const CompiledWorld budget32=compileMaterial(mediumPath,32u);
    verifyParameterBinding(budget16,package,16u);
    verifyParameterBinding(budget32,package,32u);
    need(budget16.fingerprint!=budget32.fingerprint,
         "source compiler fingerprints failed to bind 16 versus 32 local updates");
    if(!native) {
        std::cout<<"{\"record\":\"cpu_admission\",\"status\":\"pass\","
          <<"\"run_package_sha256\":\""<<kPackageSHA<<"\","
          <<"\"candidate_nodes_sha256\":\""<<kCandidateNodesSHA<<"\","
          <<"\"accepted_state_sha256\":\""<<kAcceptedStateSHA<<"\","
          <<"\"medium_asset_sha256\":\""<<kMediumSHA<<"\","
          <<"\"compiled_budgets\":[16,32],\"native_metal_execution\":false}\n";
        return 0;
    }
    need(!std::filesystem::exists(outputPath),
         "refusing to overwrite a prior native point replay directory");
    std::filesystem::create_directories(outputPath);
    NativeRun metal;
    const float dt=1.0e-4f;
    const auto result16=metal.evaluate(budget16,objectValue,tetrahedron,
                                       candidateNodes,acceptedState,dt);
    const auto result32=metal.evaluate(budget32,objectValue,tetrahedron,
                                       candidateNodes,acceptedState,dt);
    std::ofstream stream(outputPath/"point-replay.jsonl",std::ios::out|std::ios::trunc);
    need(stream.good(),"cannot create source370 point replay ledger");
    stream<<"{\"record\":\"header\",\"schema\":\"numi.cardboard.source380-native-material-point.v1\","
      <<"\"historical_base_snapshot107_manifest_sha256\":\""<<NUMI_CARDBOARD_SOURCE107_MANIFEST_SHA256
      <<"\",\"run_package_sha256\":\""<<kPackageSHA
      <<"\",\"candidate_nodes_sha256\":\""<<kCandidateNodesSHA
      <<"\",\"accepted_state_sha256\":\""<<kAcceptedStateSHA
      <<"\",\"run_manifest_sha256\":\""<<kRunManifestSHA
      <<"\",\"medium_source_sha256\":\""<<kMediumSHA
      <<"\",\"environment\":0,\"tet_id\":674,\"material_index\":1,"
      <<"\"object_timestep_s\":"<<dt
      <<",\"run_requested_local_iterations\":32,\"run_compiled_local_iterations\":32,"
      <<"\"diagnostic_budgets_are_point_only\":[16,32],"
      <<"\"budget32_is_not_a_replay_of_the_full_380_trajectory\":true,"
      <<"\"input_state_is_accepted_rollback_history\":true,"
      <<"\"deformation_source\":\"production tetrahedronDeformation + femMaterialDeformationWords over exact rejected candidate_nodes.bin\","
      <<"\"material_source\":\"source-bound medium asset recompiled with each explicit local budget; parameter words verified against the 380 package\","
      <<"\"physical_steps\":0,\"physical_validation\":false,"
      <<"\"accepted_state_f32_words\":";
    writeBits(stream,acceptedState.data(),acceptedState.size());
    stream<<"}\n";
    stream.flush();need(stream.good(),"failed writing source370 point replay header");stream.close();
    writeResult(outputPath/"budget16.json", "accurate_source_paper_projection",16u,
                result16,budget16.materials.at(0).bulk.y,budget16.fingerprint);
    writeResult(outputPath/"budget32.json", "accurate_source_paper_projection",32u,
                result32,budget32.materials.at(0).bulk.y,budget32.fingerprint);
    std::ofstream end(outputPath/"point-replay.jsonl",std::ios::out|std::ios::app);
    need(end.good(),"cannot append source370 point replay ledger");
    end<<"{\"record\":\"end\",\"status\":\"diagnostic_complete\","
       <<"\"budget16_direct_valid\":"<<(result16[2]==1.0f?"true":"false")
       <<",\"budget32_direct_valid\":"<<(result32[2]==1.0f?"true":"false")
       <<",\"interpretation\":\"native local material point only; not full-step or coupon qualification\"}\n";
    end.flush();need(end.good(),"failed closing source370 point replay ledger");
    return 0;
}
} // namespace

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            const bool cpu=argc==4&&std::string(argv[1])=="--cpu-admission";
            const bool native=argc==5&&std::string(argv[1])=="--native";
            need(cpu||native,
                "usage: source370-material-point-check --cpu-admission RUN_DIR MEDIUM.nmatter | --native RUN_DIR MEDIUM.nmatter OUTPUT_DIR");
            const std::filesystem::path runPath=argv[2];
            const std::filesystem::path mediumPath=argv[3];
            const std::filesystem::path outputPath=native?argv[4]:"";
            return run(native,runPath,mediumPath,outputPath);
        } catch(const std::exception& error) {
            std::cerr<<"FAIL source370 native material point replay: "<<error.what()<<'\n';
            return 2;
        }
    }
}
