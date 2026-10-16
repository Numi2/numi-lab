#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>

#include "numi/matter/matter.hpp"
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
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_CARDBOARD_PAPER_METALLIB
#error "Bind the diagnostic metallib built from the source-118 paper checker shader"
#endif
#ifndef NUMI_CARDBOARD_SOURCE118_BUILD_SHA256
#error "Bind the source-118 build receipt"
#endif

namespace {
using namespace numi::matter;
constexpr std::uint32_t kTet=858u;
constexpr std::uint32_t kMedium=1u;
constexpr std::uint64_t kFingerprint=8485105279886344921ull;
constexpr char kReceiptSHA[]="bc1495a5a63e35a36182b9206350ff2f68480345ae1b205484215a1d53da7bcb";
constexpr char kRunManifestSHA[]="fbfcf51db84a3710cc3bd96e016597728163da464974ea033554e82e587cd8eb";
constexpr char kResultSHA[]="214e9619e196ff45fcf97376930fcb9c017f032959075332ff65073a8c213e3d";
constexpr char kPackageSHA[]="cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c";
constexpr char kCandidateSHA[]="89846e4efc91c1774b52d6b3732f633e78414a7d341b6cbd8b9b4966e4be0417";
constexpr char kConditioningSHA[]="ad51b0ceca073a4736ee128d0daf13eee82bf3a398919cc911a81fdff8399ced";
constexpr char kStateSHA[]="e32909672f5ef53992fd7c798419dcdcd257662daa0f0c6cc8d960a1dced5dfd";
constexpr char kAssetSHA[]="76f2bef552822708c36b5acc9f92bd44e7bb92c578e5e46adb64a1a2e82e47f9";
constexpr char kSourceBuildSHA[]="da94e6051e767ea4bfba58705ccaf321e5dd97e5e9b40b32715f056676980c9f";

void need(bool ok,const std::string& message) {
    if(!ok) throw std::runtime_error(message);
}
std::string sha256(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary);
    need(in.good(),"cannot read hash input: "+path.string());
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),{});
    need(bytes.size()<=static_cast<std::size_t>(UINT32_MAX),"SHA input too large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    need(CC_SHA256(bytes.data(),static_cast<CC_LONG>(bytes.size()),digest)!=nullptr,
         "CommonCrypto SHA-256 failed");
    static constexpr char hex[]="0123456789abcdef";
    std::string out;out.reserve(64);
    for(unsigned char b:digest){out.push_back(hex[b>>4]);out.push_back(hex[b&15u]);}
    return out;
}
id readJSON(const std::filesystem::path& path) {
    NSData* data=[NSData dataWithContentsOfFile:
        [NSString stringWithUTF8String:path.string().c_str()]];
    need(data!=nil,"cannot read JSON: "+path.string());
    NSError* error=nil;
    id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    need(value!=nil,"invalid JSON: "+path.string());
    return value;
}
NSDictionary* object(id value,const std::string& label) {
    NSDictionary* out=[value isKindOfClass:[NSDictionary class]]?(NSDictionary*)value:nil;
    need(out!=nil,label+" must be an object");return out;
}
id field(NSDictionary* value,const char* key) {
    id out=[value objectForKey:[NSString stringWithUTF8String:key]];
    need(out!=nil,std::string("missing field ")+key);return out;
}
double number(id value,const std::string& label) {
    need([value isKindOfClass:[NSNumber class]],label+" must be numeric");
    double out=[(NSNumber*)value doubleValue];
    need(std::isfinite(out),label+" is nonfinite");return out;
}
std::uint64_t uintValue(id value,const std::string& label) {
    need([value isKindOfClass:[NSNumber class]],label+" must be an integer");
    NSNumber* n=(NSNumber*)value;double d=n.doubleValue;
    need(std::isfinite(d)&&d>=0&&std::floor(d)==d,label+" must be unsigned integer");
    return n.unsignedLongLongValue;
}
std::string stringValue(id value,const std::string& label) {
    need([value isKindOfClass:[NSString class]],label+" must be a string");
    return [(NSString*)value UTF8String];
}
template<class T> std::vector<T> readRaw(const std::filesystem::path& path,std::size_t count) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    need(in.good(),"cannot open raw input: "+path.string());auto size=in.tellg();
    need(size>=0&&std::uint64_t(size)==count*sizeof(T),"raw input byte size mismatch: "+path.string());
    in.seekg(0);std::vector<T> v(count);in.read(reinterpret_cast<char*>(v.data()),std::streamsize(count*sizeof(T)));
    need(in.good(),"short raw read: "+path.string());return v;
}
template<class T> id<MTLBuffer> makeBuffer(id<MTLDevice> device,const std::vector<T>& values) {
    need(!values.empty(),"refusing empty Metal buffer");
    return [device newBufferWithBytes:values.data() length:values.size()*sizeof(T)
                              options:MTLResourceStorageModeShared];
}
void writeBits(std::ostream& out,const float* values,std::size_t count) {
    out<<'[';
    for(std::size_t i=0;i<count;++i){if(i)out<<',';out<<"\"0x"<<std::hex<<std::setw(8)
        <<std::setfill('0')<<std::bit_cast<std::uint32_t>(values[i])<<std::dec<<std::setfill(' ')<<'"';}
    out<<']';
}
void writeValues(std::ostream& out,const float* values,std::size_t count) {
    out<<'['<<std::setprecision(9);
    for(std::size_t i=0;i<count;++i){if(i)out<<',';if(std::isfinite(values[i]))out<<values[i];else out<<"null";}
    out<<']';
}
std::vector<float> parameterValues(const CompiledWorld& world) {
    std::vector<float> out;out.reserve(world.parameters.size());
    for(const auto& p:world.parameters)out.push_back(p.valueAndBounds.x);return out;
}

struct Inputs {
    CompiledWorld world;
    std::vector<NMFEMNodeStateGPU> nodes;
    std::array<float,15> accepted{};
    NMTetrahedronGPU tet{};
    NMContinuumObjectGPU object{};
    float dt=0.0f;
    float temperature=0.0f;
    std::filesystem::path run;
    std::filesystem::path asset;
    std::uint32_t tetIndex=kTet;
    std::string packageSHA,candidateSHA,conditioningSHA,stateSHA,runManifestSHA,receiptSHA,assetSHA;
};

Inputs loadInputs(const std::filesystem::path& run,const std::filesystem::path& asset,
                  std::uint32_t tetIndex) {
    need(tetIndex==kTet,"this sealed diagnostic only admits source392 tet 858");
    Inputs in;in.run=run;in.asset=asset;in.tetIndex=tetIndex;
    const auto output=run/"output";
    const auto receiptPath=run/"receipt.json";
    const auto manifestPath=output/"manifest.json";
    const auto resultPath=output/"result.json";
    const auto packagePath=output/"compiled.nmatterpack";
    const auto condPath=output/"rejected-conditioning-000001/manifest.json";
    const auto nodesPath=output/"rejected-conditioning-000001/candidate_nodes.bin";
    const auto statePath=output/"material_state_000001_bent.json";
    const auto buildPath=run/"inputs/external/source-build.json";
    auto checkHash=[&](const std::filesystem::path& p,const char* expected,const char* label){
        auto actual=sha256(p);need(actual==expected,std::string(label)+" SHA-256 mismatch");return actual;};
    in.receiptSHA=checkHash(receiptPath,kReceiptSHA,"run receipt");
    in.runManifestSHA=checkHash(manifestPath,kRunManifestSHA,"run manifest");
    checkHash(resultPath,kResultSHA,"result");
    in.packageSHA=checkHash(packagePath,kPackageSHA,"compiled package");
    in.conditioningSHA=checkHash(condPath,kConditioningSHA,"conditioning manifest");
    in.candidateSHA=checkHash(nodesPath,kCandidateSHA,"candidate nodes");
    in.stateSHA=checkHash(statePath,kStateSHA,"accepted material state");
    in.assetSHA=checkHash(asset,kAssetSHA,"medium source asset");
    checkHash(buildPath,kSourceBuildSHA,"source build receipt");

    NSDictionary* receipt=object(readJSON(receiptPath),"receipt");
    need(stringValue(field(receipt,"schema"),"receipt schema")=="numi.cardboard.native-run-receipt.v2"&&
         [(NSNumber*)field(receipt,"source_inputs_unchanged") boolValue]&&
         uintValue(field(receipt,"exit_code"),"run exit code")==2u&&
         ![(NSNumber*)field(receipt,"timed_out") boolValue],"run receipt is not the retained complete failure");
    NSDictionary* result=object(readJSON(resultPath),"result");
    need(stringValue(field(result,"status"),"run status")=="failed"&&
         uintValue(field(result,"accepted_steps"),"accepted steps")==0u&&
         uintValue(field(result,"compiled_world_fingerprint"),"result fingerprint")==kFingerprint,
         "run result no longer describes the first-step rejection");
    NSDictionary* manifest=object(readJSON(manifestPath),"run manifest");
    need(uintValue(field(manifest,"compiled_world_fingerprint"),"world fingerprint")==kFingerprint,
         "run manifest fingerprint mismatch");
    NSDictionary* mesh=object(field(manifest,"mesh"),"mesh");
    need(uintValue(field(mesh,"nodes"),"node count")==882u&&
         uintValue(field(mesh,"tetrahedra"),"tet count")==1920u,"run mesh dimensions changed");
    NSDictionary* materials=object(field(manifest,"materials"),"material manifest");
    need(stringValue(field(materials,"medium_sha256"),"medium hash") == kAssetSHA,
         "run manifest medium source binding mismatch");
    NSDictionary* solver=object(field(manifest,"solver"),"solver manifest");
    in.dt=float(number(field(solver,"dt_s"),"dt"));
    need(in.dt==1.0e-4f&&uintValue(field(solver,"local_material_newton_iterations"),"local Newton updates")==32u,
         "solver timestep or local iteration budget changed");
    NSDictionary* policy=object(field(manifest,"compensated_source_paper_projection"),"paper projection policy");
    need([(NSNumber*)field(policy,"enabled") boolValue]&&
         [(NSNumber*)field(policy,"accurate_bilinear15_projection") boolValue],
         "source-paper compensated projection was not selected");

    std::string error;
    need(readPackage(packagePath,in.world,nullptr,&error),"compiled package read failed: "+error);
    need(in.world.fingerprint==kFingerprint&&in.world.dispatch.environmentCount==2u&&
         in.world.dispatch.femNodeCount==882u&&in.world.dispatch.tetrahedronCount==1920u&&
         in.world.materials.size()==3u&&in.world.objects.size()==1u,
         "compiled package dimensions/fingerprint mismatch");
    need(in.world.materials[kMedium].stateCount==15u&&in.world.materials[kMedium].parameterCount==28u&&
         in.world.materials[kMedium].localNewtonIterations==32u&&in.world.materials[kMedium].flags==213u,
         "compiled source392 medium descriptor mismatch");
    in.nodes=readRaw<NMFEMNodeStateGPU>(nodesPath,2u*882u);
    NSDictionary* cond=object(readJSON(condPath),"conditioning sidecar");
    need(![(NSNumber*)field(cond,"accepted_physical_state") boolValue]&&
         uintValue(field(cond,"control_step"),"conditioning control step")==0u&&
         stringValue(field(cond,"source_physics_fingerprint"),"conditioning fingerprint")==std::to_string(kFingerprint),
         "candidate is not the rejected step-0 conditioning snapshot");
    NSDictionary* state=object(readJSON(statePath),"accepted-state export");
    need(uintValue(field(state,"environment"),"state environment")==0u&&
         uintValue(field(state,"status_code"),"rolled-back status")==5u,
         "accepted material state is not env0 rollback after status 5");
    NSArray* rows=(NSArray*)field(state,"tetrahedra");
    need(rows.count==1920u,"accepted history tet row count mismatch");
    NSDictionary* target=object(rows[tetIndex],"selected accepted tet history");
    need(uintValue(field(target,"index"),"history tet index")==tetIndex&&
         uintValue(field(target,"material_index"),"history material index")==kMedium,
         "selected accepted history row mismatch");
    NSArray* history=(NSArray*)field(target,"state");
    need(history.count==15u,"selected accepted history has wrong state count");
    for(unsigned i=0;i<15u;++i)in.accepted[i]=float(number(history[i],"accepted history"));
    need(std::all_of(in.accepted.begin(),in.accepted.end(),[](float x){return x==0.0f;}),
         "source392 accepted rollback history is not all-zero");

    in.tet=in.world.fem.tetrahedra.at(tetIndex);
    need(in.tet.identity.x==kMedium&&in.tet.identity.y==0u&&
         in.tet.nodes.x==290u&&in.tet.nodes.y==293u&&in.tet.nodes.z==292u&&in.tet.nodes.w==587u&&
         (in.tet.identity.w&NM_TETRAHEDRON_REFERENCE_DISPLACEMENT_GRADIENT)!=0u,
         "source392 selected tet identity/connectivity/deformation policy changed");
    in.object=in.world.objects.at(in.tet.identity.y);
    need((in.object.flags&NM_OBJECT_FEM_MATERIAL_FRAME)!=0u,
         "source392 object material-frame policy changed");
    in.temperature=in.world.materials[kMedium].bulk.y;
    return in;
}

struct Native {
    id<MTLDevice> device=nil;id<MTLCommandQueue> queue=nil;id<MTLComputePipelineState> pipeline=nil;
    Native() {
        device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");
        std::string name=[[device name] UTF8String];
        need(name.find("Apple")!=std::string::npos&&name.find("Paravirtual")==std::string::npos,
             "physical Apple GPU required; no point replay dispatched");
        queue=[device newCommandQueue];need(queue!=nil,"Metal command queue unavailable");
        NSError* error=nil;id<MTLLibrary> library=[device newLibraryWithURL:
            [NSURL fileURLWithPath:@NUMI_CARDBOARD_PAPER_METALLIB] error:&error];
        need(library!=nil,"source-118 paper diagnostic metallib unavailable");
        id<MTLFunction> function=[library newFunctionWithName:@"cardboard_replay_source15_candidate_point"];
        need(function!=nil,"source118 native source15 candidate-point kernel missing");
        pipeline=[device newComputePipelineStateWithFunction:function error:&error];
        need(pipeline!=nil,"source15 candidate-point pipeline unavailable");
        std::cout<<"{\"record\":\"device\",\"device_name\":\""<<name
                 <<"\",\"physical_transactions\":0}\n";
    }
    std::vector<float> evaluate(const Inputs& in) {
        const NMMaterialGPU selected=in.world.materials.at(kMedium);
        const std::vector<NMContinuumObjectGPU> objects{in.object};
        const std::vector<NMTetrahedronGPU> tetrahedra{in.tet};
        const std::vector<NMMaterialGPU> materials{selected};
        const std::vector<float> accepted(in.accepted.begin(),in.accepted.end());
        const std::vector<float> config{in.dt,in.temperature};
        const auto params=parameterValues(in.world);
        std::vector<float> out(160u,0.0f);
        std::array<id<MTLBuffer>,10> buffers{
            makeBuffer(device,objects),makeBuffer(device,tetrahedra),makeBuffer(device,materials),
            makeBuffer(device,in.world.scalarPrograms),makeBuffer(device,in.world.instructions),
            makeBuffer(device,params),makeBuffer(device,in.nodes),makeBuffer(device,accepted),
            makeBuffer(device,out),makeBuffer(device,config)};
        need(std::all_of(buffers.begin(),buffers.end(),[](id<MTLBuffer> b){return b!=nil;}),
             "Metal point buffers failed to allocate");
        id<MTLCommandBuffer> command=[queue commandBuffer];need(command!=nil,"command buffer unavailable");
        id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];need(encoder!=nil,"encoder unavailable");
        [encoder setComputePipelineState:pipeline];
        for(NSUInteger i=0;i<buffers.size();++i)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,
             command.error?std::string([[command.error description] UTF8String]):"Metal command failed");
        auto* values=static_cast<const float*>(buffers[8].contents);
        need(values[0]==1.0f,"point kernel did not write completion marker");
        return std::vector<float>(values,values+160u);
    }
};

void writeOutput(const std::filesystem::path& outPath,const Inputs& in,const std::vector<float>& x) {
    std::ofstream out(outPath,std::ios::out|std::ios::trunc);need(out.good(),"cannot create point output");
    out<<std::setprecision(17)
       <<"{\"schema\":\"numi.cardboard.source392-tet-point.v1\","
       <<"\"base_snapshot_build_sha256\":\""<<NUMI_CARDBOARD_SOURCE118_BUILD_SHA256<<"\","
       <<"\"receipt_sha256\":\""<<in.receiptSHA<<"\",\"run_manifest_sha256\":\""<<in.runManifestSHA<<"\","
       <<"\"package_sha256\":\""<<in.packageSHA<<"\",\"candidate_nodes_sha256\":\""<<in.candidateSHA<<"\","
       <<"\"conditioning_manifest_sha256\":\""<<in.conditioningSHA<<"\",\"accepted_state_sha256\":\""<<in.stateSHA<<"\","
       <<"\"medium_asset_sha256\":\""<<in.assetSHA<<"\",\"physics_fingerprint\":"<<kFingerprint<<","
       <<"\"environment\":0,\"tet_id\":"<<in.tetIndex<<",\"material_index\":"<<kMedium<<","
       <<"\"control_step\":0,\"candidate_is_rejected_conditioning_snapshot\":true,"
       <<"\"candidate_exact_first_failure_generation_bound\":false,"
       <<"\"dt_s\":"<<in.dt<<",\"temperature_K\":"<<in.temperature<<","
       <<"\"physical_transactions\":0,\"physical_validation\":false,"
       <<"\"direct_stress_valid\":"<<(x[2]==1.0f?"true":"false")<<",\"determinant\":"<<x[1]<<","
       <<"\"tangent_valid\":"<<(x[48]==1.0f?"true":"false")<<",\"projection_flags\":"<<x[49]<<","
       <<"\"projection_converged\":"<<(x[55]==1.0f?"true":"false")<<",\"projection_residual\":"<<x[50]<<","
       <<"\"projection_state_norm\":"<<x[51]<<",\"local_updates\":"<<x[52]<<",\"terminal_failure\":"<<x[53]<<","
       <<"\"F_high\":";writeValues(out,x.data()+3,9);
    out<<",\"F_low\":";writeValues(out,x.data()+12,9);
    out<<",\"Fdot_local\":";writeValues(out,x.data()+21,9);
    out<<",\"direct_stress_local_Pa\":";writeValues(out,x.data()+30,9);
    out<<",\"direct_stress_world_Pa\":";writeValues(out,x.data()+39,9);
    out<<",\"tangent_stress_world_Pa\":";writeValues(out,x.data()+129,9);
    out<<",\"tangent_action_world_Pa_per_direction\":";writeValues(out,x.data()+147,9);
    out<<",\"projected_state_high\":";writeValues(out,x.data()+56,15);
    out<<",\"projected_state_low\":";writeValues(out,x.data()+88,15);
    out<<",\"raw_160_f32_words\":";writeBits(out,x.data(),x.size());out<<"}\n";
    out.flush();need(out.good(),"failed writing point output");
}

int run(const bool native,const std::filesystem::path& run,const std::filesystem::path& asset,
        const std::uint32_t tet,const std::filesystem::path& output) {
    Inputs in=loadInputs(run,asset,tet);
    std::cout<<"{\"record\":\"input_admission\",\"status\":\"pass\","
      <<"\"fingerprint\":"<<kFingerprint<<",\"tet\":"<<tet<<",\"material\":"<<kMedium
      <<",\"native_dispatch\":"<<(native?"true":"false")<<"}\n";
    if(!native)return 0;
    need(!std::filesystem::exists(output),"refusing to overwrite prior native point output");
    std::filesystem::create_directories(output);
    Native metal;auto values=metal.evaluate(in);
    writeOutput(output/"point.json",in,values);
    std::ofstream ledger(output/"point-replay.jsonl",std::ios::out|std::ios::trunc);
    need(ledger.good(),"cannot create point ledger");
    ledger<<"{\"record\":\"end\",\"status\":\"diagnostic_complete\","
          <<"\"native_projection_converged\":"<<(values[55]==1.0f?"true":"false")
          <<",\"note\":\"single production-kernel point evaluation only; does not reproduce full FEM dispatch generation or certify the coupon\"}\n";
    ledger.flush();need(ledger.good(),"failed writing point ledger");return 0;
}
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            const bool validate=argc==5&&std::string(argv[1])=="--validate";
            const bool native=argc==6&&std::string(argv[1])=="--native";
            need(validate||native,"usage: source392-tet858-point-check --validate RUN_DIR 858 MEDIUM.nmatter | --native RUN_DIR 858 MEDIUM.nmatter OUTPUT_DIR");
            const auto runPath=std::filesystem::path(argv[2]);
            const auto tet=static_cast<std::uint32_t>(std::stoul(argv[3]));
            const auto asset=std::filesystem::path(argv[4]);
            return run(native,runPath,asset,tet,native?std::filesystem::path(argv[5]):std::filesystem::path{});
        } catch(const std::exception& e) {std::cerr<<"FAIL source392 tet point replay: "<<e.what()<<'\n';return 2;}
    }
}
