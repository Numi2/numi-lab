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
#include <cstring>
#include <type_traits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NUMI_TANGENT_CACHE_METALLIB
#error "Bind the diagnostic checker to its source125 metallib"
#endif
#ifndef NUMI_TANGENT_CACHE_SOURCE125_MANIFEST_SHA256
#error "Bind the diagnostic checker to source125"
#endif

namespace {
using namespace numi::matter;
constexpr char kCaptureSHA[]="548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33";
constexpr char kPackageSHA[]="3c25f41c6c26654ccf3f8b4f4b510bbb64ab0ea0f52b94d05d8b1fa4060fafe5";
constexpr char kCandidateSHA[]="dcd6b8ea62c05c3ef9a04dc4d46ece7e7fbad58101d8d55d2bee6dfd9d7e8614";
constexpr char kAcceptedSHA[]="e352376a5e63ae33802d6eedb225839e4e71d0245076679fe51b5fe938a45a2b";
constexpr char kRunManifestSHA[]="2a34ad4f3328616fb5660fe22022941ca9ace9383ca9f1f1b7262ecfd4350ec8";
constexpr char kConditionManifestSHA[]="f2775bdc8f1747dcda5136b2bbe890e1d0645804df304c66bfe3143e90d955ad";
constexpr std::uint32_t kEnvironments=2u,kNodes=882u,kTets=1920u;
void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::string sha256(const std::filesystem::path& path){
    std::ifstream in(path,std::ios::binary);need(in.good(),"cannot read SHA input: "+path.string());
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)),{});
    need(b.size()<=UINT32_MAX,"SHA input exceeds bound");unsigned char digest[CC_SHA256_DIGEST_LENGTH]{};
    need(CC_SHA256(b.data(),static_cast<CC_LONG>(b.size()),digest)!=nullptr,"CommonCrypto SHA failed");
    static constexpr char h[]="0123456789abcdef";std::string out;out.reserve(64);
    for(auto x:digest){out.push_back(h[x>>4]);out.push_back(h[x&15]);}return out;
}
id readJSON(const std::filesystem::path& p){NSData* d=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:p.string().c_str()]];need(d!=nil,"cannot read JSON: "+p.string());NSError* e=nil;id v=[NSJSONSerialization JSONObjectWithData:d options:0 error:&e];need(v!=nil,"cannot parse JSON: "+p.string());return v;}
NSDictionary* asObject(id v,const std::string& what){need([v isKindOfClass:[NSDictionary class]],what+" is not an object");return (NSDictionary*)v;}
id field(NSDictionary* d,const char* key){id v=[d objectForKey:[NSString stringWithUTF8String:key]];need(v!=nil,std::string("missing JSON field ")+key);return v;}
double number(id v,const std::string& label){need([v isKindOfClass:[NSNumber class]],label+" is not numeric");double x=[(NSNumber*)v doubleValue];need(std::isfinite(x),label+" is nonfinite");return x;}
std::uint64_t uintValue(id v,const std::string& label){double x=number(v,label);need(x>=0&&std::floor(x)==x,label+" is not an unsigned integer");return static_cast<std::uint64_t>(x);}
std::string stringValue(id v,const std::string& label){need([v isKindOfClass:[NSString class]],label+" is not a string");return [(NSString*)v UTF8String];}
template<class T> std::vector<T> raw(const std::filesystem::path& p,std::size_t count){std::ifstream in(p,std::ios::binary|std::ios::ate);need(in.good(),"cannot open binary input: "+p.string());auto n=in.tellg();need(n>=0&&static_cast<std::uint64_t>(n)==count*sizeof(T),"binary input size mismatch: "+p.string());in.seekg(0);std::vector<T> out(count);in.read(reinterpret_cast<char*>(out.data()),static_cast<std::streamsize>(count*sizeof(T)));need(in.good(),"short binary input: "+p.string());return out;}
template<class T> id<MTLBuffer> makeBuffer(id<MTLDevice> d,const std::vector<T>& values){T zero{};return [d newBufferWithBytes:values.empty()?&zero:values.data() length:std::max<std::size_t>(1,values.size())*sizeof(T) options:MTLResourceStorageModeShared];}
std::vector<float> readFlatFloats(id value,std::size_t count,const std::string& what){need([value isKindOfClass:[NSArray class]],what+" is not an array");NSArray* a=(NSArray*)value;need(a.count==count,what+" has wrong count");std::vector<float> out;out.reserve(count);for(id x in a)out.push_back(static_cast<float>(number(x,what)));return out;}
struct Capture { NSDictionary* header=nil; std::vector<std::vector<float>> states; std::vector<std::array<float,81>> savedMatrices; std::vector<std::uint8_t> savedValid; std::vector<std::uint32_t> materials; std::vector<bool> seen; std::vector<nm_float4> capturedCorrection; std::size_t nodeRecords=0,tetRecords=0; };
Capture readCapture(const std::filesystem::path& path){
    std::ifstream in(path);need(in.good(),"cannot open operator capture");Capture c;c.states.resize(kEnvironments*kTets);c.savedMatrices.resize(kEnvironments*kTets);c.savedValid.resize(kEnvironments*kTets);c.materials.resize(kEnvironments*kTets);c.seen.resize(kEnvironments*kTets,false);c.capturedCorrection.resize(kEnvironments*kNodes);std::string line;bool haveHeader=false;
    while(std::getline(in,line)){if(line.empty())continue;NSData* d=[NSData dataWithBytes:line.data() length:line.size()];NSError* e=nil;id parsed=[NSJSONSerialization JSONObjectWithData:d options:0 error:&e];NSDictionary* row=asObject(parsed,"capture row");NSString* record=(NSString*)field(row,"record");
        if([record isEqualToString:@"header"]){need(!haveHeader,"duplicate header");c.header=row;haveHeader=true;continue;}
        if([record isEqualToString:@"integrity"]){need([(NSNumber*)field(row,"production_inputs_bitwise_unchanged") boolValue],"capture says inputs changed");continue;}
        if([record isEqualToString:@"node"]){++c.nodeRecords;const auto env=static_cast<std::size_t>(uintValue(field(row,"environment"),"node env"));const auto node=static_cast<std::size_t>(uintValue(field(row,"node_id"),"node id"));need(env<kEnvironments&&node<kNodes,"captured correction node index out of range");auto delta=readFlatFloats(field(row,"correction_delta_v_m_s"),3,"captured correction");c.capturedCorrection[env*kNodes+node]={delta[0],delta[1],delta[2],0.0f};continue;}
        if(![record isEqualToString:@"tetrahedron"])continue;
        ++c.tetRecords;const auto env=static_cast<std::uint32_t>(uintValue(field(row,"environment"),"environment"));const auto tet=static_cast<std::uint32_t>(uintValue(field(row,"tet_id"),"tet id"));need(env<kEnvironments&&tet<kTets,"capture tetra index out of range");const std::size_t idx=std::size_t(env)*kTets+tet;c.materials[idx]=static_cast<std::uint32_t>(uintValue(field(row,"material_index"),"material index"));need(!c.seen[idx],"duplicate capture tetra row");c.seen[idx]=true;
        NSDictionary* center=asObject(field(row,"center"),"capture center");const auto stateCount=static_cast<std::uint32_t>(uintValue(field(row,"material_state_count"),"material state count"));if(stateCount==15u)c.states[idx]=readFlatFloats(field(center,"accepted_state"),15,"captured accepted state");else{need(stateCount==0u,"unexpected captured material state count");need([(NSArray*)field(center,"accepted_state") count]==0u,"stateless material has nonempty accepted state");}
        NSDictionary* cache=asObject(field(row,"cached_algorithmic_tangent_local"),"captured tangent cache");c.savedValid[idx]=[(NSNumber*)field(cache,"valid") boolValue]?1u:0u;
        if(c.savedValid[idx]){auto a=readFlatFloats(field(cache,"matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"),81,"captured tangent matrix");std::copy(a.begin(),a.end(),c.savedMatrices[idx].begin());}
    }
    need(in.eof()&&haveHeader,"capture read incomplete/header absent");need(c.nodeRecords==kEnvironments*kNodes&&c.tetRecords==kEnvironments*kTets,"capture row counts unexpected");need(std::all_of(c.seen.begin(),c.seen.end(),[](bool x){return x;}),"capture has missing tetra record");
    need(uintValue(field(c.header,"environment_count"),"environment count")==kEnvironments&&uintValue(field(c.header,"node_count_per_environment"),"nodes") == kNodes&&uintValue(field(c.header,"tetrahedron_count_per_environment"),"tets")==kTets,"capture dimensions mismatch");
    need(uintValue(field(c.header,"root"),"root") == 2u&&uintValue(field(c.header,"newton_iteration"),"Newton") == 23u,"capture iteration identity mismatch");
    return c;
}
void verifyCaptureNodeWords(const std::filesystem::path& capturePath,const std::vector<NMFEMNodeStateGPU>& candidate){
    std::ifstream in(capturePath);need(in.good(),"cannot reopen capture for node-word validation");std::string line;std::size_t checked=0;
    while(std::getline(in,line)){if(line.empty())continue;NSData* d=[NSData dataWithBytes:line.data() length:line.size()];NSError* e=nil;NSDictionary* row=asObject([NSJSONSerialization JSONObjectWithData:d options:0 error:&e],"capture row");if(![(NSString*)field(row,"record") isEqualToString:@"node"])continue;
        const auto env=static_cast<std::size_t>(uintValue(field(row,"environment"),"node env"));const auto node=static_cast<std::size_t>(uintValue(field(row,"node_id"),"node id"));need(env<kEnvironments&&node<kNodes,"captured node index out of range");NSDictionary* center=asObject(field(row,"center"),"node center");id rawValue=field(center,"raw_state_float4s");need([rawValue isKindOfClass:[NSArray class]]&&[(NSArray*)rawValue count]==5u,"captured node raw layout mismatch");std::array<float,20> words{};std::size_t k=0;for(id v in (NSArray*)rawValue){auto f=readFlatFloats(v,4,"captured raw node word");std::copy(f.begin(),f.end(),words.begin()+static_cast<std::ptrdiff_t>(k));k+=4;}
        const auto* nodeWords=reinterpret_cast<const float*>(&candidate[env*kNodes+node]);for(std::size_t i=0;i<20;++i)need(std::bit_cast<std::uint32_t>(words[i])==std::bit_cast<std::uint32_t>(nodeWords[i]),"candidate node does not bit-match capture center at env/node/word "+std::to_string(env)+"/"+std::to_string(node)+"/"+std::to_string(i));++checked;
    }
    need(checked==kEnvironments*kNodes,"not all captured candidate node words checked");
}
void verifyAcceptedStateSidecar(const Capture& cap,const std::filesystem::path& statePath){
    NSDictionary* doc=asObject(readJSON(statePath),"accepted-state JSON");need(uintValue(field(doc,"environment"),"state env")==0u,"sidecar is not env0 bent state");NSArray* rows=(NSArray*)field(doc,"tetrahedra");need(rows.count==kTets,"accepted-state sidecar tet count mismatch");std::size_t mismatches=0;
    for(std::size_t t=0;t<kTets;++t){NSDictionary* row=asObject(rows[t],"state tet row");need(uintValue(field(row,"index"),"state index")==t,"state sidecar ordering mismatch");if(cap.materials[t]==2u){need([(NSArray*)field(row,"state") count]==0u,"stateless sidecar row has a state");continue;}auto a=readFlatFloats(field(row,"state"),15,"sidecar state");for(std::size_t s=0;s<15;++s)if(std::bit_cast<std::uint32_t>(a[s])!=std::bit_cast<std::uint32_t>(cap.states[t][s]))++mismatches;}
    need(mismatches==0,"capture accepted states differ from accepted-state sidecar after FP32 reconstruction");
}
struct InputBuffers {
    id<MTLBuffer> objects,materials,programs,instructions,params,nodes,tets,state,schedulers,adaptive,status,solver,fields,mixed,learned,layers,weights;
    NMMatterDispatchGPU dispatch{};NMMicrostepGPU micro{};NMMixedSolverGPU mixedSolver{};std::uint32_t cacheRebuild=1u;
};
InputBuffers prepareBuffers(id<MTLDevice> device,const CompiledWorld& w,const std::vector<NMFEMNodeStateGPU>& nodes,const std::vector<float>& state){
    InputBuffers b;b.dispatch=w.dispatch;b.dispatch.environmentCount=kEnvironments;b.micro.controlStep=2u;b.micro.microtick=0u;b.micro.microtickCount=1u;b.micro.solverIteration=23u;b.micro.flags=NM_MICROSTEP_FGMRES_OPERATOR;b.micro.time={w.dispatch.gravityAndTimestep.w,1.0f/w.dispatch.gravityAndTimestep.w,0,0};b.mixedSolver=w.mixedSolver;
    auto dup=[](const auto& v){std::vector<typename std::decay_t<decltype(v)>::value_type> o;for(unsigned e=0;e<kEnvironments;++e)o.insert(o.end(),v.begin(),v.end());return o;};
    std::vector<float> params;for(unsigned e=0;e<kEnvironments;++e)for(const auto& p:w.parameters)params.push_back(p.valueAndBounds.x);
    auto sched=dup(w.schedulers);auto adapt=dup(w.adaptive);
    b.objects=makeBuffer(device,w.objects);b.materials=makeBuffer(device,w.materials);b.programs=makeBuffer(device,w.scalarPrograms);b.instructions=makeBuffer(device,w.instructions);b.params=makeBuffer(device,params);b.nodes=makeBuffer(device,nodes);b.tets=makeBuffer(device,dup(w.fem.tetrahedra));b.state=makeBuffer(device,state);b.schedulers=makeBuffer(device,sched);b.adaptive=makeBuffer(device,adapt);b.status=makeBuffer(device,std::vector<NMMatterStatusGPU>(kEnvironments));b.solver=makeBuffer(device,std::vector<NMFGMRESStateGPU>(kEnvironments));b.fields=makeBuffer(device,w.fem.fields);b.mixed=makeBuffer(device,w.mixedMaterials);b.learned=makeBuffer(device,w.learnedMaterials);b.layers=makeBuffer(device,w.learnedLayers);b.weights=makeBuffer(device,w.learnedWeights);
    return b;
}
struct MetalRun {
    id<MTLDevice> device=nil;id<MTLCommandQueue> queue=nil;id<MTLLibrary> library=nil;id<MTLComputePipelineState> oldCache=nil,reuseCache=nil,apply=nil;
    MetalRun(){device=MTLCreateSystemDefaultDevice();need(device!=nil,"Metal device unavailable");NSString* name=[device name];need([name rangeOfString:@"Apple"].location!=NSNotFound&&[name rangeOfString:@"Paravirtual"].location==NSNotFound,"physical Apple GPU required for native diagnostic");queue=[device newCommandQueue];NSError* e=nil;library=[device newLibraryWithURL:[NSURL fileURLWithPath:@NUMI_TANGENT_CACHE_METALLIB] error:&e];need(library!=nil,"bound source125 metallib unavailable");oldCache=make(@"numi_matter_metal::nm_fem_build_algorithmic_tangent_cache");reuseCache=make(@"numi_matter_metal::nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic");apply=make(@"numi_matter_metal::nm_fem_apply_operator_elements");}
    id<MTLComputePipelineState> make(NSString* n){NSError* e=nil;id<MTLFunction> f=[library newFunctionWithName:n];need(f!=nil,"required kernel absent: "+std::string([n UTF8String]));id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&e];need(p!=nil,"pipeline creation failed: "+std::string([e.localizedDescription UTF8String]));return p;}
    std::pair<std::vector<float>,std::vector<std::uint32_t>> build(InputBuffers& b,id<MTLComputePipelineState> pipeline){const std::size_t total=std::size_t(kEnvironments)*kTets;auto cache=[device newBufferWithLength:total*81u*sizeof(float) options:MTLResourceStorageModeShared];auto valid=[device newBufferWithLength:total*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];need(cache&&valid,"cache output allocation failed");std::memset(cache.contents,0,cache.length);std::memset(valid.contents,0,valid.length);id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> en=[cmd computeCommandEncoder];need(en!=nil,"cache encoder unavailable");[en setComputePipelineState:pipeline];[en setBytes:&b.dispatch length:sizeof(b.dispatch) atIndex:0];[en setBytes:&b.micro length:sizeof(b.micro) atIndex:1];[en setBuffer:b.objects offset:0 atIndex:2];[en setBuffer:b.materials offset:0 atIndex:3];[en setBuffer:b.programs offset:0 atIndex:4];[en setBuffer:b.instructions offset:0 atIndex:5];[en setBuffer:b.params offset:0 atIndex:6];[en setBuffer:b.nodes offset:0 atIndex:7];[en setBuffer:b.tets offset:0 atIndex:8];[en setBuffer:b.state offset:0 atIndex:9];[en setBuffer:b.schedulers offset:0 atIndex:10];[en setBuffer:b.adaptive offset:0 atIndex:11];[en setBuffer:cache offset:0 atIndex:12];[en setBuffer:valid offset:0 atIndex:13];[en setBuffer:b.status offset:0 atIndex:14];[en setBuffer:b.solver offset:0 atIndex:15];[en setBytes:&b.cacheRebuild length:sizeof(b.cacheRebuild) atIndex:16];[en dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[en endEncoding];[cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted,"cache command failed");const auto* cacheStatuses=static_cast<const NMMatterStatusGPU*>(b.status.contents);for(unsigned env=0;env<kEnvironments;++env)need(cacheStatuses[env].code==NM_STATUS_SUCCESS,"cache builder status failure env="+std::to_string(env));const auto* cp=(const float*)cache.contents;const auto* vp=(const std::uint32_t*)valid.contents;return {{cp,cp+total*81u},{vp,vp+total}};}
    std::vector<NMFEMElementVectorGPU> action(InputBuffers& b,const std::vector<nm_float4>& direction,const std::vector<float>& tangent,const std::vector<std::uint32_t>& valid){const std::size_t total=std::size_t(kEnvironments)*kTets;auto dir=makeBuffer(device,direction);auto out=[device newBufferWithLength:total*sizeof(NMFEMElementVectorGPU) options:MTLResourceStorageModeShared];auto cache=makeBuffer(device,tangent);auto validBuf=makeBuffer(device,valid);need(out!=nil,"operator output allocation failed");std::memset(out.contents,0,out.length);const std::uint32_t use=1u;id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> en=[cmd computeCommandEncoder];need(en!=nil,"operator encoder unavailable");[en setComputePipelineState:apply];[en setBytes:&b.dispatch length:sizeof(b.dispatch) atIndex:0];[en setBytes:&b.micro length:sizeof(b.micro) atIndex:1];[en setBuffer:b.objects offset:0 atIndex:2];[en setBuffer:b.materials offset:0 atIndex:3];[en setBuffer:b.programs offset:0 atIndex:4];[en setBuffer:b.instructions offset:0 atIndex:5];[en setBuffer:b.params offset:0 atIndex:6];[en setBuffer:b.nodes offset:0 atIndex:7];[en setBuffer:b.tets offset:0 atIndex:8];[en setBuffer:b.state offset:0 atIndex:9];[en setBuffer:dir offset:0 atIndex:10];[en setBuffer:b.schedulers offset:0 atIndex:11];[en setBuffer:b.adaptive offset:0 atIndex:12];[en setBuffer:out offset:0 atIndex:13];[en setBuffer:b.status offset:0 atIndex:14];[en setBuffer:b.mixed offset:0 atIndex:16];[en setBuffer:b.fields offset:0 atIndex:17];[en setBuffer:b.learned offset:0 atIndex:18];[en setBuffer:b.layers offset:0 atIndex:19];[en setBuffer:b.weights offset:0 atIndex:20];[en setBytes:&b.mixedSolver length:sizeof(b.mixedSolver) atIndex:21];[en setBuffer:b.solver offset:0 atIndex:22];[en setBuffer:cache offset:0 atIndex:23];[en setBuffer:validBuf offset:0 atIndex:24];[en setBytes:&use length:sizeof(use) atIndex:25];[en dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[en endEncoding];[cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted,"operator command failed");auto* status=(const NMMatterStatusGPU*)b.status.contents;for(unsigned e=0;e<kEnvironments;++e)need(status[e].code==NM_STATUS_SUCCESS,"operator status failure env="+std::to_string(e)+" code="+std::to_string(status[e].code));const auto* p=(const NMFEMElementVectorGPU*)out.contents;return {p,p+total};}
};
struct Compare {std::uint64_t count=0,bitsDifferent=0,nonfinite=0;double maxAbs=0,maxRel=0,scale=0;void add(float a,float b){++count;if(std::bit_cast<std::uint32_t>(a)!=std::bit_cast<std::uint32_t>(b))++bitsDifferent;if(!std::isfinite(a)||!std::isfinite(b)){++nonfinite;return;}maxAbs=std::max(maxAbs,std::abs(double(a)-double(b)));maxRel=std::max(maxRel,std::abs(double(a)-double(b))/std::max(1e-30,std::max(std::abs(double(a)),std::abs(double(b)))));scale=std::max(scale,std::max(std::abs(double(a)),std::abs(double(b))));}};
void execute(const std::filesystem::path& run,const std::filesystem::path& outdir){const auto output=run/"output";const auto capturePath=run/"operator-capture";const auto packagePath=output/"compiled.nmatterpack";const auto candidatePath=output/"rejected-conditioning-000003/candidate_nodes.bin";const auto conditionPath=output/"rejected-conditioning-000003/manifest.json";const auto statePath=output/"material_state_000003_bent.json";const auto manifestPath=output/"manifest.json";
    need(sha256(capturePath)==kCaptureSHA,"run360 operator capture hash mismatch");need(sha256(packagePath)==kPackageSHA,"run360 package hash mismatch");need(sha256(candidatePath)==kCandidateSHA,"run360 candidate hash mismatch");need(sha256(statePath)==kAcceptedSHA,"run360 accepted state hash mismatch");need(sha256(manifestPath)==kRunManifestSHA,"run360 manifest hash mismatch");need(sha256(conditionPath)==kConditionManifestSHA,"run360 conditioning manifest hash mismatch");
    Capture cap=readCapture(capturePath);const auto fpHex=stringValue(field(cap.header,"world_fingerprint_hex"),"world fingerprint hex");need(std::stoull(fpHex,nullptr,16)==0x6de3475316b7e8b9ull,"capture world fingerprint mismatch");need([(NSNumber*)field(field(cap.header,"cached_algorithmic_tangent_capture"),"policy_enabled") boolValue],"capture lacks production tangent cache");
    CompiledWorld w;std::string packageError;need(readPackage(packagePath,w,nullptr,&packageError),"compiled capture package rejected: "+packageError);need(w.fingerprint==0x6de3475316b7e8b9ull,"compiled package fingerprint mismatch");need(w.dispatch.environmentCount==kEnvironments&&w.dispatch.femNodeCount==kNodes&&w.dispatch.tetrahedronCount==kTets&&w.dispatch.materialStateStride>=15u,"compiled dimensions/state stride mismatch");need(w.materials.size()==3u&&w.materials[0].stateCount==15u&&w.materials[1].stateCount==15u&&w.materials[2].stateCount==0u,"compiled material/state layout mismatch");need(w.objects.size()==1u&&w.parameters.size()==84u,"compiled object/parameter layout mismatch");need(w.materials[0].flags==213u&&w.materials[1].flags==213u,"compiled source-15 policy flags differ from capture");const double captureDt=number(field(cap.header,"dt_seconds"),"capture dt");need(std::abs(captureDt-double(w.dispatch.gravityAndTimestep.w))<=1e-10,"capture/package timestep mismatch");NSArray* parameterEnvironments=(NSArray*)field(cap.header,"parameters_by_environment");need(parameterEnvironments.count==kEnvironments,"capture parameter environment count mismatch");for(unsigned env=0;env<kEnvironments;++env){auto values=readFlatFloats(parameterEnvironments[env],w.parameters.size(),"capture parameters");for(std::size_t pi=0;pi<values.size();++pi)need(std::bit_cast<std::uint32_t>(values[pi])==std::bit_cast<std::uint32_t>(w.parameters[pi].valueAndBounds.x),"capture/package parameter word mismatch env="+std::to_string(env)+" index="+std::to_string(pi));}
    auto candidate=raw<NMFEMNodeStateGPU>(candidatePath,std::size_t(kEnvironments)*kNodes);verifyCaptureNodeWords(capturePath,candidate);verifyAcceptedStateSidecar(cap,statePath);
    std::vector<float> states(std::size_t(kEnvironments)*kTets*w.dispatch.materialStateStride,0.0f);for(std::size_t i=0;i<cap.states.size();++i){const auto mat=w.fem.tetrahedra[i%kTets].identity.x;const auto n=w.materials.at(mat).stateCount;need(n==0u||n==15u,"unexpected state count in run360 material table");need(cap.materials[i]==mat,"capture/package material map mismatch");if(n==15u)std::copy(cap.states[i].begin(),cap.states[i].end(),states.begin()+static_cast<std::ptrdiff_t>(i*w.dispatch.materialStateStride));}
    std::filesystem::create_directories(outdir);MetalRun metal;auto b=prepareBuffers(metal.device,w,candidate,states);auto old=metal.build(b,metal.oldCache);auto reused=metal.build(b,metal.reuseCache);
    need(old.second.size()==reused.second.size(),"cache valid-bit count mismatch");Compare matrix;std::uint64_t validDiff=0,validOld=0,validNew=0,historicalValid=0,historicalDiff=0;double histAbs=0,histRel=0;for(std::size_t e=0;e<old.second.size();++e){if(old.second[e])++validOld;if(reused.second[e])++validNew;if(old.second[e]!=reused.second[e])++validDiff;if(old.second[e]&&reused.second[e])for(unsigned j=0;j<81;++j){float a=old.first[e*81+j],bval=reused.first[e*81+j];matrix.add(a,bval);if(cap.savedValid[e]){++historicalValid;if(std::bit_cast<std::uint32_t>(a)!=std::bit_cast<std::uint32_t>(cap.savedMatrices[e][j]))++historicalDiff;histAbs=std::max(histAbs,std::abs(double(a)-cap.savedMatrices[e][j]));histRel=std::max(histRel,std::abs(double(a)-cap.savedMatrices[e][j])/std::max(1e-30,std::abs(double(cap.savedMatrices[e][j]))));}}}
    std::vector<nm_float4> randomDirection(std::size_t(kEnvironments)*kNodes);std::uint32_t rng=0x2f39a1c7u;for(auto& v:randomDirection){auto sample=[&](){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return (float(rng&0xffffu)/32767.5f)-1.0f;};v={sample(),sample(),sample(),0};}
    auto compareActions=[&](const std::vector<nm_float4>& dir){std::vector<NMMatterStatusGPU> resetStatus(kEnvironments);b.status=makeBuffer(metal.device,resetStatus);auto a=metal.action(b,dir,old.first,old.second);b.status=makeBuffer(metal.device,resetStatus);auto z=metal.action(b,dir,reused.first,reused.second);Compare diff;for(std::size_t e=0;e<a.size();++e){const auto* x=reinterpret_cast<const float*>(&a[e]);const auto* y=reinterpret_cast<const float*>(&z[e]);for(std::size_t j=0;j<16;++j)diff.add(x[j],y[j]);}return diff;};
    const auto capturedAction=compareActions(cap.capturedCorrection);const auto randomAction=compareActions(randomDirection);
    const bool parityPass=validOld>0u&&validOld==validNew&&validDiff==0u&&
        matrix.count==validOld*81u&&matrix.bitsDifferent==0u&&matrix.nonfinite==0u&&
        capturedAction.bitsDifferent==0u&&capturedAction.nonfinite==0u&&
        randomAction.bitsDifferent==0u&&randomAction.nonfinite==0u;
    std::ofstream out(outdir/"projection-reuse-parity.json");
    need(out.good(),"cannot write result");
    out<<std::setprecision(17)
       <<"{\n  \"schema\":\"numi.cardboard.tangent-cache-projection-reuse-parity.v1\","
       <<"\n  \"source125_manifest_sha256\":\""<<NUMI_TANGENT_CACHE_SOURCE125_MANIFEST_SHA256<<"\","
       <<"\n  \"run360_capture_sha256\":\""<<kCaptureSHA<<"\","
       <<"\n  \"run360_package_sha256\":\""<<kPackageSHA<<"\","
       <<"\n  \"run360_candidate_nodes_sha256\":\""<<kCandidateSHA<<"\","
       <<"\n  \"world_fingerprint_hex\":\"6de3475316b7e8b9\","
       <<"\n  \"kernel_pair\":[\"nm_fem_build_algorithmic_tangent_cache\","
       <<"\"nm_fem_build_algorithmic_tangent_cache_projection_reuse_diagnostic\"],"
       <<"\n  \"diagnostic_scope\":\"zero physical steps; same exact compiled package, candidate node words, accepted history, parameters, frames, timestep and production operator\","
       <<"\n  \"elements\":"<<old.second.size()<<","
       <<"\n  \"old_valid_entries\":"<<validOld<<","
       <<"\n  \"reuse_valid_entries\":"<<validNew<<","
       <<"\n  \"validity_bit_differences\":"<<validDiff<<","
       <<"\n  \"exact_cache_and_action_parity\":"<<(parityPass?"true":"false")<<","
       <<"\n  \"matrix_float_comparisons\":"<<matrix.count<<","
       <<"\n  \"matrix_bit_differences\":"<<matrix.bitsDifferent<<",\n  \"matrix_nonfinite_values\":"<<matrix.nonfinite<<","
       <<"\n  \"matrix_max_abs_difference\":"<<matrix.maxAbs<<","
       <<"\n  \"matrix_max_relative_difference\":"<<matrix.maxRel<<","
       <<"\n  \"matrix_max_reference_magnitude\":"<<matrix.scale<<","
       <<"\n  \"historical_run360_cached_matrix_comparisons\":"<<historicalValid*81u<<","
       <<"\n  \"historical_matrix_bit_differences\":"<<historicalDiff<<","
       <<"\n  \"historical_matrix_max_abs_difference\":"<<histAbs<<","
       <<"\n  \"historical_matrix_max_relative_difference\":"<<histRel<<","
       <<"\n  \"captured_correction_action_float_comparisons\":"<<capturedAction.count<<","
       <<"\n  \"captured_correction_action_bit_differences\":"<<capturedAction.bitsDifferent<<",\n  \"captured_correction_action_nonfinite_values\":"<<capturedAction.nonfinite<<","
       <<"\n  \"captured_correction_action_max_abs_difference\":"<<capturedAction.maxAbs<<","
       <<"\n  \"captured_correction_action_max_relative_difference\":"<<capturedAction.maxRel<<","
       <<"\n  \"random_action_float_comparisons\":"<<randomAction.count<<","
       <<"\n  \"random_action_bit_differences\":"<<randomAction.bitsDifferent<<",\n  \"random_action_nonfinite_values\":"<<randomAction.nonfinite<<","
       <<"\n  \"random_action_max_abs_difference\":"<<randomAction.maxAbs<<","
       <<"\n  \"random_action_max_relative_difference\":"<<randomAction.maxRel<<","
       <<"\n  \"physical_validation\":false,\n  \"performance_claim\":false\n}\n";
    out.close();need(out.good(),"result write failed");
    std::cout<<"{\"record\":\"parity_complete\",\"status\":\""
             <<(parityPass?"pass":"fail")<<"\",\"validity_bit_differences\":"<<validDiff
             <<",\"matrix_bit_differences\":"<<matrix.bitsDifferent
             <<",\"captured_action_bit_differences\":"<<capturedAction.bitsDifferent
             <<",\"random_action_bit_differences\":"<<randomAction.bitsDifferent
             <<",\"physical_steps\":0}\n";
    need(parityPass,"projection-reuse cache differs from repeated production evaluation; see JSON output");
}
void cpu(const std::filesystem::path& run){need(sha256(run/"operator-capture")==kCaptureSHA,"capture SHA admission failed");need(sha256(run/"output/compiled.nmatterpack")==kPackageSHA,"package SHA admission failed");need(sha256(run/"output/rejected-conditioning-000003/candidate_nodes.bin")==kCandidateSHA,"candidate SHA admission failed");Capture c=readCapture(run/"operator-capture");need(c.nodeRecords==1764&&c.tetRecords==3840,"capture completeness test failed");need(std::string(NUMI_TANGENT_CACHE_SOURCE125_MANIFEST_SHA256).size()==64,"source binding macro malformed");std::cout<<"{\"record\":\"cpu_admission\",\"status\":\"pass\",\"capture_rows\":3840,\"node_rows\":1764,\"physical_steps\":0}\n";}
}
int main(int argc,char**argv){@autoreleasepool{try{if(argc==3&&std::string(argv[1])=="--cpu-admission"){cpu(argv[2]);return 0;}if(argc==4&&std::string(argv[1])=="--native"){execute(argv[2],argv[3]);return 0;}std::cerr<<"usage: checker --cpu-admission RUN360 | --native RUN360 OUTPUT_DIR\n";return 2;}catch(const std::exception& e){std::cerr<<"tangent-cache checker: "<<e.what()<<"\n";return 1;}}}
