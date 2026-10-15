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
#include <sstream>
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
struct FixtureBinding {
    const char* label;
    const char* captureSHA;
    const char* packageSHA;
    const char* candidateSHA;
    const char* acceptedSHA;
    const char* runManifestSHA;
    const char* conditionManifestSHA;
    const char* sourceBuildSHA;
    const char* receiptSHA;
    const char* resultSHA;
    const char* sourceSnapshot;
    std::uint32_t root;
    std::uint32_t newtonIteration;
    std::uint32_t conditioningStep;
    std::uint64_t worldFingerprint;
};
constexpr FixtureBinding kFixture360{
    "run360_root2_newton23", "548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33",
    "3c25f41c6c26654ccf3f8b4f4b510bbb64ab0ea0f52b94d05d8b1fa4060fafe5",
    "dcd6b8ea62c05c3ef9a04dc4d46ece7e7fbad58101d8d55d2bee6dfd9d7e8614",
    "e352376a5e63ae33802d6eedb225839e4e71d0245076679fe51b5fe938a45a2b",
    "2a34ad4f3328616fb5660fe22022941ca9ace9383ca9f1f1b7262ecfd4350ec8",
    "f2775bdc8f1747dcda5136b2bbe890e1d0645804df304c66bfe3143e90d955ad",
    "b40d30669b286449a3c5d50da2c865a052cd69c1e9f556132f0fc4bc66cd9fb0",
    "d623591647a91ca07626e18fef6c78358e60c0d9ab7c38d465e79a2f6c18c474",
    "4afcdb2310cf87130417399775dc132cb0509f805f537bfe0fa28b3c3ae7b2d5",
    "/Users/home/cardboard-implementation-20261014/snapshots/source-boundary-reset-order-101",
    2u, 23u, 3u, 0x6de3475316b7e8b9ull
};
constexpr FixtureBinding kFixture419{
    "run419_root1_newton48", "f2d42314cfee0428ecd285ebb9dbd5da1556624e3e92af1d12240fd8106b309a",
    "cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c",
    "c4f2b8d790494949da5be98e876b911db2655dc88e7417416906c91412557269",
    "fb270a40f77d2251451e1ca1ab2c09e1c4779775a9f073fde25a6b6218ef2ee4",
    "bcc9d5c5085635ea845d4d34f37bbf4c26b5af9086c15f5e02bb4597d2c2c3f5",
    "40e918b680ab35c6b5dd571796221311535f36215a43a80331b50b28fce4456d",
    "35b821184ca36ad57c529bc0e33e4f20519996def1c312f8ac0f8028ecadb8e6",
    "0826a69a65ea4a0d76ef95e56c5d0d7f5238f4de746dd6daf81f11a87907f67d",
    "6600e9b15bba128d2ef4ed2c5c4a181e91fdc8600a4171ecdd7e1c2eb04efcba",
    "/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125",
    1u, 48u, 2u, 0x75c1264bc96ac2d9ull
};
std::string sixDigitStep(std::uint32_t value) {
    const std::string digits = std::to_string(value);
    return std::string(digits.size() < 6u ? 6u - digits.size() : 0u, '0') + digits;
}
std::string fingerprintHex(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::nouppercase << std::setw(16) << std::setfill('0') << value;
    return out.str();
}
constexpr std::uint32_t kEnvironments=2u,kNodes=882u,kTets=1920u;
void need(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
bool isSHA256(const char* value){
    if(value==nullptr||std::strlen(value)!=64u)return false;
    for(std::size_t i=0;i<64u;++i){
        const char c=value[i];
        if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F')))return false;
    }
    return true;
}
void validateFixtureHashes(const FixtureBinding& fixture){
    const std::array<const char*,9> hashes={fixture.captureSHA,fixture.packageSHA,fixture.candidateSHA,
        fixture.acceptedSHA,fixture.runManifestSHA,fixture.conditionManifestSHA,fixture.sourceBuildSHA,
        fixture.receiptSHA,fixture.resultSHA};
    for(std::size_t i=0;i<hashes.size();++i)
        need(isSHA256(hashes[i]),std::string(fixture.label)+" has malformed 64-hex fixture SHA at slot "+std::to_string(i));
}
void testSHA256Validation(){
    const std::string valid(64u,'a');
    const std::string shortValue=valid.substr(1u);
    std::string invalidHex=valid;invalidHex[17]='g';
    need(isSHA256(valid.c_str()),"SHA validator rejected a valid 64-hex digest");
    need(!isSHA256(shortValue.c_str()),"SHA validator accepted a 63-character digest");
    need(!isSHA256(invalidHex.c_str()),"SHA validator accepted a non-hex digest");
    need(!isSHA256(nullptr),"SHA validator accepted a null digest");
}
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
Capture readCapture(const std::filesystem::path& path,const FixtureBinding& fixture){
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
    need(uintValue(field(c.header,"root"),"root") == fixture.root&&uintValue(field(c.header,"newton_iteration"),"Newton") == fixture.newtonIteration,"capture iteration identity mismatch for "+std::string(fixture.label));
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
    id<MTLComputePipelineState> make(NSString* n){NSError* e=nil;id<MTLFunction> f=[library newFunctionWithName:n];need(f!=nil,"required kernel absent: "+std::string([n UTF8String]));id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:f error:&e];if(p==nil){NSString* description=e.localizedDescription;const char* utf8=description?description.UTF8String:nullptr;throw std::runtime_error(std::string("pipeline creation failed: ")+(utf8?utf8:"unknown Metal error"));}return p;}
    std::pair<std::vector<float>,std::vector<std::uint32_t>> build(InputBuffers& b,id<MTLComputePipelineState> pipeline){const std::size_t total=std::size_t(kEnvironments)*kTets;auto cache=[device newBufferWithLength:total*81u*sizeof(float) options:MTLResourceStorageModeShared];auto valid=[device newBufferWithLength:total*sizeof(std::uint32_t) options:MTLResourceStorageModeShared];need(cache&&valid,"cache output allocation failed");std::memset(cache.contents,0,cache.length);std::memset(valid.contents,0,valid.length);id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> en=[cmd computeCommandEncoder];need(en!=nil,"cache encoder unavailable");[en setComputePipelineState:pipeline];[en setBytes:&b.dispatch length:sizeof(b.dispatch) atIndex:0];[en setBytes:&b.micro length:sizeof(b.micro) atIndex:1];[en setBuffer:b.objects offset:0 atIndex:2];[en setBuffer:b.materials offset:0 atIndex:3];[en setBuffer:b.programs offset:0 atIndex:4];[en setBuffer:b.instructions offset:0 atIndex:5];[en setBuffer:b.params offset:0 atIndex:6];[en setBuffer:b.nodes offset:0 atIndex:7];[en setBuffer:b.tets offset:0 atIndex:8];[en setBuffer:b.state offset:0 atIndex:9];[en setBuffer:b.schedulers offset:0 atIndex:10];[en setBuffer:b.adaptive offset:0 atIndex:11];[en setBuffer:cache offset:0 atIndex:12];[en setBuffer:valid offset:0 atIndex:13];[en setBuffer:b.status offset:0 atIndex:14];[en setBuffer:b.solver offset:0 atIndex:15];[en setBytes:&b.cacheRebuild length:sizeof(b.cacheRebuild) atIndex:16];[en dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[en endEncoding];[cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted,"cache command failed");const auto* cacheStatuses=static_cast<const NMMatterStatusGPU*>(b.status.contents);for(unsigned env=0;env<kEnvironments;++env)need(cacheStatuses[env].code==NM_STATUS_SUCCESS,"cache builder status failure env="+std::to_string(env));const auto* cp=(const float*)cache.contents;const auto* vp=(const std::uint32_t*)valid.contents;return {{cp,cp+total*81u},{vp,vp+total}};}
    std::vector<NMFEMElementVectorGPU> action(InputBuffers& b,const std::vector<nm_float4>& direction,const std::vector<float>& tangent,const std::vector<std::uint32_t>& valid){const std::size_t total=std::size_t(kEnvironments)*kTets;auto dir=makeBuffer(device,direction);auto out=[device newBufferWithLength:total*sizeof(NMFEMElementVectorGPU) options:MTLResourceStorageModeShared];auto cache=makeBuffer(device,tangent);auto validBuf=makeBuffer(device,valid);need(out!=nil,"operator output allocation failed");std::memset(out.contents,0,out.length);const std::uint32_t use=1u;id<MTLCommandBuffer> cmd=[queue commandBuffer];id<MTLComputeCommandEncoder> en=[cmd computeCommandEncoder];need(en!=nil,"operator encoder unavailable");[en setComputePipelineState:apply];[en setBytes:&b.dispatch length:sizeof(b.dispatch) atIndex:0];[en setBytes:&b.micro length:sizeof(b.micro) atIndex:1];[en setBuffer:b.objects offset:0 atIndex:2];[en setBuffer:b.materials offset:0 atIndex:3];[en setBuffer:b.programs offset:0 atIndex:4];[en setBuffer:b.instructions offset:0 atIndex:5];[en setBuffer:b.params offset:0 atIndex:6];[en setBuffer:b.nodes offset:0 atIndex:7];[en setBuffer:b.tets offset:0 atIndex:8];[en setBuffer:b.state offset:0 atIndex:9];[en setBuffer:dir offset:0 atIndex:10];[en setBuffer:b.schedulers offset:0 atIndex:11];[en setBuffer:b.adaptive offset:0 atIndex:12];[en setBuffer:out offset:0 atIndex:13];[en setBuffer:b.status offset:0 atIndex:14];[en setBuffer:b.mixed offset:0 atIndex:16];[en setBuffer:b.fields offset:0 atIndex:17];[en setBuffer:b.learned offset:0 atIndex:18];[en setBuffer:b.layers offset:0 atIndex:19];[en setBuffer:b.weights offset:0 atIndex:20];[en setBytes:&b.mixedSolver length:sizeof(b.mixedSolver) atIndex:21];[en setBuffer:b.solver offset:0 atIndex:22];[en setBuffer:cache offset:0 atIndex:23];[en setBuffer:validBuf offset:0 atIndex:24];[en setBytes:&use length:sizeof(use) atIndex:25];[en dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[en endEncoding];[cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted,"operator command failed");auto* status=(const NMMatterStatusGPU*)b.status.contents;for(unsigned e=0;e<kEnvironments;++e)need(status[e].code==NM_STATUS_SUCCESS,"operator status failure env="+std::to_string(e)+" code="+std::to_string(status[e].code));const auto* p=(const NMFEMElementVectorGPU*)out.contents;return {p,p+total};}
};
struct Compare {std::uint64_t count=0,bitsDifferent=0,nonfinite=0;double maxAbs=0,maxRel=0,scale=0;void add(float a,float b){++count;if(std::bit_cast<std::uint32_t>(a)!=std::bit_cast<std::uint32_t>(b))++bitsDifferent;if(!std::isfinite(a)||!std::isfinite(b)){++nonfinite;return;}maxAbs=std::max(maxAbs,std::abs(double(a)-double(b)));maxRel=std::max(maxRel,std::abs(double(a)-double(b))/std::max(1e-30,std::max(std::abs(double(a)),std::abs(double(b)))));scale=std::max(scale,std::max(std::abs(double(a)),std::abs(double(b))));}};
void execute(const std::filesystem::path& run,const std::filesystem::path& outdir,const FixtureBinding& fixture){validateFixtureHashes(fixture);const auto output=run/"output";const auto capturePath=run/"operator-capture";const auto packagePath=output/"compiled.nmatterpack";const std::string step=sixDigitStep(fixture.conditioningStep);const auto conditionDir=output/("rejected-conditioning-"+step);const auto candidatePath=conditionDir/"candidate_nodes.bin";const auto conditionPath=conditionDir/"manifest.json";const auto statePath=output/("material_state_"+step+"_bent.json");const auto manifestPath=output/"manifest.json";const auto sourceBuildPath=run/"inputs/external/source-build.json";const auto receiptPath=run/"receipt.json";const auto resultPath=output/"result.json";
    need(sha256(capturePath)==fixture.captureSHA,std::string(fixture.label)+" operator capture hash mismatch");need(sha256(packagePath)==fixture.packageSHA,std::string(fixture.label)+" package hash mismatch");need(sha256(candidatePath)==fixture.candidateSHA,std::string(fixture.label)+" candidate hash mismatch");need(sha256(statePath)==fixture.acceptedSHA,std::string(fixture.label)+" accepted state hash mismatch");need(sha256(manifestPath)==fixture.runManifestSHA,std::string(fixture.label)+" manifest hash mismatch");need(sha256(conditionPath)==fixture.conditionManifestSHA,std::string(fixture.label)+" conditioning manifest hash mismatch");
    need(sha256(sourceBuildPath)==fixture.sourceBuildSHA,std::string(fixture.label)+" source-build binding mismatch");
    need(sha256(receiptPath)==fixture.receiptSHA,std::string(fixture.label)+" execution receipt binding mismatch");
    need(sha256(resultPath)==fixture.resultSHA,std::string(fixture.label)+" native result binding mismatch");
    NSDictionary* sourceBuild=asObject(readJSON(sourceBuildPath),"source-build JSON");
    need(stringValue(field(sourceBuild,"source_snapshot"),"source snapshot")==fixture.sourceSnapshot,std::string(fixture.label)+" source snapshot mismatch");
    Capture cap=readCapture(capturePath,fixture);const auto fpHex=stringValue(field(cap.header,"world_fingerprint_hex"),"world fingerprint hex");need(std::stoull(fpHex,nullptr,16)==fixture.worldFingerprint,"capture world fingerprint mismatch");need([(NSNumber*)field(field(cap.header,"cached_algorithmic_tangent_capture"),"policy_enabled") boolValue],"capture lacks production tangent cache");
    CompiledWorld w;std::string packageError;need(readPackage(packagePath,w,nullptr,&packageError),"compiled capture package rejected: "+packageError);need(w.fingerprint==fixture.worldFingerprint,"compiled package fingerprint mismatch");need(w.dispatch.environmentCount==kEnvironments&&w.dispatch.femNodeCount==kNodes&&w.dispatch.tetrahedronCount==kTets&&w.dispatch.materialStateStride>=15u,"compiled dimensions/state stride mismatch");need(w.materials.size()==3u&&w.materials[0].stateCount==15u&&w.materials[1].stateCount==15u&&w.materials[2].stateCount==0u,"compiled material/state layout mismatch");need(w.objects.size()==1u&&w.parameters.size()==59u,"compiled object/parameter layout mismatch: objects="+std::to_string(w.objects.size())+" parameters="+std::to_string(w.parameters.size()));need(w.materials[0].flags==213u&&w.materials[1].flags==213u,"compiled source-15 policy flags differ from capture");const double captureDt=number(field(cap.header,"dt_seconds"),"capture dt");need(std::abs(captureDt-double(w.dispatch.gravityAndTimestep.w))<=1e-10,"capture/package timestep mismatch");NSArray* parameterEnvironments=(NSArray*)field(cap.header,"parameters_by_environment");need(parameterEnvironments.count==kEnvironments,"capture parameter environment count mismatch");for(unsigned env=0;env<kEnvironments;++env){auto values=readFlatFloats(parameterEnvironments[env],w.parameters.size(),"capture parameters");for(std::size_t pi=0;pi<values.size();++pi)need(std::bit_cast<std::uint32_t>(values[pi])==std::bit_cast<std::uint32_t>(w.parameters[pi].valueAndBounds.x),"capture/package parameter word mismatch env="+std::to_string(env)+" index="+std::to_string(pi));}
    auto candidate=raw<NMFEMNodeStateGPU>(candidatePath,std::size_t(kEnvironments)*kNodes);verifyCaptureNodeWords(capturePath,candidate);verifyAcceptedStateSidecar(cap,statePath);
    std::vector<float> states(std::size_t(kEnvironments)*kTets*w.dispatch.materialStateStride,0.0f);for(std::size_t i=0;i<cap.states.size();++i){const auto mat=w.fem.tetrahedra[i%kTets].identity.x;const auto n=w.materials.at(mat).stateCount;need(n==0u||n==15u,"unexpected state count in fixed run fixture material table");need(cap.materials[i]==mat,"capture/package material map mismatch");if(n==15u)std::copy(cap.states[i].begin(),cap.states[i].end(),states.begin()+static_cast<std::ptrdiff_t>(i*w.dispatch.materialStateStride));}
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
       <<"{\n  \"schema\":\"numi.cardboard.tangent-cache-projection-reuse-parity.v2\"," 
       <<"\n  \"source125_manifest_sha256\":\""<<NUMI_TANGENT_CACHE_SOURCE125_MANIFEST_SHA256<<"\","
       <<"\n  \"fixture_id\":\""<<fixture.label<<"\",\n  \"capture_sha256\":\""<<fixture.captureSHA<<"\","
       <<"\n  \"package_sha256\":\""<<fixture.packageSHA<<"\","
       <<"\n  \"source_build_sha256\":\""<<fixture.sourceBuildSHA<<"\"," 
       <<"\n  \"source_snapshot\":\""<<fixture.sourceSnapshot<<"\"," 
       <<"\n  \"receipt_sha256\":\""<<fixture.receiptSHA<<"\"," 
       <<"\n  \"result_sha256\":\""<<fixture.resultSHA<<"\","
       <<"\n  \"candidate_nodes_sha256\":\""<<fixture.candidateSHA<<"\","
       <<"\n  \"world_fingerprint_hex\":\""<<fingerprintHex(fixture.worldFingerprint)<<"\","
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
       <<"\n  \"historical_capture_matrix_comparisons\":"<<historicalValid*81u<<","
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
void cpu(const std::filesystem::path& run,const FixtureBinding& fixture){
    testSHA256Validation();
    validateFixtureHashes(fixture);
    const auto output=run/"output";
    const std::string step=sixDigitStep(fixture.conditioningStep);
    const auto conditionDir=output/("rejected-conditioning-"+step);
    const auto candidatePath=conditionDir/"candidate_nodes.bin";
    const auto conditionPath=conditionDir/"manifest.json";
    const auto statePath=output/("material_state_"+step+"_bent.json");
    need(sha256(run/"operator-capture")==fixture.captureSHA,"capture SHA admission failed");
    need(sha256(output/"compiled.nmatterpack")==fixture.packageSHA,"package SHA admission failed");
    need(sha256(candidatePath)==fixture.candidateSHA,"candidate SHA admission failed");
    need(sha256(conditionPath)==fixture.conditionManifestSHA,"conditioning manifest SHA admission failed");
    need(sha256(statePath)==fixture.acceptedSHA,"accepted-state SHA admission failed");
    need(sha256(output/"manifest.json")==fixture.runManifestSHA,"run manifest SHA admission failed");
    need(sha256(run/"inputs/external/source-build.json")==fixture.sourceBuildSHA,"source-build SHA admission failed");
    need(sha256(run/"receipt.json")==fixture.receiptSHA,"execution receipt SHA admission failed");
    need(sha256(output/"result.json")==fixture.resultSHA,"native result SHA admission failed");
    NSDictionary* sourceBuild=asObject(readJSON(run/"inputs/external/source-build.json"),"source-build JSON");
    need(stringValue(field(sourceBuild,"source_snapshot"),"source snapshot")==fixture.sourceSnapshot,"source snapshot admission failed");
    Capture cap=readCapture(run/"operator-capture",fixture);
    need(cap.nodeRecords==1764&&cap.tetRecords==3840,"capture completeness test failed");
    need([(NSNumber*)field(field(cap.header,"cached_algorithmic_tangent_capture"),"policy_enabled") boolValue]&&
         [(NSNumber*)field(field(cap.header,"cached_algorithmic_tangent_capture"),"available") boolValue],
         "capture does not contain the production tangent cache");
    CompiledWorld w;std::string error;
    need(readPackage(output/"compiled.nmatterpack",w,nullptr,&error),"compiled package rejected: "+error);
    need(w.objects.size()==1u&&w.parameters.size()==59u,"compiled object/parameter layout mismatch: objects="+std::to_string(w.objects.size())+" parameters="+std::to_string(w.parameters.size()));
    need(w.dispatch.environmentCount==kEnvironments&&w.dispatch.femNodeCount==kNodes&&w.dispatch.tetrahedronCount==kTets,"compiled package dimensions mismatch");
    need(w.fingerprint==fixture.worldFingerprint,"compiled package fingerprint mismatch");
    need(w.materials.size()==3u&&w.materials[0].stateCount==15u&&w.materials[1].stateCount==15u&&w.materials[2].stateCount==0u,"compiled material/state layout mismatch");
    need(w.materials[0].flags==213u&&w.materials[1].flags==213u,"compiled source-15 policy flags differ from capture");
    const double captureDt=number(field(cap.header,"dt_seconds"),"capture dt");
    need(std::abs(captureDt-double(w.dispatch.gravityAndTimestep.w))<=1e-10,"capture/package timestep mismatch");
    NSArray* parameterEnvironments=(NSArray*)field(cap.header,"parameters_by_environment");
    need(parameterEnvironments.count==kEnvironments,"capture parameter environment count mismatch");
    for(unsigned env=0;env<kEnvironments;++env){
        auto values=readFlatFloats(parameterEnvironments[env],w.parameters.size(),"capture parameters");
        for(std::size_t pi=0;pi<values.size();++pi)
            need(std::bit_cast<std::uint32_t>(values[pi])==std::bit_cast<std::uint32_t>(w.parameters[pi].valueAndBounds.x),
                 "capture/package parameter word mismatch env="+std::to_string(env)+" index="+std::to_string(pi));
    }
    auto candidate=raw<NMFEMNodeStateGPU>(candidatePath,std::size_t(kEnvironments)*kNodes);
    verifyCaptureNodeWords(run/"operator-capture",candidate);
    verifyAcceptedStateSidecar(cap,statePath);
    need(std::string(NUMI_TANGENT_CACHE_SOURCE125_MANIFEST_SHA256).size()==64,"source binding macro malformed");
    std::cout<<"{\"record\":\"cpu_admission\",\"fixture_id\":\""<<fixture.label<<"\",\"status\":\"pass\",\"capture_rows\":3840,\"node_rows\":1764,\"object_count\":1,\"parameter_count\":59,\"physical_steps\":0}\n";
}
}
int main(int argc,char**argv){@autoreleasepool{try{
    if(argc==3&&std::string(argv[1])=="--cpu-admission"){cpu(argv[2],kFixture360);return 0;}
    if(argc==3&&std::string(argv[1])=="--cpu-admission-419"){cpu(argv[2],kFixture419);return 0;}
    if(argc==4&&std::string(argv[1])=="--native"){execute(argv[2],argv[3],kFixture360);return 0;}
    if(argc==4&&std::string(argv[1])=="--native-419"){execute(argv[2],argv[3],kFixture419);return 0;}
    std::cerr<<"usage: checker --cpu-admission RUN360 | --cpu-admission-419 RUN419 | --native RUN360 OUTPUT_DIR | --native-419 RUN419 OUTPUT_DIR\n";
    return 2;
}catch(const std::exception& e){std::cerr<<"tangent-cache checker: "<<e.what()<<"\n";return 1;}}}
