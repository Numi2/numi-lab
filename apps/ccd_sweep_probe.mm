#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/engine_types.h"
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
using Bytes=std::vector<std::byte>;
Bytes read(const char* name){
    std::ifstream f(name,std::ios::binary|std::ios::ate);require(bool(f),"Missing snapshot");
    auto n=f.tellg();require(n>=32&&n<=1024*1024,"Snapshot outside byte bound");Bytes bytes{std::size_t(n)};
    f.seekg(0);f.read(reinterpret_cast<char*>(bytes.data()),n);require(bool(f),"Truncated snapshot");return bytes;
}
template<class T>std::vector<T> typed(const Bytes& bytes){
    require(!bytes.empty()&&bytes.size()%sizeof(T)==0,"Typed segment size mismatch");
    std::vector<T> values(bytes.size()/sizeof(T));std::memcpy(values.data(),bytes.data(),bytes.size());return values;
}
void floats(mr_float4 v){std::cout<<'['<<v.x<<','<<v.y<<','<<v.z<<','<<v.w<<']';}
void record(const MRCCDPairGPU& r){
    std::cout<<"{\"environment\":"<<r.environment<<",\"compiled_pair\":"<<r.compiledPair
      <<",\"colliders\":["<<r.colliderA<<','<<r.colliderB<<"],\"flags\":"<<r.flags<<",\"status\":"<<r.status
      <<",\"interval_distance\":";floats(r.intervalAndDistance);
    std::cout<<",\"normal_iteration\":";floats(r.normalAndIteration);std::cout<<'}';
}
void event(const MRCCDEventStateGPU& e){
    std::cout<<"{\"environment\":"<<e.environment<<",\"splits\":"<<e.splitCount<<",\"zero_time_replays\":"<<e.zeroTimeReplayCount
      <<",\"flags\":"<<e.flags<<",\"generation\":"<<e.generation<<",\"time\":";floats(e.time);std::cout<<'}';
}
int main(int argc,char** argv){@autoreleasepool{try{
    require(argc==5,"Usage: metalrobo_ccd_sweep_probe PRODUCTION_METALLIB QUERY_METALLIB SNAPSHOT NEW_OUTPUT");
    require(!std::filesystem::exists(argv[4]),"Output must be new");const auto bytes=read(argv[3]);
    require(std::memcmp(bytes.data(),"CCDSNP01",8)==0,"Wrong snapshot magic");
    std::array<unsigned,6> h{};std::memcpy(h.data(),bytes.data()+8,24);
    require((h[0]==1&&h[5]==9)||(h[0]==2&&h[5]==11),"Unsupported snapshot header");
    std::map<std::string,Bytes> segments;std::size_t offset=32;
    for(unsigned i=0;i<h[5];++i){
        require(offset<=bytes.size()&&bytes.size()-offset>=16,"Truncated segment header");
        char tag[9]{};std::memcpy(tag,bytes.data()+offset,8);std::uint64_t n{};std::memcpy(&n,bytes.data()+offset+8,8);offset+=16;
        require(n<=bytes.size()-offset,"Truncated segment");
        require(!segments.contains(tag),"Duplicate segment");segments[tag]=Bytes(bytes.begin()+offset,bytes.begin()+offset+n);offset+=n;
    }
    require(offset==bytes.size(),"Trailing snapshot bytes");
    const auto dispatch=typed<MRMetalWorldContactDispatchGPU>(segments.at("dispatch"));
    const auto status=typed<MRMetalWorldContactStatusGPU>(segments.at("status"));
    auto shapes=typed<MRShapeGPU>(segments.at("shapes"));
    const auto pairs=typed<MRCompiledCollisionPairGPU>(segments.at("pairs"));
    auto current=typed<MRProjectedColliderGPU>(segments.at("current"));
    auto future=typed<MRProjectedColliderGPU>(segments.at("future"));
    const auto candidates=typed<MRCCDPairGPU>(segments.at("ccdPairs"));
    auto a=typed<MRCCDEventStateGPU>(segments.at("eventA")),b=typed<MRCCDEventStateGPU>(segments.at("eventB"));
    require(dispatch.size()==1&&status.size()==1&&a.size()==1&&b.size()==1,"Wrong singleton sizes");
    const auto& d=dispatch[0];const auto& s=status[0];
    require(d.shapeCount==shapes.size()&&current.size()==shapes.size()&&future.size()==shapes.size(),"Shape counts differ");
    require(d.eligiblePairCount==pairs.size()&&d.ccdCandidateCapacity==candidates.size(),"Pair counts differ");
    require(h[1]==s.environment&&h[4]==s.firstFailingPair&&((h[0]==1&&h[2]==s.controlStep&&h[3]==s.physicsSubstep)||h[0]==2),"Failure header differs");
    require(h[1]<d.environmentCount&&h[4]<pairs.size()&&s.ccdCandidates<=candidates.size(),"Invalid failure range");
    const auto pair=pairs[h[4]];require(pair.colliderA<shapes.size()&&pair.colliderB<shapes.size(),"Invalid shape indices");
    require((shapes[pair.colliderA].shapeType==MR_SHAPE_BOX&&shapes[pair.colliderB].shapeType==MR_SHAPE_SPHERE)||
      (shapes[pair.colliderB].shapeType==MR_SHAPE_BOX&&shapes[pair.colliderA].shapeType==MR_SHAPE_SPHERE),"Probe requires analytic sphere-box geometry");
    const MRCCDPairGPU* stored=nullptr;
    for(unsigned i=0;i<s.ccdCandidates;++i)if(candidates[i].compiledPair==h[4])stored=&candidates[i];
    MRCCDFailureSnapshotGPU protectedSnapshot{};
    if(h[0]==2){
        const auto saved=typed<MRCCDFailureSnapshotGPU>(segments.at("failed"));
        const auto world=typed<MRMetalWorldStatusGPU>(segments.at("world"));
        require(saved.size()==1&&world.size()==1,"Wrong protected snapshot/world size");
        require(world[0].code==s.code&&world[0].environment==h[1]&&world[0].controlStep==h[2]&&world[0].failingSubstep==h[3]&&world[0].failingIndex==h[4],"World failure stamp differs");
        protectedSnapshot=saved[0];const auto& p=protectedSnapshot;
        require(p.validAndStamp.x==MR_CCD_FAILURE_SNAPSHOT_TAG&&p.candidate.environment==h[1]&&p.candidate.compiledPair==h[4]&&
            p.candidate.colliderA==pair.colliderA&&p.candidate.colliderB==pair.colliderB,"Protected candidate identity differs");
        require((p.candidate.flags&MR_CCD_PAIR_UNRESOLVED)!=0&&(p.candidate.flags&MR_CCD_PAIR_SPECULATIVE_FALLBACK)==0,"Protected record is not unsafe unresolved CCD");
        shapes[pair.colliderA]=p.sourceA;shapes[pair.colliderB]=p.sourceB;
        current[pair.colliderA]=p.startA;current[pair.colliderB]=p.startB;future[pair.colliderA]=p.endA;future[pair.colliderB]=p.endB;
        a[0]=p.event;stored=&protectedSnapshot.candidate;
    }
    require(stored!=nullptr,"Reported failed candidate absent");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();require(device!=nil,"Metal unavailable");NSError* error=nil;
    auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]] error:&error];require(library!=nil,"Cannot load sweep metallib");
    std::cerr<<"distance_pipeline_begin\n";auto function=[library newFunctionWithName:@"mr_ccd_sweep_probe"];require(function!=nil,"Missing sweep kernel");
    auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];require(pipeline!=nil,"Cannot create sweep pipeline");std::cerr<<"distance_pipeline_ready\n";
    auto production=[device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];require(production!=nil,"Cannot load production metallib");
    auto resolverFunction=[production newFunctionWithName:@"mr_world_resolve_ccd"];require(resolverFunction!=nil,"Missing production resolver");
    auto resolver=[device newComputePipelineStateWithFunction:resolverFunction error:&error];require(resolver!=nil,"Cannot create resolver pipeline");std::cerr<<"resolver_pipeline_ready\n";
    auto buffer=[&](const void* data,std::size_t size){auto x=[device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];require(x!=nil,"GPU allocation failed");return x;};
    const MRGeometryHeaderGPU gh{};const mr_float4 vertex{};const MRMeshBVHNodeGPU node{};const MRMeshTriangleGPU triangle{};
    std::array<id<MTLBuffer>,14> buffers{};
    buffers[0]=buffer(dispatch.data(),sizeof(MRMetalWorldContactDispatchGPU));
    buffers[1]=buffer(shapes.data(),shapes.size()*sizeof(MRShapeGPU));
    buffers[2]=buffer(current.data(),current.size()*sizeof(MRProjectedColliderGPU));
    buffers[3]=buffer(future.data(),future.size()*sizeof(MRProjectedColliderGPU));
    buffers[4]=buffer(pairs.data(),pairs.size()*sizeof(MRCompiledCollisionPairGPU));
    buffers[5]=buffer(a.data(),sizeof(MRCCDEventStateGPU));buffers[6]=buffer(b.data(),sizeof(MRCCDEventStateGPU));
    buffers[7]=[device newBufferWithLength:2*sizeof(MRCCDPairGPU) options:MTLResourceStorageModeShared];
    buffers[8]=[device newBufferWithLength:257*sizeof(mr_float4) options:MTLResourceStorageModeShared];
    const mr_uint4 ids{h[1],h[4],0,0};buffers[9]=buffer(&ids,sizeof(ids));buffers[10]=buffer(&gh,sizeof(gh));
    buffers[11]=buffer(&vertex,sizeof(vertex));buffers[12]=buffer(&node,sizeof(node));buffers[13]=buffer(&triangle,sizeof(triangle));
    for(auto x:buffers)require(x!=nil,"GPU output allocation failed");
    // Preserve original environment/pair identity and the complete dispatch. Only
    // this pair participates; other lanes are blocked before projection reads.
    auto replayDispatch=d;replayDispatch.flags&=~MR_METAL_WORLD_CONTACT_CAPTURE_CCD_FAILURE;
    std::vector<MRProjectedColliderGPU> projected(std::size_t(d.environmentCount)*d.shapeCount),predicted(projected.size());
    std::copy(current.begin(),current.end(),projected.begin()+std::size_t(h[1])*d.shapeCount);
    std::copy(future.begin(),future.end(),predicted.begin()+std::size_t(h[1])*d.shapeCount);
    std::vector<mr_u32> overlaps(std::size_t(d.environmentCount)*d.eligiblePairCount);
    overlaps[std::size_t(h[1])*d.eligiblePairCount+h[4]]=1;
    std::array<MRCCDPairGPU,2> replay{};auto queue=[device newCommandQueue];
    for(unsigned pass=0;pass<2;++pass){
        std::vector<MRMetalWorldContactStatusGPU> statuses(d.environmentCount);
        for(auto& status:statuses)status.code=MR_STEP_UNSUPPORTED;
        statuses[h[1]]={};statuses[h[1]].environment=h[1];
        std::vector<MRCCDEventStateGPU> events(d.environmentCount);events[h[1]]=pass?b[0]:a[0];
        std::vector<MRCCDPairGPU> candidates(std::size_t(d.environmentCount)*d.ccdCandidateCapacity);
        std::array<id<MTLBuffer>,13> rb{buffer(&replayDispatch,sizeof(replayDispatch)),buffers[1],buffers[4],
            buffer(overlaps.data(),overlaps.size()*sizeof(mr_u32)),
            buffer(projected.data(),projected.size()*sizeof(MRProjectedColliderGPU)),
            buffer(predicted.data(),predicted.size()*sizeof(MRProjectedColliderGPU)),
            buffers[10],buffers[11],buffers[12],buffers[13],
            buffer(candidates.data(),candidates.size()*sizeof(MRCCDPairGPU)),
            buffer(statuses.data(),statuses.size()*sizeof(MRMetalWorldContactStatusGPU)),
            buffer(events.data(),events.size()*sizeof(MRCCDEventStateGPU))};
        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:resolver];for(unsigned i=0;i<rb.size();++i)[encoder setBuffer:rb[i] offset:0 atIndex:i];
        [encoder dispatchThreads:MTLSizeMake(d.environmentCount,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];[encoder endEncoding];
        [command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"Resolver GPU execution failed");
        std::memcpy(&replay[pass],static_cast<const char*>(rb[10].contents)+std::size_t(h[1])*d.ccdCandidateCapacity*sizeof(MRCCDPairGPU),sizeof(MRCCDPairGPU));
    }
    auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];for(unsigned i=0;i<buffers.size();++i)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
    [encoder dispatchThreads:MTLSizeMake(257,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
    [encoder endEncoding];[command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"Sweep GPU execution failed");
    std::ofstream output(argv[4],std::ios::binary);output.write(reinterpret_cast<const char*>(replay.data()),sizeof(replay));
    output.write(static_cast<const char*>(buffers[8].contents),257*sizeof(mr_float4));require(bool(output),"Cannot write replay stream");
    std::cout<<std::setprecision(9)<<"{\"schema\":\"numi.ccd-sweep-probe.v1\",\"header\":[";
    for(unsigned i=0;i<h.size();++i)std::cout<<(i?",":"")<<h[i];
    std::cout<<"],\"record_source\":\""<<(h[0]==2?"protected-first-failure":"post-submission")<<"\",\"stored_record\":";record(*stored);std::cout<<",\"replay_records\":[";record(replay[0]);std::cout<<',';record(replay[1]);
    std::cout<<"],\"stored_record_matches\":["<<(std::memcmp(stored,&replay[0],sizeof(*stored))==0?"true":"false")<<','
      <<(std::memcmp(stored,&replay[1],sizeof(*stored))==0?"true":"false")<<"],\"events\":[";event(a[0]);std::cout<<',';event(b[0]);
    std::cout<<"],\"maximum_CA_iterations\":"<<d.maxConservativeAdvancementIterations<<",\"ccd_parameters\":";floats(d.ccdParameters);
    std::cout<<",\"status_event_times\":";floats(s.eventTimes);
    std::cout<<",\"status_counters\":{\"advance\":"<<s.ccdAdvanceCount<<",\"zero_time\":"<<s.zeroTimeCCDReplays
      <<",\"unresolved\":"<<s.unresolvedCCDCount<<",\"ccd_candidates\":"<<s.ccdCandidates<<"},\"shapes\":[";
    for(unsigned i=0;i<2;++i){const unsigned j=i?pair.colliderB:pair.colliderA;const auto& shape=shapes[j];if(i)std::cout<<',';
      std::cout<<"{\"index\":"<<j<<",\"type\":"<<shape.shapeType<<",\"dimensions\":";floats(shape.dimensions);
      std::cout<<",\"contact_rest_radius\":";floats(shape.contactRestAndBoundingRadius);
      for(unsigned e=0;e<2;++e){const auto& p=e?future[j]:current[j];std::cout<<(e?",\"end\":{":",\"start\":{")<<"\"center_radius\":";floats(p.centerAndRadius);
        std::cout<<",\"rotation\":";floats(p.rotation);std::cout<<",\"sweep_margin\":"<<std::bit_cast<float>(p.statusAndFlags.z)<<'}';}
      std::cout<<'}';}
    std::cout<<"]}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}}
