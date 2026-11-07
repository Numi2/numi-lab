#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/RetinalStream.hpp"
#include "metalrobo/task_program_types.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>

namespace {
using namespace metalrobo;
static_assert(sizeof(MRTaskStateGPU)==13*sizeof(mr_uint4));
constexpr unsigned N=3,SW=16,SH=12,W=8,H=6,EYES=2,STEPS=50;
constexpr unsigned FEATURES=EYES*W*H*3+2,OFFSET=3,STRIDE=FEATURES+5;
void require(bool yes,const std::string& why) {if (!yes) throw std::runtime_error(why);}
const char* fixtureSource=R"(
#include <metal_stdlib>
using namespace metal;
kernel void fixture(device uint4* tasks [[buffer(0)]],device float4* left [[buffer(1)]],
    device float4* right [[buffer(2)]],constant uint4& p [[buffer(3)]],uint at [[thread_position_in_grid]]) {
    if (at>=3*16*12) return;
    const uint env=at/(16*12),pixel=at%(16*12),x=pixel%16,y=pixel/16;
    const bool reset=env==1 && p.x>=17;
    if (pixel==0) {tasks[env*13]=uint4(reset ? p.x-17 : p.x,reset ? 1 : 0,0,0);tasks[env*13+2]=uint4(1,0,0,0);}
    const float tag=float(p.x+p.y*100);
    float4 a=float4((tag+1)/256.0f,float(env+1)/8.0f+(float(x)+.5f)/128.0f,1.0f/8.0f+(float(y)+.5f)/128.0f,1);
    float4 b=a;b.z+=1.0f/8.0f;
    if (p.z==2 && p.x==4 && env==0 && pixel==0) a.x=NAN;
    if (p.z==3 && p.x==4 && env==0) a.x=as_type<float>(0x7f7fffffu);
    left[at]=a;right[at]=b;
})";
struct Trial {
    unsigned numerator=3,denominator=10,latency=2;
    std::vector<float> output;
    RetinalStreamSnapshot snapshot;
    std::vector<std::array<unsigned,2>> calls;
};
Trial run(unsigned chunk,unsigned mode=0,unsigned numerator=3,unsigned denominator=10,unsigned latency=2) {
    RetinalStreamConfig c;c.environments=N;c.sourceWidth=SW;c.sourceHeight=SH;c.width=W;c.height=H;
    c.sampleNumerator=numerator;c.sampleDenominator=denominator;c.latencyTicks=latency;
    RetinalStreamContext stream(c);require(stream.featureCount()==FEATURES,"Retinal feature contract mismatch");
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();NSError* error=nil;
    id<MTLLibrary> fixture=[device newLibraryWithSource:[NSString stringWithUTF8String:fixtureSource] options:nil error:&error];
    require(fixture!=nil,"Native retinal fixture compilation failed");
    id<MTLComputePipelineState> pipeline=[device newComputePipelineStateWithFunction:[fixture newFunctionWithName:@"fixture"] error:&error];
    require(pipeline!=nil,"Native retinal fixture pipeline failed");
    id<MTLCommandQueue> queue=[device newCommandQueue];
    id<MTLBuffer> tasks=[device newBufferWithLength:N*sizeof(MRTaskStateGPU) options:MTLResourceStorageModeShared];
    std::memset(tasks.contents,0,tasks.length);
    id<MTLBuffer> left=[device newBufferWithLength:N*SW*SH*sizeof(mr_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> right=[device newBufferWithLength:left.length options:MTLResourceStorageModeShared];
    std::array<void*,2> eyes{{(__bridge void*)left,(__bridge void*)right}};
    Trial result;result.numerator=numerator;result.denominator=denominator;result.latency=latency;
    if (mode==1) result.calls={{{0,0}},{{2,0}}};
    else for (unsigned tick=0;tick<(mode>=2 ? 10u : STEPS);++tick) {
        result.calls.push_back({tick,0});
        if (tick==7 && mode==0) result.calls.push_back({tick,1});
    }
    result.output.assign(result.calls.size()*N*STRIDE,-777.0f);
    id<MTLBuffer> output=[device newBufferWithBytes:result.output.data() length:result.output.size()*sizeof(float) options:MTLResourceStorageModeShared];
    require(queue!=nil && tasks!=nil && left!=nil && right!=nil && output!=nil,"Retinal fixture allocation failed");
    // Invalid calls must reject before submitting or mutating any stream state.
    id<MTLCommandBuffer> invalid=[queue commandBuffer];
    require(!stream.encode((__bridge void*)invalid,(__bridge void*)tasks,eyes,(__bridge void*)output,FEATURES-1,0),"Short retinal output accepted");
    require(!stream.encode((__bridge void*)invalid,(__bridge void*)tasks,eyes,nullptr,STRIDE,OFFSET),"Null retinal output accepted");
    require(!stream.encode((__bridge void*)invalid,nullptr,eyes,(__bridge void*)output,STRIDE,OFFSET),"Null retinal task clock accepted");
    require(!stream.encode((__bridge void*)invalid,(__bridge void*)tasks,std::span<void* const>(eyes.data(),1),(__bridge void*)output,STRIDE,OFFSET),"Incomplete retinal eyes accepted");
    for (unsigned first=0;first<result.calls.size();first+=chunk) {
        id<MTLCommandBuffer> command=[queue commandBuffer];
        for (unsigned call=first;call<std::min<unsigned>(first+chunk,result.calls.size());++call) {
            const auto fixtureCall=result.calls[call];const mr_uint4 p{fixtureCall[0],fixtureCall[1],mode,0};
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:tasks offset:0 atIndex:0];
            [encoder setBuffer:left offset:0 atIndex:1];[encoder setBuffer:right offset:0 atIndex:2];
            [encoder setBytes:&p length:sizeof(p) atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(N*SW*SH,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
            require(stream.encode((__bridge void*)command,(__bridge void*)tasks,eyes,(__bridge void*)output,STRIDE,OFFSET,
                std::size_t(call)*N*STRIDE),"Retinal encode rejected admitted fixture");
        }
        [command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"Retinal native fixture GPU failure");
    }
    // One publication boundary, never a host-side sensory loop.
    std::memcpy(result.output.data(),output.contents,output.length);result.snapshot=stream.snapshot();return result;
}
double admit(const Trial& trial) {
    struct Sample {unsigned step,tag;};
    std::array<std::vector<Sample>,N> captures;
    std::array<unsigned,N> previous{{~0u,~0u,~0u}},epoch{{~0u,~0u,~0u}};
    double maxError=0;
    for (unsigned call=0;call<trial.calls.size();++call) for (unsigned env=0;env<N;++env) {
        const auto [tick,variant]=trial.calls[call];const bool reset=env==1 && tick>=17;
        const unsigned step=reset ? tick-17 : tick,index=reset ? 1 : 0;
        const bool fresh=epoch[env]!=index;
        if (fresh) {captures[env].clear();epoch[env]=index;}
        if (fresh || (step!=previous[env] && std::uint64_t(step)*trial.numerator/trial.denominator!=std::uint64_t(previous[env])*trial.numerator/trial.denominator))
            captures[env].push_back({step,tick+variant*100});
        previous[env]=step;
        const Sample* delivered=nullptr;
        for (const auto& sample:captures[env]) if (step>=trial.latency && sample.step<=step-trial.latency) delivered=&sample;
        const auto base=std::size_t(call)*N*STRIDE+env*STRIDE;
        for (unsigned feature=0;feature<STRIDE;++feature) {
            const float actual=trial.output[base+feature];require(std::isfinite(actual),"Retinal output is non-finite");
            if (feature<OFFSET || feature>=OFFSET+FEATURES) {require(actual==-777,"Retinal stream overwrote another actor feature");continue;}
            const unsigned at=feature-OFFSET;double expected=0;
            if (delivered && at<FEATURES-2) {
                const unsigned channel=at%3,pixel=(at/3)%(W*H),eye=(at/3)/(W*H),x=pixel%W,y=pixel/W;
                const double radiance=channel==0 ? double(delivered->tag+1)/256 : channel==1 ? double(env+1)/8+(2.0*x+1)/128 : double(eye+1)/8+(2.0*y+1)/128;
                expected=radiance/(1+radiance);
            }
            if (delivered && at==FEATURES-2) expected=1;
            if (delivered && at==FEATURES-1) expected=double(step-delivered->step)*.02;
            maxError=std::max(maxError,std::abs(actual-expected));
        }
    }
    require(maxError<=3e-6,"Retinal samples/cadence/latency disagree with independent FP64 oracle");
    for (unsigned env=0;env<N;++env) {
        const auto& s=trial.snapshot.states[env];
        require(trial.snapshot.errors[env]==0,"Valid retinal fixture produced a stream error");
        require(s.delivery.w==captures[env].size(),"Retinal capture count mismatch");
        require(s.timeline.x==previous[env] && s.timeline.y==epoch[env],"Retinal accepted clock mismatch");
    }
    return maxError;
}
void binary(const std::filesystem::path& path,const void* data,std::size_t bytes) {
    std::ofstream f(path,std::ios::binary);f.exceptions(std::ios::badbit|std::ios::failbit);f.write(static_cast<const char*>(data),bytes);
}
}
int main(int argc,char** argv) {@autoreleasepool {
    bool owns=false;
    try {
        require(argc==2,"Usage: metalrobo_retinal_stream_check NEW_OUTPUT_DIRECTORY");
        const std::filesystem::path output=argv[1];require(std::filesystem::create_directory(output),"Retinal evidence path must be new");owns=true;
        std::ofstream(output/"plan.json")<<R"({"schema":"numi.retinal-stream-admission.v2","question":"Do native RGB sample-and-hold, integer cadence and delayed publication remain causal across repeats, per-lane episode resets and command partitions?","unit":"controlled GPU input fixture, three lanes; not embodied learning","steps":50,"control_hz":50,"sample_rate_ratio":[3,10],"latency_ticks":2,"eyes":2,"source_dimensions":[16,12],"actor_dimensions":[8,6],"input":"linear RGB and accepted TaskState episode clock only; no depth, IDs or world coordinates","response":"fixed c/(1+c) per channel after exact area reduction","predictions":{"FP64_max_absolute_feature_error":0.000003,"chunk_1_8_bit_mismatches":0,"guard_feature_writes":0,"duplicate_tick_new_samples":0,"premature_or_stale_reset_frames":0},"parent":"01 passed original timing controls; this prospective extension adds parameter boundaries and finite-radiance overflow rejection with the same feature-error threshold","boundary_conditions":[[1,1,0],[1,1,1],[1,1,16],[1,50,2]],"negative_controls":["invalid configuration","finite RGB reduction overflow","short/null output","null clock","missing eye","accepted clock gap poisons stream","nonfinite captured image poisons affected lane"],"stopping":"two 51-call partition runs, four cadence/delay boundary runs, one clock-gap run and two 10-call invalid image runs; no retries or threshold changes","boundary":"synthetic native timing instrument; no renderer-to-actor integration, biological eye or ecological qualification"})";
        unsigned rejectedConfigurations=0;
        for (unsigned badCase=0;badCase<7;++badCase) {
            RetinalStreamConfig c;c.environments=N;c.sourceWidth=SW;c.sourceHeight=SH;c.width=W;c.height=H;
            if (badCase==0) c.sampleNumerator=0;
            if (badCase==1) c.sampleNumerator=11;
            if (badCase==2) c.sourceWidth=17;
            if (badCase==3) c.latencyTicks=17;
            if (badCase==4) c.maximumRetainedBytes=0;
            if (badCase==5) c.sourceWidth=5000;
            if (badCase==6) c.tickSeconds=std::numeric_limits<float>::quiet_NaN();
            try {RetinalStreamContext invalid(c);} catch (const std::invalid_argument&) {++rejectedConfigurations;}
        }
        require(rejectedConfigurations==7,"Invalid retinal configuration was accepted");
        const auto a=run(1),b=run(8);double error=std::max(admit(a),admit(b));
        std::ofstream boundaries(output/"boundary-measurements.jsonl");
        for (const auto condition : std::array<std::array<unsigned,3>,4>{{{1,1,0},{1,1,1},{1,1,16},{1,50,2}}}) {
            const auto trial=run(8,0,condition[0],condition[1],condition[2]);
            const auto comparison=admit(trial);error=std::max(error,comparison);
            boundaries<<std::setprecision(17)<<"{\"numerator\":"<<condition[0]<<",\"denominator\":"<<condition[1]
                <<",\"latency_ticks\":"<<condition[2]<<",\"maximum_feature_error\":"<<comparison<<"}\n";
            binary(output/("boundary-"+std::to_string(condition[1])+"-"+std::to_string(condition[2])+".output.f32"),trial.output.data(),trial.output.size()*sizeof(float));
        }
        require(a.output==b.output && a.snapshot.errors==b.snapshot.errors &&
            std::memcmp(a.snapshot.states.data(),b.snapshot.states.data(),N*sizeof(MRRetinalStateGPU))==0,"Retinal state depends on command partition");
        binary(output/"chunk1.output.f32",a.output.data(),a.output.size()*sizeof(float));
        binary(output/"chunk8.output.f32",b.output.data(),b.output.size()*sizeof(float));
        binary(output/"final-states.bin",a.snapshot.states.data(),N*sizeof(MRRetinalStateGPU));
        const auto gap=run(2,1);
        require(std::all_of(gap.snapshot.errors.begin(),gap.snapshot.errors.end(),[](unsigned e){return e==1;}),"Accepted clock gap was not rejected");
        for (unsigned env=0;env<N;++env) for (unsigned feature=0;feature<FEATURES;++feature)
            require(gap.output[N*STRIDE+env*STRIDE+OFFSET+feature]==0,"Clock-gap stream exposed a valid frame");
        const auto nan=run(8,2);require(nan.snapshot.errors==std::vector<unsigned>{2,0,0},"Nonfinite RGB error escaped or crossed lanes");
        for (unsigned call=4;call<nan.calls.size();++call) for (unsigned feature=0;feature<FEATURES;++feature)
            require(nan.output[std::size_t(call)*N*STRIDE+OFFSET+feature]==0,"Nonfinite stream did not fail closed");
        const auto overflow=run(8,3);require(overflow.snapshot.errors==std::vector<unsigned>{2,0,0},"Finite radiance overflow escaped or crossed lanes");
        for (unsigned call=4;call<overflow.calls.size();++call) for (unsigned feature=0;feature<FEATURES;++feature)
            require(overflow.output[std::size_t(call)*N*STRIDE+OFFSET+feature]==0,"Radiance overflow stream did not fail closed");
        binary(output/"finite-overflow.output.f32",overflow.output.data(),overflow.output.size()*sizeof(float));
        binary(output/"clock-gap.output.f32",gap.output.data(),gap.output.size()*sizeof(float));
        binary(output/"nonfinite.output.f32",nan.output.data(),nan.output.size()*sizeof(float));
        std::ofstream(output/"layout.json")<<"{\"byte_order\":\"little-endian\",\"output_order\":\"call,environment,actor_feature\",\"calls\":51,\"environments\":3,\"actor_stride\":"<<STRIDE<<",\"feature_offset\":"<<OFFSET<<",\"retinal_feature_count\":"<<FEATURES<<",\"pixel_order\":\"eye,y,x,RGB\",\"tail_features\":[\"valid\",\"age_seconds\"],\"state_record_bytes\":"<<sizeof(MRRetinalStateGPU)<<"}\n";
        std::ostringstream errorText;errorText<<std::setprecision(17)<<error;
        const auto result=std::string("{\"schema\":\"numi.retinal-stream-evidence.v1\",\"status\":\"passed\",\"FP64_max_absolute_error\":")+errorText.str()+R"(,"exact_partition_invariance":true,"duplicates_idempotent":true,"per_lane_resets_clear_frames":true,"gap_and_nonfinite_fail_closed":true,"renderer_actor_integration":false,"ecological_environment_qualified":false})";
        std::ofstream(output/"summary.json")<<std::setprecision(17)<<result<<'\n';std::cout<<result<<'\n';return 0;
    } catch (const std::exception& e) {
        if (owns) std::ofstream(std::filesystem::path(argv[1])/"failure.json")<<"{\"error\":"<<std::quoted(e.what())<<"}\n";
        std::cerr<<e.what()<<'\n';return 1;
    }
}}
