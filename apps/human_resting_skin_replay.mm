#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include "NumiHumanRestingSkinInfluencePipeline.hpp"
#include "metalrobo/numi_human_resting_visual_gpu.h"
#include "numi/matter/human_respiration.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Offline geometry evidence only. Dispatch the existing, hash-pinned viewer
// kernel over its uploaded ABI records at retained accepted body/respiratory
// states. This tool never advances physics, physiology or controller history.
namespace {
void need(bool b,const char* why) { if(!b) throw std::runtime_error(why); }
std::string sha(NSData* b) {
    need(b.length<=std::numeric_limits<CC_LONG>::max(),"hash input exceeds bounded length");
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256(b.bytes,(CC_LONG)b.length,bytes);
    constexpr char digits[]="0123456789abcdef";std::string s;
    for(auto x:bytes){s+=digits[x>>4];s+=digits[x&15];}return s;
}
NSData* read(NSString* p,NSUInteger limit) {
    NSDictionary* a=[[NSFileManager defaultManager] attributesOfItemAtPath:p error:nil];
    need(a&&[a[NSFileType] isEqualToString:NSFileTypeRegular],"input is not a regular file");
    const auto n=[a[NSFileSize] unsignedLongLongValue];need(n>0&&n<=limit,"input exceeds bounded size");
    NSData* b=[NSData dataWithContentsOfFile:p];need(b&&b.length==n,"input read failed");return b;
}
NSDictionary* object(NSData* b) {
    id o=[NSJSONSerialization JSONObjectWithData:b options:0 error:nil];
    need([o isKindOfClass:[NSDictionary class]],"receipt must be a JSON object");return o;
}
NSData* pinned(NSDictionary* row,NSUInteger limit) {
    need([row isKindOfClass:[NSDictionary class]]&&[row[@"path"] isKindOfClass:[NSString class]]&&
        [row[@"sha256"] isKindOfClass:[NSString class]],"missing pinned ABI/source record");
    NSData* b=read(row[@"path"],limit);
    need(b.length==[row[@"bytes"] unsignedLongLongValue]&&sha(b)==[row[@"sha256"] UTF8String],
        "ABI/source file differs from receipt pin");return b;
}
float scalar(id x) {
    need([x isKindOfClass:[NSNumber class]],"pose component is not numeric");
    const double d=[x doubleValue];need(std::isfinite(d)&&std::isfinite((float)d),"nonfinite pose component");return (float)d;
}
std::uint64_t integer(id x) {
    need([x isKindOfClass:[NSNumber class]],"count/index is not numeric");
    const double d=[x doubleValue];need(std::isfinite(d)&&d>=0&&d<=double(UINT32_MAX)&&std::floor(d)==d,"invalid count/index");return (std::uint64_t)d;
}
void write(NSString* p,NSData* b) {
    need(![[NSFileManager defaultManager] fileExistsAtPath:p],"refusing to overwrite evidence");
    need([b writeToFile:p options:NSDataWritingAtomic error:nil],"evidence write failed");
    need(sha(read(p,b.length))==sha(b),"written evidence hash differs");
}
}
int main(int argc,char** argv) {
    @autoreleasepool { try {
        need(argc==6||argc==7,"usage: numi_human_resting_skin_replay SOURCE_RECEIPT SHA256 POSE_RECEIPT SHA256 OUTPUT_DIRECTORY [--retain-source-cardiac-context]");
        const bool retainSourceCardiacContext=argc==7;
        if(retainSourceCardiacContext)need(std::strcmp(argv[6],"--retain-source-cardiac-context")==0,"unknown replay option");
        NSString* rp=@(argv[1]);NSString* pp=@(argv[3]);NSString* out=@(argv[5]);
        NSData* rb=read(rp,8u*1024u*1024u);NSData* pb=read(pp,8u*1024u*1024u);
        need(sha(rb)==argv[2]&&sha(pb)==argv[4],"source or pose receipt hash mismatch");
        NSDictionary* source=object(rb);NSDictionary* pose=object(pb);
        NSDictionary* m=source[@"anatomy_source_mapping"];
        const bool completeMapping=[m isKindOfClass:[NSDictionary class]]&&m.count>0;
        if(!completeMapping)m=source[@"skin_source_mapping"];need([m isKindOfClass:[NSDictionary class]],"receipt lacks source skin mapping");
        const auto count=integer(m[@"source_vertex_count"]);need(count>0&&count<=8'000'000u,"invalid skin vertex count");
        need(integer(m[@"vertex_map_record_stride_bytes"])==sizeof(MRHumanRestingVertexMap)&&
            integer(m[@"influence_record_stride_bytes"])==sizeof(MRHumanRestingInfluence),"skin ABI stride differs");
        NSData* mb=pinned(m[@"vertex_map"],count*sizeof(MRHumanRestingVertexMap));
        NSData* ib=pinned(m[@"influences"],2ull*1024u*1024u*1024u-1u);
        NSData* ab=pinned(m[@"anatomy_parameters"],sizeof(MRHumanRestingAnatomyGPU));
        need(mb.length==count*sizeof(MRHumanRestingVertexMap)&&ab.length==sizeof(MRHumanRestingAnatomyGPU)&&
            ib.length%sizeof(MRHumanRestingInfluence)==0,"skin ABI lengths differ");
        auto* anatomy=(const MRHumanRestingAnatomyGPU*)ab.bytes;
        const auto base=integer(m[@"influence_first_index"]);const auto ninf=ib.length/sizeof(MRHumanRestingInfluence);
        const bool tiled=[m[@"influence_layout"] isEqual:@"tile32"];
        need(tiled||[m[@"influence_layout"] isEqual:@"vertex-major"],"unknown skin influence layout");

        std::vector<MRHumanRestingVertexMap> maps(count);std::memcpy(maps.data(),mb.bytes,mb.length);
        const auto* influences=(const MRHumanRestingInfluence*)ib.bytes;
        NSArray* bodyRows=pose[@"accepted_body_poses"];need([bodyRows isKindOfClass:[NSArray class]]&&
            bodyRows.count>0&&bodyRows.count<=1024u,"missing complete accepted body poses");
        std::vector<MRBodyStateGPU> bodies(bodyRows.count);std::vector<bool> seen(bodyRows.count,false);
        for(NSDictionary* row in bodyRows) {
            const auto i=integer(row[@"body_index"]);need(i<bodies.size()&&!seen[i],"duplicate or incomplete body indices");seen[i]=true;
            NSArray* p=row[@"position_m"];NSArray* q=row[@"quaternion_xyzw"];
            need([p isKindOfClass:[NSArray class]]&&p.count==3&&[q isKindOfClass:[NSArray class]]&&q.count==4,"invalid body pose");
            bodies[i].position={scalar(p[0]),scalar(p[1]),scalar(p[2]),0};
            bodies[i].orientation={scalar(q[0]),scalar(q[1]),scalar(q[2]),scalar(q[3])};
            const auto qq=bodies[i].orientation;
            const double norm=double(qq.x)*qq.x+double(qq.y)*qq.y+double(qq.z)*qq.z+double(qq.w)*qq.w;
            need(std::abs(norm-1)<.001,"body quaternion is outside native normalization tolerance");
        }
        need(anatomy->bodyAndFlags.x<bodies.size(),"anatomical torso body is absent");
        for(auto& map:maps) {
            need(map.deformationKind<=12u,"unsupported native deformation kind");
            if(!map.influenceCount) {need(completeMapping,"source skin lacks positive influences");continue;}
            need((completeMapping||map.deformationKind==3u)&&map.firstInfluence>=base,
                "source owner influence interval is invalid");
            const std::size_t stride=tiled&&map.deformationKind==3u?32u:1u;
            const std::size_t first=map.firstInfluence-base;
            need(first<ninf&&std::size_t(map.influenceCount-1u)<=(ninf-1-first)/stride,"skin influence index out of range");
            double sum=0;
            for(unsigned j=0;j<map.influenceCount;++j) {
                const auto& v=influences[first+j*stride];
                need(v.body.x==MR_INVALID_INDEX||v.body.x<bodies.size(),"skin influence body is absent");
                need(v.body.y==0u||v.body.y==MR_HUMAN_RESTING_COMPENSATED_POSITION,"unknown skin position accumulation mode");
                need(std::isfinite(v.positionAndWeight.x)&&std::isfinite(v.positionAndWeight.y)&&std::isfinite(v.positionAndWeight.z)&&
                    std::isfinite(v.positionAndWeight.w)&&v.positionAndWeight.w>0,"invalid positive skin influence");
                sum+=v.positionAndWeight.w;
            }
            need(std::abs(sum-1)<.0001,"skin positive weight sum differs from native owner");
            map.firstInfluence=(std::uint32_t)first;
        }
        NSDictionary* motion=pose[@"accepted_respiratory_motion"];
        need([motion isKindOfClass:[NSDictionary class]],"missing accepted respiratory motion");
        NMHumanRespirationState respiration{};
        respiration.motion={scalar(motion[@"diaphragm_swept_volume_m3"]),scalar(motion[@"rib_swept_volume_m3"]),
            scalar(motion[@"diaphragm_volume_velocity_m3_s"]),scalar(motion[@"rib_volume_velocity_m3_s"])};
        NSDictionary* kernel=m[@"kernel"];id<MTLDevice> device=MTLCreateSystemDefaultDevice();
        need(device!=nil,"native Metal device unavailable");NSError* error=nil;id<MTLLibrary> lib=nil;
        if([m[@"kernel_mode"] isEqual:@"source"]) {
            need([kernel[@"math_mode"] isEqual:@"safe"]&&[kernel[@"floating_point_functions"] isEqual:@"precise"]&&
                [kernel[@"language_version"] isEqual:@"4.0"],"unsupported native kernel compilation policy");
            NSData* kb=pinned(kernel,8u*1024u*1024u);
            NSString* ks=[[NSString alloc] initWithData:kb encoding:NSUTF8StringEncoding];need(ks!=nil,"kernel source is not UTF-8");
            MTLCompileOptions* options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
            options.mathFloatingPointFunctions=MTLMathFloatingPointFunctionsPrecise;options.languageVersion=MTLLanguageVersion4_0;
            lib=[device newLibraryWithSource:ks options:options error:&error];
        } else {
            need([m[@"kernel_mode"] isEqual:@"native-library"],"unknown native kernel owner mode");
            NSData* kb=pinned(kernel,64u*1024u*1024u);
            lib=[device newLibraryWithURL:[NSURL fileURLWithPath:kernel[@"path"]] error:&error];
            need(sha(read(kernel[@"path"],64u*1024u*1024u))==sha(kb),"native library changed while loading");
        }
        if(!lib)throw std::runtime_error(error.localizedDescription.UTF8String!=nullptr?error.localizedDescription.UTF8String:"native kernel load failed");
        auto function=numiHumanRestingMakeSkinInfluenceFunction(lib,@"nm_human_resting_skin",
            {false,true,tiled?NumiHumanSkinInfluenceLayout::tile32:NumiHumanSkinInfluenceLayout::vertexMajor},&error);
        need(function!=nil,"existing native skin function specialization failed");
        auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];need(pipeline!=nil,"existing native skin pipeline failed");
        auto buffer=[&](const void* p,NSUInteger n){id<MTLBuffer> b=[device newBufferWithBytes:p length:n options:MTLResourceStorageModeShared];need(b!=nil,"Metal buffer allocation failed");return b;};
        auto mapsGPU=buffer(maps.data(),mb.length);auto infGPU=buffer(ib.bytes,ib.length);
        auto bodiesGPU=buffer(bodies.data(),bodies.size()*sizeof(MRBodyStateGPU));
        auto respGPU=buffer(&respiration,sizeof(respiration));auto anatomyGPU=buffer(ab.bytes,ab.length);
        std::vector<MRVisualVertexGPUV2> vertices(count);
        NSData* seed=completeMapping?pinned(m[@"seed_vertices"],count*sizeof(MRVisualVertexGPUV2)):nil;
        if(completeMapping) {
            need(seed.length==count*sizeof(MRVisualVertexGPUV2),"seed vertex ABI differs");
            std::memcpy(vertices.data(),seed.bytes,seed.length);
        }
        NSData* indices=completeMapping?pinned(m[@"indices"],512u*1024u*1024u):nil;
        NSData* wallMap=completeMapping?pinned(m[@"cardiac_wall_map"],512u*1024u*1024u):nil;
        NSData* wallParams=completeMapping?pinned(m[@"cardiac_wall_parameters"],sizeof(MRHumanRestingCardiacWallGPU)):nil;
        NSData* commonMap=completeMapping?pinned(m[@"common_field_map"],512u*1024u*1024u):nil;
        NSData* normalRanges=completeMapping?pinned(m[@"normal_ranges"],128u*1024u*1024u):nil;
        NSData* normalIncidents=completeMapping?pinned(m[@"normal_incident_triangles"],512u*1024u*1024u):nil;
        NSData* airwayRanges=completeMapping?pinned(m[@"airway_normal_ranges"],128u*1024u*1024u):nil;
        NSData* airwayIncidents=completeMapping?pinned(m[@"airway_normal_incident_triangles"],512u*1024u*1024u):nil;
        need(!retainSourceCardiacContext||completeMapping,"source-cardiac context requires complete anatomy mapping");
        NSDictionary* poseCardiacInputs=pose[@"accepted_cardiac_geometry_buffers"];
        NSDictionary* sourceCardiacInputs=source[@"accepted_cardiac_geometry_buffers"];
        NSDictionary* cardiacInputs=retainSourceCardiacContext?sourceCardiacInputs:poseCardiacInputs;
        if(completeMapping)need([cardiacInputs isKindOfClass:[NSDictionary class]],"selected receipt lacks accepted cardiac geometry buffers");
        NSData* cardiac=completeMapping?pinned(cardiacInputs[@"cardiac_q"],16):nil;
        NSData* wall=completeMapping?pinned(cardiacInputs[@"cardiac_wall_q"],16):nil;
        NSData* common=completeMapping?pinned(cardiacInputs[@"common_coordinates"],64):nil;
        if(completeMapping) {
            need(cardiac.length==16&&wall.length==16&&common.length==64&&
                indices.length%4==0&&wallMap.length%48==0&&commonMap.length%112==0&&
                wallParams.length==sizeof(MRHumanRestingCardiacWallGPU),
                "native cardiac/index owner ABI differs");
            for(const auto& map:maps) {
                if(map.deformationKind==2u||map.deformationKind==8u)
                    need(map.chamberIndex<4u,"cardiac channel is out of range");
                if(map.deformationKind==5u||map.deformationKind==7u)
                    need(map.chamberIndex<24u,"rib channel is out of range");
                if(map.deformationKind==7u)
                    need(std::isfinite(map.deformationWeight.y)&&map.deformationWeight.y>=0&&
                        map.deformationWeight.y<24&&std::floor(map.deformationWeight.y)==map.deformationWeight.y,
                        "second rib channel is out of range");
                if(map.deformationKind==10u)need(map.chamberIndex<wallMap.length/48,"wall map channel is out of range");
                if(map.deformationKind==12u)need(map.chamberIndex<commonMap.length/112,"common map channel is out of range");
            }
        }
        auto verticesGPU=buffer(vertices.data(),vertices.size()*sizeof(vertices.front()));
        std::vector<std::uint8_t> dummy(4096,0);auto unusedGPU=buffer(dummy.data(),dummy.size());
        id<MTLCommandQueue> queue=[device newCommandQueue];id<MTLCommandBuffer> command=[queue commandBuffer];
        need(queue&&command,"Metal command allocation failed");id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];const mr_uint4 d={(unsigned)count,0,0,0};[encoder setBytes:&d length:sizeof(d) atIndex:0];
        [encoder setBuffer:mapsGPU offset:0 atIndex:1];[encoder setBuffer:infGPU offset:0 atIndex:2];
        [encoder setBuffer:bodiesGPU offset:0 atIndex:3];[encoder setBuffer:verticesGPU offset:0 atIndex:4];
        [encoder setBuffer:respGPU offset:0 atIndex:5];[encoder setBuffer:anatomyGPU offset:0 atIndex:6];
        for(NSUInteger i=7;i<=12;++i)[encoder setBuffer:unusedGPU offset:0 atIndex:i];
        if(completeMapping) {
            [encoder setBuffer:buffer(cardiac.bytes,cardiac.length) offset:0 atIndex:7];
            [encoder setBuffer:buffer(wallMap.bytes,wallMap.length) offset:0 atIndex:8];
            [encoder setBuffer:buffer(wallParams.bytes,wallParams.length) offset:0 atIndex:9];
            [encoder setBuffer:buffer(wall.bytes,wall.length) offset:0 atIndex:10];
            [encoder setBuffer:buffer(commonMap.bytes,commonMap.length) offset:0 atIndex:11];
            [encoder setBuffer:buffer(common.bytes,common.length) offset:0 atIndex:12];
        }
        [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(64,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
        if(completeMapping) {
            auto normalFunction=[lib newFunctionWithName:@"nm_human_resting_cardiac_wall_normals"];
            need(normalFunction!=nil,"existing native normal owner is absent");
            auto normalPipeline=[device newComputePipelineStateWithFunction:normalFunction error:&error];
            need(normalPipeline!=nil,"existing native normal pipeline failed");
            auto normals=[&](NSData* ranges,NSData* incidents,NSUInteger n) {
                need(ranges.length%16==0&&incidents.length%4==0&&n<=ranges.length/16,
                    "normal owner range ABI differs");
                const auto* rr=(const mr_uint4*)ranges.bytes;
                const auto* ii=(const unsigned*)incidents.bytes;
                const auto* ix=(const unsigned*)indices.bytes;
                for(NSUInteger i=0;i<n;++i) {
                    const auto r=rr[i];
                    need(r.x<count&&r.y<=incidents.length/4&&r.z<=incidents.length/4-r.y,
                        "normal owner incident span is invalid");
                    for(NSUInteger j=r.y;j<std::size_t(r.y)+r.z;++j) {
                        const auto at=ii[j];need(indices.length/4>=3&&at<=indices.length/4-3,
                            "normal owner triangle start is invalid");
                        need(ix[at]<count&&ix[at+1]<count&&ix[at+2]<count,
                            "normal owner triangle vertex is invalid");
                    }
                }
                if(!n)return;
                [encoder setComputePipelineState:normalPipeline];unsigned d=(unsigned)n;
                [encoder setBytes:&d length:sizeof(d) atIndex:0];
                [encoder setBuffer:buffer(ranges.bytes,ranges.length) offset:0 atIndex:1];
                [encoder setBuffer:buffer(incidents.bytes,incidents.length) offset:0 atIndex:2];
                [encoder setBuffer:buffer(indices.bytes,indices.length) offset:0 atIndex:3];
                [encoder setBuffer:verticesGPU offset:0 atIndex:4];
                [encoder dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(
                    std::min<NSUInteger>(64,normalPipeline.maxTotalThreadsPerThreadgroup),1,1)];
            };
            normals(normalRanges,normalIncidents,integer(m[@"normal_vertex_count"]));
            normals(airwayRanges,airwayIncidents,integer(m[@"airway_normal_vertex_count"]));
        }
        [encoder endEncoding];[command commit];[command waitUntilCompleted];
        need(command.status==MTLCommandBufferStatusCompleted,"native replay GPU command failed");
        const auto* vv=(const MRVisualVertexGPUV2*)verticesGPU.contents;
        for(std::size_t i=0;i<count;++i)need(std::isfinite(vv[i].position.x)&&std::isfinite(vv[i].position.y)&&std::isfinite(vv[i].position.z)&&
            std::isfinite(vv[i].normalAndTangentSign.x)&&std::isfinite(vv[i].normalAndTangentSign.y)&&std::isfinite(vv[i].normalAndTangentSign.z),"nonfinite native replay output");
        need(![[NSFileManager defaultManager] fileExistsAtPath:out],"refusing to overwrite replay directory");
        need([[NSFileManager defaultManager] createDirectoryAtPath:out withIntermediateDirectories:NO attributes:nil error:nil],"cannot create replay output directory");
        NSData* vb=[NSData dataWithBytes:verticesGPU.contents length:verticesGPU.length];NSString* vp=[out stringByAppendingPathComponent:completeMapping?@"anatomy-vertices.bin":@"skin-vertices.bin"];write(vp,vb);
        NSDictionary* cardiacContext=@{@"mode":retainSourceCardiacContext?@"held_source_receipt_buffers":@"pose_receipt_buffers",
            @"source_receipt":@{@"path":rp,@"sha256":@(argv[2])},
            @"pose_receipt":@{@"path":pp,@"sha256":@(argv[4])},
            @"cardiac_geometry_held_source_fixed":@(retainSourceCardiacContext),
            @"counterfactual":@(retainSourceCardiacContext),
            @"non_cardiac_motion_only":@(retainSourceCardiacContext),
            @"motion_scope":retainSourceCardiacContext?@"accepted body and respiratory motion from pose receipt; cardiac geometry buffers held at source receipt":@"accepted body, respiratory, and cardiac geometry buffers from pose receipt",
            @"buffers":completeMapping?cardiacInputs:@{}};
        NSDictionary* report=@{@"status":retainSourceCardiacContext?@"PASS_SOURCE_CARDIAC_HELD_COUNTERFACTUAL_NATIVE_GEOMETRY_KERNEL_REPLAY_ONLY":(completeMapping?@"PASS_EXISTING_NATIVE_GEOMETRY_KERNEL_REPLAY_ONLY":@"PASS_EXISTING_NATIVE_SKIN_KERNEL_REPLAY_ONLY"),
            @"complete_mapping":@(completeMapping),
            @"cardiac_context":cardiacContext,
            @"source_receipt":@{@"path":rp,@"sha256":@(argv[2])},@"pose_receipt":@{@"path":pp,@"sha256":@(argv[4])},
            @"vertex_map":m[@"vertex_map"],@"influences":m[@"influences"],@"anatomy_parameters":m[@"anatomy_parameters"],@"kernel":kernel,@"kernel_mode":m[@"kernel_mode"],
            @"influence_first_index":@(base),@"influence_layout":m[@"influence_layout"],@"vertex_count":@(count),
            @"output":@{@"path":vp,@"sha256":@(sha(vb).c_str()),@"bytes":@(vb.length),@"record_type":@"MRVisualVertexGPUV2",@"stride_bytes":@(sizeof(MRVisualVertexGPUV2))},
            @"device":device.name,@"gpu_seconds":@(command.GPUEndTime-command.GPUStartTime),
            @"physical_stepping":@NO,@"anatomical_admission":@NO,@"meaning":retainSourceCardiacContext?@"Counterfactual replay of current uploaded source geometry through the unchanged native viewer Metal kernel using accepted body and respiratory state from the pose receipt while holding cardiac geometry buffers at the source receipt. This is non-cardiac-motion-only evidence; it does not claim cardiac motion, physical stepping, or anatomical admission.":@"Current uploaded source geometry replayed at a retained accepted pose through the unchanged native viewer Metal kernel. Requires native capture parity and geometry gates; no new accepted physical state is claimed."};
        NSData* reportBytes=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
        need(reportBytes!=nil,"cannot serialize replay receipt");write([out stringByAppendingPathComponent:@"receipt.json"],reportBytes);
        std::cout<<(retainSourceCardiacContext?"PASS_SOURCE_CARDIAC_HELD_COUNTERFACTUAL_NATIVE_GEOMETRY_KERNEL_REPLAY_ONLY vertices=":(completeMapping?"PASS_EXISTING_NATIVE_GEOMETRY_KERNEL_REPLAY_ONLY vertices=":"PASS_EXISTING_NATIVE_SKIN_KERNEL_REPLAY_ONLY vertices="))<<count<<" gpu_seconds="<<command.GPUEndTime-command.GPUStartTime<<"\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"numi_human_resting_skin_replay: "<<e.what()<<"\n";return 1;} }
}
