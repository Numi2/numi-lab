#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_stand_gpu.h"
#include "numi/matter/numi_human_shared.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void require(bool b,const char* s){if(!b){std::fprintf(stderr,"FAIL %s\n",s);std::exit(1);}}
static nm_float4 f4(float x,float y,float z,float w){return{x,y,z,w};}
int main(int argc,const char** argv){@autoreleasepool {
 require(argc==2,"metallib path required");NSError* error=nil;
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();require(d!=nil,"Metal device");
 id<MTLLibrary> lib=[d newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];require(lib!=nil,"load production metallib");
 auto pipeline=[&](const char* name){auto fn=[lib newFunctionWithName:[NSString stringWithFormat:@"numi_matter_metal::%s",name]];require(fn!=nil,"kernel exists");auto p=[d newComputePipelineStateWithFunction:fn error:&error];require(p!=nil,"pipeline");return p;};
 auto wrenchSerial=pipeline("nm_numi_human_assemble_fem_body_contact_wrenches");auto wrenchPartial=pipeline("nm_numi_human_fem_body_contact_wrench_partial");auto wrenchFinish=pipeline("nm_numi_human_fem_body_contact_wrench_finish");
 auto serial=pipeline("nm_numi_human_audit_fem_body_contact");auto partial=pipeline("nm_numi_human_audit_fem_body_contact_partial");auto finish=pipeline("nm_numi_human_audit_fem_body_contact_finish");
 require(partial.threadExecutionWidth==32 && finish.threadExecutionWidth==32 && partial.maxTotalThreadsPerThreadgroup>=256 && finish.maxTotalThreadsPerThreadgroup>=256,"SIMD and threadgroup admission");
 id<MTLCommandQueue> q=[d newCommandQueue];require(q!=nil,"queue");
 auto buffer=[&](const void* data,NSUInteger n){auto b=data?[d newBufferWithBytes:data length:std::max<NSUInteger>(n,16) options:MTLResourceStorageModeShared]:[d newBufferWithLength:std::max<NSUInteger>(n,16) options:MTLResourceStorageModeShared];require(b!=nil,"buffer allocation");return b;};
 auto rowQuery=pipeline("nm_numi_human_validate_fem_body_contact_samples_cached");
 auto validationOracle=pipeline("nm_numi_human_validate_fem_body_contact_samples");
 auto auditCached=pipeline("nm_numi_human_audit_fem_body_contact_partial_cached");
 auto wrenchCached=pipeline("nm_numi_human_fem_body_contact_wrench_partial_cached");
 auto forceDirect=pipeline("nm_numi_human_assemble_fem_body_contact_loads_grouped");
 auto forceCached=pipeline("nm_numi_human_assemble_fem_body_contact_loads_grouped_cached");
 double maxRelative=0,maxAbsolute=0,maxForceAbsolute=0,maxForceRelative=0;
 auto compareForces=[&](const auto& a,const auto& b){
  const float* x=reinterpret_cast<const float*>(a.data());const float* y=reinterpret_cast<const float*>(b.data());
  for(unsigned k=0;k<24;++k){require(std::isfinite(x[k])&&std::isfinite(y[k]),"finite source-order forces");
   const double diff=std::abs(double(x[k])-y[k]);maxForceAbsolute=std::max(maxForceAbsolute,diff);maxForceRelative=std::max(maxForceRelative,diff/std::max(1.e-6,std::abs(double(x[k]))));
   require(diff<=1.e-6+.01*std::abs(double(x[k])),"source-order cached force comparison1percent");}
 };
 for(unsigned count:{0u,1u,255u,256u,257u,8193u,262145u}) for(unsigned mode=0;mode<7;++mode){@autoreleasepool {
  if(mode>=4 && count>8193)continue;
  // Mode0 inactive;1 active contacts and attachments;2 malformed late sample;3 malformed attachment.
  NMNumiHumanTendonFEMLoadDispatchGPU dispatch{};dispatch.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;dispatch.environmentCount=2;dispatch.femNodeCount=3;dispatch.bodyPoseStride=3;dispatch.articulationFirstBody=0;dispatch.femBodyContactSampleCount=count;dispatch.femSurfaceAttachmentSampleCount=mode==0?0:1;
  std::vector<NMNumiHumanFEMBodyContactSampleGPU> samples(std::max(count,1u));
  for(unsigned i=0;i<count;++i){auto& a=samples[i];a.slaveNode=unsigned((std::uint64_t(i)*3u)/std::max(count,1u));a.bodyIndex=1+i%2;a.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE|NM_NUMI_HUMAN_FEM_BODY_CONTACT_ELASTIC_ATTACHMENT;a.bodyLocalPointAndArea=f4(mode==0?0.f:.003f*float(i%5),mode==0?0.f:-.002f*float(i%3),mode==0?0.f:.001f*float(i%7),1.e-6f*(1.f+float(i%7)*.01f));a.bodyLocalNormalAndReferenceSeparation=f4(0,0,1,0);a.stiffnessAndNormalStrainPerPressure=f4(30000,1.e-6f,0,0);}
  if(mode==2 && count) samples[count-1].slaveNode=99;
  NMNumiHumanFEMSurfaceAttachmentGPU attachment{};attachment.node[0]=0;attachment.node[1]=1;attachment.node[2]=mode==3?99:2;attachment.bodyIndex=1;attachment.stableID=73;attachment.flags=NM_NUMI_HUMAN_FEM_SURFACE_ATTACHMENT_ACTIVE;attachment.barycentric=f4(.2f,.3f,.5f,0);attachment.bodyLocalPointAndArea=f4(.03f,.05f,0,.02f);attachment.stiffnessAndStrainPerPressure=f4(100000,1.e-5f,0,0);
  std::array<NMFEMNodeStateGPU,6> nodes{};
  for(unsigned env=0;env<2;++env){nodes[env*3].positionAndMass=f4(0,0,0,1);nodes[env*3+1].positionAndMass=f4(mode==0?0:.1f,0,0,1);nodes[env*3+2].positionAndMass=f4(0,mode==0?0:.1f,0,1);}
  // Inactive elastic contacts coincide at the origin; active cases exercise different reaction directions.
  std::array<MRArticulatedBodyPoseGPU,6> poses{};
  for(unsigned i=0;i<6;++i){poses[i].position={0,0,mode==0?0.f:.01f*float(i%3),0};poses[i].orientation={0,0,0,1};}
  std::vector<NMNumiHumanFEMBodyContactNeighborhoodRangeGPU> queryRanges(std::max(count,1u));
  std::array<NMNumiHumanFEMBodyContactNeighborhoodTriangleGPU,2> queryTriangles{};
  if(mode>=4){
   dispatch.femBodyContactCandidateTriangleCount=2;
   for(unsigned j=0;j<2;++j){auto& t=queryTriangles[j];t.bodyIndex=j+1;t.boneStableId=j+21;t.triangleStableId=j+101;
    t.bodyLocalVertex0=f4(-1,-1,0,0);t.bodyLocalVertex1=f4(1,-1,0,0);t.bodyLocalVertex2=f4(0,1,0,0);t.authoredOutwardNormal=f4(0,0,1,0);}
   for(unsigned i=0;i<count;++i){auto& a=samples[i];a.flags=NM_NUMI_HUMAN_FEM_BODY_CONTACT_ACTIVE|NM_NUMI_HUMAN_FEM_BODY_CONTACT_TRIANGLE_CONTACT;
    a.bodyLocalPointAndArea=f4(-1,-1,0,1.e-6f);a.bodyLocalNormalAndReferenceSeparation=f4(1,-1,0,30000);a.stiffnessAndNormalStrainPerPressure=f4(0,1,0,1.e-6f);
    queryRanges[i].firstTriangle=i%2;queryRanges[i].triangleCount=1;queryRanges[i].maximumDistanceMeters=.05f;}
   if(mode==4)for(auto& v:poses)v.position.z=-.01f;
   if(mode==6 && count)queryRanges[count-1].maximumDistanceMeters=0;
  }
  auto rb=buffer(queryRanges.data(),queryRanges.size()*sizeof(queryRanges[0]));auto tb=buffer(queryTriangles.data(),sizeof(queryTriangles));
  auto sb=buffer(samples.data(),samples.size()*sizeof(samples[0]));auto nb=buffer(nodes.data(),sizeof(nodes));auto pb=buffer(poses.data(),sizeof(poses));auto ab=buffer(&attachment,sizeof(attachment));
  auto dummy=buffer(nullptr,256);const unsigned total=count+dispatch.femSurfaceAttachmentSampleCount;const unsigned tiles=total/256+(total%256!=0);auto scratch=buffer(nullptr,std::max(1u,tiles)*2*sizeof(NMNumiHumanArticularContactAuditGPU));auto out=buffer(nullptr,2*sizeof(NMNumiHumanArticularContactAuditGPU));
     auto cache=buffer(nullptr,std::max(count,1u)*2*sizeof(nm_float4));
   auto validity=buffer(nullptr,std::max(count,1u)*2*sizeof(unsigned));
   auto originalValidity=buffer(nullptr,std::max(count,1u)*2*sizeof(unsigned));
   auto query=[&](id<MTLCommandBuffer> command,bool oracle=false){auto e=[command computeCommandEncoder];
    [e setComputePipelineState:oracle?validationOracle:rowQuery];[e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];
    [e setBuffer:sb offset:0 atIndex:1];[e setBuffer:rb offset:0 atIndex:2];[e setBuffer:tb offset:0 atIndex:3];
    [e setBuffer:dummy offset:0 atIndex:4];[e setBuffer:nb offset:0 atIndex:5];[e setBuffer:pb offset:0 atIndex:6];
    [e setBuffer:oracle?originalValidity:validity offset:0 atIndex:7];if(!oracle)[e setBuffer:cache offset:0 atIndex:8];
    if(count)[e dispatchThreads:MTLSizeMake(2*count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];};
   {auto cb=[q commandBuffer];query(cb,true);query(cb);[cb commit];[cb waitUntilCompleted];
    require(cb.status==MTLCommandBufferStatusCompleted,"validation GPU completion");
    require(std::memcmp(validity.contents,originalValidity.contents,2*count*sizeof(unsigned))==0,"fused validation flags bitwise parity");}
   auto run=[&](bool parallel,bool cached=false){auto command=[q commandBuffer];require(command!=nil,"command");if(cached)query(command);auto e=[command computeCommandEncoder];[e setComputePipelineState:cached?auditCached:(parallel?partial:serial)];[e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBuffer:sb offset:0 atIndex:1];[e setBuffer:nb offset:0 atIndex:2];[e setBuffer:pb offset:0 atIndex:3];[e setBuffer:parallel?scratch:out offset:0 atIndex:4];[e setBuffer:ab offset:0 atIndex:5];[e setBuffer:dummy offset:0 atIndex:6];[e setBuffer:dummy offset:0 atIndex:7];[e setBuffer:dummy offset:0 atIndex:8];if(cached)[e setBuffer:cache offset:0 atIndex:9];[e setBuffer:rb offset:0 atIndex:6];[e setBuffer:tb offset:0 atIndex:7];
   if(parallel){if(tiles)[e dispatchThreadgroups:MTLSizeMake(2*tiles,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}else [e dispatchThreads:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(2,1,1)];[e endEncoding];
   if(parallel){e=[command computeCommandEncoder];[e setComputePipelineState:finish];[e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBuffer:scratch offset:0 atIndex:1];[e setBuffer:out offset:0 atIndex:2];[e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];}
   [command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted,"GPU command completed");std::array<NMNumiHumanArticularContactAuditGPU,2> result{};std::memcpy(result.data(),out.contents,sizeof(result));return std::pair{result,command.GPUEndTime-command.GPUStartTime};};
  auto compare=[&](const auto& a,const auto& b){for(unsigned env=0;env<2;++env){bool bad=mode==3 || ((mode==2 || mode==6) && count);const float* x=reinterpret_cast<const float*>(&a[env]);const float* y=reinterpret_cast<const float*>(&b[env]);if(bad){require(std::isnan(x[0])&&std::isnan(y[0]),"same rejection for malformed row");continue;}
   for(unsigned k=0;k<16;++k){require(std::isfinite(x[k])&&std::isfinite(y[k]),"finite audit");double diff=std::abs(double(x[k])-y[k]);double scale=std::max(1.e-6,std::abs(double(x[k])));maxAbsolute=std::max(maxAbsolute,diff);maxRelative=std::max(maxRelative,diff/scale);require(diff<=1.e-6+.01*std::abs(double(x[k])),"audit field comparison1percent");}
   require(a[env].normalForceAreaAndCounts.w==b[env].normalForceAreaAndCounts.w,"exact source row count");
  }};

  // The same source-order rows are indexed by body once, without copying samples.
  std::array<std::vector<unsigned>,3> rows;
  for(unsigned i=0;i<count;++i) rows[samples[i].bodyIndex].push_back(i);
  std::vector<unsigned> indices;std::vector<nm_uint4> tilesForBody;
  std::array<NMIncidenceRangeGPU,3> rangesForBody{};
  for(unsigned body=0;body<3;++body){rangesForBody[body].first=unsigned(tilesForBody.size());const unsigned first=unsigned(indices.size());indices.insert(indices.end(),rows[body].begin(),rows[body].end());for(unsigned i=0;i<rows[body].size();i+=256)tilesForBody.push_back({first+i,std::min(256u,unsigned(rows[body].size())-i),body,0});rangesForBody[body].count=unsigned(tilesForBody.size())-rangesForBody[body].first;}
  const unsigned wrenchTiles=unsigned(tilesForBody.size());
  auto rowIndexBuffer=buffer(indices.empty()?nullptr:indices.data(),indices.size()*sizeof(unsigned));
  auto tileBuffer=buffer(tilesForBody.empty()?nullptr:tilesForBody.data(),tilesForBody.size()*sizeof(nm_uint4));
  auto bodyRangeBuffer=buffer(rangesForBody.data(),sizeof(rangesForBody));auto wrenchScratch=buffer(nullptr,std::max(1u,wrenchTiles)*2*sizeof(NMNumiHumanBodyWrenchGPU));auto wrenchOut=buffer(nullptr,6*sizeof(NMNumiHumanBodyWrenchGPU));
  auto runWrench=[&](bool parallel,bool cached=false){auto cb=[q commandBuffer];if(cached)query(cb);auto en=[cb computeCommandEncoder];[en setComputePipelineState:cached?wrenchCached:(parallel?wrenchPartial:wrenchSerial)];[en setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[en setBuffer:sb offset:0 atIndex:1];[en setBuffer:nb offset:0 atIndex:2];[en setBuffer:pb offset:0 atIndex:3];[en setBuffer:parallel?wrenchScratch:wrenchOut offset:0 atIndex:4];[en setBuffer:dummy offset:0 atIndex:5];[en setBuffer:dummy offset:0 atIndex:6];[en setBuffer:dummy offset:0 atIndex:7];[en setBuffer:rb offset:0 atIndex:5];[en setBuffer:tb offset:0 atIndex:6];
   if(parallel){[en setBuffer:rowIndexBuffer offset:0 atIndex:8];[en setBuffer:tileBuffer offset:0 atIndex:9];[en setBytes:&wrenchTiles length:sizeof(wrenchTiles) atIndex:10];if(cached)[en setBuffer:cache offset:0 atIndex:11];if(wrenchTiles)[en dispatchThreadgroups:MTLSizeMake(2*wrenchTiles,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}else [en dispatchThreads:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(6,1,1)];[en endEncoding];
   if(parallel){en=[cb computeCommandEncoder];[en setComputePipelineState:wrenchFinish];[en setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[en setBuffer:wrenchScratch offset:0 atIndex:1];[en setBuffer:wrenchOut offset:0 atIndex:2];[en setBuffer:bodyRangeBuffer offset:0 atIndex:3];[en setBytes:&wrenchTiles length:sizeof(wrenchTiles) atIndex:4];[en dispatchThreadgroups:MTLSizeMake(6,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[en endEncoding];}
   [cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"wrench GPU completed");std::array<NMNumiHumanBodyWrenchGPU,6> result{};std::memcpy(result.data(),wrenchOut.contents,sizeof(result));return std::pair{result,cb.GPUEndTime-cb.GPUStartTime};};
  if(mode<2){const auto wa=runWrench(false),wb=runWrench(true);for(unsigned i=0;i<6;++i){const float* x=reinterpret_cast<const float*>(&wa.first[i]);const float* y=reinterpret_cast<const float*>(&wb.first[i]);for(unsigned j=0;j<8;++j){require(std::isfinite(x[j])&&std::isfinite(y[j]),"finite wrench");require(std::abs(double(x[j])-y[j])<=1.e-6+.01*std::abs(double(x[j])),"wrench comparison1percent");}}
   std::printf("{\"wrench_case_count\":%u,\"mode\":%u,\"serial_gpu_s\":%.9g,\"parallel_gpu_s\":%.9g,\"speedup\":%.9g}\n",count,mode,wa.second,wb.second,wa.second/wb.second);}
     auto a=run(false);auto b=run(true);auto c=run(true,true);compare(a.first,b.first);compare(b.first,c.first);
   auto wa=runWrench(true),wc=runWrench(true,true);
   for(unsigned i=0;i<6;++i){const float* x=reinterpret_cast<const float*>(&wa.first[i]);const float* y=reinterpret_cast<const float*>(&wc.first[i]);
    for(unsigned j=0;j<8;++j){if(!std::isfinite(x[j])){require(!std::isfinite(y[j]),"cached invalid wrench retained");continue;}
     require(std::isfinite(y[j])&&std::abs(double(x[j])-y[j])<=1.e-6+.01*std::abs(double(x[j])),"cached wrench parity");}}
   if(count<=8193){
    std::vector<unsigned> groups(std::max(count,1u),0u);
    for(unsigned i=0;i<count;){unsigned end=i+1;while(end<count && samples[end].slaveNode==samples[i].slaveNode)++end;groups[i]=end-i;i=end;}
    auto gb=buffer(groups.data(),groups.size()*sizeof(unsigned));
    auto forceOut=buffer(nullptr,6*sizeof(nm_float4));
    auto runForce=[&](bool cached){std::array<nm_float4,6> initial{};for(auto& v:initial)v=f4(.125f,-.25f,.5f,0);std::memcpy(forceOut.contents,initial.data(),sizeof(initial));
     auto cb=[q commandBuffer];if(cached)query(cb);auto e=[cb computeCommandEncoder];[e setComputePipelineState:cached?forceCached:forceDirect];
     [e setBytes:&dispatch length:sizeof(dispatch) atIndex:0];[e setBuffer:sb offset:0 atIndex:1];[e setBuffer:nb offset:0 atIndex:2];[e setBuffer:pb offset:0 atIndex:3];[e setBuffer:forceOut offset:0 atIndex:4];[e setBuffer:dummy offset:0 atIndex:5];[e setBuffer:dummy offset:0 atIndex:6];[e setBuffer:dummy offset:0 atIndex:8];[e setBuffer:gb offset:0 atIndex:9];if(cached)[e setBuffer:cache offset:0 atIndex:10];[e setBuffer:rb offset:0 atIndex:5];[e setBuffer:tb offset:0 atIndex:6];
     if(count)[e dispatchThreads:MTLSizeMake(2*count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status==MTLCommandBufferStatusCompleted,"force GPU completion");std::array<nm_float4,6> result{};std::memcpy(result.data(),forceOut.contents,sizeof(result));return result;};
    auto fd=runForce(false),fc=runForce(true);compareForces(fd,fc);
    // Mutate the accepted-node input, then rebuild scratch. A stale cache would fail.
    static_cast<NMFEMNodeStateGPU*>(nb.contents)[0].positionAndMass.x+=.017f;
    auto moved=runForce(false),movedCached=runForce(true);compareForces(moved,movedCached);
   }
   std::printf("{\"cache_case_count\":%u,\"mode\":%u,\"direct_tiled_audit_gpu_s\":%.9g,\"query_plus_cached_audit_gpu_s\":%.9g}\n",count,mode,b.second,c.second);
  std::printf("{\"case_count\":%u,\"mode\":%u,\"environments\":2,\"serial_gpu_s\":%.9g,\"parallel_gpu_s\":%.9g,\"speedup\":%.9g}\n",count,mode,a.second,b.second,a.second/b.second);std::fflush(stdout);
  if(count==262145 && mode==1){for(unsigned pair=0;pair<3;++pair){auto x=run(pair%2!=0),y=run(pair%2==0);compare(x.first,y.first);const double old=pair%2?y.second:x.second,now=pair%2?x.second:y.second;std::printf("{\"matched_pair\":%u,\"case_count\":%u,\"serial_gpu_s\":%.9g,\"parallel_gpu_s\":%.9g,\"speedup\":%.9g}\n",pair,count,old,now,old/now);std::fflush(stdout);}}
 }}
 std::printf("{\"status\":\"PASS\",\"maximum_relative_error\":%.9g,\"maximum_absolute_error\":%.9g,\"scope\":\"production current-generation contact cache; active/inactive/invalid, 2 environments, source-order forces and changed-state refresh; not integrated throughput\"}\n",maxRelative,maxAbsolute);
 std::printf("{\"maximum_force_absolute_error\":%.9g,\"maximum_force_relative_error\":%.9g}\n",maxForceAbsolute,maxForceRelative);
 return 0;
}}
