#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "metalrobo/engine_types.h"
#include "numi/matter/numi_human_shared.h"
#include "numi/matter/human_respiration.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
static void need(bool b,const char* message){if(!b){fprintf(stderr,"FAIL %s\n",message);exit(1);}}
static nm_float4 f4(float x=0,float y=0,float z=0,float w=0){return{x,y,z,w};}
struct GPU {
 id<MTLDevice> device=MTLCreateSystemDefaultDevice();id<MTLCommandQueue> queue;
 id<MTLLibrary> matter,respiration;std::map<std::string,id<MTLComputePipelineState>> cache;
 GPU(const char* a,const char* b){NSError* e=nil;need(device!=nil,"device");queue=[device newCommandQueue];
  matter=[device newLibraryWithURL:[NSURL fileURLWithPath:@(a)] error:&e];
  respiration=[device newLibraryWithURL:[NSURL fileURLWithPath:@(b)] error:&e];need(matter&&respiration,"production libraries");}
 id<MTLBuffer> buf(const void* p,size_t n){auto b=[device newBufferWithLength:std::max(n,size_t(16)) options:MTLResourceStorageModeShared];need(b!=nil,"buffer");memset(b.contents,0,b.length);if(p&&n)memcpy(b.contents,p,n);return b;}
 template<class T> id<MTLBuffer> array(const std::vector<T>& a){if(a.empty())return buf(nullptr,16);return buf(a.data(),a.size()*sizeof(T));}
 id<MTLComputePipelineState> pipe(const char* n,bool blood=false){std::string key=n;if(cache.count(key))return cache[key];NSError* e=nil;auto lib=blood?respiration:matter;
  NSString* name=blood?@(n):[NSString stringWithFormat:@"numi_matter_metal::%s",n];auto f=[lib newFunctionWithName:name];need(f!=nil,n);auto p=[device newComputePipelineStateWithFunction:f error:&e];if(!p)fprintf(stderr,"%s\n",e.localizedDescription.UTF8String);need(p!=nil,"pipeline");cache[key]=p;return p;}
 using Bind=std::pair<unsigned,id<MTLBuffer>>;
 void encode(id<MTLCommandBuffer> cb,const char* name,const void* dispatch,size_t size,std::initializer_list<Bind> buffers,size_t count,bool groups=false,unsigned width=256,const NMFGMRESLayoutGPU* layout=nullptr,bool blood=false,std::initializer_list<std::pair<unsigned,unsigned>> constants={}){
  auto p=pipe(name,blood);auto e=[cb computeCommandEncoder];[e setComputePipelineState:p];[e setBytes:dispatch length:size atIndex:0];if(layout)[e setBytes:layout length:sizeof(*layout) atIndex:30];
  for(auto x:buffers)[e setBuffer:x.second offset:0 atIndex:x.first];for(auto x:constants)[e setBytes:&x.second length:sizeof(x.second) atIndex:x.first];
  need(width<=p.maxTotalThreadsPerThreadgroup,"group admission");if(count){if(groups)[e dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];else [e dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];}[e endEncoding];}
 double done(id<MTLCommandBuffer> cb){[cb commit];[cb waitUntilCompleted];if(cb.error)fprintf(stderr,"%s\n",cb.error.localizedDescription.UTF8String);need(cb.status==MTLCommandBufferStatusCompleted,"GPU completion");return cb.GPUEndTime-cb.GPUStartTime;}
};
static void near(float a,float b,const char* name){if(std::isnan(a)||std::isnan(b)){need(std::isnan(a)&&std::isnan(b),name);return;}need(std::isfinite(a)&&std::isfinite(b),name);if(std::abs(double(a)-b)>1.e-6+.01*std::abs(double(a))){fprintf(stderr,"%s: %.9g %.9g\n",name,a,b);need(false,"1percent gate");}}
static void nodes(GPU& g){
 for(unsigned N:{1u,255u,256u,257u,8193u,318974u}) for(unsigned fault=0;fault<6;++fault){@autoreleasepool {
  NMNumiHumanTendonFEMLoadDispatchGPU d{};d.abiVersion=NM_NUMI_HUMAN_TENDON_FEM_LOAD_ABI_VERSION;d.environmentCount=2;d.femNodeCount=N;d.endpointCount=2;d.dofCount=3;d.transferStride=2;d.generalizedForceStride=3;d.muscleCount=1;d.bodyPoseStride=1;d.pointJacobianStride=36;d.bodyJacobianPointOffset=0;
  std::vector<NMNumiHumanTendonFEMNodeLoadGPU> loads(N);std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> anchors(N);
  std::vector<MRNumiHumanTendonTransferResultGPU> transfers(4);std::vector<MRArticulatedBodyPoseGPU> poses(2);std::vector<nm_float4> forces(2*N);
  std::vector<unsigned> active;
  for(unsigned i=0;i<N;++i){for(auto& v:loads[i].endpointIndex)v=NM_INVALID_INDEX;anchors[i].bodyIndex=MR_INVALID_INDEX;if(i%17==0||i==N-1){anchors[i].flags=NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;anchors[i].bodyIndex=0;anchors[i].localPoint=f4(.01f*(i%3),.02f,.03f,0);active.push_back(i);}for(unsigned env=0;env<2;++env)forces[env*N+i]=f4(float(int(i%7)-3)*.003f,float(int(i%5)-2)*.002f,.001f,0);}
  for(unsigned i=0;i<4;++i){transfers[i].status=MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS;transfers[i].environment=i/2;transfers[i].bindingIndex=i%2;}
  for(auto& p:poses)p.orientation={0,0,0,1};
  if(fault==1)loads[N-1].scale.x=NAN;
  if(fault==2){loads[N-1].endpointIndex[0]=0;loads[N-1].endpointIndex[1]=0;loads[N-1].scale=f4(1,1,0,0);}
  if(fault==3){loads[N-1].endpointIndex[0]=0;loads[N-1].scale.x=1;transfers[2].status=999;}
  if(fault==4)anchors[N-1].reserved0=1;
  if(fault==5){loads[N-1].endpointIndex[0]=0;loads[N-1].scale.x=1;poses[1].position.x=NAN;}
  auto l=g.array(loads),a=g.array(anchors),t=g.array(transfers),p=g.array(poses),f=g.array(forces),st=g.buf(nullptr,2*sizeof(NMMatterStatusGPU));
  const unsigned tiles=(N+255)/256;auto scratch=g.buf(nullptr,2*tiles*sizeof(nm_float4)),witness=g.buf(nullptr,8),out=g.buf(nullptr,2*sizeof(nm_float4));
  auto cb=[g.queue commandBuffer];
  g.encode(cb,"nm_numi_human_validate_node_load_tiles",&d,sizeof(d),{{1,l},{2,a},{3,t},{4,p},{5,scratch}},2*tiles,true,256,nullptr,false,{{6,tiles}});
  g.encode(cb,"nm_numi_human_finish_node_load_validation",&d,sizeof(d),{{1,scratch},{2,witness}},2,true,256,nullptr,false,{{3,tiles}});
  double validation=g.done(cb);auto w=(unsigned*)witness.contents;
  need(w[0]==((fault==1||fault==2||fault==4)?N-1:NM_INVALID_INDEX),"env0 validation witness");
  need(w[1]==(fault==5?0u:(fault?N-1:NM_INVALID_INDEX)),"env1 validation witness");
  auto dummy=g.buf(nullptr,1024),world=g.buf(nullptr,2*sizeof(MRMetalWorldStatusGPU));
  auto provisional=[&](bool fast){auto command=[g.queue commandBuffer];
   g.encode(command,"nm_numi_human_prepare_provisional_status",&d,sizeof(d),
    {{1,l},{2,a},{3,dummy},{4,dummy},{5,t},{6,p},{7,world},{8,dummy},{9,dummy},{10,dummy},{11,dummy},{12,dummy},{13,dummy},{14,dummy},{15,dummy},{16,dummy},{17,dummy},{18,witness}},2,false,1,nullptr,false,{{19,fast?tiles:0u}});
   double time=g.done(command);std::array<MRMetalWorldStatusGPU,2> result;memcpy(result.data(),world.contents,sizeof(result));return std::pair{result,time};};
  auto originalStatus=provisional(false),tiledStatus=provisional(true);
  need(memcmp(originalStatus.first.data(),tiledStatus.first.data(),sizeof(originalStatus.first))==0,"provisional acceptance/rejection exact parity");
  printf("{\"path\":\"node_validation\",\"nodes\":%u,\"fault\":%u,\"serial_gpu_s\":%.9g,\"parallel_gpu_s\":%.9g}\n",N,fault,originalStatus.second,validation+tiledStatus.second);
  if(fault)continue;
  for(unsigned mode=0;mode<4;++mode){
   if(mode==2)((nm_float4*)f.contents)[2*N-1].x=NAN;
   if(mode==3)((NMMatterStatusGPU*)st.contents)[1].code=NM_STATUS_CONTACT_FAILURE;
   bool reaction=mode!=0;
   cb=[g.queue commandBuffer];
   if(reaction)g.encode(cb,"nm_numi_human_audit_fem_anchor_reactions",&d,sizeof(d),{{1,a},{2,f},{3,st},{4,out}},2,false,1);
   else g.encode(cb,"nm_numi_human_audit_tendon_fem_loads",&d,sizeof(d),{{1,f},{2,out}},2,false,1);
   double old=g.done(cb);std::array<nm_float4,2> reference;memcpy(reference.data(),out.contents,sizeof(reference));
   cb=[g.queue commandBuffer];g.encode(cb,"nm_numi_human_node_force_audit_tiles",&d,sizeof(d),{{1,f},{2,a},{3,st},{4,scratch}},2*tiles,true,256,nullptr,false,{{5,unsigned(reaction)}});
   g.encode(cb,"nm_numi_human_finish_node_force_audit",&d,sizeof(d),{{1,scratch},{2,out}},2,true);
   double now=g.done(cb);
   for(unsigned i=0;i<8;++i)near(((float*)reference.data())[i],((float*)out.contents)[i],"force audit");
   printf("{\"path\":\"node_audit\",\"nodes\":%u,\"mode\":%u,\"validation_gpu_s\":%.9g,\"serial_gpu_s\":%.9g,\"parallel_gpu_s\":%.9g}\n",N,mode,validation,old,now);
  }
  // Deferred anchor projection retains precisely the same source-order sum.
  if(true){d.environmentCount=1;d.dofCount=128;d.generalizedForceStride=128;d.pointJacobianStride=1536;auto indices=g.array(active);auto J=g.buf(nullptr,1536*sizeof(float));for(unsigned j=0;j<1536;++j)((float*)J.contents)[j]=.01f*float(int(j%11)-5);auto pending=g.buf(nullptr,128*sizeof(float));
   // Restore valid loads/poses; force buffer was deliberately corrupted above.
   memcpy(f.contents,forces.data(),forces.size()*sizeof(nm_float4));memset(st.contents,0,2*sizeof(NMMatterStatusGPU));
   auto run=[&](bool sparse){memset(pending.contents,0,128*sizeof(float));auto c=[g.queue commandBuffer];if(sparse)g.encode(c,"nm_numi_human_add_fem_anchor_reactions_to_pending_load_sparse",&d,sizeof(d),{{1,a},{2,f},{3,p},{4,J},{5,pending},{6,st},{7,indices}},d.dofCount,false,32,nullptr,false,{{8,unsigned(active.size())}});
    else g.encode(c,"nm_numi_human_add_fem_anchor_reactions_to_pending_load",&d,sizeof(d),{{1,a},{2,f},{3,p},{4,J},{5,pending},{6,st}},d.dofCount,false,32);
    double time=g.done(c);printf("{\"path\":\"anchor_projection\",\"nodes\":%u,\"anchors\":%zu,\"dofs\":%u,\"sparse\":%s,\"gpu_s\":%.9g}\n",N,active.size(),d.dofCount,sparse?"true":"false",time);std::array<float,128> v;memcpy(v.data(),pending.contents,sizeof(v));return v;};
   auto oldProjection=run(false),newProjection=run(true);need(oldProjection==newProjection,"bitwise sparse anchor projection");need(oldProjection[0]!=0.f||oldProjection[1]!=0.f||oldProjection[2]!=0.f,"active sparse projection exercised");
  }
 }}
}
static void topology(GPU& g){
 for(unsigned N:{1u,255u,256u,257u,8193u})for(unsigned mode=0;mode<3;++mode){@autoreleasepool {
  NMMatterDispatchGPU d{};d.environmentCount=2;d.femNodeCount=N;d.tetrahedronCount=N*3;d.materialCount=1;unsigned T=d.tetrahedronCount,blocks=(N+255)/256;
  std::vector<NMTetrahedronGPU> ts(2*T);std::vector<NMMaterialGPU> mats(1);mats[0].bulk.x=1050.6f;
  for(unsigned e=0;e<2;++e)for(unsigned i=0;i<T;++i){auto& t=ts[e*T+i];for(unsigned c=0;c<4;++c)reinterpret_cast<unsigned*>(&t.nodes)[c]=(i*7+c*13)%N;t.identity.w=(i%13==0)?0:NM_OBJECT_ACTIVE;t.inverseRestRow0.w=1.e-7f*(1+float(i%9)*.01f);if(mode==1)reinterpret_cast<unsigned*>(&t.nodes)[1]=reinterpret_cast<unsigned*>(&t.nodes)[0];if(mode==2 && i%9==0){reinterpret_cast<unsigned*>(&t.nodes)[0]=reinterpret_cast<unsigned*>(&t.nodes)[1]=reinterpret_cast<unsigned*>(&t.nodes)[2]=reinterpret_cast<unsigned*>(&t.nodes)[3]=0;}}
  auto tet=g.array(ts),material=g.array(mats),node=g.buf(nullptr,2*N*sizeof(NMFEMNodeStateGPU)),ranges=g.buf(nullptr,2*N*sizeof(NMIncidenceRangeGPU)),cursors=g.buf(nullptr,2*N*4),incidence=g.buf(nullptr,2*T*4*4),scratch=g.buf(nullptr,2*blocks*4),status=g.buf(nullptr,2*sizeof(NMMatterStatusGPU));
  auto run=[&](bool fast){memset(status.contents,0,status.length);memset(incidence.contents,0,incidence.length);auto cb=[g.queue commandBuffer];
   g.encode(cb,"nm_topology_rebuild_clear",&d,sizeof(d),{{1,node},{2,ranges},{3,cursors},{4,status}},2*N);
   if(fast){
    g.encode(cb,"nm_topology_rebuild_count_corners",&d,sizeof(d),{{1,tet},{2,cursors},{3,status}},2*T);
    g.encode(cb,"nm_topology_rebuild_prefix_tiles",&d,sizeof(d),{{1,cursors},{2,ranges},{3,scratch},{4,status}},2*blocks,true);
    g.encode(cb,"nm_topology_rebuild_prefix_block_offsets",&d,sizeof(d),{{1,scratch},{2,status}},2,true,32);
    g.encode(cb,"nm_topology_rebuild_add_block_offsets",&d,sizeof(d),{{1,ranges},{2,scratch},{3,status}},2*N);
    g.encode(cb,"nm_topology_rebuild_scatter_corners",&d,sizeof(d),{{1,tet},{2,ranges},{3,cursors},{4,incidence},{5,status}},2*T);
    g.encode(cb,"nm_topology_rebuild_sort_and_mass",&d,sizeof(d),{{1,material},{2,tet},{3,node},{4,ranges},{5,incidence},{6,status}},2*N);
   }else{
    g.encode(cb,"nm_topology_rebuild_count_mass",&d,sizeof(d),{{1,material},{2,tet},{3,node},{4,ranges},{5,status}},2*N);
    g.encode(cb,"nm_topology_rebuild_prefix",&d,sizeof(d),{{1,ranges},{2,cursors},{3,status}},2,true,32);
    g.encode(cb,"nm_topology_rebuild_scatter",&d,sizeof(d),{{1,tet},{2,ranges},{3,cursors},{4,incidence},{5,status}},2*N);
   }
   double seconds=g.done(cb);need(((NMMatterStatusGPU*)status.contents)[0].code==0&&((NMMatterStatusGPU*)status.contents)[1].code==0,"topology status");
   std::vector<unsigned char> data(node.length+ranges.length+incidence.length);size_t o=0;for(auto b:{node,ranges,incidence}){memcpy(data.data()+o,b.contents,b.length);o+=b.length;}return std::pair{data,seconds};};
  auto x=run(false),y=run(true);need(x.first==y.first,"bitwise topology mass ranges and ordered incidence");
  printf("{\"path\":\"topology\",\"nodes\":%u,\"tets\":%u,\"mode\":%u,\"serial_gpu_s\":%.9g,\"sparse_gpu_s\":%.9g}\n",N,T,mode,x.second,y.second);
 }}
}
static void reset(GPU& g){
 for(unsigned N:{1u,257u,318974u})for(unsigned mode=0;mode<5;++mode){@autoreleasepool {
  NMMatterDispatchGPU d{};d.environmentCount=2;d.femNodeCount=N;d.rigidGeneralizedCapacity=128;d.mpmActiveNodeCapacity=37;
  NMFGMRESLayoutGPU l{};l.supportContactCount=2;l.supportBase=2*(2*N+128+37);l.vascularUnknownCount=11;l.vascularBase=l.supportBase+4;l.unknownCount=l.vascularBase+22;
  NMHumanSupportDispatchGPU support{};support.contactCount=2;support.groundNormal=f4(0,1,0,0);
  std::vector<nm_float4> values(l.unknownCount,f4(1,2,3,4));for(unsigned e=0;e<2;++e){values[l.supportBase+e*2]=f4(0,(mode==0||(mode>=3&&e==0))?1:-1,0,0);values[l.supportBase+e*2+1]=f4(0,1,0,0);}if(mode==2)values[N-1].x=NAN;
  auto solution=g.array(values),hist=g.buf(nullptr,4*sizeof(nm_float4)),ws=g.buf(nullptr,16),changed=g.buf(nullptr,8),status=g.buf(nullptr,2*sizeof(NMMatterStatusGPU)),contacts=g.buf(nullptr,2*sizeof(NMHumanSupportContactGPU)),cone=g.buf(nullptr,16),targets=g.buf(nullptr,4*sizeof(nm_float4)),samples=g.buf(nullptr,4*sizeof(NMContactSampleGPU));
  auto args=g.buf(nullptr,16);
  auto encodeClear=[&](id<MTLCommandBuffer> cb) {
   g.encode(cb,"nm_fgmres_prepare_changed_direction_dispatch",&d,sizeof(d),
    {{1,changed},{2,args}},1,true,32,&l);
   auto e=[cb computeCommandEncoder];[e setComputePipelineState:g.pipe("nm_fgmres_clear_changed_direction_rows")];
   [e setBytes:&d length:sizeof(d) atIndex:0];[e setBytes:&l length:sizeof(l) atIndex:30];
   [e setBuffer:solution offset:0 atIndex:1];[e setBuffer:changed offset:0 atIndex:2];[e setBuffer:status offset:0 atIndex:3];
   [e dispatchThreadgroupsWithIndirectBuffer:args indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
  };
  auto execute=[&](bool fast){memcpy(solution.contents,values.data(),values.size()*sizeof(nm_float4));for(auto b:{ws,changed,status,cone,targets})memset(b.contents,0,b.length);if(mode==4)((NMMatterStatusGPU*)status.contents)[0].code=NM_STATUS_CONTACT_FAILURE;auto cb=[g.queue commandBuffer];auto e=[cb computeCommandEncoder];[e setComputePipelineState:g.pipe(fast?"nm_human_support_resolve_working_set_decision":"nm_human_support_resolve_working_set")];[e setBytes:&d length:sizeof(d) atIndex:0];[e setBytes:&support length:sizeof(support) atIndex:1];[e setBytes:&l length:sizeof(l) atIndex:30];for(auto b:std::initializer_list<GPU::Bind>{{2,solution},{3,hist},{4,ws},{5,changed},{6,status},{7,contacts},{8,cone},{9,targets},{10,samples}})[e setBuffer:b.second offset:0 atIndex:b.first];[e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
   if(fast)encodeClear(cb);
   double time=g.done(cb);std::vector<unsigned char> result(solution.length+ws.length+changed.length+status.length+cone.length+targets.length);size_t o=0;for(auto b:{solution,ws,changed,status,cone,targets}){memcpy(result.data()+o,b.contents,b.length);o+=b.length;}return std::pair{result,time};};
  auto a=execute(false),b=execute(true);need(a.first==b.first,"working-set decision and full-domain clear exact parity");
  printf("{\"path\":\"working_set\",\"nodes\":%u,\"mode\":%u,\"serial_gpu_s\":%.9g,\"parallel_gpu_s\":%.9g}\n",N,mode,a.second,b.second);
  need(((unsigned*)args.contents)[0]==(mode?((l.unknownCount+255)/256):0u),"GPU no-pivot suppression");
  NMVascularLayoutGPU graph{};graph.counts.y=1;graph.ranges.z=11;graph.offsets.y=2;
  NMVascularConnectionGPU edge{};edge.identity.w=1;
  auto edges=g.buf(&edge,sizeof(edge)),vascularCandidate=g.buf(nullptr,22*sizeof(nm_float4)),dummy=g.buf(nullptr,4096);
  for(unsigned env=0;env<2;++env)values[l.vascularBase+env*11+2]=f4(mode?-1:1,0,0,0);
  auto vascular=[&](bool fast){memcpy(solution.contents,values.data(),values.size()*sizeof(nm_float4));for(auto b:{ws,changed,status})memset(b.contents,0,b.length);if(mode==4)((NMMatterStatusGPU*)status.contents)[0].code=NM_STATUS_CONTACT_FAILURE;auto cb=[g.queue commandBuffer];auto e=[cb computeCommandEncoder];
   [e setComputePipelineState:g.pipe(fast?"nm_vascular_resolve_working_set_decision":"nm_vascular_resolve_working_set")];[e setBytes:&d length:sizeof(d) atIndex:0];[e setBytes:&graph length:sizeof(graph) atIndex:1];[e setBytes:&l length:sizeof(l) atIndex:30];
   // Working-set is indexed by every vascular unknown.
   auto vascularWS=g.buf(nullptr,22*sizeof(unsigned));for(auto x:std::initializer_list<GPU::Bind>{{2,edges},{3,vascularCandidate},{4,solution},{5,vascularWS},{6,changed},{7,status},{8,dummy},{9,dummy},{10,dummy},{11,dummy},{12,dummy}})[e setBuffer:x.second offset:0 atIndex:x.first];
   [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];if(fast)encodeClear(cb);
   double seconds=g.done(cb);std::vector<unsigned char> out(solution.length+vascularWS.length+changed.length+status.length);size_t o=0;for(auto x:{solution,vascularWS,changed,status}){memcpy(out.data()+o,x.contents,x.length);o+=x.length;}return std::pair{out,seconds};};
  auto va=vascular(false),vb=vascular(true);need(va.first==vb.first,"vascular working-set and clear exact parity");
  printf("{\"path\":\"vascular_working_set\",\"nodes\":%u,\"mode\":%u,\"serial_gpu_s\":%.9g,\"parallel_gpu_s\":%.9g}\n",N,mode,va.second,vb.second);
 }}
}
static void blood(GPU& g){
 for(unsigned N:{0u,32u,257u,1024u})for(unsigned mode=0;mode<7;++mode){@autoreleasepool{
  NMHumanRespirationDispatch d{};d.environmentCount=2;d.bloodMpmEnabled=1;d.particleCount=N;d.bloodMpmParticleCount=N;d.bloodMpmDensity=1050.6f;d.bloodMpmBulkModulus=2.e9f;d.bloodMpmGridCellWidth=.01f;d.bloodMpmGridNodeMinimum={-100,-100,-100,0};d.bloodMpmGridNodeDimensions={200,200,200,0};d.materialStateStride=2;d.bloodMpmMaterialStateCount=2;
  std::vector<NMParticleStateGPU> particles(2*N);std::vector<NMHumanRespirationState> respiration(2);float volume=1.e-9f,mass=d.bloodMpmDensity*volume;
  for(unsigned e=0;e<2;++e){for(unsigned i=0;i<N;++i){auto& p=particles[e*N+i];p.identity={0,0,0,NM_OBJECT_ACTIVE};float x=float(i%13)*.001f,y=float((i/13)%13)*.001f,z=float(i/169)*.001f;if(mode==2)x=float(i)*.02f;if(mode==3){x=float(i%7)*.001f;y=float((i/7)%7)*.001f;}if(mode==6){x=16777216.f;y=z=0.f;}
    p.positionAndMass=f4(x,y,z,mass);p.velocityAndReferenceVolume=f4(mode==4?.01f:0,0,0,volume);p.deformationRow0=f4(1,0,0,0);p.deformationRow1=f4(0,1,0,0);p.deformationRow2=f4(0,0,1,0);p.referenceAndTemperature=f4(x,y,z,mode==5?310.f+float(i):310.f);if(mode==4){p.affineRow0=f4(0,-1,0,0);p.affineRow1=f4(1,0,0,0);}
    if(mode==0&&i<8)p.identity.w=0;
   }
   auto& n=respiration[e];float batchVolume=volume*(mode==1?4:mode==3?8:1);n.externalBlood.pendingParcel=f4(batchVolume,d.bloodMpmDensity*batchVolume,0,0);n.externalBlood.pendingPositionMass=f4(0,0,0,n.externalBlood.pendingParcel.y);n.externalBlood.total=n.externalBlood.pendingParcel;
  }
  unsigned hashCapacity=1;while(hashCapacity<2*N)hashCapacity*=2;
  auto accepted=g.array(particles),candidate=g.array(particles),checkpoint=g.array(particles),state=g.array(respiration),status=g.buf(nullptr,2*sizeof(NMMatterStatusGPU)),defaults=g.buf(nullptr,4*N*sizeof(float)),sa=g.buf(nullptr,4*N*sizeof(float)),sc=g.buf(nullptr,4*N*sizeof(float)),heads=g.buf(nullptr,2*hashCapacity*4),next=g.buf(nullptr,2*N*4);
  auto run=[&](bool fast){if(!particles.empty()){memcpy(accepted.contents,particles.data(),particles.size()*sizeof(particles[0]));memcpy(candidate.contents,particles.data(),particles.size()*sizeof(particles[0]));}memcpy(state.contents,respiration.data(),state.length);memset(status.contents,0,status.length);memset(sa.contents,0,sa.length);memset(sc.contents,0,sc.length);
   auto cb=[g.queue commandBuffer];g.encode(cb,fast?"nm_human_blood_mpm_handoff_local":"nm_human_blood_mpm_handoff",&d,sizeof(d),{{1,state},{2,status},{3,accepted},{4,candidate},{5,checkpoint},{6,defaults},{7,sa},{8,sc},{9,heads},{10,next}},2,fast,fast?256:1,nullptr,true,{{11,hashCapacity}});double time=g.done(cb);
   auto st=(NMMatterStatusGPU*)status.contents;need((st[0].code==0)==(N>0&&mode!=2&&mode!=5),"blood capacity success/failure");
   // Published state, particles and material history must agree bit for bit:
   // hash order must not change the chosen pair or conservation ledger.
   std::vector<unsigned char> result(state.length+status.length+accepted.length+candidate.length+sa.length+sc.length);size_t o=0;for(auto b:{state,status,accepted,candidate,sa,sc}){memcpy(result.data()+o,b.contents,b.length);o+=b.length;}return std::pair{result,time};};
  auto a=run(false),b=run(true);need(a.first==b.first,"blood closest-pair tie order, particles, history and ledgers exact parity");
  printf("{\"path\":\"blood\",\"particles\":%u,\"mode\":%u,\"serial_gpu_s\":%.9g,\"local_gpu_s\":%.9g}\n",N,mode,a.second,b.second);fflush(stdout);
 }}
}
int main(int argc,const char** argv){@autoreleasepool{need(argc==3,"production matter and respiration libraries");GPU g(argv[1],argv[2]);nodes(g);topology(g);reset(g);blood(g);puts("{\"status\":\"PASS\",\"scope\":\"production GPU algorithm parity, not integrated Human throughput\"}");}}
