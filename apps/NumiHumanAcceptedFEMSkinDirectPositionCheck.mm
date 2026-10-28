#include "NumiHumanAcceptedFEMSkinRenderPass.hpp"
#include <bit>
#include <iostream>
#include <limits>
using Pass=numi::human::AcceptedFEMSkinRenderPass;
static metalrobo::MetalHybridComputeEncoderCallbacks callbacks(id<MTLComputeCommandEncoder> e){
 metalrobo::MetalHybridComputeEncoderCallbacks c;c.context=(__bridge void*)e;
 c.useHeap=[](void*,void*){};
 c.setPipeline=[](void*p,void*v){[(__bridge id<MTLComputeCommandEncoder>)p setComputePipelineState:(__bridge id<MTLComputePipelineState>)v];};
 c.setBuffer=[](void*p,void*v,std::size_t o,std::uint32_t i){[(__bridge id<MTLComputeCommandEncoder>)p setBuffer:(__bridge id<MTLBuffer>)v offset:o atIndex:i];};
 c.setBytes=[](void*p,const void*v,std::size_t n,std::uint32_t i){[(__bridge id<MTLComputeCommandEncoder>)p setBytes:v length:n atIndex:i];};
 c.dispatchThreads=[](void*p,std::size_t n,std::size_t t){[(__bridge id<MTLComputeCommandEncoder>)p dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(t,1,1)];};
 c.dispatchThreadgroups=[](void*p,std::size_t n,std::size_t t){[(__bridge id<MTLComputeCommandEncoder>)p dispatchThreadgroups:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(t,1,1)];};
 return c;
}
int main(){@autoreleasepool{
 id<MTLDevice>d=MTLCreateSystemDefaultDevice();if(!d)return 2;std::string error;
 std::vector<Pass::Row> rows(3);
 rows[0]={{0,1,2},7,{0,0,1},0};rows[1]={{1,2,0},7,{1,0,0},0};rows[2]={{2,0,1},7,{0,1,0},0};
 const std::vector<Pass::Point> baseline{{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
 const std::vector<std::array<float,4>> accepted{{-0.29977738857269287f,-1.0083298683166504f,0.06120483577251434f,1},
 {-0.2997773587703705f,-1.0083297491073608f,0.06120484322309494f,1},
 {-0.2997783422470093f,-1.0083261728286743f,0.06121484190225601f,1},{0,0,1,1}};
 Pass pass;if(!pass.initialize(d,rows,4,&error)||!pass.updateAcceptedPositions(accepted,baseline,&error)){std::cerr<<error;return 3;}
 std::vector<MRVisualVertexGPUV2> initial(3);for(auto&v:initial){v.position={10,-10,5,1};v.color={0.2,0.3,0.4,1};v.normalAndTangentSign={0,0,1,1};v.tangent={1,0,0,0};}
 id<MTLBuffer>buffer=[d newBufferWithBytes:initial.data() length:initial.size()*sizeof(initial[0]) options:MTLResourceStorageModeShared];
 id<MTLCommandQueue>q=[d newCommandQueue];const auto run=[&]{
  id<MTLCommandBuffer>cmd=[q commandBuffer];id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];
  auto c=callbacks(e);if(!pass.encode(c,(__bridge void*)buffer,0))return false;
  [e endEncoding];[cmd commit];[cmd waitUntilCompleted];return cmd.status==MTLCommandBufferStatusCompleted;};
 if(!run())return 4;
 auto check=[&]{const auto*v=static_cast<const MRVisualVertexGPUV2*>([buffer contents]);const unsigned selected[]={2,1,0};
  for(unsigned i=0;i<3;++i){const float* a=&v[i].position.x;for(unsigned j=0;j<3;++j)
   if(std::bit_cast<std::uint32_t>(a[j])!=std::bit_cast<std::uint32_t>(accepted[selected[i]][j]))return false;
   if(std::memcmp(&v[i].color,&initial[i].color,sizeof(v[i].color)))return false;}return true;};
 if(!check())return 5;
 auto bad=accepted;bad[3][0]=std::numeric_limits<float>::quiet_NaN();
 if(pass.updateAcceptedPositions(bad,baseline,&error)||!run()||!check())return 6;
 auto projected=rows;projected[0].weight[0]=0.5f;projected[0].weight[2]=0.5f;
 Pass projection;if(!projection.initialize(d,projected,4,&error)||projection.updateAcceptedPositions(accepted,baseline,&error))return 7;
 auto invalid=rows;invalid[0].coarseVertex[2]=4;Pass invalidPass;if(invalidPass.initialize(d,invalid,4,&error))return 8;
 std::vector<std::array<float,4>> delta(4,{1,2,3,0});if(!pass.updateAcceptedState(delta,baseline,&error)||!run())return 9;
 const auto*v=static_cast<const MRVisualVertexGPUV2*>([buffer contents]);const unsigned selected[]={2,1,0};
 for(unsigned i=0;i<3;++i){const float*a=&v[i].position.x;for(unsigned j=0;j<3;++j)if(a[j]!=accepted[selected[i]][j]+delta[selected[i]][j])return 10;}
 std::cout<<"accepted_skin_direct_position=PASS device="<<d.name.UTF8String
  <<" one_hot_slots=3 absolute_f32_bits=exact invalid_update_preserves_state=1 projected_row_rejected=1 invalid_node_rejected=1 legacy_delta=PASS physical_qualification=not_claimed\n";
 return 0;
}}
