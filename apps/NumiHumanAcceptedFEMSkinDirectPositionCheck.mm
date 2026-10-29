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

 // A nonzero off-surface embedding must follow the same affine motion as its
 // physical tetrahedron, including rotation. No body-driven animation is used.
 const std::vector<Pass::Point> tetReference{{0.12,-0.23,0.31},{1.12,-0.03,0.41},
     {0.22,1.27,0.61},{0.32,-0.13,2.31}};
 std::vector<Pass::TetrahedralRow> tetRows(3);
 tetRows[0]={{0,1,2,3},{-0.1f,0.4f,0.3f,0.4f}};
 tetRows[1]={{0,1,2,3},{0.25f,0.25f,0.25f,0.25f}};
 tetRows[2]={{0,1,2,3},{1,0,0,0}};
 Pass tetPass;if(!tetPass.initializeTetrahedral(d,tetRows,tetReference,&error)){std::cerr<<error;return 11;}
 std::vector<MRVisualVertexGPUV2> tetInitial(5);
 for(auto&vertex:tetInitial){vertex.position={7,8,9,1};vertex.color={.2,.3,.4,1};vertex.normalAndTangentSign={0,0,1,1};vertex.tangent={1,0,0,0};}
 id<MTLBuffer>tetBuffer=[d newBufferWithBytes:tetInitial.data() length:tetInitial.size()*sizeof(tetInitial[0]) options:MTLResourceStorageModeShared];
 const auto runTet=[&]{
  std::memcpy([tetBuffer contents],tetInitial.data(),tetInitial.size()*sizeof(tetInitial[0]));
  id<MTLCommandBuffer>cmd=[q commandBuffer];id<MTLComputeCommandEncoder>e=[cmd computeCommandEncoder];
  const auto c=callbacks(e);if(!tetPass.encode(c,(__bridge void*)tetBuffer,2))return false;
  [e endEncoding];[cmd commit];[cmd waitUntilCompleted];return cmd.status==MTLCommandBufferStatusCompleted;};
 using Matrix=std::array<std::array<double,3>,3>;
 const Matrix identity{{{1,0,0},{0,1,0},{0,0,1}}},rotation{{{0,-1,0},{1,0,0},{0,0,1}}},
     affine{{{1.1,.2,-.1},{.1,.9,.15},{.02,-.03,1.2}}};
 double maximumTetPositionError=0,maximumTetDirectionError=0;
 const auto checkTet=[&](const Matrix&F,const Pass::Point&translation){
  const auto*vertices=static_cast<const MRVisualVertexGPUV2*>([tetBuffer contents]);
  if(std::memcmp(vertices,tetInitial.data(),2*sizeof(tetInitial[0])))return false;
  double normal[3]={F[1][0]*F[2][1]-F[2][0]*F[1][1],
                   F[2][0]*F[0][1]-F[0][0]*F[2][1],
                   F[0][0]*F[1][1]-F[1][0]*F[0][1]};
  double tangent[3]={F[0][0],F[1][0],F[2][0]};
  const double nl=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
  const double tl=std::sqrt(tangent[0]*tangent[0]+tangent[1]*tangent[1]+tangent[2]*tangent[2]);
  for(unsigned row=0;row<3;++row){
   Pass::Point expected=translation;
   for(unsigned a=0;a<3;++a)for(unsigned c=0;c<4;++c)for(unsigned b=0;b<3;++b)
    expected[a]+=double(tetRows[row].weight[c])*F[a][b]*tetReference[tetRows[row].node[c]][b];
   const auto&vertex=vertices[row+2];const float*position=&vertex.position.x;
   const float*n=&vertex.normalAndTangentSign.x;const float*t=&vertex.tangent.x;
   for(unsigned a=0;a<3;++a){
    maximumTetPositionError=std::max(maximumTetPositionError,std::abs(double(position[a])-expected[a]));
    maximumTetDirectionError=std::max({maximumTetDirectionError,std::abs(double(n[a])-normal[a]/nl),std::abs(double(t[a])-tangent[a]/tl)});
   }
   if(std::memcmp(&vertex.color,&tetInitial[row+2].color,sizeof(vertex.color)))return false;
  }
  return maximumTetPositionError<2e-6&&maximumTetDirectionError<2e-6;
 };
 const auto applyTet=[&](const Matrix&F,const Pass::Point&translation){
  std::vector<std::array<float,4>> positions(4);
  for(unsigned i=0;i<4;++i){positions[i][3]=1;for(unsigned a=0;a<3;++a){
   double value=translation[a];for(unsigned b=0;b<3;++b)value+=F[a][b]*tetReference[i][b];positions[i][a]=float(value);}}
  return tetPass.updateAcceptedTetrahedralPositions(positions,&error)&&runTet()&&checkTet(F,translation);
 };
 if(!runTet()||!checkTet(identity,{0,0,0})||
    !applyTet(identity,{.4,-.3,.2})||!applyTet(rotation,{.4,-.3,.2})||
    !applyTet(affine,{-.2,.1,.05}))return 12;
 auto badTetPositions=std::vector<std::array<float,4>>(4,{0,0,0,1});badTetPositions[3][0]=std::numeric_limits<float>::quiet_NaN();
 if(tetPass.updateAcceptedTetrahedralPositions(badTetPositions,&error)||!runTet()||!checkTet(affine,{-.2,.1,.05}))return 13;
 if(tetPass.updateAcceptedState(delta,tetReference,&error))return 14;
 auto invalidTetRows=tetRows;invalidTetRows[0].node[3]=4;Pass invalidTet;
 if(invalidTet.initializeTetrahedral(d,invalidTetRows,tetReference,&error))return 15;
 invalidTetRows=tetRows;invalidTetRows[0].weight[0]=.1f;
 if(invalidTet.initializeTetrahedral(d,invalidTetRows,tetReference,&error))return 16;
 invalidTetRows=tetRows;invalidTetRows[0].node[3]=2;
 if(invalidTet.initializeTetrahedral(d,invalidTetRows,tetReference,&error))return 17;
 auto invalidReference=tetReference;invalidReference[3]=invalidReference[2];
 if(invalidTet.initializeTetrahedral(d,tetRows,invalidReference,&error))return 18;
 std::cout<<"accepted_skin_tetrahedral=PASS zero_motion=PASS translation=PASS rotation_90=PASS affine_patch=PASS invalid_update_preserves_state=1"
     <<" maximum_position_error_m="<<maximumTetPositionError<<" maximum_direction_error="<<maximumTetDirectionError
     <<" physical_qualification=not_claimed\n";
 std::cout<<"accepted_skin_direct_position=PASS device="<<d.name.UTF8String
  <<" one_hot_slots=3 absolute_f32_bits=exact invalid_update_preserves_state=1 projected_row_rejected=1 invalid_node_rejected=1 legacy_delta=PASS physical_qualification=not_claimed\n";
 return 0;
}}
