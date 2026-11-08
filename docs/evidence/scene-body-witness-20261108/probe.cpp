#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
void vec(mr_float4 v){std::cout<<"["<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<"]";}
int main(int argc,char** argv){
 if(argc!=2||!std::filesystem::is_regular_file(argv[1]))return 2;
 std::cout<<std::setprecision(10)<<"{\"cases\":[";
 const std::array<float,4> gaps{0.f,.0006f,.0014f,.004f};
 for(size_t c=0;c<gaps.size();++c){
  auto model=metalrobo::makeFreeSphereEngineModel();model.world.gravityAndTimestep={0,0,0,.01f};model.defaultQ[0]=10.f;model.defaultQ[1]=10.f;
  for(auto& m:model.materials){m.friction.x=.5f;m.friction.y=.5f;m.response.x=0;m.response.y=0;}
  MRBodyPropertiesGPU body{};body.articulationIndex=MR_INVALID_INDEX;body.parentBody=MR_INVALID_INDEX;body.inboundJoint=MR_INVALID_INDEX;body.motionType=MR_MOTION_DYNAMIC;
  body.massAndInverseMass={1.f,1.f,0,0};body.inertiaRow0={.001f,0,0,0};body.inertiaRow1={0,.001f,0,0};body.inertiaRow2={0,0,.001f,0};body.inverseInertiaRow0={1000.f,0,0,0};body.inverseInertiaRow1={0,1000.f,0,0};body.inverseInertiaRow2={0,0,1000.f,0};body.dampingAndSpeedLimits={0,0,100,100};
  const auto global=static_cast<std::uint32_t>(model.bodies.size());model.bodies.push_back(body);
  MRShapeGPU shape{};shape.bodyIndex=global;shape.shapeType=MR_SHAPE_SPHERE;shape.materialIndex=0;shape.collisionGroup=1;shape.collisionMask=~0u;shape.slotGeneration=512;shape.localPosition.w=1;shape.localRotation.w=1;shape.dimensions.x=.05f;shape.contactRestAndBoundingRadius={.002f,0,.05f,0};model.shapes.push_back(shape);
  if(!model.bodyNames.empty())model.bodyNames.push_back("audit_scene_sphere");if(!model.shapeNames.empty())model.shapeNames.push_back("audit_scene_sphere_shape");
  model.world.bodyCount=static_cast<std::uint32_t>(model.bodies.size());model.world.shapeCount=static_cast<std::uint32_t>(model.shapes.size());
  metalrobo::CompiledWorld world;auto comp=metalrobo::compileMetalWorld(model,0,world);if(!comp.succeeded()){std::cerr<<"compile "<<comp.message<<"\n";return 3;}
  if(world.sceneBodyCount()!=2){std::cerr<<"scene count "<<world.sceneBodyCount()<<"\n";return 4;}
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;ground.flagsAndIndices[2]=0;
  MRBodyStateGPU sphere{};sphere.position={0,.05f+gaps[c],0,1};sphere.orientation.w=1;sphere.linearVelocityAndInverseMass={.2f,c==3?0.f:-.3f,0,1};sphere.angularVelocity={0,0,0,0};sphere.inverseInertiaWorldRow0={1000,0,0,0};sphere.inverseInertiaWorldRow1={0,1000,0,0};sphere.inverseInertiaWorldRow2={0,0,1000,0};sphere.flagsAndIndices[0]=MR_MOTION_DYNAMIC;sphere.flagsAndIndices[1]=MR_INVALID_INDEX;sphere.flagsAndIndices[2]=global;
  std::vector<MRBodyStateGPU> scene{ground,sphere};std::vector<float> effort(world.nv(),0);
  metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,.initialV=model.defaultV,.efforts=effort,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.01f;cfg.physicsSubsteps=1;cfg.velocityIterations=2;cfg.finalVelocityIterations=0;cfg.ccdMode=metalrobo::MetalWorldCCDMode::speculative;cfg.applyBodyDamping=false;cfg.captureContactEvidence=true;
  metalrobo::MetalWorldConfig oc;oc.metallibPath=argv[1];metalrobo::MetalWorldContext owner(oc);metalrobo::MetalWorldResult a,b;auto x=owner.run(world,batch,cfg,a);auto y=owner.run(world,batch,cfg,b);
  if(a.contactStatuses.empty()||a.finalSceneBodies.size()!=2){std::cerr<<"missing status or scene count="<<a.finalSceneBodies.size()<<" status="<<static_cast<unsigned>(x.status)<<" message="<<x.message<<"\n";return 5;}
  const auto& st=a.contactStatuses[0];const auto& end=a.finalSceneBodies[1];const bool replay=same(a.finalQ,b.finalQ)&&same(a.finalV,b.finalV)&&same(a.finalSceneBodies,b.finalSceneBodies)&&x.firstGPUStatusCode==y.firstGPUStatusCode;
  if(c)std::cout<<",";std::cout<<"{\"gap\":"<<gaps[c]<<",\"accepted\":"<<(x.succeeded()?"true":"false")<<",\"status\":"<<x.firstGPUStatusCode<<",\"contacts\":"<<st.activeContacts<<",\"replay\":"<<(replay?"true":"false")<<",\"global_sphere\":"<<global<<",\"start_position\":";vec(sphere.position);std::cout<<",\"start_velocity\":";vec(sphere.linearVelocityAndInverseMass);std::cout<<",\"end_position\":";vec(end.position);std::cout<<",\"end_velocity\":";vec(end.linearVelocityAndInverseMass);std::cout<<",\"end_angular\":";vec(end.angularVelocity);std::cout<<",\"rows\":[";
  const size_t n=std::min<size_t>(st.activeContacts,a.contactEvidence.contacts.size());
  for(size_t i=0;i<n;++i){const auto& row=a.contactEvidence.contacts[i];const auto& meta=a.contactEvidence.contactMetadata.at(i);if(i)std::cout<<",";std::cout<<"{\"a\":"<<row.bodyA<<",\"b\":"<<row.bodyB<<",\"point\":";vec(row.pointAndSeparation);std::cout<<",\"normal\":";vec(row.normal);std::cout<<",\"tangent\":";vec(row.tangent);std::cout<<",\"impulse\":";vec(row.impulses);std::cout<<",\"anchor_a\":";vec(meta.localAnchorA);std::cout<<",\"anchor_b\":";vec(meta.localAnchorB);std::cout<<"}";}
  std::cout<<"]}";
 }
 std::cout<<"]}\n";
}
