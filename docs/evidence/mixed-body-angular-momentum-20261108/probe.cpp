#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <array>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
template<class T> bool same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
void vec(mr_float4 v){std::cout<<"["<<v.x<<","<<v.y<<","<<v.z<<","<<v.w<<"]";}
void floats(const std::vector<float>& v){std::cout<<"[";for(size_t i=0;i<v.size();i++){if(i)std::cout<<",";std::cout<<v[i];}std::cout<<"]";}
int main(int argc,char** argv){
 if(argc!=2||!std::filesystem::is_regular_file(argv[1]))return 2;
 std::cout<<std::setprecision(10)<<"{\"cases\":[";
 const std::array<float,3> gaps{0.f,.0014f,.0014f};
 for(size_t c=0;c<gaps.size();c++){
  auto model=metalrobo::makeFreeSphereEngineModel();model.world.gravityAndTimestep={0,0,0,.01f};model.defaultQ[0]=0;model.defaultQ[1]=1;model.defaultQ[2]=0;
  model.shapes[1].dimensions.x=.05f;model.shapes[1].contactRestAndBoundingRadius={.002f,0,.05f,0};
  auto& art=model.bodies[1];art.inertiaRow0={.001f,0,0,0};art.inertiaRow1={0,.001f,0,0};art.inertiaRow2={0,0,.001f,0};art.inverseInertiaRow0={1000,0,0,0};art.inverseInertiaRow1={0,1000,0,0};art.inverseInertiaRow2={0,0,1000,0};
  for(auto& m:model.materials){m.friction.x=c==2?0:.5f;m.friction.y=c==2?0:.5f;m.response.x=0;m.response.y=0;}
  MRBodyPropertiesGPU body=art;body.articulationIndex=MR_INVALID_INDEX;body.parentBody=MR_INVALID_INDEX;body.inboundJoint=MR_INVALID_INDEX;
  const auto global=static_cast<std::uint32_t>(model.bodies.size());model.bodies.push_back(body);
  MRShapeGPU shape=model.shapes[1];shape.bodyIndex=global;shape.slotGeneration=512;model.shapes.push_back(shape);
  if(!model.bodyNames.empty())model.bodyNames.push_back("audit_scene_sphere");if(!model.shapeNames.empty())model.shapeNames.push_back("audit_scene_sphere_shape");
  model.world.bodyCount=static_cast<std::uint32_t>(model.bodies.size());model.world.shapeCount=static_cast<std::uint32_t>(model.shapes.size());
  metalrobo::CompiledWorld world;auto comp=metalrobo::compileMetalWorld(model,0,world);if(!comp.succeeded()){std::cerr<<"compile "<<comp.message<<"\n";return 3;}
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;ground.flagsAndIndices[2]=0;
  MRBodyStateGPU sphere{};sphere.position={.1f+gaps[c],1,0,1};sphere.orientation.w=1;sphere.linearVelocityAndInverseMass={-.3f,.2f,0,1};sphere.inverseInertiaWorldRow0={1000,0,0,0};sphere.inverseInertiaWorldRow1={0,1000,0,0};sphere.inverseInertiaWorldRow2={0,0,1000,0};sphere.flagsAndIndices[0]=MR_MOTION_DYNAMIC;sphere.flagsAndIndices[1]=MR_INVALID_INDEX;sphere.flagsAndIndices[2]=global;
  std::vector<MRBodyStateGPU> scene{ground,sphere};std::vector<float> effort(world.nv(),0);
  metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,.initialV=model.defaultV,.efforts=effort,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.01f;cfg.physicsSubsteps=1;cfg.velocityIterations=2;cfg.finalVelocityIterations=0;cfg.ccdMode=metalrobo::MetalWorldCCDMode::speculative;cfg.applyBodyDamping=false;cfg.captureContactEvidence=true;
  metalrobo::MetalWorldConfig oc;oc.metallibPath=argv[1];metalrobo::MetalWorldContext owner(oc);metalrobo::MetalWorldResult a,b;auto x=owner.run(world,batch,cfg,a);auto y=owner.run(world,batch,cfg,b);
  if(a.contactStatuses.empty()||a.finalSceneBodies.size()!=2){std::cerr<<"missing state or status: "<<x.message<<"\n";return 5;}
  const auto& st=a.contactStatuses[0];const auto& end=a.finalSceneBodies[1];const bool replay=same(a.finalQ,b.finalQ)&&same(a.finalV,b.finalV)&&same(a.finalSceneBodies,b.finalSceneBodies)&&x.firstGPUStatusCode==y.firstGPUStatusCode;
  if(c)std::cout<<",";std::cout<<"{\"gap\":"<<gaps[c]<<",\"friction\":"<<(c==2?0:.5)<<",\"accepted\":"<<(x.succeeded()?"true":"false")<<",\"status\":"<<x.firstGPUStatusCode<<",\"contacts\":"<<st.activeContacts<<",\"replay\":"<<(replay?"true":"false")<<",\"initial_q\":";floats(model.defaultQ);std::cout<<",\"initial_v\":";floats(model.defaultV);std::cout<<",\"final_q\":";floats(a.finalQ);std::cout<<",\"final_v\":";floats(a.finalV);std::cout<<",\"scene_start_position\":";vec(sphere.position);std::cout<<",\"scene_start_velocity\":";vec(sphere.linearVelocityAndInverseMass);std::cout<<",\"scene_end_position\":";vec(end.position);std::cout<<",\"scene_end_velocity\":";vec(end.linearVelocityAndInverseMass);std::cout<<",\"scene_end_angular\":";vec(end.angularVelocity);std::cout<<",\"rows\":[";
  const size_t n=std::min<size_t>(st.activeContacts,a.contactEvidence.contacts.size());for(size_t i=0;i<n;i++){const auto& row=a.contactEvidence.contacts[i];const auto& meta=a.contactEvidence.contactMetadata.at(i);if(i)std::cout<<",";std::cout<<"{\"a\":"<<row.bodyA<<",\"b\":"<<row.bodyB<<",\"point\":";vec(row.pointAndSeparation);std::cout<<",\"normal\":";vec(row.normal);std::cout<<",\"tangent\":";vec(row.tangent);std::cout<<",\"impulse\":";vec(row.impulses);std::cout<<",\"anchor_a\":";vec(meta.localAnchorA);std::cout<<",\"anchor_b\":";vec(meta.localAnchorB);std::cout<<"}";}
  std::cout<<"]}";
 }
 std::cout<<"]}\n";
}
