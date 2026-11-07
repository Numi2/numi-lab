#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
template<class T> bool eq(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
int main(int argc,char**argv){
 if(argc!=2||!std::filesystem::is_regular_file(argv[1]))return 2;
 std::cout<<std::setprecision(10)<<"{\"cases\":[";bool first=true;
 for(int geom=0;geom<3;++geom)for(float mu:{12345.f})for(float vn:{-1.f,0.f,1.f})for(int far:{0,1}) {
  if(far && vn!=0.f)continue;
  auto model=metalrobo::makeFreeSphereEngineModel();
  model.world.gravityAndTimestep={0,0,0,.01f};
  for(auto& m:model.materials){m.friction.x=mu;m.friction.y=mu;m.response.x=0;m.response.y=0;}
  auto& shape=model.shapes[1];
  float support=.01f;
  if(geom==0){shape.shapeType=MR_SHAPE_SPHERE;shape.dimensions={.01f,0,0,0};shape.contactRestAndBoundingRadius={.001f,0,.01f,0};}
  if(geom==1){shape.shapeType=MR_SHAPE_CAPSULE;shape.dimensions={.01f,.02f,0,0};shape.contactRestAndBoundingRadius={.001f,0,.03f,0};support=.03f;}
  if(geom==2){shape.shapeType=MR_SHAPE_BOX;shape.dimensions={.02f,.01f,.015f,0};shape.contactRestAndBoundingRadius={.001f,0,.027f,0};}
  model.defaultQ[1]=support+(far?.05f:-.0001f);
  model.defaultV[1]=vn;model.defaultV[0]=mu>0?1.f:0.f;
  metalrobo::CompiledWorld world;auto c=metalrobo::compileMetalWorld(model,0,world);
  if(!c.succeeded()){std::cerr<<"compile failure\n";return 3;}
  const std::vector<float> effort(world.nv(),0.f);
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  const std::vector<MRBodyStateGPU> scene{ground};
  const metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,.initialV=model.defaultV,.efforts=effort,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.01f;cfg.physicsSubsteps=1;cfg.velocityIterations=2;cfg.finalVelocityIterations=0;
  cfg.ccdMode=metalrobo::MetalWorldCCDMode::disabled;cfg.applyBodyDamping=false;cfg.captureContactEvidence=true;
  metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];metalrobo::MetalWorldContext ctx(owner);
  metalrobo::MetalWorldResult a,b;auto d=ctx.run(world,batch,cfg,a);auto d2=ctx.run(world,batch,cfg,b);
  if(a.finalV.size()<6||a.contactStatuses.empty())return 4;
  auto status=a.contactStatuses[0];double J[3]={},L[3]={},sumN=0.;
  for(std::size_t i=0;i<std::min<std::size_t>(status.activeContacts,a.contactEvidence.contacts.size());++i){
   const auto&x=a.contactEvidence.contacts[i];auto n=x.normal,t=x.tangent;
   double tangent2[3]={n.y*t.z-n.z*t.y,n.z*t.x-n.x*t.z,n.x*t.y-n.y*t.x};
   double j[3]={n.x*x.impulses.x+t.x*x.impulses.y+tangent2[0]*x.impulses.z,
    n.y*x.impulses.x+t.y*x.impulses.y+tangent2[1]*x.impulses.z,n.z*x.impulses.x+t.z*x.impulses.y+tangent2[2]*x.impulses.z};
   // Both witnesses are body-local. The initial rotation is identity, so the
   // dynamic body's local anchor is its pre-solve COM lever arm.
   if(i>=a.contactEvidence.contactMetadata.size() || (x.bodyA!=1 && x.bodyB!=1)) return 5;
   const auto& meta=a.contactEvidence.contactMetadata[i];
   const auto& anchor=x.bodyB==1?meta.localAnchorB:meta.localAnchorA;
   if(x.bodyA==1)for(double& value:j)value=-value;
   double r[3]={anchor.x,anchor.y,anchor.z};
   for(int k=0;k<3;++k)J[k]+=j[k];
   L[0]+=r[1]*j[2]-r[2]*j[1];L[1]+=r[2]*j[0]-r[0]*j[2];L[2]+=r[0]*j[1]-r[1]*j[0];sumN+=x.impulses.x;
  }
  double me=0,ae=0,work=0;for(int k=0;k<3;++k){
   me=std::max(me,std::abs(double(a.finalV[k]-model.defaultV[k])-J[k]));
   ae=std::max(ae,std::abs(.1*double(a.finalV[k+3]-model.defaultV[k+3])-L[k]));
   work+=.5*(double(a.finalV[k])*a.finalV[k]-double(model.defaultV[k])*model.defaultV[k]);
   work+=.05*(double(a.finalV[k+3])*a.finalV[k+3]-double(model.defaultV[k+3])*model.defaultV[k+3]);
  }
  bool replay=eq(a.finalQ,b.finalQ)&&eq(a.finalV,b.finalV)&&eq(a.finalSceneBodies,b.finalSceneBodies)&&d.firstGPUStatusCode==d2.firstGPUStatusCode;
  bool restore=d.succeeded()||(eq(a.finalQ,model.defaultQ)&&eq(a.finalV,model.defaultV)&&eq(a.finalSceneBodies,scene));
  bool emptyCache=std::all_of(a.contactEvidence.manifoldCounts.begin(),a.contactEvidence.manifoldCounts.end(),[](auto n){return n==0;});
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"geometry\":"<<geom<<",\"friction\":"<<mu<<",\"initial_vn\":"<<vn<<",\"far\":"<<(far?"true":"false")
   <<",\"accepted\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<d.firstGPUStatusCode
   <<",\"contacts\":"<<status.activeContacts<<",\"normal_residual\":"<<status.residuals.y
   <<",\"final_vn\":"<<a.finalV[1]<<",\"final_vt\":"<<a.finalV[0]
   <<",\"replay\":"<<(replay?"true":"false")<<",\"restore\":"<<(restore?"true":"false")<<",\"rejected_cache_empty\":"<<((d.succeeded()||emptyCache)?"true":"false");
  if(d.succeeded())std::cout<<",\"normal_impulse_ns\":"<<sumN<<",\"linear_momentum_error\":"<<me<<",\"angular_momentum_error\":"<<ae<<",\"kinetic_work_j\":"<<work;
  else std::cout<<",\"normal_impulse_ns\":null,\"linear_momentum_error\":null,\"angular_momentum_error\":null,\"kinetic_work_j\":null";
  std::cout<<",\"rows\":[";
  for(std::size_t i=0;i<std::min<std::size_t>(status.activeContacts,a.contactEvidence.contacts.size());++i){
   if(i) std::cout<<",";
   const auto& x=a.contactEvidence.contacts[i]; const auto& m=a.contactEvidence.contactMetadata[i];
   const auto& row=a.contactEvidence.evaluatedRows.at(3*i);
   std::cout<<"{\"body_a\":"<<x.bodyA<<",\"body_b\":"<<x.bodyB
    <<",\"anchor_a\":["<<m.localAnchorA.x<<","<<m.localAnchorA.y<<","<<m.localAnchorA.z<<"]"
    <<",\"anchor_b\":["<<m.localAnchorB.x<<","<<m.localAnchorB.y<<","<<m.localAnchorB.z<<"]"
    <<",\"normal_impulse\":"<<x.impulses.x<<",\"tangent_impulses\":["<<x.impulses.y<<","<<x.impulses.z<<"]"
    <<",\"target\":"<<row.targetVelocity<<",\"pre_velocity\":"<<row.preSolveVelocity
    <<",\"separation\":"<<x.pointAndSeparation.w<<"}";
  }
  std::cout<<"]}";
 }
 std::cout<<"]}\n";
}
