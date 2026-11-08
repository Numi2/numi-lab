#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::cout<<std::setprecision(12)<<"{\"cases\":["; bool first=true;
 for(float mass : {1.e-7f,1.e-6f,1.f}) for(float angle : {-0.37f,0.29f}) for(int iterations : {2,32}) for(int testcase=0;testcase<3;testcase++) {
  auto m=metalrobo::makeFreeSphereEngineModel();
  m.world.gravityAndTimestep={0,0,0,0.001f};
  for(auto &mat:m.materials) {mat.friction={};mat.response={};}
  auto& sh=m.shapes[1];sh.shapeType=MR_SHAPE_BOX;
  constexpr float hx=.2f,hy=.02f,hz=.03f, gap=.0001f;
  sh.dimensions={hx,hy,hz,0};sh.contactRestAndBoundingRadius={.001f,0,std::sqrt(hx*hx+hy*hy+hz*hz),0};
  auto &b=m.bodies[1];b.massAndInverseMass={mass,1.f/mass,0,0};const float ix=mass*(hy*hy+hz*hz)/3.f,iy=mass*(hx*hx+hz*hz)/3.f,iz=mass*(hx*hx+hy*hy)/3.f;
  b.inertiaRow0={ix,0,0,0};b.inertiaRow1={0,iy,0,0};b.inertiaRow2={0,0,iz,0};
  b.inverseInertiaRow0={1/ix,0,0,0};b.inverseInertiaRow1={0,1/iy,0,0};b.inverseInertiaRow2={0,0,1/iz,0};
  m.defaultQ[1]=hx*std::abs(std::sin(angle))+hy*std::abs(std::cos(angle))+gap;
  m.defaultQ[3]=0;m.defaultQ[4]=0;m.defaultQ[5]=std::sin(angle/2);m.defaultQ[6]=std::cos(angle/2);
  std::fill(m.defaultV.begin(),m.defaultV.end(),0.f);
  m.defaultV[0]=float(testcase-1);
  m.defaultV[1]=-1.f;
  metalrobo::CompiledWorld world;auto compiled=metalrobo::compileMetalWorld(m,0,world);
  if(!compiled.succeeded()){std::cerr<<"compile: "<<compiled.message<<"\n";return 3;}
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  std::vector<MRBodyStateGPU> scene{ground}; std::vector<float> efforts(world.nv(),0.f);
  const metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=m.defaultQ,.initialV=m.defaultV,.efforts=efforts,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.001f;cfg.physicsSubsteps=1;cfg.velocityIterations=iterations;cfg.finalVelocityIterations=0;
  cfg.ccdMode=metalrobo::MetalWorldCCDMode::disabled;cfg.captureContactEvidence=true;cfg.applyBodyDamping=false;
  metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];metalrobo::MetalWorldContext ctx(owner);
  metalrobo::MetalWorldResult r,rr;auto s=ctx.run(world,batch,cfg,r);auto ss=ctx.run(world,batch,cfg,rr);
  const auto same=[](auto& a,auto& c){return a.size()==c.size()&&(a.empty()||std::memcmp(a.data(),c.data(),a.size()*sizeof(a[0]))==0);};
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"mass\":"<<mass<<",\"angle\":"<<angle<<",\"iterations\":"<<iterations<<",\"case\":"<<testcase<<",\"initial_y\":"<<m.defaultQ[1]<<",\"success\":"<<(s.succeeded()?"true":"false")<<",\"gpu_status\":"<<s.firstGPUStatusCode<<",\"repeat\":"<<((same(r.finalQ,rr.finalQ)&&same(r.finalV,rr.finalV)&&s.firstGPUStatusCode==ss.firstGPUStatusCode)?"true":"false")<<",\"contacts\":"<<(r.contactStatuses.empty()?0:r.contactStatuses[0].activeContacts)<<",\"gap\":"<<gap<<",\"q\":[";
  for(size_t i=0;i<r.finalQ.size();i++){if(i)std::cout<<",";std::cout<<r.finalQ[i];}
  std::cout<<"],\"v\":[";for(size_t i=0;i<r.finalV.size();i++){if(i)std::cout<<",";std::cout<<r.finalV[i];}
  std::cout<<"],\"solver_residual\":"<<(r.contactStatuses.empty() ? -1 : r.contactStatuses[0].residuals.y)<<",\"rows\":[";
  unsigned count=r.contactStatuses.empty()?0:r.contactStatuses[0].activeContacts;
  if (count>r.contactEvidence.contacts.size() || 3*count>r.contactEvidence.evaluatedRows.size()) return 4;
  for(unsigned i=0;i<count;++i) {
   if(i)std::cout<<",";const auto &c=r.contactEvidence.contacts[i];const auto &e=r.contactEvidence.evaluatedRows[3*i];
   std::cout<<"{\"bodyA\":"<<c.bodyA<<",\"bodyB\":"<<c.bodyB<<",\"point\":["<<c.pointAndSeparation.x<<","<<c.pointAndSeparation.y<<","<<c.pointAndSeparation.z<<"],\"direction\":["<<e.direction.x<<","<<e.direction.y<<","<<e.direction.z<<"],\"lambda\":"<<c.impulses.x<<",\"tangent\":["<<c.impulses.y<<","<<c.impulses.z<<"],\"target\":"<<e.targetVelocity<<",\"reg\":"<<e.regularization<<"}";
  }
  std::cout<<"]}";
 }
 std::cout<<"]}\n";
}
