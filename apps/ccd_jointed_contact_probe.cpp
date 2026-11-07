#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
template<class T> bool eq(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
int main(int argc,char**argv){
 if(argc!=2||!std::filesystem::is_regular_file(argv[1]))return 2;
 std::cout<<std::setprecision(10)<<"{\"cases\":[";bool first=true;
 for(auto speeds: {std::pair{-1.f,0.f},std::pair{-.25f,-.75f},std::pair{0.f,0.f},std::pair{1.f,0.f}}){
  auto m=metalrobo::makeFreeSphereEngineModel();m.name="jointed_contact_analytic";
  m.world.bodyCount=3;m.world.jointCount=1;m.world.nq=8;m.world.nv=7;m.world.gravityAndTimestep={0,0,0,.01f};
  auto& art=m.articulations[0];art.bodyCount=2;art.jointCount=1;art.nq=8;art.nv=7;
  MRJointDescriptorGPU joint{};joint.parentBody=1;joint.childBody=2;joint.jointType=MR_JOINT_PRISMATIC;joint.qOffset=7;joint.nq=1;joint.vOffset=6;joint.nv=1;joint.axis0={0,1,0,0};joint.parentRotation={0,0,0,1};joint.childRotation={0,0,0,1};m.joints.push_back(joint);
  auto child=m.bodies[1];child.parentBody=1;child.inboundJoint=0;child.massAndInverseMass={.5f,2,0,0};child.inertiaRow0={.05f,0,0,0};child.inertiaRow1={0,.05f,0,0};child.inertiaRow2={0,0,.05f,0};child.inverseInertiaRow0={20,0,0,0};child.inverseInertiaRow1={0,20,0,0};child.inverseInertiaRow2={0,0,20,0};m.bodies.push_back(child);
  MRDofPropertiesGPU dof{};dof.articulationIndex=0;dof.jointIndex=0;dof.qIndex=7;dof.vIndex=6;m.dofs.push_back(dof);
  m.shapes[1].bodyIndex=2;m.shapes[1].dimensions={.01f,0,0,0};m.shapes[1].contactRestAndBoundingRadius={.001f,0,.01f,0};
  for(auto& mat:m.materials){mat.friction={0,0,0,0};mat.response={0,0,0,0};}
  m.defaultQ[1]=.0099f;m.defaultQ.push_back(0);m.defaultV[1]=speeds.first;m.defaultV.push_back(speeds.second);
  std::string why;if(!m.valid(&why)){std::cerr<<"invalid model "<<why<<"\n";return 3;}
  metalrobo::CompiledWorld w;auto c=metalrobo::compileMetalWorld(m,0,w);if(!c.succeeded()){std::cerr<<"compile "<<c.message<<"\n";return 4;}
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  const std::vector<MRBodyStateGPU> scene{ground};const std::vector<float> effort(w.nv(),0.f);
  metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=m.defaultQ,.initialV=m.defaultV,.efforts=effort,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.01f;cfg.physicsSubsteps=1;cfg.velocityIterations=2;cfg.finalVelocityIterations=0;cfg.ccdMode=metalrobo::MetalWorldCCDMode::disabled;cfg.applyBodyDamping=false;cfg.captureContactEvidence=true;
  metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];metalrobo::MetalWorldContext ctx(owner);metalrobo::MetalWorldResult a,b;auto d=ctx.run(w,batch,cfg,a);auto d2=ctx.run(w,batch,cfg,b);
  if(a.finalV.size()<7||a.contactStatuses.empty())return 5;
  double J=0;for(size_t i=0;i<std::min<size_t>(a.contactStatuses[0].activeContacts,a.contactEvidence.contacts.size());++i)J+=a.contactEvidence.contacts[i].impulses.x;
  double p0=speeds.first, c0=speeds.first+speeds.second,p1=a.finalV[1],c1=p1+a.finalV[6];
  double ke0=.5*p0*p0+.25*c0*c0,ke1=.5*p1*p1+.25*c1*c1;
  double expectedJ=.5*std::max(-c0,0.0),expectedP=p0,expectedC=std::max(c0,0.0);
  bool replay=eq(a.finalQ,b.finalQ)&&eq(a.finalV,b.finalV)&&eq(a.finalSceneBodies,b.finalSceneBodies)&&d.firstGPUStatusCode==d2.firstGPUStatusCode;
  bool restore=d.succeeded()||(eq(a.finalQ,m.defaultQ)&&eq(a.finalV,m.defaultV)&&eq(a.finalSceneBodies,scene));
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"initial_parent_v\":"<<p0<<",\"initial_joint_v\":"<<speeds.second<<",\"accepted\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<d.firstGPUStatusCode<<",\"contacts\":"<<a.contactStatuses[0].activeContacts<<",\"normal_residual\":"<<a.contactStatuses[0].residuals.y<<",\"normal_impulse\":"<<J<<",\"reference_impulse\":"<<expectedJ<<",\"parent_v\":"<<p1<<",\"reference_parent_v\":"<<expectedP<<",\"child_v\":"<<c1<<",\"reference_child_v\":"<<expectedC<<",\"work_j\":"<<ke1-ke0<<",\"reference_work_j\":"<<-.25*std::min(c0,0.0)*std::min(c0,0.0)<<",\"replay\":"<<(replay?"true":"false")<<",\"restore\":"<<(restore?"true":"false")<<"}";
 }
 std::cout<<"]}\n";
}
