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
 for(auto speeds: {std::pair{0.f,-5.f},std::pair{-.25f,-3.75f},std::pair{0.f,0.f},std::pair{0.f,5.f}}){
  auto m=metalrobo::makeFreeSphereEngineModel();m.name="jointed_contact_analytic";
  m.world.bodyCount=3;m.world.jointCount=1;m.world.nq=8;m.world.nv=7;m.world.gravityAndTimestep={0,0,0,.01f};
  auto& art=m.articulations[0];art.bodyCount=2;art.jointCount=1;art.nq=8;art.nv=7;
  MRJointDescriptorGPU joint{};joint.parentBody=1;joint.childBody=2;joint.jointType=MR_JOINT_REVOLUTE;joint.qOffset=7;joint.nq=1;joint.vOffset=6;joint.nv=1;joint.axis0={0,0,1,0};joint.parentRotation={0,0,0,1};joint.childRotation={0,0,0,1};m.joints.push_back(joint);
  auto child=m.bodies[1];child.parentBody=1;child.inboundJoint=0;child.massAndInverseMass={.5f,2,0,0};child.inertiaRow0={.05f,0,0,0};child.inertiaRow1={0,.05f,0,0};child.inertiaRow2={0,0,.05f,0};child.inverseInertiaRow0={20,0,0,0};child.inverseInertiaRow1={0,20,0,0};child.inverseInertiaRow2={0,0,20,0};m.bodies.push_back(child);
  MRDofPropertiesGPU dof{};dof.articulationIndex=0;dof.jointIndex=0;dof.qIndex=7;dof.vIndex=6;m.dofs.push_back(dof);
  m.shapes[1].bodyIndex=2;m.shapes[1].localPosition={.2f,0,0,0};m.shapes[1].dimensions={.01f,0,0,0};m.shapes[1].contactRestAndBoundingRadius={.001f,0,.01f,0};
  for(auto& mat:m.materials){mat.friction={.5f,.5f,0,0};mat.response={0,0,0,0};}
  m.defaultQ[1]=.0099f;m.defaultQ.push_back(0);m.defaultV[0]=1.f;m.defaultV[1]=speeds.first;m.defaultV.push_back(speeds.second);
  std::string why;if(!m.valid(&why)){std::cerr<<"invalid model "<<why<<"\n";return 3;}
  metalrobo::CompiledWorld w;auto c=metalrobo::compileMetalWorld(m,0,w);if(!c.succeeded()){std::cerr<<"compile "<<c.message<<"\n";return 4;}
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  const std::vector<MRBodyStateGPU> scene{ground};const std::vector<float> effort(w.nv(),0.f);
  metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=m.defaultQ,.initialV=m.defaultV,.efforts=effort,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.01f;cfg.physicsSubsteps=1;cfg.velocityIterations=2;cfg.finalVelocityIterations=0;cfg.ccdMode=metalrobo::MetalWorldCCDMode::disabled;cfg.applyBodyDamping=false;cfg.captureContactEvidence=true;
  metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];metalrobo::MetalWorldContext ctx(owner);metalrobo::MetalWorldResult a,b;auto d=ctx.run(w,batch,cfg,a);auto d2=ctx.run(w,batch,cfg,b);
  if(a.finalV.size()<7||a.contactStatuses.empty())return 5;
  double J=0,Jx=0;for(size_t i=0;i<std::min<size_t>(a.contactStatuses[0].activeContacts,a.contactEvidence.contacts.size());++i){
   const auto& c=a.contactEvidence.contacts[i];J+=c.impulses.x;
   const double t2x=c.normal.y*c.tangent.z-c.normal.z*c.tangent.y;
   Jx+=c.tangent.x*c.impulses.y+t2x*c.impulses.z;
  }
  // Independent sliding reference from planar contact impulse matrix.
  // Wnn=1/1.5+0.2^2/.05, Wnt=0.2*.01/.05, Wtt=1/1.5+.01^2/.05.
  double p0=speeds.first,omega0=speeds.second,vn0=p0+.2*omega0,vt0=1.+.01*omega0;
  const double Wnn=1./1.5+.2*.2/.05,Wnt=.2*.01/.05,Wtt=1./1.5+.01*.01/.05;
  double expectedJ=std::max(-vn0,0.0)/(Wnn-.5*Wnt),expectedJx=-.5*expectedJ;
  double expectedP=p0+expectedJ/1.5,expectedC=vn0+Wnn*expectedJ+Wnt*expectedJx;
  double vx=a.finalV[0],p1=a.finalV[1],omega1=a.finalV[6]+a.finalV[5],c1=p1+.2*omega1;
  double vt1=vx+.01*omega1;
  double ke0=.75*(1.+p0*p0)+.025*omega0*omega0;
  double ke1=.75*(vx*vx+p1*p1)+.05*a.finalV[5]*a.finalV[5]+.025*omega1*omega1;
  double expectedWork=vn0*expectedJ+vt0*expectedJx+.5*(Wnn*expectedJ*expectedJ+2*Wnt*expectedJ*expectedJx+Wtt*expectedJx*expectedJx);
  bool replay=eq(a.finalQ,b.finalQ)&&eq(a.finalV,b.finalV)&&eq(a.finalSceneBodies,b.finalSceneBodies)&&d.firstGPUStatusCode==d2.firstGPUStatusCode;
  bool restore=d.succeeded()||(eq(a.finalQ,m.defaultQ)&&eq(a.finalV,m.defaultV)&&eq(a.finalSceneBodies,scene));
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"initial_parent_v\":"<<p0<<",\"initial_joint_v\":"<<speeds.second<<",\"accepted\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<d.firstGPUStatusCode<<",\"contacts\":"<<a.contactStatuses[0].activeContacts<<",\"normal_residual\":"<<a.contactStatuses[0].residuals.y<<",\"normal_impulse\":"<<J<<",\"reference_impulse\":"<<expectedJ<<",\"tangent_impulse\":"<<Jx<<",\"reference_tangent_impulse\":"<<expectedJx<<",\"parent_v\":"<<p1<<",\"reference_parent_v\":"<<expectedP<<",\"child_v\":"<<c1<<",\"reference_child_v\":"<<expectedC<<",\"tangent_slip\":"<<vt1<<",\"work_j\":"<<ke1-ke0<<",\"reference_work_j\":"<<expectedWork<<",\"replay\":"<<(replay?"true":"false")<<",\"restore\":"<<(restore?"true":"false")<<"}";
 }
 std::cout<<"]}\n";
}
