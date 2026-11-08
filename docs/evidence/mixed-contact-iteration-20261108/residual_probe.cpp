#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/MetalWorld.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
template<class T>bool eq(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
int main(int argc,char**argv){
 if(argc!=2||!std::filesystem::is_regular_file(argv[1]))return 2;
 std::cout<<std::setprecision(10)<<"{\"cases\":[";bool first=true;
 for(int iterations:{1,2,4,8,16,32,64}) for(float mu:{.5f,0.f}) for(float vn:{-1.f,0.f,1.f}){
  auto m=metalrobo::makeFreeSphereEngineModel();m.world.gravityAndTimestep={0,0,0,.001f};m.defaultQ[1]=.0099f;m.defaultV[0]=1.f;m.defaultV[1]=vn;
  m.shapes[1].dimensions={.01f,0,0,0};m.shapes[1].contactRestAndBoundingRadius={.001f,0,.01f,0};
  for(auto&x:m.materials){x.friction={mu,mu,0,0};x.response={0,0,0,0};}
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;std::vector<MRBodyStateGPU> scene{ground};
  metalrobo::DiscreteRodMaterial mat;mat.radius=.003;mat.density=1000;mat.youngModulus=1.e5;
  metalrobo::HeterogeneousRodProgram rod;rod.instanceId="mixed_test_rod";rod.model=metalrobo::makeStraightSutureRod(3,.04,mat);rod.defaultState=metalrobo::makeDiscreteElasticRodDefaultState(rod.model);
  for(auto&p:rod.defaultState.positions){p[0]-=.02;p[1]=.0228;}
  rod.collision.enableCCD=false;rod.collision.contactOffset=.001;rod.stepConfig.gravity={0,0,0};rod.stepConfig.timestep=.001;
  const std::vector<metalrobo::HeterogeneousWorldComponent> components{{&m,"body",scene}};
  const std::vector<metalrobo::HeterogeneousRodProgram> rods{rod};metalrobo::HeterogeneousWorld hw;
  metalrobo::EngineModelComposeConfig compose;compose.gravityAndTimestep={0,0,0,.001f};auto hd=metalrobo::composeHeterogeneousWorld(components,rods,hw,compose);if(!hd.succeeded()){std::cerr<<"compose "<<hd.message<<"\n";return 3;}
  metalrobo::CompiledWorld world;auto cd=metalrobo::compileMetalWorld(hw,world);if(!cd.succeeded()){std::cerr<<"compile "<<cd.message<<"\n";return 4;}
  std::vector<float> effort(world.nv(),0);metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=hw.model.defaultQ,.initialV=hw.model.defaultV,.efforts=effort,.initialSceneBodies=hw.defaultSceneBodies};
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.001f;cfg.physicsSubsteps=1;cfg.velocityIterations=iterations;cfg.finalVelocityIterations=0;cfg.ccdMode=metalrobo::MetalWorldCCDMode::disabled;cfg.applyBodyDamping=false;cfg.captureContactEvidence=true;
  metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];metalrobo::MetalWorldContext ctx(owner);metalrobo::MetalWorldResult a,b;auto da=ctx.run(world,batch,cfg,a);auto db=ctx.run(world,batch,cfg,b);
  if(a.finalV.size()<6||a.contactStatuses.empty())return 5;
  auto st=a.contactStatuses[0];bool mixed=false;unsigned countRod=0,countRigid=0;double groundJ[3]={};
  for(unsigned i=0;i<std::min<size_t>(st.activeContacts,a.contactEvidence.contacts.size());++i){const auto& c=a.contactEvidence.contacts[i];bool isrod=(c.flags&MR_CONSTRAINT_FLAG_ROD_ENDPOINT)!=0;countRod+=isrod;countRigid+=!isrod;
   if(!isrod && c.bodyA==0 && c.bodyB==1){auto n=c.normal,t=c.tangent;double t2[3]={n.y*t.z-n.z*t.y,n.z*t.x-n.x*t.z,n.x*t.y-n.y*t.x};groundJ[0]+=n.x*c.impulses.x+t.x*c.impulses.y+t2[0]*c.impulses.z;groundJ[1]+=n.y*c.impulses.x+t.y*c.impulses.y+t2[1]*c.impulses.z;groundJ[2]+=n.z*c.impulses.x+t.z*c.impulses.y+t2[2]*c.impulses.z;}
  }
  for(auto& island:a.contactEvidence.islands)if(island.rodNodeCount && island.articulationNodeCount && island.constraintCount>=2)mixed=true;
  double P[3]={a.finalV[0]-m.defaultV[0],a.finalV[1]-m.defaultV[1],a.finalV[2]-m.defaultV[2]};
  for(size_t i=0;i<std::min(a.finalRodNodes.size(),rod.model.nodeMasses.size());++i){P[0]+=rod.model.nodeMasses[i]*a.finalRodNodes[i].velocity.x;P[1]+=rod.model.nodeMasses[i]*a.finalRodNodes[i].velocity.y;P[2]+=rod.model.nodeMasses[i]*a.finalRodNodes[i].velocity.z;}
  double err=0;for(int i=0;i<3;i++)err=std::max(err,std::abs(P[i]-groundJ[i]));
  auto sameSpan=[](const auto& v,const auto& span){using T=typename std::decay_t<decltype(v)>::value_type;return v.size()==span.size()&&(v.empty()||std::memcmp(v.data(),span.data(),v.size()*sizeof(T))==0);};
  bool restored=eq(a.finalQ,hw.model.defaultQ)&&eq(a.finalV,hw.model.defaultV)&&eq(a.finalSceneBodies,hw.defaultSceneBodies)&&sameSpan(a.finalRodNodes,world.defaultRodNodes())&&sameSpan(a.finalRodEdges,world.defaultRodEdges());
  bool replay=eq(a.finalQ,b.finalQ)&&eq(a.finalV,b.finalV)&&eq(a.finalRodNodes,b.finalRodNodes)&&eq(a.finalRodEdges,b.finalRodEdges)&&da.firstGPUStatusCode==db.firstGPUStatusCode;
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"iterations\":"<<iterations<<",\"mu\":"<<mu<<",\"initial_vn\":"<<vn<<",\"accepted\":"<<(da.succeeded()?"true":"false")<<",\"status\":"<<da.firstGPUStatusCode<<",\"contacts\":"<<st.activeContacts<<",\"rod_contacts\":"<<countRod<<",\"rigid_contacts\":"<<countRigid<<",\"mixed_island\":"<<(mixed?"true":"false")<<",\"normal_residual\":"<<st.residuals.y<<",\"restored\":"<<(restored?"true":"false")<<",\"candidate_momentum_error_uninterpretable_on_rejection\":"<<err<<",\"ground_impulse\":["<<groundJ[0]<<","<<groundJ[1]<<","<<groundJ[2]<<"],\"replay\":"<<(replay?"true":"false")<<",\"first_failing_constraint\":"<<st.firstFailingConstraint<<",\"queue_flags\":"<<st.queueFlags<<",\"rows\":[";
  for(unsigned i=0;i<std::min<size_t>(st.activeContacts,a.contactEvidence.contacts.size());++i){if(i)std::cout<<",";auto& c=a.contactEvidence.contacts[i];auto& r=a.contactEvidence.evaluatedRows.at(3*i);auto& md=a.contactEvidence.contactMetadata.at(i);auto& cone=a.contactEvidence.evaluatedCones.at(i);
   std::cout<<"{\"flags\":"<<c.flags<<",\"island\":"<<c.islandIndex<<",\"body_a\":"<<c.bodyA<<",\"body_b\":"<<c.bodyB<<",\"normal\":["<<c.normal.x<<","<<c.normal.y<<","<<c.normal.z<<"],\"anchor_a\":["<<md.localAnchorA.x<<","<<md.localAnchorA.y<<","<<md.localAnchorA.z<<"],\"anchor_b\":["<<md.localAnchorB.x<<","<<md.localAnchorB.y<<","<<md.localAnchorB.z<<"],\"cone\":["<<cone.effectiveFrictionU<<","<<cone.effectiveFrictionV<<"],\"row_residual\":["<<c.friction.x<<","<<c.friction.y<<","<<c.friction.z<<"],\"target\":"<<r.targetVelocity<<",\"pre\":"<<r.preSolveVelocity<<",\"impulse\":["<<c.impulses.x<<","<<c.impulses.y<<","<<c.impulses.z<<"]}";
  }
  std::cout<<"],\"final_v\":[";
  for(size_t i=0;i<a.finalV.size();i++){if(i)std::cout<<",";std::cout<<a.finalV[i];}
  std::cout<<"],\"rod_v\":[";
  for(size_t i=0;i<a.finalRodNodes.size();i++){if(i)std::cout<<",";const auto &v=a.finalRodNodes[i].velocity;std::cout<<"["<<v.x<<","<<v.y<<","<<v.z<<"]";}
  std::cout<<"]}";
 }
 std::cout<<"]}\n";
}
