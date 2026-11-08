#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
template<class T> bool eq(const std::vector<T>&a,const std::vector<T>&b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
int main(int argc,char**argv){
 if(argc<2||argc>4||!std::filesystem::is_regular_file(argv[1]))return 2; bool warm=argc==2||std::string(argv[2])!="cold"; unsigned subsequentSweeps=argc==4?std::stoul(argv[3]):2; if(!subsequentSweeps)return 2;
 std::cout<<std::setprecision(10)<<"{\"cases\":[";
 for(int k=0;k<5;++k){
  auto m=metalrobo::makeFreeSphereEngineModel();m.world.gravityAndTimestep={0,k>=3?-9.81f:0.f,0,.01f};
  for(auto& x:m.materials){x.friction.x=.5f;x.friction.y=.5f;x.response.x=0;x.response.y=0;}
  auto& sh=m.shapes[1];sh.shapeType=MR_SHAPE_BOX;sh.dimensions={.02f,.01f,.015f,0};sh.contactRestAndBoundingRadius={.001f,0,.027f,0};
  m.defaultQ[1]=.01f+(k==0?-.0001f:k==2?.0015f:0.f);m.defaultV[0]=k==3?0.f:k==4?.2f:1.f;m.defaultV[1]=k<2?-1.f:0.f;
  metalrobo::CompiledWorld w;auto c=metalrobo::compileMetalWorld(m,0,w);if(!c.succeeded())return 3;
  std::vector<float> effort(w.nv(),0);MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  std::vector<MRBodyStateGPU> scene{ground};
  const metalrobo::MetalWorldBatch initial{.environmentCount=1,.controlStepCount=1,.initialQ=m.defaultQ,.initialV=m.defaultV,.efforts=effort,.initialSceneBodies=scene};
  const metalrobo::MetalWorldBatch next{.environmentCount=1,.controlStepCount=1,.efforts=effort};
  metalrobo::MetalWorldStepConfig cfg;cfg.timestepSeconds=.01f;cfg.physicsSubsteps=1;cfg.velocityIterations=2;cfg.finalVelocityIterations=0;cfg.ccdMode=metalrobo::MetalWorldCCDMode::disabled;cfg.applyBodyDamping=false;cfg.captureContactEvidence=true;cfg.warmStart=warm;
  metalrobo::MetalWorldConfig o;o.metallibPath=argv[1];metalrobo::MetalWorldContext a(o),b(o);metalrobo::MetalWorldResidentState sa,sb;metalrobo::MetalWorldSubmission ta,tb;metalrobo::MetalWorldResult ra,rb;
  std::vector<float> prevQ=m.defaultQ,prevV=m.defaultV;metalrobo::MetalWorldResult prevResult;
  if(k)std::cout<<",";std::cout<<"{\"near_miss\":"<<(k==2?"true":"false")<<",\"initial_gap\":"<<(k==0?-.0001:k==2?.0015:0)<<",\"gravity\":"<<(k>=3?-9.81:0)<<",\"steps\":[";
  for(int i=0;i<100;++i){cfg.velocityIterations=i?subsequentSweeps:2;auto da=i?a.submitResident(w,next,cfg,sa,ta):a.initializeResidentState(w,initial,cfg,sa,ta);auto db=i?b.submitResident(w,next,cfg,sb,tb):b.initializeResidentState(w,initial,cfg,sb,tb);if(!da.succeeded()||!db.succeeded())return 4;
   auto xa=ta.wait(ra),xb=tb.wait(rb);if(ra.finalV.size()<6||ra.finalQ.size()<7||ra.contactStatuses.empty())return 5;
   bool replay=eq(ra.finalQ,rb.finalQ)&&eq(ra.finalV,rb.finalV)&&eq(ra.finalSceneBodies,rb.finalSceneBodies)&&eq(ra.contactEvidence.manifoldCounts,rb.contactEvidence.manifoldCounts)&&eq(ra.contactEvidence.manifoldHeaders,rb.contactEvidence.manifoldHeaders)&&eq(ra.contactEvidence.manifoldPoints,rb.contactEvidence.manifoldPoints)&&xa.firstGPUStatusCode==xb.firstGPUStatusCode;
   bool restored=i>0&&!xa.succeeded()&&eq(ra.finalQ,prevResult.finalQ)&&eq(ra.finalV,prevResult.finalV)&&eq(ra.finalSceneBodies,prevResult.finalSceneBodies)&&eq(ra.contactEvidence.manifoldCounts,prevResult.contactEvidence.manifoldCounts)&&eq(ra.contactEvidence.manifoldHeaders,prevResult.contactEvidence.manifoldHeaders)&&eq(ra.contactEvidence.manifoldPoints,prevResult.contactEvidence.manifoldPoints);
   auto st=ra.contactStatuses[0];double j=0;for(std::size_t z=0;z<std::min<std::size_t>(st.activeContacts,ra.contactEvidence.contacts.size());++z)j+=ra.contactEvidence.contacts[z].impulses.x;
   double J[3]={},L[3]={};for(std::size_t z=0;z<std::min<std::size_t>(st.activeContacts,ra.contactEvidence.contacts.size());++z){
 const auto& p=ra.contactEvidence.contacts[z];const auto& meta=ra.contactEvidence.contactMetadata.at(z);auto n=p.normal,t=p.tangent;double b[3]={n.y*t.z-n.z*t.y,n.z*t.x-n.x*t.z,n.x*t.y-n.y*t.x};
 double jv[3]={n.x*p.impulses.x+t.x*p.impulses.y+b[0]*p.impulses.z,n.y*p.impulses.x+t.y*p.impulses.y+b[1]*p.impulses.z,n.z*p.impulses.x+t.z*p.impulses.y+b[2]*p.impulses.z};
 if(p.bodyA!=1&&p.bodyB!=1)return 6;if(p.bodyA==1)for(double& x:jv)x=-x;const auto& a=p.bodyB==1?meta.localAnchorB:meta.localAnchorA;
 double u[3]={prevQ[3],prevQ[4],prevQ[5]},qw=prevQ[6],r0[3]={a.x,a.y,a.z},cross[3]={u[1]*r0[2]-u[2]*r0[1],u[2]*r0[0]-u[0]*r0[2],u[0]*r0[1]-u[1]*r0[0]};double dot=u[0]*r0[0]+u[1]*r0[1]+u[2]*r0[2],uu=u[0]*u[0]+u[1]*u[1]+u[2]*u[2],r[3];for(int t0=0;t0<3;++t0){r[t0]=(qw*qw-uu)*r0[t0]+2*u[t0]*dot+2*qw*cross[t0];J[t0]+=jv[t0];}L[0]+=r[1]*jv[2]-r[2]*jv[1];L[1]+=r[2]*jv[0]-r[0]*jv[2];L[2]+=r[0]*jv[1]-r[1]*jv[0];}
 double energy=0,oldEnergy=0,contactWork=0,linearError=0,angularError=0;for(int z=0;z<3;++z){energy+=.5*double(ra.finalV[z])*ra.finalV[z]+.05*double(ra.finalV[z+3])*ra.finalV[z+3];oldEnergy+=.5*double(prevV[z])*prevV[z]+.05*double(prevV[z+3])*prevV[z+3];contactWork+=.5*J[z]*(ra.finalV[z]+prevV[z]+(z==1&&k>=3?-.0981:0.0))+.5*L[z]*(ra.finalV[z+3]+prevV[z+3]);linearError=std::max(linearError,std::abs(double(ra.finalV[z]-prevV[z])-J[z]-(z==1&&k>=3?-.0981:0.0)));angularError=std::max(angularError,std::abs(.1*double(ra.finalV[z+3]-prevV[z+3])-L[z]));}
   if(i)std::cout<<",";std::cout<<"{\"accepted\":"<<(xa.succeeded()?"true":"false")<<",\"status\":"<<xa.firstGPUStatusCode<<",\"contacts\":"<<st.activeContacts<<",\"residual\":"<<st.residuals.y<<",\"normal_impulse\":"<<j<<",\"energy\":"<<energy<<",\"kinetic_change\":"<<(energy-oldEnergy)<<",\"contact_work\":"<<contactWork<<",\"linear_error\":"<<linearError<<",\"angular_error\":"<<angularError<<",\"restored\":"<<(restored?"true":"false")<<",\"replay\":"<<(replay?"true":"false")<<",\"q\":[";
   for(std::size_t z=0;z<ra.finalQ.size();++z){if(z)std::cout<<",";std::cout<<ra.finalQ[z];}std::cout<<"],\"v\":[";for(std::size_t z=0;z<ra.finalV.size();++z){if(z)std::cout<<",";std::cout<<ra.finalV[z];}std::cout<<"],\"rows\":[";for(std::size_t z=0;z<std::min<std::size_t>(st.activeContacts,ra.contactEvidence.contacts.size());++z){if(z)std::cout<<",";const auto& p=ra.contactEvidence.contacts[z];const auto& row=ra.contactEvidence.evaluatedRows.at(3*z);std::cout<<"{\"impulse\":"<<p.impulses.x<<",\"target\":"<<row.targetVelocity<<",\"pre\":"<<row.preSolveVelocity<<",\"gap\":"<<p.pointAndSeparation.w<<"}";}std::cout<<"]}";prevQ=ra.finalQ;prevV=ra.finalV;prevResult=ra;
  }std::cout<<"]}";
 }std::cout<<"]}\n";
}
