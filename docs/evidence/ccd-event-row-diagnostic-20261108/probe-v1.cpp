#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/EngineModel.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
int main(int argc,char** argv) {
 if(argc<2||!std::filesystem::is_regular_file(argv[1]))return 2;
 std::cout<<std::setprecision(12)<<"{\"schema\":\"numi.ccd.complete-steps.v1\",\"runs\":[";bool first=true;
 for(double scale:{1.,.5})for(int limit:{16})for(int mode:{0})for(int fixture:{1,2,12,13,14,15}) {
  auto model=metalrobo::makeFreeSphereEngineModel();
  model.world.gravityAndTimestep={0,0,0,.01f};
  auto& shape=model.shapes[1];shape.shapeType=MR_SHAPE_BOX;shape.flags|=MR_SHAPE_FLAG_ENABLE_CCD;
  const float hx=.2*scale,hy=(fixture>=12?.0002:.002)*scale,hz=.03*scale;
  shape.dimensions={hx,hy,hz,0};shape.contactRestAndBoundingRadius={float(.001*scale),0,std::sqrt(hx*hx+hy*hy+hz*hz),0};
  auto& body=model.bodies[1];const float ix=(hy*hy+hz*hz)/3.f,iy=(hx*hx+hz*hz)/3.f,iz=(hx*hx+hy*hy)/3.f;
  body.inertiaRow0={ix,0,0,0};body.inertiaRow1={0,iy,0,0};body.inertiaRow2={0,0,iz,0};
  body.inverseInertiaRow0={1/ix,0,0,0};body.inverseInertiaRow1={0,1/iy,0,0};body.inverseInertiaRow2={0,0,1/iz,0};
  const bool hit=fixture==1||fixture==2||fixture==8||fixture==9||fixture==12||fixture==14;
  const double center=fixture==0||fixture==5?.12:(fixture==3||fixture==4?.23:(hit?.1998:.2003));
  model.defaultQ[1]=center*scale;
  model.defaultV[1]=(fixture==0?-22.f:(fixture==5?-1.f:0.f))*scale;
  model.defaultV[5]=fixture==0||fixture==5?0.f:
      (fixture==8||fixture==10?211.f:(fixture==9||fixture==11?293.f:
      (fixture==2||fixture==4||fixture==7||fixture==14||fixture==15?177.f:250.f)));
  metalrobo::CompiledWorld world;auto compiled=metalrobo::compileMetalWorld(model,0,world);
  if(!compiled.succeeded()){std::cerr<<"world compile failed\n";return 1;}
  std::vector<float> efforts(world.nv(),0.f);
  MRBodyStateGPU ground{};ground.position.w=1;ground.orientation.w=1;ground.flagsAndIndices[0]=MR_MOTION_STATIC;ground.flagsAndIndices[1]=MR_INVALID_INDEX;
  const std::vector<MRBodyStateGPU> scene{ground};
  const metalrobo::MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.initialQ=model.defaultQ,.initialV=model.defaultV,.efforts=efforts,.initialSceneBodies=scene};
  metalrobo::MetalWorldStepConfig config;config.timestepSeconds=.01f;config.physicsSubsteps=1;
  config.velocityIterations=2;config.finalVelocityIterations=1;config.ccdMode=metalrobo::MetalWorldCCDMode::hybrid;
  config.maxConservativeAdvancementIterations=limit;config.directionalCCDAdvancement=mode>0;config.intervalLocalCCDAdvancement=mode==2;
  config.captureContactEvidence=true;config.applyBodyDamping=false;
  metalrobo::MetalWorldConfig owner;owner.metallibPath=argv[1];metalrobo::MetalWorldContext context(owner);
  if(std::getenv("NUMI_CCD_EVENT_TRACE"))std::cerr<<"ccd_case={\"limit\":"<<limit<<",\"mode\":"<<mode<<",\"fixture\":"<<fixture<<",\"scale\":"<<scale<<"}\n";
  metalrobo::MetalWorldResult result;auto start=std::chrono::steady_clock::now();auto d=context.run(world,batch,config,result);
  const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  metalrobo::MetalWorldResult repeated;auto d2=context.run(world,batch,config,repeated);
  const auto bytes=[](const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);};
  const bool replay=bytes(result.finalQ,repeated.finalQ)&&bytes(result.finalV,repeated.finalV)&&d.firstGPUStatusCode==d2.firstGPUStatusCode;
  const bool rollback=d.succeeded()||(bytes(result.finalQ,model.defaultQ)&&bytes(result.finalV,model.defaultV));
  const double impulse=result.finalV.size()>1?result.finalV[1]-model.defaultV[1]:0.;
  const unsigned contacts=result.contactStatuses.empty()?0:result.contactStatuses[0].activeContacts;

  const unsigned unresolved=result.environmentStatuses.empty()?0:result.environmentStatuses[0].unresolvedCCDCount;
  // Both torque-free rotation fixtures pass pi/2 during the step. The analytic
  // minimum plane gap of their unforced sweep is independent of endpoint pose.
  const double freeSweepGap=fixture==0?-.102*scale:(fixture==5?.108*scale:model.defaultQ[1]-std::hypot(hx,hy));
  const bool unresponded=d.succeeded()&&fixture>0&&impulse==0.&&freeSweepGap<0.;
  double gap=0.;if(result.finalQ.size()>=7){const auto& q=result.finalQ;
   const double nx=2*(q[3]*q[4]+q[5]*q[6]),ny=1-2*(q[3]*q[3]+q[5]*q[5]),nz=2*(q[4]*q[5]-q[3]*q[6]);
   gap=q[1]-(hx*std::abs(nx)+hy*std::abs(ny)+hz*std::abs(nz));}
  if(!first)std::cout<<",";first=false;
  std::cout<<"{\"limit\":"<<limit<<",\"mode\":"<<mode<<",\"fixture\":"<<fixture<<",\"scale\":"<<scale<<",\"half_extents\":["<<hx<<","<<hy<<","<<hz<<"],\"initial_q_y\":"<<model.defaultQ[1]<<",\"accepted\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<d.firstGPUStatusCode<<",\"host_status\":"<<int(d.status)<<",\"accepted_steps\":"<<d.successfulStepCount<<",\"gap\":"<<gap<<",\"normal_impulse_Ns\":"<<impulse<<",\"rejected_contact_cache_empty\":"<<((d.succeeded()||std::all_of(result.contactEvidence.manifoldCounts.begin(),result.contactEvidence.manifoldCounts.end(),[](auto n){return n==0;}))?"true":"false")<<",\"unresolved_ccd\":"<<unresolved<<",\"free_sweep_min_gap\":"<<freeSweepGap<<",\"accepted_unresponded_swept_hit\":"<<(unresponded?"true":"false")<<",\"active_contacts\":"<<contacts<<",\"rejected_state_restored\":"<<(rollback?"true":"false")<<",\"repeat_identical\":"<<(replay?"true":"false")<<",\"elapsed_with_initialization\":"<<elapsed<<"}";
 }
 std::cout<<"]}\n";
}
