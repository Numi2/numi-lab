#define main toe_free_admission_main
#include "toe_admission_armature.cpp"
#undef main
#include "metalrobo/MetalWorld.hpp"
int main(){try{
 for(bool flex:{false,true}){
  RunManifest m;m.id="toe_flex_probe";m.robot=toeModel();m.scene=makeBirdFlowAmericanCrowFlightScenePack();m.sensors.id="proprioception";m.task=makeAmericanCrowEmbodiedTaskPack(m.sensors.observation,m.reality.reset);m.reality.reset={};m.reality.id="reality";m.teacher.id="no_teacher";m.profile.id="toe_probe";
  m.profile.capacities={.candidatePairs=512,.rawContacts=2048,.manifolds=512,.constraintBlocks=2048,.constraintRows=6144};m.task.capacities=m.profile.capacities;m.profile.environmentCount=1;m.profile.controlSteps=50;m.profile.physicsSubsteps=4;m.profile.controlTimestepSeconds=.02f;
  CompiledRun run;auto cc=compileRun(m,run);need(cc.succeeded(),cc.message);auto q=run.model().defaultQ,v=run.model().defaultV;const unsigned nq=q.size(),nv=v.size();std::vector<float> effort(50*nv,0);
  for(unsigned t=0;t<50;t++)for(const auto& d:run.model().dofs)if(d.vIndex<nv && d.qIndex<nq && (d.flags&MR_DOF_FLAG_DRIVE))effort[t*nv+d.vIndex]=(flex && t>=10 && t<30 && d.vIndex>=nv-8)?0.35f:q[d.qIndex];
  auto scene=run.defaultSceneBodies();MetalWorldBatch batch{.environmentCount=1,.controlStepCount=50,.initialQ=q,.initialV=v,.efforts=effort,.initialSceneBodies=scene};MetalWorldStepConfig c;c.timestepSeconds=.02f;c.physicsSubsteps=4;c.solverMode=MetalWorldSolverMode::temporalCone;c.actuationMode=MetalWorldActuationMode::implicitPositionDrive;c.velocityIterations=4;c.finalVelocityIterations=2;c.captureContactEvidence=true;
  MetalWorldContext ctx;MetalWorldResult out;auto d=ctx.run(run.world(),batch,c,out);need(d.succeeded() && d.failedStepCount==0 && d.successfulStepCount==50,"physical sequence failed: "+d.message);
  need(out.observations.size()==50*(nq+nv),"physical trajectory missing");double maxDiff=0;for(unsigned j=0;j<nq;j++)maxDiff=std::max(maxDiff,std::abs(double(out.observations[49*(nq+nv)+j]-out.finalQ[j])));
  std::cout<<"arm="<<(flex?"flex_release":"fixed")<<" final_trajectory_q_max_difference="<<maxDiff<<"\n";
  for(unsigned t=0;t<50;t++){const unsigned base=t*(nq+nv);std::cout<<"state "<<(flex?1:0)<<" "<<t<<" "<<out.observations[base+2];for(unsigned j=0;j<8;j++)std::cout<<" "<<out.observations[base+nq-8+j];for(unsigned j=0;j<8;j++)std::cout<<" "<<out.observations[base+nq+nv-8+j];std::cout<<"\n";}
  auto cs=out.contactStatuses.at(0);double toeGround=0;unsigned positive=0;for(unsigned i=0;i<cs.requiredConstraints;i++){const auto&r=out.contactEvidence.contacts.at(i);const auto&a=run.model().bodyNames.at(r.bodyA),&b=run.model().bodyNames.at(r.bodyB);if(((a.find("_toe")!=std::string::npos && b=="locomotion_ground")||(b.find("_toe")!=std::string::npos && a=="locomotion_ground"))&&r.impulses.x>0){positive++;toeGround+=r.impulses.x;}}std::cout<<"final_toe_ground_positive="<<positive<<" impulse="<<toeGround<<"\n";
 }
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
