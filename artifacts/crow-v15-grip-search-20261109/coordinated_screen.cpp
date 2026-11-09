#define main toe_free_admission_main
#include "../crow-v15-toe-prototype-20261109/toe_admission_armature.cpp"
#undef main
#include "metalrobo/MetalWorld.hpp"
int main() {
 for (auto bounds : {std::pair<float,float>{-.020f,.040f}})
 for (float target : {.85f})
 for (unsigned arm=0;arm<8;arm++) { try {
  RunManifest m;m.id="toe_grip_geometry_screen";m.robot=toeModel();
  if(arm==2)for(unsigned i=0;i<m.robot.mechanics.shapeNames.size();i++)if(m.robot.mechanics.shapeNames[i].find("_toe")!=std::string::npos)m.robot.mechanics.shapes[i].collisionMask=0;
  m.scene=makeBirdFlowAmericanCrowFlightScenePack();auto &perch=m.scene.objects.at(0).mechanics.shapes.at(0);perch.shapeType=MR_SHAPE_BOX;perch.localPosition={(bounds.first+bounds.second)/2,0,-.02f,1};perch.localRotation={0,0,0,1};perch.dimensions={(bounds.second-bounds.first)/2,.2f,.02f,0};perch.contactRestAndBoundingRadius.z=std::sqrt(perch.dimensions.x*perch.dimensions.x+.2f*.2f+.02f*.02f);
  m.sensors.id="proprioception";m.task=makeAmericanCrowEmbodiedTaskPack(m.sensors.observation,m.reality.reset);m.reality.reset={};m.reality.id="reality";m.teacher.id="no_teacher";m.profile.id="toe_probe";
  m.profile.capacities={.candidatePairs=512,.rawContacts=2048,.manifolds=512,.constraintBlocks=2048,.constraintRows=6144};m.task.capacities=m.profile.capacities;m.profile.environmentCount=1;m.profile.controlSteps=50;m.profile.physicsSubsteps=4;m.profile.controlTimestepSeconds=.02f;
  CompiledRun run;auto cc=compileRun(m,run);need(cc.succeeded(),cc.message);auto q=run.model().defaultQ,v=run.model().defaultV;const unsigned nq=q.size(),nv=v.size();std::vector<float> effort(50*nv,0);
  for(unsigned t=0;t<50;t++)for(const auto& d:run.model().dofs)if(d.vIndex<nv && d.qIndex<nq && (d.flags&MR_DOF_FLAG_DRIVE)) {
   float desired=q[d.qIndex];if(t>=10 && d.vIndex>=nv-8){bool rear=((d.vIndex-(nv-8))%4)==3;switch(arm){case 1:case 2:desired=target;break;case 3:desired=rear?0:target;break;case 4:desired=rear?target:0;break;case 5:desired=(rear||t>=20)?target:0;break;case 6:desired=(!rear||t>=20)?target:0;break;case 7:desired=rear?target:.35f;break;}}effort[t*nv+d.vIndex]=desired;}
  auto scene=run.defaultSceneBodies();MetalWorldBatch batch{.environmentCount=1,.controlStepCount=50,.initialQ=q,.initialV=v,.efforts=effort,.initialSceneBodies=scene};MetalWorldStepConfig c;c.timestepSeconds=.02f;c.physicsSubsteps=4;c.solverMode=MetalWorldSolverMode::temporalCone;c.actuationMode=MetalWorldActuationMode::implicitPositionDrive;c.velocityIterations=4;c.finalVelocityIterations=2;c.captureContactEvidence=true;
  MetalWorldContext ctx;MetalWorldResult out;auto d=ctx.run(run.world(),batch,c,out);need(d.succeeded() && d.failedStepCount==0 && d.successfulStepCount==50,"physical sequence failed: "+d.message);
  const unsigned stride=out.layout.dispatch.observationEnvironmentStride;need(out.observations.size()==50*stride,"trajectory missing");double maxdiff=0;for(unsigned j=0;j<nq;j++)maxdiff=std::max(maxdiff,std::abs(double(out.observations[49*stride+j]-out.finalQ[j])));need(maxdiff<1e-6,"trajectory not aligned");
  double pos=0,neg=0,up=0,total=0;unsigned rows=0;auto cs=out.contactStatuses.at(0);
  for(unsigned i=0;i<cs.requiredConstraints;i++){const auto&r=out.contactEvidence.contacts.at(i);const auto&a=run.model().bodyNames.at(r.bodyA),&b=run.model().bodyNames.at(r.bodyB);bool forward=a.find("_toe")!=std::string::npos && b=="locomotion_ground",reverse=b.find("_toe")!=std::string::npos && a=="locomotion_ground";if(!(forward||reverse)||r.impulses.x<=0)continue;double nx=(forward?1:-1)*r.normal.x,nz=(forward?1:-1)*r.normal.z,imp=r.impulses.x;rows++;total+=imp;if(nx>.5)pos+=imp;if(nx<-.5)neg+=imp;if(nz<-.5)up+=imp;std::cout<<"contact "<<bounds.first<<" "<<bounds.second<<" "<<target<<" "<<arm<<" "<<a<<" "<<b<<" "<<nx<<" "<<nz<<" "<<imp<<" "<<r.pointAndSeparation.w<<"\n";}
  std::cout<<"result "<<bounds.first<<" "<<bounds.second<<" "<<target<<" "<<arm<<" accepted "<<rows<<" "<<pos<<" "<<neg<<" "<<up<<" "<<total<<" "<<out.finalQ[0]<<" "<<out.finalQ[1]<<" "<<out.finalQ[2]<<" "<<out.finalV[0]<<" "<<out.finalV[1]<<" "<<out.finalV[2];for(unsigned j=nq-8;j<nq;j++)std::cout<<" "<<out.finalQ[j];std::cout<<std::endl;
 }catch(const std::exception&e){std::cout<<"result "<<bounds.first<<" "<<bounds.second<<" "<<target<<" "<<arm<<" failure "<<e.what()<<std::endl;} }
}
