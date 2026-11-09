#define main toe_free_admission_main
#include "../crow-v15-toe-prototype-20261109/toe_admission_armature.cpp"
#undef main
#include "metalrobo/MetalWorld.hpp"
int main(){for(unsigned arm=0;arm<4;arm++){try{
 RunManifest m;m.id="toe_resident_trace";m.robot=toeModel();if(arm==3)for(unsigned i=0;i<m.robot.mechanics.shapeNames.size();i++)if(m.robot.mechanics.shapeNames[i].find("_toe")!=std::string::npos)m.robot.mechanics.shapes[i].collisionMask=0;
 m.scene=makeBirdFlowAmericanCrowFlightScenePack();auto& b=m.scene.objects.at(0).mechanics.shapes.at(0);b.shapeType=MR_SHAPE_BOX;b.localPosition={.01f,0,-.02f,1};b.localRotation={0,0,0,1};b.dimensions={.03f,.2f,.02f,0};b.contactRestAndBoundingRadius.z=std::sqrt(.03f*.03f+.2f*.2f+.02f*.02f);
 m.sensors.id="proprioception";m.task=makeAmericanCrowEmbodiedTaskPack(m.sensors.observation,m.reality.reset);m.reality.reset={};m.reality.id="reality";m.teacher.id="no_teacher";m.profile.id="probe";m.profile.capacities={.candidatePairs=512,.rawContacts=2048,.manifolds=512,.constraintBlocks=2048,.constraintRows=6144};m.task.capacities=m.profile.capacities;m.profile.environmentCount=1;m.profile.controlSteps=60;m.profile.physicsSubsteps=4;m.profile.controlTimestepSeconds=.02f;
 CompiledRun run;auto compile=compileRun(m,run);need(compile.succeeded(),compile.message);auto q=run.model().defaultQ,v=run.model().defaultV;auto scene=run.defaultSceneBodies();const unsigned nq=q.size(),nv=v.size();std::vector<float> commands(60*nv,0);
 for(unsigned t=0;t<60;t++)for(const auto& d:run.model().dofs)if(d.vIndex<nv && d.qIndex<nq && (d.flags&MR_DOF_FLAG_DRIVE)){float target=q[d.qIndex];if(t>=10 && t<35 && d.vIndex>=nv-8){bool rear=(d.vIndex-(nv-8))%4==3;if(arm==1||arm==3||(arm==2&&rear))target=.85f;}commands[t*nv+d.vIndex]=target;}
 MetalWorldStepConfig c;c.timestepSeconds=.02f;c.physicsSubsteps=4;c.solverMode=MetalWorldSolverMode::temporalCone;c.actuationMode=MetalWorldActuationMode::implicitPositionDrive;c.velocityIterations=4;c.finalVelocityIterations=2;c.captureContactEvidence=true;
 MetalWorldContext wholeContext;MetalWorldResult whole;MetalWorldBatch all{.environmentCount=1,.controlStepCount=60,.initialQ=q,.initialV=v,.efforts=commands,.initialSceneBodies=scene};auto a=wholeContext.run(run.world(),all,c,whole);need(a.succeeded()&&a.failedStepCount==0,"whole rollout failed: "+a.message);auto stride=whole.layout.dispatch.observationEnvironmentStride;need(whole.observations.size()==60*stride,"whole trace missing");
 MetalWorldContext ctx;MetalWorldResidentState resident;double worst=0;unsigned dual=0;for(unsigned t=0;t<60;t++){
  std::span<const float> control(commands.data()+t*nv,nv);MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.efforts=control};if(t==0){batch.initialQ=q;batch.initialV=v;batch.initialSceneBodies=scene;}
  MetalWorldSubmission submission;auto status=t==0?ctx.initializeResidentState(run.world(),batch,c,resident,submission):ctx.submitResident(run.world(),batch,c,resident,submission);need(status.succeeded(),"resident submit: "+status.message);MetalWorldResult out;status=submission.wait(out);need(status.succeeded() && status.failedStepCount==0 && status.successfulStepCount==1,"resident step: "+status.message);
  for(unsigned j=0;j<nq;j++)worst=std::max(worst,std::abs(double(whole.observations[t*stride+j]-out.finalQ[j])));for(unsigned j=0;j<nv;j++)worst=std::max(worst,std::abs(double(whole.observations[t*stride+nq+j]-out.finalV[j])));
  double pos=0,neg=0,up=0,total=0;auto cs=out.contactStatuses.at(0);for(unsigned i=0;i<cs.requiredConstraints;i++){const auto&r=out.contactEvidence.contacts.at(i);const auto&an=run.model().bodyNames.at(r.bodyA),&bn=run.model().bodyNames.at(r.bodyB);bool f=an.find("_toe")!=std::string::npos&&bn=="locomotion_ground", rev=bn.find("_toe")!=std::string::npos&&an=="locomotion_ground";if(!(f||rev)||r.impulses.x<=0)continue;const double nx=(f?1:-1)*r.normal.x,nz=(f?1:-1)*r.normal.z,imp=r.impulses.x;total+=imp;if(nx>.5)pos+=imp;if(nx<-.5)neg+=imp;if(nz<-.5)up+=imp;}
  if(pos>0.00005&&neg>0.00005)dual++;
  std::cout<<"trace "<<arm<<" "<<t<<" "<<out.finalQ[0]<<" "<<out.finalQ[2]<<" "<<out.finalV[0]<<" "<<out.finalV[2]<<" "<<pos<<" "<<neg<<" "<<up<<" "<<total;for(unsigned j=nq-8;j<nq;j++)std::cout<<" "<<out.finalQ[j];std::cout<<std::endl;
 }
 std::cout<<"arm_result "<<arm<<" completed 60 max_qv_difference "<<worst<<" dual_boundaries "<<dual<<std::endl;
 }catch(const std::exception&e){std::cout<<"arm_result "<<arm<<" failed "<<e.what()<<std::endl;}}
}
