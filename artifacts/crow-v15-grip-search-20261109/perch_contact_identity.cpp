#define main toe_free_admission_main
#include "../crow-v15-toe-prototype-20261109/toe_admission_armature.cpp"
#undef main
#include "metalrobo/MetalWorld.hpp"
int main(){for(unsigned arm=0;arm<4;arm++){try{
 RunManifest m;m.id="toe_feedback_probe";m.robot=toeModel();if(arm==3)for(unsigned i=0;i<m.robot.mechanics.shapeNames.size();i++)if(m.robot.mechanics.shapeNames[i].find("_toe")!=std::string::npos)m.robot.mechanics.shapes[i].collisionMask=0;
 m.scene=makeBirdFlowAmericanCrowFlightScenePack();auto& b=m.scene.objects.at(0).mechanics.shapes.at(0);b.shapeType=MR_SHAPE_BOX;b.localPosition={0.f,0,-.02f,1};b.localRotation={0,0,0,1};b.dimensions={.025f,.2f,.02f,0};b.contactRestAndBoundingRadius.z=std::sqrt(.025f*.025f+.2f*.2f+.02f*.02f);
 m.sensors.id="proprioception";m.task=makeAmericanCrowEmbodiedTaskPack(m.sensors.observation,m.reality.reset);m.reality.reset={};m.reality.id="reality";m.teacher.id="no_teacher";m.profile.id="probe";m.profile.capacities={.candidatePairs=512,.rawContacts=2048,.manifolds=512,.constraintBlocks=2048,.constraintRows=6144};m.task.capacities=m.profile.capacities;m.profile.environmentCount=1;m.profile.controlSteps=150;m.profile.physicsSubsteps=4;m.profile.controlTimestepSeconds=.02f;
 CompiledRun run;auto compiled=compileRun(m,run);need(compiled.succeeded(),compiled.message);auto q=run.model().defaultQ,v=run.model().defaultV;auto scene=run.defaultSceneBodies();unsigned nq=q.size(),nv=v.size();
 MetalWorldStepConfig c;c.timestepSeconds=.02f;c.physicsSubsteps=4;c.solverMode=MetalWorldSolverMode::temporalCone;c.actuationMode=MetalWorldActuationMode::implicitPositionDrive;c.velocityIterations=4;c.finalVelocityIterations=2;c.captureContactEvidence=true;
 MetalWorldContext ctx;MetalWorldResidentState resident;unsigned phase=0,stable=0,dual=0,hold=0,transition=999,success=999,release=999;float front=0,rear=0;double lastPos=0,lastNeg=0;
 for(unsigned t=0;t<150;t++){
  if(t==10)phase=1;if(t>=120&&phase!=4){phase=4;release=t;}if(phase==1)rear=.85f;if(phase==2){rear=.85f;front=std::clamp(front+(lastPos>.00005?.025f:-.015f),0.f,.85f);}if(phase==3){rear=.85f;if(++hold>=10){phase=4;release=t;}}if(phase==4){front=0;rear=0;}
  std::vector<float> commands(nv,0);for(const auto& d:run.model().dofs)if(d.vIndex<nv && d.qIndex<nq && (d.flags&MR_DOF_FLAG_DRIVE)){float target=q[d.qIndex];if(d.vIndex>=nv-8){bool isRear=(d.vIndex-(nv-8))%4==3;target=isRear?rear:front;}commands[d.vIndex]=target;}
  MetalWorldBatch batch{.environmentCount=1,.controlStepCount=1,.efforts=commands};if(t==0){batch.initialQ=q;batch.initialV=v;batch.initialSceneBodies=scene;}MetalWorldSubmission sub;auto status=t==0?ctx.initializeResidentState(run.world(),batch,c,resident,sub):ctx.submitResident(run.world(),batch,c,resident,sub);need(status.succeeded(),"submit: "+status.message);MetalWorldResult out;status=sub.wait(out);need(status.succeeded()&&status.failedStepCount==0&&status.successfulStepCount==1,"step "+std::to_string(t)+": "+status.message);
  double pos=0,neg=0,up=0;auto cs=out.contactStatuses.at(0);for(unsigned i=0;i<cs.requiredConstraints;i++){const auto&r=out.contactEvidence.contacts.at(i);const auto&an=run.model().bodyNames.at(r.bodyA),&bn=run.model().bodyNames.at(r.bodyB);bool f=an.find("_toe")!=std::string::npos&&bn=="locomotion_ground",rev=bn.find("_toe")!=std::string::npos&&an=="locomotion_ground";if(!(f||rev)||r.impulses.x<=0)continue;double nx=(f?1:-1)*r.normal.x,nz=(f?1:-1)*r.normal.z,imp=r.impulses.x;std::cout<<"contact "<<arm<<" "<<t<<" "<<(f?an:bn)<<" "<<nx<<" "<<nz<<" "<<imp<<std::endl;if(nx>.5)pos+=imp;if(nx<-.5)neg+=imp;if(nz<-.5)up+=imp;}
  std::cout<<"trace "<<arm<<" "<<t<<" "<<phase<<" "<<front<<" "<<rear<<" "<<pos<<" "<<neg<<" "<<up<<" "<<out.finalQ[0]<<" "<<out.finalQ[2]<<" "<<out.finalV[0];for(unsigned j=nq-8;j<nq;j++)std::cout<<" "<<out.finalQ[j];std::cout<<std::endl;
  lastPos=(arm==2)?0:pos;lastNeg=(arm==2)?0:neg;
  if(phase==1 && arm!=1){stable=lastPos>.00005?stable+1:0;if(stable>=2){phase=2;transition=t;}}
  if(phase==2){dual=(lastPos>.00005&&lastNeg>.00005)?dual+1:0;if(dual>=3){phase=3;success=t;}}
 }
 std::cout<<"arm_result "<<arm<<" completed transition "<<transition<<" success "<<success<<" release "<<release<<std::endl;
 }catch(const std::exception&e){std::cout<<"arm_result "<<arm<<" failed "<<e.what()<<std::endl;}}
}
