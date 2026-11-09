#define main toe_free_admission_main
#include "toe_admission_armature.cpp"
#undef main
#include "metalrobo/MetalWorld.hpp"
int main(){ try {
 for (bool enabled : {false,true}) {
  RunManifest m; m.id="toe_ground_collision_admission";m.robot=toeModel();
  if(!enabled) for(unsigned i=0;i<m.robot.mechanics.shapeNames.size();i++) if(m.robot.mechanics.shapeNames[i].find("_toe")!=std::string::npos)m.robot.mechanics.shapes[i].collisionMask=0;
  m.scene=makeBirdFlowAmericanCrowFlightScenePack();m.sensors.id="toe_probe_proprioception";
  m.task=makeAmericanCrowEmbodiedTaskPack(m.sensors.observation,m.reality.reset);m.reality.reset={};m.reality.id="toe_probe_reality";m.teacher.id="no_teacher";m.profile.id="toe_probe_profile";
  m.profile.capacities={.candidatePairs=512,.rawContacts=2048,.manifolds=512,.constraintBlocks=2048,.constraintRows=6144};
  m.task.capacities=m.profile.capacities;
  m.profile.environmentCount=1;m.profile.controlSteps=50;m.profile.physicsSubsteps=4;m.profile.controlTimestepSeconds=.02f;
  CompiledRun run;auto compile=compileRun(m,run);need(compile.succeeded(),"compile run failed: "+compile.message);
  auto q=run.model().defaultQ;auto v=run.model().defaultV;std::vector<float> effort(50*v.size(),0.f);for(unsigned t=0;t<50;t++)for(const auto& dof:run.model().dofs) if(dof.vIndex<v.size() && dof.qIndex<q.size() && (dof.flags & MR_DOF_FLAG_DRIVE)) effort[t*v.size()+dof.vIndex]=q[dof.qIndex];auto scene=run.defaultSceneBodies();
  MetalWorldBatch batch{.environmentCount=1,.controlStepCount=50,.initialQ=q,.initialV=v,.efforts=effort,.initialSceneBodies=scene};
  MetalWorldStepConfig c;c.timestepSeconds=.02f;c.physicsSubsteps=4;c.solverMode=MetalWorldSolverMode::temporalCone;c.actuationMode=MetalWorldActuationMode::implicitPositionDrive;c.captureContactEvidence=true;c.velocityIterations=4;c.finalVelocityIterations=2;
  MetalWorldContext context;MetalWorldResult out;auto d=context.run(run.world(),batch,c,out);need(d.succeeded() && d.failedStepCount==0 && d.successfulStepCount==50,"native step failed: "+d.message+" gpu="+std::to_string(d.firstGPUStatusCode));
  auto status=out.contactStatuses.at(0);unsigned toes=0,positive=0;double impulse=0;
  std::cout<<"arm="<<(enabled?"enabled":"disabled")<<" contacts="<<status.requiredConstraints<<"\n";
  for(unsigned i=0;i<status.requiredConstraints;i++){const auto& row=out.contactEvidence.contacts.at(i);std::string an=run.model().bodyNames.at(row.bodyA),bn=run.model().bodyNames.at(row.bodyB);bool toe=an.find("_toe")!=std::string::npos||bn.find("_toe")!=std::string::npos;if(toe){toes++;if(row.impulses.x>0)positive++;impulse+=row.impulses.x;}std::cout<<an<<" "<<bn<<" impulse="<<row.impulses.x<<" separation="<<row.pointAndSeparation.w<<"\n";}
  std::cout<<"toe_contacts="<<toes<<" positive="<<positive<<" impulse="<<impulse<<" final_root_z="<<out.finalQ.at(2)<<"\n";
  
 }
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;} }
