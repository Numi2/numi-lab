// Bounded native composition of actual task, tracker, rendering and physics.
// Compares an intended mixed-overflow arm with an all-clear control.
#include "/Users/numi/numi-g1-tracker-camera-owner-20261107/src/c_api_retained_research.cpp"
#include <array>
#include <bit>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
static std::unique_ptr<MRTaskRolloutHandle> make(const char* metallib,const char* assetDir,bool failing) {
 std::array<MRTaskRolloutDynamicSphereC,6> balls{};
 for(std::size_t i=0;i<balls.size();++i) {balls[i].position[0]=4.f+float(i); balls[i].position[2]=2.f; balls[i].radius=.1f; balls[i].mass=1.f; balls[i].launch_step=0;}
 MRTaskRolloutConfigC cfg{};cfg.environment_count=2;cfg.physics_substeps=4;cfg.velocity_iterations=4;cfg.final_velocity_iterations=2;cfg.control_timestep_seconds=.02f;cfg.seed=4081015u;cfg.dynamic_spheres=balls.data();cfg.dynamic_sphere_count=6;
 balls[5].position[0]=.08f; balls[5].position[1]=0.f; balls[5].position[2]=2.f;
 auto manifest=makeUnitreeG1RunManifest(metalrobo::LocomotionSurface::ground,metalrobo::UnitreeG1Task::ballDisturbanceRecovery,cfg);
 manifest.task.capacities.rawContacts=8;
 // Preserve deterministic authored positions for the overflow intervention.
 std::erase_if(manifest.reality.reset.operators,[](const auto& op) {return op.operation==metalrobo::TaskRandomizationOperator::sceneBodyPosition || op.operation==metalrobo::TaskRandomizationOperator::sceneBodyVelocity || op.operation==metalrobo::TaskRandomizationOperator::sceneBodyLaunchStep || op.operation==metalrobo::TaskRandomizationOperator::sceneBodyEventImpact;});
 std::array<std::string,6> names,paths;
 std::array<MRTaskVisualPackC,6> packs{};
 for(std::size_t i=0;i<6;++i) {
  names[i]="locomotion_dynamic_sphere_"+std::to_string(i);
  paths[i]=(std::filesystem::path(assetDir)/(names[i]+".mrvpack")).string();
  packs[i]={paths[i].c_str(),names[i].c_str(),2u,static_cast<std::uint32_t>(100u+i)};
 }
 MRTaskVisualObservationConfigC visual{};visual.packs=packs.data();visual.pack_count=6;
 visual.camera_parent_body="torso_link";visual.camera_position[0]=.08f;visual.camera_position[2]=.45f;
 visual.camera_orientation[3]=1.f;
 visual.width=16;visual.height=9;visual.minimum_visible_pixels=1;visual.vertical_field_of_view_degrees=54;visual.nominal_rate_hz=10;
 auto h=createCompiledRunTaskRollout(std::move(manifest),metallib,"visual rejection fixture",&visual);
 require(h->environmentCount==2 && h->visualRuntime!=nullptr,"visual owner unavailable");
 h->stepConfig.publishStateTrajectory=true; h->stepConfig.publishFinalState=true;
 // Both robots start high enough to avoid ordinary floor contact. The failing
 // arm alone places five converging rigid spheres away from the elevated robot.
 const auto layout=mr_task_rollout_layout(h.get());
 h->resetQ[2]+=1.f;h->resetQ[layout.nq+2]+=1.f;
 const auto n=h->defaultSceneBodies.size();
 require(n>=7,"ground plus six rigid spheres required");
 if(failing) {
  for(std::size_t i=1;i<=5;++i) {
   const float angle=float(i-1)*2.f*3.14159265358979323846f/5.f;
   const float x=std::cos(angle), y=std::sin(angle);
   h->resetSceneBodies[i].position.x=4.f+.38f*x;
   h->resetSceneBodies[i].position.y=.38f*y;
   h->resetSceneBodies[i].position.z=2.f;
   h->resetSceneBodies[i].linearVelocityAndInverseMass.x=-5.f*x;
   h->resetSceneBodies[i].linearVelocityAndInverseMass.y=-5.f*y;
  }
 }
 metalrobo::PolicyPack policy;
 policy.id="visual_rejection_zero_policy";
 const auto& policyLayout=h->taskProgram.layout();
 policy.layers.push_back({.inputCount=policyLayout.actorObservationSize,
  .outputCount=policyLayout.actionCount,
  .activation=metalrobo::PolicyActivation::identity,
  .weights=std::vector<float>(policyLayout.actorObservationSize*policyLayout.actionCount,0.f),
  .bias=std::vector<float>(policyLayout.actionCount,0.f)});
 metalrobo::bindPolicyPack(policy,h->taskProgram);
 installPolicyPack(*h,policy);
 return h;
}
int main(int argc,char** argv) {
 try {
  require(argc==4,"usage: tracker_probe METALLIB ASSET_DIRECTORY SNAPSHOT_TSV");
  const bool instrumented=std::string(argv[3])!="-";
  std::ofstream snapshotFile;
  if(instrumented) snapshotFile.open(argv[3]);
  require(!instrumented || bool(snapshotFile),"snapshot file creation failed");
  if(instrumented) snapshotFile << "arm\tphase\tarray\tindex\tbits\n";
  auto capture=[&](MRTaskRolloutHandle* h,const char* arm,int phase) {
   if(!instrumented) return;
   metalrobo::TrackerProbeSnapshot x;
   require(h->visualRuntime->tracker.researchSnapshot(x),"snapshot failed");
   std::cerr << "snapshot arm="<<arm<<" phase="<<phase
    <<" sequence="<<x.observationSequence<<" sensor="<<x.lastSensorSequence
    <<" envs="<<x.environments<<" tracks="<<x.tracks
    <<" track_history="<<x.trackHistoryLength<<" depth_width="<<x.depthWidth
    <<" depth_height="<<x.depthHeight<<" depth_slots="<<x.depthSlots<<'\n';
   for(std::size_t i=0;i<x.trackHistory.size();++i)
    snapshotFile<<arm<<'\t'<<phase<<"\ttrack\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.trackHistory[i])<<'\n';
   for(std::size_t i=0;i<x.cameraState.size();++i)
    snapshotFile<<arm<<'\t'<<phase<<"\tcamera\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.cameraState[i])<<'\n';
   for(std::size_t i=0;i<x.depthHistory.size();++i)
    snapshotFile<<arm<<'\t'<<phase<<"\tdepth\t"<<i<<'\t'<<std::bit_cast<std::uint32_t>(x.depthHistory[i])<<'\n';
  };
  auto control=make(argv[1],argv[2],false); auto failure=make(argv[1],argv[2],true);
  const auto c=mr_task_rollout_layout(control.get()), f=mr_task_rollout_layout(failure.get());
  require(c.world_fingerprint==f.world_fingerprint && c.task_fingerprint==f.task_fingerprint && c.observation_fingerprint==f.observation_fingerprint && c.action_fingerprint==f.action_fingerprint,"arm contract mismatch");
  std::cerr << "world="<<f.world_fingerprint<<" task="<<f.task_fingerprint<<" observation="<<f.observation_fingerprint<<" action="<<f.action_fingerprint<<" policy="<<failure->stepConfig.policyProgram.fingerprint()<<" sensor_offset="<<(failure->taskProgram.layout().actorFrameSize*failure->taskProgram.layout().actorHistoryLength+failure->taskProgram.header().counts3.w)<<" actor_count="<<f.actor_observation_count<<'\n';
  std::cout << "arm\tstep\tenv\tstatus\tphysics_error\tdone\tactor_component\tvalue\n" << std::setprecision(9);
  for(int arm=0;arm<2;++arm) {
   auto* h=arm?failure.get():control.get();
   capture(h,arm?"failure":"control",0);
   
   for(int step=1;step<=3;++step) {
    MRTaskRolloutAdvanceC advance{}; int rc=mr_task_rollout_advance(h,nullptr,0,nullptr,0,1,0,0,&advance);
    if(rc!=0) std::cerr << (arm?"failure":"control") << " step=" << step << " advance error: " << mr_last_error() << std::endl;
    for(std::size_t e=0;e<h->result.contactStatuses.size();++e) {
     const auto& contact=h->result.contactStatuses[e];
     std::cerr<<"contact arm="<<(arm?"failure":"control")<<" step="<<step<<" env="<<e<<" code="<<contact.code<<" raw_required="<<contact.requiredRawContacts<<" active="<<contact.activeContacts<<'\n';
    }
    const auto count=h->defaultSceneBodies.size();
    for(std::size_t e=0;e<2;++e) for(std::size_t i=1;i<count;++i) {
     if(e*count+i>=h->result.finalSceneBodies.size()) continue;
     const auto& b=h->result.finalSceneBodies[e*count+i];
     std::cerr<<"body arm="<<(arm?"failure":"control")<<" step="<<step<<" env="<<e<<" index="<<i<<" p="<<b.position.x<<","<<b.position.y<<","<<b.position.z<<" v="<<b.linearVelocityAndInverseMass.x<<","<<b.linearVelocityAndInverseMass.y<<","<<b.linearVelocityAndInverseMass.z<<'\n';
    }
    auto* values=mr_task_rollout_actor_observations(h);auto* status=mr_task_rollout_status_codes(h);auto* trans=mr_task_rollout_transitions(h);
    require(values && status && trans,"missing outputs");
    for(int e=0;e<2;++e) for(std::uint32_t j=0;j<c.actor_observation_count;++j)
     std::cout << (arm?"failure":"control") << '\t' << step << '\t' << e << '\t' << status[e] << '\t' << trans[e].physics_error << '\t' << trans[e].done << '\t' << j << '\t' << values[e*c.actor_observation_count+j] << '\n';
    capture(h,arm?"failure":"control",step);
    if(rc!=0) break;
   }
  }
  return 0;
 } catch(const std::exception& e) {std::cerr<<"fixture failure: "<<e.what()<<'\n';return 2;}
}
