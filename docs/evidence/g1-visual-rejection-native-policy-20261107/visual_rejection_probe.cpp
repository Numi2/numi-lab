// Bounded native composition of actual task, tracker, rendering and physics.
// Compares an intended mixed-overflow arm with an all-clear control.
#include "/Users/numi/numi-g1-delay-fix-owner-20261107/src/c_api.cpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
static std::unique_ptr<MRTaskRolloutHandle> make(const char* metallib,const char* assetDir,bool failing) {
 std::array<MRTaskRolloutDynamicSphereC,6> balls{};
 for(std::size_t i=0;i<balls.size();++i) {balls[i].position[0]=4.f+float(i); balls[i].position[2]=2.f; balls[i].radius=.1f; balls[i].mass=1.f; balls[i].launch_step=1000;}
 MRTaskRolloutConfigC cfg{};cfg.environment_count=2;cfg.physics_substeps=4;cfg.velocity_iterations=4;cfg.final_velocity_iterations=2;cfg.control_timestep_seconds=.02f;cfg.seed=4081015u;cfg.dynamic_spheres=balls.data();cfg.dynamic_sphere_count=6;
 auto manifest=makeUnitreeG1RunManifest(metalrobo::LocomotionSurface::ground,metalrobo::UnitreeG1Task::ballDodge,cfg);
 manifest.task.capacities.rawContacts=1;
 // Preserve deterministic authored positions for the overflow intervention.
 std::erase_if(manifest.reality.reset.operators,[](const auto& op) {return op.operation==metalrobo::TaskRandomizationOperator::sceneBodyPosition || op.operation==metalrobo::TaskRandomizationOperator::sceneBodyVelocity;});
 std::array<std::string,6> names,paths;
 std::array<MRTaskVisualPackC,6> packs{};
 for(std::size_t i=0;i<6;++i) {
  names[i]="locomotion_dynamic_sphere_"+std::to_string(i);
  paths[i]=(std::filesystem::path(assetDir)/(names[i]+".mrvpack")).string();
  packs[i]={paths[i].c_str(),names[i].c_str(),2u,static_cast<std::uint32_t>(100u+i)};
 }
 MRTaskVisualObservationConfigC visual{};visual.packs=packs.data();visual.pack_count=6;
 visual.camera_parent_body="torso_link";visual.camera_position[0]=.08f;visual.camera_position[2]=.45f;
 visual.camera_orientation[1]=std::sin(35.f*3.14159265358979323846f/180.f);visual.camera_orientation[3]=std::cos(35.f*3.14159265358979323846f/180.f);
 visual.width=16;visual.height=9;visual.minimum_visible_pixels=1;visual.vertical_field_of_view_degrees=54;visual.nominal_rate_hz=10;
 auto h=createCompiledRunTaskRollout(std::move(manifest),metallib,"visual rejection fixture",&visual);
 require(h->environmentCount==2 && h->visualRuntime!=nullptr,"visual owner unavailable");
 // Both robots start high enough to avoid ordinary floor contact. The failing
 // arm alone places two rigid spheres at the elevated pelvis.
 const auto layout=mr_task_rollout_layout(h.get());
 h->resetQ[2]+=1.f;h->resetQ[layout.nq+2]+=1.f;
 const auto n=h->defaultSceneBodies.size();
 require(n>=7,"ground plus six rigid spheres required");
 if(failing) {
  for(std::size_t i=1;i<=2;++i) {
   h->resetSceneBodies[i].position.x=0.f;
   h->resetSceneBodies[i].position.y=i==1?-.05f:.05f;
   h->resetSceneBodies[i].position.z=1.8f;
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
  require(argc==3,"usage: visual_rejection_probe METALLIB ASSET_DIRECTORY");
  auto control=make(argv[1],argv[2],false); auto failure=make(argv[1],argv[2],true);
  const auto c=mr_task_rollout_layout(control.get()), f=mr_task_rollout_layout(failure.get());
  require(c.world_fingerprint==f.world_fingerprint && c.task_fingerprint==f.task_fingerprint && c.observation_fingerprint==f.observation_fingerprint && c.action_fingerprint==f.action_fingerprint,"arm contract mismatch");
  std::cerr << "world="<<f.world_fingerprint<<" task="<<f.task_fingerprint<<" observation="<<f.observation_fingerprint<<" action="<<f.action_fingerprint<<" policy="<<failure->stepConfig.policyProgram.fingerprint()<<" sensor_offset="<<(failure->taskProgram.layout().actorFrameSize*failure->taskProgram.layout().actorHistoryLength+failure->taskProgram.header().counts3.w)<<" actor_count="<<f.actor_observation_count<<'\n';
  std::cout << "arm\tstep\tenv\tstatus\tphysics_error\tdone\tactor_component\tvalue\n" << std::setprecision(9);
  for(int arm=0;arm<2;++arm) {
   auto* h=arm?failure.get():control.get();
   
   for(int step=1;step<=3;++step) {
    MRTaskRolloutAdvanceC advance{}; int rc=mr_task_rollout_advance(h,nullptr,0,nullptr,0,1,0,0,&advance);
    if(rc!=0) throw std::runtime_error(std::string("advance error: ")+mr_last_error());
    auto* values=mr_task_rollout_actor_observations(h);auto* status=mr_task_rollout_status_codes(h);auto* trans=mr_task_rollout_transitions(h);
    require(values && status && trans,"missing outputs");
    for(int e=0;e<2;++e) for(std::uint32_t j=0;j<c.actor_observation_count;++j)
     std::cout << (arm?"failure":"control") << '\t' << step << '\t' << e << '\t' << status[e] << '\t' << trans[e].physics_error << '\t' << trans[e].done << '\t' << j << '\t' << values[e*c.actor_observation_count+j] << '\n';
   }
  }
  return 0;
 } catch(const std::exception& e) {std::cerr<<"fixture failure: "<<e.what()<<'\n';return 2;}
}
