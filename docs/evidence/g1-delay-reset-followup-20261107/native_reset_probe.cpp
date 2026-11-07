// Research-only customized owner fixture; no policy is rebound or evaluated.
#include "/Users/numi/numi-g1-delay-fix-owner-20261107/src/c_api.cpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
static float action(int step,int env,int j) {return 0.04f*step+0.003f*env+0.0001f*j;}
int main(int argc,char** argv) {
 try {
  require(argc==2,"usage: native_depth_probe METALLIB");
  MRTaskRolloutConfigC config{}; config.environment_count=8; config.physics_substeps=4;
  config.velocity_iterations=4; config.final_velocity_iterations=2;
  config.control_timestep_seconds=0.02f; config.seed=4081014u;
  int comparisons=0, mismatches=0;
  std::cout << "delay\tstep\tenv\taction\treset\texpected\tobserved\tdifference\tstatus\tdone\n" << std::setprecision(9);
  for(int delay=0;delay<=3;++delay) {
   auto manifest=makeUnitreeG1RunManifest(metalrobo::LocomotionSurface::ground,metalrobo::UnitreeG1Task::g1LegsLocomotion,config);
   manifest.reality.reset.maximumActionDelaySteps=2;
   auto& ops=manifest.sensors.observation.actorFrame;
   require(ops.size()==48,"actor frame width mismatch");
   for(int j=0;j<12;++j) {
    require(ops[j].source==metalrobo::TaskObservationSource::delayedAction,"unexpected operator");
    ops[j].component=delay;
   }
   auto h=createCompiledRunTaskRollout(std::move(manifest),argv[1],"delay-depth probe",nullptr);
   auto layout=mr_task_rollout_layout(h.get());
   require(layout.environment_count==8 && layout.action_count==29 && layout.actor_observation_count==240,"layout mismatch");
   std::cerr << "delay=" << delay << " world=" << layout.world_fingerprint << " task=" << layout.task_fingerprint << " observation=" << layout.observation_fingerprint << " action=" << layout.action_fingerprint << '\n';
   for(int step=1;step<=10;++step) {
    std::vector<float> actions(8*29);
    std::vector<std::uint32_t> resets(8,0);
    for(int e=0;e<8;++e) for(int j=0;j<29;++j) actions[e*29+j]=action(step,e,j);
    if(step==5) resets[1]=1;
    MRTaskRolloutAdvanceC advance{};
    if(mr_task_rollout_advance(h.get(),actions.data(),actions.size(),resets.data(),resets.size(),1,0,0,&advance)!=0) throw std::runtime_error(mr_last_error());
    auto* obs=mr_task_rollout_actor_observations(h.get());
    auto* transitions=mr_task_rollout_transitions(h.get());
    auto* statuses=mr_task_rollout_status_codes(h.get());
    require(obs && transitions && statuses && advance.failed_environment_steps==0,"missing or failed publication");
    for(int e=0;e<8;++e) {
     require(statuses[e]==0 && transitions[e].physics_error==0 && transitions[e].done==0,"failed physics or terminated");
     for(int j=0;j<12;++j) {
      const int source=step-1-delay;
      const float expected=source<1 || (e==1 && step>=5 && source<5) ? 0.f : action(source,e,j);
      const float observed=obs[e*240+192+j], difference=observed-expected;
      std::cout << delay << '\t' << step << '\t' << e << '\t' << j << '\t' << resets[e] << '\t' << expected << '\t' << observed << '\t' << difference << '\t' << statuses[e] << '\t' << transitions[e].done << '\n';
      ++comparisons; if(!std::isfinite(observed) || std::abs(difference)>1e-6f) ++mismatches;
     }
    }
   }
  }
  std::cerr << "comparisons=" << comparisons << " mismatches=" << mismatches << '\n';
  return mismatches ? 1 : 0;
 } catch(const std::exception& e) {std::cerr << "probe failure: " << e.what() << '\n'; return 2;}
}
