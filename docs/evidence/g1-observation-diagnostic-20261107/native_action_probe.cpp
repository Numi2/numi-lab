// Owner-level fault isolation: externally supplied commands and native actor
// observations. No policy, source contract migration, or shader substitution.
#include "metalrobo/c_api.h"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
float requested(int step, int env, int action) {
    return 0.04f*step + 0.003f*env + 0.0001f*action;
}
int main(int argc, char** argv) {
    MRTaskRolloutHandle* h=nullptr;
    try {
        check(argc==2,"usage: native_action_probe METALLIB");
        MRRunManifestC manifest{};
        manifest.source=MR_RUN_SOURCE_UNITREE_G1;
        manifest.surface=MR_LOCOMOTION_SURFACE_GROUND;
        manifest.task=MR_UNITREE_G1_TASK_G1_LEGS_LOCOMOTION;
        manifest.metallib_path=argv[1];
        manifest.profile.environment_count=8;
        manifest.profile.physics_substeps=4;
        manifest.profile.velocity_iterations=4;
        manifest.profile.final_velocity_iterations=2;
        manifest.profile.control_timestep_seconds=0.02f;
        manifest.profile.seed=4081012u;
        h=mr_create_task_rollout(&manifest);
        if (!h) throw std::runtime_error(mr_last_error());
        const auto layout=mr_task_rollout_layout(h);
        check(layout.environment_count==8 && layout.action_count==29 && layout.actor_observation_count==240,"layout mismatch");
        check(layout.world_fingerprint==3978934575658225183ull && layout.task_fingerprint==922021427268694894ull && layout.observation_fingerprint==15900562835394482329ull && layout.action_fingerprint==9497262916478774679ull,"exact task contract mismatch");
        std::cout << "step\tenv\taction\treset\texpected_delayed_action\tobserved_delayed_action\tdifference\tphysics_status\tdone\n" << std::setprecision(9);
        int mismatch=0, comparisons=0;
        for (int step=1; step<=8; ++step) {
            std::vector<float> action(8*29);
            std::vector<std::uint32_t> resets(8,0);
            for (int e=0;e<8;++e) for(int j=0;j<29;++j) action[e*29+j]=requested(step,e,j);
            if(step==5) resets[1]=1;
            MRTaskRolloutAdvanceC advance{};
            if(mr_task_rollout_advance(h,action.data(),action.size(),resets.data(),resets.size(),1,0,0,&advance)!=0) throw std::runtime_error(mr_last_error());
            const auto* obs=mr_task_rollout_actor_observations(h);
            const auto* transitions=mr_task_rollout_transitions(h);
            const auto* statuses=mr_task_rollout_status_codes(h);
            check(obs && transitions && statuses,"missing native publication");
            check(advance.failed_environment_steps==0,"failed physics step");
            for(int e=0;e<8;++e) {
                check(statuses[e]==0 && transitions[e].physics_error==0 && transitions[e].done==0,"unexpected physical/task termination");
                // Env 1 is excluded after reset from the delay-value oracle:
                // its reset history is an explicit separate contract.
                if(e==1 && step>=5) continue;
                if(step<3) continue;
                for(int j=0;j<12;++j) {
                    const float observed=obs[e*240+192+j];
                    const float expected=requested(step-2,e,j);
                    const float error=observed-expected;
                    std::cout << step << '\t' << e << '\t' << j << '\t' << resets[e] << '\t' << expected << '\t' << observed << '\t' << error << '\t' << statuses[e] << '\t' << transitions[e].done << '\n';
                    ++comparisons;
                    if(!std::isfinite(observed) || std::abs(error)>1.e-6f) ++mismatch;
                }
            }
        }
        std::cerr << "comparisons=" << comparisons << " mismatches=" << mismatch << '\n';
        mr_task_rollout_destroy(h);
        return mismatch==0 ? 0 : 1;
    } catch(const std::exception& e) {
        std::cerr << "probe failure: " << e.what() << '\n';
        if(h) mr_task_rollout_destroy(h);
        return 2;
    }
}
