// Extract only measured actor inputs from the retained historical rollout.
#include "metalrobo/LearningPacks.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
static void check(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
int main(int argc,char** argv) {
  try {
    check(argc==2,"usage: extract_observations ROLLOUTPACK");
    metalrobo::PolicyRolloutPack pack;
    auto result=metalrobo::readPolicyRolloutPack(argv[1],pack);
    check(result.succeeded(),"pack read failed");
    check(pack.taskFingerprint==922021427268694894ull && pack.policyFingerprint==17940125103817627925ull,"fingerprint mismatch");
    check(pack.environmentCount==8 && pack.controlStepCount==1000 && pack.actorObservationCount==240 && pack.actionCount==29,"unexpected dimensions");
    check(pack.actorObservations.size()==std::size_t(1000)*8*240,"actor observation shape mismatch");
    std::cout << "step\tdelayed_ankle_action\tankle_position_error_rad\tankle_finite_difference_velocity_rad_s\tgravity_x\tgravity_y\tgravity_z\tgyro_x_rad_s\tgyro_y_rad_s\tgyro_z_rad_s\n" << std::setprecision(9);
    for(std::size_t step=0;step<1000;++step) {
      // Actor history is oldest first, with five frames of 48 values each.
      const auto* x=pack.actorObservations.data()+(step*8)*240+192;
      for(int i : {5,21,33,40,41,42,45,46,47}) check(std::isfinite(x[i]),"nonfinite observation");
      std::cout << step+1;
      for(int i : {5,21,33,40,41,42,45,46,47}) std::cout << '\t' << x[i];
      std::cout << '\n';
    }
    return 0;
  } catch(const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}
