#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){auto m=hipRollModel().mechanics;std::cout<<std::setprecision(12)<<"model_gravity "<<m.world.gravityAndTimestep.x<<" "<<m.world.gravityAndTimestep.y<<" "<<m.world.gravityAndTimestep.z<<" model_timestep "<<m.world.gravityAndTimestep.w<<std::endl;}
