#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){auto p=hipRollModel();std::cout<<std::setprecision(9);for(unsigned i=13;i<=18;i++){const auto& d=p.mechanics.dofs.at(i);std::cout<<i<<" "<<p.mechanics.dofNames.at(i)<<" "<<d.limits.x<<" "<<d.limits.y<<" "<<d.limits.z<<" "<<d.limits.w<<"\n";}return 0;}
