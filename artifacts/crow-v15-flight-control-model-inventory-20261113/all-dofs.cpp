#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){auto r=hipRollModel();auto&m=r.mechanics;std::cout<<std::setprecision(12);for(unsigned i=0;i<m.dofs.size();++i){const auto&d=m.dofs[i];if(i<100)std::cout<<i<<" "<<m.dofNames[i]<<" "<<d.qIndex<<" "<<d.flags<<" "<<d.limits.x<<" "<<d.limits.y<<" "<<d.limits.z<<" "<<d.limits.w<<"\n";}return 0;}
