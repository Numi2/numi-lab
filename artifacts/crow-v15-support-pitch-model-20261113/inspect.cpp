#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){auto r=hipRollModel();auto&m=r.mechanics;std::cout<<std::setprecision(9);for(unsigned i:{13u,14u,15u,16u,17u,18u}){auto&d=m.dofs.at(i);auto&j=m.joints.at(d.jointIndex);std::cout<<m.dofNames.at(i)<<" q "<<d.qIndex<<" joint "<<d.jointIndex<<" parent "<<j.parentBody<<" child "<<j.childBody<<" axis "<<j.axis0.x<<" "<<j.axis0.y<<" "<<j.axis0.z<<" bounds "<<d.limits.x<<" "<<d.limits.y<<" effort "<<d.limits.w<<"\n";}return 0;}
