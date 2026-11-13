#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iostream>
int main(){auto p=hipRollModel(); auto &m=p.mechanics; for(size_t i=8;i<=9;i++){const auto& d=m.dofs[i];std::cout<<i<<" "<<m.dofNames[i]<<" limits="<<d.limits.x<<","<<d.limits.y<<","<<d.limits.z<<","<<d.limits.w<<" drive="<<d.drive.x<<","<<d.drive.y<<","<<d.drive.z<<","<<d.drive.w<<std::endl;}}
