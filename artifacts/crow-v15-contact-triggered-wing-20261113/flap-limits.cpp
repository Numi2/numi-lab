#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){auto m=hipRollModel().mechanics;for(int i:{8,9}){auto d=m.dofs[i];std::cout<<m.dofNames[i]<<" "<<d.limits.x<<" "<<d.limits.y<<" "<<d.limits.z<<"\n";}}
