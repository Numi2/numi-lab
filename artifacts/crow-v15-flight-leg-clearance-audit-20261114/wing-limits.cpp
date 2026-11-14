#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){auto m=hipRollModel().mechanics;for(auto d:m.dofs)if(d.qIndex>=9 && d.qIndex<=13)std::cout<<m.dofNames[d.vIndex]<<" "<<d.qIndex<<" "<<d.limits.x<<" "<<d.limits.y<<" "<<d.limits.z<<" "<<d.limits.w<<"\n";}
