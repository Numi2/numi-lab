#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){ auto r=hipRollModel(); auto&m=r.mechanics; for(unsigned i=0;i<6;i++){const auto&j=m.joints[i];std::cout<<i<<" "<<m.jointNames[i]<<" axis "<<j.axis0.x<<" "<<j.axis0.y<<" "<<j.axis0.z<<" q "<<j.qOffset<<" default "<<m.defaultQ[j.qOffset]<<" parent "<<j.parentBody<<" child "<<j.childBody<<"\n";} for(unsigned i=0;i<6;i++){const auto&a=r.actuators[i];std::cout<<i<<" "<<a.id<<" target "<<a.target<<" scale "<<a.scale<<"\n";} }
