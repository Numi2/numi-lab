#define CROW_HIP_ROLL_LIBRARY
#include "/Users/n/Developer/numi-crow-rl-20261107/artifacts/crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){auto p=hipRollModel(); auto &m=p.mechanics; if(m.joints.at(4).axis0.y!=1.f || m.joints.at(5).axis0.y!=-1.f)return 2; m.joints.at(5).axis0.y=1.f; std::cout<<std::setprecision(10); for(unsigned i=0;i<m.bodies.size();++i)std::cout<<i<<"\t"<<m.bodyNames[i]<<"\t"<<m.bodies[i].massAndInverseMass.x<<"\t"<<m.bodies[i].parentBody<<"\n";}
