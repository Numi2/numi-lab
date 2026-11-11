#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){auto p=hipRollModel();for(unsigned i=0;i<p.mechanics.joints.size();++i){auto j=p.mechanics.joints[i];std::cout<<p.mechanics.jointNames[i]<<" q="<<j.qOffset<<" v="<<j.vOffset<<"\n";}return 0;}
