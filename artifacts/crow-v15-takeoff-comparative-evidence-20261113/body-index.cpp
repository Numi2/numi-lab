#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){auto m=hipRollModel().mechanics;for(unsigned i=0;i<m.bodyNames.size();++i)std::cout<<i<<" "<<m.bodyNames[i]<<"\n";}
