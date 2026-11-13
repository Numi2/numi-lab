#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){auto m=hipRollModel().mechanics;for(int i:{8,9})std::cout<<m.dofNames[i]<<" q "<<m.dofs[i].qIndex<<" v "<<m.dofs[i].vIndex<<"\n";}
