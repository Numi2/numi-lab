#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){auto model=hipRollModel();for(unsigned i=0;i<model.mechanics.dofs.size();i++)std::cout<<model.mechanics.dofNames[i]<<" q="<<model.mechanics.dofs[i].qIndex<<" v="<<model.mechanics.dofs[i].vIndex<<"\n";}
