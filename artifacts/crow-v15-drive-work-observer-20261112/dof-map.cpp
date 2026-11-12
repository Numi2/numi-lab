#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
int main(){try{const auto p=hipRollModel();const auto& m=p.mechanics;for(size_t i=0;i<m.dofs.size();++i)std::cout<<i<<" "<<m.dofNames[i]<<" "<<m.dofs[i].qIndex<<" "<<m.dofs[i].flags<<" "<<m.dofs[i].drive.x<<" "<<m.dofs[i].drive.y<<" "<<m.dofs[i].limits.z<<"\n";}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
