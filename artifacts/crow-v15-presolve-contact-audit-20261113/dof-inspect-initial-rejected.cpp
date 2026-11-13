#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iostream>
int main(){auto p=hipRollModel(); auto &m=p.mechanics; for(size_t i=0;i<m.dofs.size();i++){const auto& d=m.dofs[i];std::cout<<i<<" "<<m.dofNames[i]<<" q="<<d.qIndex<<" v="<<d.vIndex<<" limits="<<d.limits[0]<<","<<d.limits[1]<<"
";} }
