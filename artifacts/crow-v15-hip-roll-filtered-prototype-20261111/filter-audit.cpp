#define CROW_HIP_ROLL_LIBRARY
#include "probe.cpp"
int main(){auto p=hipRollModel();auto &m=p.mechanics;for(const auto &x:m.collisionExclusions)std::cout<<x.colliderA<<" "<<m.shapeNames[x.colliderA]<<" | "<<x.colliderB<<" "<<m.shapeNames[x.colliderB]<<"\n";}
