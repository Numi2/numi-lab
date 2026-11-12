#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){auto p=hipRollModel();const auto&m=p.mechanics;std::cout<<std::setprecision(9);for(unsigned i=0;i<m.materials.size();i++){const auto&x=m.materials[i];std::cout<<"material "<<i<<" friction "<<x.friction.x<<" "<<x.friction.y<<" response "<<x.response.x<<" "<<x.response.y<<" "<<x.response.z<<" "<<x.response.w<<" geometry "<<x.geometry.x<<"\n";}for(unsigned i=0;i<m.bodies.size();i++)std::cout<<"mass "<<m.bodyNames[i]<<" "<<m.bodies[i].massAndInverseMass.x<<"\n";for(unsigned i=0;i<m.shapes.size();i++){const auto&x=m.shapes[i];std::cout<<"shape "<<i<<" "<<m.shapeNames[i]<<" mat "<<x.materialIndex<<" body "<<x.bodyIndex<<"\n";}return 0;}
