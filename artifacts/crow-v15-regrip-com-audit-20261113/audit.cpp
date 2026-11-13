#define main toe_probe_main
#include "../crow-v15-two-segment-toe-prototype-20261111/probe.cpp"
#undef main
#include "../crow-v15-toe-release-regrip-20261111/wing_model.hpp"
#include <iomanip>
int main(){try{auto robot=toeModel();crowResearchWingInertia(robot);const auto&m=robot.mechanics;std::cout<<std::setprecision(12);int lane,step;while(std::cin>>lane>>step){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(auto&x:q)need(bool(std::cin>>x),"q missing");for(auto&x:v)need(bool(std::cin>>x),"v missing");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());need(computeArticulatedBodyKinematics(m,0,q,v,k).succeeded(),"FK failure");double mass=0,x=0,z=0,vx=0,vz=0;for(unsigned i=0;i<k.size();i++){double w=m.bodies[i].massAndInverseMass.x;mass+=w;x+=w*k[i].centerOfMassPosition[0];z+=w*k[i].centerOfMassPosition[2];vx+=w*k[i].linearVelocity[0];vz+=w*k[i].linearVelocity[2];}std::cout<<lane<<" "<<step<<" "<<mass<<" "<<x/mass<<" "<<z/mass<<" "<<vx/mass<<" "<<vz/mass<<" "<<q[0]<<" "<<q[2]<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
