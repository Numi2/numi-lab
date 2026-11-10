#define main toe_free_admission_main
#include "toe_model.cpp"
#undef main
#include "wing_model.hpp"
#include <fstream>
int main(int argc,char** argv){auto robot=toeModel();crowResearchWingInertia(robot);auto& m=robot.mechanics;std::ifstream f(argv[1]);unsigned t;std::cout<<std::setprecision(12);while(f>>t){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(auto& x:q)f>>x;for(auto& x:v)f>>x;std::vector<ArticulatedBodyKinematics> k(m.bodies.size());need(computeArticulatedBodyKinematics(m,0,q,v,k).succeeded(),"kinematics");for(unsigned i=0;i<m.bodyNames.size();i++)if(m.bodyNames[i].find("_toe")!=std::string::npos){std::cout<<t<<" "<<m.bodyNames[i];for(auto x:k[i].centerOfMassPosition)std::cout<<" "<<x;for(auto x:k[i].orientation)std::cout<<" "<<x;std::cout<<"\n";}}}
