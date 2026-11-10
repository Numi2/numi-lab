#define main toe_free_admission_main
#include "../crow-v15-impact-grip-probe-20261110/toe_model.cpp"
#undef main
#include "wing_model.hpp"
#include <fstream>
int main(int argc,char**argv){auto robot=toeModel();crowResearchWingInertia(robot);auto& m=robot.mechanics;std::ifstream f(argv[1]);unsigned t;while(f>>t){std::vector<double>q(m.defaultQ.size()),v(m.defaultV.size());for(auto&x:q)f>>x;for(auto&x:v)f>>x;std::vector<ArticulatedBodyKinematics> k(m.bodies.size());auto status=computeArticulatedBodyKinematics(m,0,q,v,k);need(status.succeeded(),"kinematics failed");std::cout<<"sample "<<t<<" "<<q[0]<<" "<<q[2]<<" "<<v[0]<<" "<<v[2];for(unsigned i=0;i<m.bodyNames.size();i++)if(m.bodyNames[i]=="crow_left_foot"||m.bodyNames[i]=="crow_right_foot"||m.bodyNames[i]=="crow_left_rear_toe"||m.bodyNames[i]=="crow_right_rear_toe"){auto p=k[i].centerOfMassPosition;std::cout<<" "<<m.bodyNames[i]<<" "<<p[0]<<" "<<p[1]<<" "<<p[2];}std::cout<<std::endl;}}
