#define main toe_model_admission_main
#include "toe_model.cpp"
#undef main
#include <iomanip>
int main(){try{
 auto robot=toeModel();const auto& m=robot.mechanics;std::cout<<std::setprecision(17);int lane,step;
 while(std::cin>>lane>>step){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(double& x:q)need(bool(std::cin>>x),"missing q");for(double& x:v)need(bool(std::cin>>x),"missing v");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,q,v,k);need(s.succeeded(),"kinematics");
  for(unsigned i=0;i<m.bodyNames.size();++i){const auto& n=m.bodyNames[i];if(n.find("_foot")==std::string::npos&&n.find("_toe")==std::string::npos)continue;std::cout<<lane<<" "<<step<<" "<<n;for(auto x:k[i].centerOfMassPosition)std::cout<<" "<<x;for(auto x:k[i].orientation)std::cout<<" "<<x;for(auto x:k[i].linearVelocity)std::cout<<" "<<x;for(auto x:k[i].angularVelocity)std::cout<<" "<<x;std::cout<<"\n";}
 }
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} }
