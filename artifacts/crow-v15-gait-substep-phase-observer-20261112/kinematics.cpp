#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{
 auto robot=hipRollModel();const auto& m=robot.mechanics;std::cout<<std::setprecision(17);int lane,step;
 while(std::cin>>lane>>step){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(double& x:q)need(bool(std::cin>>x),"missing q");for(double& x:v)need(bool(std::cin>>x),"missing v");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,q,v,k);need(s.succeeded(),"kinematics");
  double mass=0,c[3]={0,0,0};
  for(unsigned i=0;i<m.bodyNames.size();++i){if(m.bodyNames[i]!="crow_left_thigh" && m.bodyNames[i]!="crow_right_thigh" && m.bodyNames[i]!="crow_left_shank" && m.bodyNames[i]!="crow_right_shank" && m.bodyNames[i]!="crow_left_foot" && m.bodyNames[i]!="crow_right_foot" && m.bodyNames[i]!="crow_left_front_toe_2_proximal" && m.bodyNames[i]!="crow_right_front_toe_2_proximal")continue;std::cout<<"body "<<lane<<" "<<step<<" "<<m.bodyNames[i];for(auto x:k[i].centerOfMassPosition)std::cout<<" "<<x;for(auto x:k[i].orientation)std::cout<<" "<<x;std::cout<<"\n";}

 }
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} }
