#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{auto robot=hipRollModel();const auto& m=robot.mechanics;std::cout<<std::setprecision(17);int lane,step,sub;
 while(std::cin>>lane>>step>>sub){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size(),0.0);for(double& x:q)need(bool(std::cin>>x),"missing q");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,q,v,k);need(s.succeeded(),"kinematics");
  for(unsigned i=0;i<m.bodyNames.size();++i){const auto& n=m.bodyNames[i];if(n!="crow_left_hip_roll_link"&&n!="crow_right_hip_roll_link"&&n!="crow_left_thigh"&&n!="crow_right_thigh"&&n!="crow_left_shank"&&n!="crow_right_shank"&&n!="crow_left_foot"&&n!="crow_right_foot"&&n!="crow_left_front_toe_2_proximal"&&n!="crow_right_front_toe_2_proximal"&&n!="crow_left_front_toe_2_distal"&&n!="crow_right_front_toe_2_distal")continue;std::cout<<"body "<<lane<<" "<<step<<" "<<sub<<" "<<n;for(auto x:k[i].centerOfMassPosition)std::cout<<" "<<x;for(auto x:k[i].orientation)std::cout<<" "<<x;std::cout<<"\n";}
 }
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
