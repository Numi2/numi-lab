#define CROW_HIP_ROLL_LIBRARY
#include "hip-roll-ratio.cpp"
#include <iomanip>
int main(){try{
 auto robot=hipRollModel();const auto& m=robot.mechanics;std::cout<<std::setprecision(17);int lane,step;
 while(std::cin>>lane>>step){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(double& x:q)need(bool(std::cin>>x),"missing q");for(double& x:v)need(bool(std::cin>>x),"missing v");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());auto s=computeArticulatedBodyKinematics(m,0,q,v,k);need(s.succeeded(),"kinematics");
  double mass=0,c[3]={0,0,0};
  for(unsigned i=0;i<m.bodyNames.size();++i){double w=m.bodies[i].massAndInverseMass.x;mass+=w;for(int j=0;j<3;++j)c[j]+=w*k[i].centerOfMassPosition[j];std::cout<<"body "<<lane<<" "<<step<<" "<<m.bodyNames[i];for(auto x:k[i].centerOfMassPosition)std::cout<<" "<<x;for(auto x:k[i].orientation)std::cout<<" "<<x;std::cout<<"\n";}
  std::cout<<"com "<<lane<<" "<<step<<" "<<mass;for(double x:c)std::cout<<" "<<x/mass;std::cout<<"\n";
 }
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} }
