#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{
 auto m=hipRollModel().mechanics;std::cout<<std::setprecision(17);int lane,index;
 while(std::cin>>lane>>index){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(double& x:q)need(bool(std::cin>>x),"missing q");for(double& x:v)need(bool(std::cin>>x),"missing v");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());need(computeArticulatedBodyKinematics(m,0,q,v,k).succeeded(),"kinematics");
  std::cout<<lane<<" "<<index;for(unsigned i:{0u,5u,6u,7u})for(double x:k[i].centerOfMassPosition)std::cout<<" "<<x;std::cout<<"\n";
 }
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;} }
