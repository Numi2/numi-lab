#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try {
 auto robot=hipRollModel();
 need(robot.mechanics.joints.at(5).axis0.y==-1.f,"unexpected pronation axis");
 robot.mechanics.joints.at(5).axis0.y=1.f;
 auto &m=robot.mechanics;
 std::cout<<std::setprecision(17);
 int lane,step,sub;
 while(std::cin>>lane>>step>>sub) {
  std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());
  for(auto &x:q)need(bool(std::cin>>x),"missing q");
  for(auto &x:v)need(bool(std::cin>>x),"missing v");
  std::vector<ArticulatedBodyKinematics> k(m.bodies.size());
  need(computeArticulatedBodyKinematics(m,0,q,v,k).succeeded(),"FK failed");
  for(unsigned i: {11u,15u,27u,28u,29u,30u,31u,32u,33u,34u}) {
   std::cout<<"body "<<lane<<" "<<step<<" "<<sub<<" "<<i;
   for(auto x:k.at(i).centerOfMassPosition)std::cout<<" "<<x;
   for(auto x:k.at(i).orientation)std::cout<<" "<<x;
   for(auto x:k.at(i).linearVelocity)std::cout<<" "<<x;
   for(auto x:k.at(i).angularVelocity)std::cout<<" "<<x;
   std::cout<<"\n";
  }
 }
 }catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}
