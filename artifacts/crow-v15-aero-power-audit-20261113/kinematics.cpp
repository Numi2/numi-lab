#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{auto robot=hipRollModel();const auto&m=robot.mechanics;std::cout<<std::setprecision(17);int idx;while(std::cin>>idx){std::vector<double>q(m.defaultQ.size()),v(m.defaultV.size());for(auto&x:q)need(bool(std::cin>>x),"missing q");for(auto&x:v)need(bool(std::cin>>x),"missing v");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());need(computeArticulatedBodyKinematics(m,0,q,v,k).succeeded(),"FK failed");std::cout<<idx;for(unsigned i:{0,5,6,7}){for(double x:k[i].linearVelocity)std::cout<<" "<<x;for(double x:k[i].angularVelocity)std::cout<<" "<<x;}std::cout<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
