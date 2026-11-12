#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{auto model=hipRollModel();auto&m=model.mechanics;unsigned id;std::cout<<std::setprecision(14);while(std::cin>>id){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(auto&x:q)need(bool(std::cin>>x),"missing q");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());need(computeArticulatedBodyKinematics(m,0,q,v,k).succeeded(),"FK failed");for(unsigned i:{11u,15u,20u,22u,24u,26u}){std::cout<<id<<" "<<m.bodyNames[i];for(auto x:k[i].centerOfMassPosition)std::cout<<" "<<x;std::cout<<"\n";}}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
