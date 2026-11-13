#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{auto r=hipRollModel();const auto&m=r.mechanics;std::cout<<std::setprecision(17);int id;while(std::cin>>id){std::vector<double> q(m.defaultQ.size()),v(m.defaultV.size());for(auto&x:q)need(bool(std::cin>>x),"missing q");for(auto&x:v)need(bool(std::cin>>x),"missing v");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());need(computeArticulatedBodyKinematics(m,0,q,v,k).succeeded(),"FK failed");std::cout<<id;for(int b:{5,6}){for(int j=0;j<3;++j)std::cout<<" "<<k[b].centerOfMassPosition[j]-k[0].centerOfMassPosition[j];}std::cout<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
