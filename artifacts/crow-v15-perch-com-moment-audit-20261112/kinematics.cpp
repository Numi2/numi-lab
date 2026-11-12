#define CROW_HIP_ROLL_LIBRARY
#include "../crow-v15-hip-roll-strong-drive-prototype-20261111/probe.cpp"
#include <iomanip>
int main(){try{
 auto model=hipRollModel();auto &m=model.mechanics;const unsigned nq=m.defaultQ.size();std::cout<<std::setprecision(14);
 unsigned lane,step;while(std::cin>>lane>>step){std::vector<double> q(nq),v(m.defaultV.size());for(auto& x:q)need(bool(std::cin>>x),"missing q");std::vector<ArticulatedBodyKinematics> k(m.bodies.size());need(computeArticulatedBodyKinematics(m,0,q,v,k).succeeded(),"FK failed");double mass=0,c[3]={};for(size_t i=0;i<k.size();i++){double w=m.bodies[i].massAndInverseMass.x;mass+=w;for(int j=0;j<3;j++)c[j]+=w*k[i].centerOfMassPosition[j];}
 std::cout<<"C "<<lane<<" "<<step<<" "<<mass;for(double x:c)std::cout<<" "<<x/mass;std::cout<<"\n";
 if(lane==5 && (step==4084||step==4259))for(size_t i=0;i<k.size();i++){std::cout<<"B "<<lane<<" "<<step<<" "<<m.bodyNames[i];for(auto x:k[i].centerOfMassPosition)std::cout<<" "<<x;for(auto x:k[i].orientation)std::cout<<" "<<x;std::cout<<"\n";}
 }
 return 0;}catch(const std::exception&e){std::cerr<<"error "<<e.what()<<"\n";return 1;}}
