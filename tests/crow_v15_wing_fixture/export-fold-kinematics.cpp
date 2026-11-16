#include "v15_fixture.hpp"
#include "metalrobo/CrowArticulatedWing.hpp"
#include "metalrobo/ArticulatedDynamics.hpp"
#include <fstream>
#include <iomanip>
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("expected pose table");
 auto p=metalrobo::makeCrowArticulatedWingMechanicsFixture(crowHipRollModel());auto&m=p.mechanics;auto&a=m.articulations[0];
 std::ifstream in(argv[1]);if(!in)throw std::runtime_error("cannot read input");std::vector<double>q(a.nq),v(a.nv,0);std::vector<metalrobo::ArticulatedBodyKinematics> bodies(m.bodies.size());
 std::cout<<std::setprecision(10)<<"{\"names\":[";for(size_t i=0;i<m.bodyNames.size();++i){if(i)std::cout<<',';std::cout<<std::quoted(m.bodyNames[i]);}std::cout<<"],\"frames\":[";
 for(int s=0;s<480;++s){for(auto&x:q)if(!(in>>x))throw std::runtime_error("short pose table");auto d=metalrobo::computeArticulatedBodyKinematics(m,0,q,v,bodies);if(!d.succeeded())throw std::runtime_error("body kinematics failed");if(s)std::cout<<',';std::cout<<'[';for(size_t i=0;i<bodies.size();++i){if(i)std::cout<<',';auto&b=bodies[i];std::cout<<'[';for(int j=0;j<3;++j)std::cout<<b.centerOfMassPosition[j]<<',';for(int j=0;j<4;++j){if(j)std::cout<<',';std::cout<<b.orientation[j];}std::cout<<']';}std::cout<<']';}std::cout<<"]}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
