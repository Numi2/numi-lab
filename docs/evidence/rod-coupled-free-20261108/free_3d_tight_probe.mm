#include "metalrobo/MetalCoupledRod.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <numbers>
using namespace metalrobo;
static void vec(const std::array<double,3>& a){std::cout<<"["<<a[0]<<","<<a[1]<<","<<a[2]<<"]";}
int main(int argc,char**argv){
 if(argc!=2)return 2; unsigned n,cap; double h; if(!(std::cin>>n>>cap>>h)||n<3||n>9)return 3;
 constexpr double R=2e-5,RHO=1100.,EA=2.3818718058345718,L=.11758755785232791;
 DiscreteRodMaterial m;m.radius=R;m.density=RHO;m.youngModulus=EA/(std::numbers::pi*R*R);m.poissonRatio=.35;
 auto model=makeStraightSutureRod(n,L,m);auto state=makeDiscreteElasticRodDefaultState(model);
 for(auto& a:state.positions)for(auto& x:a)std::cin>>x;
 for(auto& a:state.velocities)for(auto& x:a)std::cin>>x;
 for(auto& x:state.twists)std::cin>>x;for(auto& x:state.twistRates)std::cin>>x;
 if(!std::cin)return 4;
 MetalCoupledRodConfig config;config.timestep=h;config.maximumIterations=cap;config.linearResidualTolerance=.000025;config.angularResidualTolerance=.000025;config.gravity={0,-9.81,0};config.metallibPath=argv[1];
 MetalCoupledRodResult out;auto d=runResearchMetalCoupledRod(model,state,{},out,config);bool ok=d.succeeded()&&d.published;
 std::cout<<std::setprecision(17)<<"{\"status\":"<<unsigned(d.status)<<",\"gpu\":"<<d.firstGPUStatusCode<<",\"published\":"<<(ok?"true":"false");
 if(ok){
  auto vectors=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(const auto& x:values){if(!first)std::cout<<",";first=false;vec(x);}std::cout<<"]";};
  auto scalars=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(double x:values){if(!first)std::cout<<",";first=false;std::cout<<x;}std::cout<<"]";};
  vectors("q",out.state.positions);vectors("v",out.state.velocities);scalars("theta",out.state.twists);scalars("omega",out.state.twistRates);
  std::cout<<",\"iterations\":"<<out.iterations<<",\"shifted\":"<<out.regularizedSteps<<",\"max_shift\":"<<out.maximumInertiaShift<<",\"reactions\":"<<out.reactionCount;
 }
 std::cout<<"}\n";return 0;
}
