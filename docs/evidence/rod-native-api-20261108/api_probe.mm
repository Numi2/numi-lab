#include "metalrobo/MetalCoupledRod.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <cmath>
using namespace metalrobo;
static void vec(const std::array<double,3>& a){std::cout<<"["<<a[0]<<","<<a[1]<<","<<a[2]<<"]";}
int main(int argc,char**argv){
 if(argc!=2)return 2;unsigned n,cap;if(!(std::cin>>n>>cap)||n<3||n>9)return 3;
 constexpr double R=2e-5,RHO=1100.,EA=2.3818718058345718,L=.11758755785232791,H=.001;
 DiscreteRodMaterial m;m.radius=R;m.density=RHO;m.youngModulus=EA/(std::numbers::pi*R*R);m.poissonRatio=.35;
 auto model=makeStraightSutureRod(n,L,m);auto state=makeDiscreteElasticRodDefaultState(model);
 for(auto& a:state.positions)for(auto& x:a)std::cin>>x;
 for(auto& a:state.velocities)for(auto& x:a)std::cin>>x;
 for(auto& x:state.twists)std::cin>>x;for(auto& x:state.twistRates)std::cin>>x;
 std::array<DiscreteRodAttachment,2> attachments{};attachments[0].nodeIndex=0;attachments[1].nodeIndex=n-1;
 for(auto& a:attachments)for(auto& x:a.targetPosition)std::cin>>x;if(!std::cin)return 4;
 for(auto& a:attachments)for(unsigned j=0;j<3;j++)a.targetVelocity[j]=(a.targetPosition[j]-state.positions[a.nodeIndex][j])/double(float(H));
 MetalCoupledRodConfig config;config.timestep=H;config.maximumIterations=cap;config.metallibPath=argv[1];
 MetalCoupledRodResult out;auto d=runResearchMetalCoupledRod(model,state,attachments,out,config);bool ok=d.succeeded()&&d.published;
 std::cout<<std::setprecision(17)<<"{\"status\":"<<unsigned(d.status)<<",\"gpu\":"<<d.firstGPUStatusCode<<",\"published\":"<<(ok?"true":"false")<<",\"states\":"<<(ok?1:0)<<",\"reactions\":"<<(ok?2:0);
 if(ok){
  auto vectors=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(const auto& x:values){if(!first)std::cout<<",";first=false;vec(x);}std::cout<<"]";};
  auto scalars=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(double x:values){if(!first)std::cout<<",";first=false;std::cout<<x;}std::cout<<"]";};
  vectors("q",out.state.positions);vectors("v",out.state.velocities);scalars("theta",out.state.twists);scalars("omega",out.state.twistRates);
  std::array<std::array<double,3>,2> f{out.reactions[0].averageForceOnTarget,out.reactions[1].averageForceOnTarget};vectors("reaction",f);
  std::cout<<",\"iterations\":"<<out.iterations<<",\"internal_linear\":"<<out.maximumLinearResidual<<",\"internal_angular\":"<<out.maximumAngularResidual;
 }
 std::cout<<"}\n";return 0;
}
