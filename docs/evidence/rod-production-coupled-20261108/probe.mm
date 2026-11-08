#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>
using namespace metalrobo;
static void vec(const std::array<double,3>& v) { std::cout << "[" << v[0] << "," << v[1] << "," << v[2] << "]"; }
int main(int argc,char**argv) {
 if(argc!=2) return 2;
 unsigned n, it; if(!(std::cin>>n>>it) || n<3 || n>128) return 3;
 DiscreteRodMaterial m; m.radius=2e-5; m.density=1100.; m.youngModulus=2.3818718058345718/(std::numbers::pi*m.radius*m.radius); m.poissonRatio=.35;
 auto model=makeStraightSutureRod(n,.11758755785232791,m); auto s=makeDiscreteElasticRodDefaultState(model);
 for(auto& x:s.positions) for(auto& v:x) std::cin>>v;
 for(auto& x:s.velocities) for(auto& v:x) std::cin>>v;
 for(auto& x:s.twists) std::cin>>x;
 for(auto& x:s.twistRates) std::cin>>x;
 std::array<DiscreteRodAttachment,2> a{}; a[0].nodeIndex=0; a[1].nodeIndex=n-1;
 for(auto& x:a) for(auto& v:x.targetPosition) std::cin>>v;
 if(!std::cin) return 4;
 std::array<DiscreteElasticRodState,1> ss{s}; MetalDiscreteElasticRodInput input; input.states=ss; input.attachmentCount=2; input.attachments=a;
 MetalDiscreteElasticRodConfig cfg; cfg.metallibPath=argv[1]; cfg.step.timestep=.001; cfg.step.gravity={0,0,0}; cfg.step.linearDamping=0; cfg.step.twistDamping=0; cfg.step.solverIterations=it; cfg.step.constraintTolerance=1e-7;
 MetalDiscreteElasticRodResult out; auto d=runMetalDiscreteElasticRod(model,input,out,cfg);
 std::cout<<std::setprecision(17)<<"{\"status\":"<<unsigned(d.status)<<",\"gpu\":"<<d.firstGPUStatusCode<<",\"published\":"<<(d.published?"true":"false")<<",\"states\":"<<out.states.size()<<",\"reactions\":"<<out.reactions.size();
 if(d.published && out.states.size()==1 && out.reactions.size()==2) {
  auto& t=out.states[0];
  auto vectors=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(const auto& v:values){if(!first)std::cout<<",";first=false;vec(v);}std::cout<<"]";};
  auto scalars=[&](const char* name,const auto& values){std::cout<<",\""<<name<<"\":[";bool first=true;for(double v:values){if(!first)std::cout<<",";first=false;std::cout<<v;}std::cout<<"]";};
  vectors("q",t.positions);vectors("v",t.velocities);scalars("theta",t.twists);scalars("omega",t.twistRates);
  std::array<std::array<double,3>,2> f{out.reactions[0].averageForceOnTarget,out.reactions[1].averageForceOnTarget};vectors("reaction",f);
 }
 std::cout<<"}\n";return 0;
}
