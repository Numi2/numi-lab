#include "metalrobo/MetalCoupledRod.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
using namespace metalrobo;
int main(int argc,char**argv){
 if(argc!=2)return 2;
 auto model=makeStraightSutureRod(3,.12);auto state=makeDiscreteElasticRodDefaultState(model);
 std::array<DiscreteRodAttachment,2> a{};a[0].nodeIndex=0;a[1].nodeIndex=2;
 a[0].targetPosition=state.positions[0];a[1].targetPosition=state.positions[2];
 MetalCoupledRodConfig c;c.metallibPath=argv[1];
 auto run=[&](const char* name,const auto& m,const auto& s,const auto& at,const auto& cfg,bool reject){
  MetalCoupledRodResult out;out.state.positions={{{123,456,789}}};out.iterations=42;out.reactions[0].averageForceOnTarget[0]=999;
  auto d=runResearchMetalCoupledRod(m,s,at,out,cfg);bool untouched=out.state.positions.size()==1 && out.state.positions[0][0]==123 && out.iterations==42 && out.reactions[0].averageForceOnTarget[0]==999;
  std::cout<<name<<" status="<<unsigned(d.status)<<" published="<<d.published<<" untouched="<<untouched;
  if(d.published) std::cout<<" twist="<<out.state.twists[0]<<" rate="<<out.state.twistRates[0];
  std::cout<<"\n";
  return reject ? (!d.succeeded()&&!d.published&&untouched) : (d.succeeded()&&d.published);
 };
 bool ok=run("ordinary",model,state,a,c,false);
 auto badmodel=model;badmodel.stretchStiffness[1]*=1.1;ok &=run("nonuniform",badmodel,state,a,c,true);
 auto badstate=state;badstate.positions[1]=badstate.positions[0];ok &=run("degenerate",model,badstate,a,c,true);
 badstate=state;badstate.velocities[1][0]=std::numeric_limits<double>::infinity();ok &=run("nonfinite",model,badstate,a,c,true);
 auto badattachment=a;badattachment[0].compliance=1e-9;ok &=run("compliant",model,state,badattachment,c,true);
 auto badconfig=c;badconfig.metallibPath+=".missing";ok &=run("library",model,state,a,badconfig,true);
 badstate=state;for(auto& x:badstate.twists)x=1e12;for(auto& x:badstate.twistRates)x=.001;
 ok &=run("large_twist",model,badstate,a,c,true);
 return ok?0:1;
}
