#include "metalrobo/DiscreteElasticRod.hpp"
#include <iomanip>
#include <iostream>
using namespace metalrobo;
int main(){
 double radius,young,x0,x1,x2,y;
 while(std::cin>>radius>>young>>x0>>x1>>x2>>y){
  DiscreteRodMaterial m;m.radius=radius;m.youngModulus=young;m.density=1100;m.poissonRatio=.35;
  auto model=makeStraightSutureRod(3,.12,m);auto state=makeDiscreteElasticRodDefaultState(model);
  state.positions[0][0]=x0;state.positions[1][0]=x1;state.positions[2][0]=x2;state.positions[2][1]=y;
  DiscreteElasticRodEnergy e;auto d=evaluateDiscreteElasticRodEnergy(model,state,e);if(!d.succeeded())return 2;
  std::cout<<std::setprecision(17)<<e.total()<<"\n";
 }
}
