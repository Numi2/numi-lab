#include "metalrobo/MetalCoupledRod.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace metalrobo;
int main(int argc,char**argv){
 if(argc!=2)return 2;DiscreteRodMaterial m;m.radius=2e-5;m.density=1100;m.youngModulus=5e8;m.poissonRatio=.35;
 auto model=makeStraightSutureRod(3,.12,m);auto state=makeDiscreteElasticRodDefaultState(model);
 state.positions[1][0]=.059;state.positions[2][0]=.118;
 std::array<DiscreteRodAttachment,2> a{};a[0].nodeIndex=0;a[1].nodeIndex=2;a[0].targetPosition=state.positions[0];a[1].targetPosition=state.positions[2];
 MetalCoupledRodConfig c;c.timestep=.002;c.metallibPath=argv[1];
 double h=float(c.timestep),ell=model.restLengths[0],length=.059,k=model.stretchStiffness[0]/ell;
 double curvature=model.nodeMasses[1]/(h*h)+2*k*(length-ell)/length+4*model.bendStiffness[0]/(ell*ell*ell);
 MetalCoupledRodResult out;out.iterations=42;out.state.positions={{{123,456,789}}};auto d=runResearchMetalCoupledRod(model,state,a,out,c);
 bool unchanged=out.iterations==42&&out.state.positions.size()==1&&out.state.positions[0][0]==123;
 bool pass=curvature<0&&!d.published&&!d.succeeded()&&unchanged;
 std::cout<<std::setprecision(17)<<"curvature="<<curvature<<" status="<<unsigned(d.status)<<" published="<<d.published<<" unchanged="<<unchanged<<" pass="<<pass<<" message=["<<d.message<<"]\n";
 return pass?0:1;
}
