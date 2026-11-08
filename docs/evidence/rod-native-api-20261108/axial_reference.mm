#include "metalrobo/MetalCoupledRod.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
using namespace metalrobo;
int main(int argc,char**argv){
 if(argc!=2)return 2;int fail=0;std::cout<<std::setprecision(17);
 for(double radius:{1e-5,2e-5,5e-5}) for(double young:{1e7,1e8,5e8}) for(double h:{.0005,.001,.002}) {
  DiscreteRodMaterial mat;mat.radius=radius;mat.density=1100;mat.youngModulus=young;mat.poissonRatio=.35;
  auto m=makeStraightSutureRod(3,.12,mat);auto s=makeDiscreteElasticRodDefaultState(m);
  s.positions[1][0]+=.0001;s.velocities[1][0]=.003;
  std::array<DiscreteRodAttachment,2> a{};a[0].nodeIndex=0;a[1].nodeIndex=2;
  a[0].targetPosition=s.positions[0];a[1].targetPosition=s.positions[2];
  a[0].targetPosition[0]+=1e-5;a[1].targetPosition[0]-=2e-5;
  const double metalH=float(h);
  a[0].targetVelocity[0]=(a[0].targetPosition[0]-s.positions[0][0])/metalH;
  a[1].targetVelocity[0]=(a[1].targetPosition[0]-s.positions[2][0])/metalH;
  MetalCoupledRodConfig cfg;cfg.timestep=h;cfg.metallibPath=argv[1];
  MetalCoupledRodResult out;auto d=runResearchMetalCoupledRod(m,s,a,out,cfg);
  bool pass=d.succeeded()&&d.published;double ep=0,ev=0,er=0,ef=0,fgate=0;
  if(pass){
   const double k=m.stretchStiffness[0]/m.restLengths[0],M=m.nodeMasses[1]/(metalH*metalH);
   const double x=(M*(s.positions[1][0]+metalH*s.velocities[1][0])+k*(a[0].targetPosition[0]+a[1].targetPosition[0]))/(M+2*k);
   const double v=(x-s.positions[1][0])/metalH;
   const double f0=-(m.nodeMasses[0]/(metalH*metalH)*(a[0].targetPosition[0]-s.positions[0][0]-metalH*s.velocities[0][0])-k*(x-a[0].targetPosition[0]-m.restLengths[0]));
   const double f1=-(m.nodeMasses[2]/(metalH*metalH)*(a[1].targetPosition[0]-s.positions[2][0]-metalH*s.velocities[2][0])+k*(a[1].targetPosition[0]-x-m.restLengths[1]));
   ep=std::abs(out.state.positions[1][0]-x);ev=std::abs(out.state.velocities[1][0]-v);
   er=std::abs(out.state.velocities[1][0]-s.velocities[1][0]+metalH*k*(2*out.state.positions[1][0]-a[0].targetPosition[0]-a[1].targetPosition[0])/m.nodeMasses[1]);
   ef=std::max(std::abs(out.reactions[0].averageForceOnTarget[0]-f0),std::abs(out.reactions[1].averageForceOnTarget[0]-f1));
   fgate=std::max(1e-6,.01*std::max(std::abs(f0),std::abs(f1)));
   pass=ep<=1e-6&&ev<=.001&&er<=.001&&std::abs(out.reactions[0].averageForceOnTarget[0]-f0)<=std::max(1e-6,.01*std::abs(f0))&&std::abs(out.reactions[1].averageForceOnTarget[0]-f1)<=std::max(1e-6,.01*std::abs(f1));
  }
  std::cout<<"radius="<<radius<<" young="<<young<<" h="<<h<<" accepted="<<d.published<<" pass="<<pass<<" position="<<ep<<" velocity="<<ev<<" momentum="<<er<<" force="<<ef<<" force_gate_max="<<fgate<<"\n";
  if(!pass)fail++;
 }
 return fail?1:0;
}
