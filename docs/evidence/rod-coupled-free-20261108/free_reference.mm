#include "metalrobo/MetalCoupledRod.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
using namespace metalrobo;
using A3=std::array<double,3>;
static A3 tridiagonal(A3 diag,A3 upper,A3 lower,A3 b){
 for(int i=1;i<3;i++){double q=lower[i]/diag[i-1];diag[i]-=q*upper[i-1];b[i]-=q*b[i-1];}
 A3 x{};x[2]=b[2]/diag[2];for(int i=1;i>=0;i--)x[i]=(b[i]-upper[i]*x[i+1])/diag[i];return x;
}
int main(int argc,char**argv){
 if(argc!=2)return 2;int failed=0;std::cout<<std::setprecision(17);
 for(double radius:{1e-5,2e-5,5e-5}) for(double young:{1e7,1e8,5e8}) for(double h:{.0005,.001,.002}) {
  DiscreteRodMaterial material;material.radius=radius;material.density=1100;material.youngModulus=young;material.poissonRatio=.35;
  auto model=makeStraightSutureRod(3,.12,material);auto initial=makeDiscreteElasticRodDefaultState(model);
  initial.positions[0][0]-=.00004;initial.positions[1][0]+=.00013;initial.positions[2][0]+=.00002;
  initial.velocities[0][0]=-.014;initial.velocities[1][0]=.007;initial.velocities[2][0]=.021;
  MetalCoupledRodConfig config;config.timestep=h;config.gravity={-9.81,0,0};config.metallibPath=argv[1];
  MetalCoupledRodResult out;auto d=runResearchMetalCoupledRod(model,initial,{},out,config);
  bool pass=d.succeeded()&&d.published&&out.reactionCount==0;double ep=0,ev=0,er=0,total=0,neg=0;
  if(pass){
   double hh=double(float(h)),g=config.gravity[0],ell=model.restLengths[0],k=model.stretchStiffness[0]/ell;
   A3 diag{},up{-k,-k,0},lo{0,-k,-k},rhs{};
   for(int i=0;i<3;i++){double mi=model.nodeMasses[i]/(hh*hh);diag[i]=mi+(i==1?2:1)*k;rhs[i]=mi*(initial.positions[i][0]+hh*initial.velocities[i][0]+hh*hh*g);}
   rhs[0]-=k*ell;rhs[2]+=k*ell;const auto ref=tridiagonal(diag,up,lo,rhs);
   A3 q{},v{},res{};for(int i=0;i<3;i++){q[i]=out.state.positions[i][0];v[i]=out.state.velocities[i][0];ep=std::max(ep,std::abs(q[i]-ref[i]));ev=std::max(ev,std::abs(v[i]-(ref[i]-initial.positions[i][0])/hh));}
   A3 f{k*(q[0]-q[1]+ell),k*(2*q[1]-q[0]-q[2]),k*(q[2]-q[1]-ell)};
   for(int i=0;i<3;i++){res[i]=v[i]-initial.velocities[i][0]-hh*g+hh*f[i]/model.nodeMasses[i];er=std::max(er,std::abs(res[i]));total+=model.nodeMasses[i]*(v[i]-initial.velocities[i][0]-hh*g);}
   neg=std::abs(res[0]+std::copysign(.002,res[0]==0?1:res[0]));
   pass=ep<=1e-6&&ev<=.001&&er<=.001&&std::abs(total)<=1e-12&&neg>.001;
  }
  std::cout<<"radius="<<radius<<" young="<<young<<" h="<<h<<" accepted="<<d.published<<" pass="<<pass<<" position="<<ep<<" velocity="<<ev<<" local="<<er<<" momentum="<<std::abs(total)<<" negative="<<neg<<"\n";if(!pass)failed++;
 }
 auto m=makeStraightSutureRod(3,.12);auto s=makeDiscreteElasticRodDefaultState(m);
 MetalCoupledRodConfig c;c.metallibPath=argv[1];c.gravity={-9.81,0,0};MetalCoupledRodResult out;
 auto d=runResearchMetalCoupledRod(m,s,{},out,c);bool control=d.succeeded()&&out.reactionCount==0;
 if(control){double h=double(float(c.timestep));for(int i=0;i<3;i++){control &= std::abs(out.state.velocities[i][0]-h*c.gravity[0])<1e-6;control &= std::abs(out.state.positions[i][0]-s.positions[i][0]-h*h*c.gravity[0])<1e-9;}for(int i=0;i<2;i++)control &= std::abs((out.state.positions[i+1][0]-out.state.positions[i][0])-m.restLengths[i])<1e-9;}
 std::cout<<"freefall="<<control<<"\n";if(!control)failed++;
 auto reject=[&](const char* name,const auto& state,const auto& a,const auto& config){MetalCoupledRodResult v;v.state.positions={{{123,456,789}}};v.iterations=42;auto z=runResearchMetalCoupledRod(m,state,a,v,config);bool ok=!z.succeeded()&&!z.published&&v.iterations==42&&v.state.positions.size()==1&&v.state.positions[0][0]==123;std::cout<<name<<"="<<ok<<" status="<<unsigned(z.status)<<"\n";if(!ok)failed++;};
 std::array<DiscreteRodAttachment,1> one{};reject("one_attachment",s,one,c);
 auto bad=c;bad.gravity[0]=std::numeric_limits<double>::infinity();reject("invalid_gravity",s,std::span<const DiscreteRodAttachment>{},bad);
 auto degenerate=s;degenerate.positions[1]=degenerate.positions[0];reject("degenerate",degenerate,std::span<const DiscreteRodAttachment>{},c);
 return failed?1:0;
}
