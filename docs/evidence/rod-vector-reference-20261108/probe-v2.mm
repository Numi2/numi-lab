#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <string_view>
using V=std::array<double,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V mul(V a,double s){return {a[0]*s,a[1]*s,a[2]*s};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V unit(V a){return mul(a,1/std::sqrt(dot(a,a)));}
double norm(V a){return std::sqrt(dot(a,a));}
V sub(V a,V b){return add(a,mul(b,-1));}
void arr(V a){std::cout<<"["<<a[0]<<","<<a[1]<<","<<a[2]<<"]";}
int main(int argc,char** argv){
 if(argc!=5)return 2;
 const std::string axisName=argv[2],mode=argv[4];
 V axis{};
 if(axisName=="x")axis={1,0,0}; else if(axisName=="y")axis={0,1,0};
 else if(axisName=="z")axis={0,0,1}; else if(axisName=="minusx")axis={-1,0,0};
 else if(axisName=="oblique1")axis=unit({1,2,3});
 else if(axisName=="oblique2")axis=unit({-2,3,1});
 else if(axisName=="oblique3")axis=unit({3,-4,7});
 else if(axisName=="oblique4")axis=unit({-1,-5,2});
 else if(axisName=="oblique5")axis=unit({4,1,-3}); else return 2;
 const double offset=std::stod(argv[3]); const V origin{offset,offset*.5,-offset*.25};
 V perp=unit(std::abs(axis[2])<.9 ? V{-axis[1],axis[0],0} : V{0,-axis[2],axis[1]});
 constexpr double rest=.11758755785232791,ea=2.3818718058345718,radius=.000020,h=.001,right=.122;
 metalrobo::DiscreteRodMaterial mat; mat.radius=radius; mat.density=1100; mat.youngModulus=ea/(std::numbers::pi*radius*radius);
 auto model=metalrobo::makeStraightSutureRod(3,rest,mat);
 const double L0=model.restLengths[0],L1=model.restLengths[1];
 const double k0=model.stretchStiffness[0]/L0,k1=model.stretchStiffness[1]/L1,m=model.nodeMasses[1];
 struct Sample{const char* name;double x,v;};
 constexpr std::array<Sample,3> cases{{{"symmetric",.061,0},{"asymmetric",.060,0},{"moving",.060,.01}}};
 std::cout<<std::setprecision(17);
 for(auto test:cases){
  auto state=metalrobo::makeDiscreteElasticRodDefaultState(model);
  state.positions[0]=origin;state.positions[1]=add(origin,mul(axis,test.x));state.positions[2]=add(origin,mul(axis,right));
  state.velocities[1]=mul(axis,test.v);
  if(mode=="bent")state.positions[1]=add(state.positions[1],mul(perp,.002));
  if(mode=="transverse")state.velocities[1]=add(state.velocities[1],mul(perp,.01));
  if(mode!="axial" && mode!="bent" && mode!="transverse")return 2;
  std::array<metalrobo::DiscreteElasticRodState,1> states{state};
  std::array<metalrobo::DiscreteRodAttachment,2> anchors{};
  anchors[0].nodeIndex=0;anchors[0].targetPosition=origin;
  anchors[1].nodeIndex=2;anchors[1].targetPosition=add(origin,mul(axis,right));
  metalrobo::MetalDiscreteElasticRodInput input;input.states=states;input.attachmentCount=2;input.attachments=anchors;
  metalrobo::MetalDiscreteElasticRodConfig cfg;cfg.metallibPath=argv[1];cfg.step.timestep=h;cfg.step.gravity={0,0,0};
  cfg.step.linearDamping=0;cfg.step.twistDamping=0;cfg.step.solverIterations=128;cfg.step.constraintTolerance=1e-7;
  const double predicted=test.x+h*test.v;
  const double xr=(m*predicted/(h*h)+k0*L0+k1*(right-L1))/(m/(h*h)+k0+k1);
  const double vr=(xr-test.x)/h,fl=k0*(xr-L0),fr=-k1*(right-xr-L1);
  metalrobo::MetalDiscreteElasticRodResult result;
  auto d=metalrobo::runMetalDiscreteElasticRod(model,input,result,cfg);
  bool accepted=d.succeeded()&&d.published&&result.states.size()==1&&result.reactions.size()==2;
  V p{},v{},f0{},f1{};double pe=0,ve=0,fe0=0,fe1=0,pt=0,vt=0,ft0=0,ft1=0;
  if(accepted){p=result.states[0].positions[1];v=result.states[0].velocities[1];f0=result.reactions[0].averageForceOnTarget;f1=result.reactions[1].averageForceOnTarget;
   V rel=sub(p,origin);pe=std::abs(dot(rel,axis)-xr);ve=std::abs(dot(v,axis)-vr);
   fe0=std::abs(dot(f0,axis)-fl);fe1=std::abs(dot(f1,axis)-fr);
   pt=norm(sub(rel,mul(axis,dot(rel,axis))));vt=norm(sub(v,mul(axis,dot(v,axis))));
   ft0=norm(sub(f0,mul(axis,dot(f0,axis))));ft1=norm(sub(f1,mul(axis,dot(f1,axis))));
  }
  bool pass=accepted&&pe<=1e-6&&ve<=.001&&fe0<=std::max(1e-6,.01*std::abs(fl))&&fe1<=std::max(1e-6,.01*std::abs(fr))&&pt<=1e-6&&vt<=.001&&ft0<=1e-6&&ft1<=1e-6;
  std::cout<<"{\"axis\":\""<<axisName<<"\",\"offset\":"<<offset<<",\"mode\":\""<<mode<<"\",\"case\":\""<<test.name<<"\",\"accepted\":"<<(accepted?"true":"false")<<",\"gpu\":"<<d.firstGPUStatusCode<<",\"passed\":"<<(pass?"true":"false")<<",\"pos_error\":"<<pe<<",\"vel_error\":"<<ve<<",\"force0_error\":"<<fe0<<",\"force1_error\":"<<fe1<<",\"pos_transverse\":"<<pt<<",\"vel_transverse\":"<<vt<<",\"force0_transverse\":"<<ft0<<",\"force1_transverse\":"<<ft1<<",\"pos\":";arr(p);std::cout<<",\"vel\":";arr(v);std::cout<<",\"force0\":";arr(f0);std::cout<<",\"force1\":";arr(f1);std::cout<<"}\n";
 }
 return 0;
}
