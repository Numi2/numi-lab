#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>
using V=std::array<double,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V mul(V a,double s){return {a[0]*s,a[1]*s,a[2]*s};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
double norm(V a){return std::sqrt(dot(a,a));}
void out(V a){std::cout<<"["<<a[0]<<","<<a[1]<<","<<a[2]<<"]";}
double energy(const metalrobo::DiscreteElasticRodModel& m,const metalrobo::DiscreteElasticRodState& s){metalrobo::DiscreteElasticRodEnergy e;auto x=metalrobo::evaluateDiscreteElasticRodEnergy(m,s,e);if(x.status!=metalrobo::DiscreteElasticRodStatus::success){std::cerr<<"energy failed";std::exit(3);}return e.total();}
V gradient(const metalrobo::DiscreteElasticRodModel& m,const metalrobo::DiscreteElasticRodState& state,double eps){V g{};for(int j=0;j<3;j++){auto a=state,b=state;a.positions[1][j]+=eps;b.positions[1][j]-=eps;g[j]=(energy(m,a)-energy(m,b))/(2*eps);}return g;}
int main(int argc,char**argv){
 if(argc!=4)return 2;const std::string angle=argv[2];double bend=std::stod(argv[3]);
 V axis,perp;if(angle=="x"){axis={1,0,0};perp={0,1,0};}else if(angle=="oblique"){axis={3/std::sqrt(74.0),-4/std::sqrt(74.0),7/std::sqrt(74.0)};perp={.8,.6,0};}else return 2;
 constexpr double rest=.11758755785232791,ea=2.3818718058345718,radius=.000020,h=.001;
 metalrobo::DiscreteRodMaterial mat;mat.radius=radius;mat.density=1100;mat.youngModulus=ea/(std::numbers::pi*radius*radius);
 auto m=metalrobo::makeStraightSutureRod(3,rest,mat);auto state=metalrobo::makeDiscreteElasticRodDefaultState(m);
 state.positions[0]={0,0,0};state.positions[1]=add(mul(axis,.061),mul(perp,bend));state.positions[2]=mul(axis,.122);
 std::array<metalrobo::DiscreteElasticRodState,1> init{state};std::array<metalrobo::DiscreteRodAttachment,2> anchors{};anchors[0].nodeIndex=0;anchors[1].nodeIndex=2;anchors[1].targetPosition=state.positions[2];
 metalrobo::MetalDiscreteElasticRodInput in;in.states=init;in.attachmentCount=2;in.attachments=anchors;
 metalrobo::MetalDiscreteElasticRodConfig c;c.metallibPath=argv[1];c.step.timestep=h;c.step.gravity={0,0,0};c.step.linearDamping=0;c.step.twistDamping=0;c.step.solverIterations=128;c.step.constraintTolerance=1e-7;
 metalrobo::MetalDiscreteElasticRodResult r;auto diag=metalrobo::runMetalDiscreteElasticRod(m,in,r,c);bool accepted=diag.succeeded()&&diag.published&&r.states.size()==1&&r.reactions.size()==2;
 V pos{},vel{},g1{},g2{},gs{},r1{},r2{};if(accepted){const auto &s=r.states[0];pos=s.positions[1];vel=s.velocities[1];g1=gradient(m,s,1e-8);g2=gradient(m,s,5e-9);
 for(int e=0;e<2;e++){V delta=sub(s.positions[1],s.positions[e?2:0]);double len=norm(delta);gs=add(gs,mul(delta,m.stretchStiffness[e]/m.restLengths[e]*(len-m.restLengths[e])/len));}
 r1=add(sub(vel,state.velocities[1]),mul(g1,h/m.nodeMasses[1]));r2=add(sub(vel,state.velocities[1]),mul(g2,h/m.nodeMasses[1]));}
 std::cout<<std::setprecision(17)<<"{\"axis\":\""<<angle<<"\",\"bend\":"<<bend<<",\"accepted\":"<<(accepted?"true":"false")<<",\"gpu\":"<<diag.firstGPUStatusCode<<",\"mass\":"<<m.nodeMasses[1]<<",\"position\":";out(pos);std::cout<<",\"velocity\":";out(vel);std::cout<<",\"gradient10\":";out(g1);std::cout<<",\"gradient5\":";out(g2);std::cout<<",\"stretch_gradient\":";out(gs);std::cout<<",\"residual10\":";out(r1);std::cout<<",\"residual5\":";out(r2);std::cout<<"}\n";
}
