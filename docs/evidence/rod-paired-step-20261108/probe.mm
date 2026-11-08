#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>
using V=std::array<double,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V mul(V a,double s){return {a[0]*s,a[1]*s,a[2]*s};}
void printv(V x){std::cout<<"["<<x[0]<<","<<x[1]<<","<<x[2]<<"]";}
template<class T>void printlist(const T& x){std::cout<<"[";bool first=true;for(auto a:x){if(!first)std::cout<<",";std::cout<<a;first=false;}std::cout<<"]";}
double energy(const metalrobo::DiscreteElasticRodModel& m,const metalrobo::DiscreteElasticRodState& s){metalrobo::DiscreteElasticRodEnergy e;auto d=metalrobo::evaluateDiscreteElasticRodEnergy(m,s,e);if(d.status!=metalrobo::DiscreteElasticRodStatus::success){std::cerr<<"energy failed";std::exit(3);}return e.total();}
V grad(const metalrobo::DiscreteElasticRodModel& m,const metalrobo::DiscreteElasticRodState& s,int i,double eps){V g{};for(int j=0;j<3;j++){auto a=s,b=s;a.positions[i][j]+=eps;b.positions[i][j]-=eps;g[j]=(energy(m,a)-energy(m,b))/(2*eps);}return g;}
int main(int argc,char**argv){
 if(argc!=6)return 2;const int n=std::stoi(argv[2]);const std::string ax=argv[3],kind=argv[5];const double off=std::stod(argv[4]);if(n!=5&&n!=8)return 2;
 V axis=ax=="x"?V{1,0,0}:V{3/std::sqrt(74.0),-4/std::sqrt(74.0),7/std::sqrt(74.0)};V origin{off,off*.5,-off*.25};
 constexpr double rest=.11758755785232791,ea=2.3818718058345718,radius=.000020,h=.001;
 metalrobo::DiscreteRodMaterial mat;mat.radius=radius;mat.density=1100;mat.youngModulus=ea/(std::numbers::pi*radius*radius);
 auto model=metalrobo::makeStraightSutureRod(n,rest,mat);auto state=metalrobo::makeDiscreteElasticRodDefaultState(model);
 std::vector<double> oldx(n),oldv(n),targets(n),targetv(n);std::vector<int> pinned(n,0);
 for(int i=0;i<n;i++){oldx[i]=.122*i/(n-1)+(i>0&&i<n-1?.0002*std::sin(i*1.1):0);oldv[i]=(i>0&&i<n-1?.02*std::cos(i*.8):0);}
 pinned[0]=pinned[n-1]=1;
 if(kind=="internal")pinned[n/2]=1;else if(kind!="fixed"&&kind!="moving")return 2;
 for(int i=0;i<n;i++){if(pinned[i])oldv[i]=0;targets[i]=oldx[i];if(kind!="fixed"&&i==n-1)targets[i]+=.00004;if(kind=="internal"&&i==n/2)targets[i]+=.00002;targetv[i]=(targets[i]-oldx[i])/h;state.positions[i]=add(origin,mul(axis,oldx[i]));state.velocities[i]=mul(axis,oldv[i]);}
 std::vector<metalrobo::DiscreteRodAttachment> anchors;
 for(int i=0;i<n;i++)if(pinned[i]){metalrobo::DiscreteRodAttachment a;a.nodeIndex=i;a.targetPosition=add(origin,mul(axis,targets[i]));a.targetVelocity=mul(axis,targetv[i]);anchors.push_back(a);}
 std::array<metalrobo::DiscreteElasticRodState,1> states{state};metalrobo::MetalDiscreteElasticRodInput input;input.states=states;input.attachmentCount=anchors.size();input.attachments=anchors;
 metalrobo::MetalDiscreteElasticRodConfig cfg;cfg.metallibPath=argv[1];cfg.step.timestep=h;cfg.step.gravity={0,0,0};cfg.step.linearDamping=0;cfg.step.twistDamping=0;cfg.step.solverIterations=128;cfg.step.constraintTolerance=1e-7;
 metalrobo::MetalDiscreteElasticRodResult out;auto d=metalrobo::runMetalDiscreteElasticRod(model,input,out,cfg);bool accepted=d.succeeded()&&d.published&&out.states.size()==1&&out.reactions.size()==anchors.size();
 std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"axis\":\""<<ax<<"\",\"offset\":"<<off<<",\"kind\":\""<<kind<<"\",\"h\":"<<h<<",\"axis_v\":";printv(axis);std::cout<<",\"origin\":";printv(origin);
 std::cout<<",\"x0\":";printlist(oldx);std::cout<<",\"v0\":";printlist(oldv);std::cout<<",\"pinned\":";printlist(pinned);std::cout<<",\"targets\":";printlist(targets);std::cout<<",\"mass\":";printlist(model.nodeMasses);std::cout<<",\"lengths\":";printlist(model.restLengths);std::cout<<",\"ea\":";printlist(model.stretchStiffness);
 std::cout<<",\"accepted\":"<<(accepted?"true":"false")<<",\"gpu\":"<<d.firstGPUStatusCode<<",\"positions\":[";
 if(accepted)for(int i=0;i<n;i++){if(i)std::cout<<",";printv(out.states[0].positions[i]);}
 std::cout<<"],\"velocities\":[";if(accepted)for(int i=0;i<n;i++){if(i)std::cout<<",";printv(out.states[0].velocities[i]);}
 std::cout<<"],\"forces\":[";if(accepted)for(size_t i=0;i<anchors.size();i++){if(i)std::cout<<",";printv(out.reactions[i].averageForceOnTarget);}std::cout<<"],\"localres5\":[";if(accepted)for(int i=0;i<n;i++){if(i)std::cout<<",";V r{};if(!pinned[i])r=add(sub(out.states[0].velocities[i],state.velocities[i]),mul(grad(model,out.states[0],i,5e-9),h/model.nodeMasses[i]));printv(r);}
 std::cout<<"],\"localres10\":[";if(accepted)for(int i=0;i<n;i++){if(i)std::cout<<",";V r{};if(!pinned[i])r=add(sub(out.states[0].velocities[i],state.velocities[i]),mul(grad(model,out.states[0],i,1e-8),h/model.nodeMasses[i]));printv(r);}std::cout<<"]}\n";
}
