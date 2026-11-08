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
#include <cstring>
using namespace metalrobo;
bool sameState(const DiscreteElasticRodState&a,const DiscreteElasticRodState&b){return a.positions==b.positions&&a.velocities==b.velocities&&a.twists==b.twists&&a.twistRates==b.twistRates;}
bool sameOutput(const MetalDiscreteElasticRodResult&a,const MetalDiscreteElasticRodResult&b){
 if(a.states.size()!=b.states.size()||a.statuses.size()!=b.statuses.size()||a.reactions.size()!=b.reactions.size()||a.rigidBodies.size()!=b.rigidBodies.size()||a.toolContactCounts!=b.toolContactCounts||a.toolContacts.size()!=b.toolContacts.size())return false;
 for(size_t i=0;i<a.states.size();++i)if(!sameState(a.states[i],b.states[i]))return false;
 for(size_t i=0;i<a.statuses.size();++i)if(std::memcmp(&a.statuses[i],&b.statuses[i],sizeof(a.statuses[i])))return false;
 for(size_t i=0;i<a.reactions.size();++i){const auto&x=a.reactions[i];const auto&y=b.reactions[i];if(x.nodeIndex!=y.nodeIndex||x.bodyIndex!=y.bodyIndex||x.impulseOnTarget!=y.impulseOnTarget||x.averageForceOnTarget!=y.averageForceOnTarget||x.finalPositionError!=y.finalPositionError)return false;}
 return a.rigidBodies.empty()&&a.toolContacts.empty();
}
int main(int argc,char**argv){
 if(argc!=5)return 2;const double offset=std::stod(argv[2]);const std::string kind=argv[3];const int run=std::stoi(argv[4]);
 constexpr int n=8, steps=32;constexpr double rest=.11758755785232791,ea=2.3818718058345718,radius=.000020,h=.001;
 if(kind!="fixed"&&kind!="moving"&&kind!="internal")return 2;
 DiscreteRodMaterial mat;mat.radius=radius;mat.density=1100;mat.youngModulus=ea/(std::numbers::pi*radius*radius);
 auto model=makeStraightSutureRod(n,rest,mat);auto initial=makeDiscreteElasticRodDefaultState(model);std::vector<int> pinned(n,0);pinned[0]=pinned[n-1]=1;if(kind=="internal")pinned[n/2]=1;
 for(int i=0;i<n;i++){double x=.122*i/(n-1)+(i>0&&i<n-1?.0002*std::sin(i*1.1):0);initial.positions[i]={offset+x,offset*.5,-offset*.25};initial.velocities[i]={!pinned[i]?.02*std::cos(i*.8):0,0,0};}
 MetalDiscreteElasticRodConfig cfg;cfg.metallibPath=argv[1];cfg.step.timestep=h;cfg.step.gravity={0,0,0};cfg.step.linearDamping=0;cfg.step.twistDamping=0;cfg.step.solverIterations=128;cfg.step.constraintTolerance=1e-7;
 std::cout<<std::setprecision(17);
 std::vector<MetalDiscreteElasticRodResult> baseline;
 for(int arm=0;arm<2;arm++){
  std::array<DiscreteElasticRodState,1> states{initial};MetalDiscreteElasticRodResult out;
  for(int step=0;step<steps;step++){
   std::vector<DiscreteRodAttachment> anchors;std::vector<double> targets(n),targetv(n);
   for(int i=0;i<n;i++)if(pinned[i]){DiscreteRodAttachment a;a.nodeIndex=i;a.targetPosition=initial.positions[i];double amplitude=kind!="fixed"&&i==n-1?.00004:kind=="internal"&&i==n/2?.00002:0.;double prev=amplitude*std::sin(.3*step),next=amplitude*std::sin(.3*(step+1));a.targetPosition[0]+=next;a.targetVelocity={(next-prev)/h,0,0};targets[i]=a.targetPosition[0];targetv[i]=a.targetVelocity[0];anchors.push_back(a);}
   MetalDiscreteElasticRodInput in;in.states=states;in.attachmentCount=anchors.size();in.attachments=anchors;
   if(arm==1&&step==16){auto inputBefore=states[0];auto outputBefore=out;auto bad=anchors;bad.back().targetPosition[0]=states[0].positions[0][0]-.1;in.attachments=bad;auto d=runMetalDiscreteElasticRod(model,in,out,cfg);
    std::cout<<"{\"type\":\"rejection\",\"run\":"<<run<<",\"offset\":"<<offset<<",\"kind\":\""<<kind<<"\",\"step\":"<<step<<",\"dispatched\":"<<(d.dispatched?"true":"false")<<",\"published\":"<<(d.published?"true":"false")<<",\"succeeded\":"<<(d.succeeded()?"true":"false")<<",\"gpu\":"<<d.firstGPUStatusCode<<",\"input_unchanged\":"<<(sameState(states[0],inputBefore)?"true":"false")<<",\"output_unchanged\":"<<(sameOutput(out,outputBefore)?"true":"false")<<"}\n";in.attachments=anchors;}
   auto previous=states[0];auto d=runMetalDiscreteElasticRod(model,in,out,cfg);bool accepted=d.succeeded()&&d.published&&out.states.size()==1&&out.reactions.size()==anchors.size();
   bool outputEqual=accepted && (arm==0 || (size_t(step)<baseline.size() && sameOutput(out,baseline[step]))); if(arm==0 && accepted)baseline.push_back(out);
   std::cout<<"{\"type\":\"step\",\"run\":"<<run<<",\"arm\":"<<arm<<",\"step\":"<<step<<",\"n\":"<<n<<",\"offset\":"<<offset<<",\"kind\":\""<<kind<<"\",\"h\":"<<h<<",\"accepted\":"<<(accepted?"true":"false")<<",\"output_equal\":"<<(outputEqual?"true":"false")<<",\"gpu\":"<<d.firstGPUStatusCode<<",\"pinned\":";printlist(pinned);
   std::vector<double> oldx(n),oldv(n);for(int i=0;i<n;i++){oldx[i]=previous.positions[i][0];oldv[i]=previous.velocities[i][0];}
   std::cout<<",\"x0\":";printlist(oldx);std::cout<<",\"v0\":";printlist(oldv);std::cout<<",\"targets\":";printlist(targets);std::cout<<",\"targetv\":";printlist(targetv);std::cout<<",\"mass\":";printlist(model.nodeMasses);std::cout<<",\"lengths\":";printlist(model.restLengths);std::cout<<",\"ea\":";printlist(model.stretchStiffness);
   std::cout<<",\"positions\":[";if(accepted)for(int i=0;i<n;i++){if(i)std::cout<<",";printv(out.states[0].positions[i]);}
   std::cout<<"],\"velocities\":[";if(accepted)for(int i=0;i<n;i++){if(i)std::cout<<",";printv(out.states[0].velocities[i]);}
   std::cout<<"],\"forces\":[";if(accepted)for(size_t i=0;i<anchors.size();i++){if(i)std::cout<<",";printv(out.reactions[i].averageForceOnTarget);}
   std::cout<<"],\"localres5\":[";if(accepted)for(int i=0;i<n;i++){if(i)std::cout<<",";V r{};if(!pinned[i])r=add(sub(out.states[0].velocities[i],previous.velocities[i]),mul(grad(model,out.states[0],i,5e-9),h/model.nodeMasses[i]));printv(r);}
   std::cout<<"],\"localres10\":[";if(accepted)for(int i=0;i<n;i++){if(i)std::cout<<",";V r{};if(!pinned[i])r=add(sub(out.states[0].velocities[i],previous.velocities[i]),mul(grad(model,out.states[0],i,1e-8),h/model.nodeMasses[i]));printv(r);}std::cout<<"]}\n";
   if(!accepted)break;states[0]=out.states[0];
  }
 }
}
