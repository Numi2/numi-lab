#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo;
constexpr double L=.025,E=2.129e9,I=3.960e-12,area=3.449e-6,radius=.00185,density=1200,delta=.000025,dh=.00000001;
DiscreteElasticRodModel model(unsigned n){DiscreteRodMaterial mat{.radius=radius,.density=density,.youngModulus=E,.poissonRatio=.35};auto m=makeStraightSutureRod(n,L,mat);m.name="related_crow_rachis_energy_instrument";m.fidelityBoundary="uniform short-segment surrogate; carrion crow sample";std::fill(m.bendStiffness.begin(),m.bendStiffness.end(),E*I);std::fill(m.stretchStiffness.begin(),m.stretchStiffness.end(),E*area);double dm=density*area*L/(n-1);for(unsigned j=0;j<n;++j)m.nodeMasses[j]=(j==0||j==n-1)?.5*dm:dm;return m;}
DiscreteElasticRodState shaped(const DiscreteElasticRodModel&m,double d){auto s=makeDiscreteElasticRodDefaultState(m);for(auto& p:s.positions){double x=p[0]/L;if(x>.5)x=1-x;p[1]=d*(3*x-4*x*x*x);}return s;}
double energy(const DiscreteElasticRodModel&m,double d){auto s=shaped(m,d);DiscreteElasticRodEnergy e;auto diag=evaluateDiscreteElasticRodEnergy(m,s,e);if(!diag.succeeded())throw std::runtime_error("energy reference failed");return e.total();}
void stateJson(const DiscreteElasticRodState&s){std::cout<<"[";for(size_t i=0;i<s.positions.size();++i){if(i)std::cout<<',';std::cout<<'[';for(int j=0;j<3;++j)std::cout<<s.positions[i][j]<<',';for(int j=0;j<3;++j){if(j)std::cout<<',';std::cout<<s.velocities[i][j];}std::cout<<']';}std::cout<<']';}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("expected metallib");std::cout<<std::setprecision(14);bool failed=false;
 for(unsigned n:{9,17,33}){auto m=model(n);double e0=energy(m,0),ep=energy(m,delta),em=energy(m,-delta),fd=(energy(m,delta+dh)-energy(m,delta-dh))/(2*dh);std::cout<<"{\"event\":\"energy\",\"nodes\":"<<n<<",\"zero\":"<<e0<<",\"plus\":"<<ep<<",\"minus\":"<<em<<",\"force_derivative\":"<<fd<<"}"<<std::endl;}
 auto m=model(9);DiscreteElasticRodStepConfig cfg;cfg.timestep=.001;cfg.gravity={0,0,0};cfg.solverIterations=4096;cfg.constraintTolerance=1e-10;cfg.linearDamping=100;cfg.twistDamping=100;
 std::vector<DiscreteRodAttachment> att;std::vector<DiscreteElasticRodState> initial;for(int lane=0;lane<3;++lane){double d=(lane-1)*delta;initial.push_back(shaped(m,d));att.push_back({.nodeIndex=0,.targetPosition={0,0,0}});att.push_back({.nodeIndex=4,.targetPosition={L/2,d,0}});att.push_back({.nodeIndex=8,.targetPosition={L,0,0}});}
 for(int lane=0;lane<3;++lane){auto s=initial[lane];for(int step=0;step<3;++step){std::array<DiscreteRodAttachmentReaction,3> reaction{};auto d=stepDiscreteElasticRodCpu(m,s,std::span(att.data()+3*lane,3),cfg,reaction);std::cout<<"{\"event\":\"cpu\",\"lane\":"<<lane<<",\"step\":"<<step<<",\"success\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<std::quoted(discreteElasticRodStatusName(d.status))<<",\"message\":"<<std::quoted(d.message)<<",\"iterations\":"<<d.iterations<<",\"force\":["<<reaction[0].averageForceOnTarget[1]<<','<<reaction[1].averageForceOnTarget[1]<<','<<reaction[2].averageForceOnTarget[1]<<"],\"state\":";stateJson(s);std::cout<<"}"<<std::endl;if(!d.succeeded()){failed=true;break;}}}
 auto states=initial;MetalDiscreteElasticRodConfig gc;gc.step=cfg;gc.metallibPath=argv[1];for(int step=0;step<3;++step){MetalDiscreteElasticRodInput in{.states=states,.attachmentCount=3,.attachments=att};MetalDiscreteElasticRodResult out;auto d=runMetalDiscreteElasticRod(m,in,out,gc);std::cout<<"{\"event\":\"gpu_status\",\"step\":"<<step<<",\"success\":"<<(d.succeeded()?"true":"false")<<",\"dispatched\":"<<(d.dispatched?"true":"false")<<",\"published\":"<<(d.published?"true":"false")<<",\"code\":"<<d.firstGPUStatusCode<<",\"device\":"<<std::quoted(d.deviceName)<<",\"message\":"<<std::quoted(d.message)<<"}"<<std::endl;if(!d.succeeded()||out.states.size()!=3||out.reactions.size()!=9){failed=true;break;}for(int lane=0;lane<3;++lane){int a=3*lane;std::cout<<"{\"event\":\"gpu\",\"lane\":"<<lane<<",\"step\":"<<step<<",\"force\":["<<out.reactions[a].averageForceOnTarget[1]<<','<<out.reactions[a+1].averageForceOnTarget[1]<<','<<out.reactions[a+2].averageForceOnTarget[1]<<"],\"state\":";stateJson(out.states[lane]);std::cout<<"}"<<std::endl;}states=std::move(out.states);}
 return failed?2:0;
 }catch(const std::exception&e){std::cerr<<"exception: "<<e.what()<<'\n';return 1;}}
