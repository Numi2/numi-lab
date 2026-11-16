#include "metalrobo/DiscreteElasticRod.hpp"
#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

using namespace metalrobo;
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("expected MetalRobo metallib");
 constexpr double L=.025,E=2.129e9,I=3.960e-12,area=3.449e-6,radius=.00185,density=1200,delta=.000025;
 DiscreteRodMaterial mat{.radius=radius,.density=density,.youngModulus=E,.poissonRatio=.35};
 auto m=makeStraightSutureRod(9,L,mat);m.name="related_crow_rachis_three_point_instrument";m.fidelityBoundary="uniform short-segment surrogate; carrion crow sample, not whole feather";
 std::fill(m.bendStiffness.begin(),m.bendStiffness.end(),E*I);std::fill(m.stretchStiffness.begin(),m.stretchStiffness.end(),E*area);
 const double dm=density*area*(L/8);for(int j=0;j<9;++j)m.nodeMasses[j]=(j==0||j==8)?.5*dm:dm;
 std::string why;if(!m.valid(&why))throw std::runtime_error(why);
 DiscreteElasticRodStepConfig cfg;cfg.timestep=.001;cfg.gravity={0,0,0};cfg.solverIterations=256;cfg.constraintTolerance=1e-7;cfg.linearDamping=100;cfg.twistDamping=100;
 std::vector<DiscreteRodAttachment> att;for(int lane=0;lane<3;++lane){double shift=(lane-1)*delta;att.push_back({.nodeIndex=0,.targetPosition={0,0,0}});att.push_back({.nodeIndex=4,.targetPosition={L/2,shift,0}});att.push_back({.nodeIndex=8,.targetPosition={L,0,0}});}
 std::cout<<std::setprecision(12)<<"{\"event\":\"setup\",\"EI\":"<<E*I<<",\"theory_force\":"<<48*E*I*delta/(L*L*L)<<",\"steps\":500,\"nodes\":9}"<<std::endl;
 bool failed=false;
 for(int lane=0;lane<3;++lane){auto state=makeDiscreteElasticRodDefaultState(m);for(int step=0;step<500;++step){std::array<DiscreteRodAttachmentReaction,3> reaction{};auto d=stepDiscreteElasticRodCpu(m,state,std::span(att.data()+lane*3,3),cfg,reaction);
  std::cout<<"{\"event\":\"cpu\",\"lane\":"<<lane<<",\"step\":"<<step<<",\"success\":"<<(d.succeeded()?"true":"false")<<",\"status\":"<<std::quoted(discreteElasticRodStatusName(d.status))<<",\"message\":"<<std::quoted(d.message)<<",\"iterations\":"<<d.iterations<<",\"force\":["<<reaction[0].averageForceOnTarget[1]<<','<<reaction[1].averageForceOnTarget[1]<<','<<reaction[2].averageForceOnTarget[1]<<"],\"mid_y\":"<<state.positions[4][1]<<",\"mid_vy\":"<<state.velocities[4][1]<<"}"<<std::endl;
  if(!d.succeeded()){failed=true;break;}
 }}
 std::vector<DiscreteElasticRodState> states(3,makeDiscreteElasticRodDefaultState(m));MetalDiscreteElasticRodConfig gcfg;gcfg.step=cfg;gcfg.metallibPath=argv[1];
 for(int step=0;step<500;++step){MetalDiscreteElasticRodInput input{.states=states,.attachmentCount=3,.attachments=att};MetalDiscreteElasticRodResult out;auto d=runMetalDiscreteElasticRod(m,input,out,gcfg);
  std::cout<<"{\"event\":\"gpu_status\",\"step\":"<<step<<",\"success\":"<<(d.succeeded()?"true":"false")<<",\"dispatched\":"<<(d.dispatched?"true":"false")<<",\"published\":"<<(d.published?"true":"false")<<",\"code\":"<<d.firstGPUStatusCode<<",\"device\":"<<std::quoted(d.deviceName)<<",\"message\":"<<std::quoted(d.message)<<"}"<<std::endl;
  if(!d.succeeded()||out.states.size()!=3||out.reactions.size()!=9){failed=true;break;}
  for(int lane=0;lane<3;++lane){const auto a=lane*3;std::cout<<"{\"event\":\"gpu\",\"lane\":"<<lane<<",\"step\":"<<step<<",\"force\":["<<out.reactions[a].averageForceOnTarget[1]<<','<<out.reactions[a+1].averageForceOnTarget[1]<<','<<out.reactions[a+2].averageForceOnTarget[1]<<"],\"mid_y\":"<<out.states[lane].positions[4][1]<<",\"mid_vy\":"<<out.states[lane].velocities[4][1]<<"}"<<std::endl;}
  states=std::move(out.states);
 }
 return failed?2:0;
 }catch(const std::exception&e){std::cerr<<"instrument exception: "<<e.what()<<'\n';return 1;}}
