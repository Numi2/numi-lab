#include "rod_local_frame.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

static bool markerIntact(const metalrobo::MetalDiscreteElasticRodResult& out) {
    return out.states.size()==1 && out.states[0].positions.size()==1 &&
           out.states[0].positions[0]==std::array<double,3>{42,43,44};
}
static metalrobo::MetalDiscreteElasticRodResult marker() {
    metalrobo::MetalDiscreteElasticRodResult out;
    out.states.resize(1);out.states[0].positions.push_back({42,43,44});
    return out;
}
int main(int argc,char**argv) {
    if(argc!=2) return 2;
    constexpr double rest=.11758755785232791,ea=2.3818718058345718,radius=.000020,h=.001,speed=.01;
    metalrobo::DiscreteRodMaterial mat;mat.radius=radius;mat.density=1100;
    mat.youngModulus=ea/(std::numbers::pi*radius*radius);
    const auto model=metalrobo::makeStraightSutureRod(5,rest,mat);
    const std::array<double,3> origin{0,1,16};
    std::vector<metalrobo::DiscreteElasticRodState> states;
    std::vector<metalrobo::DiscreteRodAttachment> attachments(6);
    for(std::size_t env=0;env<origin.size();++env) {
        auto state=metalrobo::makeDiscreteElasticRodDefaultState(model);
        for(unsigned i=0;i<5;++i) {
            state.positions[i][0]=origin[env]+.122*double(i)/4;
            state.velocities[i][0]=speed;
        }
        states.push_back(state);
        attachments[2*env].nodeIndex=0;
        attachments[2*env+1].nodeIndex=4;
        attachments[2*env].targetVelocity={speed,0,0};
        attachments[2*env+1].targetVelocity={speed,0,0};
    }
    metalrobo::MetalDiscreteElasticRodConfig cfg;
    cfg.metallibPath=argv[1];cfg.step.timestep=h;cfg.step.gravity={0,0,0};
    cfg.step.linearDamping=0;cfg.step.twistDamping=0;cfg.step.solverIterations=128;
    cfg.step.constraintTolerance=1e-7;
    bool all=true;double maxPosition=0,maxVelocity=0,maxForce=0;
    for(unsigned step=1;step<=6;++step) {
        for(std::size_t env=0;env<origin.size();++env) {
            const double left=origin[env]+double(step)*h*speed;
            attachments[2*env].targetPosition={left,0,0};
            attachments[2*env+1].targetPosition={left+.122,0,0};
        }
        metalrobo::MetalDiscreteElasticRodInput input;
        input.states=states;input.attachmentCount=2;input.attachments=attachments;
        metalrobo::MetalDiscreteElasticRodResult first,second;
        const auto d1=rod_local_research::run(model,input,first,cfg);
        const auto d2=rod_local_research::run(model,input,second,cfg);
        bool ok=d1.succeeded()&&d1.published&&d2.succeeded()&&d2.published&&
            first.states.size()==3&&second.states.size()==3&&first.reactions.size()==6&&second.reactions.size()==6;
        if(ok) {
            for(std::size_t env=0;env<origin.size();++env) {
                for(unsigned i=0;i<5;++i) {
                    maxPosition=std::max(maxPosition,std::abs((first.states[env].positions[i][0]-origin[env])-
                                                               first.states[0].positions[i][0]));
                    maxVelocity=std::max(maxVelocity,std::abs(first.states[env].velocities[i][0]-
                                                               first.states[0].velocities[i][0]));
                    ok &= first.states[env].positions[i]==second.states[env].positions[i] &&
                          first.states[env].velocities[i]==second.states[env].velocities[i];
                }
                for(unsigned j=0;j<2;++j) {
                    maxForce=std::max(maxForce,std::abs(first.reactions[2*env+j].averageForceOnTarget[0]-
                                                        first.reactions[j].averageForceOnTarget[0]));
                    ok &= first.reactions[2*env+j].averageForceOnTarget==second.reactions[2*env+j].averageForceOnTarget;
                }
            }
        }
        ok &= maxPosition<=1e-6&&maxVelocity<=.001&&maxForce<=1e-6;
        all &= ok;
        std::cout<<"step="<<step<<" accepted_replayed="<<ok<<"\n";
        if(!ok) break;
        states=first.states;
    }
    metalrobo::MetalDiscreteElasticRodInput input;
    input.states=states;input.attachmentCount=2;input.attachments=attachments;
    auto rejectCase=[&](const char* name,metalrobo::MetalDiscreteElasticRodInput bad,
                        metalrobo::MetalDiscreteElasticRodConfig badCfg,bool configReject) {
        auto out=marker();
        auto d=rod_local_research::run(model,bad,out,badCfg);
        bool ok=!d.succeeded()&&!d.published&&markerIntact(out);
        if(configReject) ok &= d.status==metalrobo::MetalDiscreteElasticRodHostStatus::invalidConfiguration;
        all &= ok;
        std::cout<<name<<" rejected_and_output_preserved="<<ok<<" status="
                 <<metalrobo::metalDiscreteElasticRodHostStatusName(d.status)<<"\n";
    };
    auto badCfg=cfg;badCfg.step.enableSelfCollision=true;rejectCase("self_contact",input,badCfg,true);
    badCfg=cfg;badCfg.tool.enabled=true;rejectCase("tool_contact",input,badCfg,true);
    auto badInput=input;badInput.rigidBodyCount=1;rejectCase("rigid",badInput,cfg,true);
    std::array<MRRodToolWitnessGPU,1> cache{};badInput=input;badInput.previousToolContacts=cache;
    rejectCase("tool_cache",badInput,cfg,true);
    auto invalid=attachments;invalid[0].targetPosition[0]=invalid[1].targetPosition[0]+.01;
    badInput=input;badInput.attachments=invalid;rejectCase("reversed",badInput,cfg,false);
    std::cout<<std::setprecision(17)<<"max_translation_position_error_m="<<maxPosition
             <<" max_translation_velocity_error_m_s="<<maxVelocity
             <<" max_translation_force_error_N="<<maxForce<<" passed="<<all<<"\n";
    return all?0:1;
}
