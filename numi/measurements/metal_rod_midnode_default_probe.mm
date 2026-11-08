#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string_view>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    constexpr double rest=.11758755785232791, ea=2.3818718058345718;
    constexpr double radius=.000020, h=.001, left=0, right=.122;
    metalrobo::DiscreteRodMaterial mat;
    mat.radius=radius;
    mat.density=1100;
    mat.youngModulus=ea/(std::numbers::pi*radius*radius);
    const auto model=metalrobo::makeStraightSutureRod(3,rest,mat);
    const double L0=model.restLengths[0], L1=model.restLengths[1];
    const double k0=model.stretchStiffness[0]/L0;
    const double k1=model.stretchStiffness[1]/L1;
    const double m=model.nodeMasses[1];
    struct Sample { std::string_view name; double x,v; };
    constexpr std::array<Sample,3> cases{{{"symmetric",.061,0},
                                        {"asymmetric",.060,0},
                                        {"moving",.060,.01}}};
    bool all=true;
    std::cout<<std::setprecision(17);
    for (const auto& test:cases) {
        auto state=metalrobo::makeDiscreteElasticRodDefaultState(model);
        state.positions[0]={left,0,0}; state.positions[1]={test.x,0,0};
        state.positions[2]={right,0,0}; state.velocities[1]={test.v,0,0};
        std::array<metalrobo::DiscreteElasticRodState,1> states{state};
        std::array<metalrobo::DiscreteRodAttachment,2> attachments{};
        attachments[0].nodeIndex=0; attachments[0].targetPosition={left,0,0};
        attachments[1].nodeIndex=2; attachments[1].targetPosition={right,0,0};
        metalrobo::MetalDiscreteElasticRodInput input;
        input.states=states; input.attachmentCount=2; input.attachments=attachments;
        metalrobo::MetalDiscreteElasticRodConfig cfg;
        cfg.metallibPath=argv[1]; cfg.step.timestep=h; cfg.step.gravity={0,0,0};
        cfg.step.linearDamping=0; cfg.step.twistDamping=0;
        cfg.step.solverIterations=128; cfg.step.constraintTolerance=1e-7;
        const double predicted=test.x+h*test.v;
        const double xr=(m*predicted/(h*h)+k0*(left+L0)+k1*(right-L1)) /
                        (m/(h*h)+k0+k1);
        const double vr=(xr-test.x)/h;
        const double fl=k0*(xr-left-L0), fr=-k1*(right-xr-L1);
        const bool referenceValid=std::isfinite(xr) && xr>left && xr<right;
        metalrobo::MetalDiscreteElasticRodResult result;
        const auto d=metalrobo::runMetalDiscreteElasticRod(model,input,result,cfg);
        const bool accepted=d.succeeded() && d.published &&
                            result.states.size()==1 && result.reactions.size()==2;
        bool forcePass=false, statePass=false;
        double xl=0,vl=0,al=0,ar=0;
        if (accepted) {
            xl=result.states[0].positions[1][0]; vl=result.states[0].velocities[1][0];
            al=result.reactions[0].averageForceOnTarget[0];
            ar=result.reactions[1].averageForceOnTarget[0];
            forcePass=std::isfinite(al)&&std::isfinite(ar)&&
                std::abs(al-fl)<=std::max(1e-6,.01*std::abs(fl)) &&
                std::abs(ar-fr)<=std::max(1e-6,.01*std::abs(fr));
            statePass=std::isfinite(xl)&&std::isfinite(vl)&&
                std::abs(xl-xr)<=1e-6&&std::abs(vl-vr)<=.001;
        }
        const bool pass=referenceValid && accepted && forcePass && statePass;
        all &= pass;
        std::cout<<"case="<<test.name<<" reference_valid="<<referenceValid
                 <<" accepted="<<accepted<<" host="<<metalrobo::metalDiscreteElasticRodHostStatusName(d.status)
                 <<" gpu="<<d.firstGPUStatusCode<<" x_ref="<<xr<<" x_actual="<<xl
                 <<" v_ref="<<vr<<" v_actual="<<vl
                 <<" force_left_ref="<<fl<<" force_left_actual="<<al
                 <<" force_right_ref="<<fr<<" force_right_actual="<<ar
                 <<" state_gate="<<statePass<<" force_gate="<<forcePass<<" passed="<<pass<<"\n";
    }
    return all?0:1;
}
