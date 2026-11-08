#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
bool close(double a,double b,double t) { return std::abs(a-b)<=t; }
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    metalrobo::DiscreteRodMaterial m;
    m.radius=2e-5; m.density=1100; m.youngModulus=1.2e9;
    auto model=metalrobo::makeStraightSutureRod(2,.117,m);
    auto state=metalrobo::makeDiscreteElasticRodDefaultState(model);
    state.velocities[0]={.1,-.2,.3}; state.velocities[1]={-.1,.2,-.1};
    state.twists[0]=.4; state.twistRates[0]=.3;
    std::array<metalrobo::DiscreteRodAttachment,2> a{};
    a[0].nodeIndex=1; a[0].targetPosition={.119,.001,0}; a[0].targetVelocity={.02,.01,0};
    a[1].nodeIndex=0; a[1].targetPosition={0,0,0}; a[1].targetVelocity={-.01,0,0};
    metalrobo::MetalDiscreteElasticRodConfig c;
    c.metallibPath=argv[1]; c.step.timestep=.001; c.step.gravity={0,0,-9.81};
    c.step.linearDamping=0; c.step.twistDamping=.02;
    c.step.solverIterations=128; c.step.constraintTolerance=1e-10;
    auto run=[&](const metalrobo::DiscreteElasticRodState& s, auto& attach, auto& result) {
        std::array<metalrobo::DiscreteElasticRodState,1> arr{s};
        metalrobo::MetalDiscreteElasticRodInput in;
        in.states=arr; in.attachmentCount=2; in.attachments=attach;
        return metalrobo::runMetalDiscreteElasticRod(model,in,result,c);
    };
    metalrobo::MetalDiscreteElasticRodResult out;
    auto d=run(state,a,out);
    if (!d.succeeded() || !d.published || out.states.size()!=1 || out.reactions.size()!=2) return 1;
    const double L=std::hypot(.119,.001);
    const double F=model.stretchStiffness[0]/model.restLengths[0]*(L-model.restLengths[0]);
    const std::array<double,3> direction{.119/L,.001/L,0};
    for(int k=0;k<3;++k) {
        double left=model.nodeMasses[0]*(a[1].targetVelocity[k]-state.velocities[0][k]-c.step.gravity[k]*c.step.timestep);
        double right=model.nodeMasses[1]*(a[0].targetVelocity[k]-state.velocities[1][k]-c.step.gravity[k]*c.step.timestep);
        if (!close(out.reactions[1].impulseOnTarget[k],F*direction[k]*c.step.timestep-left,1e-9) ||
            !close(out.reactions[0].impulseOnTarget[k],-F*direction[k]*c.step.timestep-right,1e-9)) return 3;
    }
    auto saved=out.states[0];
    metalrobo::MetalDiscreteElasticRodResult next1,next2;
    auto d1=run(saved,a,next1); auto d2=run(saved,a,next2);
    if (!d1.succeeded() || !d2.succeeded() || next1.states.size()!=1 || next2.states.size()!=1 ||
        next1.states[0].positions!=next2.states[0].positions || next1.states[0].velocities!=next2.states[0].velocities ||
        next1.states[0].twists!=next2.states[0].twists || next1.states[0].twistRates!=next2.states[0].twistRates ||
        next1.reactions[0].impulseOnTarget!=next2.reactions[0].impulseOnTarget ||
        next1.reactions[1].impulseOnTarget!=next2.reactions[1].impulseOnTarget) return 4;
    auto invalid=a; invalid[0].targetPosition=invalid[1].targetPosition;
    metalrobo::MetalDiscreteElasticRodResult sentinel;
    sentinel.states.push_back(state);
    auto bad=run(state,invalid,sentinel);
    if (bad.succeeded() || sentinel.states.size()!=1 || sentinel.states[0].positions!=state.positions ||
        sentinel.states[0].velocities!=state.velocities || sentinel.states[0].twists!=state.twists ||
        sentinel.states[0].twistRates!=state.twistRates) return 5;
    std::cout<<"Metal momentum, swapped anchors, repeated saved-state continuation and degenerate output restoration passed\n";
}
