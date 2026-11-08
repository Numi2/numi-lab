#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
int main(int argc, char** argv) {
    if (argc != 3 || std::filesystem::exists(argv[2])) return 2;
    constexpr double a=.04886680223398625, b=.020256155067237984, dt=.1, speed=.0005;
    constexpr double radius=.000020;
    const double rest=.120-a/(1000*b), ea=1000*b*rest;
    metalrobo::DiscreteRodMaterial mat;
    mat.radius=radius; mat.density=1100; mat.youngModulus=ea/(std::numbers::pi*radius*radius);
    auto model=metalrobo::makeStraightSutureRod(2,rest,mat);
    auto state=metalrobo::makeDiscreteElasticRodDefaultState(model);
    state.positions[1][0]=.122; state.velocities[1][0]=speed;
    std::array<metalrobo::DiscreteRodAttachment,2> attachments{};
    attachments[0].nodeIndex=0; attachments[1].nodeIndex=1;
    attachments[1].targetVelocity={speed,0,0};
    metalrobo::MetalDiscreteElasticRodConfig config;
    config.metallibPath=argv[1]; config.step.timestep=dt;
    config.step.gravity={0,0,0}; config.step.linearDamping=0; config.step.twistDamping=0;
    config.step.solverIterations=128; config.step.constraintTolerance=1e-10;
    std::ofstream out(argv[2]);
    if (!out) return 2;
    out << std::setprecision(17) << "step,extension_mm,force_N,elastic_J,kinetic_J,work_J,status\n";
    double work=0, oldForce=0, oldTarget=0, firstEnergy=0, maxForceError=0;
    for (int i=0;i<=40;++i) {
        double x=2+i*dt*speed*1000, target=.120+x/1000;
        attachments[1].targetPosition={target,0,0};
        std::array<metalrobo::DiscreteElasticRodState,1> in{state};
        metalrobo::MetalDiscreteElasticRodInput input;
        input.states=in; input.attachmentCount=2; input.attachments=attachments;
        metalrobo::MetalDiscreteElasticRodResult result;
        auto d=metalrobo::runMetalDiscreteElasticRod(model,input,result,config);
        if (!d.succeeded() || !d.published || result.states.size()!=1 || result.reactions.size()!=2) {
            std::cerr << "step="<<i<<" host="<<metalrobo::metalDiscreteElasticRodHostStatusName(d.status)
                      <<" gpu="<<d.firstGPUStatusCode<<" msg="<<d.message<<"\n";
            return 1;
        }
        state=result.states[0];
        metalrobo::DiscreteElasticRodEnergy energy;
        if (!metalrobo::evaluateDiscreteElasticRodEnergy(model,state,energy).succeeded()) return 1;
        double kinetic=0;
        for (std::size_t n=0;n<state.velocities.size();++n) {
            auto v=state.velocities[n];
            kinetic += .5*model.nodeMasses[n]*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        }
        double force=-result.reactions[1].averageForceOnTarget[0];
        double error=std::abs(force-(a+b*x))/(a+b*x);
        if (!std::isfinite(error)) return 1;
        maxForceError=std::max(maxForceError,error);
        if (i==0) firstEnergy=energy.total()+kinetic;
        else work+=.5*(force+oldForce)*(target-oldTarget);
        oldForce=force; oldTarget=target;
        out<<i<<','<<x<<','<<force<<','<<energy.total()<<','<<kinetic<<','<<work<<",ok\n";
        if (i==40) {
            double change=energy.total()+kinetic-firstEnergy;
            double workError=std::abs(work-change)/std::abs(change);
            std::cout<<std::setprecision(17)<<"accepted=41 max_force_relative_error="<<maxForceError
                     <<" work_J="<<work<<" energy_change_J="<<change<<" work_relative_error="<<workError<<"\n";
            return maxForceError<=.01 && std::isfinite(workError) && workError<=.01 ? 0 : 1;
        }
    }
    return 1;
}
