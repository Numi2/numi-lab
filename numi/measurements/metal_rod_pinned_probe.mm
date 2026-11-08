#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    constexpr double rest=.11758755785232791, ea=2.3818718058345718, radius=.000020;
    metalrobo::DiscreteRodMaterial mat;
    mat.radius=radius; mat.density=1100;
    mat.youngModulus=ea/(std::numbers::pi*radius*radius);
    auto model=metalrobo::makeStraightSutureRod(2,rest,mat);
    auto state=metalrobo::makeDiscreteElasticRodDefaultState(model);
    state.positions[1][0]=.122;
    std::array<metalrobo::DiscreteElasticRodState,1> states{state};
    std::array<metalrobo::DiscreteRodAttachment,2> attachments{};
    attachments[0].nodeIndex=0;
    attachments[1].nodeIndex=1; attachments[1].targetPosition={.122,0,0};
    metalrobo::MetalDiscreteElasticRodInput input;
    input.states=states; input.attachmentCount=2; input.attachments=attachments;
    metalrobo::MetalDiscreteElasticRodConfig config;
    config.metallibPath=argv[1]; config.step.timestep=.001;
    config.step.gravity={0,0,0}; config.step.linearDamping=0; config.step.twistDamping=0;
    config.step.solverIterations=128; config.step.constraintTolerance=1e-10;
    metalrobo::MetalDiscreteElasticRodResult output;
    auto d=metalrobo::runMetalDiscreteElasticRod(model,input,output,config);
    std::cout<<std::setprecision(17)<<"host_status="<<metalrobo::metalDiscreteElasticRodHostStatusName(d.status)
             <<" gpu_status="<<d.firstGPUStatusCode<<" dispatched="<<d.dispatched
             <<" published="<<d.published<<" message="<<d.message<<"\n";
    if (!d.succeeded() || output.reactions.size()!=2) return 2;
    const double force=-output.reactions[1].averageForceOnTarget[0];
    const double expected=ea/rest*(.122-rest);
    const double relative=std::abs(force-expected)/expected;
    std::cout<<"force_N="<<force<<" expected_N="<<expected<<" relative_error="<<relative
             <<" force_gate="<<(relative<=.01)<<"\n";
    return relative<=.01 ? 0:1;
}
