#include "rod_local_frame.hpp"
#include <array>
#include <iostream>

namespace metalrobo {
MetalDiscreteElasticRodDiagnostics runMetalDiscreteElasticRod(
    const DiscreteElasticRodModel&, const MetalDiscreteElasticRodInput&,
    MetalDiscreteElasticRodResult& output, const MetalDiscreteElasticRodConfig&) {
    output.states.resize(1);
    output.states[0].positions.push_back({1,2,3});
    MetalDiscreteElasticRodDiagnostics d;
    d.status=MetalDiscreteElasticRodHostStatus::success;
    d.published=false;
    return d;
}
}
int main() {
    metalrobo::DiscreteElasticRodModel model;
    metalrobo::DiscreteElasticRodState state;
    state.positions.push_back({16,0,0});
    std::array<metalrobo::DiscreteElasticRodState,1> states{state};
    metalrobo::MetalDiscreteElasticRodInput input;input.states=states;
    metalrobo::MetalDiscreteElasticRodConfig cfg;
    metalrobo::MetalDiscreteElasticRodResult output;
    output.states.resize(1);output.states[0].positions.push_back({42,43,44});
    auto d=rod_local_research::run(model,input,output,cfg);
    const bool ok=d.status==metalrobo::MetalDiscreteElasticRodHostStatus::internalFailure &&
        !d.published && output.states.size()==1 && output.states[0].positions.size()==1 &&
        output.states[0].positions[0]==std::array<double,3>{42,43,44};
    std::cout<<"success_without_publication_rejected="<<ok<<"\n";
    return ok?0:1;
}
