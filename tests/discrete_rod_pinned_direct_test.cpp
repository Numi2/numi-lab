#include "metalrobo/DiscreteElasticRod.hpp"
#include <array>
#include <cmath>
#include <iostream>

bool close(double a, double b, double tol) { return std::abs(a-b) <= tol; }
int main() {
    using namespace metalrobo;
    DiscreteRodMaterial mat;
    mat.radius = 2e-5; mat.density = 1100; mat.youngModulus = 1.2e9;
    auto model = makeStraightSutureRod(2, .117, mat);
    auto start = makeDiscreteElasticRodDefaultState(model);
    start.velocities[0] = {0.1, -0.2, 0.3};
    start.velocities[1] = {-0.1, 0.2, -0.1};
    start.twists[0] = 0.4; start.twistRates[0] = 0.3;
    auto state = start;
    std::array<DiscreteRodAttachment,2> a{};
    a[0].nodeIndex=1; a[0].targetPosition={.119,.001,0}; a[0].targetVelocity={.02,.01,0};
    a[1].nodeIndex=0; a[1].targetPosition={0,0,0}; a[1].targetVelocity={-.01,0,0};
    DiscreteElasticRodStepConfig c;
    c.timestep=.001; c.gravity={0,0,-9.81}; c.linearDamping=0; c.twistDamping=.02;
    c.solverIterations=1;
    std::array<DiscreteRodAttachmentReaction,2> r{};
    const auto result=stepDiscreteElasticRodCpu(model,state,a,c,r);
    if (!result.succeeded() || r[0].nodeIndex!=1 || r[1].nodeIndex!=0 ||
        state.positions[0]!=a[1].targetPosition || state.positions[1]!=a[0].targetPosition ||
        state.velocities[0]!=a[1].targetVelocity || state.velocities[1]!=a[0].targetVelocity) return 1;
    const double L=std::hypot(.119,.001);
    const double F=model.stretchStiffness[0]/model.restLengths[0]*(L-model.restLengths[0]);
    const std::array<double,3> dir{.119/L,.001/L,0};
    for (int dim=0;dim<3;++dim) {
        const double leftNet=model.nodeMasses[0]*(a[1].targetVelocity[dim]-start.velocities[0][dim]-c.gravity[dim]*c.timestep);
        const double rightNet=model.nodeMasses[1]*(a[0].targetVelocity[dim]-start.velocities[1][dim]-c.gravity[dim]*c.timestep);
        if (!close(r[1].impulseOnTarget[dim], F*dir[dim]*c.timestep-leftNet,1e-14) ||
            !close(r[0].impulseOnTarget[dim], -F*dir[dim]*c.timestep-rightNet,1e-14)) return 2;
    }
    if (!close(state.twists[0],.4003,1e-14) ||
        !close(state.twistRates[0],.3*std::exp(-.02*.001),1e-14)) return 3;
    auto replay=start;
    std::array<DiscreteRodAttachmentReaction,2> rr{};
    auto replayStatus=stepDiscreteElasticRodCpu(model,replay,a,c,rr);
    if (!replayStatus.succeeded() || replay.positions!=state.positions || replay.velocities!=state.velocities ||
        replay.twists!=state.twists || replay.twistRates!=state.twistRates ||
        rr[0].impulseOnTarget!=r[0].impulseOnTarget || rr[1].impulseOnTarget!=r[1].impulseOnTarget) return 4;
    auto rejected=start;
    a[0].targetPosition=a[1].targetPosition;
    r[0].impulseOnTarget[0]=42;
    const auto fail=stepDiscreteElasticRodCpu(model,rejected,a,c,r);
    if (fail.succeeded() || rejected.positions!=start.positions || rejected.velocities!=start.velocities ||
        rejected.twists!=start.twists || rejected.twistRates!=start.twistRates ||
        r[0].impulseOnTarget[0]!=42) return 5;
    std::cout << "direct pinned reaction, momentum, swapped anchors, replay, and degenerate rollback passed\n";
}
