#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metalrobo/numi_human_stand_gpu.h"
#include "metalrobo/compensated_translation_gpu.h"
#include "metalrobo/numi_human_tendon_gpu.h"
#include "metalrobo/numi_human_joint_equality_gpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

// Bounded production-kernel fixture for the optional unilateral hip term.
// Three independent environments cover an active toe, slack-to-active,
// and active-to-slack crossings. A double reduced-mass solve is the oracle.
namespace {
constexpr unsigned envs = 3u;
constexpr unsigned nv = 9u;
constexpr unsigned nq = 10u;
constexpr unsigned bodies = 4u;
constexpr unsigned points = 4u * bodies;
constexpr std::array<float,3u> stepSizes{{0.002f,0.001f,0.0005f}};
constexpr double inertia[bodies] = {0.2, 0.01, 2.0, 0.3};
constexpr double degree = std::numbers::pi / 180.0;
constexpr double x5 = 14.5 * degree;
constexpr double referenceA = (15.0 - (0.8 / degree) * x5) / (x5 * x5);
constexpr double referenceB = ((0.8 / degree) * x5 - 10.0) / (x5 * x5 * x5);
constexpr float toeA = static_cast<float>(referenceA);
constexpr float toeB = static_cast<float>(referenceB);

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

using Buffers = std::array<id<MTLBuffer>, 26u>;

Buffers allocate(id<MTLDevice> device, const float timestep) {
    const std::array<std::size_t, 26u> sizes{{
        sizeof(MRWorldGPU),
        sizeof(MRArticulationGPU),
        nv * sizeof(MRDofPropertiesGPU),
        bodies * sizeof(MRBodyPropertiesGPU),
        sizeof(MRNumiHumanStandDispatchGPU),
        envs * nq * sizeof(float),
        envs * nv * sizeof(float),
        envs * bodies * sizeof(MRArticulatedBodyPoseGPU),
        envs * points * sizeof(MRArticulatedPointWorldGPU),
        envs * points * 3u * nv * sizeof(float),
        envs * nv * sizeof(float),
        sizeof(MRNumiHumanStandContactGPU),
        envs * bodies * MR_NUMI_HUMAN_STAND_SPATIAL_SCRATCH_ROWS * nv * sizeof(float),
        envs * bodies * 2u * sizeof(mr_float4),
        envs * nv * nv * sizeof(float),
        envs * 4u * nv * sizeof(float),
        envs * nv * nv * sizeof(float),
        envs * sizeof(MRNumiHumanStandStatusGPU),
        sizeof(MRNumiHumanTendonBindingGPU),
        sizeof(MRNumiHumanTendonTransferResultGPU),
        sizeof(MRNumiHumanJointEqualityGPU),
        envs * sizeof(MRCompensatedRootTranslationGPU),
        envs * bodies * sizeof(mr_float4),
        envs * points * sizeof(mr_float4),
        nv * (nv + 1u) * sizeof(float),
        envs * 3u * nv * sizeof(float)
    }};
    Buffers b{};
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = [device newBufferWithLength:sizes[i] options:MTLResourceStorageModeShared];
        require(b[i] != nil, "Metal fixture buffer allocation failed");
        std::memset(b[i].contents, 0, sizes[i]);
    }

    auto* world = static_cast<MRWorldGPU*>(b[0].contents);
    world->abiVersion = MR_ENGINE_ABI_VERSION;
    world->bodyCount = bodies;
    world->articulationCount = 1u;
    world->nq = nq;
    world->nv = nv;
    world->gravityAndTimestep = {0, 0, 0, timestep};

    auto* articulation = static_cast<MRArticulationGPU*>(b[1].contents);
    articulation->bodyCount = bodies;
    articulation->rootType = MR_ROOT_FLOATING;
    articulation->nq = nq;
    articulation->nv = nv;

    auto* dofs = static_cast<MRDofPropertiesGPU*>(b[2].contents);
    for (unsigned d = 0; d < nv; ++d) {
        dofs[d].vIndex = d;
        dofs[d].qIndex = d < 3u ? d :
            d < 6u ? MR_INVALID_INDEX : d + 1u;
    }

    auto* body = static_cast<MRBodyPropertiesGPU*>(b[3].contents);
    for (unsigned i = 0; i < bodies; ++i) {
        body[i].massAndInverseMass = {1.0f, 1.0f, 0.0f, 0.0f};
        body[i].inertiaRow0 = {static_cast<float>(inertia[i]), 0, 0, 0};
        body[i].inertiaRow1 = {0, static_cast<float>(inertia[i]), 0, 0};
        body[i].inertiaRow2 = {0, 0, static_cast<float>(inertia[i]), 0};
    }

    auto* dispatch = static_cast<MRNumiHumanStandDispatchGPU*>(b[4].contents);
    dispatch->abiVersion = MR_NUMI_HUMAN_STAND_ABI_VERSION;
    dispatch->environmentCount = envs;
    dispatch->stepCount = 2u;
    dispatch->qStride = nq;
    dispatch->vStride = nv;
    dispatch->pointWorldStride = points;
    dispatch->pointJacobianStride = points * 3u * nv;
    dispatch->bodyPoseStride = bodies;
    dispatch->generalizedForceStride = nv;
    dispatch->contactIterationCount = 8u;
    dispatch->groundPointAndTimestep = {0, 0, 0, timestep};
    dispatch->groundNormal = {0, 0, 1, 0};
    dispatch->targetRootOrientation = {0, 0, 0, 1};
    dispatch->hipCapsuleTermCount = 1u;
    auto& term = dispatch->hipCapsuleTerms[0];
    term.dofIndex0 = 7u;
    term.dofIndex1 = 8u;
    term.coordinate0 = 0.8f;
    term.coordinate1 = -0.7f;
    term.threshold = 0.02f;
    term.toeQuadratic = toeA;
    term.toeCubic = toeB;

    auto* q = static_cast<float*>(b[5].contents);
    auto* v = static_cast<float*>(b[6].contents);
    for (unsigned e = 0; e < envs; ++e) {
        q[e * nq + 6u] = 1.0f;
        if (e == 0u) {
            q[e * nq + 8u] = 0.09f;
            q[e * nq + 9u] = -0.01f;
            v[e * nv + 5u] = 0.05f;
            v[e * nv + 7u] = 0.03f;
            v[e * nv + 8u] = -0.01f;
        } else if (e == 1u) {
            q[e * nq + 8u] = 0.024f;
            q[e * nq + 9u] = 0.0f;
            v[e * nv + 7u] = 2.5f;
        } else {
            q[e * nq + 8u] = 0.0253125f;
            q[e * nq + 9u] = 0.0f;
            v[e * nv + 7u] = -1.0f;
        }
    }

    auto* pose = static_cast<MRArticulatedBodyPoseGPU*>(b[7].contents);
    auto* position = static_cast<MRArticulatedPointWorldGPU*>(b[8].contents);
    auto* jac = static_cast<float*>(b[9].contents);
    const std::array<std::array<float, 3u>, 4u> offsets{{
        {{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}
    }};
    for (unsigned e = 0; e < envs; ++e) {
        for (unsigned bodyIndex = 0; bodyIndex < bodies; ++bodyIndex) {
            pose[e * bodies + bodyIndex].orientation = {0, 0, 0, 1};
            for (unsigned p = 0; p < 4u; ++p) {
                const unsigned pointIndex = e * points + 4u * bodyIndex + p;
                position[pointIndex].position = {
                    offsets[p][0], offsets[p][1], offsets[p][2], 0
                };
                for (unsigned axis = 0; axis < 3u; ++axis)
                    jac[pointIndex * 3u * nv + axis * nv + axis] = 1.0f;
                for (unsigned d = 3u; d < nv; ++d) {
                    std::array<float, 3u> angularAxis{};
                    if (d < 6u) angularAxis[d - 3u] = 1.0f;
                    else if (bodyIndex > 0u && d == 5u + bodyIndex)
                        angularAxis[2] = 1.0f;
                    const auto& r = offsets[p];
                    const std::array<float, 3u> motion{{
                        angularAxis[1]*r[2] - angularAxis[2]*r[1],
                        angularAxis[2]*r[0] - angularAxis[0]*r[2],
                        angularAxis[0]*r[1] - angularAxis[1]*r[0]
                    }};
                    for (unsigned c = 0; c < 3u; ++c)
                        jac[pointIndex * 3u * nv + c * nv + d] = motion[c];
                }
            }
        }
    }
    return b;
}

std::array<double, 4u> solve4(
    std::array<std::array<double, 4u>, 4u> a,
    std::array<double, 4u> rhs
) {
    for (unsigned k = 0; k < 4u; ++k) {
        unsigned pivot = k;
        for (unsigned row = k + 1u; row < 4u; ++row)
            if (std::abs(a[row][k]) > std::abs(a[pivot][k])) pivot = row;
        require(std::abs(a[pivot][k]) > 1.0e-14, "double oracle matrix is singular");
        std::swap(a[pivot], a[k]);
        std::swap(rhs[pivot], rhs[k]);
        for (unsigned row = k + 1u; row < 4u; ++row) {
            const double scale = a[row][k] / a[k][k];
            for (unsigned col = k; col < 4u; ++col) a[row][col] -= scale * a[k][col];
            rhs[row] -= scale * rhs[k];
        }
    }
    std::array<double, 4u> x{};
    for (int row = 3; row >= 0; --row) {
        double value = rhs[static_cast<unsigned>(row)];
        for (unsigned col = static_cast<unsigned>(row) + 1u; col < 4u; ++col)
            value -= a[static_cast<unsigned>(row)][col] * x[col];
        x[static_cast<unsigned>(row)] =
            value / a[static_cast<unsigned>(row)][static_cast<unsigned>(row)];
    }
    return x;
}

double potential(double gap, double a, double b) {
    const double positive = std::max(0.0, gap);
    return a*positive*positive*positive/3.0 +
        b*positive*positive*positive*positive/4.0;
}

double energy(const std::array<double, 4u>& v, const double gap,
              const double a, const double b) {
    const std::array<std::array<double, 4u>, 4u> mass{{
        {{2.51, 0.01, 2.0, 0.3}},
        {{0.01, 0.01, 0.0, 0.0}},
        {{2.0, 0.0, 2.0, 0.0}},
        {{0.3, 0.0, 0.0, 0.3}}
    }};
    double kinetic = 0.0;
    for (unsigned i = 0; i < 4u; ++i)
        for (unsigned j = 0; j < 4u; ++j)
            kinetic += 0.5 * v[i] * mass[i][j] * v[j];
    return kinetic + potential(gap, a, b);
}

std::array<double, 4u> oracleStep(
    const std::array<double, 4u>& oldV,
    const double q0, const double q1,
    const double timestep, const double c0, const double c1,
    const double threshold, const double toeA, const double toeB,
    double& gapOut
) {
    const std::array<std::array<double, 4u>, 4u> mass{{
        {{2.51, 0.01, 2.0, 0.3}},
        {{0.01, 0.01, 0.0, 0.0}},
        {{2.0, 0.0, 2.0, 0.0}},
        {{0.3, 0.0, 0.0, 0.3}}
    }};
    gapOut = c0*q0 + c1*q1 - threshold;
    const double p = std::max(0.0, gapOut);
    const double grad = toeA*p*p + toeB*p*p*p;
    const double tangent = 2.0*toeA*p + 3.0*toeB*p*p;
    const double qdot = c0*oldV[2] + c1*oldV[3];
    std::array<std::array<double, 4u>, 4u> effective = mass;
    effective[2][2] += timestep*timestep*tangent*c0*c0;
    effective[2][3] += timestep*timestep*tangent*c0*c1;
    effective[3][2] += timestep*timestep*tangent*c1*c0;
    effective[3][3] += timestep*timestep*tangent*c1*c1;
    std::array<double, 4u> rhs{{
        0.0, 0.0,
        -c0*(grad+timestep*tangent*qdot),
        -c1*(grad+timestep*tangent*qdot)
    }};
    const auto acceleration = solve4(effective, rhs);
    std::array<double, 4u> next{};
    for (unsigned i=0; i<4u; ++i) next[i]=oldV[i]+timestep*acceleration[i];
    return next;
}

void dispatchKernel(id<MTLDevice> device,id<MTLComputePipelineState> pipeline,
                    id<MTLCommandQueue> queue,const Buffers& b,
                    const unsigned xGroups,const unsigned yGroups) {
    require(device!=nil&&queue!=nil,"Metal fixture lacks device or queue");
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    require(command!=nil&&encoder!=nil,"cannot create production-kernel command");
    [encoder setComputePipelineState:pipeline];
    for (unsigned i=0;i<b.size();++i) [encoder setBuffer:b[i] offset:0 atIndex:i];
    [encoder dispatchThreadgroups:MTLSizeMake(xGroups,yGroups,1)
           threadsPerThreadgroup:MTLSizeMake(pipeline.threadExecutionWidth,1,1)];
    [encoder endEncoding];
    dispatch_semaphore_t done=dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer>) {dispatch_semaphore_signal(done);}];
    [command commit];
    require(dispatch_semaphore_wait(done,
        dispatch_time(DISPATCH_TIME_NOW,30*NSEC_PER_SEC))==0,
        "production hip-capsule Metal fixture timed out");
    require(command.status==MTLCommandBufferStatusCompleted,
            "production hip-capsule Metal command failed");
}

std::size_t exerciseMonolithic(id<MTLDevice> device,
    id<MTLComputePipelineState> step,id<MTLCommandQueue> queue,
    const float timestep) {
    auto b=allocate(device,timestep);
    auto* d=static_cast<MRNumiHumanStandDispatchGPU*>(b[4].contents);
    auto* q=static_cast<float*>(b[5].contents);
    auto* v=static_cast<float*>(b[6].contents);
    std::array<std::array<double,4u>,envs> expectedV{{
        {{0.05,0.0,0.03,-0.01}},
        {{0.0,0.0,2.5,0.0}},
        {{0.0,0.0,-1.0,0.0}}
    }};
    std::array<std::array<double,3u>,envs> expectedQ{{
        {{0.0,0.09,-0.01}},
        {{0.0,0.024,0.0}},
        {{0.0,0.0253125,0.0}}
    }};
    std::size_t checks=0u;
    bool crossingBecameActive=false;
    bool crossingBecameSlack=false;
    double maxAbsEnergyOracleError=0.0;
    double maxAbsEnergyChange=0.0;
    double maxPositiveEnergyChange=0.0;
    for (unsigned stepIndex=0u; stepIndex<2u; ++stepIndex) {
        d->stepIndex=stepIndex;
        std::memset(b[17].contents,0,b[17].length);
        std::array<double,envs> oldEnergy{},newEnergy{},beforeGap{},afterGap{};
        for (unsigned e=0;e<envs;++e) {
            oldEnergy[e]=energy(expectedV[e],
                0.8*expectedQ[e][1]-0.7*expectedQ[e][2]-0.02,
                referenceA,referenceB);
            const auto next=oracleStep(expectedV[e],expectedQ[e][1],expectedQ[e][2],
                timestep,0.8,-0.7,0.02,double(toeA),double(toeB),beforeGap[e]);
            for (unsigned i=0;i<4u;++i) expectedV[e][i]=next[i];
            expectedQ[e][0]+=timestep*expectedV[e][1];
            expectedQ[e][1]+=timestep*expectedV[e][2];
            expectedQ[e][2]+=timestep*expectedV[e][3];
            afterGap[e]=0.8*expectedQ[e][1]-0.7*expectedQ[e][2]-0.02;
        }
        dispatchKernel(device,step,queue,b,envs,1u);
        auto* status=static_cast<MRNumiHumanStandStatusGPU*>(b[17].contents);
        for (unsigned e=0;e<envs;++e) {
            require(status[e].code==MR_NUMI_HUMAN_STAND_SUCCESS &&
                    status[e].completedSteps==stepIndex+1u,
                    "monolithic production step failed");
            ++checks;
            const std::array<unsigned,4u> index{{5u,6u,7u,8u}};
            for (unsigned i=0;i<4u;++i) {
                const double actual=v[e*nv+index[i]];
                require(std::abs(actual-expectedV[e][i])<3.0e-4*(1.0+std::abs(expectedV[e][i])),
                    "Metal hip velocity differs from independent double reduced-mass solve");
                ++checks;
            }
            for (unsigned i=0;i<3u;++i) {
                const double actual=q[e*nq+7u+i];
                require(std::abs(actual-expectedQ[e][i])<2.0e-5,
                    "Metal hip coordinate differs from independent double reduced-mass solve");
                ++checks;
            }
            newEnergy[e]=energy(expectedV[e],afterGap[e],double(toeA),double(toeB));
            std::array<double,4u> actualVelocity{};
            for (unsigned i=0;i<4u;++i) actualVelocity[i]=v[e*nv+index[i]];
            const double actualGap=0.8*q[e*nq+8u]-0.7*q[e*nq+9u]-0.02;
            const double actualEnergy=energy(actualVelocity,actualGap,double(toeA),double(toeB));
            const double energyOracleError=std::abs(actualEnergy-newEnergy[e]);
            require(std::isfinite(actualEnergy) &&
                    energyOracleError < 1.0e-3*(1.0+std::abs(newEnergy[e])),
                    "Metal kinetic plus potential energy differs from double oracle");
            maxAbsEnergyOracleError=std::max(maxAbsEnergyOracleError,energyOracleError);
            const double energyChange=newEnergy[e]-oldEnergy[e];
            maxAbsEnergyChange=std::max(maxAbsEnergyChange,std::abs(energyChange));
            maxPositiveEnergyChange=std::max(maxPositiveEnergyChange,energyChange);
            ++checks;
            if (e==1u && stepIndex==0u) {
                require(beforeGap[e]<0.0 && afterGap[e]>0.0,
                        "fixture did not cross from slack to active between accepted steps");
                ++checks;
                crossingBecameActive=true;
            }
            if (e==2u && stepIndex==0u) {
                require(beforeGap[e]>0.0 && afterGap[e]<0.0,
                        "fixture did not cross from active to slack between accepted steps");
                ++checks;
                crossingBecameSlack=true;
            }
        }
    }
    require(crossingBecameActive,"slack-to-active crossing was not observed");
    require(crossingBecameSlack,"active-to-slack crossing was not observed");
    std::cout << " timestep_s=" << timestep
        << " max_energy_oracle_error_j=" << maxAbsEnergyOracleError
        << " max_abs_energy_change_j=" << maxAbsEnergyChange
        << " max_positive_energy_change_j=" << maxPositiveEnergyChange;
    return checks;
}

std::size_t exerciseSplitMass(id<MTLDevice> device,
    id<MTLComputePipelineState> step,
    id<MTLComputePipelineState> mass,
    id<MTLCommandQueue> queue,const float timestep) {
    auto b=allocate(device,timestep);
    auto* d=static_cast<MRNumiHumanStandDispatchGPU*>(b[4].contents);
    d->stepCount=1u;
    d->flags=MR_NUMI_HUMAN_STAND_PREPARE_ONLY|MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY;
    auto* q=static_cast<float*>(b[5].contents);
    q[6u]=1.0f;q[8u]=0.09f;q[9u]=-0.01f;
    q[nq+6u]=1.0f;q[nq+8u]=0.024f;q[nq+9u]=0.0f;
    q[2u*nq+6u]=1.0f;q[2u*nq+8u]=0.0253125f;q[2u*nq+9u]=0.0f;
    dispatchKernel(device,step,queue,b,envs,1u);
    auto* status=static_cast<MRNumiHumanStandStatusGPU*>(b[17].contents);
    for(unsigned e=0;e<envs;++e)
        require(status[e].code==MR_NUMI_HUMAN_STAND_SUCCESS &&
                (status[e].flags&MR_NUMI_HUMAN_STAND_MASS_PREREQUISITES_ONLY)!=0u,
                "split mass prerequisite stage failed");
    dispatchKernel(device,mass,queue,b,(nv*nv+mass.threadExecutionWidth-1u)/
        mass.threadExecutionWidth,envs);
    auto* factor=static_cast<float*>(b[14].contents);
    auto* d2=static_cast<MRNumiHumanStandDispatchGPU*>(b[4].contents);
    const double c0=d2->hipCapsuleTerms[0].coordinate0;
    const double c1=d2->hipCapsuleTerms[0].coordinate1;
    const double a=d2->hipCapsuleTerms[0].toeQuadratic;
    const double cubic=d2->hipCapsuleTerms[0].toeCubic;
    std::size_t checks=0u;
    for(unsigned e=0;e<envs;++e) {
        const double gap=c0*q[e*nq+8u]+c1*q[e*nq+9u]-
            d2->hipCapsuleTerms[0].threshold;
        const double positive=std::max(0.0,gap);
        const double tangent=2.0*a*positive+3.0*cubic*positive*positive;
        const std::array<unsigned,4u> dofs{{5u,6u,7u,8u}};
        const double inertiaByDof[4u][4u]={
            {2.51,0.01,2.0,0.3},
            {0.01,0.01,0.0,0.0},
            {2.0,0.0,2.0,0.0},
            {0.3,0.0,0.0,0.3}
        };
        for(unsigned i=0;i<4u;++i) for(unsigned j=0;j<4u;++j) {
            double expected=inertiaByDof[i][j];
            const double ai=dofs[i]==7u?c0:dofs[i]==8u?c1:0.0;
            const double aj=dofs[j]==7u?c0:dofs[j]==8u?c1:0.0;
            expected+=double(timestep)*timestep*tangent*ai*aj;
            const double actual=factor[e*nv*nv+dofs[i]*nv+dofs[j]];
            require(std::abs(actual-expected)<4.0e-4*(1.0+std::abs(expected)),
                    "split mass assembly omitted or mis-scaled hip tangent");
            ++checks;
        }
    }
    return checks;
}

struct RefinementState {
    std::array<double,4u> v{};
    std::array<double,3u> q{};
};

std::array<RefinementState,envs> initialRefinementStates() {
    return {{
        {{{0.05,0.0,0.03,-0.01}},{{0.0,0.09,-0.01}}},
        {{{0.0,0.0,2.5,0.0}},{{0.0,0.024,0.0}}},
        {{{0.0,0.0,-1.0,0.0}},{{0.0,0.0253125,0.0}}}
    }};
}

void advanceRefinementState(RefinementState& state, const double timestep) {
    double unusedGap=0.0;
    state.v=oracleStep(state.v,state.q[1],state.q[2],timestep,
        0.8,-0.7,0.02,double(toeA),double(toeB),unusedGap);
    state.q[0]+=timestep*state.v[1];
    state.q[1]+=timestep*state.v[2];
    state.q[2]+=timestep*state.v[3];
}

std::size_t exerciseTemporalRefinement(
    id<MTLDevice> device,id<MTLComputePipelineState> step,
    id<MTLCommandQueue> queue) {
    constexpr double horizon=0.02;
    constexpr double referenceTimestep=0.000015625;
    const unsigned referenceSteps=static_cast<unsigned>(
        std::llround(horizon/referenceTimestep));
    auto reference=initialRefinementStates();
    for(unsigned n=0;n<referenceSteps;++n)
        for(auto& state:reference) advanceRefinementState(state,referenceTimestep);

    const auto stateEnergy=[](const RefinementState& state) {
        const double gap=0.8*state.q[1]-0.7*state.q[2]-0.02;
        return energy(state.v,gap,double(toeA),double(toeB));
    };
    std::array<double,stepSizes.size()> refinementError{};
    std::size_t checks=0u;
    for(std::size_t dtIndex=0u;dtIndex<stepSizes.size();++dtIndex) {
        const float timestep=stepSizes[dtIndex];
        const unsigned steps=static_cast<unsigned>(
            std::llround(horizon/double(timestep)));
        auto b=allocate(device,timestep);
        auto* d=static_cast<MRNumiHumanStandDispatchGPU*>(b[4].contents);
        auto* q=static_cast<float*>(b[5].contents);
        auto* v=static_cast<float*>(b[6].contents);
        d->stepCount=steps;
        auto expected=initialRefinementStates();
        const auto initial=expected;
        double maxEnergyOracleError=0.0;
        double maxAbsEnergyChange=0.0;
        double maxPositiveEnergyChange=0.0;
        const std::array<unsigned,4u> vIndices{{5u,6u,7u,8u}};
        for(unsigned stepIndex=0u;stepIndex<steps;++stepIndex) {
            for(auto& state:expected) advanceRefinementState(state,double(timestep));
            d->stepIndex=stepIndex;
            std::memset(b[17].contents,0,b[17].length);
            dispatchKernel(device,step,queue,b,envs,1u);
            auto* status=static_cast<MRNumiHumanStandStatusGPU*>(b[17].contents);
            for(unsigned e=0;e<envs;++e) {
                require(status[e].code==MR_NUMI_HUMAN_STAND_SUCCESS &&
                        status[e].completedSteps==stepIndex+1u,
                        "temporal-refinement production step failed");
                ++checks;
                for(unsigned i=0;i<4u;++i) {
                    const double actual=v[e*nv+vIndices[i]];
                    require(std::abs(actual-expected[e].v[i])<
                                4.0e-4*(1.0+std::abs(expected[e].v[i])),
                            "temporal-refinement Metal velocity differs from double oracle");
                    ++checks;
                }
                for(unsigned i=0;i<3u;++i) {
                    const double actual=q[e*nq+7u+i];
                    require(std::abs(actual-expected[e].q[i])<2.0e-5,
                            "temporal-refinement Metal coordinate differs from double oracle");
                    ++checks;
                }
            }
        }
        double linf=0.0;
        for(unsigned e=0;e<envs;++e) {
            for(unsigned i=0;i<4u;++i)
                linf=std::max(linf,std::abs(expected[e].v[i]-reference[e].v[i]));
            for(unsigned i=0;i<3u;++i)
                linf=std::max(linf,std::abs(expected[e].q[i]-reference[e].q[i]));
            const double oracleEnergy=stateEnergy(expected[e]);
            RefinementState actual{};
            for(unsigned i=0;i<4u;++i) actual.v[i]=v[e*nv+vIndices[i]];
            for(unsigned i=0;i<3u;++i) actual.q[i]=q[e*nq+7u+i];
            const double actualEnergy=stateEnergy(actual);
            const double energyError=std::abs(actualEnergy-oracleEnergy);
            require(std::isfinite(actualEnergy) &&
                    energyError<1.0e-3*(1.0+std::abs(oracleEnergy)),
                    "same-horizon GPU energy differs from double oracle");
            maxEnergyOracleError=std::max(maxEnergyOracleError,energyError);
            const double energyChange=oracleEnergy-stateEnergy(initial[e]);
            maxAbsEnergyChange=std::max(maxAbsEnergyChange,std::abs(energyChange));
            maxPositiveEnergyChange=std::max(maxPositiveEnergyChange,energyChange);
        }
        refinementError[dtIndex]=linf;
        std::cout << " horizon_s=" << horizon << " steps=" << steps
            << " dt_to_ref_linf=" << linf
            << " max_energy_oracle_error_j=" << maxEnergyOracleError
            << " max_abs_energy_change_j=" << maxAbsEnergyChange
            << " max_positive_energy_change_j=" << maxPositiveEnergyChange;
    }
    require(refinementError[0]>refinementError[1] &&
            refinementError[1]>refinementError[2],
            "same-horizon timestep refinement did not reduce double-oracle error");
    ++checks;
    return checks;
}
}

int main(int argc,char** argv) {
    @autoreleasepool {
        try {
            require(argc==2,"usage: hip-capsule-metal-test /path/MetalRobo.metallib");
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            if(device==nil){std::cout<<"gpu_available=false execution=not_run\n";return 77;}
            NSError* error=nil;
            id<MTLLibrary> library=[device newLibraryWithURL:
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
            require(library!=nil,"cannot load fresh production MetalRobo library");
            MTLFunctionConstantValues* constants=[[MTLFunctionConstantValues alloc] init];
            const bool disabled=false;
            [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:2];
            [constants setConstantValue:&disabled type:MTLDataTypeBool atIndex:3];
            id<MTLFunction> stepFunction=[library newFunctionWithName:
                @"mr_numi_human_stand_step" constantValues:constants error:&error];
            id<MTLFunction> massFunction=[library newFunctionWithName:
                @"mr_numi_human_stand_mass_assemble" constantValues:constants error:&error];
            require(stepFunction!=nil&&massFunction!=nil,"required production stand kernels are absent");
            id<MTLComputePipelineState> step=[device newComputePipelineStateWithFunction:stepFunction error:&error];
            require(step!=nil,"cannot compile production stand-step pipeline");
            id<MTLComputePipelineState> mass=[device newComputePipelineStateWithFunction:massFunction error:&error];
            require(mass!=nil,"cannot compile production split-mass pipeline");
            id<MTLCommandQueue> queue=[device newCommandQueue];
            require(queue!=nil,"cannot create Metal command queue");
            std::size_t mono=0u,split=0u,refinement=0u;
            for (const float timestep : stepSizes) {
                mono+=exerciseMonolithic(device,step,queue,timestep);
                split+=exerciseSplitMass(device,step,mass,queue,timestep);
            }
            refinement=exerciseTemporalRefinement(device,step,queue);
            std::cout<<"\ngpu_available=true device=\""<<device.name.UTF8String
                <<"\" kernels=production_stand_step+production_stand_mass_assemble"
                <<" timesteps_s=0.002,0.001,0.0005 monolithic_checks="<<mono
                <<" split_mass_checks="<<split<<" refinement_checks="<<refinement
                <<" total_checks="<<(mono+split+refinement)
                <<" status=passed scope=synthetic_two_coordinate_toe_not_full_human\n";
            return 0;
        } catch(const std::exception& error) {
            std::cerr<<error.what()<<'\n';return 1;
        }
    }
}
