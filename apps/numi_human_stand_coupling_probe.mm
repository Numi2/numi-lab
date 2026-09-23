#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MujocoMuscleReference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef METALROBO_DEFAULT_METALLIB
#error "METALROBO_DEFAULT_METALLIB must name the build-tree Metal library"
#endif

// This is deliberately a small production-path discriminator, not a standing
// qualification.  It combines a floating root, two scalar joints, source
// MyoSim, a source-style plane support witness, passive preload, tendon
// transfer, and one bilateral scalar equality.  The no-contact one-step
// branch is compared with an independent FP64 source route/dynamics/equality
// projection.  The frictionless normal-contact/equality/upper-limit triad
// also has a one-step FP64 KKT reference; the full frictional plane-contact
// solve remains outside this probe's CPU oracle.
namespace {

using metalrobo::ArticulatedBodyWrench;
using metalrobo::EngineModel;
using metalrobo::MetalArticulatedOperatorConfig;
using metalrobo::MetalArticulatedOperatorContext;
using metalrobo::MetalArticulatedOperatorDiagnostics;
using metalrobo::MetalArticulatedOperatorInput;
using metalrobo::MetalArticulatedOperatorResult;
using metalrobo::MujocoCompliantMuscleArchitecture;
using metalrobo::MujocoCompliantMuscleResult;
using metalrobo::MujocoCompliantMuscleState;
using metalrobo::MujocoMuscleDefinition;
using metalrobo::MujocoMuscleResult;
using metalrobo::MujocoMuscleSite;
using metalrobo::MujocoMuscleState;
using metalrobo::MujocoRouteNode;
using metalrobo::MujocoRouteNodeType;

constexpr float kDefaultTimestepSeconds = 100.0e-6f;
constexpr std::uint32_t kMasterQ = 7u;
constexpr std::uint32_t kDependentQ = 8u;
constexpr std::uint32_t kMasterV = 6u;
constexpr std::uint32_t kDependentV = 7u;
constexpr std::uint32_t kFirstArticulatedBody = 1u;
constexpr std::uint32_t kTerminalBody = 3u;
constexpr std::uint32_t kBodyCount = 3u;
constexpr std::uint32_t kBodyProbeCount = 4u * kBodyCount;
constexpr float kEqualitySlope = 2.0f;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] mr_float4 f4(
    const float x = 0.0f,
    const float y = 0.0f,
    const float z = 0.0f,
    const float w = 0.0f
) {
    return {x, y, z, w};
}

[[nodiscard]] std::vector<double> asDouble(const std::vector<float>& input) {
    return {input.begin(), input.end()};
}

[[nodiscard]] bool finiteVector(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), [](const float value) {
        return std::isfinite(value);
    });
}

template <typename T>
[[nodiscard]] bool sameBytes(
    const std::vector<T>& first,
    const std::vector<T>& second
) {
    static_assert(std::is_trivially_copyable_v<T>);
    return first.size() == second.size() &&
        (first.empty() || std::memcmp(
            first.data(), second.data(), first.size() * sizeof(T)
        ) == 0);
}

[[nodiscard]] double maximumDifference(
    const std::vector<float>& first,
    const std::vector<float>& second
) {
    require(first.size() == second.size(), "comparison dimensions disagree");
    double maximum = 0.0;
    for (std::size_t index = 0u; index < first.size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(first[index]) - second[index])
        );
    }
    return maximum;
}

[[nodiscard]] EngineModel makeModel(const float timestepSeconds) {
    EngineModel model = metalrobo::makeFreeSphereEngineModel();
    model.name = "numi_human_stand_coupling_minimal";
    model.world.bodyCount = 4u;
    model.world.jointCount = 2u;
    model.world.nq = 9u;
    model.world.nv = 8u;
    model.world.gravityAndTimestep = f4(0.0f, -9.81f, 0.0f, timestepSeconds);

    MRArticulationGPU& articulation = model.articulations.front();
    articulation.bodyCount = kBodyCount;
    articulation.jointCount = 2u;
    articulation.nq = 9u;
    articulation.nv = 8u;

    model.joints.clear();
    model.joints.reserve(2u);
    model.bodies.reserve(4u);
    model.dofs.reserve(8u);
    for (std::uint32_t jointIndex = 0u; jointIndex < 2u; ++jointIndex) {
        const std::uint32_t parent = kFirstArticulatedBody + jointIndex;
        const std::uint32_t child = parent + 1u;
        MRJointDescriptorGPU joint{};
        joint.parentBody = parent;
        joint.childBody = child;
        joint.jointType = MR_JOINT_PRISMATIC;
        joint.qOffset = kMasterQ + jointIndex;
        joint.nq = 1u;
        joint.vOffset = kMasterV + jointIndex;
        joint.nv = 1u;
        joint.axis0 = f4(0.0f, 1.0f, 0.0f, 0.0f);
        joint.parentRotation = f4(0.0f, 0.0f, 0.0f, 1.0f);
        joint.childRotation = joint.parentRotation;
        model.joints.push_back(joint);

        MRBodyPropertiesGPU body = model.bodies.at(kFirstArticulatedBody);
        body.parentBody = parent;
        body.inboundJoint = jointIndex;
        body.massAndInverseMass = f4(0.5f, 2.0f, 0.0f, 0.0f);
        body.inertiaRow0 = f4(0.05f, 0.0f, 0.0f, 0.0f);
        body.inertiaRow1 = f4(0.0f, 0.05f, 0.0f, 0.0f);
        body.inertiaRow2 = f4(0.0f, 0.0f, 0.05f, 0.0f);
        body.inverseInertiaRow0 = f4(20.0f, 0.0f, 0.0f, 0.0f);
        body.inverseInertiaRow1 = f4(0.0f, 20.0f, 0.0f, 0.0f);
        body.inverseInertiaRow2 = f4(0.0f, 0.0f, 20.0f, 0.0f);
        model.bodies.push_back(body);

        MRDofPropertiesGPU dof{};
        dof.articulationIndex = 0u;
        dof.jointIndex = jointIndex;
        dof.qIndex = joint.qOffset;
        dof.vIndex = joint.vOffset;
        dof.localDof = 0u;
        dof.flags = MR_DOF_FLAG_POSITION_LIMIT;
        dof.limits = jointIndex == 0u
            ? f4(-0.10f, 0.10f, 0.0f, 0.0f)
            : f4(-0.20f, 0.02f, 0.0f, 0.0f);
        model.dofs.push_back(dof);
    }

    model.defaultQ.assign(9u, 0.0f);
    model.defaultQ[6u] = 1.0f;
    model.defaultQ[kMasterQ] = 0.01f;
    model.defaultQ[kDependentQ] = 0.02f;
    model.defaultV.assign(8u, 0.0f);

    std::string reason;
    require(model.valid(&reason), "minimal coupling model is invalid: " + reason);
    return model;
}

[[nodiscard]] std::vector<MRArticulatedPointImpulseGPU> makeQueries() {
    constexpr std::array<mr_float4, 4u> kBodyProbeLocals{{
        {0.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    }};
    std::vector<MRArticulatedPointImpulseGPU> points;
    points.reserve(kBodyProbeCount + 1u);
    for (std::uint32_t localBody = 0u; localBody < kBodyCount; ++localBody) {
        for (const mr_float4 localPoint : kBodyProbeLocals) {
            MRArticulatedPointImpulseGPU point{};
            point.bodyIndex = kFirstArticulatedBody + localBody;
            point.localPoint = localPoint;
            points.push_back(point);
        }
    }
    MRArticulatedPointImpulseGPU support{};
    support.bodyIndex = kTerminalBody;
    support.flags = MR_ARTICULATED_POINT_SPHERE_SUPPORT;
    support.localPoint = f4();
    support.supportPlaneNormalAndRadius = f4(0.0f, 1.0f, 0.0f, 0.03f);
    points.push_back(support);
    return points;
}

struct Fixture {
    EngineModel model;
    std::vector<float> q;
    std::vector<float> v;
    std::vector<float> passivePreload;
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::vector<MRNumiHumanStandContactGPU> contacts;
    std::vector<MRNumiHumanJointEqualityGPU> equalities;
    std::vector<MRNumiHumanTendonBindingGPU> tendonBindings;
    std::vector<MRMujocoMuscleSiteGPU> gpuSites;
    std::vector<MRMujocoMuscleRouteNodeGPU> gpuRoutes;
    std::vector<MRMujocoMuscleGPU> gpuMuscles;
    std::vector<MRMujocoMuscleStateGPU> gpuStates;
    std::vector<MujocoMuscleSite> referenceSites;
    MujocoMuscleDefinition referenceMuscle;
    MujocoCompliantMuscleArchitecture architecture;
    MujocoMuscleState referenceState;
    float timestepSeconds = kDefaultTimestepSeconds;

    explicit Fixture(const float timestep)
        : model(makeModel(timestep)),
          q(model.defaultQ),
          v(model.defaultV),
          passivePreload(model.world.nv, 0.0f),
          points(makeQueries()),
          timestepSeconds(timestep) {
        // The preload represents the production passive/equality/limit input
        // channel.  It is intentionally modest and not a standing tune.
        passivePreload[kMasterV] = 0.35f;
        passivePreload[kDependentV] = -0.20f;

        MRNumiHumanStandContactGPU support{};
        support.bodyIndex = kTerminalBody;
        support.pointQueryIndex = kBodyProbeCount;
        support.sourceGeometryIndex = 0u;
        support.frictionSlopAndStabilization = f4(0.70f, 0.002f, 0.20f, 0.0f);
        contacts.push_back(support);

        MRNumiHumanJointEqualityGPU equality{};
        equality.indices = {kDependentQ, kDependentV, kMasterQ, kMasterV};
        equality.referencesAndCoefficients0 = f4(0.0f, 0.0f, 0.0f, kEqualitySlope);
        equality.coefficients1 = f4();
        equality.solref = f4(0.02f, 1.0f, 0.0f, 0.0f);
        equality.solimp0 = f4(0.9f, 0.95f, 0.001f, 0.5f);
        equality.solimp1 = f4(2.0f, 0.0f, 0.0f, 0.0f);
        equalities.push_back(equality);

        referenceSites = {
            {kFirstArticulatedBody, {0.15, 0.00, 0.00}},
            {kTerminalBody, {-0.15, 0.10, 0.00}},
        };
        referenceMuscle.route = {
            {MujocoRouteNodeType::site, 0u, MR_INVALID_INDEX},
            {MujocoRouteNodeType::site, 1u, MR_INVALID_INDEX},
        };
        referenceMuscle.lengthRange = {0.20, 0.50};
        referenceMuscle.accelerationScale = 1.0;
        referenceMuscle.controlRange = {0.0, 1.0};
        referenceMuscle.gainParameters = {
            0.75, 1.05, 1.0, 200.0, 0.5,
            1.60, 1.5, 1.30, 1.20, 0.0,
        };
        referenceMuscle.biasParameters = referenceMuscle.gainParameters;
        referenceMuscle.dynamicParameters = {
            0.01, 0.04, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        };
        referenceState = {.excitation = 0.45, .activation = 0.45};
        architecture = {
            .optimalFiberLength = 0.14,
            .tendonSlackLength = 0.16,
            .tendonStrainAtOneNormalizedForce = 0.049,
            .tendonStiffnessAtOneNormalizedForce = 1.375 / 0.049,
            .tendonNormalizedForceAtToeEnd = 2.0 / 3.0,
            .tendonCurviness = 0.5,
            .normalizedFiberDamping = 0.1,
            .fitNormalizedRmse = 0.0,
        };

        for (const MujocoMuscleSite& source : referenceSites) {
            MRMujocoMuscleSiteGPU site{};
            site.bodyIndex = source.bodyIndex;
            site.localPoint = f4(
                static_cast<float>(source.localPoint[0]),
                static_cast<float>(source.localPoint[1]),
                static_cast<float>(source.localPoint[2]),
                0.0f
            );
            gpuSites.push_back(site);
        }
        for (const MujocoRouteNode& source : referenceMuscle.route) {
            MRMujocoMuscleRouteNodeGPU node{};
            node.type = MR_MUJOCO_MUSCLE_ROUTE_SITE;
            node.targetIndex = source.targetIndex;
            node.sideSiteIndex = source.sideSiteIndex;
            gpuRoutes.push_back(node);
        }

        MRMujocoMuscleGPU muscle{};
        muscle.route = {0u, static_cast<mr_u32>(gpuRoutes.size()), 0u, 0u};
        muscle.lengthRangeAndAcceleration = f4(0.20f, 0.50f, 1.0f, 0.0f);
        muscle.controlRange = f4(0.0f, 1.0f, 0.0f, 0.0f);
        for (std::size_t parameter = 0u; parameter < 10u; ++parameter) {
            (&muscle.gainParameters[parameter / 4u].x)[parameter % 4u] =
                static_cast<float>(referenceMuscle.gainParameters[parameter]);
            (&muscle.biasParameters[parameter / 4u].x)[parameter % 4u] =
                static_cast<float>(referenceMuscle.biasParameters[parameter]);
            (&muscle.dynamicParameters[parameter / 4u].x)[parameter % 4u] =
                static_cast<float>(referenceMuscle.dynamicParameters[parameter]);
        }
        muscle.compliantArchitecture0 = f4(
            static_cast<float>(architecture.optimalFiberLength),
            static_cast<float>(architecture.tendonSlackLength),
            static_cast<float>(architecture.tendonStrainAtOneNormalizedForce),
            static_cast<float>(architecture.tendonStiffnessAtOneNormalizedForce)
        );
        muscle.compliantArchitecture1 = f4(
            static_cast<float>(architecture.tendonNormalizedForceAtToeEnd),
            static_cast<float>(architecture.tendonCurviness),
            static_cast<float>(architecture.normalizedFiberDamping),
            static_cast<float>(architecture.fitNormalizedRmse)
        );
        gpuMuscles.push_back(muscle);

        const MujocoMuscleResult source = sourceAt(asDouble(q), asDouble(v));
        MujocoCompliantMuscleResult stationary{};
        const MujocoCompliantMuscleState zeroLengthRequest{
            .excitation = referenceState.excitation,
            .activation = referenceState.activation,
            .fiberLength = 0.0,
            .fiberVelocity = 0.0,
        };
        const auto stationaryStatus = metalrobo::evaluateMujocoCompliantMuscle(
            source.path.length, source.path.velocity, timestepSeconds,
            referenceMuscle, architecture, zeroLengthRequest, stationary
        );
        require(stationaryStatus.succeeded() &&
                    stationary.candidateFiberLength > 0.0 &&
                    std::isfinite(stationary.candidateFiberVelocity),
                "FP64 stationary fibre initialization failed");
        MRMujocoMuscleStateGPU state{};
        state.excitationAndActivation = f4(
            static_cast<float>(referenceState.excitation),
            static_cast<float>(referenceState.activation),
            static_cast<float>(stationary.candidateFiberLength),
            0.0f
        );
        gpuStates.push_back(state);

        for (std::uint32_t endpoint = 0u; endpoint < 2u; ++endpoint) {
            MRNumiHumanTendonBindingGPU binding{};
            binding.muscleIndex = 0u;
            binding.endpointOrdinal = endpoint;
            binding.bodyIndex = endpoint == 0u
                ? kFirstArticulatedBody : kTerminalBody;
            binding.mode = MR_NUMI_HUMAN_TENDON_TRANSFER_SOURCE_POINT;
            binding.envelopeIndex = MR_INVALID_INDEX;
            binding.sourceLocalPoint = gpuSites.at(endpoint).localPoint;
            tendonBindings.push_back(binding);
        }
    }

    void moveOffDependentPositionLimit() {
        // Preserve the terminal support height exactly while moving the
        // dependent coordinate away from its upper bound.  This keeps the
        // source limit capability declared, but makes the limit solve
        // inactive for the control horizon.
        q[1u] = 0.015f;
        q[kMasterQ] = 0.005f;
        q[kDependentQ] = 0.010f;
        require(
            std::abs(q[kDependentQ] - kEqualitySlope * q[kMasterQ]) <= 1.0e-7f,
            "inactive-limit control left the equality manifold"
        );
        const MujocoMuscleResult source = sourceAt(asDouble(q), asDouble(v));
        MujocoCompliantMuscleResult stationary{};
        const MujocoCompliantMuscleState zeroLengthRequest{
            .excitation = referenceState.excitation,
            .activation = referenceState.activation,
            .fiberLength = 0.0,
            .fiberVelocity = 0.0,
        };
        const auto status = metalrobo::evaluateMujocoCompliantMuscle(
            source.path.length, source.path.velocity, timestepSeconds,
            referenceMuscle, architecture, zeroLengthRequest, stationary
        );
        require(
            status.succeeded() && stationary.candidateFiberLength > 0.0,
            "inactive-limit control could not seed its stationary fibre state"
        );
        gpuStates.front().excitationAndActivation.z =
            static_cast<float>(stationary.candidateFiberLength);
        gpuStates.front().excitationAndActivation.w = 0.0f;
    }

    [[nodiscard]] MujocoMuscleResult sourceAt(
        const std::vector<double>& configuration,
        const std::vector<double>& velocity
    ) const {
        MujocoMuscleResult result{};
        const auto status = metalrobo::evaluateMujocoMuscle(
            model, 0u, configuration, velocity, referenceSites, {},
            referenceMuscle, referenceState, result
        );
        require(status.succeeded(), "FP64 source route evaluation failed");
        return result;
    }

    [[nodiscard]] MujocoCompliantMuscleResult compliantAtInitialState() const {
        const MujocoMuscleResult source = sourceAt(asDouble(q), asDouble(v));
        MujocoCompliantMuscleResult result{};
        const MujocoCompliantMuscleState accepted{
            .excitation = referenceState.excitation,
            .activation = referenceState.activation,
            .fiberLength = gpuStates.front().excitationAndActivation.z,
            .fiberVelocity = gpuStates.front().excitationAndActivation.w,
        };
        const auto status = metalrobo::evaluateMujocoCompliantMuscle(
            source.path.length, source.path.velocity, timestepSeconds,
            referenceMuscle, architecture, accepted, result
        );
        require(status.succeeded(), "FP64 compliant fibre update failed");
        return result;
    }
};

[[nodiscard]] metalrobo::ArticulatedPointQuery fp64SupportQuery(
    const Fixture& fixture
) {
    require(fixture.contacts.size() == 1u,
            "FP64 support oracle requires exactly one contact");
    const MRNumiHumanStandContactGPU& contact = fixture.contacts.front();
    require(contact.pointQueryIndex < fixture.points.size(),
            "FP64 support oracle point index is out of range");
    const MRArticulatedPointImpulseGPU& source =
        fixture.points[contact.pointQueryIndex];
    require(source.bodyIndex == contact.bodyIndex,
            "FP64 support oracle body differs from the production contact");
    metalrobo::ArticulatedPointQuery result{};
    require(
        metalrobo::widenArticulatedPointQueryFromGPU(source, result) ==
            metalrobo::ArticulatedDynamicsStatus::success,
        "FP64 support oracle rejected the production point record"
    );
    return result;
}

struct Run {
    MetalArticulatedOperatorDiagnostics diagnostics;
    MetalArticulatedOperatorResult result;
};

struct AbsoluteStepAudit {
    std::vector<std::uint32_t> preDynamics;
    std::vector<std::uint32_t> postValidation;
    std::uint32_t abortCount = 0u;
};

bool recordAbsoluteStepPreDynamics(
    void* opaque,
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    auto* audit = static_cast<AbsoluteStepAudit*>(opaque);
    if (audit == nullptr || pass.commandBuffer == nullptr) return false;
    audit->preDynamics.push_back(pass.stepIndex);
    return true;
}

bool recordAbsoluteStepPostValidation(
    void* opaque,
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    auto* audit = static_cast<AbsoluteStepAudit*>(opaque);
    if (audit == nullptr || pass.commandBuffer == nullptr) return false;
    audit->postValidation.push_back(pass.stepIndex);
    return true;
}

void abortAbsoluteStepAudit(void* opaque, void*) {
    auto* audit = static_cast<AbsoluteStepAudit*>(opaque);
    if (audit != nullptr) ++audit->abortCount;
}

struct VelocityComparison {
    double maximumDifference = 0.0;
    std::size_t dof = 0u;
    double referenceVelocity = 0.0;
    double observedVelocity = 0.0;
};

[[nodiscard]] VelocityComparison compareVelocities(
    const std::vector<double>& reference,
    const std::vector<float>& observed
) {
    require(reference.size() == observed.size(),
            "velocity comparison dimensions disagree");
    VelocityComparison comparison{};
    for (std::size_t dof = 0u; dof < reference.size(); ++dof) {
        const double observedVelocity = static_cast<double>(observed[dof]);
        const double difference = std::abs(reference[dof] - observedVelocity);
        if (difference > comparison.maximumDifference) {
            comparison.maximumDifference = difference;
            comparison.dof = dof;
            comparison.referenceVelocity = reference[dof];
            comparison.observedVelocity = observedVelocity;
        }
    }
    return comparison;
}

[[nodiscard]] VelocityComparison compareVelocities(
    const std::vector<double>& reference,
    const std::vector<double>& observed
) {
    require(reference.size() == observed.size(),
            "FP64 comparison dimensions disagree");
    VelocityComparison comparison{};
    for (std::size_t dof = 0u; dof < reference.size(); ++dof) {
        const double difference = std::abs(reference[dof] - observed[dof]);
        if (difference > comparison.maximumDifference) {
            comparison.maximumDifference = difference;
            comparison.dof = dof;
            comparison.referenceVelocity = reference[dof];
            comparison.observedVelocity = observed[dof];
        }
    }
    return comparison;
}

[[nodiscard]] Run runHorizon(
    const Fixture& fixture,
    const std::uint32_t stepCount,
    const bool enableContact,
    const bool includeEquality,
    const std::uint32_t contactIterationCount = 16u
) {
    MetalArticulatedOperatorConfig configuration{};
    configuration.pointJacobiansOnly = true;
    configuration.mujocoActivationTimestepSeconds = fixture.timestepSeconds;
    configuration.metallibPath = METALROBO_DEFAULT_METALLIB;
    MetalArticulatedOperatorContext context(configuration);

    MetalArticulatedOperatorInput input{};
    input.articulationIndex = 0u;
    input.environmentCount = 1u;
    input.pointCount = fixture.points.size();
    input.q = fixture.q;
    input.v = fixture.v;
    input.points = fixture.points;
    input.mujoco.muscles = fixture.gpuMuscles;
    input.mujoco.states = fixture.gpuStates;
    input.mujoco.sites = fixture.gpuSites;
    input.mujoco.wraps = {};
    input.mujoco.routeNodes = fixture.gpuRoutes;
    input.mujoco.bodyJacobianPointOffset = 0u;
    input.stand.v = fixture.v;
    input.stand.preloadedGeneralizedForce = fixture.passivePreload;
    input.stand.contacts = enableContact
        ? std::span<const MRNumiHumanStandContactGPU>(fixture.contacts)
        : std::span<const MRNumiHumanStandContactGPU>{};
    input.stand.jointEqualities = includeEquality
        ? std::span<const MRNumiHumanJointEqualityGPU>(fixture.equalities)
        : std::span<const MRNumiHumanJointEqualityGPU>{};
    input.stand.tendonBindings = fixture.tendonBindings;
    input.stand.tendonEnvelopes = {};
    input.stand.stepCount = stepCount;
    input.stand.contactIterationCount = contactIterationCount;
    input.stand.enableContact = enableContact;
    input.stand.enableRootAssistance = false;
    input.stand.groundPoint = f4();
    input.stand.groundNormal = f4(0.0f, 1.0f, 0.0f, 0.0f);
    input.stand.targetRootPosition = f4(
        fixture.q[0u], fixture.q[1u], fixture.q[2u], 0.0f
    );
    input.stand.targetRootOrientation = f4(
        fixture.q[3u], fixture.q[4u], fixture.q[5u], fixture.q[6u]
    );
    input.stand.assistanceGains = f4();

    Run run{};
    run.diagnostics = context.run(fixture.model, input, run.result);
    require(run.diagnostics.succeeded() && run.diagnostics.published &&
                run.diagnostics.completedStandSteps == stepCount &&
                run.result.standQ.size() == fixture.q.size() &&
                run.result.standV.size() == fixture.v.size() &&
                run.result.standStatuses.size() == 1u &&
                run.result.standStatuses.front().code == MR_NUMI_HUMAN_STAND_SUCCESS,
            "Metal minimal Human horizon failed: " + run.diagnostics.message);
    require(finiteVector(run.result.standQ) && finiteVector(run.result.standV),
            "Metal minimal Human horizon published non-finite state");
    return run;
}

void checkSplitAuthoritativeHorizon(const Fixture& fixture) {
    constexpr std::uint32_t kSteps = 8u;
    MetalArticulatedOperatorConfig configuration{};
    configuration.pointJacobiansOnly = true;
    configuration.mujocoActivationTimestepSeconds = fixture.timestepSeconds;
    configuration.metallibPath = METALROBO_DEFAULT_METALLIB;
    MetalArticulatedOperatorContext context(configuration);

    AbsoluteStepAudit audit;
    MetalArticulatedOperatorInput input{};
    input.articulationIndex = 0u;
    input.environmentCount = 1u;
    input.pointCount = fixture.points.size();
    input.q = fixture.q;
    input.v = fixture.v;
    input.points = fixture.points;
    input.mujoco.muscles = fixture.gpuMuscles;
    input.mujoco.states = fixture.gpuStates;
    input.mujoco.sites = fixture.gpuSites;
    input.mujoco.routeNodes = fixture.gpuRoutes;
    input.mujoco.bodyJacobianPointOffset = 0u;
    input.stand.v = fixture.v;
    input.stand.preloadedGeneralizedForce = fixture.passivePreload;
    input.stand.jointEqualities = fixture.equalities;
    input.stand.tendonBindings = fixture.tendonBindings;
    input.stand.tendonLoadProgram = {
        .context = &audit,
        .encodePreDynamics = &recordAbsoluteStepPreDynamics,
        .encodePostValidation = &recordAbsoluteStepPostValidation,
        .abort = &abortAbsoluteStepAudit,
        .fingerprint = 0x53504c4954385354ull,
    };
    input.stand.stepCount = 1u;
    input.stand.authoritativeStepCount = kSteps;
    input.stand.contactIterationCount = 16u;
    input.stand.enableContact = false;
    input.stand.enableRootAssistance = false;
    input.stand.groundNormal = f4(0.0f, 1.0f, 0.0f, 0.0f);
    input.stand.targetRootPosition = f4(
        fixture.q[0u], fixture.q[1u], fixture.q[2u], 0.0f);
    input.stand.targetRootOrientation = f4(
        fixture.q[3u], fixture.q[4u], fixture.q[5u], fixture.q[6u]);

    std::vector<float> q = fixture.q;
    std::vector<float> v = fixture.v;
    std::vector<MRMujocoMuscleStateGPU> states = fixture.gpuStates;
    std::vector<MRCompensatedRootTranslationGPU> roots;
    MetalArticulatedOperatorResult finalResult;
    for (std::uint32_t step = 0u; step < kSteps; ++step) {
        input.q = q;
        input.stand.v = v;
        input.mujoco.states = states;
        input.rootTranslations = roots;
        input.stand.stepIndexOffset = step;
        MetalArticulatedOperatorResult result;
        const auto diagnostics = context.run(fixture.model, input, result);
        require(diagnostics.succeeded() && diagnostics.published &&
                    diagnostics.completedStandSteps == step + 1u &&
                    result.standStatuses.size() == 1u &&
                    result.standStatuses.front().code ==
                        MR_NUMI_HUMAN_STAND_SUCCESS &&
                    result.standStatuses.front().completedSteps == step + 1u &&
                    result.standStatuses.front().tendonTransferCount ==
                        fixture.tendonBindings.size(),
                "split authoritative Human horizon lost its absolute step");
        q = result.standQ;
        v = result.standV;
        states = result.mujocoActivationStates;
        roots = result.standRootTranslations;
        finalResult = std::move(result);
    }
    require(audit.abortCount == 0u && audit.preDynamics.size() == kSteps &&
                audit.postValidation.size() == kSteps,
            "split authoritative Human callbacks were incomplete");
    for (std::uint32_t step = 0u; step < kSteps; ++step) {
        require(audit.preDynamics[step] == step &&
                    audit.postValidation[step] == step,
                "split authoritative Human callback step sequence drifted");
    }

    const Run legacy = runHorizon(fixture, kSteps, false, true);
    require(sameBytes(finalResult.standQ, legacy.result.standQ) &&
                sameBytes(finalResult.standV, legacy.result.standV) &&
                sameBytes(finalResult.mujocoActivationStates,
                          legacy.result.mujocoActivationStates) &&
                sameBytes(finalResult.standRootTranslations,
                          legacy.result.standRootTranslations),
            "split authoritative Human horizon differs from legacy execution");

    const auto expectInvalidRange = [&](const std::uint32_t offset,
                                        const std::uint32_t local,
                                        const std::uint32_t total) {
        input.q = fixture.q;
        input.stand.v = fixture.v;
        input.mujoco.states = fixture.gpuStates;
        input.rootTranslations = {};
        input.stand.stepIndexOffset = offset;
        input.stand.stepCount = local;
        input.stand.authoritativeStepCount = total;
        MetalArticulatedOperatorResult sentinel;
        sentinel.standQ = {-123.0f};
        const auto diagnostics = context.run(fixture.model, input, sentinel);
        require(!diagnostics.succeeded() && !diagnostics.dispatched &&
                    !diagnostics.published && sentinel.standQ.size() == 1u &&
                    sentinel.standQ.front() == -123.0f,
                "malformed authoritative Human step range did not fail closed");
    };
    expectInvalidRange(1u, 1u, 0u);
    expectInvalidRange(7u, 2u, 8u);
    expectInvalidRange(
        0u, 1u, MR_NUMI_HUMAN_STAND_MAX_HORIZON_STEPS + 1u);

    // A nonzero global offset is not a caller-selected label. It must name
    // the exact immediately preceding state and immutable boundary retained
    // by this context.
    AbsoluteStepAudit boundaryAudit;
    input.stand.tendonLoadProgram.context = &boundaryAudit;
    input.q = fixture.q;
    input.stand.v = fixture.v;
    input.mujoco.states = fixture.gpuStates;
    input.rootTranslations = {};
    input.stand.stepIndexOffset = 0u;
    input.stand.stepCount = 1u;
    input.stand.authoritativeStepCount = 2u;
    input.stand.contactIterationCount = 16u;
    MetalArticulatedOperatorContext boundaryContext(configuration);
    MetalArticulatedOperatorResult firstSegment;
    const auto firstDiagnostics = boundaryContext.run(
        fixture.model, input, firstSegment);
    require(firstDiagnostics.succeeded() && firstDiagnostics.published &&
                firstDiagnostics.completedStandSteps == 1u,
            "split predecessor fixture failed its first segment");

    q = firstSegment.standQ;
    v = firstSegment.standV;
    states = firstSegment.mujocoActivationStates;
    roots = firstSegment.standRootTranslations;
    input.q = q;
    input.stand.v = v;
    input.mujoco.states = states;
    input.rootTranslations = roots;
    input.stand.stepIndexOffset = 1u;

    MetalArticulatedOperatorContext freshContext(configuration);
    MetalArticulatedOperatorResult freshSentinel;
    freshSentinel.standQ = {-321.0f};
    const auto freshRejected = freshContext.run(
        fixture.model, input, freshSentinel);
    require(!freshRejected.succeeded() && !freshRejected.dispatched &&
                !freshRejected.published &&
                freshSentinel.standQ.size() == 1u &&
                freshSentinel.standQ.front() == -321.0f,
            "fresh context admitted a nonzero authoritative offset");

    input.stand.contactIterationCount = 15u;
    MetalArticulatedOperatorResult boundarySentinel;
    boundarySentinel.standQ = {-654.0f};
    const auto boundaryRejected = boundaryContext.run(
        fixture.model, input, boundarySentinel);
    require(!boundaryRejected.succeeded() &&
                !boundaryRejected.dispatched &&
                !boundaryRejected.published &&
                boundarySentinel.standQ.size() == 1u &&
                boundarySentinel.standQ.front() == -654.0f,
            "split continuation admitted a mutated immutable boundary");

    input.stand.contactIterationCount = 16u;
    MetalArticulatedOperatorResult secondSegment;
    const auto secondDiagnostics = boundaryContext.run(
        fixture.model, input, secondSegment);
    require(secondDiagnostics.succeeded() && secondDiagnostics.published &&
                secondDiagnostics.completedStandSteps == 2u &&
                boundaryAudit.abortCount == 0u &&
                boundaryAudit.preDynamics.size() == 2u &&
                boundaryAudit.postValidation.size() == 2u,
            "valid exact split predecessor did not remain admissible");
    std::cout << "split_authoritative_horizon=pass steps=8 callbacks=0..7 "
                 "malformed_ranges=3 legacy_byte_identity=true "
                 "fresh_offset=rejected mutated_boundary=rejected\n";
}

[[nodiscard]] std::vector<double> freeReferenceAcceleration(
    const Fixture& fixture
) {
    const std::vector<double> q = asDouble(fixture.q);
    const std::vector<double> v = asDouble(fixture.v);
    const MujocoMuscleResult source = fixture.sourceAt(q, v);
    const MujocoCompliantMuscleResult compliant =
        fixture.compliantAtInitialState();
    require(source.path.lengthJacobian.size() == fixture.model.world.nv,
            "FP64 source Jacobian has wrong dimensions");

    std::vector<double> force = asDouble(fixture.passivePreload);
    for (std::size_t dof = 0u; dof < force.size(); ++dof) {
        force[dof] += compliant.actuatorForce * source.path.lengthJacobian[dof];
    }
    // The CPU API indexes external wrenches by global body index; retain the
    // static floor slot even though this minimal reference applies no wrenches.
    std::vector<ArticulatedBodyWrench> wrenches(fixture.model.bodies.size());
    std::vector<double> freeAcceleration(fixture.model.world.nv, 0.0);
    metalrobo::ArticulatedDynamicsConfig dynamicsConfig{};
    dynamicsConfig.gravity = {0.0, -9.81, 0.0};
    dynamicsConfig.timestep = fixture.timestepSeconds;
    const auto forward = metalrobo::computeArticulatedForwardDynamics(
        fixture.model, 0u, q, v, force, wrenches, freeAcceleration,
        dynamicsConfig
    );
    require(
        forward.succeeded(),
        "FP64 free dynamics failed status=" +
            std::to_string(static_cast<std::uint32_t>(forward.status)) +
            " pivot=" + std::to_string(forward.minimumCholeskyPivot)
    );

    return freeAcceleration;
}

[[nodiscard]] std::vector<double> constrainedReferenceAcceleration(
    const Fixture& fixture
) {
    const std::vector<double> q = asDouble(fixture.q);
    std::vector<double> freeAcceleration = freeReferenceAcceleration(fixture);

    metalrobo::ArticulatedDynamicsConfig dynamicsConfig{};
    dynamicsConfig.gravity = {0.0, -9.81, 0.0};
    dynamicsConfig.timestep = fixture.timestepSeconds;
    std::vector<double> equalityRow(fixture.model.world.nv, 0.0);
    equalityRow[kMasterV] = -static_cast<double>(kEqualitySlope);
    equalityRow[kDependentV] = 1.0;
    std::vector<double> inverseMassRow(fixture.model.world.nv, 0.0);
    const auto inverse = metalrobo::computeArticulatedInverseMassResponses(
        fixture.model, 0u, q, equalityRow, inverseMassRow, dynamicsConfig
    );
    require(
        inverse.succeeded(),
        "FP64 inverse-mass equality response failed status=" +
            std::to_string(static_cast<std::uint32_t>(inverse.status)) +
            " pivot=" + std::to_string(inverse.minimumCholeskyPivot)
    );

    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t dof = 0u; dof < equalityRow.size(); ++dof) {
        numerator += equalityRow[dof] * freeAcceleration[dof];
        denominator += equalityRow[dof] * inverseMassRow[dof];
    }
    require(std::isfinite(numerator) && std::isfinite(denominator) &&
                denominator > 1.0e-12,
            "FP64 equality Schur scalar is invalid");
    const double lambda = numerator / denominator;
    for (std::size_t dof = 0u; dof < freeAcceleration.size(); ++dof) {
        freeAcceleration[dof] -= inverseMassRow[dof] * lambda;
    }
    return freeAcceleration;
}

struct DenseThreeSolve {
    std::array<double, 3u> solution{};
    double minimumAbsolutePivot = std::numeric_limits<double>::infinity();
};

[[nodiscard]] DenseThreeSolve solveDenseThreeByThree(
    const std::array<std::array<double, 3u>, 3u>& coefficients,
    const std::array<double, 3u>& rightHandSide
) {
    std::array<std::array<double, 4u>, 3u> augmented{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            augmented[row][column] = coefficients[row][column];
        }
        augmented[row][3u] = rightHandSide[row];
    }

    DenseThreeSolve result{};
    for (std::size_t column = 0u; column < 3u; ++column) {
        std::size_t pivotRow = column;
        for (std::size_t candidate = column + 1u;
             candidate < 3u;
             ++candidate) {
            if (std::abs(augmented[candidate][column]) >
                std::abs(augmented[pivotRow][column])) {
                pivotRow = candidate;
            }
        }
        const double pivot = augmented[pivotRow][column];
        require(
            std::isfinite(pivot) && std::abs(pivot) > 1.0e-12,
            "FP64 simultaneous contact/equality/limit Schur matrix is singular"
        );
        if (pivotRow != column) {
            std::swap(augmented[pivotRow], augmented[column]);
        }
        result.minimumAbsolutePivot = std::min(
            result.minimumAbsolutePivot, std::abs(augmented[column][column])
        );
        const double normalizedPivot = augmented[column][column];
        for (std::size_t entry = column; entry < 4u; ++entry) {
            augmented[column][entry] /= normalizedPivot;
        }
        for (std::size_t row = 0u; row < 3u; ++row) {
            if (row == column) continue;
            const double scale = augmented[row][column];
            for (std::size_t entry = column; entry < 4u; ++entry) {
                augmented[row][entry] -= scale * augmented[column][entry];
            }
        }
    }
    for (std::size_t row = 0u; row < 3u; ++row) {
        result.solution[row] = augmented[row][3u];
        require(
            std::isfinite(result.solution[row]),
            "FP64 simultaneous contact/equality/limit solution is non-finite"
        );
    }
    return result;
}

struct SimultaneousTriadReference {
    std::vector<double> freeVelocity;
    std::vector<double> constrainedVelocity;
    std::array<double, 3u> impulses{};
    std::array<double, 3u> targetVelocity{};
    std::array<double, 3u> constraintVelocity{};
    std::array<double, 3u> regularizedResidual{};
    std::array<std::array<double, 3u>, 3u> physicalDelassus{};
    double initialContactGap = 0.0;
    double minimumAbsolutePivot = 0.0;
};

// This follows the current production solve in FP64: normal contact, two
// bilateral-block refinements, then every source position limit using a
// response projected through the complete equality block, followed by the
// existing exact equality-coordinate overwrite. It is intentionally not a
// new solver or a candidate runtime policy. Comparing it with the simultaneous
// KKT reference separates finite-sweep/projection behavior from FP32 arithmetic
// and the exact-surface contact target.
struct ProductionOrderTriadReference {
    std::vector<double> preProjectionVelocity;
    std::vector<double> postProjectionVelocity;
    std::array<double, 3u> accumulatedImpulses{};
    std::array<double, 3u> targetVelocity{};
    std::array<double, 3u> preProjectionTargetResidual{};
    std::array<double, 3u> postProjectionTargetResidual{};
    double initialContactGap = 0.0;
    bool upperLimitAdmitted = false;
};

[[nodiscard]] ProductionOrderTriadReference productionOrderTriadReference(
    const Fixture& fixture,
    const std::uint32_t coupledSweepCount,
    const double contactTargetOverride = std::numeric_limits<double>::quiet_NaN()
) {
    constexpr double kRegularization = 1.0e-7;
    constexpr std::size_t kContactNormalRow = 0u;
    constexpr std::size_t kEqualityRow = 1u;
    constexpr std::size_t kMasterLimitRow = 2u;
    constexpr std::size_t kDependentLimitRow = 3u;
    constexpr std::size_t kRowCount = 4u;
    constexpr std::size_t kLimitCount = 2u;
    require(coupledSweepCount != 0u,
            "production-order reference needs at least one coupled sweep");
    require(fixture.contacts.size() == 1u &&
                fixture.contacts.front().frictionSlopAndStabilization.x == 0.0f,
            "production-order reference only models the frictionless normal triad");

    const std::vector<double> q = asDouble(fixture.q);
    const std::vector<double> v = asDouble(fixture.v);
    metalrobo::ArticulatedDynamicsConfig dynamicsConfig{};
    dynamicsConfig.gravity = {0.0, -9.81, 0.0};
    dynamicsConfig.timestep = fixture.timestepSeconds;

    metalrobo::ArticulatedPointQuery support = fp64SupportQuery(fixture);
    std::array<metalrobo::ArticulatedPointKinematics, 1u> points{};
    const std::size_t nv = fixture.model.world.nv;
    std::vector<double> pointJacobians(3u * nv, 0.0);
    const auto pointDiagnostics = metalrobo::computeArticulatedPointJacobians(
        fixture.model, 0u, q, v, std::span(&support, 1u), points,
        pointJacobians, dynamicsConfig
    );
    require(pointDiagnostics.succeeded(),
            "FP64 support-point Jacobian failed for production-order reference");

    ProductionOrderTriadReference result{};
    result.initialContactGap = points.front().position[1u];
    const auto& dependentProperties = fixture.model.dofs.at(kDependentV);
    const double lowerLimit = static_cast<double>(dependentProperties.limits.x);
    const double upperLimit = static_cast<double>(dependentProperties.limits.y);
    const double dependentPosition = q[kDependentQ];
    require(
        std::abs(result.initialContactGap) <= 1.0e-7 &&
            std::abs(dependentPosition - upperLimit) <= 1.0e-7,
        "production-order reference did not begin at the support/upper-limit intersection"
    );

    std::vector<double> rows(kRowCount * nv, 0.0);
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        rows[kContactNormalRow * nv + dof] = pointJacobians[nv + dof];
        rows[kEqualityRow * nv + dof] =
            dof == kMasterV ? -static_cast<double>(kEqualitySlope) :
            dof == kDependentV ? 1.0 : 0.0;
        // Production prepares every authored scalar interval. The response
        // stores +e_dof; the interval projection gives an upper stop a
        // negative impulse and a lower stop a positive impulse.
        rows[kMasterLimitRow * nv + dof] = dof == kMasterV ? 1.0 : 0.0;
        rows[kDependentLimitRow * nv + dof] =
            dof == kDependentV ? 1.0 : 0.0;
    }
    std::vector<double> responses(kRowCount * nv, 0.0);
    const auto responseDiagnostics = metalrobo::computeArticulatedInverseMassResponses(
        fixture.model, 0u, q, rows, responses, dynamicsConfig
    );
    require(responseDiagnostics.succeeded(),
            "FP64 inverse-mass production-order triad responses failed");

    const auto rowVelocity = [&](const std::size_t row,
                                 const std::vector<double>& velocity) {
        double value = 0.0;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            value += rows[row * nv + dof] * velocity[dof];
        }
        return value;
    };
    const auto applyResponse = [&](std::vector<double>& velocity,
                                   const std::span<const double> response,
                                   const double impulse) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            velocity[dof] += impulse * response[dof];
        }
    };
    const auto responseFor = [&](const std::size_t row) {
        return std::span<const double>(responses.data() + row * nv, nv);
    };
    const auto contraction = [&](const std::size_t row,
                                 const std::span<const double> response) {
        double value = 0.0;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            value += rows[row * nv + dof] * response[dof];
        }
        return value;
    };

    const double contactStabilization = static_cast<double>(
        fixture.contacts.front().frictionSlopAndStabilization.z
    );
    const double rawContactTarget = std::max(
        0.0,
        -contactStabilization * std::min(result.initialContactGap, 0.0) /
            static_cast<double>(fixture.timestepSeconds)
    );
    result.targetVelocity[kContactNormalRow] =
        std::isfinite(contactTargetOverride) ? contactTargetOverride : rawContactTarget;
    const double equalityError = dependentPosition -
        static_cast<double>(kEqualitySlope) * q[kMasterQ];
    result.targetVelocity[kEqualityRow] = std::clamp(
        -0.2 * equalityError / static_cast<double>(fixture.timestepSeconds),
        -4.0, 4.0
    );
    const auto positionLimitSlop = [](const double position,
                                      const double bound) {
        constexpr double kFloatEpsilon = 1.1920928955078125e-7;
        return 16.0 * kFloatEpsilon *
            std::max({std::abs(position), std::abs(bound), 1.0});
    };
    const auto lowerLimitTarget = [&](const double position,
                                      const double lower) {
        const double gap = position - lower;
        if (gap >= 0.0) {
            return -gap / static_cast<double>(fixture.timestepSeconds);
        }
        const double slop = positionLimitSlop(position, lower);
        if (gap >= -slop) return 0.0;
        return std::min(
            4.0,
            -0.2 * (gap + slop) /
                static_cast<double>(fixture.timestepSeconds)
        );
    };
    const auto upperLimitTarget = [&](const double position,
                                      const double upper) {
        return -lowerLimitTarget(-position, -upper);
    };
    result.targetVelocity[2u] = upperLimitTarget(
        dependentPosition, upperLimit
    );

    const double contactMass = kRegularization + contraction(
        kContactNormalRow, responseFor(kContactNormalRow)
    );
    const double equalityMass = contraction(
        kEqualityRow, responseFor(kEqualityRow)
    );
    require(std::isfinite(contactMass) && contactMass > kRegularization &&
                std::isfinite(equalityMass) && equalityMass > 1.0e-12,
            "FP64 production-order contact/equality response is invalid");

    const std::array<std::size_t, kLimitCount> limitRows{
        kMasterLimitRow, kDependentLimitRow};
    const std::array<std::size_t, kLimitCount> limitDofs{
        kMasterV, kDependentV};
    const std::array<double, kLimitCount> limitPositions{
        q[kMasterQ], q[kDependentQ]};
    const std::array<double, kLimitCount> limitLowerBounds{
        static_cast<double>(fixture.model.dofs.at(kMasterV).limits.x),
        lowerLimit};
    const std::array<double, kLimitCount> limitUpperBounds{
        static_cast<double>(fixture.model.dofs.at(kMasterV).limits.y),
        upperLimit};
    std::array<std::vector<double>, kLimitCount> projectedLimitResponses;
    std::array<double, kLimitCount> limitMasses{};
    for (std::size_t limit = 0u; limit < kLimitCount; ++limit) {
        const std::span<const double> rawResponse = responseFor(limitRows[limit]);
        projectedLimitResponses[limit].assign(
            rawResponse.begin(), rawResponse.end()
        );
        const double rawDiagonal = rawResponse[limitDofs[limit]];
        for (std::size_t refinement = 0u; refinement < 2u; ++refinement) {
            const double residual = contraction(
                kEqualityRow, projectedLimitResponses[limit]
            );
            const double correction = residual / equalityMass;
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                projectedLimitResponses[limit][dof] -=
                    correction * responses[kEqualityRow * nv + dof];
            }
        }
        if (!(projectedLimitResponses[limit][limitDofs[limit]] >
              1.0e-6 * rawDiagonal)) {
            projectedLimitResponses[limit].assign(
                rawResponse.begin(), rawResponse.end()
            );
        }
        limitMasses[limit] =
            projectedLimitResponses[limit][limitDofs[limit]] +
            kRegularization;
        require(std::isfinite(limitMasses[limit]) &&
                    limitMasses[limit] > kRegularization,
                "FP64 production-order projected limit response is invalid");
    }

    std::vector<double> candidate = v;
    const std::vector<double> freeAcceleration = freeReferenceAcceleration(fixture);
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        candidate[dof] += static_cast<double>(fixture.timestepSeconds) *
            freeAcceleration[dof];
    }
    double contactLambda = 0.0;
    double equalityLambda = 0.0;
    std::array<double, kLimitCount> limitLambdas{};
    for (std::uint32_t sweep = 0u; sweep < coupledSweepCount; ++sweep) {
        const double oldContactLambda = contactLambda;
        contactLambda = std::max(
            oldContactLambda +
                (result.targetVelocity[kContactNormalRow] -
                 rowVelocity(kContactNormalRow, candidate)) / contactMass,
            0.0
        );
        applyResponse(candidate, responseFor(kContactNormalRow),
                      contactLambda - oldContactLambda);

        // The production bilateral block runs two residual refinements per
        // coupled sweep. With this fixture's one row, the second correction is
        // normally roundoff-sized but remains part of the reference contract.
        for (std::size_t refinement = 0u; refinement < 2u; ++refinement) {
            const double equalityDelta =
                (result.targetVelocity[kEqualityRow] -
                 rowVelocity(kEqualityRow, candidate)) / equalityMass;
            equalityLambda += equalityDelta;
            applyResponse(candidate, responseFor(kEqualityRow), equalityDelta);
        }

        for (std::size_t limit = 0u; limit < kLimitCount; ++limit) {
            const double lowerVelocity = lowerLimitTarget(
                limitPositions[limit], limitLowerBounds[limit]
            );
            const double upperVelocity = upperLimitTarget(
                limitPositions[limit], limitUpperBounds[limit]
            );
            const double lowerCandidate = limitLambdas[limit] +
                (lowerVelocity - candidate[limitDofs[limit]]) /
                    limitMasses[limit];
            const double upperCandidate = limitLambdas[limit] +
                (upperVelocity - candidate[limitDofs[limit]]) /
                    limitMasses[limit];
            double nextImpulse = 0.0;
            if (lowerCandidate > 0.0) {
                nextImpulse = lowerCandidate;
            } else if (upperCandidate < 0.0) {
                nextImpulse = upperCandidate;
            }
            const double impulse = nextImpulse - limitLambdas[limit];
            limitLambdas[limit] = nextImpulse;
            applyResponse(candidate, projectedLimitResponses[limit], impulse);
        }
    }
    result.upperLimitAdmitted = limitLambdas[1u] < 0.0;
    require(result.upperLimitAdmitted && limitLambdas[0u] == 0.0,
            "FP64 production-order triad did not retain the source upper limit");
    result.accumulatedImpulses = {
        contactLambda, equalityLambda, limitLambdas[1u]};
    result.preProjectionVelocity = candidate;
    result.preProjectionTargetResidual[kContactNormalRow] = std::max(
        0.0, result.targetVelocity[kContactNormalRow] -
            rowVelocity(kContactNormalRow, candidate)
    );
    result.preProjectionTargetResidual[2u] = std::max({
        0.0,
        lowerLimitTarget(dependentPosition, lowerLimit) -
            candidate[kDependentV],
        candidate[kDependentV] - result.targetVelocity[2u]
    });
    const double postIntegrationEqualityError =
        q[kDependentQ] + static_cast<double>(fixture.timestepSeconds) *
            candidate[kDependentV] -
        static_cast<double>(kEqualitySlope) *
            (q[kMasterQ] + static_cast<double>(fixture.timestepSeconds) *
                candidate[kMasterV]);
    const double postIntegrationEqualityTarget = std::clamp(
        -0.2 * postIntegrationEqualityError /
            static_cast<double>(fixture.timestepSeconds),
        -4.0, 4.0
    );
    result.preProjectionTargetResidual[kEqualityRow] = std::abs(
        rowVelocity(kEqualityRow, candidate) - postIntegrationEqualityTarget
    );

    result.postProjectionVelocity = candidate;
    result.postProjectionVelocity[kDependentV] =
        static_cast<double>(kEqualitySlope) * result.postProjectionVelocity[kMasterV];
    result.postProjectionTargetResidual[kContactNormalRow] = std::max(
        0.0, result.targetVelocity[kContactNormalRow] -
            rowVelocity(kContactNormalRow, result.postProjectionVelocity)
    );
    result.postProjectionTargetResidual[2u] = std::max({
        0.0,
        lowerLimitTarget(dependentPosition, lowerLimit) -
            result.postProjectionVelocity[kDependentV],
        result.postProjectionVelocity[kDependentV] - result.targetVelocity[2u]
    });
    result.postProjectionTargetResidual[kEqualityRow] = std::abs(
        rowVelocity(kEqualityRow, result.postProjectionVelocity) -
            result.targetVelocity[kEqualityRow]
    );
    require(
        std::all_of(result.preProjectionVelocity.begin(),
                    result.preProjectionVelocity.end(),
                    [](const double value) { return std::isfinite(value); }) &&
            std::all_of(result.postProjectionVelocity.begin(),
                        result.postProjectionVelocity.end(),
                        [](const double value) { return std::isfinite(value); }) &&
            result.postProjectionTargetResidual[kEqualityRow] <= 1.0e-14,
        "FP64 production-order triad produced an invalid final projection"
    );
    return result;
}

[[nodiscard]] SimultaneousTriadReference simultaneousTriadReference(
    const Fixture& fixture,
    const double contactTargetOverride = std::numeric_limits<double>::quiet_NaN()
) {
    constexpr double kRegularization = 1.0e-7;
    constexpr std::size_t kContactNormalRow = 0u;
    constexpr std::size_t kEqualityRow = 1u;
    constexpr std::size_t kUpperLimitRow = 2u;
    constexpr std::size_t kRowCount = 3u;

    const std::vector<double> q = asDouble(fixture.q);
    const std::vector<double> v = asDouble(fixture.v);
    metalrobo::ArticulatedDynamicsConfig dynamicsConfig{};
    dynamicsConfig.gravity = {0.0, -9.81, 0.0};
    dynamicsConfig.timestep = fixture.timestepSeconds;

    metalrobo::ArticulatedPointQuery support = fp64SupportQuery(fixture);
    std::array<metalrobo::ArticulatedPointKinematics, 1u> points{};
    std::vector<double> pointJacobians(3u * fixture.model.world.nv, 0.0);
    const auto pointDiagnostics = metalrobo::computeArticulatedPointJacobians(
        fixture.model, 0u, q, v, std::span(&support, 1u), points,
        pointJacobians, dynamicsConfig
    );
    require(
        pointDiagnostics.succeeded(),
        "FP64 support-point Jacobian failed for simultaneous-triad reference"
    );

    SimultaneousTriadReference result{};
    result.initialContactGap = points.front().position[1u];
    require(
        std::abs(result.initialContactGap) <= 1.0e-7 &&
            std::abs(q[kDependentQ] -
                     static_cast<double>(fixture.model.dofs.at(kDependentV).limits.y)) <=
                1.0e-7,
        "simultaneous-triad reference did not begin at the support/upper-limit intersection"
    );
    const double rawContactTargetVelocity = std::max(
        0.0,
        -0.2 * std::min(result.initialContactGap, 0.0) /
            static_cast<double>(fixture.timestepSeconds)
    );
    result.targetVelocity[kContactNormalRow] = std::isfinite(contactTargetOverride)
        ? contactTargetOverride
        : rawContactTargetVelocity;
    const double equalityError = q[kDependentQ] -
        static_cast<double>(kEqualitySlope) * q[kMasterQ];
    result.targetVelocity[kEqualityRow] = std::clamp(
        -0.2 * equalityError / static_cast<double>(fixture.timestepSeconds),
        -4.0,
        4.0
    );
    result.targetVelocity[kUpperLimitRow] = std::min(
        0.0,
        std::max(
            -4.0,
            -0.2 * (q[kDependentQ] -
                    static_cast<double>(fixture.model.dofs.at(kDependentV).limits.y)) /
                static_cast<double>(fixture.timestepSeconds)
        )
    );

    const std::size_t nv = fixture.model.world.nv;
    std::vector<double> rows(kRowCount * nv, 0.0);
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        rows[kContactNormalRow * nv + dof] = pointJacobians[nv + dof];
        rows[kEqualityRow * nv + dof] =
            dof == kMasterV ? -static_cast<double>(kEqualitySlope) :
            dof == kDependentV ? 1.0 : 0.0;
        // An upper limit is a non-negative row in the outward (-qdot) direction.
        rows[kUpperLimitRow * nv + dof] =
            dof == kDependentV ? -1.0 : 0.0;
    }
    std::vector<double> responses(kRowCount * nv, 0.0);
    const auto responseDiagnostics =
        metalrobo::computeArticulatedInverseMassResponses(
            fixture.model, 0u, q, rows, responses, dynamicsConfig
        );
    require(
        responseDiagnostics.succeeded(),
        "FP64 inverse-mass triad responses failed"
    );

    const std::vector<double> freeAcceleration =
        freeReferenceAcceleration(fixture);
    result.freeVelocity = v;
    for (std::size_t dof = 0u; dof < nv; ++dof) {
        result.freeVelocity[dof] +=
            static_cast<double>(fixture.timestepSeconds) * freeAcceleration[dof];
    }

    std::array<std::array<double, 3u>, 3u> system{};
    std::array<double, 3u> rightHandSide{};
    for (std::size_t row = 0u; row < kRowCount; ++row) {
        for (std::size_t column = 0u; column < kRowCount; ++column) {
            double value = 0.0;
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                value += rows[row * nv + dof] *
                    responses[column * nv + dof];
            }
            result.physicalDelassus[row][column] = value;
            system[row][column] = value + (row == column ? kRegularization : 0.0);
        }
        rightHandSide[row] = result.targetVelocity[row];
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            rightHandSide[row] -=
                rows[row * nv + dof] * result.freeVelocity[dof];
        }
    }
    const DenseThreeSolve solve = solveDenseThreeByThree(system, rightHandSide);
    result.impulses = solve.solution;
    require(
        result.impulses[kContactNormalRow] >= -1.0e-12 &&
            result.impulses[kUpperLimitRow] >= -1.0e-12,
        "FP64 simultaneous-triad active set requires a pulling contact or limit impulse"
    );
    result.minimumAbsolutePivot = solve.minimumAbsolutePivot;
    result.constrainedVelocity = result.freeVelocity;
    for (std::size_t row = 0u; row < kRowCount; ++row) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            result.constrainedVelocity[dof] +=
                result.impulses[row] * responses[row * nv + dof];
        }
    }
    for (std::size_t row = 0u; row < kRowCount; ++row) {
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            result.constraintVelocity[row] +=
                rows[row * nv + dof] * result.constrainedVelocity[dof];
        }
        result.regularizedResidual[row] = result.constraintVelocity[row] -
            result.targetVelocity[row] + kRegularization * result.impulses[row];
        require(
            std::isfinite(result.regularizedResidual[row]) &&
                std::abs(result.regularizedResidual[row]) <= 2.0e-10,
            "FP64 simultaneous-triad regularized KKT residual is too large"
        );
    }
    return result;
}

void checkSimultaneousTriadReference() {
    Fixture fixture(kDefaultTimestepSeconds);
    fixture.contacts.front().frictionSlopAndStabilization.x = 0.0f;
    const SimultaneousTriadReference reference =
        simultaneousTriadReference(fixture);
    std::cout << "simultaneous_triad_initial_gap_m="
              << reference.initialContactGap
              << " simultaneous_triad_contact_impulse_ns="
              << reference.impulses[0u]
              << " simultaneous_triad_contact_target_velocity_m_s="
              << reference.targetVelocity[0u]
              << " simultaneous_triad_equality_impulse_ns="
              << reference.impulses[1u]
              << " simultaneous_triad_upper_limit_impulse_ns="
              << reference.impulses[2u]
              << " simultaneous_triad_contact_equality_cross_delassus="
              << reference.physicalDelassus[0u][1u]
              << " simultaneous_triad_contact_limit_cross_delassus="
              << reference.physicalDelassus[0u][2u]
              << " simultaneous_triad_equality_limit_cross_delassus="
              << reference.physicalDelassus[1u][2u]
              << " simultaneous_triad_minimum_pivot="
              << reference.minimumAbsolutePivot
              << " simultaneous_triad_max_regularized_kkt_residual="
              << std::max({
                     std::abs(reference.regularizedResidual[0u]),
                     std::abs(reference.regularizedResidual[1u]),
                     std::abs(reference.regularizedResidual[2u]),
                 }) << '\n';
    constexpr std::array<std::uint32_t, 3u> iterationCounts{{4u, 16u, 64u}};
    for (const std::uint32_t contactIterationCount : iterationCounts) {
        const Run metal = runHorizon(
            fixture, 1u, true, true, contactIterationCount
        );
        double maximumVelocityDifference = 0.0;
        for (std::size_t dof = 0u;
             dof < reference.constrainedVelocity.size();
             ++dof) {
            maximumVelocityDifference = std::max(
                maximumVelocityDifference,
                std::abs(reference.constrainedVelocity[dof] -
                         static_cast<double>(metal.result.standV[dof]))
            );
        }
        std::cout << "one_step_metal_to_simultaneous_fp64_velocity_difference_m_s="
                  << maximumVelocityDifference
                  << " coupled_iteration_count=" << contactIterationCount
                  << '\n';
        if (contactIterationCount == 16u) {
            require(
                maximumVelocityDifference <= 5.0e-4,
                "sixteen coupled sweeps did not approach the FP64 triad reference"
            );
        } else if (contactIterationCount == 64u) {
            require(
                maximumVelocityDifference <= 1.0e-5,
                "sixty-four coupled sweeps did not converge toward the FP64 triad reference"
            );
        }
    }
}

// Keep the one-step FP64 KKT comparison on the same timestep grid as the
// common-duration discriminator.  This does not claim a full plane-contact
// oracle or a long-horizon reference; it establishes whether the active
// normal-contact/equality/upper-limit triad itself remains well-conditioned
// and accurately solved at each candidate step size.
void checkSimultaneousTriadTimestepReference() {
    constexpr double kVelocityTolerance = 1.0e-5;
    constexpr std::array<float, 4u> timesteps{{
        100.0e-6f,
        50.0e-6f,
        25.0e-6f,
        12.5e-6f,
    }};
    bool referenceGatePassed = true;
    for (const float timestep : timesteps) {
        Fixture fixture(timestep);
        fixture.contacts.front().frictionSlopAndStabilization.x = 0.0f;
        const SimultaneousTriadReference reference =
            simultaneousTriadReference(fixture);
        const Run metal = runHorizon(fixture, 1u, true, true, 64u);
        double maximumVelocityDifference = 0.0;
        for (std::size_t dof = 0u;
             dof < reference.constrainedVelocity.size();
             ++dof) {
            maximumVelocityDifference = std::max(
                maximumVelocityDifference,
                std::abs(reference.constrainedVelocity[dof] -
                         static_cast<double>(metal.result.standV[dof]))
            );
        }
        require(std::isfinite(maximumVelocityDifference),
                "timestep-swept coupled solve produced a non-finite FP64 comparison");
        const bool passesReferenceGate =
            maximumVelocityDifference <= kVelocityTolerance;
        referenceGatePassed = referenceGatePassed && passesReferenceGate;
        std::cout << "simultaneous_triad_timestep_reference dt_us="
                  << static_cast<double>(timestep) * 1.0e6
                  << " coupled_iteration_count=64"
                  << " fp64_minimum_pivot="
                  << reference.minimumAbsolutePivot
                  << " fp64_velocity_tolerance_m_s=" << kVelocityTolerance
                  << " fp64_reference_gate="
                  << (passesReferenceGate ? "pass" : "fail")
                  << " metal_to_fp64_velocity_difference_m_s="
                  << maximumVelocityDifference << '\n';
    }
    std::cout << "simultaneous_triad_timestep_reference_gate="
              << (referenceGatePassed ? "pass" : "fail")
              << " scope=one_step_frictionless_normal_contact_equality_upper_limit"
              << " standing_qualified=false\n";
}

// The timestep sweep leaves the finest case just outside the one-step
// reference tolerance at the production 64-sweep budget. Hold all other
// inputs fixed and sweep only supported budgets before changing the coupled
// formulation. This is diagnostic evidence: it does not alter a runtime
// default or claim a full-contact reference.
void checkSimultaneousTriadFinestTimestepIterationConvergence() {
    constexpr float kFinestTimestep = 12.5e-6f;
    constexpr double kVelocityTolerance = 1.0e-5;
    constexpr std::array<std::uint32_t, 5u> iterationCounts{{
        1u, 4u, 16u, 32u, 64u,
    }};
    Fixture fixture(kFinestTimestep);
    fixture.contacts.front().frictionSlopAndStabilization.x = 0.0f;
    const SimultaneousTriadReference reference =
        simultaneousTriadReference(fixture);
    std::uint32_t firstPassingIterationCount = 0u;
    for (const std::uint32_t contactIterationCount : iterationCounts) {
        const Run metal = runHorizon(
            fixture, 1u, true, true, contactIterationCount
        );
        double maximumVelocityDifference = 0.0;
        for (std::size_t dof = 0u;
             dof < reference.constrainedVelocity.size();
             ++dof) {
            maximumVelocityDifference = std::max(
                maximumVelocityDifference,
                std::abs(reference.constrainedVelocity[dof] -
                         static_cast<double>(metal.result.standV[dof]))
            );
        }
        require(std::isfinite(maximumVelocityDifference),
                "finest-timestep coupled solve produced a non-finite FP64 comparison");
        const bool passesReferenceGate =
            maximumVelocityDifference <= kVelocityTolerance;
        if (passesReferenceGate && firstPassingIterationCount == 0u) {
            firstPassingIterationCount = contactIterationCount;
        }
        std::cout << "simultaneous_triad_finest_timestep_supported_iteration_reference"
                  << " dt_us=" << static_cast<double>(kFinestTimestep) * 1.0e6
                  << " coupled_iteration_count=" << contactIterationCount
                  << " fp64_minimum_pivot=" << reference.minimumAbsolutePivot
                  << " fp64_velocity_tolerance_m_s=" << kVelocityTolerance
                  << " fp64_reference_gate="
                  << (passesReferenceGate ? "pass" : "fail")
                  << " metal_to_fp64_velocity_difference_m_s="
                  << maximumVelocityDifference << '\n';
    }
    std::cout << "simultaneous_triad_finest_timestep_supported_iteration_reference_gate="
              << (firstPassingIterationCount == 0u ? "fail" : "pass")
              << " first_passing_coupled_iteration_count="
              << firstPassingIterationCount
              << " scope=one_step_frictionless_normal_contact_equality_upper_limit"
              << " device_abi_max_coupled_iteration_count=64"
              << " runtime_default_unchanged=true"
              << " standing_qualified=false\n";
}

// The final equality-coordinate assignment is deliberately outside the
// coupled sweeps. Record which of the same pre-step rows it leaves violated
// in the smallest frictionless production triad before changing formulation
// or numerical policy. This is an observation-only discriminator.
void checkPostProjectionPreStepConstraintDiagnostics() {
    constexpr float kFinestTimestep = 12.5e-6f;
    struct Variant {
        const char* name = "";
        bool contact = false;
        bool equality = false;
        bool dependentLimitActive = true;
    };
    constexpr std::array<Variant, 4u> variants{{
        {"full_triad", true, true, true},
        {"no_contact", false, true, true},
        {"inactive_dependent_limit", true, true, false},
        {"no_equality", true, false, true},
    }};
    for (const Variant& variant : variants) {
        Fixture fixture(kFinestTimestep);
        fixture.contacts.front().frictionSlopAndStabilization.x = 0.0f;
        if (!variant.dependentLimitActive) {
            fixture.moveOffDependentPositionLimit();
        }
        const Run metal = runHorizon(
            fixture, 1u, variant.contact, variant.equality, 64u
        );
        const auto& preProjection = metal.result.standStatuses.front()
            .preProjectionPreStepConstraintDiagnostics;
        const auto& postProjection = metal.result.standStatuses.front()
            .postProjectionPreStepConstraintDiagnostics;
        require(
            std::isfinite(preProjection.x) && std::isfinite(preProjection.y) &&
                std::isfinite(preProjection.z) && std::isfinite(preProjection.w) &&
                preProjection.w >= std::max({preProjection.x, preProjection.y,
                                              preProjection.z}) &&
                std::isfinite(postProjection.x) && std::isfinite(postProjection.y) &&
                std::isfinite(postProjection.z) && std::isfinite(postProjection.w) &&
                postProjection.w >= std::max({postProjection.x, postProjection.y,
                                               postProjection.z}),
            "post-projection pre-step constraint diagnostic is invalid"
        );
        if (!variant.equality) {
            require(
                preProjection.x == 0.0f && preProjection.y == 0.0f &&
                    preProjection.z == 0.0f && preProjection.w == 0.0f &&
                    postProjection.x == 0.0f && postProjection.y == 0.0f &&
                    postProjection.z == 0.0f && postProjection.w == 0.0f,
                "no-equality control unexpectedly recorded a final equality projection residual"
            );
        }
        std::cout << "simultaneous_triad_postprojection_prestep_constraint_residual"
                  << " variant=" << variant.name
                  << " dt_us=" << static_cast<double>(kFinestTimestep) * 1.0e6
                  << " coupled_iteration_count=64"
                  << " pre_projection_normal_contact_target_velocity_violation_m_s="
                  << preProjection.x
                  << " post_projection_normal_contact_target_velocity_violation_m_s="
                  << postProjection.x
                  << " pre_projection_source_limit_target_velocity_violation_m_s_or_rad_s="
                  << preProjection.y
                  << " post_projection_source_limit_target_velocity_violation_m_s_or_rad_s="
                  << postProjection.y
                  << " pre_projection_equality_target_velocity_residual_m_s_or_rad_s="
                  << preProjection.z
                  << " post_projection_equality_target_velocity_residual_m_s_or_rad_s="
                  << postProjection.z
                  << " pre_projection_maximum_target_velocity_residual_m_s_or_rad_s="
                  << preProjection.w
                  << " post_projection_maximum_target_velocity_residual_m_s_or_rad_s="
                  << postProjection.w
                  << " scope=one_step_frictionless_normal_contact_equality_upper_limit"
                  << " standing_qualified=false\n";
    }
}

// At the exact support surface, the FP64 oracle widens the same source point
// record consumed by Metal. The constructed coincident witness and its contact
// target must therefore agree exactly; free and equality-only controls retain
// the surrounding arithmetic discriminator.
void checkExactContactPrecisionDiagnostic() {
    constexpr float kFinestTimestep = 12.5e-6f;
    Fixture fixture(kFinestTimestep);
    fixture.contacts.front().frictionSlopAndStabilization.x = 0.0f;
    const SimultaneousTriadReference reference =
        simultaneousTriadReference(fixture);
    std::vector<double> equalityOnlyReferenceVelocity = asDouble(fixture.v);
    const std::vector<double> equalityOnlyReferenceAcceleration =
        constrainedReferenceAcceleration(fixture);
    for (std::size_t dof = 0u;
         dof < equalityOnlyReferenceVelocity.size();
         ++dof) {
        equalityOnlyReferenceVelocity[dof] +=
            static_cast<double>(kFinestTimestep) *
            equalityOnlyReferenceAcceleration[dof];
    }
    const Run free = runHorizon(fixture, 1u, false, false, 64u);
    const Run equalityOnly = runHorizon(fixture, 1u, false, true, 64u);
    const Run projected = runHorizon(fixture, 1u, true, true, 64u);
    const MRNumiHumanStandStatusGPU& projectedStatus =
        projected.result.standStatuses.front();
    const VelocityComparison freeComparison = compareVelocities(
        reference.freeVelocity, free.result.standV
    );
    const VelocityComparison equalityOnlyComparison = compareVelocities(
        equalityOnlyReferenceVelocity, equalityOnly.result.standV
    );
    const VelocityComparison projectedComparison = compareVelocities(
        reference.constrainedVelocity, projected.result.standV
    );
    const double gpuMinimumGap = static_cast<double>(
        projectedStatus.contactAndAcceleration.x
    );
    const double contactStabilization = static_cast<double>(
        fixture.contacts.front().frictionSlopAndStabilization.z
    );
    const double gpuContactTargetVelocity = std::max(
        0.0,
        -contactStabilization * std::min(gpuMinimumGap, 0.0) /
            static_cast<double>(kFinestTimestep)
    );
    // Retain an independently supplied-target reference after source geometry
    // agreement. It should now receive the same zero target, while continuing
    // to isolate any later solver-path remainder.
    const SimultaneousTriadReference metalTargetReference =
        simultaneousTriadReference(fixture, gpuContactTargetVelocity);
    const VelocityComparison targetMatchedComparison = compareVelocities(
        metalTargetReference.constrainedVelocity, projected.result.standV
    );
    const double contactTargetVelocityDifference = std::abs(
        reference.targetVelocity[0u] - gpuContactTargetVelocity
    );
    const float projectedEqualityVelocityResidual = std::abs(
        projected.result.standV[kDependentV] -
        kEqualitySlope * projected.result.standV[kMasterV]
    );
    require(
        std::isfinite(freeComparison.maximumDifference) &&
            std::isfinite(equalityOnlyComparison.maximumDifference) &&
            std::isfinite(projectedComparison.maximumDifference) &&
            std::isfinite(targetMatchedComparison.maximumDifference) &&
            std::isfinite(gpuMinimumGap) &&
            std::isfinite(gpuContactTargetVelocity) &&
            std::isfinite(contactTargetVelocityDifference) &&
            std::isfinite(projectedEqualityVelocityResidual),
        "exact-contact precision diagnostic produced a non-finite comparison"
    );
    require(
        freeComparison.maximumDifference <= 1.0e-8 &&
            equalityOnlyComparison.maximumDifference <= 1.0e-8,
        "exact-contact precision diagnostic cannot isolate the contact path"
    );
    require(
        reference.initialContactGap == 0.0 && gpuMinimumGap == 0.0 &&
            reference.targetVelocity[0u] == 0.0 &&
            gpuContactTargetVelocity == 0.0 &&
            contactTargetVelocityDifference == 0.0,
        "source-identical support witness changed its exact contact target"
    );
    std::cout << "exact_contact_precision_diagnostic"
              << " dt_us=" << static_cast<double>(kFinestTimestep) * 1.0e6
              << " fp64_initial_contact_gap_m=" << reference.initialContactGap
              << " metal_initial_contact_gap_m=" << gpuMinimumGap
              << " fp64_contact_target_velocity_m_s="
              << reference.targetVelocity[0u]
              << " metal_contact_target_velocity_m_s="
              << gpuContactTargetVelocity
              << " contact_target_velocity_difference_m_s="
              << contactTargetVelocityDifference
              << " free_metal_to_fp64_velocity_difference_m_s="
              << freeComparison.maximumDifference
              << " equality_only_metal_to_fp64_velocity_difference_m_s="
              << equalityOnlyComparison.maximumDifference
              << " triad_metal_to_fp64_velocity_difference_m_s="
              << projectedComparison.maximumDifference
              << " triad_metal_to_fp64_matched_contact_target_velocity_difference_m_s="
              << targetMatchedComparison.maximumDifference
              << " triad_worst_dof=" << projectedComparison.dof
              << " triad_fp64_velocity_m_s="
              << projectedComparison.referenceVelocity
              << " triad_metal_velocity_m_s="
              << projectedComparison.observedVelocity
              << " triad_post_limit_residual_m_s="
              << projectedStatus.postProjectionPreStepConstraintDiagnostics.y
              << " triad_post_equality_residual_m_s="
              << projectedStatus.postProjectionPreStepConstraintDiagnostics.z
              << " triad_equality_velocity_residual_m_s="
              << projectedEqualityVelocityResidual
              << " scope=diagnostic_not_production_policy"
              << " standing_qualified=false\n";
}

// The simultaneous KKT reference establishes the ideal coupled-row target,
// while this second reference follows the production contact/equality/limit
// order and its final equality overwrite.  Keeping both references on the
// same source-authored contact target separates finite-sweep
// ordering/projection from the remaining FP32 path difference. It does not
// choose a corrective runtime formulation.
void checkProductionOrderTriadReference() {
    constexpr float kFinestTimestep = 12.5e-6f;
    constexpr std::uint32_t kCoupledSweepCount = 64u;
    Fixture fixture(kFinestTimestep);
    fixture.contacts.front().frictionSlopAndStabilization.x = 0.0f;
    const Run metal = runHorizon(
        fixture, 1u, true, true, kCoupledSweepCount
    );
    const MRNumiHumanStandStatusGPU& status = metal.result.standStatuses.front();
    const double metalGap = static_cast<double>(status.contactAndAcceleration.x);
    const double contactStabilization = static_cast<double>(
        fixture.contacts.front().frictionSlopAndStabilization.z
    );
    const double metalContactTarget = std::max(
        0.0,
        -contactStabilization * std::min(metalGap, 0.0) /
            static_cast<double>(kFinestTimestep)
    );
    const ProductionOrderTriadReference fp64GeometryOrder =
        productionOrderTriadReference(fixture, kCoupledSweepCount);
    const ProductionOrderTriadReference metalTargetOrder =
        productionOrderTriadReference(
            fixture, kCoupledSweepCount, metalContactTarget
        );
    const SimultaneousTriadReference metalTargetKkt =
        simultaneousTriadReference(fixture, metalContactTarget);
    const VelocityComparison metalToProductionOrder = compareVelocities(
        metalTargetOrder.postProjectionVelocity, metal.result.standV
    );
    const VelocityComparison productionOrderToKkt = compareVelocities(
        metalTargetKkt.constrainedVelocity,
        metalTargetOrder.postProjectionVelocity
    );
    const VelocityComparison preProjectionOrderToKkt = compareVelocities(
        metalTargetKkt.constrainedVelocity,
        metalTargetOrder.preProjectionVelocity
    );
    const VelocityComparison geometryTargetToMetalTargetOrder = compareVelocities(
        fp64GeometryOrder.postProjectionVelocity,
        metalTargetOrder.postProjectionVelocity
    );
    require(
        std::isfinite(metalGap) && std::isfinite(metalContactTarget) &&
            std::isfinite(metalToProductionOrder.maximumDifference) &&
            std::isfinite(productionOrderToKkt.maximumDifference) &&
            std::isfinite(preProjectionOrderToKkt.maximumDifference) &&
            std::isfinite(geometryTargetToMetalTargetOrder.maximumDifference) &&
            metalTargetOrder.upperLimitAdmitted,
        "production-order triad discriminator produced an invalid comparison"
    );
    require(
        metalToProductionOrder.maximumDifference <= 1.0e-8,
        "Metal path no longer matches the FP64 replay of its production ordering"
    );
    std::cout << "production_order_triad_reference"
              << " dt_us=" << static_cast<double>(kFinestTimestep) * 1.0e6
              << " coupled_iteration_count=" << kCoupledSweepCount
              << " metal_contact_target_velocity_m_s=" << metalContactTarget
              << " fp64_geometry_contact_target_velocity_m_s="
              << fp64GeometryOrder.targetVelocity[0u]
              << " fp64_production_order_pre_contact_residual_m_s="
              << metalTargetOrder.preProjectionTargetResidual[0u]
              << " fp64_production_order_pre_limit_residual_m_s="
              << metalTargetOrder.preProjectionTargetResidual[2u]
              << " fp64_production_order_pre_equality_residual_m_s="
              << metalTargetOrder.preProjectionTargetResidual[1u]
              << " fp64_production_order_post_contact_residual_m_s="
              << metalTargetOrder.postProjectionTargetResidual[0u]
              << " fp64_production_order_post_limit_residual_m_s="
              << metalTargetOrder.postProjectionTargetResidual[2u]
              << " fp64_production_order_post_equality_residual_m_s="
              << metalTargetOrder.postProjectionTargetResidual[1u]
              << " metal_post_limit_residual_m_s="
              << status.postProjectionPreStepConstraintDiagnostics.y
              << " metal_post_equality_residual_m_s="
              << status.postProjectionPreStepConstraintDiagnostics.z
              << " metal_to_fp64_production_order_velocity_difference_m_s="
              << metalToProductionOrder.maximumDifference
              << " metal_to_fp64_production_order_worst_dof="
              << metalToProductionOrder.dof
              << " fp64_production_order_to_simultaneous_kkt_velocity_difference_m_s="
              << productionOrderToKkt.maximumDifference
              << " fp64_pre_projection_order_to_simultaneous_kkt_velocity_difference_m_s="
              << preProjectionOrderToKkt.maximumDifference
              << " fp64_geometry_to_metal_target_production_order_velocity_difference_m_s="
              << geometryTargetToMetalTargetOrder.maximumDifference
              << " scope=diagnostic_not_production_policy"
              << " standing_qualified=false\n";
}

void checkSourceDerivative(const Fixture& fixture) {
    const std::vector<double> v = asDouble(fixture.v);
    std::vector<double> q = asDouble(fixture.q);
    const MujocoMuscleResult baseline = fixture.sourceAt(q, v);
    constexpr double h = 1.0e-6;
    auto sample = [&](const double master) {
        std::vector<double> candidate = q;
        candidate[kMasterQ] = master;
        candidate[kDependentQ] = static_cast<double>(kEqualitySlope) * master;
        return fixture.sourceAt(candidate, v).path.length;
    };
    const double finiteDifference =
        (sample(q[kMasterQ] + h) - sample(q[kMasterQ] - h)) / (2.0 * h);
    const double tangent = baseline.path.lengthJacobian[kMasterV] +
        static_cast<double>(kEqualitySlope) *
            baseline.path.lengthJacobian[kDependentV];
    require(std::isfinite(finiteDifference) && std::isfinite(tangent) &&
                std::abs(finiteDifference - tangent) <= 2.0e-6,
            "FP64 MyoSim/equality tangent finite difference disagrees");
    std::cout << "fp64_route_length_m=" << baseline.path.length
              << " fp64_route_tangent=" << tangent
              << " fp64_route_fd=" << finiteDifference << '\n';
}

void checkOneStepReference(const Fixture& fixture) {
    checkSourceDerivative(fixture);
    const MujocoMuscleResult source = fixture.sourceAt(
        asDouble(fixture.q), asDouble(fixture.v)
    );
    const MujocoCompliantMuscleResult compliant =
        fixture.compliantAtInitialState();
    const Run metal = runHorizon(fixture, 1u, false, true);
    require(metal.result.mujocoResults.size() == 1u &&
                metal.result.mujocoActivationStates.size() == 1u &&
                metal.result.standTendonTransfers.size() == 2u,
            "Metal MyoSim/tendon result coverage is incomplete");
    const MRMujocoMuscleResultGPU& gpu = metal.result.mujocoResults.front();
    require(gpu.status == MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS &&
                std::abs(static_cast<double>(gpu.pathForceAndActivationDerivative.x) -
                         source.path.length) <= 2.0e-5 &&
                std::abs(static_cast<double>(gpu.fiberStateTendonForceResidual.x) -
                         compliant.candidateFiberLength) <= 2.0e-4,
            "Metal MyoSim result disagrees with its FP64 source/fibre reference");
    for (const MRNumiHumanTendonTransferResultGPU& transfer :
         metal.result.standTendonTransfers) {
        require(transfer.status == MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS,
                "Metal source-point tendon transfer failed");
    }

    const std::vector<double> reference = constrainedReferenceAcceleration(fixture);
    double maximumAccelerationDifference = 0.0;
    for (std::size_t dof = 0u; dof < reference.size(); ++dof) {
        const double observed = static_cast<double>(metal.result.standV[dof]) /
            static_cast<double>(fixture.timestepSeconds);
        maximumAccelerationDifference = std::max(
            maximumAccelerationDifference, std::abs(observed - reference[dof])
        );
    }
    const auto& status = metal.result.standStatuses.front();
    const double velocityEqualityError = std::abs(
        static_cast<double>(metal.result.standV[kDependentV]) -
        static_cast<double>(kEqualitySlope) * metal.result.standV[kMasterV]
    );
    const double positionEqualityError = std::abs(
        static_cast<double>(metal.result.standQ[kDependentQ]) -
        static_cast<double>(kEqualitySlope) * metal.result.standQ[kMasterQ]
    );
    require(velocityEqualityError <= 2.0e-6 && positionEqualityError <= 2.0e-6 &&
                status.jointEqualityCounts.x == 1u &&
                status.jointEqualityCounts.z == 0u,
            "Metal equality projection did not retain the source manifold");
    require(maximumAccelerationDifference <= 2.0e-4,
            "Metal no-contact acceleration disagrees with the FP64 source/equality reference");
    std::cout << "one_step_fp64_to_metal_max_acceleration_difference_m_s2="
              << maximumAccelerationDifference
              << " equality_velocity_error_m_s=" << velocityEqualityError
              << " equality_position_error_m=" << positionEqualityError
              << " tendon_force_residual_n=" << status.tendonDiagnostics.x
              << " tendon_moment_residual_nm=" << status.tendonDiagnostics.y
              << '\n';
}

void checkContactAndReplay(const Fixture& fixture) {
    const Run first = runHorizon(fixture, 8u, true, true);
    const Run replay = runHorizon(fixture, 8u, true, true);
    require(sameBytes(first.result.standQ, replay.result.standQ) &&
                sameBytes(first.result.standV, replay.result.standV) &&
                sameBytes(first.result.standStatuses, replay.result.standStatuses) &&
                sameBytes(first.result.mujocoActivationStates,
                          replay.result.mujocoActivationStates),
            "minimal contact/equality horizon is not bitwise replayable");
    const MRNumiHumanStandStatusGPU& status = first.result.standStatuses.front();
    require(status.maximumActiveContactCount >= 1u &&
                status.contactAndAcceleration.y <= 1.0e-5f &&
                status.tendonTransferCount == 16u &&
                status.jointEqualityCounts.x == 1u &&
                status.jointEqualityCounts.z == 0u,
            "minimal production contact/tendon/equality path was not exercised");
    std::cout << "contact_replay=bitwise active_contacts="
              << status.maximumActiveContactCount
              << " max_penetration_m=" << status.contactAndAcceleration.y
              << " peak_preprojection_acceleration_m_s2="
              << status.contactAndAcceleration.w
              << " equality_abs_impulse=" << status.jointEqualityDiagnostics.z
              << " equality_total_abs_impulse=" << status.jointEqualityDiagnostics.w
              << '\n';
}

// Exercise the terminal coordinate owner with a deliberately off-manifold
// state. The coupled block can legitimately arrive at the terminal projection
// with a zero correction, so non-zero diagnostic magnitude is proven here
// rather than inferred from a contact/limit interaction.
void checkFinalEqualityProjection() {
    Fixture fixture(12.5e-6f);
    fixture.moveOffDependentPositionLimit();
    fixture.q[kDependentQ] += 1.0e-4f;
    const Run run = runHorizon(fixture, 1u, false, true, 64u);
    const MRNumiHumanStandStatusGPU& status =
        run.result.standStatuses.front();
    const double velocityError = std::abs(
        static_cast<double>(run.result.standV[kDependentV]) -
        static_cast<double>(kEqualitySlope) *
            run.result.standV[kMasterV]
    );
    const double positionError = std::abs(
        static_cast<double>(run.result.standQ[kDependentQ]) -
        static_cast<double>(kEqualitySlope) *
            run.result.standQ[kMasterQ]
    );
    require(
        velocityError <= 2.0e-6 && positionError <= 2.0e-6 &&
            status.jointEqualityProjectionDiagnostics.x > 1.0e-6f &&
            status.jointEqualityProjectionDiagnostics.y >=
                status.jointEqualityProjectionDiagnostics.x &&
            status.jointEqualityProjectionDiagnostics.z > 1.0e-6f &&
            status.jointEqualityProjectionDiagnostics.w >=
                status.jointEqualityProjectionDiagnostics.z,
        "off-manifold control omitted its terminal equality projection"
    );
    std::cout << "terminal_equality_projection=pass"
              << " position_projection_m_or_rad="
              << status.jointEqualityProjectionDiagnostics.x
              << " velocity_projection_m_s_or_rad_s="
              << status.jointEqualityProjectionDiagnostics.z
              << " position_residual_m_or_rad=" << positionError
              << " velocity_residual_m_s_or_rad_s=" << velocityError
              << '\n';
}

struct RefinementSummary {
    float timestepSeconds = 0.0f;
    std::uint32_t steps = 0u;
    std::vector<float> q;
    std::vector<float> v;
    MRNumiHumanStandStatusGPU status{};
};

void checkCommonDurationRefinement() {
    constexpr std::array<std::pair<float, std::uint32_t>, 4u> cases{{
        {100.0e-6f, 8u},
        {50.0e-6f, 16u},
        {25.0e-6f, 32u},
        {12.5e-6f, 64u},
    }};
    struct Variant {
        const char* name = "";
        bool contact = false;
        bool equality = false;
        bool passivePreload = false;
        bool dependentLimitActive = true;
        float contactFriction = 0.70f;
    };
    // The dependent coordinate begins at its source upper limit.  Keep an
    // explicit inactive-limit control so a later limit projection is not
    // silently attributed to the contact/equality pair.
    constexpr std::array<Variant, 6u> variants{{
        {"full", true, true, true, true},
        {"no_contact", false, true, true, true},
        {"no_equality", true, false, true, true},
        {"no_passive_preload", true, true, false, true},
        {"inactive_dependent_limit", true, true, true, false},
        {"frictionless_contact", true, true, true, true, 0.0f},
    }};
    for (const Variant& variant : variants) {
        std::vector<RefinementSummary> summaries;
        summaries.reserve(cases.size());
        for (const auto [timestep, steps] : cases) {
            Fixture fixture(timestep);
            if (!variant.dependentLimitActive) {
                fixture.moveOffDependentPositionLimit();
            }
            fixture.contacts.front().frictionSlopAndStabilization.x =
                variant.contactFriction;
            if (!variant.passivePreload) {
                std::fill(
                    fixture.passivePreload.begin(), fixture.passivePreload.end(), 0.0f
                );
            }
            const Run run = runHorizon(
                fixture, steps, variant.contact, variant.equality
            );
            const MRNumiHumanStandStatusGPU& status =
                run.result.standStatuses.front();
            require(
                (!variant.contact ||
                    (status.maximumActiveContactCount >= 1u &&
                     status.contactAndAcceleration.y <= 1.0e-5f)) &&
                    (!variant.equality || status.jointEqualityCounts.z == 0u) &&
                    (variant.equality || status.jointEqualityCounts.x == 0u),
                "common-duration discriminator did not retain its selected coupling"
            );
            if (variant.equality) {
                const double velocityEqualityError = std::abs(
                    static_cast<double>(run.result.standV[kDependentV]) -
                    static_cast<double>(kEqualitySlope) *
                        run.result.standV[kMasterV]
                );
                const double positionEqualityError = std::abs(
                    static_cast<double>(run.result.standQ[kDependentQ]) -
                    static_cast<double>(kEqualitySlope) *
                        run.result.standQ[kMasterQ]
                );
                require(
                    velocityEqualityError <= 2.0e-6 &&
                        positionEqualityError <= 2.0e-6 &&
                        std::isfinite(
                            status.jointEqualityProjectionDiagnostics.x) &&
                        std::isfinite(
                            status.jointEqualityProjectionDiagnostics.y) &&
                        std::isfinite(
                            status.jointEqualityProjectionDiagnostics.z) &&
                        std::isfinite(
                            status.jointEqualityProjectionDiagnostics.w),
                    "common-duration equality path left the source manifold"
                );
            } else {
                require(
                    status.jointEqualityProjectionDiagnostics.x == 0.0f &&
                        status.jointEqualityProjectionDiagnostics.y == 0.0f &&
                        status.jointEqualityProjectionDiagnostics.z == 0.0f &&
                        status.jointEqualityProjectionDiagnostics.w == 0.0f,
                    "uncoupled control unexpectedly used final equality projection"
                );
            }
            const bool coupledTriad = variant.contact && variant.equality &&
                variant.dependentLimitActive;
            if (coupledTriad) {
                require(
                    status.constraintImpulseDiagnostics.x > 0.0f &&
                        status.constraintImpulseDiagnostics.z > 0.0f &&
                        status.constraintImpulseOwners.x == 0u &&
                        status.constraintImpulseOwners.z == kDependentV &&
                        status.constraintImpulseOwners.w == 0u,
                    "coupled contact/equality/limit path did not retain its impulse owners"
                );
            }
            summaries.push_back({
                .timestepSeconds = timestep,
                .steps = steps,
                .q = run.result.standQ,
                .v = run.result.standV,
                .status = status,
            });
        }
        const RefinementSummary& finest = summaries.back();
        if (variant.contact && variant.equality && variant.passivePreload &&
            variant.dependentLimitActive && variant.contactFriction > 0.0f) {
            const RefinementSummary& coarsest = summaries.front();
            require(
                maximumDifference(coarsest.q, finest.q) <= 1.0e-5 &&
                    maximumDifference(coarsest.v, finest.v) <= 5.0e-4,
                "coupled contact/equality/limit refinement did not approach its fine state"
            );
        }
        for (const RefinementSummary& summary : summaries) {
            std::cout << "refinement_variant=" << variant.name
                      << " dt_us="
                      << static_cast<double>(summary.timestepSeconds) * 1.0e6
                      << " steps=" << summary.steps
                      << " max_penetration_m="
                      << summary.status.contactAndAcceleration.y
                      << " peak_preprojection_acceleration_m_s2="
                      << summary.status.contactAndAcceleration.w
                      << " tendon_force_residual_n="
                      << summary.status.tendonDiagnostics.x
                      << " equality_abs_impulse="
                      << summary.status.jointEqualityDiagnostics.z
                      << " equality_impulse_index="
                      << summary.status.constraintImpulseOwners.w
                      << " max_normal_contact_impulse_ns="
                      << summary.status.constraintImpulseDiagnostics.x
                      << " max_normal_contact_impulse_index="
                      << summary.status.constraintImpulseOwners.x
                      << " max_tangential_contact_impulse_ns="
                      << summary.status.constraintImpulseDiagnostics.y
                      << " max_tangential_contact_impulse_index="
                      << summary.status.constraintImpulseOwners.y
                      << " max_source_limit_impulse_ns_or_nms="
                      << summary.status.constraintImpulseDiagnostics.z
                      << " total_source_limit_abs_impulse_ns_or_nms="
                      << summary.status.constraintImpulseDiagnostics.w
                      << " max_source_limit_impulse_dof="
                      << summary.status.constraintImpulseOwners.z
                      << " equality_position_projection_max_m_or_rad="
                      << summary.status.jointEqualityProjectionDiagnostics.x
                      << " equality_position_projection_total_m_or_rad="
                      << summary.status.jointEqualityProjectionDiagnostics.y
                      << " equality_velocity_projection_max_m_s_or_rad_s="
                      << summary.status.jointEqualityProjectionDiagnostics.z
                      << " equality_velocity_projection_total_m_s_or_rad_s="
                      << summary.status.jointEqualityProjectionDiagnostics.w
                      << " endpoint_q_difference_to_12p5us_m="
                      << maximumDifference(summary.q, finest.q)
                      << " endpoint_v_difference_to_12p5us_m_s="
                      << maximumDifference(summary.v, finest.v)
                      << '\n';
        }
    }
}

} // namespace

int main() {
    try {
        std::cout << std::setprecision(17);
        const Fixture fixture(kDefaultTimestepSeconds);
        checkOneStepReference(fixture);
        checkSplitAuthoritativeHorizon(fixture);
        checkSimultaneousTriadReference();
        checkSimultaneousTriadTimestepReference();
        checkSimultaneousTriadFinestTimestepIterationConvergence();
        checkPostProjectionPreStepConstraintDiagnostics();
        checkExactContactPrecisionDiagnostic();
        checkProductionOrderTriadReference();
        checkContactAndReplay(fixture);
        checkFinalEqualityProjection();
        checkCommonDurationRefinement();
        std::cout << "numi_human_stand_coupling_probe=passed "
                  << "scope=minimal_production_path "
                  << "standing_qualified=false "
                  << "simultaneous_fp64_normal_triad_reference=true "
                  << "full_fp64_plane_contact_oracle=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numi_human_stand_coupling_probe: " << error.what() << '\n';
        return 1;
    }
}
