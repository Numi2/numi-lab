#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {
void require(const bool value, const std::string_view message) {
    if (value) return;
    std::cerr << "numi_human_static_support_test=failed error=\""
              << message << "\"\n";
    std::exit(1);
}
bool near(const double a, const double b, const double tolerance = 1.0e-7) {
    return std::abs(a - b) <= tolerance;
}
mr_float4 f4(
    const double x,
    const double y,
    const double z,
    const double w = 0.0
) {
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z),
        static_cast<float>(w),
    };
}

MRDofPropertiesGPU rootDof(const std::uint32_t localDof) {
    MRDofPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.jointIndex = MR_INVALID_INDEX;
    result.qIndex = localDof < 3u
        ? localDof
        : MR_INVALID_INDEX;
    result.vIndex = localDof;
    result.localDof = localDof;
    result.flags = MR_DOF_FLAG_ROOT;
    return result;
}

MRBodyPropertiesGPU body(
    const std::uint32_t parent,
    const std::uint32_t inboundJoint,
    const double mass,
    const std::array<double, 3> inertia
) {
    MRBodyPropertiesGPU result{};
    result.articulationIndex = 0u;
    result.parentBody = parent;
    result.inboundJoint = inboundJoint;
    result.motionType = MR_MOTION_DYNAMIC;
    result.massAndInverseMass = f4(mass, 1.0 / mass, 0.0, 0.0);
    result.inertiaRow0 = f4(inertia[0], 0.0, 0.0);
    result.inertiaRow1 = f4(0.0, inertia[1], 0.0);
    result.inertiaRow2 = f4(0.0, 0.0, inertia[2]);
    result.inverseInertiaRow0 =
        f4(1.0 / inertia[0], 0.0, 0.0);
    result.inverseInertiaRow1 =
        f4(0.0, 1.0 / inertia[1], 0.0);
    result.inverseInertiaRow2 =
        f4(0.0, 0.0, 1.0 / inertia[2]);
    result.dampingAndSpeedLimits =
        f4(0.0, 0.0, 1.0e6, 1.0e6);
    return result;
}

metalrobo::EngineModel makeFreeBodyModel() {
    metalrobo::EngineModel model;
    model.name = "analytic_contact_free_body";
    MRArticulationGPU articulation{};
    articulation.rootBody = 0u;
    articulation.rootType = MR_ROOT_FLOATING;
    articulation.firstBody = 0u;
    articulation.bodyCount = 1u;
    articulation.firstJoint = 0u;
    articulation.jointCount = 0u;
    articulation.qOffset = 0u;
    articulation.nq = 7u;
    articulation.vOffset = 0u;
    articulation.nv = 6u;
    model.articulations.push_back(articulation);
    for (std::uint32_t localDof = 0u;
         localDof < 6u;
         ++localDof) {
        model.dofs.push_back(rootDof(localDof));
    }
    model.bodies.push_back(body(
        MR_INVALID_INDEX,
        MR_INVALID_INDEX,
        2.5,
        {0.7, 1.1, 1.6}
    ));
    model.defaultQ = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    model.defaultV.assign(6u, 0.0f);
    return model;
}


struct Fixture {
    metalrobo::EngineModel model = makeFreeBodyModel();
    std::vector<double> q;
    const std::vector<metalrobo::MujocoMuscleSite> sites{
        {0u, {-0.5, 0.0, 0.0}}, {0u, {0.5, 0.0, 0.0}}};
    std::vector<metalrobo::MujocoMuscleDefinition> muscles;
    const std::vector<metalrobo::MujocoCompliantMuscleArchitecture> architectures{1u};
    metalrobo::NumiHumanMuscleEquilibriumConfig config;

    Fixture() {
        model.world.gravityAndTimestep = {0.0f, 0.0f, -9.81f, 0.0001f};
        q.assign(model.defaultQ.begin(), model.defaultQ.end());
        metalrobo::MujocoMuscleDefinition muscle;
        muscle.route = {
            {metalrobo::MujocoRouteNodeType::site, 0u},
            {metalrobo::MujocoRouteNodeType::site, 1u}};
        muscle.lengthRange = {0.8, 1.2};
        muscle.accelerationScale = 1.0;
        muscle.controlRange = {0.0, 1.0};
        muscle.gainParameters = {0.75, 1.05, 1.0, 200.0, 0.5, 1.6,
                                 1.5, 1.3, 1.2, 0.0};
        muscle.biasParameters = muscle.gainParameters;
        muscle.dynamicParameters = {0.01, 0.04, 0.0};
        muscles.push_back(muscle);
        config.activationSamples = 3u;
        config.activationSweeps = 8u;
        config.poseSweeps = 0u;
        config.globalActivationPolishIterations = 2u;
    }

    auto compile(const std::vector<metalrobo::NumiHumanStaticSupportContact>& supports,
                 metalrobo::NumiHumanMuscleEquilibriumResult& result) const {
        return metalrobo::compileNumiHumanMuscleEquilibrium(
            model, 0u, q, sites, {}, muscles, architectures, {}, {}, supports,
            result, config);
    }
    double weight() const {
        return -2.5 * static_cast<double>(model.world.gravityAndTimestep.z);
    }
};
} // namespace

int main() {
    using namespace metalrobo;
    Fixture fixture;
    const NumiHumanStaticSupportContact touching{.bodyIndex = 0u};
    auto separated = touching;
    separated.localPoint[2] = 0.02;
    NumiHumanMuscleEquilibriumResult accepted;
    const auto grounded = fixture.compile({touching, separated}, accepted);
    require(grounded.succeeded() && grounded.balanced,
            "analytic grounded mass did not balance");
    require(accepted.supportNormalForce.size() == 2u &&
                accepted.supportPlaneGapMeters.size() == 2u &&
                near(accepted.supportNormalForce[0], fixture.weight()) &&
                accepted.supportNormalForce[1] == 0.0 &&
                near(accepted.supportPlaneGapMeters[0], 0.0) &&
                near(accepted.supportPlaneGapMeters[1], 0.02),
            "separated witness carried weight or lost its source index");
    require(near(accepted.generalizedSupportForce[2], fixture.weight()),
            "support generalized force disagrees with analytic weight");

    NumiHumanMuscleEquilibriumResult replay;
    require(fixture.compile({touching, separated}, replay).succeeded() &&
                replay.q == accepted.q && replay.activation == accepted.activation &&
                replay.supportNormalForce == accepted.supportNormalForce &&
                replay.supportPlaneGapMeters == accepted.supportPlaneGapMeters,
            "static support compile did not replay exactly");

    NumiHumanMuscleEquilibriumResult airborne;
    fixture.config.supportForceRegularization = 0.0;
    require(fixture.compile({separated}, airborne).succeeded() &&
                !airborne.diagnostics.balanced &&
                airborne.supportNormalForce[0] == 0.0 &&
                near(airborne.diagnostics.maximumFloatingRootForceResidual,
                     fixture.weight()),
            "airborne mass received a fictitious ground reaction");

    auto penetrating = touching;
    penetrating.localPoint[2] = -0.014;
    const auto rejected = fixture.compile({penetrating}, accepted);
    require(rejected.status == NumiHumanMuscleEquilibriumStatus::supportPenetration &&
                rejected.failingIndex == 0u && accepted.q == replay.q &&
                accepted.activation == replay.activation &&
                accepted.supportNormalForce == replay.supportNormalForce &&
                accepted.supportPlaneGapMeters == replay.supportPlaneGapMeters,
            "penetrating initial pose was admitted or mutated accepted output");

    // A shared world translation changes neither contact nor supported load.
    fixture.q[2] = 1.25;
    auto elevatedPlane = touching;
    elevatedPlane.planePoint[2] = 1.25;
    NumiHumanMuscleEquilibriumResult translated;
    require(fixture.compile({elevatedPlane}, translated).succeeded() &&
                translated.diagnostics.balanced &&
                near(translated.supportNormalForce[0], fixture.weight()) &&
                near(translated.supportPlaneGapMeters[0], 0.0),
            "authored nonzero plane was ignored");
    fixture.q[2] = 0.0;

    // The gap and force follow the authored normal, not a hard-coded z axis.
    fixture.model.world.gravityAndTimestep = {0.0f, -9.81f, 0.0f, 0.0001f};
    auto sidePlane = touching;
    sidePlane.normal = {0.0, 1.0, 0.0};
    NumiHumanMuscleEquilibriumResult rotated;
    require(fixture.compile({sidePlane}, rotated).succeeded() &&
                rotated.diagnostics.balanced &&
                near(rotated.generalizedSupportForce[1],
                     -2.5 * fixture.model.world.gravityAndTimestep.y),
            "rotated plane did not support the analytic load");

    // Numerical geometry tolerance is an explicit admission bound.
    auto withinTolerance = sidePlane;
    withinTolerance.localPoint[1] = -0.5 * fixture.config.supportGapToleranceMeters;
    require(fixture.compile({withinTolerance}, rotated).succeeded(),
            "bounded geometric roundoff was rejected");
    fixture.config.supportGapToleranceMeters = 0.0;
    require(fixture.compile({withinTolerance}, rotated).status ==
                NumiHumanMuscleEquilibriumStatus::supportPenetration,
            "zero tolerance was not enforced");
    fixture.config.supportGapToleranceMeters = -1.0;
    require(fixture.compile({sidePlane}, rotated).status ==
                NumiHumanMuscleEquilibriumStatus::invalidConfiguration,
            "negative support tolerance was accepted");
    fixture.config.supportGapToleranceMeters = std::numeric_limits<double>::quiet_NaN();
    require(fixture.compile({sidePlane}, rotated).status ==
                NumiHumanMuscleEquilibriumStatus::invalidConfiguration,
            "nonfinite support tolerance was accepted");
    fixture.config.supportGapToleranceMeters = 1.0e-6;
    sidePlane.planePoint[0] = std::numeric_limits<double>::quiet_NaN();
    require(fixture.compile({sidePlane}, rotated).status ==
                NumiHumanMuscleEquilibriumStatus::invalidDimensions,
            "nonfinite authored plane was accepted");
    std::cout << "numi_human_static_support_test=passed"
              << " analytic_weight_n=" << replay.supportNormalForce[0]
              << " replay=exact penetration=rejected airborne_force_n=0\n";
}
