#include "metalrobo/DiscreteElasticRod.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

double distance(
    const std::array<double, 3>& a,
    const std::array<double, 3>& b
) {
    const double x = a[0] - b[0];
    const double y = a[1] - b[1];
    const double z = a[2] - b[2];
    return std::sqrt(x * x + y * y + z * z);
}

double dot(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

std::array<double, 3> subtract(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right
) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

struct SegmentWitness {
    double first = 0.0;
    double second = 0.0;
    std::array<double, 3> delta{};
    double distance = 0.0;
};

SegmentWitness closestSegmentWitness(
    const std::array<double, 3>& firstA,
    const std::array<double, 3>& firstB,
    const std::array<double, 3>& secondA,
    const std::array<double, 3>& secondB
) {
    const auto first = subtract(firstB, firstA);
    const auto second = subtract(secondB, secondA);
    const auto offset = subtract(firstA, secondA);
    const double aa = dot(first, first);
    const double bb = dot(second, second);
    const double ab = dot(first, second);
    const double ao = dot(first, offset);
    const double bo = dot(second, offset);
    const double denominator = aa * bb - ab * ab;
    double firstParameter = denominator > 1.0e-14 * aa * bb
        ? std::clamp(
              (ab * bo - ao * bb) / denominator,
              0.0,
              1.0
          )
        : 0.0;
    const double secondNumerator =
        ab * firstParameter + bo;
    double secondParameter = 0.0;
    if (secondNumerator < 0.0) {
        firstParameter = std::clamp(-ao / aa, 0.0, 1.0);
    } else if (secondNumerator > bb) {
        secondParameter = 1.0;
        firstParameter = std::clamp(
            (ab - ao) / aa,
            0.0,
            1.0
        );
    } else {
        secondParameter = secondNumerator / bb;
    }
    const std::array<double, 3> firstPoint{
        firstA[0] + firstParameter * first[0],
        firstA[1] + firstParameter * first[1],
        firstA[2] + firstParameter * first[2],
    };
    const std::array<double, 3> secondPoint{
        secondA[0] + secondParameter * second[0],
        secondA[1] + secondParameter * second[1],
        secondA[2] + secondParameter * second[2],
    };
    SegmentWitness witness;
    witness.first = firstParameter;
    witness.second = secondParameter;
    witness.delta = subtract(secondPoint, firstPoint);
    witness.distance = std::sqrt(dot(witness.delta, witness.delta));
    return witness;
}

double tangentialContactSlip(
    const metalrobo::DiscreteElasticRodState& state
) {
    const SegmentWitness witness = closestSegmentWitness(
        state.positions[0],
        state.positions[1],
        state.positions[2],
        state.positions[3]
    );
    require(
        witness.distance > 0.0,
        "DER slip witness has no contact normal"
    );
    std::array<double, 3> normal = witness.delta;
    for (double& component : normal) {
        component /= witness.distance;
    }
    const std::array<double, 3> firstVelocity{
        (1.0 - witness.first) * state.velocities[0][0] +
            witness.first * state.velocities[1][0],
        (1.0 - witness.first) * state.velocities[0][1] +
            witness.first * state.velocities[1][1],
        (1.0 - witness.first) * state.velocities[0][2] +
            witness.first * state.velocities[1][2],
    };
    const std::array<double, 3> secondVelocity{
        (1.0 - witness.second) * state.velocities[2][0] +
            witness.second * state.velocities[3][0],
        (1.0 - witness.second) * state.velocities[2][1] +
            witness.second * state.velocities[3][1],
        (1.0 - witness.second) * state.velocities[2][2] +
            witness.second * state.velocities[3][2],
    };
    const auto relativeVelocity = subtract(
        secondVelocity,
        firstVelocity
    );
    const double normalSpeed = dot(relativeVelocity, normal);
    return std::sqrt(std::max(
        dot(relativeVelocity, relativeVelocity) -
            normalSpeed * normalSpeed,
        0.0
    ));
}

double segmentDistance(
    const std::array<double, 3>& firstA,
    const std::array<double, 3>& firstB,
    const std::array<double, 3>& secondA,
    const std::array<double, 3>& secondB
) {
    return closestSegmentWitness(
        firstA,
        firstB,
        secondA,
        secondB
    ).distance;
}

} // namespace

int main() {
    try {
        const metalrobo::DiscreteElasticRodModel model =
            metalrobo::makeStraightSutureRod(
                9u,
                0.12
            );
        metalrobo::DiscreteElasticRodState state =
            metalrobo::makeDiscreteElasticRodDefaultState(model);
        state.positions[4][1] = 0.008;
        state.positions.back()[0] += 0.006;
        state.twists.back() = 0.6;
        metalrobo::DiscreteElasticRodEnergy initial{};
        const auto initialDiagnostics =
            metalrobo::evaluateDiscreteElasticRodEnergy(
                model,
                state,
                initial
            );
        require(
            initialDiagnostics.succeeded() &&
                initial.stretch > 0.0 &&
                initial.bend > 0.0 &&
                initial.twist > 0.0,
            "deformed DER energy is incomplete"
        );

        metalrobo::DiscreteRodAttachment attachment{
            .nodeIndex = 0u,
            .targetPosition = model.restPositions[0],
            .targetVelocity = {0.0, 0.0, 0.0},
            .compliance = 0.0,
        };
        metalrobo::DiscreteElasticRodStepConfig config;
        config.gravity = {0.0, 0.0, 0.0};
        config.solverIterations = 96u;
        config.constraintTolerance = 1.0e-5;
        std::array<
            metalrobo::DiscreteRodAttachmentReaction,
            1
        > reactions{};
        const auto diagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                model,
                state,
                {&attachment, 1u},
                config,
                reactions
            );
        require(
            diagnostics.succeeded(),
            "implicit DER step failed: " +
                diagnostics.message
        );
        require(
            state.positions[0] == attachment.targetPosition &&
                distance(
                    state.positions[0],
                    state.positions[1]
                ) < 1.1 * model.restLengths[0],
            "hard attachment or stretch projection failed"
        );
        require(
            reactions[0].nodeIndex == 0u &&
                std::isfinite(
                    reactions[0].averageForceOnTarget[0]
                ) &&
                std::isfinite(
                    reactions[0].averageForceOnTarget[1]
                ) &&
                std::isfinite(
                    reactions[0].averageForceOnTarget[2]
                ) &&
                reactions[0].finalPositionError <= 1.0e-12,
            "DER attachment reaction evidence is invalid"
        );
        require(
            diagnostics.after.total() <
                initial.total(),
            "DER projection did not reduce elastic energy"
        );

        metalrobo::DiscreteElasticRodState replay =
            metalrobo::makeDiscreteElasticRodDefaultState(model);
        replay.positions[4][1] = 0.008;
        replay.positions.back()[0] += 0.006;
        replay.twists.back() = 0.6;
        const auto replayDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                model,
                replay,
                {&attachment, 1u},
                config
            );
        require(
            replayDiagnostics.succeeded() &&
                replay.positions == state.positions &&
                replay.velocities == state.velocities &&
                replay.twists == state.twists &&
                replay.twistRates == state.twistRates,
            "DER replay is not deterministic"
        );

        metalrobo::DiscreteElasticRodState rejected = state;
        rejected.positions[3][0] =
            std::numeric_limits<double>::quiet_NaN();
        const auto sentinel = rejected;
        const auto failed =
            metalrobo::stepDiscreteElasticRodCpu(
                model,
                rejected,
                {&attachment, 1u},
                config
            );
        require(
            !failed.succeeded() &&
                std::isnan(rejected.positions[3][0]) &&
                rejected.positions[2] == sentinel.positions[2] &&
                rejected.twists == sentinel.twists,
            "DER non-finite rejection changed state"
        );

        metalrobo::DiscreteElasticRodModel collisionModel =
            metalrobo::makeStraightSutureRod(4u, 0.12);
        collisionModel.stretchStiffness.assign(
            collisionModel.stretchStiffness.size(),
            1.0e-6
        );
        collisionModel.bendStiffness.assign(
            collisionModel.bendStiffness.size(),
            1.0e-12
        );
        collisionModel.twistStiffness.assign(
            collisionModel.twistStiffness.size(),
            1.0e-12
        );
        metalrobo::DiscreteElasticRodState crossing =
            metalrobo::makeDiscreteElasticRodDefaultState(
                collisionModel
            );
        crossing.positions = {{
            {-0.02, 0.00, 0.0},
            { 0.02, 0.00, 0.0},
            { 0.00, -0.03464101615137755, 0.0},
            { 0.00, 0.00535898384862245, 0.0},
        }};
        metalrobo::DiscreteElasticRodStepConfig collisionConfig;
        collisionConfig.gravity = {0.0, 0.0, 0.0};
        collisionConfig.solverIterations = 256u;
        collisionConfig.constraintTolerance = 1.0e-6;
        collisionConfig.enableSelfCollision = true;
        const auto crossingInput = crossing;
        const auto collisionDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                collisionModel,
                crossing,
                {},
                collisionConfig
            );
        const double selfContactClearance = segmentDistance(
            crossing.positions[0],
            crossing.positions[1],
            crossing.positions[2],
            crossing.positions[3]
        );
        require(
            collisionDiagnostics.succeeded() &&
                collisionDiagnostics.projectedSelfContacts > 0u &&
                collisionDiagnostics.maximumSelfPenetration >=
                    1.9 * collisionModel.radius &&
                selfContactClearance >=
                    0.95 * 2.0 * collisionModel.radius,
            "DER capsule self-contact did not separate crossing edges: " +
                collisionDiagnostics.message +
                " contacts=" +
                std::to_string(
                    collisionDiagnostics.projectedSelfContacts
                ) +
                " penetration=" +
                std::to_string(
                    collisionDiagnostics.maximumSelfPenetration
                ) +
                " separation=" +
                std::to_string(selfContactClearance)
        );
        auto crossingReplay = crossingInput;
        const auto crossingReplayDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                collisionModel,
                crossingReplay,
                {},
                collisionConfig
            );
        require(
            crossingReplayDiagnostics.succeeded() &&
                crossingReplay.positions == crossing.positions &&
                crossingReplay.velocities == crossing.velocities,
            "DER self-contact replay is not deterministic"
        );

        auto slidingInput = crossingInput;
        slidingInput.velocities[2] = {0.02, 0.0, 0.0};
        slidingInput.velocities[3] = {0.02, 0.0, 0.0};
        auto frictionlessSliding = slidingInput;
        const auto frictionlessSlidingDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                collisionModel,
                frictionlessSliding,
                {},
                collisionConfig
            );
        auto frictionConfig = collisionConfig;
        frictionConfig.selfCollisionFriction = 0.12;
        auto frictionalSliding = slidingInput;
        const auto frictionalSlidingDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                collisionModel,
                frictionalSliding,
                {},
                frictionConfig
            );
        const double frictionlessSlip =
            tangentialContactSlip(frictionlessSliding);
        const double frictionalSlip =
            tangentialContactSlip(frictionalSliding);
        std::array<double, 3> frictionMomentumDelta{};
        for (std::size_t node = 0u;
             node < collisionModel.nodeMasses.size();
             ++node) {
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                frictionMomentumDelta[axis] +=
                    collisionModel.nodeMasses[node] *
                    (
                        frictionalSliding.velocities[node][axis] -
                        frictionlessSliding.velocities[node][axis]
                    );
            }
        }
        require(
            frictionlessSlidingDiagnostics.succeeded() &&
                frictionalSlidingDiagnostics.succeeded() &&
                frictionalSlip < 0.9 * frictionlessSlip &&
                std::sqrt(
                    frictionMomentumDelta[0] *
                        frictionMomentumDelta[0] +
                    frictionMomentumDelta[1] *
                        frictionMomentumDelta[1] +
                    frictionMomentumDelta[2] *
                        frictionMomentumDelta[2]
                ) <= 1.0e-12,
            "DER Coulomb self-contact did not reduce slip with "
            "equal-and-opposite impulse: frictionless_slip=" +
                std::to_string(frictionlessSlip) +
                " frictional_slip=" +
                std::to_string(frictionalSlip) +
                " momentum_delta=" +
                std::to_string(std::sqrt(
                    frictionMomentumDelta[0] *
                        frictionMomentumDelta[0] +
                    frictionMomentumDelta[1] *
                        frictionMomentumDelta[1] +
                    frictionMomentumDelta[2] *
                        frictionMomentumDelta[2]
                ))
        );
        auto frictionReplay = slidingInput;
        const auto frictionReplayDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                collisionModel,
                frictionReplay,
                {},
                frictionConfig
            );
        require(
            frictionReplayDiagnostics.succeeded() &&
                frictionReplay.positions == frictionalSliding.positions &&
                frictionReplay.velocities == frictionalSliding.velocities,
            "DER frictional self-contact replay is not deterministic"
        );

        std::cout
            << "discrete_elastic_rod=ok"
            << " nodes=" << model.restPositions.size()
            << " edges=" << model.restLengths.size()
            << " iterations=" << diagnostics.iterations
            << " stretch=" << diagnostics.after.stretch
            << " bend=" << diagnostics.after.bend
            << " twist=" << diagnostics.after.twist
            << " max_error="
            << diagnostics.maximumConstraintError
            << " anchor_force_n="
            << std::sqrt(
                reactions[0].averageForceOnTarget[0] *
                    reactions[0].averageForceOnTarget[0] +
                reactions[0].averageForceOnTarget[1] *
                    reactions[0].averageForceOnTarget[1] +
                reactions[0].averageForceOnTarget[2] *
                reactions[0].averageForceOnTarget[2]
            )
            << " self_contacts="
            << collisionDiagnostics.projectedSelfContacts
            << " self_penetration="
            << collisionDiagnostics.maximumSelfPenetration
            << " self_contact_clearance_m="
            << selfContactClearance
            << " frictionless_slip_mps=" << frictionlessSlip
            << " frictional_slip_mps=" << frictionalSlip
            << " deterministic=yes transactional=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            << "discrete_elastic_rod=failed reason=\""
            << error.what()
            << "\"\n";
        return 1;
    }
}
