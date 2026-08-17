#include "metalrobo/MetalDiscreteElasticRod.hpp"
#include "metalrobo/SurgicalKnot.hpp"
#include "metalrobo/SurgicalWorld.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Vec3 = std::array<double, 3>;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Vec3 add(const Vec3& left, const Vec3& right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

Vec3 subtract(const Vec3& left, const Vec3& right) {
    return {
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    };
}

Vec3 multiply(const Vec3& value, const double scale) {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

double dot(const Vec3& left, const Vec3& right) {
    return left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

double length(const Vec3& value) {
    return std::sqrt(dot(value, value));
}

Vec3 normalized(const Vec3& value) {
    const double magnitude = length(value);
    require(magnitude > 0.0, "knot direction is degenerate");
    return multiply(value, 1.0 / magnitude);
}

struct SegmentWitness {
    double first = 0.0;
    double second = 0.0;
    Vec3 delta{};
    double distance = 0.0;
};

SegmentWitness closestSegmentWitness(
    const Vec3& firstA,
    const Vec3& firstB,
    const Vec3& secondA,
    const Vec3& secondB
) {
    const Vec3 first = subtract(firstB, firstA);
    const Vec3 second = subtract(secondB, secondA);
    const Vec3 offset = subtract(firstA, secondA);
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
    const Vec3 firstPoint = add(
        firstA,
        multiply(first, firstParameter)
    );
    const Vec3 secondPoint = add(
        secondA,
        multiply(second, secondParameter)
    );
    SegmentWitness result;
    result.first = firstParameter;
    result.second = secondParameter;
    result.delta = subtract(secondPoint, firstPoint);
    result.distance = length(result.delta);
    return result;
}

struct ContactMotion {
    std::uint32_t count = 0u;
    double minimumCenterlineDistance =
        std::numeric_limits<double>::infinity();
    double maximumTangentialSlip = 0.0;
    double tangentialSlipSquaredSum = 0.0;

    [[nodiscard]] double rmsTangentialSlip() const noexcept {
        return count > 0u
            ? std::sqrt(
                  tangentialSlipSquaredSum /
                  static_cast<double>(count)
              )
            : 0.0;
    }
};

ContactMotion contactMotion(
    const metalrobo::DiscreteElasticRodModel& model,
    const metalrobo::DiscreteElasticRodState& state,
    const double contactMargin,
    const double shellTolerance
) {
    ContactMotion result;
    const double contactDistance =
        2.0 * model.radius + contactMargin;
    for (std::size_t firstEdge = 0u;
         firstEdge < model.restLengths.size();
         ++firstEdge) {
        for (std::size_t secondEdge = firstEdge + 2u;
             secondEdge < model.restLengths.size();
             ++secondEdge) {
            const SegmentWitness witness = closestSegmentWitness(
                state.positions[firstEdge],
                state.positions[firstEdge + 1u],
                state.positions[secondEdge],
                state.positions[secondEdge + 1u]
            );
            result.minimumCenterlineDistance = std::min(
                result.minimumCenterlineDistance,
                witness.distance
            );
            if (witness.distance >
                    contactDistance + shellTolerance ||
                !(witness.distance > 0.0)) {
                continue;
            }
            const Vec3 normal = multiply(
                witness.delta,
                1.0 / witness.distance
            );
            const Vec3 firstVelocity = add(
                multiply(
                    state.velocities[firstEdge],
                    1.0 - witness.first
                ),
                multiply(
                    state.velocities[firstEdge + 1u],
                    witness.first
                )
            );
            const Vec3 secondVelocity = add(
                multiply(
                    state.velocities[secondEdge],
                    1.0 - witness.second
                ),
                multiply(
                    state.velocities[secondEdge + 1u],
                    witness.second
                )
            );
            const Vec3 relative = subtract(
                secondVelocity,
                firstVelocity
            );
            const Vec3 tangent = subtract(
                relative,
                multiply(normal, dot(relative, normal))
            );
            ++result.count;
            result.tangentialSlipSquaredSum += dot(tangent, tangent);
            result.maximumTangentialSlip = std::max(
                result.maximumTangentialSlip,
                length(tangent)
            );
        }
    }
    return result;
}

struct KnotFixture {
    metalrobo::DiscreteElasticRodModel model;
    metalrobo::DiscreteElasticRodState state;
    Vec3 leftPullDirection{};
    Vec3 rightPullDirection{};
    double coreLength = 0.0;
};

KnotFixture makeKnotFixture() {
    constexpr std::uint32_t nodeCount = 128u;
    constexpr std::uint32_t tailEdgeCount = 16u;
    constexpr std::uint32_t coreNodeCount = 96u;
    constexpr double cutAngle = 0.12;
    constexpr double inPlaneScale = 1.8e-3;
    constexpr double crossingHeightScale = 1.02e-4;

    const auto spec =
        metalrobo::makeBowelAnastomosisSutureSpec();
    const auto worldConfig =
        metalrobo::makeBowelAnastomosisNeedleThreadWorldConfig(
            spec
        );
    const auto& material = worldConfig.threadMaterial;
    KnotFixture fixture;
    fixture.model = metalrobo::makeStraightSutureRod(
        nodeCount,
        spec.threadLengthM.value,
        material
    );
    fixture.model.name =
        "pdo_3_0_loaded_five_crossing_knot_fixture";
    fixture.model.fidelityBoundary =
        "authored pre-tied five-crossing mechanics fixture; "
        "not robot-tied, package-calibrated, or clinical evidence";

    std::array<Vec3, coreNodeCount> core{};
    for (std::uint32_t node = 0u;
         node < coreNodeCount;
         ++node) {
        const double fraction =
            static_cast<double>(node) /
            static_cast<double>(coreNodeCount - 1u);
        const double angle =
            cutAngle +
            (2.0 * std::numbers::pi - 2.0 * cutAngle) *
                fraction;
        const double radial = 2.0 + std::cos(5.0 * angle);
        core[node] = {
            inPlaneScale * radial * std::cos(2.0 * angle),
            inPlaneScale * radial * std::sin(2.0 * angle),
            crossingHeightScale * std::sin(5.0 * angle),
        };
        if (node > 0u) {
            fixture.coreLength += length(subtract(
                core[node],
                core[node - 1u]
            ));
        }
    }
    const auto derivative = [=](const double angle) {
        const double radial = 2.0 + std::cos(5.0 * angle);
        const double radialDerivative =
            -5.0 * std::sin(5.0 * angle);
        return Vec3{
            inPlaneScale *
                (
                    radialDerivative * std::cos(2.0 * angle) -
                    2.0 * radial * std::sin(2.0 * angle)
                ),
            inPlaneScale *
                (
                    radialDerivative * std::sin(2.0 * angle) +
                    2.0 * radial * std::cos(2.0 * angle)
                ),
            5.0 * crossingHeightScale * std::cos(5.0 * angle),
        };
    };
    const Vec3 startTangent = normalized(derivative(cutAngle));
    const Vec3 endTangent = normalized(derivative(
        2.0 * std::numbers::pi - cutAngle
    ));
    fixture.leftPullDirection = multiply(startTangent, -1.0);
    fixture.rightPullDirection = endTangent;
    // Route the two long tails to opposite sides of the cut before applying
    // load. Without this small out-of-plane lead, the coarse 16-edge tails
    // introduce an unintended connector intersection at the authored cut.
    fixture.leftPullDirection[2] += 0.1;
    fixture.rightPullDirection[2] -= 0.1;
    fixture.leftPullDirection = normalized(
        fixture.leftPullDirection
    );
    fixture.rightPullDirection = normalized(
        fixture.rightPullDirection
    );
    const double tailLength = 0.5 *
        (spec.threadLengthM.value - fixture.coreLength);
    require(
        tailLength > 0.09,
        "knot fixture leaves insufficient PDO tail length"
    );

    fixture.model.restPositions.assign(nodeCount, Vec3{});
    for (std::uint32_t node = 0u;
         node < tailEdgeCount;
         ++node) {
        const double distance = tailLength *
            static_cast<double>(tailEdgeCount - node) /
            static_cast<double>(tailEdgeCount);
        fixture.model.restPositions[node] = add(
            core.front(),
            multiply(fixture.leftPullDirection, distance)
        );
    }
    for (std::uint32_t node = 0u;
         node < coreNodeCount;
         ++node) {
        fixture.model.restPositions[tailEdgeCount + node] =
            core[node];
    }
    constexpr std::uint32_t coreEnd =
        tailEdgeCount + coreNodeCount - 1u;
    for (std::uint32_t edge = 1u;
         edge <= tailEdgeCount;
         ++edge) {
        const double distance = tailLength *
            static_cast<double>(edge) /
            static_cast<double>(tailEdgeCount);
        fixture.model.restPositions[coreEnd + edge] = add(
            core.back(),
            multiply(fixture.rightPullDirection, distance)
        );
    }

    const double area =
        std::numbers::pi * material.radius * material.radius;
    const double secondMoment =
        std::numbers::pi * std::pow(material.radius, 4.0) / 4.0;
    const double polarMoment = 2.0 * secondMoment;
    const double shearModulus =
        material.youngModulus /
        (2.0 * (1.0 + material.poissonRatio));
    fixture.model.nodeMasses.assign(nodeCount, 0.0);
    for (std::size_t edge = 0u;
         edge < fixture.model.restLengths.size();
         ++edge) {
        const double restLength = length(subtract(
            fixture.model.restPositions[edge + 1u],
            fixture.model.restPositions[edge]
        ));
        fixture.model.restLengths[edge] = restLength;
        fixture.model.edgeRotationalInertias[edge] =
            material.density * polarMoment * restLength;
        fixture.model.stretchStiffness[edge] =
            material.youngModulus * area;
        const double edgeMass =
            material.density * area * restLength;
        fixture.model.nodeMasses[edge] += 0.5 * edgeMass;
        fixture.model.nodeMasses[edge + 1u] += 0.5 * edgeMass;
    }
    std::fill(
        fixture.model.bendStiffness.begin(),
        fixture.model.bendStiffness.end(),
        material.youngModulus * secondMoment
    );
    std::fill(
        fixture.model.twistStiffness.begin(),
        fixture.model.twistStiffness.end(),
        shearModulus * polarMoment
    );
    std::string reason;
    require(
        fixture.model.valid(&reason),
        "knot DER model is invalid: " + reason
    );
    fixture.state =
        metalrobo::makeDiscreteElasticRodDefaultState(fixture.model);
    return fixture;
}

double reactionLoad(
    const std::span<const metalrobo::DiscreteRodAttachmentReaction>
        reactions
) {
    double maximum = 0.0;
    for (const auto& reaction : reactions) {
        maximum = std::max(
            maximum,
            length(reaction.averageForceOnTarget)
        );
    }
    return maximum;
}

} // namespace

int main() {
    try {
        constexpr std::size_t environmentCount = 2u;
        constexpr double timestep = 1.0e-3;
        constexpr double pullDisplacement = 5.0e-4;
        constexpr double contactMargin = 5.0e-5;
        constexpr double pdoFriction = 0.12;

        KnotFixture fixture = makeKnotFixture();
        const ContactMotion authored = contactMotion(
            fixture.model,
            fixture.state,
            contactMargin,
            0.0
        );
        const auto authoredContactCertificate =
            metalrobo::certifySurgicalKnotContacts(
                fixture.state.positions,
                {
                    .threadRadiusM = fixture.model.radius,
                    .contactMarginM = contactMargin,
                    .separationToleranceM = 1.0e-12,
                    .minimumMaterialEdgeSeparation = 2u,
                    .minimumContactPairCount = 5u,
                }
            );
        require(
            authored.count >= 5u &&
                authoredContactCertificate.succeeded() &&
                authoredContactCertificate.contactPairCount ==
                    authored.count &&
                authored.minimumCenterlineDistance >=
                    2.0 * fixture.model.radius &&
                authored.minimumCenterlineDistance <=
                    2.0 * fixture.model.radius + contactMargin,
            "authored knot does not expose five separated contact "
            "crossings: count=" + std::to_string(authored.count) +
                " minimum=" +
                std::to_string(authored.minimumCenterlineDistance) +
                " diameter=" +
                std::to_string(2.0 * fixture.model.radius)
        );

        const std::array<metalrobo::DiscreteRodAttachment, 2>
            attachments{{
                {
                    .nodeIndex = 0u,
                    .targetPosition = add(
                        fixture.state.positions.front(),
                        multiply(
                            fixture.leftPullDirection,
                            pullDisplacement
                        )
                    ),
                    .targetVelocity = {},
                    .compliance = 0.0,
                },
                {
                    .nodeIndex = static_cast<std::uint32_t>(
                        fixture.state.positions.size() - 1u
                    ),
                    .targetPosition = add(
                        fixture.state.positions.back(),
                        multiply(
                            fixture.rightPullDirection,
                            pullDisplacement
                        )
                    ),
                    .targetVelocity = {},
                    .compliance = 0.0,
                },
            }};

        metalrobo::DiscreteElasticRodStepConfig frictionlessConfig;
        frictionlessConfig.timestep = timestep;
        frictionlessConfig.gravity = {0.0, 0.0, 0.0};
        frictionlessConfig.solverIterations = 192u;
        frictionlessConfig.constraintTolerance = 5.0e-6;
        frictionlessConfig.linearDamping = 2.0;
        frictionlessConfig.twistDamping = 2.0;
        frictionlessConfig.derivativeStep = 3.5e-4;
        frictionlessConfig.enableSelfCollision = true;
        frictionlessConfig.selfCollisionMargin = contactMargin;
        auto frictionConfig = frictionlessConfig;
        frictionConfig.selfCollisionFriction = pdoFriction;

        auto cpuFrictionless = fixture.state;
        std::array<metalrobo::DiscreteRodAttachmentReaction, 2>
            cpuFrictionlessReactions{};
        const auto cpuFrictionlessDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                fixture.model,
                cpuFrictionless,
                attachments,
                frictionlessConfig,
                cpuFrictionlessReactions
            );
        auto cpuFriction = fixture.state;
        std::array<metalrobo::DiscreteRodAttachmentReaction, 2>
            cpuFrictionReactions{};
        const auto cpuFrictionDiagnostics =
            metalrobo::stepDiscreteElasticRodCpu(
                fixture.model,
                cpuFriction,
                attachments,
                frictionConfig,
                cpuFrictionReactions
            );
        require(
            cpuFrictionlessDiagnostics.succeeded() &&
                cpuFrictionDiagnostics.succeeded() &&
                cpuFrictionDiagnostics.projectedSelfContacts >= 3u &&
                cpuFriction.positions == cpuFrictionless.positions,
            "FP64 loaded-knot solve failed"
        );
        const ContactMotion cpuFrictionlessMotion = contactMotion(
            fixture.model,
            cpuFrictionless,
            contactMargin,
            frictionlessConfig.constraintTolerance
        );
        const ContactMotion cpuFrictionMotion = contactMotion(
            fixture.model,
            cpuFriction,
            contactMargin,
            frictionConfig.constraintTolerance
        );
        const double cpuAttachmentLoad = reactionLoad(
            cpuFrictionReactions
        );
        require(
            cpuFrictionMotion.count >= 5u &&
                cpuFrictionMotion.rmsTangentialSlip() <
                    0.9 *
                        cpuFrictionlessMotion.rmsTangentialSlip() &&
                cpuAttachmentLoad > 0.0,
            "FP64 PDO knot friction did not reduce loaded crossing slip: "
            "frictionless_rms=" + std::to_string(
                cpuFrictionlessMotion.rmsTangentialSlip()
            ) + " frictional=" + std::to_string(
                cpuFrictionMotion.rmsTangentialSlip()
            ) + " contacts=" + std::to_string(cpuFrictionMotion.count)
        );

        std::vector<metalrobo::DiscreteElasticRodState> states(
            environmentCount,
            fixture.state
        );
        std::vector<metalrobo::DiscreteRodAttachment>
            metalAttachments;
        metalAttachments.reserve(environmentCount * attachments.size());
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            metalAttachments.insert(
                metalAttachments.end(),
                attachments.begin(),
                attachments.end()
            );
        }
        const metalrobo::MetalDiscreteElasticRodInput input{
            .states = states,
            .attachmentCount = 2u,
            .attachments = metalAttachments,
        };
        metalrobo::MetalDiscreteElasticRodConfig
            metalFrictionlessConfig;
        metalFrictionlessConfig.step = frictionlessConfig;
        metalrobo::MetalDiscreteElasticRodResult
            metalFrictionless;
        const auto metalFrictionlessDiagnostics =
            metalrobo::runMetalDiscreteElasticRod(
                fixture.model,
                input,
                metalFrictionless,
                metalFrictionlessConfig
            );
        auto metalFrictionConfig = metalFrictionlessConfig;
        metalFrictionConfig.step = frictionConfig;
        metalrobo::MetalDiscreteElasticRodResult metalFriction;
        const auto metalFrictionDiagnostics =
            metalrobo::runMetalDiscreteElasticRod(
                fixture.model,
                input,
                metalFriction,
                metalFrictionConfig
            );
        require(
            metalFrictionlessDiagnostics.succeeded() &&
                metalFrictionDiagnostics.succeeded() &&
                metalFrictionless.states.size() == environmentCount &&
                metalFriction.states.size() == environmentCount &&
                metalFrictionless.reactions.size() ==
                    environmentCount * attachments.size() &&
                metalFriction.reactions.size() ==
                    environmentCount * attachments.size() &&
                metalFrictionless.statuses.size() == environmentCount &&
                metalFriction.statuses.size() == environmentCount,
            "Apple Metal loaded-knot solve failed: frictionless=\"" +
                metalFrictionlessDiagnostics.message +
                "\" friction=\"" +
                metalFrictionDiagnostics.message + "\""
        );

        double maximumMetalSlip = 0.0;
        double maximumMetalFrictionlessSlip = 0.0;
        double maximumMetalOracleError = 0.0;
        double maximumMetalLoad = 0.0;
        double maximumSelfFrictionContactCount = 0.0;
        double maximumSelfNormalImpulseNs = 0.0;
        double maximumSelfTangentialImpulseNs = 0.0;
        double maximumSelfFrictionUtilization = 0.0;
        for (std::size_t environment = 0u;
             environment < environmentCount;
             ++environment) {
            const ContactMotion frictionlessMotion = contactMotion(
                fixture.model,
                metalFrictionless.states[environment],
                contactMargin,
                frictionlessConfig.constraintTolerance
            );
            const ContactMotion frictionMotion = contactMotion(
                fixture.model,
                metalFriction.states[environment],
                contactMargin,
                frictionConfig.constraintTolerance
            );
            const mr_float4 frictionlessEvidence =
                metalFrictionless.statuses[environment]
                    .selfContactFriction;
            const mr_float4 frictionEvidence =
                metalFriction.statuses[environment]
                    .selfContactFriction;
            require(
                frictionMotion.count >= 5u &&
                    metalFrictionless.statuses[environment].code ==
                        MR_ROD_GPU_SUCCESS &&
                    metalFriction.statuses[environment].code ==
                        MR_ROD_GPU_SUCCESS &&
                    metalFrictionless.statuses[environment].environment ==
                        environment &&
                    metalFriction.statuses[environment].environment ==
                        environment &&
                    frictionlessEvidence.x == 0.0F &&
                    frictionlessEvidence.y == 0.0F &&
                    frictionlessEvidence.z == 0.0F &&
                    frictionlessEvidence.w == 0.0F &&
                    frictionEvidence.x >= 5.0F &&
                    frictionEvidence.y > 0.0F &&
                    frictionEvidence.z > 0.0F &&
                    frictionEvidence.w <= 1.0001F &&
                    frictionMotion.rmsTangentialSlip() <
                        0.9 *
                            frictionlessMotion.rmsTangentialSlip() &&
                    metalFriction.states[environment].positions ==
                        metalFrictionless.states[environment].positions,
                "Apple Metal PDO knot friction did not reduce loaded slip"
            );
            maximumSelfFrictionContactCount = std::max(
                maximumSelfFrictionContactCount,
                static_cast<double>(frictionEvidence.x)
            );
            maximumSelfNormalImpulseNs = std::max(
                maximumSelfNormalImpulseNs,
                static_cast<double>(frictionEvidence.y)
            );
            maximumSelfTangentialImpulseNs = std::max(
                maximumSelfTangentialImpulseNs,
                static_cast<double>(frictionEvidence.z)
            );
            maximumSelfFrictionUtilization = std::max(
                maximumSelfFrictionUtilization,
                static_cast<double>(frictionEvidence.w)
            );
            maximumMetalSlip = std::max(
                maximumMetalSlip,
                frictionMotion.rmsTangentialSlip()
            );
            maximumMetalFrictionlessSlip = std::max(
                maximumMetalFrictionlessSlip,
                frictionlessMotion.rmsTangentialSlip()
            );
            maximumMetalOracleError = std::max(
                maximumMetalOracleError,
                std::abs(
                    frictionMotion.rmsTangentialSlip() -
                    cpuFrictionMotion.rmsTangentialSlip()
                )
            );
            maximumMetalLoad = std::max(
                maximumMetalLoad,
                reactionLoad({
                    metalFriction.reactions.data() +
                        environment * attachments.size(),
                    attachments.size(),
                })
            );
        }
        require(
            maximumMetalOracleError <= 3.0e-3 &&
                maximumMetalLoad > 0.0,
            "Apple Metal knot response diverged from FP64 or carried no "
            "load: oracle_error=" +
                std::to_string(maximumMetalOracleError) +
                " load=" + std::to_string(maximumMetalLoad) +
                " metal_slip=" + std::to_string(maximumMetalSlip) +
                " fp64_slip=" + std::to_string(
                    cpuFrictionMotion.rmsTangentialSlip()
                )
        );

        metalrobo::MetalDiscreteElasticRodResult replay;
        const auto replayDiagnostics =
            metalrobo::runMetalDiscreteElasticRod(
                fixture.model,
                input,
                replay,
                metalFrictionConfig
            );
        bool replayExact =
            replayDiagnostics.succeeded() &&
            replay.states.size() == metalFriction.states.size() &&
            replay.reactions.size() == metalFriction.reactions.size() &&
            replay.statuses.size() == metalFriction.statuses.size();
        for (std::size_t environment = 0u;
             replayExact && environment < replay.states.size();
             ++environment) {
            replayExact =
                replay.states[environment].positions ==
                    metalFriction.states[environment].positions &&
                replay.states[environment].velocities ==
                    metalFriction.states[environment].velocities &&
                replay.statuses[environment]
                        .selfContactFriction.x ==
                    metalFriction.statuses[environment]
                        .selfContactFriction.x &&
                replay.statuses[environment]
                        .selfContactFriction.y ==
                    metalFriction.statuses[environment]
                        .selfContactFriction.y &&
                replay.statuses[environment]
                        .selfContactFriction.z ==
                    metalFriction.statuses[environment]
                        .selfContactFriction.z &&
                replay.statuses[environment]
                        .selfContactFriction.w ==
                    metalFriction.statuses[environment]
                        .selfContactFriction.w;
        }
        for (std::size_t reaction = 0u;
             replayExact && reaction < replay.reactions.size();
             ++reaction) {
            replayExact =
                replay.reactions[reaction].nodeIndex ==
                    metalFriction.reactions[reaction].nodeIndex &&
                replay.reactions[reaction].bodyIndex ==
                    metalFriction.reactions[reaction].bodyIndex &&
                replay.reactions[reaction].impulseOnTarget ==
                    metalFriction.reactions[reaction].impulseOnTarget &&
                replay.reactions[reaction].averageForceOnTarget ==
                    metalFriction.reactions[reaction].averageForceOnTarget &&
                replay.reactions[reaction].finalPositionError ==
                    metalFriction.reactions[reaction].finalPositionError;
        }
        require(
            replayExact,
            "Apple Metal loaded-knot replay is not deterministic"
        );

        std::cout << std::setprecision(9)
            << "surgical_knot_mechanics=ok"
            << " device=\"" << metalFrictionDiagnostics.deviceName
            << "\""
            << " material=PDO_3_0"
            << " thread_diameter_mm="
            << 2000.0 * fixture.model.radius
            << " thread_length_m=0.25"
            << " nodes=" << fixture.model.restPositions.size()
            << " core_length_m=" << fixture.coreLength
            << " topological_crossings=5"
            << " certified_contact_pairs="
            << authoredContactCertificate.contactPairCount
            << " minimum_contact_surface_gap_m="
            << authoredContactCertificate.minimumContactSurfaceGapM
            << " maximum_contact_surface_gap_m="
            << authoredContactCertificate.maximumContactSurfaceGapM
            << " contact_edge_pairs=" << authored.count
            << " authored_min_clearance_m="
            << authored.minimumCenterlineDistance -
                2.0 * fixture.model.radius
            << " friction_coefficient=" << pdoFriction
            << " opposing_pull_displacement_m="
            << pullDisplacement
            << " fp64_frictionless_slip_mps="
            << cpuFrictionlessMotion.rmsTangentialSlip()
            << " fp64_frictional_slip_mps="
            << cpuFrictionMotion.rmsTangentialSlip()
            << " fp64_attachment_load_n="
            << cpuAttachmentLoad
            << " metal_frictionless_slip_mps="
            << maximumMetalFrictionlessSlip
            << " metal_frictional_slip_mps="
            << maximumMetalSlip
            << " metal_self_friction_contacts="
            << maximumSelfFrictionContactCount
            << " metal_self_normal_impulse_ns="
            << maximumSelfNormalImpulseNs
            << " metal_self_tangential_impulse_ns="
            << maximumSelfTangentialImpulseNs
            << " metal_self_friction_utilization="
            << maximumSelfFrictionUtilization
            << " metal_oracle_error_mps="
            << maximumMetalOracleError
            << " maximum_attachment_load_n="
            << maximumMetalLoad
            << " deterministic=yes"
            << " boundary=authored_pre_tied_fixture_not_robot_tied"
            << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr
            << "surgical_knot_mechanics=failed reason=\""
            << exception.what() << "\"\n";
        return 1;
    }
}
