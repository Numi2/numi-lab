#include "metalrobo/DiscreteElasticRod.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

using Vec3 = std::array<double, 3>;

Vec3 add(const Vec3 a, const Vec3 b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 subtract(const Vec3 a, const Vec3 b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 multiply(const Vec3 value, const double scale) {
    return {
        value[0] * scale,
        value[1] * scale,
        value[2] * scale,
    };
}

double dot(const Vec3 a, const Vec3 b) {
    return a[0] * b[0] +
        a[1] * b[1] +
        a[2] * b[2];
}

Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

double norm(const Vec3 value) {
    return std::sqrt(dot(value, value));
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vec3 value) {
    return finite(value[0]) &&
        finite(value[1]) &&
        finite(value[2]);
}

bool normalize(const Vec3 value, Vec3& result) {
    const double length = norm(value);
    if (!(length > 1.0e-14) || !finite(length)) {
        return false;
    }
    result = multiply(value, 1.0 / length);
    return finite(result);
}

Vec3 rotateAroundAxis(
    const Vec3 value,
    const Vec3 axis,
    const double angle
) {
    return add(
        add(
            multiply(value, std::cos(angle)),
            multiply(cross(axis, value), std::sin(angle))
        ),
        multiply(
            axis,
            dot(axis, value) * (1.0 - std::cos(angle))
        )
    );
}

bool transport(
    const Vec3 director,
    const Vec3 from,
    const Vec3 to,
    Vec3& transported
) {
    const Vec3 axis = cross(from, to);
    const double sine = norm(axis);
    const double cosine =
        std::clamp(dot(from, to), -1.0, 1.0);
    if (sine <= 1.0e-14) {
        if (cosine < 0.0) {
            return false;
        }
        transported = director;
        return true;
    }
    transported = rotateAroundAxis(
        director,
        multiply(axis, 1.0 / sine),
        std::atan2(sine, cosine)
    );
    transported = subtract(
        transported,
        multiply(to, dot(transported, to))
    );
    return normalize(transported, transported);
}

Vec3 leastAlignedDirector(const Vec3 tangent) {
    const std::array<Vec3, 3> axes{{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }};
    std::size_t selected = 0u;
    for (std::size_t index = 1u; index < axes.size(); ++index) {
        if (std::abs(dot(tangent, axes[index])) <
            std::abs(dot(tangent, axes[selected]))) {
            selected = index;
        }
    }
    Vec3 director = subtract(
        axes[selected],
        multiply(tangent, dot(axes[selected], tangent))
    );
    (void)normalize(director, director);
    return director;
}

struct RodFrames {
    std::vector<Vec3> tangents;
    std::vector<Vec3> director1;
    std::vector<Vec3> director2;
    std::vector<std::array<double, 2>> curvature;
};

bool buildFrames(
    const std::span<const Vec3> positions,
    const std::span<const double> twists,
    RodFrames& frames
) {
    const std::size_t edgeCount = positions.size() - 1u;
    frames = {};
    frames.tangents.resize(edgeCount);
    frames.director1.resize(edgeCount);
    frames.director2.resize(edgeCount);
    frames.curvature.resize(
        edgeCount > 0u ? edgeCount - 1u : 0u
    );
    for (std::size_t edge = 0u; edge < edgeCount; ++edge) {
        if (!normalize(
                subtract(
                    positions[edge + 1u],
                    positions[edge]
                ),
                frames.tangents[edge]
            )) {
            return false;
        }
    }
    Vec3 reference =
        leastAlignedDirector(frames.tangents[0]);
    for (std::size_t edge = 0u; edge < edgeCount; ++edge) {
        if (edge != 0u &&
            !transport(
                reference,
                frames.tangents[edge - 1u],
                frames.tangents[edge],
                reference
            )) {
            return false;
        }
        frames.director1[edge] = rotateAroundAxis(
            reference,
            frames.tangents[edge],
            twists[edge]
        );
        frames.director2[edge] = cross(
            frames.tangents[edge],
            frames.director1[edge]
        );
        if (!finite(frames.director1[edge]) ||
            !finite(frames.director2[edge])) {
            return false;
        }
    }
    for (std::size_t vertex = 1u;
         vertex + 1u < positions.size();
         ++vertex) {
        const Vec3 left = frames.tangents[vertex - 1u];
        const Vec3 right = frames.tangents[vertex];
        const double denominator = 1.0 + dot(left, right);
        if (!(denominator > 1.0e-8) ||
            !finite(denominator)) {
            return false;
        }
        const Vec3 binormal = multiply(
            cross(left, right),
            2.0 / denominator
        );
        frames.curvature[vertex - 1u] = {
            0.5 * dot(
                binormal,
                add(
                    frames.director2[vertex - 1u],
                    frames.director2[vertex]
                )
            ),
            -0.5 * dot(
                binormal,
                add(
                    frames.director1[vertex - 1u],
                    frames.director1[vertex]
                )
            ),
        };
    }
    return true;
}

DiscreteElasticRodDiagnostics fail(
    DiscreteElasticRodDiagnostics diagnostics,
    const DiscreteElasticRodStatus status,
    std::string message
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    return diagnostics;
}

bool validState(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& state
) {
    const std::size_t nodes = model.restPositions.size();
    const std::size_t edges = nodes - 1u;
    return
        state.positions.size() == nodes &&
        state.velocities.size() == nodes &&
        state.twists.size() == edges &&
        state.twistRates.size() == edges &&
        std::ranges::all_of(
            state.positions,
            [](const Vec3 value) { return finite(value); }
        ) &&
        std::ranges::all_of(
            state.velocities,
            [](const Vec3 value) { return finite(value); }
        ) &&
        std::ranges::all_of(
            state.twists,
            [](const double value) { return finite(value); }
        ) &&
        std::ranges::all_of(
            state.twistRates,
            [](const double value) { return finite(value); }
        );
}

bool energy(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& state,
    DiscreteElasticRodEnergy& result
) {
    RodFrames restFrames;
    RodFrames frames;
    if (!buildFrames(
            model.restPositions,
            model.restTwists,
            restFrames
        ) ||
        !buildFrames(
            state.positions,
            state.twists,
            frames
        )) {
        return false;
    }
    result = {};
    for (std::size_t edge = 0u;
         edge < model.restLengths.size();
         ++edge) {
        const double currentLength = norm(
            subtract(
                state.positions[edge + 1u],
                state.positions[edge]
            )
        );
        const double extension =
            currentLength - model.restLengths[edge];
        result.stretch +=
            0.5 * model.stretchStiffness[edge] /
            model.restLengths[edge] *
            extension * extension;
    }
    for (std::size_t vertex = 0u;
         vertex < model.bendStiffness.size();
         ++vertex) {
        const double voronoi =
            0.5 * (
                model.restLengths[vertex] +
                model.restLengths[vertex + 1u]
            );
        const double first =
            frames.curvature[vertex][0] -
            restFrames.curvature[vertex][0];
        const double second =
            frames.curvature[vertex][1] -
            restFrames.curvature[vertex][1];
        result.bend +=
            0.5 * model.bendStiffness[vertex] /
            voronoi *
            (first * first + second * second);
        const double twist =
            (
                state.twists[vertex + 1u] -
                state.twists[vertex]
            ) -
            (
                model.restTwists[vertex + 1u] -
                model.restTwists[vertex]
            );
        result.twist +=
            0.5 * model.twistStiffness[vertex] /
            voronoi *
            twist * twist;
    }
    return finite(result.stretch) &&
        finite(result.bend) &&
        finite(result.twist);
}

bool validConfig(const DiscreteElasticRodStepConfig& config) {
    return
        finite(config.timestep) &&
        config.timestep > 0.0 &&
        std::ranges::all_of(
            config.gravity,
            [](const double value) { return finite(value); }
        ) &&
        config.solverIterations > 0u &&
        finite(config.constraintTolerance) &&
        config.constraintTolerance > 0.0 &&
        finite(config.linearDamping) &&
        config.linearDamping >= 0.0 &&
        finite(config.twistDamping) &&
        config.twistDamping >= 0.0 &&
        finite(config.derivativeStep) &&
        config.derivativeStep > 0.0;
}

double stretchCompliance(
    const DiscreteElasticRodModel& model,
    const std::size_t edge
) {
    return model.restLengths[edge] /
        model.stretchStiffness[edge];
}

bool projectStretch(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const double timestep,
    const std::size_t edge,
    double& maximumError,
    double& maximumCorrection
) {
    Vec3 direction;
    const Vec3 delta = subtract(
        state.positions[edge + 1u],
        state.positions[edge]
    );
    const double length = norm(delta);
    if (!normalize(delta, direction)) {
        return false;
    }
    const double constraint =
        length - model.restLengths[edge];
    maximumError = std::max(
        maximumError,
        std::abs(constraint)
    );
    const double inverseA = 1.0 / model.nodeMasses[edge];
    const double inverseB =
        1.0 / model.nodeMasses[edge + 1u];
    const double alpha =
        stretchCompliance(model, edge) /
        (timestep * timestep);
    const double lambda =
        -constraint / (inverseA + inverseB + alpha);
    const Vec3 firstCorrection =
        multiply(direction, -inverseA * lambda);
    const Vec3 secondCorrection =
        multiply(direction, inverseB * lambda);
    state.positions[edge] =
        add(state.positions[edge], firstCorrection);
    state.positions[edge + 1u] =
        add(state.positions[edge + 1u], secondCorrection);
    maximumCorrection = std::max({
        maximumCorrection,
        norm(firstCorrection),
        norm(secondCorrection),
    });
    return true;
}

bool localCurvature(
    const std::array<Vec3, 3>& positions,
    const std::array<double, 2>& twists,
    std::array<double, 2>& curvature
) {
    RodFrames frames;
    if (!buildFrames(positions, twists, frames) ||
        frames.curvature.size() != 1u) {
        return false;
    }
    curvature = frames.curvature[0];
    return true;
}

bool projectBendComponent(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const RodFrames& restFrames,
    const DiscreteElasticRodStepConfig& config,
    const std::size_t vertex,
    const std::size_t component,
    double& maximumError,
    double& maximumCorrection
) {
    std::array<Vec3, 3> localPositions{{
        state.positions[vertex],
        state.positions[vertex + 1u],
        state.positions[vertex + 2u],
    }};
    std::array<double, 2> localTwists{{
        state.twists[vertex],
        state.twists[vertex + 1u],
    }};
    std::array<double, 2> current{};
    if (!localCurvature(
            localPositions,
            localTwists,
            current
        )) {
        return false;
    }
    const double constraint =
        current[component] -
        restFrames.curvature[vertex][component];
    maximumError = std::max(
        maximumError,
        std::abs(constraint)
    );

    std::array<Vec3, 3> positionGradient{};
    std::array<double, 2> twistGradient{};
    for (std::size_t node = 0u; node < 3u; ++node) {
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            const double scale = std::max({
                model.restLengths[vertex],
                model.restLengths[vertex + 1u],
                std::abs(localPositions[node][axis]),
                1.0e-3,
            });
            const double step =
                config.derivativeStep * scale;
            localPositions[node][axis] += step;
            std::array<double, 2> plus{};
            const bool plusOk = localCurvature(
                localPositions,
                localTwists,
                plus
            );
            localPositions[node][axis] -= 2.0 * step;
            std::array<double, 2> minus{};
            const bool minusOk = localCurvature(
                localPositions,
                localTwists,
                minus
            );
            localPositions[node][axis] += step;
            if (!plusOk || !minusOk) {
                return false;
            }
            positionGradient[node][axis] =
                (plus[component] - minus[component]) /
                (2.0 * step);
        }
    }
    for (std::size_t edge = 0u; edge < 2u; ++edge) {
        const double step = config.derivativeStep;
        localTwists[edge] += step;
        std::array<double, 2> plus{};
        const bool plusOk = localCurvature(
            localPositions,
            localTwists,
            plus
        );
        localTwists[edge] -= 2.0 * step;
        std::array<double, 2> minus{};
        const bool minusOk = localCurvature(
            localPositions,
            localTwists,
            minus
        );
        localTwists[edge] += step;
        if (!plusOk || !minusOk) {
            return false;
        }
        twistGradient[edge] =
            (plus[component] - minus[component]) /
            (2.0 * step);
    }

    double denominator = 0.0;
    for (std::size_t node = 0u; node < 3u; ++node) {
        denominator +=
            dot(positionGradient[node], positionGradient[node]) /
            model.nodeMasses[vertex + node];
    }
    for (std::size_t edge = 0u; edge < 2u; ++edge) {
        denominator +=
            twistGradient[edge] * twistGradient[edge] /
            model.edgeRotationalInertias[vertex + edge];
    }
    const double voronoi =
        0.5 * (
            model.restLengths[vertex] +
            model.restLengths[vertex + 1u]
        );
    const double compliance =
        voronoi / model.bendStiffness[vertex];
    const double alpha =
        compliance /
        (config.timestep * config.timestep);
    if (!(denominator + alpha > 0.0) ||
        !finite(denominator)) {
        return false;
    }
    const double lambda =
        -constraint / (denominator + alpha);
    for (std::size_t node = 0u; node < 3u; ++node) {
        const Vec3 correction = multiply(
            positionGradient[node],
            lambda / model.nodeMasses[vertex + node]
        );
        state.positions[vertex + node] =
            add(state.positions[vertex + node], correction);
        maximumCorrection = std::max(
            maximumCorrection,
            norm(correction)
        );
    }
    for (std::size_t edge = 0u; edge < 2u; ++edge) {
        state.twists[vertex + edge] +=
            lambda * twistGradient[edge] /
            model.edgeRotationalInertias[vertex + edge];
    }
    return true;
}

bool projectTwist(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const DiscreteElasticRodStepConfig& config,
    const std::size_t vertex,
    double& maximumError
) {
    const double constraint =
        (
            state.twists[vertex + 1u] -
            state.twists[vertex]
        ) -
        (
            model.restTwists[vertex + 1u] -
            model.restTwists[vertex]
        );
    maximumError = std::max(
        maximumError,
        std::abs(constraint)
    );
    const double inverseA =
        1.0 / model.edgeRotationalInertias[vertex];
    const double inverseB =
        1.0 / model.edgeRotationalInertias[vertex + 1u];
    const double voronoi =
        0.5 * (
            model.restLengths[vertex] +
            model.restLengths[vertex + 1u]
        );
    const double alpha =
        voronoi /
        model.twistStiffness[vertex] /
        (config.timestep * config.timestep);
    const double lambda =
        -constraint / (inverseA + inverseB + alpha);
    state.twists[vertex] -= inverseA * lambda;
    state.twists[vertex + 1u] += inverseB * lambda;
    return finite(state.twists[vertex]) &&
        finite(state.twists[vertex + 1u]);
}

bool projectAttachment(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const DiscreteRodAttachment& attachment,
    const double timestep,
    double& maximumError,
    double& maximumCorrection
) {
    const Vec3 delta = subtract(
        state.positions[attachment.nodeIndex],
        attachment.targetPosition
    );
    maximumError = std::max(maximumError, norm(delta));
    const double inverseMass =
        1.0 / model.nodeMasses[attachment.nodeIndex];
    const double alpha =
        attachment.compliance / (timestep * timestep);
    const Vec3 correction = multiply(
        delta,
        -inverseMass / (inverseMass + alpha)
    );
    state.positions[attachment.nodeIndex] =
        add(
            state.positions[attachment.nodeIndex],
            correction
        );
    maximumCorrection = std::max(
        maximumCorrection,
        norm(correction)
    );
    return finite(state.positions[attachment.nodeIndex]);
}

} // namespace

bool DiscreteElasticRodModel::valid(
    std::string* reason
) const {
    const auto reject = [reason](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (restPositions.size() < 2u ||
        restTwists.size() + 1u != restPositions.size() ||
        restLengths.size() != restTwists.size() ||
        nodeMasses.size() != restPositions.size() ||
        edgeRotationalInertias.size() != restTwists.size() ||
        stretchStiffness.size() != restTwists.size() ||
        bendStiffness.size() + 1u != restTwists.size() ||
        twistStiffness.size() != bendStiffness.size() ||
        !(radius > 0.0) || !finite(radius)) {
        return reject("rod dimensions or radius are invalid");
    }
    if (!std::ranges::all_of(
            restPositions,
            [](const Vec3 value) { return finite(value); }
        ) ||
        !std::ranges::all_of(
            restTwists,
            [](const double value) { return finite(value); }
        ) ||
        !std::ranges::all_of(
            restLengths,
            [](const double value) {
                return finite(value) && value > 0.0;
            }
        ) ||
        !std::ranges::all_of(
            nodeMasses,
            [](const double value) {
                return finite(value) && value > 0.0;
            }
        ) ||
        !std::ranges::all_of(
            edgeRotationalInertias,
            [](const double value) {
                return finite(value) && value > 0.0;
            }
        ) ||
        !std::ranges::all_of(
            stretchStiffness,
            [](const double value) {
                return finite(value) && value > 0.0;
            }
        ) ||
        !std::ranges::all_of(
            bendStiffness,
            [](const double value) {
                return finite(value) && value > 0.0;
            }
        ) ||
        !std::ranges::all_of(
            twistStiffness,
            [](const double value) {
                return finite(value) && value > 0.0;
            }
        )) {
        return reject("rod physical parameters are non-finite or non-positive");
    }
    for (std::size_t edge = 0u;
         edge < restLengths.size();
         ++edge) {
        const double measured = norm(
            subtract(
                restPositions[edge + 1u],
                restPositions[edge]
            )
        );
        if (std::abs(measured - restLengths[edge]) >
            1.0e-10 * std::max(measured, restLengths[edge])) {
            return reject("rod rest length disagrees with rest geometry");
        }
    }
    RodFrames frames;
    if (!buildFrames(restPositions, restTwists, frames)) {
        return reject("rod rest geometry is degenerate");
    }
    return true;
}

DiscreteElasticRodModel makeStraightSutureRod(
    const std::uint32_t nodeCount,
    const double length,
    const DiscreteRodMaterial& material
) {
    if (nodeCount < 2u ||
        !(length > 0.0) || !finite(length) ||
        !(material.radius > 0.0) ||
        !(material.density > 0.0) ||
        !(material.youngModulus > 0.0) ||
        !(material.poissonRatio > -1.0) ||
        !(material.poissonRatio < 0.5) ||
        !finite(material.radius) ||
        !finite(material.density) ||
        !finite(material.youngModulus) ||
        !finite(material.poissonRatio)) {
        throw std::invalid_argument(
            "straight suture rod configuration is invalid"
        );
    }
    DiscreteElasticRodModel model;
    model.name = "straight_suture_der_research";
    model.fidelityBoundary =
        "research material defaults; not package-calibrated or clinical";
    model.radius = material.radius;
    const std::size_t edgeCount = nodeCount - 1u;
    const double restLength =
        length / static_cast<double>(edgeCount);
    const double area =
        std::numbers::pi *
        material.radius * material.radius;
    const double secondMoment =
        std::numbers::pi *
        std::pow(material.radius, 4.0) / 4.0;
    const double polarMoment = 2.0 * secondMoment;
    const double shearModulus =
        material.youngModulus /
        (2.0 * (1.0 + material.poissonRatio));
    model.restPositions.resize(nodeCount);
    for (std::size_t node = 0u; node < nodeCount; ++node) {
        model.restPositions[node] = {
            restLength * static_cast<double>(node),
            0.0,
            0.0,
        };
    }
    model.restTwists.assign(edgeCount, 0.0);
    model.restLengths.assign(edgeCount, restLength);
    model.stretchStiffness.assign(
        edgeCount,
        material.youngModulus * area
    );
    model.bendStiffness.assign(
        edgeCount - 1u,
        material.youngModulus * secondMoment
    );
    model.twistStiffness.assign(
        edgeCount - 1u,
        shearModulus * polarMoment
    );
    model.edgeRotationalInertias.assign(
        edgeCount,
        material.density * polarMoment * restLength
    );
    model.nodeMasses.assign(nodeCount, 0.0);
    const double edgeMass =
        material.density * area * restLength;
    for (std::size_t edge = 0u; edge < edgeCount; ++edge) {
        model.nodeMasses[edge] += 0.5 * edgeMass;
        model.nodeMasses[edge + 1u] += 0.5 * edgeMass;
    }
    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "internal straight rod is invalid: " + reason
        );
    }
    return model;
}

DiscreteElasticRodState makeDiscreteElasticRodDefaultState(
    const DiscreteElasticRodModel& model
) {
    std::string reason;
    if (!model.valid(&reason)) {
        throw std::invalid_argument(
            "invalid rod model: " + reason
        );
    }
    DiscreteElasticRodState state;
    state.positions = model.restPositions;
    state.velocities.assign(
        model.restPositions.size(),
        Vec3{}
    );
    state.twists = model.restTwists;
    state.twistRates.assign(
        model.restTwists.size(),
        0.0
    );
    return state;
}

DiscreteElasticRodDiagnostics
evaluateDiscreteElasticRodEnergy(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& state,
    DiscreteElasticRodEnergy& output
) {
    DiscreteElasticRodDiagnostics diagnostics;
    std::string reason;
    if (!model.valid(&reason)) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::invalidModel,
            std::move(reason)
        );
    }
    if (!validState(model, state)) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::invalidState,
            "rod state dimensions or values are invalid"
        );
    }
    DiscreteElasticRodEnergy staged;
    if (!energy(model, state, staged)) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::degenerateGeometry,
            "rod frame or curvature evaluation is degenerate"
        );
    }
    output = staged;
    diagnostics.after = staged;
    return diagnostics;
}

DiscreteElasticRodDiagnostics stepDiscreteElasticRodCpu(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const std::span<const DiscreteRodAttachment> attachments,
    const DiscreteElasticRodStepConfig& config
) {
    DiscreteElasticRodDiagnostics diagnostics;
    std::string reason;
    if (!model.valid(&reason)) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::invalidModel,
            std::move(reason)
        );
    }
    if (!validConfig(config)) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::invalidConfiguration,
            "rod step configuration is invalid"
        );
    }
    if (!validState(model, state)) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::invalidState,
            "rod state dimensions or values are invalid"
        );
    }
    for (const DiscreteRodAttachment& attachment :
         attachments) {
        if (attachment.nodeIndex >=
                model.restPositions.size() ||
            !finite(attachment.targetPosition) ||
            !finite(attachment.targetVelocity) ||
            !finite(attachment.compliance) ||
            attachment.compliance < 0.0) {
            return fail(
                std::move(diagnostics),
                DiscreteElasticRodStatus::invalidAttachment,
                "rod attachment is invalid"
            );
        }
    }
    if (!energy(model, state, diagnostics.before)) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::degenerateGeometry,
            "initial rod geometry is degenerate"
        );
    }

    DiscreteElasticRodState candidate = state;
    const auto oldPositions = state.positions;
    const auto oldTwists = state.twists;
    const double h = config.timestep;
    for (std::size_t node = 0u;
         node < candidate.positions.size();
         ++node) {
        candidate.velocities[node] = add(
            candidate.velocities[node],
            multiply(config.gravity, h)
        );
        candidate.positions[node] = add(
            candidate.positions[node],
            multiply(candidate.velocities[node], h)
        );
    }
    for (std::size_t edge = 0u;
         edge < candidate.twists.size();
         ++edge) {
        candidate.twists[edge] +=
            h * candidate.twistRates[edge];
    }

    RodFrames restFrames;
    if (!buildFrames(
            model.restPositions,
            model.restTwists,
            restFrames
        )) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::invalidModel,
            "rod rest frames are invalid"
        );
    }

    bool converged = false;
    for (std::uint32_t iteration = 0u;
         iteration < config.solverIterations;
         ++iteration) {
        const auto iterationPositions =
            candidate.positions;
        const auto iterationTwists =
            candidate.twists;
        double maximumError = 0.0;
        for (std::size_t edge = 0u;
             edge < model.restLengths.size();
             ++edge) {
            if (!projectStretch(
                    model,
                    candidate,
                    h,
                    edge,
                    maximumError,
                    diagnostics.maximumPositionCorrection
                )) {
                return fail(
                    std::move(diagnostics),
                    DiscreteElasticRodStatus::degenerateGeometry,
                    "stretch projection encountered a zero edge"
                );
            }
            ++diagnostics.projectedStretchConstraints;
        }
        for (std::size_t vertex = 0u;
             vertex < model.bendStiffness.size();
             ++vertex) {
            for (std::size_t component = 0u;
                 component < 2u;
                 ++component) {
                if (!projectBendComponent(
                        model,
                        candidate,
                        restFrames,
                        config,
                        vertex,
                        component,
                        maximumError,
                        diagnostics.maximumPositionCorrection
                    )) {
                    return fail(
                        std::move(diagnostics),
                        DiscreteElasticRodStatus::degenerateGeometry,
                        "bend projection encountered a degenerate frame"
                    );
                }
                ++diagnostics.projectedBendConstraints;
            }
            if (!projectTwist(
                    model,
                    candidate,
                    config,
                    vertex,
                    maximumError
                )) {
                return fail(
                    std::move(diagnostics),
                    DiscreteElasticRodStatus::nonfiniteResult,
                    "twist projection became non-finite"
                );
            }
            ++diagnostics.projectedTwistConstraints;
        }
        for (const DiscreteRodAttachment& attachment :
             attachments) {
            if (!projectAttachment(
                    model,
                    candidate,
                    attachment,
                    h,
                    maximumError,
                    diagnostics.maximumPositionCorrection
                )) {
                return fail(
                    std::move(diagnostics),
                    DiscreteElasticRodStatus::nonfiniteResult,
                    "attachment projection became non-finite"
                );
            }
            ++diagnostics.projectedAttachments;
        }
        diagnostics.iterations = iteration + 1u;
        diagnostics.maximumConstraintError = maximumError;
        double iterationCorrection = 0.0;
        for (std::size_t node = 0u;
             node < candidate.positions.size();
             ++node) {
            iterationCorrection = std::max(
                iterationCorrection,
                norm(
                    subtract(
                        candidate.positions[node],
                        iterationPositions[node]
                    )
                )
            );
        }
        for (std::size_t edge = 0u;
             edge < candidate.twists.size();
             ++edge) {
            iterationCorrection = std::max(
                iterationCorrection,
                std::abs(
                    candidate.twists[edge] -
                    iterationTwists[edge]
                )
            );
        }
        if (iterationCorrection <=
            config.constraintTolerance) {
            converged = true;
            break;
        }
    }

    const double linearDecay =
        std::exp(-config.linearDamping * h);
    const double twistDecay =
        std::exp(-config.twistDamping * h);
    for (std::size_t node = 0u;
         node < candidate.positions.size();
         ++node) {
        candidate.velocities[node] = multiply(
            subtract(
                candidate.positions[node],
                oldPositions[node]
            ),
            linearDecay / h
        );
    }
    for (std::size_t edge = 0u;
         edge < candidate.twists.size();
         ++edge) {
        candidate.twistRates[edge] =
            (
                candidate.twists[edge] -
                oldTwists[edge]
            ) * twistDecay / h;
    }
    for (const DiscreteRodAttachment& attachment :
         attachments) {
        if (attachment.compliance == 0.0) {
            candidate.positions[attachment.nodeIndex] =
                attachment.targetPosition;
            candidate.velocities[attachment.nodeIndex] =
                attachment.targetVelocity;
        }
    }
    if (!validState(model, candidate) ||
        !energy(model, candidate, diagnostics.after)) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::nonfiniteResult,
            "candidate rod state or energy is non-finite"
        );
    }
    if (!converged) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::didNotConverge,
            "rod constraint solve exhausted its iteration budget"
        );
    }
    state = std::move(candidate);
    return diagnostics;
}

const char* discreteElasticRodStatusName(
    const DiscreteElasticRodStatus status
) noexcept {
    switch (status) {
    case DiscreteElasticRodStatus::success:
        return "success";
    case DiscreteElasticRodStatus::invalidConfiguration:
        return "invalid_configuration";
    case DiscreteElasticRodStatus::invalidModel:
        return "invalid_model";
    case DiscreteElasticRodStatus::invalidState:
        return "invalid_state";
    case DiscreteElasticRodStatus::invalidAttachment:
        return "invalid_attachment";
    case DiscreteElasticRodStatus::degenerateGeometry:
        return "degenerate_geometry";
    case DiscreteElasticRodStatus::didNotConverge:
        return "did_not_converge";
    case DiscreteElasticRodStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
