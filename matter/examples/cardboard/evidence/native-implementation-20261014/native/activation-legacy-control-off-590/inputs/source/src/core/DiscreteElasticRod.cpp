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

struct RodReferenceFrames {
    std::vector<Vec3> tangents;
    std::vector<Vec3> directors;
};

bool buildReferenceFrames(
    const std::span<const Vec3> restPositions,
    RodReferenceFrames& references
) {
    if (restPositions.size() < 2u) {
        return false;
    }
    const std::size_t edgeCount = restPositions.size() - 1u;
    references = {};
    references.tangents.resize(edgeCount);
    references.directors.resize(edgeCount);
    for (std::size_t edge = 0u; edge < edgeCount; ++edge) {
        if (!normalize(
                subtract(
                    restPositions[edge + 1u],
                    restPositions[edge]
                ),
                references.tangents[edge]
            )) {
            return false;
        }
    }
    references.directors[0] =
        leastAlignedDirector(references.tangents[0]);
    for (std::size_t edge = 1u; edge < edgeCount; ++edge) {
        if (!transport(
                references.directors[edge - 1u],
                references.tangents[edge - 1u],
                references.tangents[edge],
                references.directors[edge]
            )) {
            return false;
        }
    }
    return true;
}

bool buildFrames(
    const std::span<const Vec3> positions,
    const std::span<const double> twists,
    const RodReferenceFrames& references,
    RodFrames& frames
) {
    const std::size_t edgeCount = positions.size() - 1u;
    if (twists.size() != edgeCount ||
        references.tangents.size() != edgeCount ||
        references.directors.size() != edgeCount) {
        return false;
    }
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
    Vec3 reference;
    if (!transport(
            references.directors[0],
            references.tangents[0],
            frames.tangents[0],
            reference
        )) {
        return false;
    }
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
    RodReferenceFrames references;
    RodFrames restFrames;
    RodFrames frames;
    if (!buildReferenceFrames(
            model.restPositions,
            references
        ) ||
        !buildFrames(
            model.restPositions,
            model.restTwists,
            references,
            restFrames
        ) ||
        !buildFrames(
            state.positions,
            state.twists,
            references,
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
        config.derivativeStep > 0.0 &&
        finite(config.selfCollisionMargin) &&
        config.selfCollisionMargin >= 0.0 &&
        finite(config.selfCollisionCompliance) &&
        config.selfCollisionCompliance >= 0.0 &&
        finite(config.selfCollisionFriction) &&
        config.selfCollisionFriction >= 0.0;
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
    double& accumulatedMultiplier,
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
        (-constraint - alpha * accumulatedMultiplier) /
        (inverseA + inverseB + alpha);
    accumulatedMultiplier += lambda;
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
    const std::array<Vec3, 2>& referenceTangents,
    const std::array<Vec3, 2>& referenceDirectors,
    std::array<double, 2>& curvature
) {
    RodReferenceFrames references{
        .tangents = {
            referenceTangents[0],
            referenceTangents[1],
        },
        .directors = {
            referenceDirectors[0],
            referenceDirectors[1],
        },
    };
    RodFrames frames;
    if (!buildFrames(positions, twists, references, frames) ||
        frames.curvature.size() != 1u) {
        return false;
    }
    curvature = frames.curvature[0];
    return true;
}

bool projectBend(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const RodReferenceFrames& references,
    const RodFrames& restFrames,
    const DiscreteElasticRodStepConfig& config,
    const std::size_t vertex,
    std::array<double, 2>& accumulatedMultiplier,
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
    const std::array<Vec3, 2> referenceTangents{{
        references.tangents[vertex],
        references.tangents[vertex + 1u],
    }};
    const std::array<Vec3, 2> referenceDirectors{{
        references.directors[vertex],
        references.directors[vertex + 1u],
    }};
    std::array<double, 2> current{};
    if (!localCurvature(
            localPositions,
            localTwists,
            referenceTangents,
            referenceDirectors,
            current
        )) {
        return false;
    }
    const std::array<double, 2> constraint{{
        current[0] - restFrames.curvature[vertex][0],
        current[1] - restFrames.curvature[vertex][1],
    }};
    const bool zeroIntrinsicCurvature =
        restFrames.curvature[vertex][0] *
                restFrames.curvature[vertex][0] +
            restFrames.curvature[vertex][1] *
                restFrames.curvature[vertex][1] <=
        1.0e-24;
    maximumError = std::max(
        maximumError,
        std::max(
            std::abs(constraint[0]),
            std::abs(constraint[1])
        )
    );

    std::array<std::array<Vec3, 3>, 2> positionGradient{};
    std::array<std::array<double, 2>, 2> twistGradient{};
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
                referenceTangents,
                referenceDirectors,
                plus
            );
            localPositions[node][axis] -= 2.0 * step;
            std::array<double, 2> minus{};
            const bool minusOk = localCurvature(
                localPositions,
                localTwists,
                referenceTangents,
                referenceDirectors,
                minus
            );
            localPositions[node][axis] += step;
            if (!plusOk || !minusOk) {
                return false;
            }
            for (std::size_t component = 0u;
                 component < 2u;
                 ++component) {
                positionGradient[component][node][axis] =
                    (plus[component] - minus[component]) /
                    (2.0 * step);
            }
        }
    }
    // A circular rod with a straight intrinsic centerline has isotropic
    // bending energy: spinning its material frame cannot change EI*kappa^2.
    // Leaving finite-difference twist gradients in this block lets roundoff
    // pump the extremely small polar inertia. Curved intrinsic rods retain
    // the full material-frame coupling below.
    if (!zeroIntrinsicCurvature) {
        for (std::size_t edge = 0u; edge < 2u; ++edge) {
            const double step = config.derivativeStep;
            localTwists[edge] += step;
            std::array<double, 2> plus{};
            const bool plusOk = localCurvature(
                localPositions,
                localTwists,
                referenceTangents,
                referenceDirectors,
                plus
            );
            localTwists[edge] -= 2.0 * step;
            std::array<double, 2> minus{};
            const bool minusOk = localCurvature(
                localPositions,
                localTwists,
                referenceTangents,
                referenceDirectors,
                minus
            );
            localTwists[edge] += step;
            if (!plusOk || !minusOk) {
                return false;
            }
            for (std::size_t component = 0u;
                 component < 2u;
                 ++component) {
                twistGradient[component][edge] =
                    (plus[component] - minus[component]) /
                    (2.0 * step);
            }
        }
    }

    double effective00 = 0.0;
    double effective01 = 0.0;
    double effective11 = 0.0;
    for (std::size_t node = 0u; node < 3u; ++node) {
        const double inverseMass =
            1.0 / model.nodeMasses[vertex + node];
        effective00 += inverseMass * dot(
            positionGradient[0][node],
            positionGradient[0][node]
        );
        effective01 += inverseMass * dot(
            positionGradient[0][node],
            positionGradient[1][node]
        );
        effective11 += inverseMass * dot(
            positionGradient[1][node],
            positionGradient[1][node]
        );
    }
    if (!zeroIntrinsicCurvature) {
        for (std::size_t edge = 0u; edge < 2u; ++edge) {
            const double inverseInertia =
                1.0 / model.edgeRotationalInertias[vertex + edge];
            effective00 += inverseInertia *
                twistGradient[0][edge] * twistGradient[0][edge];
            effective01 += inverseInertia *
                twistGradient[0][edge] * twistGradient[1][edge];
            effective11 += inverseInertia *
                twistGradient[1][edge] * twistGradient[1][edge];
        }
    }
    const double voronoi =
        0.5 * (
            model.restLengths[vertex] +
            model.restLengths[vertex + 1u]
        );
    const double alpha =
        voronoi / model.bendStiffness[vertex] /
        (config.timestep * config.timestep);
    effective00 += alpha;
    effective11 += alpha;
    const double determinant =
        effective00 * effective11 - effective01 * effective01;
    const double determinantScale =
        std::max(effective00 * effective11, 1.0);
    if (!(effective00 > 0.0) ||
        !(effective11 > 0.0) ||
        !(determinant >
            32.0 * std::numeric_limits<double>::epsilon() *
                determinantScale) ||
        !finite(effective00) ||
        !finite(effective01) ||
        !finite(effective11) ||
        !finite(determinant)) {
        return false;
    }
    const std::array<double, 2> rhs{{
        constraint[0] + alpha * accumulatedMultiplier[0],
        constraint[1] + alpha * accumulatedMultiplier[1],
    }};
    const std::array<double, 2> lambda{{
        (-effective11 * rhs[0] + effective01 * rhs[1]) /
            determinant,
        (effective01 * rhs[0] - effective00 * rhs[1]) /
            determinant,
    }};
    accumulatedMultiplier[0] += lambda[0];
    accumulatedMultiplier[1] += lambda[1];
    for (std::size_t node = 0u; node < 3u; ++node) {
        const Vec3 correction = multiply(
            add(
                multiply(positionGradient[0][node], lambda[0]),
                multiply(positionGradient[1][node], lambda[1])
            ),
            1.0 / model.nodeMasses[vertex + node]
        );
        state.positions[vertex + node] =
            add(state.positions[vertex + node], correction);
        maximumCorrection = std::max(
            maximumCorrection,
            norm(correction)
        );
    }
    if (!zeroIntrinsicCurvature) {
        for (std::size_t edge = 0u; edge < 2u; ++edge) {
            state.twists[vertex + edge] +=
                (
                    lambda[0] * twistGradient[0][edge] +
                    lambda[1] * twistGradient[1][edge]
                ) /
                model.edgeRotationalInertias[vertex + edge];
        }
    }
    return true;
}

bool projectTwist(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const DiscreteElasticRodStepConfig& config,
    const std::size_t vertex,
    double& accumulatedMultiplier,
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
        (-constraint - alpha * accumulatedMultiplier) /
        (inverseA + inverseB + alpha);
    accumulatedMultiplier += lambda;
    state.twists[vertex] -= inverseA * lambda;
    state.twists[vertex + 1u] += inverseB * lambda;
    return finite(state.twists[vertex]) &&
        finite(state.twists[vertex + 1u]);
}

struct ClosestSegments {
    double first = 0.0;
    double second = 0.0;
    Vec3 delta{};
    double distance = 0.0;
};

bool closestSegments(
    const Vec3 firstA,
    const Vec3 firstB,
    const Vec3 secondA,
    const Vec3 secondB,
    ClosestSegments& result
) {
    const Vec3 firstDirection =
        subtract(firstB, firstA);
    const Vec3 secondDirection =
        subtract(secondB, secondA);
    const Vec3 offset =
        subtract(firstA, secondA);
    const double firstLengthSquared =
        dot(firstDirection, firstDirection);
    const double secondLengthSquared =
        dot(secondDirection, secondDirection);
    if (!(firstLengthSquared > 1.0e-28) ||
        !(secondLengthSquared > 1.0e-28)) {
        return false;
    }
    const double firstOffset =
        dot(firstDirection, offset);
    const double secondOffset =
        dot(secondDirection, offset);
    const double coupling =
        dot(firstDirection, secondDirection);
    const double denominator =
        firstLengthSquared * secondLengthSquared -
        coupling * coupling;
    if (denominator >
        1.0e-14 * firstLengthSquared *
            secondLengthSquared) {
        result.first = std::clamp(
            (
                coupling * secondOffset -
                firstOffset * secondLengthSquared
            ) / denominator,
            0.0,
            1.0
        );
    } else {
        result.first = 0.0;
    }
    double secondNumerator =
        coupling * result.first + secondOffset;
    if (secondNumerator < 0.0) {
        result.second = 0.0;
        result.first = std::clamp(
            -firstOffset / firstLengthSquared,
            0.0,
            1.0
        );
    } else if (secondNumerator >
               secondLengthSquared) {
        result.second = 1.0;
        result.first = std::clamp(
            (
                coupling - firstOffset
            ) / firstLengthSquared,
            0.0,
            1.0
        );
    } else {
        result.second =
            secondNumerator / secondLengthSquared;
    }
    const Vec3 firstPoint = add(
        firstA,
        multiply(firstDirection, result.first)
    );
    const Vec3 secondPoint = add(
        secondA,
        multiply(secondDirection, result.second)
    );
    result.delta = subtract(secondPoint, firstPoint);
    result.distance = norm(result.delta);
    return finite(result.first) &&
        finite(result.second) &&
        finite(result.delta) &&
        finite(result.distance);
}

bool projectSelfContact(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const DiscreteElasticRodStepConfig& config,
    const std::size_t firstEdge,
    const std::size_t secondEdge,
    bool& projected,
    double& maximumError,
    double& maximumPenetration,
    double& maximumCorrection
) {
    projected = false;
    ClosestSegments closest;
    if (!closestSegments(
            state.positions[firstEdge],
            state.positions[firstEdge + 1u],
            state.positions[secondEdge],
            state.positions[secondEdge + 1u],
            closest
        )) {
        return false;
    }
    const double contactDistance =
        2.0 * model.radius +
        config.selfCollisionMargin;
    const double penetration =
        contactDistance - closest.distance;
    if (!(penetration > 0.0)) {
        return true;
    }
    Vec3 normal;
    if (closest.distance > 1.0e-14) {
        normal = multiply(
            closest.delta,
            1.0 / closest.distance
        );
    } else {
        Vec3 firstDirection;
        Vec3 secondDirection;
        if (!normalize(
                subtract(
                    state.positions[firstEdge + 1u],
                    state.positions[firstEdge]
                ),
                firstDirection
            ) ||
            !normalize(
                subtract(
                    state.positions[secondEdge + 1u],
                    state.positions[secondEdge]
                ),
                secondDirection
            )) {
            return false;
        }
        normal = cross(firstDirection, secondDirection);
        if (!normalize(normal, normal)) {
            normal = leastAlignedDirector(firstDirection);
        }
        if (((firstEdge ^ secondEdge) & 1u) != 0u) {
            normal = multiply(normal, -1.0);
        }
    }
    const std::array<double, 4> weights{
        1.0 - closest.first,
        closest.first,
        1.0 - closest.second,
        closest.second,
    };
    const std::array<std::size_t, 4> nodes{
        firstEdge,
        firstEdge + 1u,
        secondEdge,
        secondEdge + 1u,
    };
    double denominator = 0.0;
    for (std::size_t slot = 0u;
         slot < nodes.size();
         ++slot) {
        denominator +=
            weights[slot] * weights[slot] /
            model.nodeMasses[nodes[slot]];
    }
    const double alpha =
        config.selfCollisionCompliance /
        (config.timestep * config.timestep);
    if (!(denominator + alpha > 0.0) ||
        !finite(denominator)) {
        return false;
    }
    const double lambda =
        penetration / (denominator + alpha);
    for (std::size_t slot = 0u;
         slot < nodes.size();
         ++slot) {
        const double sign = slot < 2u ? -1.0 : 1.0;
        const Vec3 correction = multiply(
            normal,
            sign * weights[slot] * lambda /
                model.nodeMasses[nodes[slot]]
        );
        state.positions[nodes[slot]] = add(
            state.positions[nodes[slot]],
            correction
        );
        maximumCorrection = std::max(
            maximumCorrection,
            norm(correction)
        );
    }
    maximumError = std::max(maximumError, penetration);
    maximumPenetration = std::max(
        maximumPenetration,
        penetration
    );
    projected = true;
    return std::ranges::all_of(
        nodes,
        [&](const std::size_t node) {
            return finite(state.positions[node]);
        }
    );
}

bool applySelfContactFriction(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const DiscreteElasticRodStepConfig& config,
    const std::span<const Vec3> unconstrainedVelocities
) {
    struct FrictionContact {
        std::size_t firstEdge = 0u;
        std::size_t secondEdge = 0u;
        Vec3 accumulatedImpulse{};
        double normalImpulse = 0.0;
    };
    const double contactDistance =
        2.0 * model.radius + config.selfCollisionMargin;
    const double contactThreshold =
        contactDistance + config.constraintTolerance;
    const auto contactKinematics = [&model, &state](
        const std::size_t firstEdge,
        const std::size_t secondEdge,
        ClosestSegments& closest,
        Vec3& normal,
        std::array<double, 4>& weights,
        std::array<std::size_t, 4>& nodes,
        double& inverseEffectiveMass
    ) {
        if (!closestSegments(
                state.positions[firstEdge],
                state.positions[firstEdge + 1u],
                state.positions[secondEdge],
                state.positions[secondEdge + 1u],
                closest
            )) {
            return false;
        }
        const Vec3 firstDirection = subtract(
            state.positions[firstEdge + 1u],
            state.positions[firstEdge]
        );
        const Vec3 secondDirection = subtract(
            state.positions[secondEdge + 1u],
            state.positions[secondEdge]
        );
        if (closest.distance > 1.0e-14) {
            normal = multiply(
                closest.delta,
                1.0 / closest.distance
            );
        } else {
            normal = cross(firstDirection, secondDirection);
            if (!normalize(normal, normal)) {
                Vec3 firstTangent;
                if (!normalize(firstDirection, firstTangent)) {
                    return false;
                }
                normal = leastAlignedDirector(firstTangent);
            }
            if (((firstEdge ^ secondEdge) & 1u) != 0u) {
                normal = multiply(normal, -1.0);
            }
        }
        weights = {
            1.0 - closest.first,
            closest.first,
            1.0 - closest.second,
            closest.second,
        };
        nodes = {
            firstEdge,
            firstEdge + 1u,
            secondEdge,
            secondEdge + 1u,
        };
        inverseEffectiveMass = 0.0;
        for (std::size_t slot = 0u; slot < nodes.size(); ++slot) {
            inverseEffectiveMass +=
                weights[slot] * weights[slot] /
                model.nodeMasses[nodes[slot]];
        }
        return inverseEffectiveMass > 0.0 &&
            finite(inverseEffectiveMass) && finite(normal);
    };
    const auto relativeVelocity = [](
        const std::span<const Vec3> velocities,
        const std::array<double, 4>& weights,
        const std::array<std::size_t, 4>& nodes
    ) {
        Vec3 first{};
        Vec3 second{};
        for (std::size_t slot = 0u; slot < nodes.size(); ++slot) {
            if (slot < 2u) {
                first = add(
                    first,
                    multiply(velocities[nodes[slot]], weights[slot])
                );
            } else {
                second = add(
                    second,
                    multiply(velocities[nodes[slot]], weights[slot])
                );
            }
        }
        return subtract(second, first);
    };

    std::vector<FrictionContact> contacts;
    for (std::size_t firstEdge = 0u;
         firstEdge < model.restLengths.size();
         ++firstEdge) {
        for (std::size_t secondEdge = firstEdge + 2u;
             secondEdge < model.restLengths.size();
             ++secondEdge) {
            ClosestSegments closest;
            Vec3 normal;
            std::array<double, 4> weights{};
            std::array<std::size_t, 4> nodes{};
            double inverseEffectiveMass = 0.0;
            if (!contactKinematics(
                    firstEdge,
                    secondEdge,
                    closest,
                    normal,
                    weights,
                    nodes,
                    inverseEffectiveMass
                )) {
                return false;
            }
            if (closest.distance > contactThreshold) {
                continue;
            }
            const Vec3 constrainedRelativeVelocity = relativeVelocity(
                state.velocities,
                weights,
                nodes
            );
            const Vec3 unconstrainedRelativeVelocity = relativeVelocity(
                unconstrainedVelocities,
                weights,
                nodes
            );
            const double normalImpulse = std::max(
                dot(
                    subtract(
                        constrainedRelativeVelocity,
                        unconstrainedRelativeVelocity
                    ),
                    normal
                ),
                0.0
            ) / inverseEffectiveMass;
            if (normalImpulse > 0.0) {
                contacts.push_back({
                    .firstEdge = firstEdge,
                    .secondEdge = secondEdge,
                    .normalImpulse = normalImpulse,
                });
            }
        }
    }

    constexpr std::size_t frictionIterations = 8u;
    for (std::size_t iteration = 0u;
         iteration < frictionIterations;
         ++iteration) {
        for (std::size_t order = 0u;
             order < contacts.size();
             ++order) {
            const std::size_t contactIndex =
                (iteration & 1u) == 0u
                ? order
                : contacts.size() - 1u - order;
            FrictionContact& contact = contacts[contactIndex];
            ClosestSegments closest;
            Vec3 normal;
            std::array<double, 4> weights{};
            std::array<std::size_t, 4> nodes{};
            double inverseEffectiveMass = 0.0;
            if (!contactKinematics(
                    contact.firstEdge,
                    contact.secondEdge,
                    closest,
                    normal,
                    weights,
                    nodes,
                    inverseEffectiveMass
                )) {
                return false;
            }
            const Vec3 currentRelativeVelocity = relativeVelocity(
                state.velocities,
                weights,
                nodes
            );
            const Vec3 tangentVelocity = subtract(
                currentRelativeVelocity,
                multiply(
                    normal,
                    dot(currentRelativeVelocity, normal)
                )
            );
            Vec3 projectedImpulse = subtract(
                contact.accumulatedImpulse,
                multiply(
                    tangentVelocity,
                    1.0 / inverseEffectiveMass
                )
            );
            const double frictionLimit =
                config.selfCollisionFriction *
                contact.normalImpulse;
            const double projectedMagnitude = norm(projectedImpulse);
            if (projectedMagnitude > frictionLimit &&
                projectedMagnitude > 0.0) {
                projectedImpulse = multiply(
                    projectedImpulse,
                    frictionLimit / projectedMagnitude
                );
            }
            const Vec3 deltaImpulse = subtract(
                projectedImpulse,
                contact.accumulatedImpulse
            );
            contact.accumulatedImpulse = projectedImpulse;
            for (std::size_t slot = 0u;
                 slot < nodes.size();
                 ++slot) {
                const double sign = slot < 2u ? -1.0 : 1.0;
                state.velocities[nodes[slot]] = add(
                    state.velocities[nodes[slot]],
                    multiply(
                        deltaImpulse,
                        sign * weights[slot] /
                            model.nodeMasses[nodes[slot]]
                    )
                );
                if (!finite(state.velocities[nodes[slot]])) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool projectAttachment(
    const DiscreteElasticRodModel& model,
    DiscreteElasticRodState& state,
    const DiscreteRodAttachment& attachment,
    const double timestep,
    Vec3& accumulatedMultiplier,
    Vec3& impulseOnTarget,
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
    const Vec3 deltaMultiplier = multiply(
        add(delta, multiply(accumulatedMultiplier, alpha)),
        -1.0 / (inverseMass + alpha)
    );
    accumulatedMultiplier = add(
        accumulatedMultiplier,
        deltaMultiplier
    );
    const Vec3 correction = multiply(
        deltaMultiplier,
        inverseMass
    );
    state.positions[attachment.nodeIndex] =
        add(
            state.positions[attachment.nodeIndex],
            correction
        );
    // Convert the positional correction into its step impulse and expose the
    // equal-and-opposite support reaction. Repeated ordered projections are
    // accumulated so internal rod forces reaching the anchor are retained.
    impulseOnTarget = subtract(
        impulseOnTarget,
        multiply(
            correction,
            1.0 / (inverseMass * timestep)
        )
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
    RodReferenceFrames references;
    RodFrames frames;
    if (!buildReferenceFrames(restPositions, references) ||
        !buildFrames(
            restPositions,
            restTwists,
            references,
            frames
        )) {
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
    const DiscreteElasticRodStepConfig& config,
    const std::span<DiscreteRodAttachmentReaction> reactions
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
    if (!reactions.empty() &&
        reactions.size() != attachments.size()) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::invalidAttachment,
            "rod reaction output must match attachment count"
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
    std::vector<DiscreteRodAttachmentReaction>
        candidateReactions(attachments.size());
    for (std::size_t attachmentIndex = 0u;
         attachmentIndex < attachments.size();
         ++attachmentIndex) {
        candidateReactions[attachmentIndex].nodeIndex =
            attachments[attachmentIndex].nodeIndex;
    }
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

    RodReferenceFrames references;
    RodFrames restFrames;
    if (!buildReferenceFrames(
            model.restPositions,
            references
        ) ||
        !buildFrames(
            model.restPositions,
            model.restTwists,
            references,
            restFrames
        )) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::invalidModel,
            "rod rest frames are invalid"
        );
    }

    // XPBD compliance is iteration-count independent only when every
    // constraint keeps its Lagrange multiplier for the whole time step.  A
    // zeroed multiplier per nonlinear sweep turns these projections back into
    // an order-dependent PBD iteration and can stall under a stiff swage load.
    std::vector<double> stretchMultipliers(
        model.restLengths.size(),
        0.0
    );
    std::vector<std::array<double, 2>> bendMultipliers(
        model.bendStiffness.size(),
        std::array<double, 2>{0.0, 0.0}
    );
    std::vector<double> twistMultipliers(
        model.twistStiffness.size(),
        0.0
    );
    std::vector<Vec3> attachmentMultipliers(
        attachments.size(),
        Vec3{}
    );

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
                    stretchMultipliers[edge],
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
            if (!projectBend(
                    model,
                    candidate,
                    references,
                    restFrames,
                    config,
                    vertex,
                    bendMultipliers[vertex],
                    maximumError,
                    diagnostics.maximumPositionCorrection
                )) {
                return fail(
                    std::move(diagnostics),
                    DiscreteElasticRodStatus::degenerateGeometry,
                    "bend projection encountered a degenerate frame"
                );
            }
            diagnostics.projectedBendConstraints += 2u;
            if (!projectTwist(
                    model,
                    candidate,
                    config,
                    vertex,
                    twistMultipliers[vertex],
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
        if (config.enableSelfCollision) {
            for (std::size_t firstEdge = 0u;
                 firstEdge < model.restLengths.size();
                 ++firstEdge) {
                for (
                    std::size_t secondEdge = firstEdge + 2u;
                    secondEdge < model.restLengths.size();
                    ++secondEdge
                ) {
                    bool projected = false;
                    if (!projectSelfContact(
                            model,
                            candidate,
                            config,
                            firstEdge,
                            secondEdge,
                            projected,
                            maximumError,
                            diagnostics.maximumSelfPenetration,
                            diagnostics.maximumPositionCorrection
                        )) {
                        return fail(
                            std::move(diagnostics),
                            DiscreteElasticRodStatus::
                                degenerateGeometry,
                            "self-contact projection encountered "
                            "a degenerate edge"
                        );
                    }
                    if (projected) {
                        ++diagnostics.projectedSelfContacts;
                    }
                }
            }
        }
        for (std::size_t attachmentIndex = 0u;
             attachmentIndex < attachments.size();
             ++attachmentIndex) {
            const DiscreteRodAttachment& attachment =
                attachments[attachmentIndex];
            if (!projectAttachment(
                    model,
                    candidate,
                    attachment,
                    h,
                    attachmentMultipliers[attachmentIndex],
                    candidateReactions[
                        attachmentIndex
                    ].impulseOnTarget,
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
        double iterationPositionCorrection = 0.0;
        for (std::size_t node = 0u;
             node < candidate.positions.size();
             ++node) {
            iterationPositionCorrection = std::max(
                iterationPositionCorrection,
                norm(
                    subtract(
                        candidate.positions[node],
                        iterationPositions[node]
                    )
                )
            );
        }
        double iterationTwistCorrection = 0.0;
        for (std::size_t edge = 0u;
             edge < candidate.twists.size();
             ++edge) {
            iterationTwistCorrection = std::max(
                iterationTwistCorrection,
                std::abs(
                    candidate.twists[edge] -
                    iterationTwists[edge]
                )
            );
        }
        diagnostics.maximumPositionCorrection = std::max(
            diagnostics.maximumPositionCorrection,
            model.radius * iterationTwistCorrection
        );
        if (std::max(
                iterationPositionCorrection,
                model.radius * iterationTwistCorrection
            ) <= config.constraintTolerance) {
            converged = true;
            break;
        }
    }

    const double linearDecay =
        std::exp(-config.linearDamping * h);
    const double twistDecay =
        std::exp(-config.twistDamping * h);
    std::vector<Vec3> unconstrainedVelocities =
        candidate.velocities;
    for (Vec3& velocity : unconstrainedVelocities) {
        velocity = multiply(velocity, linearDecay);
    }
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
    if (config.enableSelfCollision &&
        config.selfCollisionFriction > 0.0 &&
        diagnostics.projectedSelfContacts > 0u &&
        !applySelfContactFriction(
            model,
            candidate,
            config,
            unconstrainedVelocities
        )) {
        return fail(
            std::move(diagnostics),
            DiscreteElasticRodStatus::nonfiniteResult,
            "self-contact friction encountered a degenerate pair"
        );
    }
    for (std::size_t attachmentIndex = 0u;
         attachmentIndex < attachments.size();
         ++attachmentIndex) {
        const DiscreteRodAttachment& attachment =
            attachments[attachmentIndex];
        DiscreteRodAttachmentReaction& reaction =
            candidateReactions[attachmentIndex];
        if (attachment.compliance == 0.0) {
            const Vec3 velocityCorrection = subtract(
                attachment.targetVelocity,
                candidate.velocities[attachment.nodeIndex]
            );
            reaction.impulseOnTarget = subtract(
                reaction.impulseOnTarget,
                multiply(
                    velocityCorrection,
                    model.nodeMasses[attachment.nodeIndex]
                )
            );
            candidate.positions[attachment.nodeIndex] =
                attachment.targetPosition;
            candidate.velocities[attachment.nodeIndex] =
                attachment.targetVelocity;
        }
        reaction.finalPositionError = norm(subtract(
            candidate.positions[attachment.nodeIndex],
            attachment.targetPosition
        ));
        reaction.averageForceOnTarget = multiply(
            reaction.impulseOnTarget,
            1.0 / h
        );
        if (!finite(reaction.impulseOnTarget) ||
            !finite(reaction.averageForceOnTarget) ||
            !finite(reaction.finalPositionError)) {
            return fail(
                std::move(diagnostics),
                DiscreteElasticRodStatus::nonfiniteResult,
                "attachment reaction became non-finite"
            );
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
    if (!reactions.empty()) {
        std::ranges::copy(candidateReactions, reactions.begin());
    }
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
