#include "metalrobo/HeterogeneousWorld.hpp"
#include "metalrobo/SurgicalPSM.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numbers>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

using Vec3 = std::array<double, 3>;

bool setReason(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return
        std::isfinite(value.x) &&
        std::isfinite(value.y) &&
        std::isfinite(value.z) &&
        std::isfinite(value.w);
}

bool finite(const Vec3& value) {
    return std::ranges::all_of(value, [](const double item) {
        return finite(item);
    });
}

bool validMaterial(const MRMaterialGPU& material) {
    return
        finite(material.friction) &&
        finite(material.response) &&
        finite(material.geometry) &&
        material.friction.x >= material.friction.y &&
        material.friction.y >= 0.0f &&
        material.friction.z >= 0.0f &&
        material.friction.w >= 0.0f &&
        material.response.x >= 0.0f &&
        material.response.x <= 1.0f &&
        material.response.y >= 0.0f &&
        material.response.z >= 0.0f &&
        material.response.w >= 0.0f &&
        material.geometry.x >= 0.0f &&
        material.geometry.y >= 0.0f;
}

bool equal(const mr_float4 left, const mr_float4 right) {
    return
        left.x == right.x &&
        left.y == right.y &&
        left.z == right.z &&
        left.w == right.w;
}

bool equal(
    const MRMaterialGPU& left,
    const MRMaterialGPU& right
) {
    return
        equal(left.friction, right.friction) &&
        equal(left.response, right.response) &&
        equal(left.geometry, right.geometry);
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
    return
        left[0] * right[0] +
        left[1] * right[1] +
        left[2] * right[2];
}

double segmentSegmentDistance(
    const Vec3& first0,
    const Vec3& first1,
    const Vec3& second0,
    const Vec3& second1
) {
    constexpr double kDegenerateSquared = 1.0e-24;
    const Vec3 firstDirection = subtract(first1, first0);
    const Vec3 secondDirection = subtract(second1, second0);
    const Vec3 offset = subtract(first0, second0);
    const double firstLengthSquared = dot(firstDirection, firstDirection);
    const double secondLengthSquared = dot(secondDirection, secondDirection);
    const double secondProjection = dot(secondDirection, offset);
    double firstParameter = 0.0;
    double secondParameter = 0.0;
    if (firstLengthSquared <= kDegenerateSquared &&
        secondLengthSquared <= kDegenerateSquared) {
        return std::sqrt(dot(offset, offset));
    }
    if (firstLengthSquared <= kDegenerateSquared) {
        secondParameter = std::clamp(
            secondProjection / secondLengthSquared,
            0.0,
            1.0
        );
    } else {
        const double firstProjection = dot(firstDirection, offset);
        if (secondLengthSquared <= kDegenerateSquared) {
            firstParameter = std::clamp(
                -firstProjection / firstLengthSquared,
                0.0,
                1.0
            );
        } else {
            const double coupling = dot(
                firstDirection,
                secondDirection
            );
            const double denominator =
                firstLengthSquared * secondLengthSquared -
                coupling * coupling;
            if (denominator > kDegenerateSquared) {
                firstParameter = std::clamp(
                    (
                        coupling * secondProjection -
                        firstProjection * secondLengthSquared
                    ) / denominator,
                    0.0,
                    1.0
                );
            }
            secondParameter =
                (coupling * firstParameter + secondProjection) /
                secondLengthSquared;
            if (secondParameter < 0.0) {
                secondParameter = 0.0;
                firstParameter = std::clamp(
                    -firstProjection / firstLengthSquared,
                    0.0,
                    1.0
                );
            } else if (secondParameter > 1.0) {
                secondParameter = 1.0;
                firstParameter = std::clamp(
                    (coupling - firstProjection) /
                        firstLengthSquared,
                    0.0,
                    1.0
                );
            }
        }
    }
    const Vec3 closest = subtract(
        add(offset, multiply(firstDirection, firstParameter)),
        multiply(secondDirection, secondParameter)
    );
    return std::sqrt(dot(closest, closest));
}

Vec3 cross(const Vec3& left, const Vec3& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

double norm(const Vec3& value) {
    return std::sqrt(
        value[0] * value[0] +
        value[1] * value[1] +
        value[2] * value[2]
    );
}

bool normalize(const Vec3& value, Vec3& result) {
    const double magnitude = norm(value);
    if (!(magnitude > 1.0e-12) || !finite(magnitude)) {
        return false;
    }
    result = multiply(value, 1.0 / magnitude);
    return finite(result);
}

bool transportDirector(
    const Vec3& director,
    const Vec3& from,
    const Vec3& to,
    Vec3& result
) {
    const double cosine = std::clamp(dot(from, to), -1.0, 1.0);
    if (!(cosine > -1.0 + 1.0e-10)) {
        return false;
    }
    const Vec3 axis = cross(from, to);
    const Vec3 first = cross(axis, director);
    const Vec3 second = cross(axis, first);
    result = add(
        add(director, first),
        multiply(second, 1.0 / (1.0 + cosine))
    );
    result = subtract(result, multiply(to, dot(result, to)));
    return normalize(result, result);
}

Vec3 rotate(const mr_float4 quaternion, const Vec3& value) {
    const Vec3 imaginary{
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
    return add(
        value,
        multiply(
            cross(
                imaginary,
                add(
                    cross(imaginary, value),
                    multiply(value, quaternion.w)
                )
            ),
            2.0
        )
    );
}

mr_float4 conjugate(const mr_float4 quaternion) {
    return {
        -quaternion.x,
        -quaternion.y,
        -quaternion.z,
        quaternion.w,
    };
}

Vec3 transformPointToLocal(
    const SurgicalBasePose& pose,
    const Vec3& point
) {
    return rotate(
        conjugate({
            pose.orientation[0],
            pose.orientation[1],
            pose.orientation[2],
            pose.orientation[3],
        }),
        subtract(point, {
            pose.position[0],
            pose.position[1],
            pose.position[2],
        })
    );
}

Vec3 transformPointToWorld(
    const SurgicalBasePose& pose,
    const Vec3& point
) {
    return add(
        {
            pose.position[0],
            pose.position[1],
            pose.position[2],
        },
        rotate({
            pose.orientation[0],
            pose.orientation[1],
            pose.orientation[2],
            pose.orientation[3],
        }, point)
    );
}

void layThreadOnNeutralZone(
    DualPsmNeedleThreadWorld& surgical,
    const DualPsmNeedleThreadNeutralZoneConfig& config
) {
    DiscreteElasticRodModel& model = surgical.threadModel;
    const std::size_t nodeCount = model.restPositions.size();
    if (nodeCount < 2u ||
        !(config.threadCoilPitchM > 2.0 * model.radius) ||
        !(config.threadCoilInnerRadiusM > model.radius) ||
        !(config.threadMinimumNonNeighbourSurfaceClearanceM >= 0.0) ||
        config.threadDescentEdgeCount == 0u ||
        config.threadDescentEdgeCount >= nodeCount ||
        !(config.threadContactOffsetM >= 0.0) ||
        !(config.threadRestOffsetM >= 0.0) ||
        config.threadRestOffsetM > config.threadContactOffsetM ||
        config.threadSolverIterations == 0u ||
        !(config.threadConstraintToleranceM > 0.0) ||
        !(config.threadLinearDampingRate >= 0.0) ||
        !(config.threadTwistDampingRate >= 0.0) ||
        !(config.threadSupportPenetrationM >= 0.0) ||
        config.threadSupportPenetrationM >
            config.threadRestOffsetM + model.radius ||
        !finite(config.threadCoilPitchM) ||
        !finite(config.threadCoilInnerRadiusM) ||
        !finite(config.threadMinimumNonNeighbourSurfaceClearanceM) ||
        !finite(config.threadContactOffsetM) ||
        !finite(config.threadRestOffsetM) ||
        !finite(config.threadConstraintToleranceM) ||
        !finite(config.threadLinearDampingRate) ||
        !finite(config.threadTwistDampingRate) ||
        !finite(config.threadSupportPenetrationM)) {
        throw std::invalid_argument(
            "neutral-zone thread coil configuration is invalid"
        );
    }

    const Vec3 swageLocal = transformPointToLocal(
        config.padPose,
        surgical.metadata.swageAnchorWorld
    );
    const double supportZ =
        0.5 * config.pad.thicknessM.value + model.radius +
        config.threadRestOffsetM - config.threadSupportPenetrationM;
    if (!(swageLocal[2] >= supportZ - model.radius)) {
        throw std::invalid_argument(
            "needle swage begins below the neutral-zone support surface"
        );
    }

    const double innerRadius = config.threadCoilInnerRadiusM;
    const double radialGrowth =
        config.threadCoilPitchM / (2.0 * std::numbers::pi);
    const std::size_t boundaryNodeCount =
        surgical.metadata.threadBoundaryNodeCount;
    if (surgical.metadata.hardSwagedThreadNodeCount !=
            surgical.attachments.size() ||
        surgical.metadata.hardSwagedThreadNodeCount != 1u ||
        boundaryNodeCount != 2u ||
        surgical.tangentBindings.size() != 1u ||
        surgical.tangentBindings[0].edgeIndex != 0u ||
        boundaryNodeCount >= nodeCount) {
        throw std::logic_error(
            "neutral-zone thread lost its finite swage segment"
        );
    }
    std::vector<Vec3> local(nodeCount);
    for (std::size_t node = 0u; node < boundaryNodeCount; ++node) {
        local[node] = transformPointToLocal(
            config.padPose,
            model.restPositions[node]
        );
    }
    const std::size_t transitionEdgeCount =
        config.threadDescentEdgeCount;
    if (boundaryNodeCount + transitionEdgeCount >= nodeCount) {
        throw std::invalid_argument(
            "neutral-zone thread transition consumes the coil topology"
        );
    }
    const Vec3 exitDelta = subtract(
        local[boundaryNodeCount - 1u],
        local[boundaryNodeCount - 2u]
    );
    const double planarExitLength = std::hypot(
        exitDelta[0],
        exitDelta[1]
    );
    if (!(planarExitLength > 0.5 * model.restLengths.front())) {
        throw std::invalid_argument(
            "neutral-zone thread exit is not aligned with the support plane"
        );
    }
    const Vec3 exitDirection{
        exitDelta[0] / planarExitLength,
        exitDelta[1] / planarExitLength,
        0.0,
    };
    const auto rotatePlanar = [](const Vec3 value, const double angle) {
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        return Vec3{
            cosine * value[0] - sine * value[1],
            sine * value[0] + cosine * value[1],
            0.0,
        };
    };
    // Continue the actual swage tangent through a quarter-turn while the
    // strand descends to the pad. This removes the former 90-degree kink at
    // the first free node. Every chord retains its DER rest length; the
    // transition is package-layout geometry, not a hidden constraint.
    constexpr double kTransitionAngle = -0.5 * std::numbers::pi;
    for (std::size_t edge = 0u; edge < transitionEdgeCount; ++edge) {
        const std::size_t node = boundaryNodeCount + edge;
        const double restLength = model.restLengths[node - 1u];
        const double t = static_cast<double>(edge + 1u) /
            static_cast<double>(transitionEdgeCount);
        const double smooth = t * t * (3.0 - 2.0 * t);
        const double z = swageLocal[2] +
            smooth * (supportZ - swageLocal[2]);
        const double dz = z - local[node - 1u][2];
        if (!(std::abs(dz) < restLength)) {
            throw std::invalid_argument(
                "neutral-zone thread descent exceeds one rest edge"
            );
        }
        const double horizontalLength = std::sqrt(
            std::max(restLength * restLength - dz * dz, 0.0)
        );
        const double midpointAngle = kTransitionAngle *
            (static_cast<double>(edge) + 0.5) /
            static_cast<double>(transitionEdgeCount);
        const Vec3 direction = rotatePlanar(
            exitDirection,
            midpointAngle
        );
        local[node] = {
            local[node - 1u][0] + horizontalLength * direction[0],
            local[node - 1u][1] + horizontalLength * direction[1],
            z,
        };
    }
    const std::size_t transitionEndNode =
        boundaryNodeCount + transitionEdgeCount - 1u;
    const double halfX = 0.5 * config.pad.sizeXM.value;
    const double halfY = 0.5 * config.pad.sizeYM.value;
    // Begin at the outside and wind inward. The former inside-out spiral
    // necessarily crossed its incoming descent after roughly one revolution,
    // even though adjacent turns respected the nominal pitch. Here the outer
    // radius is solved so the last exact-length chord lands at the authored
    // inner radius. Starting on the outside keeps the incoming tangent out of
    // every later turn and makes the no-self-contact premise measurable.
    const auto relativeSpiralPoint = [&](const double outerRadius,
                                         const double theta) {
        const double radius = outerRadius - radialGrowth * theta;
        return Vec3{
            radius * std::cos(theta),
            radius * std::sin(theta),
            0.0,
        };
    };
    const auto nextSpiralTheta = [&](const double outerRadius,
                                     const double theta,
                                     const double chordLength) {
        const Vec3 previous = relativeSpiralPoint(outerRadius, theta);
        const auto chord = [&](const double candidate) {
            const Vec3 point = relativeSpiralPoint(outerRadius, candidate);
            return std::hypot(
                point[0] - previous[0],
                point[1] - previous[1]
            );
        };
        double lower = theta;
        double upper = theta + 0.25;
        while (outerRadius - radialGrowth * upper > model.radius &&
               chord(upper) < chordLength &&
               upper - theta < std::numbers::pi) {
            upper = theta + 2.0 * (upper - theta);
        }
        if (!(outerRadius - radialGrowth * upper > model.radius) ||
            chord(upper) < chordLength) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        for (std::uint32_t iteration = 0u; iteration < 64u; ++iteration) {
            const double midpoint = 0.5 * (lower + upper);
            if (chord(midpoint) < chordLength) {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
        }
        return 0.5 * (lower + upper);
    };
    const auto terminalRadius = [&](const double outerRadius) {
        double theta = 0.0;
        for (std::size_t node = transitionEndNode + 1u;
             node < nodeCount;
             ++node) {
            theta = nextSpiralTheta(
                outerRadius,
                theta,
                model.restLengths[node - 1u]
            );
            if (!finite(theta)) {
                return -std::numeric_limits<double>::infinity();
            }
        }
        return outerRadius - radialGrowth * theta;
    };
    double outerLower = innerRadius;
    double outerUpper = std::min(halfX, halfY) - model.radius;
    if (!(outerUpper > outerLower) ||
        terminalRadius(outerUpper) < innerRadius) {
        throw std::invalid_argument(
            "neutral-zone pad cannot contain the exact-length inward coil"
        );
    }
    for (std::uint32_t iteration = 0u; iteration < 96u; ++iteration) {
        const double midpoint = 0.5 * (outerLower + outerUpper);
        if (terminalRadius(midpoint) < innerRadius) {
            outerLower = midpoint;
        } else {
            outerUpper = midpoint;
        }
    }
    const double outerRadius = 0.5 * (outerLower + outerUpper);
    if (std::abs(terminalRadius(outerRadius) - innerRadius) > 1.0e-10) {
        throw std::logic_error(
            "neutral-zone inward coil radius solve did not converge"
        );
    }
    const Vec3 incomingDelta = subtract(
        local[transitionEndNode],
        local[transitionEndNode - 1u]
    );
    const double incomingLength = std::hypot(
        incomingDelta[0],
        incomingDelta[1]
    );
    if (!(incomingLength > 0.0)) {
        throw std::logic_error(
            "neutral-zone inward coil has no incoming planar tangent"
        );
    }
    const Vec3 incomingDirection{
        incomingDelta[0] / incomingLength,
        incomingDelta[1] / incomingLength,
        0.0,
    };
    const Vec3 incomingLeftNormal{
        -incomingDirection[1],
        incomingDirection[0],
        0.0,
    };
    const double tangentScale = std::hypot(outerRadius, radialGrowth);
    const Vec3 spiralRadial = multiply(
        add(
            multiply(incomingDirection, radialGrowth),
            multiply(incomingLeftNormal, outerRadius)
        ),
        -1.0 / tangentScale
    );
    const Vec3 spiralTangent = multiply(
        subtract(
            multiply(incomingDirection, outerRadius),
            multiply(incomingLeftNormal, radialGrowth)
        ),
        1.0 / tangentScale
    );
    const Vec3 spiralCenter = subtract(
        local[transitionEndNode],
        multiply(spiralRadial, outerRadius)
    );
    double theta = 0.0;
    for (std::size_t node = transitionEndNode + 1u;
         node < nodeCount;
         ++node) {
        theta = nextSpiralTheta(
            outerRadius,
            theta,
            model.restLengths[node - 1u]
        );
        if (!finite(theta)) {
            throw std::logic_error(
                "neutral-zone inward spiral could not resolve a rest edge"
            );
        }
        const Vec3 relative = relativeSpiralPoint(outerRadius, theta);
        local[node] = add(
            spiralCenter,
            add(
                multiply(spiralRadial, relative[0]),
                multiply(spiralTangent, relative[1])
            )
        );
        local[node][2] = supportZ;
    }

    for (std::size_t node = 0u; node < nodeCount; ++node) {
        if (std::abs(local[node][0]) + model.radius > halfX ||
            std::abs(local[node][1]) + model.radius > halfY) {
            throw std::invalid_argument(
                "neutral-zone pad is too small for the authored thread coil"
            );
        }
        model.restPositions[node] = transformPointToWorld(
            config.padPose,
            local[node]
        );
        if (node != 0u) {
            model.restLengths[node - 1u] = norm(subtract(
                model.restPositions[node],
                model.restPositions[node - 1u]
            ));
        }
    }
    double minimumNonNeighbourSurfaceClearance =
        std::numeric_limits<double>::infinity();
    const std::size_t edgeCount = nodeCount - 1u;
    for (std::size_t first = 0u; first < edgeCount; ++first) {
        for (std::size_t second = first + 2u;
             second < edgeCount;
             ++second) {
            minimumNonNeighbourSurfaceClearance = std::min(
                minimumNonNeighbourSurfaceClearance,
                segmentSegmentDistance(
                    model.restPositions[first],
                    model.restPositions[first + 1u],
                    model.restPositions[second],
                    model.restPositions[second + 1u]
                ) - 2.0 * model.radius
            );
        }
    }
    if (minimumNonNeighbourSurfaceClearance <
        config.threadMinimumNonNeighbourSurfaceClearanceM) {
        throw std::invalid_argument(
            "neutral-zone thread reset violates its non-neighbour "
            "surface-clearance certificate"
        );
    }
    model.name = "neutral_zone_coiled_pds_3_0_thread";
    model.fidelityBoundary =
        "source-sized PDO DER with a hard swage root/material-frame weld, "
        "two-axis clamped first-edge tangent, and exact-length "
        "tangent-continuous training-scene package coil; layout and contact "
        "remain research calibration, not product-package or clinical data";
    std::string reason;
    if (!model.valid(&reason)) {
        throw std::logic_error(
            "neutral-zone thread coil is invalid: " + reason
        );
    }
    surgical.threadState = makeDiscreteElasticRodDefaultState(model);
}

bool validSceneState(
    const MRBodyPropertiesGPU& properties,
    const MRBodyStateGPU& state
) {
    const double quaternionNorm =
        static_cast<double>(state.orientation.x) *
            state.orientation.x +
        static_cast<double>(state.orientation.y) *
            state.orientation.y +
        static_cast<double>(state.orientation.z) *
            state.orientation.z +
        static_cast<double>(state.orientation.w) *
            state.orientation.w;
    return
        properties.articulationIndex == MR_INVALID_INDEX &&
        state.flagsAndIndices[0] == properties.motionType &&
        state.flagsAndIndices[1] == MR_INVALID_INDEX &&
        finite(state.position) &&
        finite(state.orientation) &&
        finite(state.linearVelocityAndInverseMass) &&
        finite(state.angularVelocity) &&
        finite(state.inverseInertiaWorldRow0) &&
        finite(state.inverseInertiaWorldRow1) &&
        finite(state.inverseInertiaWorldRow2) &&
        std::abs(quaternionNorm - 1.0) <= 2.0e-4 &&
        (
            properties.motionType == MR_MOTION_DYNAMIC
            ? state.linearVelocityAndInverseMass.w > 0.0f
            : state.linearVelocityAndInverseMass.w == 0.0f
        );
}

bool validRodState(
    const DiscreteElasticRodModel& model,
    const DiscreteElasticRodState& state
) {
    DiscreteElasticRodEnergy energy;
    return evaluateDiscreteElasticRodEnergy(
        model,
        state,
        energy
    ).succeeded();
}

std::uint64_t hashBytes(
    std::uint64_t hash,
    const void* data,
    const std::size_t size
) {
    const auto* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0u; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
std::uint64_t hashPod(std::uint64_t hash, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return hashBytes(hash, &value, sizeof(value));
}

template <typename T>
std::uint64_t hashVector(
    std::uint64_t hash,
    const std::vector<T>& values
) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::uint64_t count = values.size();
    hash = hashPod(hash, count);
    return values.empty()
        ? hash
        : hashBytes(
              hash,
              values.data(),
              values.size() * sizeof(T)
          );
}

std::uint64_t hashString(
    std::uint64_t hash,
    const std::string& value
) {
    const std::uint64_t count = value.size();
    hash = hashPod(hash, count);
    return value.empty()
        ? hash
        : hashBytes(hash, value.data(), value.size());
}

std::uint64_t hashModel(
    std::uint64_t hash,
    const EngineModel& model
) {
    hash = hashPod(hash, model.world);
    hash = hashVector(hash, model.articulations);
    hash = hashVector(hash, model.joints);
    hash = hashVector(hash, model.dofs);
    hash = hashVector(hash, model.actuatorProfiles);
    hash = hashVector(hash, model.bodies);
    hash = hashVector(hash, model.shapes);
    hash = hashVector(hash, model.materials);
    hash = hashVector(hash, model.geometryHeaders);
    hash = hashVector(hash, model.geometryVertices);
    hash = hashVector(hash, model.geometryIndices);
    hash = hashVector(hash, model.convexFaces);
    hash = hashVector(hash, model.convexHalfEdges);
    hash = hashVector(hash, model.meshBvhNodes);
    hash = hashVector(hash, model.meshTriangles);
    hash = hashVector(hash, model.collisionExclusions);
    hash = hashPod(hash, model.constraintProgram.abiVersion);
    hash = hashVector(hash, model.constraintProgram.blocks);
    hash = hashVector(hash, model.constraintProgram.endpoints);
    hash = hashVector(hash, model.constraintProgram.rows);
    hash = hashVector(hash, model.constraintProgram.cones);
    hash = hashVector(hash, model.constraintProgram.warmImpulses);
    hash = hashVector(hash, model.defaultQ);
    hash = hashVector(hash, model.defaultV);
    return hashString(hash, model.name);
}

std::uint64_t hashRod(
    std::uint64_t hash,
    const HeterogeneousRodProgram& rod
) {
    hash = hashString(hash, rod.instanceId);
    hash = hashString(hash, rod.model.name);
    hash = hashString(hash, rod.model.fidelityBoundary);
    hash = hashPod(hash, rod.model.radius);
    hash = hashVector(hash, rod.model.restPositions);
    hash = hashVector(hash, rod.model.restTwists);
    hash = hashVector(hash, rod.model.restLengths);
    hash = hashVector(hash, rod.model.nodeMasses);
    hash = hashVector(hash, rod.model.edgeRotationalInertias);
    hash = hashVector(hash, rod.model.stretchStiffness);
    hash = hashVector(hash, rod.model.bendStiffness);
    hash = hashVector(hash, rod.model.twistStiffness);
    hash = hashVector(hash, rod.defaultState.positions);
    hash = hashVector(hash, rod.defaultState.velocities);
    hash = hashVector(hash, rod.defaultState.twists);
    hash = hashVector(hash, rod.defaultState.twistRates);
    hash = hashPod(hash, rod.stepConfig.timestep);
    hash = hashPod(hash, rod.stepConfig.gravity);
    hash = hashPod(hash, rod.stepConfig.solverIterations);
    hash = hashPod(hash, rod.stepConfig.constraintTolerance);
    hash = hashPod(hash, rod.stepConfig.linearDamping);
    hash = hashPod(hash, rod.stepConfig.twistDamping);
    hash = hashPod(hash, rod.stepConfig.derivativeStep);
    const std::uint32_t selfCollision =
        rod.stepConfig.enableSelfCollision ? 1u : 0u;
    hash = hashPod(hash, selfCollision);
    hash = hashPod(hash, rod.stepConfig.selfCollisionMargin);
    hash = hashPod(
        hash,
        rod.stepConfig.selfCollisionCompliance
    );
    const std::uint32_t ownsMaterial =
        rod.collision.ownedMaterial.has_value() ? 1u : 0u;
    hash = hashPod(hash, ownsMaterial);
    if (rod.collision.ownedMaterial.has_value()) {
        hash = hashPod(hash, *rod.collision.ownedMaterial);
    }
    hash = hashPod(hash, rod.collision.materialIndex);
    hash = hashPod(hash, rod.collision.collisionGroup);
    hash = hashPod(hash, rod.collision.collisionMask);
    hash = hashPod(hash, rod.collision.topologyGeneration);
    hash = hashPod(hash, rod.collision.contactOffset);
    hash = hashPod(hash, rod.collision.restOffset);
    const std::uint32_t toolCollision =
        rod.collision.enableToolCollision ? 1u : 0u;
    const std::uint32_t ccd =
        rod.collision.enableCCD ? 1u : 0u;
    hash = hashPod(hash, toolCollision);
    hash = hashPod(hash, ccd);
    hash = hashVector(hash, rod.attachments);
    hash = hashVector(hash, rod.rigidBindings);
    hash = hashVector(hash, rod.tangentBindings);
    return hashVector(hash, rod.twistBindings);
}

HeterogeneousWorldComposeDiagnostics fail(
    HeterogeneousWorldComposeDiagnostics diagnostics,
    const HeterogeneousWorldComposeStatus status,
    std::string message,
    const std::uint32_t component = MR_INVALID_INDEX
) {
    diagnostics.status = status;
    diagnostics.message = std::move(message);
    diagnostics.firstFailingComponent = component;
    return diagnostics;
}

EngineModel makeRigidSceneModel(
    const SurgicalRigidAsset& asset,
    const std::string_view name,
    const mr_float4 gravityAndTimestep
) {
    EngineModel model;
    model.name = std::string(name);
    model.bodies.push_back(asset.body);
    MRBodyPropertiesGPU& body = model.bodies.front();
    body.articulationIndex = MR_INVALID_INDEX;
    body.parentBody = MR_INVALID_INDEX;
    body.inboundJoint = MR_INVALID_INDEX;
    model.materials.push_back(asset.material);
    model.shapes = asset.shapes;
    for (MRShapeGPU& shape : model.shapes) {
        shape.bodyIndex = 0u;
        shape.materialIndex = 0u;
    }
    MRWorldGPU& world = model.world;
    world.abiVersion = MR_ENGINE_ABI_VERSION;
    world.bodyCount = 1u;
    world.shapeCount =
        static_cast<std::uint32_t>(model.shapes.size());
    world.materialCount = 1u;
    const std::uint64_t shapeCount = model.shapes.size();
    const std::uint64_t pairCount =
        shapeCount * (shapeCount - 1u) / 2u;
    world.pairCapacity = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(pairCount, 1u)
    );
    world.contactCapacity = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(4u * pairCount, 8u)
    );
    world.constraintCapacity =
        std::max<std::uint32_t>(world.contactCapacity, 8u);
    world.islandCapacity = 1u;
    world.solverType = MR_SOLVER_TEMPORAL_CONE;
    world.frictionConeType = MR_FRICTION_CONE_ELLIPTIC;
    world.gravityAndTimestep = gravityAndTimestep;
    world.solverScales = {
        1.0e-7f,
        1.0e-9f,
        2.0f,
        1.0e-5f,
    };
    return model;
}

EngineModel makeNeedleSceneModel(
    const CurvedSutureNeedleAsset& needle,
    const mr_float4 gravityAndTimestep
) {
    return makeRigidSceneModel(
        needle.rigid,
        "dynamic_curved_suture_needle_scene",
        gravityAndTimestep
    );
}

void calibrateDualPsmNeedleInsertPair(
    EngineModel& robots,
    const MRMaterialGPU& needleMaterial
) {
    const SurgicalPSMModelMetadata& metadata = surgicalPSMMetadata();
    if (robots.articulations.size() != 2u ||
        robots.materials.size() != 4u ||
        !(needleMaterial.friction.x > 0.0f) ||
        !(needleMaterial.friction.y > 0.0f) ||
        !(metadata.targetNeedleInsertStaticFriction > 0.0f) ||
        !(metadata.targetNeedleInsertDynamicFriction > 0.0f)) {
        throw std::invalid_argument(
            "dual PSM/needle material calibration is incomplete"
        );
    }
    const float rawStatic =
        metadata.targetNeedleInsertStaticFriction *
        metadata.targetNeedleInsertStaticFriction /
        needleMaterial.friction.x;
    const float rawDynamic =
        metadata.targetNeedleInsertDynamicFriction *
        metadata.targetNeedleInsertDynamicFriction /
        needleMaterial.friction.y;
    if (!std::isfinite(rawStatic) || !std::isfinite(rawDynamic) ||
        rawStatic < rawDynamic || rawDynamic < 0.0f) {
        throw std::invalid_argument(
            "dual PSM/needle effective friction is invalid"
        );
    }
    // DualPsmWorld appends each source PSM's generic and insert material in
    // arm order. Adjust only the two insert records; the needle and carrier
    // retain their authored contact behavior against the table and robot.
    for (const std::uint32_t insertMaterial : {1u, 3u}) {
        robots.materials[insertMaterial].friction.x = rawStatic;
        robots.materials[insertMaterial].friction.y = rawDynamic;
    }
}

MRBodyStateGPU staticSceneState(
    const SurgicalBasePose& pose
) {
    MRBodyStateGPU state{};
    state.position = {
        pose.position[0],
        pose.position[1],
        pose.position[2],
        1.0f,
    };
    state.orientation = {
        pose.orientation[0],
        pose.orientation[1],
        pose.orientation[2],
        pose.orientation[3],
    };
    state.flagsAndIndices[0] = MR_MOTION_STATIC;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = 0u;
    return state;
}

} // namespace

std::uint64_t heterogeneousWorldFingerprint(
    const HeterogeneousWorld& world
) noexcept {
    try {
        std::uint64_t hash = kFnvOffset;
        hash = hashPod(hash, world.formatVersion);
        hash = hashModel(hash, world.model);
        hash = hashVector(hash, world.sceneBodyIndices);
        hash = hashVector(hash, world.defaultSceneBodies);
        const std::uint64_t rodCount = world.rods.size();
        hash = hashPod(hash, rodCount);
        for (const HeterogeneousRodProgram& rod : world.rods) {
            hash = hashRod(hash, rod);
        }
        const std::uint64_t componentCount =
            world.componentInstanceIds.size();
        hash = hashPod(hash, componentCount);
        for (const std::string& id :
             world.componentInstanceIds) {
            hash = hashString(hash, id);
        }
        return hash;
    } catch (...) {
        return 0u;
    }
}

bool HeterogeneousWorld::valid(std::string* reason) const {
    if (reason != nullptr) {
        reason->clear();
    }
    if (formatVersion != kHeterogeneousWorldFormatVersion) {
        return setReason(
            reason,
            "heterogeneous world format version is unsupported"
        );
    }
    std::string modelReason;
    if (!model.valid(&modelReason)) {
        return setReason(
            reason,
            "heterogeneous model is invalid: " + modelReason
        );
    }
    std::vector<std::uint32_t> expectedSceneBodies;
    for (std::uint32_t body = 0u;
         body < model.bodies.size();
         ++body) {
        if (model.bodies[body].articulationIndex ==
            MR_INVALID_INDEX) {
            expectedSceneBodies.push_back(body);
        }
    }
    if (sceneBodyIndices != expectedSceneBodies ||
        defaultSceneBodies.size() != sceneBodyIndices.size()) {
        return setReason(
            reason,
            "scene body packing does not match model ownership"
        );
    }
    for (std::size_t local = 0u;
         local < sceneBodyIndices.size();
         ++local) {
        if (!validSceneState(
                model.bodies[sceneBodyIndices[local]],
                defaultSceneBodies[local]
            )) {
            return setReason(
                reason,
                "scene reset state is inconsistent with body properties"
            );
        }
    }
    std::set<std::string> identities;
    for (const std::string& id : componentInstanceIds) {
        if (id.empty() || !identities.insert(id).second) {
            return setReason(
                reason,
                "component instance identities are empty or duplicated"
            );
        }
    }
    for (const HeterogeneousRodProgram& rod : rods) {
        if (rod.instanceId.empty() ||
            !identities.insert(rod.instanceId).second ||
            !rod.model.valid(&modelReason) ||
            !validRodState(rod.model, rod.defaultState) ||
            rod.attachments.size() !=
                rod.rigidBindings.size()) {
            return setReason(
                reason,
                "rod program identity, model, state, or binding count is invalid"
            );
        }
        if (!finite(rod.stepConfig.timestep) ||
            !(rod.stepConfig.timestep > 0.0) ||
            !finite(rod.stepConfig.gravity) ||
            rod.stepConfig.solverIterations == 0u ||
            !finite(rod.stepConfig.constraintTolerance) ||
            !(rod.stepConfig.constraintTolerance > 0.0) ||
            !finite(rod.stepConfig.linearDamping) ||
            rod.stepConfig.linearDamping < 0.0 ||
            !finite(rod.stepConfig.twistDamping) ||
            rod.stepConfig.twistDamping < 0.0 ||
            !finite(rod.stepConfig.derivativeStep) ||
            !(rod.stepConfig.derivativeStep > 0.0) ||
            !finite(rod.stepConfig.selfCollisionMargin) ||
            rod.stepConfig.selfCollisionMargin < 0.0 ||
            !finite(
                rod.stepConfig.selfCollisionCompliance
            ) ||
            rod.stepConfig.selfCollisionCompliance < 0.0) {
            return setReason(
                reason,
                "rod step or self-contact configuration is invalid"
            );
        }
        if (rod.collision.materialIndex >=
                model.materials.size() ||
            (rod.collision.ownedMaterial.has_value() &&
             (!validMaterial(*rod.collision.ownedMaterial) ||
              !equal(
                  model.materials[rod.collision.materialIndex],
                  *rod.collision.ownedMaterial
              ))) ||
            rod.collision.topologyGeneration == 0u ||
            !finite(rod.collision.contactOffset) ||
            rod.collision.contactOffset < 0.0 ||
            !finite(rod.collision.restOffset) ||
            rod.collision.restOffset < 0.0 ||
            rod.collision.restOffset >
                rod.collision.contactOffset ||
            (
                rod.collision.enableToolCollision &&
                (
                    rod.collision.collisionGroup == 0u ||
                    rod.collision.collisionMask == 0u
                )
            )) {
            return setReason(
                reason,
                "rod collision configuration is invalid"
            );
        }
        std::set<std::uint32_t> nodes;
        std::vector<DiscreteRodRigidAttachmentBinding>
            acceptedRigidBindings;
        for (std::size_t index = 0u;
             index < rod.attachments.size();
             ++index) {
            const DiscreteRodAttachment& attachment =
                rod.attachments[index];
            const DiscreteRodRigidAttachmentBinding& binding =
                rod.rigidBindings[index];
            if (attachment.nodeIndex >=
                    rod.model.restPositions.size() ||
                !nodes.insert(attachment.nodeIndex).second ||
                !finite(attachment.targetPosition) ||
                !finite(attachment.targetVelocity) ||
                !finite(attachment.compliance) ||
                attachment.compliance < 0.0 ||
                !finite(binding.localAnchor)) {
                return setReason(
                    reason,
                    "rod attachment payload is invalid"
                );
            }
            if (binding.bodyIndex ==
                kDiscreteRodNoRigidBody) {
                continue;
            }
            if (binding.bodyIndex >=
                    defaultSceneBodies.size() ||
                defaultSceneBodies[binding.bodyIndex].
                        flagsAndIndices[0] !=
                    MR_MOTION_DYNAMIC) {
                return setReason(
                    reason,
                    "rod rigid binding does not name a dynamic scene body"
                );
            }
            const bool duplicateAnchor = std::ranges::any_of(
                acceptedRigidBindings,
                [&](const DiscreteRodRigidAttachmentBinding& previous) {
                    return
                        previous.bodyIndex == binding.bodyIndex &&
                        norm(subtract(
                            previous.localAnchor,
                            binding.localAnchor
                        )) <= 1.0e-9;
                }
            );
            if (duplicateAnchor) {
                return setReason(
                    reason,
                    "rod rigid bindings repeat a body-local anchor"
                );
            }
            acceptedRigidBindings.push_back(binding);
            const MRBodyStateGPU& body =
                defaultSceneBodies[binding.bodyIndex];
            const Vec3 offset = rotate(
                body.orientation,
                binding.localAnchor
            );
            const Vec3 expectedPosition = add(
                {
                    body.position.x,
                    body.position.y,
                    body.position.z,
                },
                offset
            );
            const Vec3 expectedVelocity = add(
                {
                    body.linearVelocityAndInverseMass.x,
                    body.linearVelocityAndInverseMass.y,
                    body.linearVelocityAndInverseMass.z,
                },
                cross(
                    {
                        body.angularVelocity.x,
                        body.angularVelocity.y,
                        body.angularVelocity.z,
                    },
                    offset
                )
            );
            if (norm(subtract(
                    attachment.targetPosition,
                    expectedPosition
                )) > 2.0e-6 ||
                norm(subtract(
                    attachment.targetVelocity,
                    expectedVelocity
                )) > 2.0e-6) {
                return setReason(
                    reason,
                    "rod reset target disagrees with its rigid anchor"
                );
            }
        }
        std::set<std::uint32_t> tangentEdges;
        for (const DiscreteRodRigidTangentAttachmentBinding& binding :
             rod.tangentBindings) {
            if (binding.edgeIndex >=
                    rod.model.restPositions.size() - 1u ||
                binding.bodyIndex == kDiscreteRodNoRigidBody ||
                binding.bodyIndex >= defaultSceneBodies.size() ||
                defaultSceneBodies[binding.bodyIndex]
                        .flagsAndIndices[0] != MR_MOTION_DYNAMIC ||
                !tangentEdges.insert(binding.edgeIndex).second ||
                !finite(binding.localAnchor) ||
                !finite(binding.localTangent) ||
                !finite(binding.localDirector) ||
                !finite(binding.complianceRadPerNm) ||
                binding.complianceRadPerNm < 0.0) {
                return setReason(
                    reason,
                    "rod tangent attachment payload is invalid"
                );
            }
            Vec3 localTangent{};
            Vec3 localDirector{};
            if (!normalize(binding.localTangent, localTangent) ||
                !normalize(binding.localDirector, localDirector) ||
                std::abs(dot(localTangent, localDirector)) > 1.0e-6 ||
                std::abs(norm(binding.localTangent) - 1.0) > 1.0e-6 ||
                std::abs(norm(binding.localDirector) - 1.0) > 1.0e-6) {
                return setReason(
                    reason,
                    "rod tangent attachment basis is not orthonormal"
                );
            }
            const MRBodyStateGPU& body =
                defaultSceneBodies[binding.bodyIndex];
            const Vec3 targetPoint = add(
                {
                    body.position.x,
                    body.position.y,
                    body.position.z,
                },
                rotate(body.orientation, binding.localAnchor)
            );
            const Vec3 worldTangent = rotate(
                body.orientation,
                localTangent
            );
            const Vec3 lineDelta = subtract(
                rod.defaultState.positions[binding.edgeIndex + 1u],
                targetPoint
            );
            const Vec3 transverse = subtract(
                lineDelta,
                multiply(worldTangent, dot(lineDelta, worldTangent))
            );
            // Checkpoint states may retain finite elastic line error; the live
            // ConstraintIR rows own its correction just as the twist row owns
            // a resumed material-frame phase error.
            if (!finite(transverse)) {
                return setReason(
                    reason,
                    "rod tangent attachment line error is non-finite"
                );
            }
        }
        std::set<std::uint32_t> twistEdges;
        for (const DiscreteRodRigidTwistAttachmentBinding& binding :
             rod.twistBindings) {
            if (binding.edgeIndex >=
                    rod.model.restPositions.size() - 1u ||
                binding.bodyIndex == kDiscreteRodNoRigidBody ||
                binding.bodyIndex >= defaultSceneBodies.size() ||
                defaultSceneBodies[binding.bodyIndex]
                        .flagsAndIndices[0] != MR_MOTION_DYNAMIC ||
                !twistEdges.insert(binding.edgeIndex).second ||
                !finite(binding.localTangent) ||
                !finite(binding.localMaterialDirector) ||
                !finite(binding.referenceTangentWorld) ||
                !finite(binding.referenceMaterialDirectorWorld) ||
                !finite(binding.complianceRadPerNm) ||
                binding.complianceRadPerNm < 0.0) {
                return setReason(
                    reason,
                    "rod material-frame attachment payload is invalid"
                );
            }
            Vec3 localTangent{};
            Vec3 localDirector{};
            Vec3 referenceTangent{};
            Vec3 referenceDirector{};
            if (!normalize(binding.localTangent, localTangent) ||
                !normalize(
                    binding.localMaterialDirector,
                    localDirector
                ) ||
                !normalize(
                    binding.referenceTangentWorld,
                    referenceTangent
                ) ||
                !normalize(
                    binding.referenceMaterialDirectorWorld,
                    referenceDirector
                ) ||
                std::abs(dot(localTangent, localDirector)) > 1.0e-6 ||
                std::abs(dot(referenceTangent, referenceDirector)) >
                    1.0e-6 ||
                std::abs(norm(binding.localTangent) - 1.0) > 1.0e-6 ||
                std::abs(norm(binding.localMaterialDirector) - 1.0) >
                    1.0e-6 ||
                std::abs(norm(binding.referenceTangentWorld) - 1.0) >
                    1.0e-6 ||
                std::abs(
                    norm(binding.referenceMaterialDirectorWorld) - 1.0
                ) > 1.0e-6) {
                return setReason(
                    reason,
                    "rod material-frame attachment basis is not orthonormal"
                );
            }
            const MRBodyStateGPU& body =
                defaultSceneBodies[binding.bodyIndex];
            Vec3 edgeTangent{};
            if (!normalize(
                    subtract(
                        rod.defaultState.positions[
                            binding.edgeIndex + 1u
                        ],
                        rod.defaultState.positions[binding.edgeIndex]
                    ),
                    edgeTangent
                )) {
                return setReason(
                    reason,
                    "rod material-frame attachment edge is degenerate"
                );
            }
            const Vec3 worldDirector = rotate(
                body.orientation,
                localDirector
            );
            const Vec3 worldTangent = rotate(
                body.orientation,
                localTangent
            );
            Vec3 transportedReference{};
            Vec3 transportedDirector{};
            if (!transportDirector(
                    referenceDirector,
                    referenceTangent,
                    edgeTangent,
                    transportedReference
                ) ||
                !transportDirector(
                    worldDirector,
                    worldTangent,
                    edgeTangent,
                    transportedDirector
                )) {
                return setReason(
                    reason,
                    "rod material-frame attachment transport is singular"
                );
            }
            const double targetTwist = std::atan2(
                dot(
                    edgeTangent,
                    cross(
                        transportedReference,
                        transportedDirector
                    )
                ),
                dot(transportedReference, transportedDirector)
            );
            // A persistent reset is allowed to carry elastic tangent and
            // material-frame error from a prior accepted step. The live
            // ConstraintIR row owns correction of that state; validity only
            // rejects a basis that cannot define a finite phase. Requiring
            // reset coincidence here would make checkpoint/resume stricter
            // than the same state in an uninterrupted transaction.
            if (!finite(targetTwist)) {
                return setReason(
                    reason,
                    "rod material-frame attachment phase is non-finite"
                );
            }
        }
    }
    const std::uint64_t expectedFingerprint =
        heterogeneousWorldFingerprint(*this);
    if (fingerprint == 0u ||
        fingerprint != expectedFingerprint) {
        return setReason(
            reason,
            "heterogeneous world fingerprint mismatch"
        );
    }
    return true;
}

HeterogeneousWorldComposeDiagnostics
composeHeterogeneousWorld(
    const std::span<const HeterogeneousWorldComponent> components,
    const std::span<const HeterogeneousRodProgram> rods,
    HeterogeneousWorld& output,
    const EngineModelComposeConfig& config
) {
    HeterogeneousWorldComposeDiagnostics diagnostics;
    diagnostics.componentCount =
        static_cast<std::uint32_t>(components.size());
    diagnostics.rodCount =
        static_cast<std::uint32_t>(rods.size());
    if (components.empty()) {
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::
                invalidConfiguration,
            "heterogeneous world requires at least one component"
        );
    }
    try {
        std::vector<EngineModelComponent> modelComponents;
        modelComponents.reserve(components.size());
        std::vector<MRBodyStateGPU> sceneStates;
        std::vector<std::string> componentIds;
        componentIds.reserve(components.size());
        for (std::uint32_t componentIndex = 0u;
             componentIndex < components.size();
             ++componentIndex) {
            const HeterogeneousWorldComponent& component =
                components[componentIndex];
            if (component.model == nullptr ||
                component.instanceId.empty()) {
                return fail(
                    std::move(diagnostics),
                    HeterogeneousWorldComposeStatus::
                        invalidComponent,
                    "heterogeneous component is empty",
                    componentIndex
                );
            }
            std::size_t expectedSceneStates = 0u;
            for (const MRBodyPropertiesGPU& body :
                 component.model->bodies) {
                expectedSceneStates +=
                    body.articulationIndex == MR_INVALID_INDEX
                    ? 1u
                    : 0u;
            }
            if (component.defaultSceneBodies.size() !=
                expectedSceneStates) {
                return fail(
                    std::move(diagnostics),
                    HeterogeneousWorldComposeStatus::
                        invalidSceneState,
                    "component scene-state count disagrees with topology",
                    componentIndex
                );
            }
            std::size_t localScene = 0u;
            for (const MRBodyPropertiesGPU& body :
                 component.model->bodies) {
                if (body.articulationIndex != MR_INVALID_INDEX) {
                    continue;
                }
                if (!validSceneState(
                        body,
                        component.defaultSceneBodies[
                            localScene++
                        ]
                    )) {
                    return fail(
                        std::move(diagnostics),
                        HeterogeneousWorldComposeStatus::
                            invalidSceneState,
                        "component scene reset is invalid",
                        componentIndex
                    );
                }
            }
            modelComponents.push_back({
                .model = component.model,
                .instanceId = component.instanceId,
            });
            sceneStates.insert(
                sceneStates.end(),
                component.defaultSceneBodies.begin(),
                component.defaultSceneBodies.end()
            );
            componentIds.emplace_back(component.instanceId);
        }

        EngineModel composed;
        const EngineModelComposeDiagnostics modelDiagnostics =
            composeEngineModels(
                modelComponents,
                composed,
                config
            );
        if (!modelDiagnostics.succeeded()) {
            return fail(
                std::move(diagnostics),
                HeterogeneousWorldComposeStatus::
                    modelCompositionFailure,
                modelDiagnostics.message,
                modelDiagnostics.firstFailingComponent
            );
        }

        HeterogeneousWorld staged;
        staged.model = std::move(composed);
        staged.defaultSceneBodies = std::move(sceneStates);
        staged.rods.assign(rods.begin(), rods.end());
        for (HeterogeneousRodProgram& rod : staged.rods) {
            if (!rod.collision.ownedMaterial.has_value()) {
                continue;
            }
            if (!validMaterial(*rod.collision.ownedMaterial) ||
                staged.model.materials.size() >=
                    std::numeric_limits<std::uint32_t>::max()) {
                return fail(
                    std::move(diagnostics),
                    HeterogeneousWorldComposeStatus::invalidRodProgram,
                    "rod-owned contact material is invalid"
                );
            }
            rod.collision.materialIndex =
                static_cast<std::uint32_t>(
                    staged.model.materials.size()
                );
            staged.model.materials.push_back(
                *rod.collision.ownedMaterial
            );
        }
        staged.model.world.materialCount =
            static_cast<std::uint32_t>(
                staged.model.materials.size()
            );
        staged.componentInstanceIds =
            std::move(componentIds);
        for (std::uint32_t body = 0u;
             body < staged.model.bodies.size();
             ++body) {
            if (staged.model.bodies[body].articulationIndex ==
                MR_INVALID_INDEX) {
                staged.sceneBodyIndices.push_back(body);
            }
        }
        // Scene state is composed in component order, which is also the
        // canonical non-articulated body order produced by the model
        // composer. Rebase the optional identity field to the composed global
        // body index so the persistent world can validate and consume the
        // state without accepting stale component-local identities.
        for (std::size_t localScene = 0u;
             localScene < staged.defaultSceneBodies.size();
             ++localScene) {
            staged.defaultSceneBodies[localScene]
                .flagsAndIndices[2] =
                staged.sceneBodyIndices[localScene];
        }
        staged.fingerprint =
            heterogeneousWorldFingerprint(staged);
        std::string reason;
        if (!staged.valid(&reason)) {
            return fail(
                std::move(diagnostics),
                HeterogeneousWorldComposeStatus::invalidWorld,
                std::move(reason)
            );
        }
        diagnostics.articulationCount =
            static_cast<std::uint32_t>(
                staged.model.articulations.size()
            );
        diagnostics.sceneBodyCount =
            static_cast<std::uint32_t>(
                staged.defaultSceneBodies.size()
            );
        diagnostics.fingerprint = staged.fingerprint;
        output = std::move(staged);
        return diagnostics;
    } catch (const std::bad_alloc&) {
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::allocationFailure,
            "heterogeneous world allocation failed"
        );
    } catch (const std::exception& exception) {
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::invalidWorld,
            exception.what()
        );
    }
}

HeterogeneousWorldComposeDiagnostics
makeDualDvrkPsmNeedleThreadHeterogeneousWorld(
    HeterogeneousWorld& output,
    const DualPsmNeedleThreadWorldConfig& config
) {
    try {
        DualPsmNeedleThreadWorld surgical =
            makeDualDvrkPsmNeedleThreadWorld(config);
        EngineModel needleModel = makeNeedleSceneModel(
            surgical.needle,
            surgical.robots.model.world.gravityAndTimestep
        );
        calibrateDualPsmNeedleInsertPair(
            surgical.robots.model,
            needleModel.materials.front()
        );
        std::string reason;
        if (!needleModel.valid(&reason)) {
            HeterogeneousWorldComposeDiagnostics diagnostics;
            return fail(
                std::move(diagnostics),
                HeterogeneousWorldComposeStatus::invalidComponent,
                "needle scene model is invalid: " + reason,
                1u
            );
        }
        const std::array<MRBodyStateGPU, 1> needleStates{
            surgical.needleState,
        };
        const std::array<HeterogeneousWorldComponent, 2>
            components{{
                {
                    .model = &surgical.robots.model,
                    .instanceId = "dual_psm",
                },
                {
                    .model = &needleModel,
                    .instanceId = "curved_needle",
                    .defaultSceneBodies = needleStates,
                },
            }};
        HeterogeneousRodProgram thread;
        thread.instanceId = "suture_thread";
        thread.model = surgical.threadModel;
        thread.defaultState = surgical.threadState;
        thread.stepConfig.timestep =
            surgical.robots.model.world.gravityAndTimestep.w;
        thread.stepConfig.gravity = {
            surgical.robots.model.world.gravityAndTimestep.x,
            surgical.robots.model.world.gravityAndTimestep.y,
            surgical.robots.model.world.gravityAndTimestep.z,
        };
        thread.stepConfig.enableSelfCollision = true;
        thread.collision.ownedMaterial =
            surgical.threadContactMaterial;
        thread.attachments.assign(
            surgical.attachments.begin(),
            surgical.attachments.end()
        );
        thread.rigidBindings.assign(
            surgical.rigidBindings.begin(),
            surgical.rigidBindings.end()
        );
        thread.tangentBindings.assign(
            surgical.tangentBindings.begin(),
            surgical.tangentBindings.end()
        );
        thread.twistBindings.assign(
            surgical.twistBindings.begin(),
            surgical.twistBindings.end()
        );
        const std::array<HeterogeneousRodProgram, 1> rods{
            std::move(thread),
        };
        EngineModelComposeConfig composeConfig;
        composeConfig.name =
            "dual_psm_curved_needle_thread_world";
        composeConfig.gravityAndTimestep =
            surgical.robots.model.world.gravityAndTimestep;
        return composeHeterogeneousWorld(
            components,
            rods,
            output,
            composeConfig
        );
    } catch (const std::bad_alloc&) {
        HeterogeneousWorldComposeDiagnostics diagnostics;
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::allocationFailure,
            "surgical heterogeneous world allocation failed"
        );
    } catch (const std::exception& exception) {
        HeterogeneousWorldComposeDiagnostics diagnostics;
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::invalidWorld,
            exception.what()
        );
    }
}

HeterogeneousWorldComposeDiagnostics
makeDualDvrkPsmNeedleThreadNeutralZoneHeterogeneousWorld(
    HeterogeneousWorld& output,
    const DualPsmNeedleThreadNeutralZoneConfig& config
) {
    try {
        DualPsmNeedleThreadWorld surgical =
            makeDualDvrkPsmNeedleThreadWorld(config.surgical);
        layThreadOnNeutralZone(surgical, config);
        const mr_float4 gravityAndTimestep =
            surgical.robots.model.world.gravityAndTimestep;
        EngineModel needleModel = makeNeedleSceneModel(
            surgical.needle,
            gravityAndTimestep
        );
        calibrateDualPsmNeedleInsertPair(
            surgical.robots.model,
            needleModel.materials.front()
        );
        const SurgicalNeutralZonePadAsset pad =
            makeSurgicalNeutralZonePadAsset({
                .bodyIndex = 0u,
                .materialIndex = 0u,
                .slotGenerationBase = 700001u,
                .collisionGroup = 2u,
                .collisionMask = ~0u,
                .motionType = MR_MOTION_STATIC,
            }, config.pad);
        EngineModel padModel = makeRigidSceneModel(
            pad.rigid,
            "sterile_neutral_zone_pad_scene",
            gravityAndTimestep
        );
        std::string reason;
        if (!needleModel.valid(&reason) ||
            !padModel.valid(&reason)) {
            HeterogeneousWorldComposeDiagnostics diagnostics;
            return fail(
                std::move(diagnostics),
                HeterogeneousWorldComposeStatus::invalidComponent,
                "surgical scene model is invalid: " + reason,
                1u
            );
        }
        const std::array<MRBodyStateGPU, 1> needleStates{
            surgical.needleState,
        };
        const std::array<MRBodyStateGPU, 1> padStates{
            staticSceneState(config.padPose),
        };
        const std::array<HeterogeneousWorldComponent, 3>
            components{{
                {
                    .model = &surgical.robots.model,
                    .instanceId = "dual_psm",
                },
                {
                    .model = &needleModel,
                    .instanceId = "bowel_suture_needle",
                    .defaultSceneBodies = needleStates,
                },
                {
                    .model = &padModel,
                    .instanceId = "sterile_neutral_zone_pad",
                    .defaultSceneBodies = padStates,
                },
            }};
        HeterogeneousRodProgram thread;
        thread.instanceId = "pds_3_0_thread";
        thread.model = surgical.threadModel;
        thread.defaultState = surgical.threadState;
        thread.stepConfig.timestep = gravityAndTimestep.w;
        thread.stepConfig.gravity = {
            gravityAndTimestep.x,
            gravityAndTimestep.y,
            gravityAndTimestep.z,
        };
        thread.stepConfig.enableSelfCollision = true;
        thread.stepConfig.selfCollisionMargin =
            config.threadMinimumNonNeighbourSurfaceClearanceM;
        thread.collision.ownedMaterial =
            surgical.threadContactMaterial;
        thread.stepConfig.solverIterations =
            config.threadSolverIterations;
        thread.stepConfig.constraintTolerance =
            config.threadConstraintToleranceM;
        thread.stepConfig.linearDamping =
            config.threadLinearDampingRate;
        thread.stepConfig.twistDamping =
            config.threadTwistDampingRate;
        thread.collision.contactOffset = config.threadContactOffsetM;
        thread.collision.restOffset = config.threadRestOffsetM;
        thread.attachments.assign(
            surgical.attachments.begin(),
            surgical.attachments.end()
        );
        thread.rigidBindings.assign(
            surgical.rigidBindings.begin(),
            surgical.rigidBindings.end()
        );
        thread.tangentBindings.assign(
            surgical.tangentBindings.begin(),
            surgical.tangentBindings.end()
        );
        thread.twistBindings.assign(
            surgical.twistBindings.begin(),
            surgical.twistBindings.end()
        );
        const std::array<HeterogeneousRodProgram, 1> rods{
            std::move(thread),
        };
        EngineModelComposeConfig composeConfig;
        composeConfig.name =
            "dual_psm_bowel_suture_neutral_zone_world";
        composeConfig.gravityAndTimestep = gravityAndTimestep;
        return composeHeterogeneousWorld(
            components,
            rods,
            output,
            composeConfig
        );
    } catch (const std::bad_alloc&) {
        HeterogeneousWorldComposeDiagnostics diagnostics;
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::allocationFailure,
            "neutral-zone surgical world allocation failed"
        );
    } catch (const std::exception& exception) {
        HeterogeneousWorldComposeDiagnostics diagnostics;
        return fail(
            std::move(diagnostics),
            HeterogeneousWorldComposeStatus::invalidWorld,
            exception.what()
        );
    }
}

const char* heterogeneousWorldComposeStatusName(
    const HeterogeneousWorldComposeStatus status
) noexcept {
    switch (status) {
    case HeterogeneousWorldComposeStatus::success:
        return "success";
    case HeterogeneousWorldComposeStatus::invalidConfiguration:
        return "invalid_configuration";
    case HeterogeneousWorldComposeStatus::invalidComponent:
        return "invalid_component";
    case HeterogeneousWorldComposeStatus::invalidSceneState:
        return "invalid_scene_state";
    case HeterogeneousWorldComposeStatus::invalidRodProgram:
        return "invalid_rod_program";
    case HeterogeneousWorldComposeStatus::modelCompositionFailure:
        return "model_composition_failure";
    case HeterogeneousWorldComposeStatus::invalidWorld:
        return "invalid_world";
    case HeterogeneousWorldComposeStatus::allocationFailure:
        return "allocation_failure";
    }
    return "unknown";
}

} // namespace metalrobo
