#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MetalWorld.hpp"
#include "metalrobo/MillardMuscleReference.hpp"
#include "metalrobo/OpenSimSpatialTransform.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr std::array<char, 8u> kMagic{
    'N', 'H', 'R', 'I', 'G', 'I', 'D', '1',
};
constexpr std::uint32_t kPayloadAbi = 1u;
constexpr std::array<char, 8u> kMillardMagic{
    'N', 'H', 'M', 'U', 'S', 'C', '1', '\0',
};
constexpr std::uint32_t kMillardPayloadAbi = 3u;

#pragma pack(push, 1)
struct PayloadHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineAbi = 0u;
    std::uint32_t sourceBodyCount = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t jointCount = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t functionProgramCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct MillardPayloadHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t sourceBodyCount = 0u;
    std::uint32_t muscleCount = 0u;
    std::uint32_t pathPointCount = 0u;
    std::uint32_t pathWrapCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct MillardMuscleRecord {
    float maxIsometricForce = 0.0f;
    float optimalFiberLength = 0.0f;
    float tendonSlackLength = 0.0f;
    float pennationAngleAtOptimal = 0.0f;
    float fiberDamping = 0.0f;
    float defaultActivation = 0.0f;
    float minimumActivation = 0.0f;
    std::uint32_t pathPointOffset = 0u;
    std::uint32_t pathPointCount = 0u;
    std::uint32_t pathWrapOffset = 0u;
    std::uint32_t pathWrapCount = 0u;
    std::uint32_t flags = 0u;
};

struct MillardPathPointRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct MillardCurveRecord {
    float minNormActiveFiberLength = 0.0f;
    float transitionNormFiberLength = 0.0f;
    float maxNormActiveFiberLength = 0.0f;
    float shallowAscendingSlope = 0.0f;
    float activeMinimumValue = 0.0f;
    float concentricSlopeAtVmax = 0.0f;
    float concentricSlopeNearVmax = 0.0f;
    float isometricSlope = 0.0f;
    float eccentricSlopeAtVmax = 0.0f;
    float eccentricSlopeNearVmax = 0.0f;
    float maxEccentricVelocityForceMultiplier = 0.0f;
    float concentricCurviness = 0.0f;
    float eccentricCurviness = 0.0f;
    float fiberStrainAtZeroForce = 0.0f;
    float fiberStrainAtOneNormForce = 0.0f;
    float fiberStiffnessAtLowForce = 0.0f;
    float fiberStiffnessAtOneNormForce = 0.0f;
    float fiberCurviness = 0.0f;
    float tendonStrainAtOneNormForce = 0.0f;
    float tendonStiffnessAtOneNormForce = 0.0f;
    float tendonNormForceAtToeEnd = 0.0f;
    float tendonCurviness = 0.0f;
};

struct MillardWrapRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;
    float rotationX = 0.0f;
    float rotationY = 0.0f;
    float rotationZ = 0.0f;
    float radius = 0.0f;
    float length = 0.0f;
    std::int32_t startPoint = -1;
    std::int32_t endPoint = -1;
    std::uint32_t method = MR_MILLARD_PATH_WRAP_HYBRID;
};
#pragma pack(pop)

static_assert(sizeof(PayloadHeader) == 76u);
static_assert(sizeof(MRWorldGPU) == 96u);
static_assert(sizeof(MRArticulationGPU) == 48u);
static_assert(sizeof(MRBodyPropertiesGPU) == 160u);
static_assert(sizeof(MRJointDescriptorGPU) == 144u);
static_assert(sizeof(MRDofPropertiesGPU) == 64u);
static_assert(sizeof(MROpenSimSpatialTransformGPU) == 2512u);
static_assert(sizeof(MillardPayloadHeader) == 64u);
static_assert(sizeof(MillardMuscleRecord) == 48u);
static_assert(sizeof(MillardPathPointRecord) == 16u);
static_assert(sizeof(MillardCurveRecord) == 88u);
static_assert(sizeof(MillardWrapRecord) == 48u);
static_assert(sizeof(MRMillardMuscleGPU) == 64u);
static_assert(sizeof(MRMillardActivationDispatchGPU) == 32u);
static_assert(sizeof(MRMillardSourceCurveGPU) == 96u);
static_assert(sizeof(MRMillardCylinderWrapGPU) == 64u);
static_assert(sizeof(MRMillardMuscleResultGPU) == 32u);

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void readObject(std::istream& input, T& value, const char* what) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    require(input.good(), std::string("truncated ") + what);
}

template <typename T>
std::vector<T> readVector(
    std::istream& input,
    const std::size_t count,
    const char* what
) {
    std::vector<T> values(count);
    if (!values.empty()) {
        input.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T))
        );
        require(input.good(), std::string("truncated ") + what);
    }
    return values;
}

std::string hexSha256(const std::array<std::uint8_t, 32u>& value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(value.size() * 2u);
    for (const std::uint8_t byte : value) {
        result.push_back(digits[byte >> 4u]);
        result.push_back(digits[byte & 0x0fu]);
    }
    return result;
}

metalrobo::EngineModel loadReference(const char* path, PayloadHeader& header) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), std::string("cannot open payload ") + path);
    readObject(input, header, "payload header");
    require(header.magic == kMagic, "payload magic is not NHRIGID1");
    require(header.payloadAbi == kPayloadAbi, "unsupported Numi Human payload ABI");
    require(header.engineAbi == MR_ENGINE_ABI_VERSION, "payload/Core engine ABI mismatch");
    require(header.reserved0 == 0u, "payload reserved field is nonzero");
    require(header.sourceBodyCount > 0u, "payload has no source bodies");
    require(
        header.engineBodyCount == header.sourceBodyCount + 1u,
        "payload must contain exactly one synthetic fixed-root anchor"
    );
    require(
        header.jointCount == header.sourceBodyCount,
        "payload tree must have one inbound joint per source body"
    );
    require(header.nq == header.nv && header.nv > 0u, "payload generalized state is invalid");

    metalrobo::EngineModel model;
    model.name = "numilab_human_rajagopal_core_reference";
    readObject(input, model.world, "world record");
    MRArticulationGPU articulation{};
    readObject(input, articulation, "articulation record");
    model.articulations.push_back(articulation);
    model.bodies = readVector<MRBodyPropertiesGPU>(
        input, header.engineBodyCount, "body records"
    );
    model.joints = readVector<MRJointDescriptorGPU>(
        input, header.jointCount, "joint records"
    );
    model.dofs = readVector<MRDofPropertiesGPU>(input, header.nv, "DoF records");
    model.functionBasedJointPrograms.reserve(header.functionProgramCount);
    for (std::uint32_t index = 0u; index < header.functionProgramCount; ++index) {
        std::uint32_t jointIndex = MR_INVALID_INDEX;
        std::uint32_t reserved = 0u;
        MROpenSimSpatialTransformGPU packed{};
        readObject(input, jointIndex, "FunctionBased joint index");
        readObject(input, reserved, "FunctionBased reserved field");
        readObject(input, packed, "FunctionBased program");
        require(reserved == 0u, "FunctionBased program reserved field is nonzero");
        const auto decoded = metalrobo::unpackOpenSimSpatialTransformGPU(packed);
        require(decoded.succeeded(), "FunctionBased program is not canonical");
        model.functionBasedJointPrograms.push_back({jointIndex, decoded.transform});
    }
    model.defaultQ = readVector<float>(input, header.nq, "default q");
    model.defaultV = readVector<float>(input, header.nv, "default v");
    require(input.peek() == std::char_traits<char>::eof(), "payload contains trailing bytes");

    require(model.world.abiVersion == header.engineAbi, "world/header ABI mismatch");
    require(model.world.bodyCount == header.engineBodyCount, "world/header body count mismatch");
    require(model.world.articulationCount == 1u, "payload must contain one articulation");
    require(model.world.jointCount == header.jointCount, "world/header joint count mismatch");
    require(model.world.nq == header.nq && model.world.nv == header.nv, "world/header state mismatch");
    require(
        articulation.rootType == MR_ROOT_FIXED && articulation.rootBody == 0u &&
            articulation.bodyCount == header.engineBodyCount &&
            articulation.jointCount == header.jointCount &&
            articulation.nq == header.nq && articulation.nv == header.nv,
        "articulation/header tree mismatch"
    );
    std::string reason;
    require(model.valid(&reason), "Core reference model invalid: " + reason);
    return model;
}

struct MillardPayload {
    MillardPayloadHeader header{};
    std::vector<MillardMuscleRecord> muscles;
    std::vector<MillardCurveRecord> curves;
    std::vector<MillardPathPointRecord> pathPoints;
    std::vector<MillardWrapRecord> wraps;
};

MillardPayload loadMillardReference(
    const char* path,
    const PayloadHeader& rigidHeader,
    const metalrobo::EngineModel& model
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), std::string("cannot open Millard payload ") + path);
    MillardPayload payload;
    readObject(input, payload.header, "Millard payload header");
    require(payload.header.magic == kMillardMagic, "Millard payload magic is not NHMUSC1");
    require(
        payload.header.payloadAbi == kMillardPayloadAbi,
        "unsupported Numi Human Millard payload ABI"
    );
    require(
        payload.header.reserved0 == 0u,
        "Millard payload reserved field is nonzero"
    );
    require(
        payload.header.sourceSha256 == rigidHeader.sourceSha256,
        "Millard payload source hash does not match rigid payload"
    );
    require(
        payload.header.sourceBodyCount == rigidHeader.sourceBodyCount,
        "Millard payload source body count does not match rigid payload"
    );
    require(
        payload.header.muscleCount > 0u && payload.header.pathPointCount > 0u,
        "Millard payload has no muscles or path points"
    );
    payload.muscles = readVector<MillardMuscleRecord>(
        input, payload.header.muscleCount, "Millard muscle records"
    );
    payload.curves = readVector<MillardCurveRecord>(
        input, payload.header.muscleCount, "Millard curve records"
    );
    payload.pathPoints = readVector<MillardPathPointRecord>(
        input, payload.header.pathPointCount, "Millard path point records"
    );
    payload.wraps = readVector<MillardWrapRecord>(
        input, payload.header.pathWrapCount, "Millard wrap records"
    );
    require(
        input.peek() == std::char_traits<char>::eof(),
        "Millard payload contains trailing bytes"
    );

    const MRArticulationGPU& articulation = model.articulations.at(0u);
    const auto bodyIsValid = [&](const std::uint32_t bodyIndex) {
        return bodyIndex >= articulation.firstBody &&
            bodyIndex < articulation.firstBody + articulation.bodyCount;
    };
    for (std::size_t index = 0u; index < payload.muscles.size(); ++index) {
        const MillardMuscleRecord& muscle = payload.muscles[index];
        require(
            muscle.flags <= 1u,
            "Millard muscle record has unsupported flags at index " +
                std::to_string(index)
        );
        require(
            muscle.pathPointCount >= 2u &&
                muscle.pathPointOffset <= payload.pathPoints.size() &&
                muscle.pathPointCount <=
                    payload.pathPoints.size() - muscle.pathPointOffset,
            "Millard muscle has invalid path-point range at index " +
                std::to_string(index)
        );
        require(
            muscle.pathWrapOffset <= payload.wraps.size() &&
                muscle.pathWrapCount <= payload.wraps.size() - muscle.pathWrapOffset,
            "Millard muscle has invalid path-wrap range at index " +
                std::to_string(index)
        );
        for (std::uint32_t wrapIndex = 0u;
             wrapIndex < muscle.pathWrapCount; ++wrapIndex) {
            const MillardWrapRecord& wrap =
                payload.wraps[muscle.pathWrapOffset + wrapIndex];
            const auto validEndpoint = [&muscle](const std::int32_t endpoint) {
                return endpoint == -1 ||
                    (endpoint >= 1 &&
                        static_cast<std::uint32_t>(endpoint) <= muscle.pathPointCount);
            };
            require(
                validEndpoint(wrap.startPoint) && validEndpoint(wrap.endPoint) &&
                    (wrap.startPoint == -1 || wrap.endPoint == -1 ||
                        wrap.startPoint <= wrap.endPoint) &&
                    wrap.method <= MR_MILLARD_PATH_WRAP_AXIAL,
                "Millard PathWrap range or method is invalid at index " +
                    std::to_string(muscle.pathWrapOffset + wrapIndex)
            );
        }
    }
    for (std::size_t index = 0u; index < payload.curves.size(); ++index) {
        const MillardCurveRecord& curve = payload.curves[index];
        const std::array<float, 22u> values{
            curve.minNormActiveFiberLength,
            curve.transitionNormFiberLength,
            curve.maxNormActiveFiberLength,
            curve.shallowAscendingSlope,
            curve.activeMinimumValue,
            curve.concentricSlopeAtVmax,
            curve.concentricSlopeNearVmax,
            curve.isometricSlope,
            curve.eccentricSlopeAtVmax,
            curve.eccentricSlopeNearVmax,
            curve.maxEccentricVelocityForceMultiplier,
            curve.concentricCurviness,
            curve.eccentricCurviness,
            curve.fiberStrainAtZeroForce,
            curve.fiberStrainAtOneNormForce,
            curve.fiberStiffnessAtLowForce,
            curve.fiberStiffnessAtOneNormForce,
            curve.fiberCurviness,
            curve.tendonStrainAtOneNormForce,
            curve.tendonStiffnessAtOneNormForce,
            curve.tendonNormForceAtToeEnd,
            curve.tendonCurviness,
        };
        require(
            std::all_of(values.begin(), values.end(), [](const float value) {
                return std::isfinite(value);
            }),
            "Millard curve record is non-finite at index " + std::to_string(index)
        );
    }
    for (std::size_t index = 0u; index < payload.pathPoints.size(); ++index) {
        require(
            bodyIsValid(payload.pathPoints[index].bodyIndex) &&
                std::isfinite(payload.pathPoints[index].x) &&
                std::isfinite(payload.pathPoints[index].y) &&
                std::isfinite(payload.pathPoints[index].z),
            "Millard path point is invalid at index " + std::to_string(index)
        );
    }
    for (std::size_t index = 0u; index < payload.wraps.size(); ++index) {
        const MillardWrapRecord& wrap = payload.wraps[index];
        require(
            bodyIsValid(wrap.bodyIndex) && std::isfinite(wrap.centerX) &&
                std::isfinite(wrap.centerY) && std::isfinite(wrap.centerZ) &&
                std::isfinite(wrap.rotationX) && std::isfinite(wrap.rotationY) &&
                std::isfinite(wrap.rotationZ) && std::isfinite(wrap.radius) &&
                std::isfinite(wrap.length) && wrap.radius > 0.0f && wrap.length > 0.0f,
            "Millard cylinder wrap is invalid at index " + std::to_string(index)
        );
    }
    return payload;
}

metalrobo::MillardSourceCurveDefinition millardCurveDefinition(
    const MillardCurveRecord& record
) {
    return {
        .minNormActiveFiberLength = record.minNormActiveFiberLength,
        .transitionNormFiberLength = record.transitionNormFiberLength,
        .maxNormActiveFiberLength = record.maxNormActiveFiberLength,
        .shallowAscendingSlope = record.shallowAscendingSlope,
        .activeMinimumValue = record.activeMinimumValue,
        .concentricSlopeAtVmax = record.concentricSlopeAtVmax,
        .concentricSlopeNearVmax = record.concentricSlopeNearVmax,
        .isometricSlope = record.isometricSlope,
        .eccentricSlopeAtVmax = record.eccentricSlopeAtVmax,
        .eccentricSlopeNearVmax = record.eccentricSlopeNearVmax,
        .maxEccentricVelocityForceMultiplier =
            record.maxEccentricVelocityForceMultiplier,
        .concentricCurviness = record.concentricCurviness,
        .eccentricCurviness = record.eccentricCurviness,
        .fiberStrainAtZeroForce = record.fiberStrainAtZeroForce,
        .fiberStrainAtOneNormForce = record.fiberStrainAtOneNormForce,
        .fiberStiffnessAtLowForce = record.fiberStiffnessAtLowForce,
        .fiberStiffnessAtOneNormForce = record.fiberStiffnessAtOneNormForce,
        .fiberCurviness = record.fiberCurviness,
        .tendonStrainAtOneNormForce = record.tendonStrainAtOneNormForce,
        .tendonStiffnessAtOneNormForce = record.tendonStiffnessAtOneNormForce,
        .tendonNormForceAtToeEnd = record.tendonNormForceAtToeEnd,
        .tendonCurviness = record.tendonCurviness,
    };
}

metalrobo::MillardMuscleDefinition millardDefinition(
    const MillardPayload& payload,
    const MillardMuscleRecord& record
) {
    metalrobo::MillardMuscleDefinition definition;
    definition.maxIsometricForce = record.maxIsometricForce;
    definition.optimalFiberLength = record.optimalFiberLength;
    definition.tendonSlackLength = record.tendonSlackLength;
    definition.pennationAngleAtOptimal = record.pennationAngleAtOptimal;
    definition.fiberDamping = record.fiberDamping;
    definition.minimumActivation = record.minimumActivation;
    definition.pathPoints.reserve(record.pathPointCount);
    for (std::uint32_t index = 0u; index < record.pathPointCount; ++index) {
        const MillardPathPointRecord& point =
            payload.pathPoints[record.pathPointOffset + index];
        definition.pathPoints.push_back({
            point.bodyIndex,
            {point.x, point.y, point.z},
        });
    }
    definition.cylinderWraps.reserve(record.pathWrapCount);
    for (std::uint32_t index = 0u; index < record.pathWrapCount; ++index) {
        const MillardWrapRecord& wrap =
            payload.wraps[record.pathWrapOffset + index];
        definition.cylinderWraps.push_back({
            wrap.bodyIndex,
            {wrap.centerX, wrap.centerY, wrap.centerZ},
            {wrap.rotationX, wrap.rotationY, wrap.rotationZ},
            wrap.radius,
            wrap.length,
            wrap.startPoint,
            wrap.endPoint,
            static_cast<MRMillardPathWrapMethod>(wrap.method),
        });
    }
    return definition;
}

struct MillardReferenceMetrics {
    std::uint32_t muscleCount = 0u;
    std::uint32_t pathPointCount = 0u;
    std::uint32_t sourceWrapCount = 0u;
    std::uint32_t appliedCylinderWrapCount = 0u;
    std::uint32_t ignoredTendonComplianceCount = 0u;
    double totalPathLength = 0.0;
    double generalizedForceL1 = 0.0;
    double maximumEquilibriumResidual = 0.0;
    struct MuscleSample {
        double pathLength = 0.0;
        double tendonForce = 0.0;
        double equilibriumResidual = 0.0;
        double generalizedForceL1 = 0.0;
        std::uint32_t appliedCylinderWrapCount = 0u;
        std::vector<double> generalizedForces;
    };
    std::vector<MuscleSample> samples;
};

bool allFinite(const std::vector<double>& values);

bool evaluateStaticMillardState(
    const metalrobo::MillardMuscleDefinition& definition,
    const metalrobo::MillardSourceCurveDefinition& curves,
    const double activation,
    const double pathLength,
    const double fiberLength,
    metalrobo::MillardMuscleState& state,
    metalrobo::MillardMuscleForce& force
) {
    const double thickness = definition.optimalFiberLength *
        std::sin(definition.pennationAngleAtOptimal);
    const double sinePennation = thickness / fiberLength;
    if (!(fiberLength > thickness) || !std::isfinite(sinePennation) ||
        std::abs(sinePennation) >= 1.0) {
        return false;
    }
    const double tendonLength = pathLength -
        fiberLength * std::sqrt(1.0 - sinePennation * sinePennation);
    state.activation = activation;
    state.normalizedFiberVelocity = 0.0;
    state.fiberLength = fiberLength;
    const auto curveDiagnostics = metalrobo::evaluateMillardSourceCurves(
        curves,
        fiberLength / definition.optimalFiberLength,
        state.normalizedFiberVelocity,
        tendonLength / definition.tendonSlackLength,
        state.curves
    );
    if (!curveDiagnostics.succeeded()) {
        throw std::runtime_error(
            "Millard source-curve evaluation status=" + std::string(
                metalrobo::millardMuscleReferenceStatusName(curveDiagnostics.status)
            )
        );
    }
    const auto forceDiagnostics = metalrobo::evaluateMillardMuscleForce(
        definition, state, force
    );
    if (!forceDiagnostics.succeeded()) {
        throw std::runtime_error(
            "Millard force evaluation status=" + std::string(
                metalrobo::millardMuscleReferenceStatusName(forceDiagnostics.status)
            )
        );
    }
    return true;
}

bool solveStaticMillardEquilibrium(
    const metalrobo::MillardMuscleDefinition& definition,
    const metalrobo::MillardSourceCurveDefinition& curves,
    const double activation,
    const double pathLength,
    metalrobo::MillardMuscleState& state,
    metalrobo::MillardMuscleForce& force
) {
    const double thickness = definition.optimalFiberLength *
        std::sin(definition.pennationAngleAtOptimal);
    const double minimumFiberLength = std::max(
        curves.minNormActiveFiberLength * definition.optimalFiberLength,
        thickness / std::sqrt(1.0 - 0.01)
    );
    double lower = minimumFiberLength * (1.0 + 1.0e-10);
    double upper = std::max(
        lower * 1.01,
        pathLength + definition.optimalFiberLength
    );
    metalrobo::MillardMuscleState lowerState;
    metalrobo::MillardMuscleForce lowerForce;
    metalrobo::MillardMuscleState upperState;
    metalrobo::MillardMuscleForce upperForce;
    if (!evaluateStaticMillardState(
            definition, curves, activation, pathLength, lower, lowerState,
            lowerForce
        ) ||
        !evaluateStaticMillardState(
            definition, curves, activation, pathLength, upper, upperState,
            upperForce
        )) {
        throw std::runtime_error("Millard source-curve evaluation failed");
    }
    if (lowerForce.equilibriumResidual > 0.0 ||
        upperForce.equilibriumResidual < 0.0) {
        throw std::runtime_error(
            "Millard static equilibrium is unbracketed path_length=" +
            std::to_string(pathLength) + " lower_residual=" +
            std::to_string(lowerForce.equilibriumResidual) +
            " upper_residual=" + std::to_string(upperForce.equilibriumResidual)
        );
    }
    for (std::uint32_t iteration = 0u; iteration < 96u; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        metalrobo::MillardMuscleState middleState;
        metalrobo::MillardMuscleForce middleForce;
        if (!evaluateStaticMillardState(
                definition, curves, activation, pathLength, middle,
                middleState, middleForce
            )) {
            return false;
        }
        if (middleForce.equilibriumResidual < 0.0) {
            lower = middle;
        } else {
            upper = middle;
        }
        state = middleState;
        force = middleForce;
    }
    return std::abs(force.equilibriumResidual) < 1.0e-5;
}

MillardReferenceMetrics verifyMillardReference(
    const metalrobo::EngineModel& model,
    const MillardPayload& payload
) {
    const MRArticulationGPU& articulation = model.articulations.at(0u);
    std::vector<double> q(
        model.defaultQ.begin() + articulation.qOffset,
        model.defaultQ.begin() + articulation.qOffset + articulation.nq
    );
    std::vector<double> v(articulation.nv, 0.0);
    MillardReferenceMetrics metrics;
    metrics.muscleCount = payload.header.muscleCount;
    metrics.pathPointCount = payload.header.pathPointCount;
    metrics.sourceWrapCount = payload.header.pathWrapCount;
    metrics.samples.reserve(payload.muscles.size());
    for (std::size_t index = 0u; index < payload.muscles.size(); ++index) {
        const MillardMuscleRecord& record = payload.muscles[index];
        const metalrobo::MillardMuscleDefinition definition =
            millardDefinition(payload, record);
        const metalrobo::MillardSourceCurveDefinition curves =
            millardCurveDefinition(payload.curves[index]);
        metalrobo::MillardMusclePathResult path;
        const auto pathDiagnostics = metalrobo::evaluateMillardMusclePath(
            model, 0u, q, v, definition, path
        );
        require(
            pathDiagnostics.succeeded() && std::isfinite(path.length) &&
                path.length > 0.0,
            "Millard GeometryPath evaluation failed at muscle " +
                std::to_string(index) + " status=" +
                metalrobo::millardMuscleReferenceStatusName(pathDiagnostics.status)
        );
        metalrobo::MillardMuscleState state;
        metalrobo::MillardMuscleForce force;
        require(
            solveStaticMillardEquilibrium(
                definition, curves,
                std::max(
                    static_cast<double>(record.defaultActivation),
                    definition.minimumActivation
                ),
                path.length, state, force
            ),
            "Millard source-curve static equilibrium failed at muscle " +
                std::to_string(index)
        );
        std::vector<double> generalizedForce(articulation.nv, 0.0);
        const auto projectionDiagnostics = metalrobo::projectMillardMuscleTension(
            model, 0u, q, v, definition, force.tendonForce, generalizedForce,
            &path
        );
        require(
            projectionDiagnostics.succeeded() && std::isfinite(path.length) &&
                path.length > 0.0 && allFinite(generalizedForce),
            "Millard GeometryPath/tension projection failed at muscle " +
                std::to_string(index) + " status=" +
                metalrobo::millardMuscleReferenceStatusName(
                    projectionDiagnostics.status
                )
        );
        metrics.appliedCylinderWrapCount += path.appliedCylinderWrapCount;
        metrics.ignoredTendonComplianceCount += (record.flags & 1u) != 0u;
        metrics.totalPathLength += path.length;
        metrics.maximumEquilibriumResidual = std::max(
            metrics.maximumEquilibriumResidual, std::abs(force.equilibriumResidual)
        );
        MillardReferenceMetrics::MuscleSample sample;
        sample.pathLength = path.length;
        sample.tendonForce = force.tendonForce;
        sample.equilibriumResidual = force.equilibriumResidual;
        sample.appliedCylinderWrapCount = path.appliedCylinderWrapCount;
        sample.generalizedForces = generalizedForce;
        for (const double effort : generalizedForce) {
            metrics.generalizedForceL1 += std::abs(effort);
            sample.generalizedForceL1 += std::abs(effort);
        }
        metrics.samples.push_back(sample);
    }
    require(
        std::isfinite(metrics.totalPathLength) &&
            std::isfinite(metrics.generalizedForceL1) &&
            metrics.totalPathLength > 0.0 && metrics.generalizedForceL1 > 0.0,
        "Millard source reference totals are non-finite or inactive"
    );
    return metrics;
}

bool allFinite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(), [](const double value) {
        return std::isfinite(value);
    });
}

struct MetalReferenceMetrics {
    std::string deviceName;
    double bodyPositionError = 0.0;
    double bodyOrientationError = 0.0;
    double pointPositionError = 0.0;
    double pointJacobianError = 0.0;
    double massError = 0.0;
    double massScaledError = 0.0;
};

struct MetalWorldFunctionBasedMetrics {
    std::string deviceName;
    double maximumAccelerationError = 0.0;
    double maximumVelocityError = 0.0;
    double maximumConfigurationError = 0.0;
    std::uint32_t successfulStepCount = 0u;
};

struct MetalWorldFunctionBasedContactMetrics {
    std::string deviceName;
    std::uint32_t successfulStepCount = 0u;
    std::uint32_t maximumActiveContacts = 0u;
    std::uint32_t maximumConstraints = 0u;
    double millardGeneralizedForceL1 = 0.0;
    double maximumMillardActivationError = 0.0;
};

MRBodyStateGPU staticGroundState(
    const std::uint32_t bodyIndex,
    const float height
) {
    MRBodyStateGPU state{};
    state.position = {0.0f, height, 0.0f, 1.0f};
    state.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    state.flagsAndIndices[0] = MR_MOTION_STATIC;
    state.flagsAndIndices[1] = MR_INVALID_INDEX;
    state.flagsAndIndices[2] = bodyIndex;
    return state;
}

MetalWorldFunctionBasedContactMetrics
verifyMetalWorldFunctionBasedContact(
    const metalrobo::EngineModel& source,
    const metalrobo::MetalWorldMillardProgram* millardProgram,
    const std::span<const float> millardExcitations = {},
    const bool taskDrivenMillardExcitation = false
) {
    metalrobo::EngineModel model = source;
    if (taskDrivenMillardExcitation) {
        // The compact Human reference payload intentionally carries no
        // display semantics. Native tasks, by contrast, bind a compiled-world
        // semantic table. Materialize deterministic probe-only identities so
        // the executable bridge exercises that normal task admission path.
        if (model.bodyNames.empty()) {
            model.bodyNames.reserve(model.bodies.size());
            for (std::size_t body = 0u; body < model.bodies.size(); ++body) {
                model.bodyNames.push_back("source_body_" +
                    std::to_string(body));
            }
        }
        if (model.jointNames.empty()) {
            model.jointNames.reserve(model.joints.size());
            for (std::size_t joint = 0u; joint < model.joints.size(); ++joint) {
                model.jointNames.push_back("source_joint_" +
                    std::to_string(joint));
            }
        }
    }
    const MRArticulationGPU& articulation = model.articulations.at(0u);
    const std::uint32_t contactBody =
        articulation.firstBody + articulation.bodyCount - 1u;
    std::vector<double> q(
        model.defaultQ.begin(), model.defaultQ.end()
    );
    std::vector<double> v(model.defaultV.size(), 0.0);
    std::vector<metalrobo::ArticulatedBodyKinematics> poses(
        articulation.bodyCount
    );
    const auto poseDiagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            model, 0u, q, v, poses
        );
    require(
        poseDiagnostics.succeeded(),
        "CPU source kinematics failed before FunctionBased contact probe"
    );
    const std::uint32_t localContactBody =
        contactBody - articulation.firstBody;
    const float contactHeight = static_cast<float>(
        poses.at(localContactBody).centerOfMassPosition[1]
    );
    require(
        std::isfinite(contactHeight),
        "source contact witness has non-finite height"
    );

    const std::uint32_t groundBody =
        static_cast<std::uint32_t>(model.bodies.size());
    MRBodyPropertiesGPU ground{};
    ground.articulationIndex = MR_INVALID_INDEX;
    ground.parentBody = MR_INVALID_INDEX;
    ground.inboundJoint = MR_INVALID_INDEX;
    ground.motionType = MR_MOTION_STATIC;
    ground.dampingAndSpeedLimits = {0.0f, 0.0f, 1.0e6f, 1.0e6f};
    model.bodies.push_back(ground);
    if (!model.bodyNames.empty()) {
        model.bodyNames.push_back("source_contact_response_ground");
    }

    MRMaterialGPU material{};
    material.friction = {0.8f, 0.6f, 0.0f, 0.0f};
    material.response = {0.0f, 0.5f, 0.0f, 0.0f};
    material.geometry = {0.001f, 0.0f, 0.0f, 0.0f};
    model.materials.push_back(material);

    constexpr float radius = 0.06f;
    MRShapeGPU plane{};
    plane.bodyIndex = groundBody;
    plane.shapeType = MR_SHAPE_PLANE;
    plane.materialIndex = 0u;
    plane.collisionGroup = 1u;
    plane.collisionMask = ~0u;
    plane.slotGeneration = 1u;
    plane.localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    model.shapes.push_back(plane);

    MRShapeGPU witness{};
    witness.bodyIndex = contactBody;
    witness.shapeType = MR_SHAPE_SPHERE;
    witness.materialIndex = 0u;
    witness.collisionGroup = 1u;
    witness.collisionMask = ~0u;
    witness.slotGeneration = 1u;
    witness.localRotation = {0.0f, 0.0f, 0.0f, 1.0f};
    witness.dimensions = {radius, 0.0f, 0.0f, 0.0f};
    witness.contactRestAndBoundingRadius =
        {0.001f, 0.0f, radius, 0.0f};
    model.shapes.push_back(witness);
    if (!model.shapeNames.empty()) {
        model.shapeNames.push_back("source_contact_response_plane");
        model.shapeNames.push_back("source_contact_response_witness");
    }
    model.world.bodyCount = static_cast<std::uint32_t>(
        model.bodies.size()
    );
    model.world.materialCount = static_cast<std::uint32_t>(
        model.materials.size()
    );
    model.world.shapeCount = static_cast<std::uint32_t>(
        model.shapes.size()
    );
    std::string modelReason;
    require(
        model.valid(&modelReason),
        "synthetic FunctionBased contact model is invalid: " + modelReason
    );

    metalrobo::CompiledWorld world;
    const auto compileDiagnostics = metalrobo::compileMetalWorld(
        model, 0u, world
    );
    require(
        compileDiagnostics.succeeded() && world.valid() &&
            world.sceneBodyCount() == 1u &&
            world.eligiblePairCount() == 1u,
        std::string("synthetic FunctionBased contact compilation failed: ") +
            metalrobo::metalWorldHostStatusName(
                compileDiagnostics.status
            ) + " " + compileDiagnostics.message
    );

    constexpr std::size_t controlSteps = 2u;
    require(
        !taskDrivenMillardExcitation ||
            (millardProgram != nullptr && millardExcitations.empty()),
        "task-driven Millard probe requires a source program and no packed host excitation stream"
    );
    const std::vector<float> efforts(
        controlSteps * static_cast<std::size_t>(world.nv()), 0.0f
    );
    std::vector<float> taskActions;
    std::vector<std::uint32_t> taskResetMasks;
    metalrobo::CompiledTaskProgram taskProgram;
    if (taskDrivenMillardExcitation) {
        const std::size_t muscleCount = millardProgram->muscles.size();
        metalrobo::TaskPack task;
        task.id = "numilab_human_source_millard_task_bridge_v1";
        task.maximumEpisodeSteps = 64u;
        task.difficultyBandCount = 1u;
        task.rewards = {{
            .operation = metalrobo::TaskRewardOperator::constant,
            .weight = 0.0f,
        }};
        metalrobo::TaskObservationProgram observations;
        observations.actorHistoryLength = 1u;
        observations.criticHistoryLength = 1u;
        std::vector<metalrobo::RobotActuatorSpec> actuators;
        actuators.reserve(muscleCount);
        for (std::size_t muscle = 0u; muscle < muscleCount; ++muscle) {
            const std::string id = "source_muscle_" +
                std::to_string(muscle);
            task.actions.push_back({.actuator = id});
            actuators.push_back({
                .id = id,
                .kind = metalrobo::RobotActuatorKind::millardExcitation,
                .target = id,
                .scale = 1.0f,
            });
            observations.actorFrame.push_back({
                .source = metalrobo::TaskObservationSource::previousAction,
                .target = id,
            });
            observations.critic.push_back({
                .source = metalrobo::TaskObservationSource::previousAction,
                .target = id,
            });
        }
        const auto taskDiagnostics = metalrobo::compileTaskProgram(
            task,
            actuators,
            observations,
            {},
            world,
            taskProgram
        );
        require(
            taskDiagnostics.succeeded() && taskProgram.valid() &&
                taskProgram.layout().actionCount == muscleCount,
            "source Millard native task compilation failed: " +
                taskDiagnostics.element + " " + taskDiagnostics.message
        );
        taskActions.assign(controlSteps * muscleCount, 1.0f);
        taskResetMasks.assign(controlSteps, 1u);
    }
    const std::vector<MRBodyStateGPU> scene{
        staticGroundState(
            groundBody,
            contactHeight - radius + 0.004f
        ),
    };
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = controlSteps,
        .initialQ = model.defaultQ,
        .initialV = model.defaultV,
        .efforts = taskDrivenMillardExcitation
            ? std::span<const float>{}
            : std::span<const float>{efforts},
        .actions = taskActions,
        .millardExcitations = millardExcitations,
        .resetMasks = taskResetMasks,
        .initialSceneBodies = scene,
    };
    metalrobo::MetalWorldStepConfig config{
        .timestepSeconds = 1.0f / 240.0f,
        .physicsSubsteps = 1u,
        .solverMode = metalrobo::MetalWorldSolverMode::temporalCone,
        .actuationMode = metalrobo::MetalWorldActuationMode::effort,
        .velocityIterations = 4u,
        .finalVelocityIterations = 2u,
        .ccdMode = metalrobo::MetalWorldCCDMode::disabled,
        .deterministic = true,
        .warmStart = false,
        .matrixFreeArticulatedContact = true,
        .streamedArticulatedContactResponses = true,
        .captureContactEvidence = true,
        .publishFinalState = true,
    };
    if (millardProgram != nullptr) {
        config.millardProgram = *millardProgram;
    }
    if (!millardExcitations.empty() || taskDrivenMillardExcitation) {
        // Explicit probe-only input contract. These values validate the
        // first-order control surface; they are not a source-parameter claim.
        config.millardActivationDynamics = {
            .activationTimeConstantSeconds = 0.01f,
            .deactivationTimeConstantSeconds = 0.04f,
        };
    }
    if (taskDrivenMillardExcitation) {
        config.taskProgram = taskProgram;
    }
    metalrobo::MetalWorldContext context;
    metalrobo::MetalWorldResult result;
    const auto deviceDiagnostics = context.run(world, batch, config, result);
    require(
        deviceDiagnostics.succeeded() && deviceDiagnostics.dispatched &&
            deviceDiagnostics.published &&
            deviceDiagnostics.successfulStepCount == controlSteps &&
            deviceDiagnostics.failedStepCount == 0u &&
            result.contactStatuses.size() == controlSteps,
        std::string("FunctionBased streamed-contact execution failed: ") +
            metalrobo::metalWorldHostStatusName(
                deviceDiagnostics.status
            ) + " " + deviceDiagnostics.message +
            " first_gpu_status=" +
            std::to_string(deviceDiagnostics.firstGPUStatusCode) +
            " final_q0=" +
            (result.finalQ.empty()
                ? std::string{"unpublished"}
                : std::to_string(result.finalQ.front()))
    );
    MetalWorldFunctionBasedContactMetrics metrics;
    metrics.deviceName = deviceDiagnostics.deviceName;
    metrics.successfulStepCount = deviceDiagnostics.successfulStepCount;
    for (const MRMetalWorldContactStatusGPU& status : result.contactStatuses) {
        metrics.maximumActiveContacts = std::max(
            metrics.maximumActiveContacts, status.activeContacts
        );
        metrics.maximumConstraints = std::max(
            metrics.maximumConstraints, status.requiredConstraints
        );
    }
    require(
        metrics.maximumActiveContacts > 0u &&
            metrics.maximumConstraints > 0u,
        "FunctionBased streamed-contact probe did not reach a device constraint"
    );
    if (millardProgram != nullptr) {
        require(
            result.millardResults.size() == millardProgram->muscles.size() &&
                result.millardGeneralizedForces.size() ==
                    millardProgram->muscles.size() * model.articulations.at(0u).nv &&
                result.millardStates.size() == millardProgram->muscles.size(),
            "FunctionBased streamed-contact probe did not publish source Millard state and forces"
        );
        for (const float force : result.millardGeneralizedForces) {
            require(
                std::isfinite(force),
                "FunctionBased streamed-contact probe published a non-finite Millard force"
            );
            metrics.millardGeneralizedForceL1 += std::abs(static_cast<double>(force));
        }
        require(
            metrics.millardGeneralizedForceL1 > 1.0e-3,
            "FunctionBased streamed-contact probe did not apply nonzero source Millard effort"
        );
        if (!millardExcitations.empty() || taskDrivenMillardExcitation) {
            const double timestep = config.timestepSeconds;
            for (std::size_t muscle = 0u;
                 muscle < millardProgram->muscles.size();
                 ++muscle) {
                double expected =
                    millardProgram->states[muscle].activationAndVelocity.x;
                const double minimumActivation =
                    millardProgram->muscles[muscle].dampingAndActivation.z;
                for (std::size_t controlStep = 0u;
                     controlStep < controlSteps;
                     ++controlStep) {
                    const double excitation = taskDrivenMillardExcitation
                        ? 1.0
                        : millardExcitations[
                              controlStep *
                                  millardProgram->muscles.size() + muscle
                          ];
                    const double target = std::clamp(
                        excitation, minimumActivation, 1.0
                    );
                    const double tau = target >= expected
                        ? config.millardActivationDynamics
                              .activationTimeConstantSeconds
                        : config.millardActivationDynamics
                              .deactivationTimeConstantSeconds;
                    expected = std::clamp(
                        target + (expected - target) *
                            std::exp(-timestep / tau),
                        minimumActivation,
                        1.0
                    );
                }
                const MRMillardMuscleStateGPU& observed =
                    result.millardStates[muscle];
                require(
                    std::isfinite(observed.activationAndVelocity.x) &&
                        observed.activationAndVelocity.y ==
                            millardProgram->states[muscle]
                                .activationAndVelocity.y &&
                        observed.activationAndVelocity.z == 0.0f &&
                        observed.activationAndVelocity.w == 0.0f,
                    "FunctionBased streamed-contact probe published malformed device activation state"
                );
                metrics.maximumMillardActivationError = std::max(
                    metrics.maximumMillardActivationError,
                    std::abs(
                        static_cast<double>(
                            observed.activationAndVelocity.x
                        ) - expected
                    )
                );
            }
            require(
                metrics.maximumMillardActivationError < 3.0e-6,
                "device Millard activation update diverged from exact first-order hold"
            );
        }
    }
    return metrics;
}

struct MetalMillardReferenceMetrics {
    std::string deviceName;
    std::uint32_t appliedCylinderWrapCount = 0u;
    double maximumPathLengthRelativeError = 0.0;
    double maximumTendonForceRelativeError = 0.0;
    double maximumGeneralizedForceL1RelativeError = 0.0;
    double maximumEquilibriumResidual = 0.0;
};

double relativeError(const double actual, const double reference) {
    return std::abs(actual - reference) /
        std::max(1.0, std::abs(reference));
}

struct MetalMillardProgramData {
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::vector<MRMillardMuscleGPU> muscles;
    std::vector<MRMillardMuscleStateGPU> states;
    std::vector<MRMillardPathPointGPU> pathPoints;
    std::vector<MRMillardSourceCurveGPU> curves;
    std::vector<MRMillardCylinderWrapGPU> wraps;

    [[nodiscard]] metalrobo::MetalWorldMillardProgram program() const {
        return {
            .articulationIndex = 0u,
            .pointQueries = points,
            .muscles = muscles,
            .states = states,
            .pathPoints = pathPoints,
            .curves = curves,
            .cylinderWraps = wraps,
        };
    }
};

MetalMillardProgramData materializeMetalMillardProgram(
    const MillardPayload& payload
) {
    MetalMillardProgramData data;
    data.points.resize(payload.pathPoints.size());
    data.pathPoints.resize(payload.pathPoints.size());
    for (std::size_t index = 0u; index < payload.pathPoints.size(); ++index) {
        const MillardPathPointRecord& source = payload.pathPoints[index];
        data.points[index] = {
            .bodyIndex = source.bodyIndex,
            .flags = 0u,
            .reserved0 = 0u,
            .reserved1 = 0u,
            .localPoint = {source.x, source.y, source.z, 0.0f},
            .worldImpulse = {0.0f, 0.0f, 0.0f, 0.0f},
        };
        data.pathPoints[index] = {
            .pointQueryIndex = static_cast<std::uint32_t>(index),
            .bodyIndex = source.bodyIndex,
            .reserved0 = 0u,
            .reserved1 = 0u,
        };
    }
    data.muscles.resize(payload.muscles.size());
    data.states.resize(payload.muscles.size());
    data.curves.resize(payload.curves.size());
    for (std::size_t index = 0u; index < payload.muscles.size(); ++index) {
        const MillardMuscleRecord& source = payload.muscles[index];
        data.muscles[index] = {
            .forceAndLengths = {
                source.maxIsometricForce,
                source.optimalFiberLength,
                source.tendonSlackLength,
                source.pennationAngleAtOptimal,
            },
            .dampingAndActivation = {
                source.fiberDamping,
                source.defaultActivation,
                source.minimumActivation,
                0.0f,
            },
            .pathAndWrap = {
                source.pathPointOffset,
                source.pathPointCount,
                source.pathWrapOffset,
                source.pathWrapCount,
            },
            .flags = {source.flags, 0u, 0u, 0u},
        };
        data.states[index] = {
            .activationAndVelocity = {
                std::max(source.defaultActivation, source.minimumActivation),
                0.0f,
                0.0f,
                0.0f,
            },
        };
        const MillardCurveRecord& sourceCurve = payload.curves[index];
        const std::array<float, 22u> sourceValues{
            sourceCurve.minNormActiveFiberLength,
            sourceCurve.transitionNormFiberLength,
            sourceCurve.maxNormActiveFiberLength,
            sourceCurve.shallowAscendingSlope,
            sourceCurve.activeMinimumValue,
            sourceCurve.concentricSlopeAtVmax,
            sourceCurve.concentricSlopeNearVmax,
            sourceCurve.isometricSlope,
            sourceCurve.eccentricSlopeAtVmax,
            sourceCurve.eccentricSlopeNearVmax,
            sourceCurve.maxEccentricVelocityForceMultiplier,
            sourceCurve.concentricCurviness,
            sourceCurve.eccentricCurviness,
            sourceCurve.fiberStrainAtZeroForce,
            sourceCurve.fiberStrainAtOneNormForce,
            sourceCurve.fiberStiffnessAtLowForce,
            sourceCurve.fiberStiffnessAtOneNormForce,
            sourceCurve.fiberCurviness,
            sourceCurve.tendonStrainAtOneNormForce,
            sourceCurve.tendonStiffnessAtOneNormForce,
            sourceCurve.tendonNormForceAtToeEnd,
            sourceCurve.tendonCurviness,
        };
        float* gpuCurveValues = &data.curves[index].values[0u].x;
        for (std::size_t value = 0u; value < sourceValues.size(); ++value) {
            gpuCurveValues[value] = sourceValues[value];
        }
    }
    data.wraps.resize(payload.wraps.size());
    for (std::size_t index = 0u; index < payload.wraps.size(); ++index) {
        const MillardWrapRecord& source = payload.wraps[index];
        data.wraps[index] = {
            .bodyIndex = source.bodyIndex,
            .startPoint = source.startPoint,
            .endPoint = source.endPoint,
            .method = source.method,
            .center = {source.centerX, source.centerY, source.centerZ, 0.0f},
            .rotationAndRadius = {
                source.rotationX,
                source.rotationY,
                source.rotationZ,
                source.radius,
            },
            .length = {source.length, 0.0f, 0.0f, 0.0f},
        };
    }
    return data;
}

MetalMillardReferenceMetrics verifyMetalMillardReference(
    const metalrobo::EngineModel& model,
    const MillardPayload& payload,
    const MillardReferenceMetrics& cpuMetrics
) {
    const MRArticulationGPU& articulation = model.articulations.at(0u);
    require(
        cpuMetrics.samples.size() == payload.muscles.size(),
        "CPU Millard source samples do not match the payload"
    );

    std::vector<MRArticulatedPointImpulseGPU> points(
        payload.pathPoints.size()
    );
    std::vector<MRMillardPathPointGPU> gpuPathPoints(
        payload.pathPoints.size()
    );
    for (std::size_t index = 0u; index < payload.pathPoints.size(); ++index) {
        const MillardPathPointRecord& source = payload.pathPoints[index];
        points[index] = {
            .bodyIndex = source.bodyIndex,
            .flags = 0u,
            .reserved0 = 0u,
            .reserved1 = 0u,
            .localPoint = {source.x, source.y, source.z, 0.0f},
            .worldImpulse = {0.0f, 0.0f, 0.0f, 0.0f},
        };
        gpuPathPoints[index] = {
            .pointQueryIndex = static_cast<std::uint32_t>(index),
            .bodyIndex = source.bodyIndex,
            .reserved0 = 0u,
            .reserved1 = 0u,
        };
    }

    std::vector<MRMillardMuscleGPU> muscles(payload.muscles.size());
    std::vector<MRMillardMuscleStateGPU> states(payload.muscles.size());
    std::vector<MRMillardSourceCurveGPU> curves(payload.curves.size());
    for (std::size_t index = 0u; index < payload.muscles.size(); ++index) {
        const MillardMuscleRecord& source = payload.muscles[index];
        muscles[index] = {
            .forceAndLengths = {
                source.maxIsometricForce,
                source.optimalFiberLength,
                source.tendonSlackLength,
                source.pennationAngleAtOptimal,
            },
            .dampingAndActivation = {
                source.fiberDamping,
                source.defaultActivation,
                source.minimumActivation,
                0.0f,
            },
            .pathAndWrap = {
                source.pathPointOffset,
                source.pathPointCount,
                source.pathWrapOffset,
                source.pathWrapCount,
            },
            .flags = {source.flags, 0u, 0u, 0u},
        };
        states[index] = {
            .activationAndVelocity = {
                std::max(source.defaultActivation, source.minimumActivation),
                0.0f,
                0.0f,
                0.0f,
            },
        };
        const MillardCurveRecord& sourceCurve = payload.curves[index];
        const std::array<float, 22u> sourceValues{
            sourceCurve.minNormActiveFiberLength,
            sourceCurve.transitionNormFiberLength,
            sourceCurve.maxNormActiveFiberLength,
            sourceCurve.shallowAscendingSlope,
            sourceCurve.activeMinimumValue,
            sourceCurve.concentricSlopeAtVmax,
            sourceCurve.concentricSlopeNearVmax,
            sourceCurve.isometricSlope,
            sourceCurve.eccentricSlopeAtVmax,
            sourceCurve.eccentricSlopeNearVmax,
            sourceCurve.maxEccentricVelocityForceMultiplier,
            sourceCurve.concentricCurviness,
            sourceCurve.eccentricCurviness,
            sourceCurve.fiberStrainAtZeroForce,
            sourceCurve.fiberStrainAtOneNormForce,
            sourceCurve.fiberStiffnessAtLowForce,
            sourceCurve.fiberStiffnessAtOneNormForce,
            sourceCurve.fiberCurviness,
            sourceCurve.tendonStrainAtOneNormForce,
            sourceCurve.tendonStiffnessAtOneNormForce,
            sourceCurve.tendonNormForceAtToeEnd,
            sourceCurve.tendonCurviness,
        };
        float* gpuCurveValues = &curves[index].values[0u].x;
        for (std::size_t value = 0u; value < sourceValues.size(); ++value) {
            gpuCurveValues[value] = sourceValues[value];
        }
    }
    std::vector<MRMillardCylinderWrapGPU> wraps(payload.wraps.size());
    for (std::size_t index = 0u; index < payload.wraps.size(); ++index) {
        const MillardWrapRecord& source = payload.wraps[index];
        wraps[index] = {
            .bodyIndex = source.bodyIndex,
            .startPoint = source.startPoint,
            .endPoint = source.endPoint,
            .method = source.method,
            .center = {source.centerX, source.centerY, source.centerZ, 0.0f},
            .rotationAndRadius = {
                source.rotationX,
                source.rotationY,
                source.rotationZ,
                source.radius,
            },
            .length = {source.length, 0.0f, 0.0f, 0.0f},
        };
    }

    metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = points.size(),
        .q = model.defaultQ,
        .points = points,
        .millard = {
            .muscles = muscles,
            .states = states,
            .pathPoints = gpuPathPoints,
            .curves = curves,
            .cylinderWraps = wraps,
        },
    };
    metalrobo::MetalArticulatedOperatorConfig config{
        .pointJacobiansOnly = true,
    };
    metalrobo::MetalArticulatedOperatorResult gpu;
    const auto diagnostics = metalrobo::runMetalArticulatedOperator(
        model, input, gpu, config
    );
    require(
        diagnostics.succeeded() && diagnostics.dispatched && diagnostics.published &&
            gpu.millardResults.size() == muscles.size() &&
            gpu.millardGeneralizedForces.size() ==
                muscles.size() * articulation.nv,
        std::string("Metal Millard reference pass failed: ") +
            metalrobo::metalArticulatedOperatorHostStatusName(diagnostics.status) +
            " " + diagnostics.message
    );

    MetalMillardReferenceMetrics metrics;
    metrics.deviceName = diagnostics.deviceName;
    for (std::size_t index = 0u; index < muscles.size(); ++index) {
        const MRMillardMuscleResultGPU& result = gpu.millardResults[index];
        const MillardReferenceMetrics::MuscleSample& cpu =
            cpuMetrics.samples[index];
        require(
            result.status == MR_MILLARD_REFERENCE_SUCCESS &&
                result.environment == 0u && result.muscleIndex == index &&
                std::isfinite(result.pathFiberTendonResidual.x) &&
                std::isfinite(result.pathFiberTendonResidual.y) &&
                std::isfinite(result.pathFiberTendonResidual.z) &&
                std::isfinite(result.pathFiberTendonResidual.w),
            "Metal Millard result is malformed at muscle " + std::to_string(index)
        );
        double forceL1 = 0.0;
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            forceL1 += std::abs(static_cast<double>(gpu.millardGeneralizedForces[
                index * articulation.nv + dof
            ]));
        }
        metrics.appliedCylinderWrapCount += result.appliedCylinderWrapCount;
        metrics.maximumPathLengthRelativeError = std::max(
            metrics.maximumPathLengthRelativeError,
            relativeError(result.pathFiberTendonResidual.x, cpu.pathLength)
        );
        metrics.maximumTendonForceRelativeError = std::max(
            metrics.maximumTendonForceRelativeError,
            relativeError(result.pathFiberTendonResidual.z, cpu.tendonForce)
        );
        metrics.maximumGeneralizedForceL1RelativeError = std::max(
            metrics.maximumGeneralizedForceL1RelativeError,
            relativeError(forceL1, cpu.generalizedForceL1)
        );
        metrics.maximumEquilibriumResidual = std::max(
            metrics.maximumEquilibriumResidual,
            std::abs(static_cast<double>(result.pathFiberTendonResidual.w))
        );
        require(
            result.appliedCylinderWrapCount == cpu.appliedCylinderWrapCount,
            "Metal Millard wrap selection disagrees with the source bridge at muscle " +
                std::to_string(index)
        );
    }
    require(
        metrics.maximumPathLengthRelativeError < 2.0e-4 &&
            metrics.maximumTendonForceRelativeError < 5.0e-3 &&
            metrics.maximumGeneralizedForceL1RelativeError < 1.0e-2 &&
            metrics.maximumEquilibriumResidual < 0.1,
        "Metal Millard reference parity exceeded its FP32 source-bridge gate"
    );
    return metrics;
}

MetalReferenceMetrics verifyMetalFunctionBasedOperator(
    const metalrobo::EngineModel& model
) {
    const MRArticulationGPU& articulation = model.articulations.at(0u);
    std::vector<float> q = model.defaultQ;
    for (std::size_t index = 0u; index < q.size(); ++index) {
        q[index] += 0.05f * std::sin(
            0.43f * static_cast<float>(index + 1u)
        );
    }

    std::vector<MRArticulatedPointImpulseGPU> gpuPoints(
        static_cast<std::size_t>(articulation.bodyCount) * 3u
    );
    std::vector<metalrobo::ArticulatedPointQuery> cpuPoints(
        static_cast<std::size_t>(articulation.bodyCount) * 3u
    );
    for (std::size_t localBody = 0u;
         localBody < articulation.bodyCount;
         ++localBody) {
        const std::uint32_t globalBody =
            articulation.firstBody + static_cast<std::uint32_t>(localBody);
        const float phase = static_cast<float>(localBody + 1u);
        const std::array<double, 3u> localPoint{
            0.004 * std::sin(0.31 * static_cast<double>(phase)),
            0.003 * std::cos(0.47 * static_cast<double>(phase)),
            0.002 * std::sin(0.59 * static_cast<double>(phase)),
        };
        const std::array<std::array<double, 3u>, 3u> pointsForBody{
            localPoint,
            {
                localPoint[0] + 0.071,
                localPoint[1] - 0.019,
                localPoint[2] + 0.013,
            },
            {
                localPoint[0] - 0.023,
                localPoint[1] + 0.067,
                localPoint[2] - 0.017,
            },
        };
        for (std::size_t pointInBody = 0u;
             pointInBody < pointsForBody.size();
             ++pointInBody) {
            const std::size_t pointIndex = localBody * 3u + pointInBody;
            const std::array<double, 3u>& point = pointsForBody[pointInBody];
            MRArticulatedPointImpulseGPU& gpuPoint = gpuPoints[pointIndex];
            gpuPoint.bodyIndex = globalBody;
            gpuPoint.flags = 0u;
            gpuPoint.reserved0 = 0u;
            gpuPoint.reserved1 = 0u;
            gpuPoint.localPoint = {
                static_cast<float>(point[0]),
                static_cast<float>(point[1]),
                static_cast<float>(point[2]),
                0.0f,
            };
            gpuPoint.worldImpulse = {0.0f, 0.0f, 0.0f, 0.0f};
            cpuPoints[pointIndex] = {globalBody, point};
        }
    }

    const std::vector<double> qDouble(q.begin(), q.end());
    const std::vector<double> zeroVelocity(articulation.nv, 0.0);
    std::vector<metalrobo::ArticulatedBodyKinematics> cpuBodies(
        articulation.bodyCount
    );
    auto cpuDiagnostics = metalrobo::computeArticulatedBodyKinematics(
        model, 0u, qDouble, zeroVelocity, cpuBodies
    );
    require(cpuDiagnostics.succeeded(), "CPU FunctionBased body reference failed");
    std::vector<metalrobo::ArticulatedPointKinematics> cpuPointKinematics(
        cpuPoints.size()
    );
    std::vector<double> cpuJacobians(
        cpuPoints.size() * 3u * articulation.nv
    );
    cpuDiagnostics = metalrobo::computeArticulatedPointJacobians(
        model,
        0u,
        qDouble,
        zeroVelocity,
        cpuPoints,
        cpuPointKinematics,
        cpuJacobians
    );
    require(cpuDiagnostics.succeeded(), "CPU FunctionBased Jacobian reference failed");
    std::vector<double> cpuMass(
        static_cast<std::size_t>(articulation.nv) * articulation.nv
    );
    cpuDiagnostics = metalrobo::computeArticulatedMassMatrix(
        model, 0u, qDouble, cpuMass
    );
    require(cpuDiagnostics.succeeded(), "CPU FunctionBased mass reference failed");

    metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = gpuPoints.size(),
        .q = q,
        .points = gpuPoints,
    };
    metalrobo::MetalArticulatedOperatorConfig config{
        .pointJacobiansOnly = true,
    };
    metalrobo::MetalArticulatedOperatorResult gpuResult;
    const auto gpuDiagnostics = metalrobo::runMetalArticulatedOperator(
        model, input, gpuResult, config
    );
    std::string gpuFailureDetail;
    if (!gpuResult.statuses.empty()) {
        const MRArticulatedOperatorStatusGPU& status =
            gpuResult.statuses.front();
        gpuFailureDetail =
            " gpu_status=" + std::to_string(status.code) +
            " gpu_failing_index=" + std::to_string(status.failingIndex) +
            " gpu_diagnostics=" +
            std::to_string(status.diagnostics.x) + "," +
            std::to_string(status.diagnostics.y) + "," +
            std::to_string(status.diagnostics.z) + "," +
            std::to_string(status.diagnostics.w);
    }
    require(
        gpuDiagnostics.succeeded() && gpuDiagnostics.dispatched &&
            gpuDiagnostics.published &&
            gpuDiagnostics.successfulEnvironmentCount == 1u &&
            gpuDiagnostics.failedEnvironmentCount == 0u,
        std::string("Metal FunctionBased kinematics/Jacobian operator failed: ") +
            metalrobo::metalArticulatedOperatorHostStatusName(
                gpuDiagnostics.status
            ) + " " + gpuDiagnostics.message +
            " first_gpu_status=" +
            std::to_string(gpuDiagnostics.firstGPUStatusCode) +
            gpuFailureDetail
    );
    require(
        gpuResult.bodyPoses.size() == cpuBodies.size() &&
            gpuResult.pointWorld.size() == cpuPointKinematics.size() &&
            gpuResult.pointJacobians.size() == cpuJacobians.size(),
        "Metal FunctionBased kinematics/Jacobian operator published an unexpected layout"
    );

    MetalReferenceMetrics metrics;
    metrics.deviceName = gpuDiagnostics.deviceName;
    for (std::size_t body = 0u; body < cpuBodies.size(); ++body) {
        const MRArticulatedBodyPoseGPU& gpuBody = gpuResult.bodyPoses[body];
        const std::array<float, 3u> gpuPosition{
            gpuBody.position.x, gpuBody.position.y, gpuBody.position.z,
        };
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            metrics.bodyPositionError = std::max(
                metrics.bodyPositionError,
                std::abs(
                    static_cast<double>(gpuPosition[axis]) -
                    cpuBodies[body].centerOfMassPosition[axis]
                )
            );
        }
        const std::array<float, 4u> gpuOrientation{
            gpuBody.orientation.x,
            gpuBody.orientation.y,
            gpuBody.orientation.z,
            gpuBody.orientation.w,
        };
        double sameSign = 0.0;
        double flippedSign = 0.0;
        for (std::size_t component = 0u; component < 4u; ++component) {
            sameSign = std::max(
                sameSign,
                std::abs(
                    static_cast<double>(gpuOrientation[component]) -
                    cpuBodies[body].orientation[component]
                )
            );
            flippedSign = std::max(
                flippedSign,
                std::abs(
                    static_cast<double>(gpuOrientation[component]) +
                    cpuBodies[body].orientation[component]
                )
            );
        }
        metrics.bodyOrientationError = std::max(
            metrics.bodyOrientationError, std::min(sameSign, flippedSign)
        );
    }
    for (std::size_t point = 0u; point < cpuPointKinematics.size(); ++point) {
        const mr_float4 gpuPoint = gpuResult.pointWorld[point].position;
        const std::array<float, 3u> gpuPosition{
            gpuPoint.x, gpuPoint.y, gpuPoint.z,
        };
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            metrics.pointPositionError = std::max(
                metrics.pointPositionError,
                std::abs(
                    static_cast<double>(gpuPosition[axis]) -
                    cpuPointKinematics[point].position[axis]
                )
            );
        }
    }
    for (std::size_t index = 0u; index < cpuJacobians.size(); ++index) {
        metrics.pointJacobianError = std::max(
            metrics.pointJacobianError,
            std::abs(
                static_cast<double>(gpuResult.pointJacobians[index]) -
                cpuJacobians[index]
            )
        );
    }
    require(
        metrics.bodyPositionError < 2.0e-4 &&
            metrics.bodyOrientationError < 2.0e-4 &&
            metrics.pointPositionError < 2.0e-4 &&
            metrics.pointJacobianError < 5.0e-4,
        "Metal FunctionBased kinematics/Jacobian parity exceeded FP32 tolerance"
    );

    metalrobo::MetalArticulatedOperatorConfig massConfig{
        .writeDiagnosticMassMatrix = true,
    };
    metalrobo::MetalArticulatedOperatorResult massResult;
    const auto massDiagnostics = metalrobo::runMetalArticulatedOperator(
        model, input, massResult, massConfig
    );
    std::string massFailureDetail;
    if (!massResult.statuses.empty()) {
        const MRArticulatedOperatorStatusGPU& status =
            massResult.statuses.front();
        massFailureDetail =
            " gpu_status=" + std::to_string(status.code) +
            " gpu_failing_index=" + std::to_string(status.failingIndex) +
            " gpu_diagnostics=" +
            std::to_string(status.diagnostics.x) + "," +
            std::to_string(status.diagnostics.y) + "," +
            std::to_string(status.diagnostics.z) + "," +
            std::to_string(status.diagnostics.w);
    }
    require(
        massDiagnostics.succeeded() && massDiagnostics.dispatched &&
            massDiagnostics.published &&
            massDiagnostics.successfulEnvironmentCount == 1u &&
            massDiagnostics.failedEnvironmentCount == 0u,
        std::string("Metal FunctionBased mass operator failed: ") +
            metalrobo::metalArticulatedOperatorHostStatusName(
                massDiagnostics.status
            ) + " " + massDiagnostics.message +
            " first_gpu_status=" +
            std::to_string(massDiagnostics.firstGPUStatusCode) +
            massFailureDetail
    );
    require(
        massResult.diagnosticMassMatrix.size() == cpuMass.size(),
        "Metal FunctionBased mass operator published an unexpected layout"
    );
    for (std::size_t index = 0u; index < cpuMass.size(); ++index) {
        const double error = std::abs(
            static_cast<double>(massResult.diagnosticMassMatrix[index]) -
            cpuMass[index]
        );
        metrics.massError = std::max(metrics.massError, error);
        metrics.massScaledError = std::max(
            metrics.massScaledError,
            error / (1.0 + std::abs(cpuMass[index]))
        );
    }
    require(
        metrics.massError < 5.0e-3 &&
            metrics.massScaledError < 2.0e-4,
        "Metal FunctionBased mass parity exceeded FP32 tolerance"
    );
    return metrics;
}

MetalWorldFunctionBasedMetrics verifyMetalWorldFunctionBasedDynamics(
    const metalrobo::EngineModel& model
) {
    const MRArticulationGPU& articulation = model.articulations.at(0u);
    constexpr std::size_t controlSteps = 3u;
    constexpr float timestep = 1.0f / 240.0f;
    std::vector<float> initialQ = model.defaultQ;
    std::vector<float> initialV = model.defaultV;
    for (std::size_t index = 0u; index < initialQ.size(); ++index) {
        initialQ[index] += 0.025f * std::sin(
            0.37f * static_cast<float>(index + 1u)
        );
        initialV[index] = 0.06f * std::cos(
            0.29f * static_cast<float>(index + 1u)
        );
    }
    std::vector<float> efforts(
        controlSteps * static_cast<std::size_t>(articulation.nv),
        0.0f
    );

    metalrobo::CompiledWorld world;
    const auto compileDiagnostics = metalrobo::compileMetalWorld(
        model, 0u, world
    );
    require(
        compileDiagnostics.succeeded() && world.valid(),
        std::string("MetalWorld FunctionBased compilation failed: ") +
            metalrobo::metalWorldHostStatusName(
                compileDiagnostics.status
            ) + " " + compileDiagnostics.message
    );

    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = controlSteps,
        .initialQ = initialQ,
        .initialV = initialV,
        .efforts = efforts,
    };
    const metalrobo::MetalWorldStepConfig config{
        .timestepSeconds = timestep,
        .physicsSubsteps = 1u,
        .solverMode = metalrobo::MetalWorldSolverMode::freeMotionABA,
        .actuationMode = metalrobo::MetalWorldActuationMode::effort,
        .applyBodyDamping = true,
        .deterministic = true,
        .publishFinalState = true,
        .publishStateTrajectory = true,
    };
    metalrobo::MetalWorldContext context;
    metalrobo::MetalWorldResult result;
    const auto deviceDiagnostics = context.run(
        world, batch, config, result
    );
    require(
        deviceDiagnostics.succeeded() && deviceDiagnostics.dispatched &&
            deviceDiagnostics.published &&
            deviceDiagnostics.successfulStepCount == controlSteps &&
            deviceDiagnostics.failedStepCount == 0u,
        std::string("MetalWorld FunctionBased dense dynamics failed: ") +
            metalrobo::metalWorldHostStatusName(
                deviceDiagnostics.status
            ) + " " + deviceDiagnostics.message +
            " first_gpu_status=" +
            std::to_string(deviceDiagnostics.firstGPUStatusCode)
    );
    require(
        result.finalQ.size() == initialQ.size() &&
            result.finalV.size() == initialV.size() &&
            result.accelerations.size() == efforts.size(),
        "MetalWorld FunctionBased dynamics published an unexpected state layout"
    );

    metalrobo::ArticulatedDynamicsConfig cpuConfig;
    cpuConfig.gravity = {
        static_cast<double>(model.world.gravityAndTimestep.x),
        static_cast<double>(model.world.gravityAndTimestep.y),
        static_cast<double>(model.world.gravityAndTimestep.z),
    };
    cpuConfig.timestep = timestep;
    cpuConfig.applyBodyDamping = true;
    cpuConfig.integrator = metalrobo::ArticulatedIntegrator::symplecticEuler;
    std::vector<double> q(initialQ.begin(), initialQ.end());
    std::vector<double> v(initialV.begin(), initialV.end());
    std::vector<double> force(articulation.nv, 0.0);
    std::vector<double> acceleration(articulation.nv, 0.0);

    MetalWorldFunctionBasedMetrics metrics;
    metrics.deviceName = deviceDiagnostics.deviceName;
    metrics.successfulStepCount = deviceDiagnostics.successfulStepCount;
    for (std::size_t step = 0u; step < controlSteps; ++step) {
        const auto forwardDiagnostics =
            metalrobo::computeArticulatedForwardDynamics(
                model, 0u, q, v, force, {}, acceleration, cpuConfig
            );
        require(
            forwardDiagnostics.succeeded(),
            "CPU FunctionBased forward dynamics failed during device parity"
        );
        for (std::size_t dof = 0u; dof < acceleration.size(); ++dof) {
            const double deviceAcceleration = result.accelerations[
                step * acceleration.size() + dof
            ];
            metrics.maximumAccelerationError = std::max(
                metrics.maximumAccelerationError,
                std::abs(deviceAcceleration - acceleration[dof])
            );
        }
        const auto integrateDiagnostics = metalrobo::integrateArticulatedState(
            model, 0u, q, v, force, {}, cpuConfig
        );
        require(
            integrateDiagnostics.succeeded(),
            "CPU FunctionBased state integration failed during device parity"
        );
    }
    for (std::size_t index = 0u; index < q.size(); ++index) {
        metrics.maximumConfigurationError = std::max(
            metrics.maximumConfigurationError,
            std::abs(static_cast<double>(result.finalQ[index]) - q[index])
        );
        metrics.maximumVelocityError = std::max(
            metrics.maximumVelocityError,
            std::abs(static_cast<double>(result.finalV[index]) - v[index])
        );
    }
    require(
        metrics.maximumAccelerationError < 2.5e-2 &&
            metrics.maximumVelocityError < 3.0e-4 &&
            metrics.maximumConfigurationError < 3.0e-5,
        "MetalWorld FunctionBased dense dynamics exceeded FP32 source parity"
    );
    return metrics;
}

struct MetalWorldMillardActuationMetrics {
    std::string deviceName;
    double maximumAccelerationError = 0.0;
    double maximumAccelerationRelativeError = 0.0;
    double accelerationL1RelativeError = 0.0;
    double generalizedForceL1RelativeError = 0.0;
    double maximumVelocityDeltaFromPassive = 0.0;
    double generalizedForceL1 = 0.0;
    std::uint32_t muscleCount = 0u;
};

MetalWorldMillardActuationMetrics verifyMetalWorldMillardActuation(
    const metalrobo::EngineModel& model,
    const MillardPayload& payload,
    const MillardReferenceMetrics& cpuMillard
) {
    const MRArticulationGPU& articulation = model.articulations.at(0u);
    constexpr float timestep = 1.0f / 240.0f;
    const std::vector<float> efforts(articulation.nv, 0.0f);
    const metalrobo::MetalWorldBatch batch{
        .environmentCount = 1u,
        .controlStepCount = 1u,
        .initialQ = model.defaultQ,
        .initialV = model.defaultV,
        .efforts = efforts,
    };
    const metalrobo::MetalWorldStepConfig baseConfig{
        .timestepSeconds = timestep,
        .physicsSubsteps = 1u,
        .solverMode = metalrobo::MetalWorldSolverMode::freeMotionABA,
        .actuationMode = metalrobo::MetalWorldActuationMode::effort,
        .applyBodyDamping = true,
        .deterministic = true,
        .publishFinalState = true,
        .publishStateTrajectory = true,
    };
    metalrobo::CompiledWorld world;
    const auto compileDiagnostics = metalrobo::compileMetalWorld(
        model, 0u, world
    );
    require(
        compileDiagnostics.succeeded() && world.valid(),
        "MetalWorld Millard actuation compilation failed"
    );

    metalrobo::MetalWorldContext passiveContext;
    metalrobo::MetalWorldResult passive;
    const auto passiveDiagnostics = passiveContext.run(
        world, batch, baseConfig, passive
    );
    require(
        passiveDiagnostics.succeeded() && passiveDiagnostics.published,
        "passive MetalWorld reference failed before Millard actuation"
    );

    const MetalMillardProgramData programData =
        materializeMetalMillardProgram(payload);
    metalrobo::MetalWorldStepConfig activeConfig = baseConfig;
    activeConfig.millardProgram = programData.program();
    metalrobo::MetalWorldContext activeContext;
    metalrobo::MetalWorldResult active;
    const auto activeDiagnostics = activeContext.run(
        world, batch, activeConfig, active
    );
    require(
        activeDiagnostics.succeeded() && activeDiagnostics.dispatched &&
            activeDiagnostics.published &&
            active.millardResults.size() == payload.muscles.size() &&
            active.millardGeneralizedForces.size() ==
                payload.muscles.size() * articulation.nv,
        std::string("MetalWorld Millard actuation failed: ") +
            metalrobo::metalWorldHostStatusName(activeDiagnostics.status) +
            " " + activeDiagnostics.message
    );

    std::vector<double> cpuEffort(articulation.nv, 0.0);
    for (const MillardReferenceMetrics::MuscleSample& sample :
         cpuMillard.samples) {
        require(
            sample.generalizedForces.size() == articulation.nv,
            "CPU Millard source force sample has an unexpected DoF layout"
        );
        for (std::size_t dof = 0u; dof < cpuEffort.size(); ++dof) {
            cpuEffort[dof] += sample.generalizedForces[dof];
        }
    }
    metalrobo::ArticulatedDynamicsConfig cpuConfig;
    cpuConfig.gravity = {
        static_cast<double>(model.world.gravityAndTimestep.x),
        static_cast<double>(model.world.gravityAndTimestep.y),
        static_cast<double>(model.world.gravityAndTimestep.z),
    };
    cpuConfig.timestep = timestep;
    cpuConfig.applyBodyDamping = true;
    cpuConfig.integrator = metalrobo::ArticulatedIntegrator::symplecticEuler;
    std::vector<double> q(model.defaultQ.begin(), model.defaultQ.end());
    std::vector<double> v(model.defaultV.begin(), model.defaultV.end());
    std::vector<double> cpuAcceleration(articulation.nv, 0.0);
    const auto cpuDiagnostics = metalrobo::computeArticulatedForwardDynamics(
        model, 0u, q, v, cpuEffort, {}, cpuAcceleration, cpuConfig
    );
    require(
        cpuDiagnostics.succeeded(),
        "CPU Millard-actuated forward dynamics failed"
    );

    MetalWorldMillardActuationMetrics metrics;
    metrics.deviceName = activeDiagnostics.deviceName;
    metrics.muscleCount = static_cast<std::uint32_t>(payload.muscles.size());
    std::vector<double> deviceEffort(articulation.nv, 0.0);
    for (std::size_t muscle = 0u; muscle < payload.muscles.size(); ++muscle) {
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            deviceEffort[dof] += active.millardGeneralizedForces[
                muscle * articulation.nv + dof
            ];
        }
    }
    double cpuForceL1 = 0.0;
    double forceDifferenceL1 = 0.0;
    double cpuAccelerationL1 = 0.0;
    double accelerationDifferenceL1 = 0.0;
    for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
        cpuForceL1 += std::abs(cpuEffort[dof]);
        forceDifferenceL1 += std::abs(deviceEffort[dof] - cpuEffort[dof]);
        metrics.generalizedForceL1 += std::abs(deviceEffort[dof]);
        metrics.maximumAccelerationError = std::max(
            metrics.maximumAccelerationError,
            std::abs(
                static_cast<double>(active.accelerations[dof]) -
                cpuAcceleration[dof]
            )
        );
        metrics.maximumAccelerationRelativeError = std::max(
            metrics.maximumAccelerationRelativeError,
            std::abs(
                static_cast<double>(active.accelerations[dof]) -
                cpuAcceleration[dof]
            ) / std::max(1.0, std::abs(cpuAcceleration[dof]))
        );
        cpuAccelerationL1 += std::abs(cpuAcceleration[dof]);
        accelerationDifferenceL1 += std::abs(
            static_cast<double>(active.accelerations[dof]) -
            cpuAcceleration[dof]
        );
        metrics.maximumVelocityDeltaFromPassive = std::max(
            metrics.maximumVelocityDeltaFromPassive,
            std::abs(
                static_cast<double>(active.finalV[dof]) -
                passive.finalV[dof]
            )
        );
    }
    metrics.generalizedForceL1RelativeError = forceDifferenceL1 /
        std::max(1.0, cpuForceL1);
    metrics.accelerationL1RelativeError = accelerationDifferenceL1 /
        std::max(1.0, cpuAccelerationL1);
    require(
        metrics.generalizedForceL1 > 1.0e-3 &&
            metrics.maximumVelocityDeltaFromPassive > 1.0e-6 &&
            metrics.generalizedForceL1RelativeError < 3.0e-3 &&
            metrics.accelerationL1RelativeError < 5.0e-4,
        "MetalWorld Millard source effort did not meet active device/CPU parity gates "
            "force_l1=" + std::to_string(metrics.generalizedForceL1) +
            " force_relative_error=" +
            std::to_string(metrics.generalizedForceL1RelativeError) +
            " acceleration_error=" +
            std::to_string(metrics.maximumAccelerationError) +
            " acceleration_relative_error=" +
            std::to_string(metrics.maximumAccelerationRelativeError) +
            " acceleration_l1_relative_error=" +
            std::to_string(metrics.accelerationL1RelativeError) +
            " velocity_delta=" +
            std::to_string(metrics.maximumVelocityDeltaFromPassive)
    );
    return metrics;
}

} // namespace

int main(const int argc, char** argv) {
    try {
        require(argc >= 2, "missing NHRIGID payload");
        bool runMetal = false;
        const char* millardPath = nullptr;
        for (int index = 2; index < argc; ++index) {
            const std::string argument(argv[index]);
            if (argument == "--metal") {
                require(!runMetal, "--metal was specified more than once");
                runMetal = true;
            } else if (argument == "--millard") {
                require(
                    millardPath == nullptr && index + 1 < argc,
                    "--millard requires exactly one payload path"
                );
                millardPath = argv[++index];
            } else {
                throw std::runtime_error(
                    "usage: metalrobo_numilab_human_core_reference_probe "
                    "<payload.nhrigid> [--metal] [--millard <payload.nhmuscle>]"
                );
            }
        }
        PayloadHeader header{};
        const metalrobo::EngineModel model = loadReference(argv[1], header);
        const MillardPayload millardPayload = millardPath != nullptr
            ? loadMillardReference(millardPath, header, model)
            : MillardPayload{};
        std::vector<double> q(model.defaultQ.begin(), model.defaultQ.end());
        std::vector<double> v(model.defaultV.size(), 0.0);
        std::vector<double> acceleration(model.defaultV.size(), 0.0);
        for (std::size_t index = 0u; index < v.size(); ++index) {
            v[index] = 0.01 * std::sin(0.37 * static_cast<double>(index + 1u));
            acceleration[index] = 0.02 * std::cos(0.23 * static_cast<double>(index + 1u));
        }

        metalrobo::ArticulatedDynamicsConfig config;
        config.gravity = {
            model.world.gravityAndTimestep.x,
            model.world.gravityAndTimestep.y,
            model.world.gravityAndTimestep.z,
        };
        config.timestep = model.world.gravityAndTimestep.w;
        config.applyBodyDamping = false;
        std::vector<metalrobo::ArticulatedBodyKinematics> kinematics(model.bodies.size());
        const auto kinematicsDiagnostics = metalrobo::computeArticulatedBodyKinematics(
            model, 0u, q, v, kinematics, config
        );
        require(kinematicsDiagnostics.succeeded(), "full Rajagopal kinematics failed");

        std::vector<double> massMatrix(v.size() * v.size(), 0.0);
        const auto massDiagnostics = metalrobo::computeArticulatedMassMatrix(
            model, 0u, q, massMatrix, config
        );
        require(massDiagnostics.succeeded(), "full Rajagopal mass-matrix assembly failed");
        require(
            massDiagnostics.minimumCholeskyPivot > 0.0 && allFinite(massMatrix),
            "full Rajagopal mass matrix is not finite positive definite"
        );
        std::vector<double> responseRhs(v.size(), 0.0);
        responseRhs.front() = 1.0;
        std::vector<double> inverseMassResponse(v.size(), 0.0);
        const auto responseDiagnostics = metalrobo::computeArticulatedInverseMassResponses(
            model, 0u, q, responseRhs, inverseMassResponse, config
        );
        require(responseDiagnostics.succeeded() && allFinite(inverseMassResponse),
            "FunctionBased inverse-mass response failed");
        double responseRecoveryError = 0.0;
        for (std::size_t row = 0u; row < v.size(); ++row) {
            double recovered = 0.0;
            for (std::size_t column = 0u; column < v.size(); ++column) {
                recovered += massMatrix[row * v.size() + column] * inverseMassResponse[column];
            }
            responseRecoveryError = std::max(responseRecoveryError, std::abs(recovered - responseRhs[row]));
        }
        require(responseRecoveryError < 1.0e-10, "FunctionBased inverse-mass response recovery failed");

        double massSymmetryError = 0.0;
        for (std::size_t row = 0u; row < v.size(); ++row) {
            for (std::size_t column = 0u; column < v.size(); ++column) {
                massSymmetryError = std::max(
                    massSymmetryError,
                    std::abs(massMatrix[row * v.size() + column] -
                             massMatrix[column * v.size() + row])
                );
            }
        }
        require(massSymmetryError < 1.0e-10, "full Rajagopal mass matrix is asymmetric");

        std::vector<double> generalizedForce(v.size(), 0.0);
        const auto inverseDiagnostics = metalrobo::computeArticulatedInverseDynamics(
            model, 0u, q, v, acceleration, {}, generalizedForce, config
        );
        require(inverseDiagnostics.succeeded(), "full Rajagopal inverse dynamics failed");
        std::vector<double> recoveredAcceleration(v.size(), 0.0);
        const auto forwardDiagnostics = metalrobo::computeArticulatedForwardDynamics(
            model, 0u, q, v, generalizedForce, {}, recoveredAcceleration, config
        );
        require(forwardDiagnostics.succeeded(), "full Rajagopal forward dynamics failed");
        double forwardInverseError = 0.0;
        for (std::size_t index = 0u; index < acceleration.size(); ++index) {
            forwardInverseError = std::max(
                forwardInverseError,
                std::abs(recoveredAcceleration[index] - acceleration[index])
            );
        }
        require(
            allFinite(generalizedForce) && allFinite(recoveredAcceleration) &&
                forwardInverseError < 2.0e-7,
            "full Rajagopal forward/inverse closure exceeded tolerance"
        );

        metalrobo::ArticulatedInvariants invariants{};
        const auto invariantDiagnostics = metalrobo::computeArticulatedInvariants(
            model, 0u, q, v, invariants, config
        );
        require(invariantDiagnostics.succeeded(), "full Rajagopal invariant evaluation failed");
        require(
            std::isfinite(invariants.kineticEnergy) &&
                std::isfinite(invariants.potentialEnergy) &&
                std::isfinite(invariants.totalEnergy),
            "full Rajagopal invariants are non-finite"
        );

        const MetalReferenceMetrics metalMetrics = runMetal
            ? verifyMetalFunctionBasedOperator(model)
            : MetalReferenceMetrics{};
        const MillardReferenceMetrics millardMetrics = millardPath != nullptr
            ? verifyMillardReference(model, millardPayload)
            : MillardReferenceMetrics{};
        const MetalMillardProgramData millardProgramData = millardPath != nullptr
            ? materializeMetalMillardProgram(millardPayload)
            : MetalMillardProgramData{};
        const metalrobo::MetalWorldMillardProgram millardProgram =
            millardPath != nullptr
            ? millardProgramData.program()
            : metalrobo::MetalWorldMillardProgram{};
        // The contact probe encodes two control periods. Drive both with a
        // normalized source-muscle excitation and let Metal update its own
        // private activation state before force projection.
        const std::vector<float> fullyExcitedMillardControls =
            millardPath != nullptr
            ? std::vector<float>(
                  2u * millardProgram.muscles.size(),
                  1.0f
              )
            : std::vector<float>{};
        const MetalWorldFunctionBasedMetrics metalWorldMetrics = runMetal
            ? verifyMetalWorldFunctionBasedDynamics(model)
            : MetalWorldFunctionBasedMetrics{};
        const MetalWorldFunctionBasedContactMetrics
            metalWorldContactMetrics = runMetal
                ? verifyMetalWorldFunctionBasedContact(
                    model, millardPath != nullptr ? &millardProgram : nullptr
                )
                : MetalWorldFunctionBasedContactMetrics{};
        const MetalWorldFunctionBasedContactMetrics
            fullyExcitedMetalWorldContactMetrics = runMetal &&
                millardPath != nullptr
                ? verifyMetalWorldFunctionBasedContact(
                    model,
                    &millardProgram,
                    fullyExcitedMillardControls
                )
                : MetalWorldFunctionBasedContactMetrics{};
        const MetalWorldFunctionBasedContactMetrics
            taskDrivenMetalWorldContactMetrics = runMetal &&
                millardPath != nullptr
                ? verifyMetalWorldFunctionBasedContact(
                    model,
                    &millardProgram,
                    {},
                    true
                )
                : MetalWorldFunctionBasedContactMetrics{};
        if (runMetal && millardPath != nullptr) {
            require(
                fullyExcitedMetalWorldContactMetrics.millardGeneralizedForceL1 >
                    metalWorldContactMetrics.millardGeneralizedForceL1 * 1.05,
                "fully excited source Millard contact probe did not exceed default activation force"
            );
            require(
                taskDrivenMetalWorldContactMetrics
                    .millardGeneralizedForceL1 >
                    metalWorldContactMetrics.millardGeneralizedForceL1 *
                        1.05,
                "native task source-Millard contact probe did not exceed default activation force"
            );
        }
        const MetalMillardReferenceMetrics metalMillardMetrics =
            runMetal && millardPath != nullptr
                ? verifyMetalMillardReference(model, millardPayload, millardMetrics)
                : MetalMillardReferenceMetrics{};
        const MetalWorldMillardActuationMetrics metalWorldMillardMetrics =
            runMetal && millardPath != nullptr
                ? verifyMetalWorldMillardActuation(
                    model, millardPayload, millardMetrics
                )
                : MetalWorldMillardActuationMetrics{};

        std::cout << std::scientific << std::setprecision(6)
                  << "numilab_human_core_reference=ok"
                  << " source_sha256=" << hexSha256(header.sourceSha256)
                  << " source_bodies=" << header.sourceBodyCount
                  << " engine_bodies=" << header.engineBodyCount
                  << " joints=" << header.jointCount
                  << " nq=" << header.nq
                  << " nv=" << header.nv
                  << " function_based_programs=" << header.functionProgramCount
                  << " mass_pivot=" << massDiagnostics.minimumCholeskyPivot
                  << " mass_symmetry_error=" << massSymmetryError
                  << " forward_inverse_error=" << forwardInverseError
                  << " kinetic_energy=" << invariants.kineticEnergy
                  << " total_energy=" << invariants.totalEnergy
                  << (runMetal
                          ? " metal_function_based_operator=ok"
                          : "")
                  << (runMetal
                          ? " metal_function_based_dynamics=ok"
                          : "")
                  << (runMetal
                          ? " metal_function_based_streamed_contact=ok"
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_function_based_millard_streamed_contact=ok"
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_function_based_millard_excitation_response=ok"
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_function_based_millard_native_task_bridge=ok"
                          : "")
                  << (runMetal
                          ? " metal_device=" + metalMetrics.deviceName
                          : "")
                  << (runMetal
                          ? " metal_world_device=" +
                                metalWorldMetrics.deviceName
                          : "")
                  << (runMetal
                          ? " metal_world_successful_steps=" +
                                std::to_string(
                                    metalWorldMetrics.successfulStepCount
                                )
                          : "")
                  << (runMetal
                          ? " metal_world_contact_device=" +
                                metalWorldContactMetrics.deviceName
                          : "")
                  << (runMetal
                          ? " metal_world_contact_successful_steps=" +
                                std::to_string(
                                    metalWorldContactMetrics.
                                        successfulStepCount
                                )
                          : "")
                  << (runMetal
                          ? " metal_world_contact_active_contacts=" +
                                std::to_string(
                                    metalWorldContactMetrics.
                                        maximumActiveContacts
                                )
                          : "")
                  << (runMetal
                          ? " metal_world_contact_constraints=" +
                                std::to_string(
                                    metalWorldContactMetrics.
                                        maximumConstraints
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_contact_millard_force_l1=" +
                                std::to_string(
                                    metalWorldContactMetrics.
                                        millardGeneralizedForceL1
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_contact_fully_excited_millard_force_l1=" +
                                std::to_string(
                                    fullyExcitedMetalWorldContactMetrics.
                                        millardGeneralizedForceL1
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_contact_millard_activation_error=" +
                                std::to_string(
                                    fullyExcitedMetalWorldContactMetrics.
                                        maximumMillardActivationError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_task_millard_force_l1=" +
                                std::to_string(
                                    taskDrivenMetalWorldContactMetrics.
                                        millardGeneralizedForceL1
                                )
                          : "")
                  << (runMetal
                          ? " metal_world_acceleration_error=" +
                                std::to_string(
                                    metalWorldMetrics.
                                        maximumAccelerationError
                                )
                          : "")
                  << (runMetal
                          ? " metal_world_velocity_error=" +
                                std::to_string(
                                    metalWorldMetrics.
                                        maximumVelocityError
                                )
                          : "")
                  << (runMetal
                          ? " metal_world_configuration_error=" +
                                std::to_string(
                                    metalWorldMetrics.
                                        maximumConfigurationError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_millard_actuation=ok"
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_millard_muscles=" +
                                std::to_string(
                                    metalWorldMillardMetrics.muscleCount
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_millard_force_l1=" +
                                std::to_string(
                                    metalWorldMillardMetrics.generalizedForceL1
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_millard_force_l1_relative_error=" +
                                std::to_string(
                                    metalWorldMillardMetrics.
                                        generalizedForceL1RelativeError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_millard_acceleration_error=" +
                                std::to_string(
                                    metalWorldMillardMetrics.
                                        maximumAccelerationError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_millard_acceleration_relative_error=" +
                                std::to_string(
                                    metalWorldMillardMetrics.
                                        maximumAccelerationRelativeError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_millard_acceleration_l1_relative_error=" +
                                std::to_string(
                                    metalWorldMillardMetrics.
                                        accelerationL1RelativeError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_world_millard_velocity_delta=" +
                                std::to_string(
                                    metalWorldMillardMetrics.
                                        maximumVelocityDeltaFromPassive
                                )
                          : "")
                  << (runMetal
                          ? " metal_body_position_error=" +
                                std::to_string(metalMetrics.bodyPositionError)
                          : "")
                  << (runMetal
                          ? " metal_body_orientation_error=" +
                                std::to_string(metalMetrics.bodyOrientationError)
                          : "")
                  << (runMetal
                          ? " metal_point_position_error=" +
                                std::to_string(metalMetrics.pointPositionError)
                          : "")
                  << (runMetal
                          ? " metal_point_jacobian_error=" +
                                std::to_string(metalMetrics.pointJacobianError)
                          : "")
                  << (runMetal
                          ? " metal_mass_error=" +
                                std::to_string(metalMetrics.massError)
                          : "")
                  << (runMetal
                          ? " metal_mass_scaled_error=" +
                                std::to_string(metalMetrics.massScaledError)
                          : "")
                  << (millardPath != nullptr
                          ? " millard_source_reference=ok"
                          : "")
                  << (millardPath != nullptr
                          ? " millard_muscles=" +
                                std::to_string(millardMetrics.muscleCount)
                          : "")
                  << (millardPath != nullptr
                          ? " millard_path_points=" +
                                std::to_string(millardMetrics.pathPointCount)
                          : "")
                  << (millardPath != nullptr
                          ? " millard_source_wraps=" +
                                std::to_string(millardMetrics.sourceWrapCount)
                          : "")
                  << (millardPath != nullptr
                          ? " millard_applied_cylinder_wraps=" +
                                std::to_string(
                                    millardMetrics.appliedCylinderWrapCount
                                )
                          : "")
                  << (millardPath != nullptr
                          ? " millard_ignore_tendon_compliance=" +
                                std::to_string(
                                    millardMetrics.ignoredTendonComplianceCount
                                )
                          : "")
                  << (millardPath != nullptr
                          ? " millard_path_length=" +
                                std::to_string(millardMetrics.totalPathLength)
                          : "")
                  << (millardPath != nullptr
                          ? " millard_generalized_force_l1=" +
                                std::to_string(millardMetrics.generalizedForceL1)
                          : "")
                  << (millardPath != nullptr
                          ? " millard_equilibrium_residual=" +
                                std::to_string(
                                    millardMetrics.maximumEquilibriumResidual
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_millard_reference=ok"
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_millard_applied_cylinder_wraps=" +
                                std::to_string(
                                    metalMillardMetrics.appliedCylinderWrapCount
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_millard_path_relative_error=" +
                                std::to_string(
                                    metalMillardMetrics.
                                        maximumPathLengthRelativeError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_millard_tendon_relative_error=" +
                                std::to_string(
                                    metalMillardMetrics.
                                        maximumTendonForceRelativeError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_millard_force_l1_relative_error=" +
                                std::to_string(
                                    metalMillardMetrics.
                                        maximumGeneralizedForceL1RelativeError
                                )
                          : "")
                  << (runMetal && millardPath != nullptr
                          ? " metal_millard_equilibrium_residual=" +
                                std::to_string(
                                    metalMillardMetrics.
                                        maximumEquilibriumResidual
                                )
                          : "")
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numilab_human_core_reference=failed " << error.what() << '\n';
        return 1;
    }
}
