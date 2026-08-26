#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
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
constexpr std::uint32_t kMillardPayloadAbi = 2u;

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
static_assert(sizeof(MillardWrapRecord) == 36u);

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
        for (const double effort : generalizedForce) {
            metrics.generalizedForceL1 += std::abs(effort);
        }
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
                          ? " metal_device=" + metalMetrics.deviceName
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
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "numilab_human_core_reference=failed " << error.what() << '\n';
        return 1;
    }
}
