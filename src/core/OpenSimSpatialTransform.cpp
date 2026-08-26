#include "metalrobo/OpenSimSpatialTransform.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace metalrobo {
namespace {

using Vector = std::array<double, 3>;
using Matrix = std::array<double, 9>;

constexpr double kAxisTolerance = 1.0e-10;

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const Vector& value) {
    return std::all_of(value.begin(), value.end(), [](const double entry) {
        return finite(entry);
    });
}

double dot(const Vector& left, const Vector& right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

Vector cross(const Vector& left, const Vector& right) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

Vector add(const Vector& left, const Vector& right) {
    return {
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    };
}

Vector scaled(const Vector& value, const double scale) {
    return {value[0] * scale, value[1] * scale, value[2] * scale};
}

Vector normalized(const Vector& value) {
    const double magnitudeSquared = dot(value, value);
    if (!finite(magnitudeSquared) || !(magnitudeSquared > kAxisTolerance)) {
        return {};
    }
    return scaled(value, 1.0 / std::sqrt(magnitudeSquared));
}

Matrix multiply(const Matrix& left, const Matrix& right) {
    Matrix result{};
    for (std::size_t row = 0u; row < 3u; ++row) {
        for (std::size_t column = 0u; column < 3u; ++column) {
            for (std::size_t cursor = 0u; cursor < 3u; ++cursor) {
                result[row * 3u + column] +=
                    left[row * 3u + cursor] * right[cursor * 3u + column];
            }
        }
    }
    return result;
}

Vector apply(const Matrix& matrix, const Vector& value) {
    return {
        matrix[0] * value[0] + matrix[1] * value[1] + matrix[2] * value[2],
        matrix[3] * value[0] + matrix[4] * value[1] + matrix[5] * value[2],
        matrix[6] * value[0] + matrix[7] * value[1] + matrix[8] * value[2],
    };
}

Matrix axisAngle(const Vector& axis, const double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double oneMinusCosine = 1.0 - cosine;
    const double x = axis[0];
    const double y = axis[1];
    const double z = axis[2];
    return {
        cosine + x * x * oneMinusCosine,
        x * y * oneMinusCosine - z * sine,
        x * z * oneMinusCosine + y * sine,
        y * x * oneMinusCosine + z * sine,
        cosine + y * y * oneMinusCosine,
        y * z * oneMinusCosine - x * sine,
        z * x * oneMinusCosine - y * sine,
        z * y * oneMinusCosine + x * sine,
        cosine + z * z * oneMinusCosine,
    };
}

bool finite(const Matrix& value) {
    return std::all_of(value.begin(), value.end(), [](const double entry) {
        return finite(entry);
    });
}

bool independent(const Vector& left, const Vector& right) {
    return dot(cross(left, right), cross(left, right)) > kAxisTolerance;
}

float packedScalar(const mr_float4* blocks, const std::size_t index) {
    const mr_float4& block = blocks[index / 4u];
    switch (index % 4u) {
    case 0u:
        return block.x;
    case 1u:
        return block.y;
    case 2u:
        return block.z;
    default:
        return block.w;
    }
}

bool finiteBlocks(const mr_float4* blocks, const std::size_t capacity) {
    for (std::size_t index = 0u; index < capacity; ++index) {
        if (!finite(static_cast<double>(packedScalar(blocks, index)))) {
            return false;
        }
    }
    return true;
}

std::vector<double> unpackBlocks(
    const mr_float4* blocks,
    const std::size_t count
) {
    std::vector<double> values;
    values.reserve(count);
    for (std::size_t index = 0u; index < count; ++index) {
        values.push_back(static_cast<double>(packedScalar(blocks, index)));
    }
    return values;
}

bool strictlyIncreasing(const std::vector<double>& values) {
    return std::adjacent_find(
        values.begin(), values.end(), [](const double left, const double right) {
            return right <= left;
        }
    ) == values.end();
}

OpenSimSpatialTransformEvaluation failure(
    const OpenSimSpatialTransformStatus status
) {
    return {.status = status};
}

} // namespace

OpenSimSpatialTransformCompilation compileOpenSimSpatialTransform(
    const OpenSimSpatialTransformDefinition& definition
) {
    OpenSimSpatialTransformCompilation compiled{};
    if (definition.coordinateCount == 0u || definition.coordinateCount > 6u) {
        compiled.status = OpenSimSpatialTransformStatus::invalidDefinition;
        return compiled;
    }
    compiled.transform.coordinateCount = definition.coordinateCount;
    for (std::size_t index = 0u; index < kOpenSimSpatialAxisCount; ++index) {
        const OpenSimSpatialAxisDefinition& source = definition.axes[index];
        CompiledOpenSimSpatialAxis& target = compiled.transform.axes[index];
        target.axis = normalized(source.axis);
        if (!finite(source.axis) || !finite(target.axis) ||
            !(dot(target.axis, target.axis) > kAxisTolerance)) {
            compiled.status = OpenSimSpatialTransformStatus::invalidDefinition;
            return compiled;
        }
        const bool constant =
            source.function.kind == OpenSimFunctionKind::constant;
        if ((source.coordinateIndex == kOpenSimNoCoordinate) != constant ||
            (!constant && source.coordinateIndex >= definition.coordinateCount)) {
            compiled.status = OpenSimSpatialTransformStatus::invalidDefinition;
            return compiled;
        }
        const OpenSimFunctionCompilation function =
            compileOpenSimFunction(source.function);
        if (!function.succeeded()) {
            compiled.status = OpenSimSpatialTransformStatus::invalidDefinition;
            return compiled;
        }
        target.coordinateIndex = source.coordinateIndex;
        target.function = function.function;
    }
    for (std::size_t group = 0u; group < 2u; ++group) {
        const std::size_t first = group * 3u;
        if (!independent(compiled.transform.axes[first].axis,
                         compiled.transform.axes[first + 1u].axis) ||
            !independent(compiled.transform.axes[first].axis,
                         compiled.transform.axes[first + 2u].axis) ||
            !independent(compiled.transform.axes[first + 1u].axis,
                         compiled.transform.axes[first + 2u].axis)) {
            compiled.status = OpenSimSpatialTransformStatus::invalidDefinition;
            return compiled;
        }
    }
    return compiled;
}

OpenSimSpatialTransformEvaluation evaluateOpenSimSpatialTransform(
    const CompiledOpenSimSpatialTransform& transform,
    const std::vector<double>& coordinates,
    const std::vector<double>& coordinateVelocities
) {
    if (transform.coordinateCount == 0u || transform.coordinateCount > 6u ||
        coordinates.size() != transform.coordinateCount ||
        coordinateVelocities.size() != transform.coordinateCount) {
        return failure(OpenSimSpatialTransformStatus::invalidCoordinates);
    }
    if (!std::all_of(
            coordinates.begin(),
            coordinates.end(),
            [](const double value) { return finite(value); }
        ) ||
        !std::all_of(
            coordinateVelocities.begin(),
            coordinateVelocities.end(),
            [](const double value) { return finite(value); }
        )) {
        return failure(OpenSimSpatialTransformStatus::nonfiniteInput);
    }

    std::array<OpenSimFunctionEvaluation, kOpenSimSpatialAxisCount> axes{};
    for (std::size_t index = 0u; index < kOpenSimSpatialAxisCount; ++index) {
        const CompiledOpenSimSpatialAxis& axis = transform.axes[index];
        const double argument = axis.coordinateIndex == kOpenSimNoCoordinate
            ? 0.0
            : coordinates[axis.coordinateIndex];
        axes[index] = evaluateOpenSimFunction(axis.function, argument);
        if (!axes[index].succeeded()) {
            return failure(
                axes[index].status == OpenSimFunctionStatus::nonfiniteArgument
                    ? OpenSimSpatialTransformStatus::nonfiniteInput
                    : OpenSimSpatialTransformStatus::invalidDefinition
            );
        }
    }

    const Matrix rotation0 = axisAngle(transform.axes[0].axis, axes[0].value);
    const Matrix rotation01 = multiply(
        rotation0,
        axisAngle(transform.axes[1].axis, axes[1].value)
    );
    OpenSimSpatialTransformEvaluation evaluation{};
    evaluation.rotation = multiply(
        rotation01,
        axisAngle(transform.axes[2].axis, axes[2].value)
    );
    for (std::size_t index = 0u; index < 3u; ++index) {
        evaluation.translation = add(
            evaluation.translation,
            scaled(transform.axes[index + 3u].axis, axes[index + 3u].value)
        );
    }

    const Vector angularAxis0 = transform.axes[0].axis;
    const Vector angularAxis1 = apply(rotation0, transform.axes[1].axis);
    const Vector angularAxis2 = apply(rotation01, transform.axes[2].axis);
    const std::array<Vector, 3u> angularAxes{
        angularAxis0,
        angularAxis1,
        angularAxis2,
    };

    const double thetaDot0 =
        transform.axes[0].coordinateIndex == kOpenSimNoCoordinate
            ? 0.0
            : axes[0].derivative *
                coordinateVelocities[transform.axes[0].coordinateIndex];
    const double thetaDot1 =
        transform.axes[1].coordinateIndex == kOpenSimNoCoordinate
            ? 0.0
            : axes[1].derivative *
                coordinateVelocities[transform.axes[1].coordinateIndex];
    const std::array<Vector, 3u> angularAxisDots{
        Vector{},
        cross(scaled(angularAxis0, thetaDot0), angularAxis1),
        cross(
            add(
                scaled(angularAxis0, thetaDot0),
                scaled(angularAxis1, thetaDot1)
            ),
            angularAxis2
        ),
    };

    evaluation.motionSubspace.resize(transform.coordinateCount);
    evaluation.motionSubspaceDot.resize(transform.coordinateCount);
    for (std::size_t index = 0u; index < kOpenSimSpatialAxisCount; ++index) {
        const CompiledOpenSimSpatialAxis& axis = transform.axes[index];
        if (axis.coordinateIndex == kOpenSimNoCoordinate) {
            continue;
        }
        const std::size_t coordinate = axis.coordinateIndex;
        const double derivative = axes[index].derivative;
        const double derivativeDot = axes[index].secondDerivative *
            coordinateVelocities[coordinate];
        if (index < 3u) {
            evaluation.motionSubspace[coordinate].angular = add(
                evaluation.motionSubspace[coordinate].angular,
                scaled(angularAxes[index], derivative)
            );
            evaluation.motionSubspaceDot[coordinate].angular = add(
                evaluation.motionSubspaceDot[coordinate].angular,
                add(
                    scaled(angularAxisDots[index], derivative),
                    scaled(angularAxes[index], derivativeDot)
                )
            );
        } else {
            evaluation.motionSubspace[coordinate].linear = add(
                evaluation.motionSubspace[coordinate].linear,
                scaled(axis.axis, derivative)
            );
            evaluation.motionSubspaceDot[coordinate].linear = add(
                evaluation.motionSubspaceDot[coordinate].linear,
                scaled(axis.axis, derivativeDot)
            );
        }
    }
    if (!finite(evaluation.rotation) || !finite(evaluation.translation)) {
        return failure(OpenSimSpatialTransformStatus::nonfiniteResult);
    }
    for (const OpenSimSpatialVector& motion : evaluation.motionSubspace) {
        if (!finite(motion.angular) || !finite(motion.linear)) {
            return failure(OpenSimSpatialTransformStatus::nonfiniteResult);
        }
    }
    for (const OpenSimSpatialVector& motion : evaluation.motionSubspaceDot) {
        if (!finite(motion.angular) || !finite(motion.linear)) {
            return failure(OpenSimSpatialTransformStatus::nonfiniteResult);
        }
    }
    return evaluation;
}

OpenSimSpatialForceProjection projectOpenSimSpatialWrench(
    const OpenSimSpatialTransformEvaluation& evaluation,
    const std::vector<double>& coordinateVelocities,
    const OpenSimSpatialWrench& wrench
) {
    OpenSimSpatialForceProjection projection{};
    if (!evaluation.succeeded()) {
        projection.status = evaluation.status;
        return projection;
    }
    if (evaluation.motionSubspace.size() != evaluation.motionSubspaceDot.size() ||
        coordinateVelocities.size() != evaluation.motionSubspace.size()) {
        projection.status = OpenSimSpatialTransformStatus::invalidCoordinates;
        return projection;
    }
    if (!finite(wrench.angular) || !finite(wrench.linear) ||
        !std::all_of(
            coordinateVelocities.begin(),
            coordinateVelocities.end(),
            [](const double value) { return finite(value); }
        )) {
        projection.status = OpenSimSpatialTransformStatus::nonfiniteInput;
        return projection;
    }
    projection.generalizedForces.resize(evaluation.motionSubspace.size());
    for (std::size_t coordinate = 0u;
         coordinate < evaluation.motionSubspace.size();
         ++coordinate) {
        const OpenSimSpatialVector& motion = evaluation.motionSubspace[coordinate];
        const OpenSimSpatialVector& motionDot = evaluation.motionSubspaceDot[coordinate];
        projection.generalizedForces[coordinate] =
            dot(motion.angular, wrench.angular) + dot(motion.linear, wrench.linear);
        projection.spatialBiasAcceleration.angular = add(
            projection.spatialBiasAcceleration.angular,
            scaled(motionDot.angular, coordinateVelocities[coordinate])
        );
        projection.spatialBiasAcceleration.linear = add(
            projection.spatialBiasAcceleration.linear,
            scaled(motionDot.linear, coordinateVelocities[coordinate])
        );
    }
    if (!finite(projection.spatialBiasAcceleration.angular) ||
        !finite(projection.spatialBiasAcceleration.linear) ||
        !std::all_of(
            projection.generalizedForces.begin(),
            projection.generalizedForces.end(),
            [](const double value) { return finite(value); }
        )) {
        projection.status = OpenSimSpatialTransformStatus::nonfiniteResult;
    }
    return projection;
}

OpenSimSpatialTransformStatus packOpenSimSpatialTransformGPU(
    const CompiledOpenSimSpatialTransform& transform,
    MROpenSimSpatialTransformGPU& program
) {
    if (transform.coordinateCount == 0u || transform.coordinateCount > 6u) {
        return OpenSimSpatialTransformStatus::invalidDefinition;
    }
    MROpenSimSpatialTransformGPU staged{};
    staged.abiVersion = MR_OPENSIM_SPATIAL_TRANSFORM_GPU_ABI_VERSION;
    staged.coordinateCount = transform.coordinateCount;
    for (std::size_t index = 0u; index < kOpenSimSpatialAxisCount; ++index) {
        const CompiledOpenSimSpatialAxis& source = transform.axes[index];
        MROpenSimFunctionGPU& target = staged.axes[index];
        if (!finite(source.axis) ||
            (source.coordinateIndex != kOpenSimNoCoordinate &&
             source.coordinateIndex >= transform.coordinateCount)) {
            return OpenSimSpatialTransformStatus::invalidDefinition;
        }
        const auto pack = [](const std::vector<double>& values,
                             mr_float4* blocks,
                             const std::size_t capacity) {
            if (values.size() > capacity ||
                !std::all_of(values.begin(), values.end(), [](const double value) {
                    return finite(value) &&
                        finite(static_cast<double>(static_cast<float>(value)));
                })) {
                return false;
            }
            for (std::size_t value = 0u; value < values.size(); ++value) {
                mr_float4& block = blocks[value / 4u];
                const float converted = static_cast<float>(values[value]);
                switch (value % 4u) {
                case 0u:
                    block.x = converted;
                    break;
                case 1u:
                    block.y = converted;
                    break;
                case 2u:
                    block.z = converted;
                    break;
                default:
                    block.w = converted;
                    break;
                }
            }
            return true;
        };
        target.kind = static_cast<mr_u32>(source.function.kind);
        target.coordinateIndex = source.coordinateIndex;
        target.coefficientCount =
            static_cast<mr_u32>(source.function.coefficients.size());
        target.knotCount = static_cast<mr_u32>(source.function.abscissae.size());
        target.axis = {
            static_cast<float>(source.axis[0]),
            static_cast<float>(source.axis[1]),
            static_cast<float>(source.axis[2]),
            0.0f,
        };
        if (!finite(static_cast<double>(target.axis.x)) ||
            !finite(static_cast<double>(target.axis.y)) ||
            !finite(static_cast<double>(target.axis.z)) ||
            !pack(
                source.function.coefficients,
                target.coefficients,
                MR_OPENSIM_SPATIAL_MAX_COEFFICIENTS
            ) ||
            !pack(
                source.function.abscissae,
                target.abscissae,
                MR_OPENSIM_SPATIAL_MAX_KNOTS
            ) ||
            !pack(
                source.function.ordinates,
                target.ordinates,
                MR_OPENSIM_SPATIAL_MAX_KNOTS
            ) ||
            !pack(
                source.function.splineSlope,
                target.splineSlope,
                MR_OPENSIM_SPATIAL_MAX_KNOTS
            ) ||
            !pack(
                source.function.splineQuadratic,
                target.splineQuadratic,
                MR_OPENSIM_SPATIAL_MAX_KNOTS
            ) ||
            !pack(
                source.function.splineCubic,
                target.splineCubic,
                MR_OPENSIM_SPATIAL_MAX_KNOTS
            )) {
            return OpenSimSpatialTransformStatus::invalidDefinition;
        }
    }
    program = staged;
    return OpenSimSpatialTransformStatus::success;
}

OpenSimSpatialTransformCompilation unpackOpenSimSpatialTransformGPU(
    const MROpenSimSpatialTransformGPU& program
) {
    OpenSimSpatialTransformCompilation decoded{};
    if (program.abiVersion != MR_OPENSIM_SPATIAL_TRANSFORM_GPU_ABI_VERSION ||
        program.coordinateCount == 0u ||
        program.coordinateCount > MR_OPENSIM_SPATIAL_MAX_COORDINATES ||
        program.reserved0 != 0u || program.reserved1 != 0u) {
        decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
        return decoded;
    }
    CompiledOpenSimSpatialTransform& transform = decoded.transform;
    transform.coordinateCount = program.coordinateCount;
    for (std::size_t index = 0u; index < kOpenSimSpatialAxisCount; ++index) {
        const MROpenSimFunctionGPU& source = program.axes[index];
        if (source.kind > MR_OPENSIM_FUNCTION_SIMM_SPLINE ||
            source.axis.w != 0.0f ||
            !finite(static_cast<double>(source.axis.x)) ||
            !finite(static_cast<double>(source.axis.y)) ||
            !finite(static_cast<double>(source.axis.z)) ||
            !finiteBlocks(
                source.coefficients, MR_OPENSIM_SPATIAL_MAX_COEFFICIENTS
            ) ||
            !finiteBlocks(source.abscissae, MR_OPENSIM_SPATIAL_MAX_KNOTS) ||
            !finiteBlocks(source.ordinates, MR_OPENSIM_SPATIAL_MAX_KNOTS) ||
            !finiteBlocks(source.splineSlope, MR_OPENSIM_SPATIAL_MAX_KNOTS) ||
            !finiteBlocks(source.splineQuadratic, MR_OPENSIM_SPATIAL_MAX_KNOTS) ||
            !finiteBlocks(source.splineCubic, MR_OPENSIM_SPATIAL_MAX_KNOTS)) {
            decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
            return decoded;
        }
        CompiledOpenSimSpatialAxis& target = transform.axes[index];
        target.axis = {
            static_cast<double>(source.axis.x),
            static_cast<double>(source.axis.y),
            static_cast<double>(source.axis.z),
        };
        if (!(dot(target.axis, target.axis) > kAxisTolerance)) {
            decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
            return decoded;
        }
        target.coordinateIndex = source.coordinateIndex;
        target.function.kind = static_cast<OpenSimFunctionKind>(source.kind);
        const bool constant = target.function.kind == OpenSimFunctionKind::constant;
        if ((constant && target.coordinateIndex != kOpenSimNoCoordinate) ||
            (!constant &&
             (target.coordinateIndex == kOpenSimNoCoordinate ||
              target.coordinateIndex >= transform.coordinateCount))) {
            decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
            return decoded;
        }
        switch (target.function.kind) {
        case OpenSimFunctionKind::constant:
            if (source.coefficientCount != 1u || source.knotCount != 0u) {
                decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
                return decoded;
            }
            target.function.coefficients = unpackBlocks(source.coefficients, 1u);
            break;
        case OpenSimFunctionKind::linear:
            if (source.coefficientCount != 2u || source.knotCount != 0u) {
                decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
                return decoded;
            }
            target.function.coefficients = unpackBlocks(source.coefficients, 2u);
            break;
        case OpenSimFunctionKind::polynomial:
            if (source.coefficientCount == 0u ||
                source.coefficientCount > MR_OPENSIM_SPATIAL_MAX_COEFFICIENTS ||
                source.knotCount != 0u) {
                decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
                return decoded;
            }
            target.function.coefficients = unpackBlocks(
                source.coefficients, source.coefficientCount
            );
            break;
        case OpenSimFunctionKind::simmSpline:
            if (source.coefficientCount != 0u || source.knotCount < 2u ||
                source.knotCount > MR_OPENSIM_SPATIAL_MAX_KNOTS) {
                decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
                return decoded;
            }
            target.function.abscissae = unpackBlocks(source.abscissae, source.knotCount);
            target.function.ordinates = unpackBlocks(source.ordinates, source.knotCount);
            target.function.splineSlope = unpackBlocks(source.splineSlope, source.knotCount);
            target.function.splineQuadratic = unpackBlocks(
                source.splineQuadratic, source.knotCount
            );
            target.function.splineCubic = unpackBlocks(source.splineCubic, source.knotCount);
            if (!strictlyIncreasing(target.function.abscissae)) {
                decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
                return decoded;
            }
            break;
        }
    }
    for (std::size_t group = 0u; group < 2u; ++group) {
        const std::size_t first = group * 3u;
        if (!independent(transform.axes[first].axis, transform.axes[first + 1u].axis) ||
            !independent(transform.axes[first].axis, transform.axes[first + 2u].axis) ||
            !independent(
                transform.axes[first + 1u].axis, transform.axes[first + 2u].axis
            )) {
            decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
            return decoded;
        }
    }
    MROpenSimSpatialTransformGPU repacked{};
    if (packOpenSimSpatialTransformGPU(transform, repacked) !=
            OpenSimSpatialTransformStatus::success ||
        std::memcmp(&program, &repacked, sizeof(program)) != 0) {
        decoded.status = OpenSimSpatialTransformStatus::invalidDefinition;
    }
    return decoded;
}

const char* openSimSpatialTransformStatusName(
    const OpenSimSpatialTransformStatus status
) noexcept {
    switch (status) {
    case OpenSimSpatialTransformStatus::success:
        return "success";
    case OpenSimSpatialTransformStatus::invalidDefinition:
        return "invalid_definition";
    case OpenSimSpatialTransformStatus::invalidCoordinates:
        return "invalid_coordinates";
    case OpenSimSpatialTransformStatus::nonfiniteInput:
        return "nonfinite_input";
    case OpenSimSpatialTransformStatus::nonfiniteResult:
        return "nonfinite_result";
    }
    return "unknown";
}

} // namespace metalrobo
