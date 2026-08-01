#pragma once

#include "metalrobo/engine_types.h"

#include <cmath>

namespace metalrobo::semantic_transform {

inline bool finite(const mr_float4 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

inline bool normalizeQuaternion(
    const mr_float4 value,
    mr_float4& normalized
) noexcept {
    if (!finite(value)) {
        return false;
    }
    const double normSquared =
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z +
        static_cast<double>(value.w) * value.w;
    if (!(normSquared > 1.0e-12) ||
        !std::isfinite(normSquared)) {
        return false;
    }
    const float inverseNorm = static_cast<float>(
        1.0 / std::sqrt(normSquared)
    );
    normalized = {
        value.x * inverseNorm,
        value.y * inverseNorm,
        value.z * inverseNorm,
        value.w * inverseNorm,
    };
    return finite(normalized);
}

inline mr_float4 quaternionProduct(
    const mr_float4 left,
    const mr_float4 right
) noexcept {
    return {
        left.w * right.x + left.x * right.w +
            left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z +
            left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y -
            left.y * right.x + left.z * right.w,
        left.w * right.w - left.x * right.x -
            left.y * right.y - left.z * right.z,
    };
}

inline mr_float4 rotateVector(
    const mr_float4 unitOrientation,
    const mr_float4 vector
) noexcept {
    const mr_float4 qv{
        unitOrientation.y * vector.z -
            unitOrientation.z * vector.y,
        unitOrientation.z * vector.x -
            unitOrientation.x * vector.z,
        unitOrientation.x * vector.y -
            unitOrientation.y * vector.x,
        0.0f,
    };
    const mr_float4 qqv{
        unitOrientation.y * qv.z -
            unitOrientation.z * qv.y,
        unitOrientation.z * qv.x -
            unitOrientation.x * qv.z,
        unitOrientation.x * qv.y -
            unitOrientation.y * qv.x,
        0.0f,
    };
    return {
        vector.x + 2.0f *
            (unitOrientation.w * qv.x + qqv.x),
        vector.y + 2.0f *
            (unitOrientation.w * qv.y + qqv.y),
        vector.z + 2.0f *
            (unitOrientation.w * qv.z + qqv.z),
        0.0f,
    };
}

inline bool compose(
    const mr_float4 parentPosition,
    const mr_float4 parentOrientation,
    const mr_float4 localPosition,
    const mr_float4 localOrientation,
    mr_float4& composedPosition,
    mr_float4& composedOrientation
) noexcept {
    mr_float4 parentUnit{};
    mr_float4 localUnit{};
    if (!finite(parentPosition) || !finite(localPosition) ||
        !normalizeQuaternion(parentOrientation, parentUnit) ||
        !normalizeQuaternion(localOrientation, localUnit)) {
        return false;
    }
    const mr_float4 offset = rotateVector(
        parentUnit,
        localPosition
    );
    composedPosition = {
        parentPosition.x + offset.x,
        parentPosition.y + offset.y,
        parentPosition.z + offset.z,
        0.0f,
    };
    return normalizeQuaternion(
        quaternionProduct(parentUnit, localUnit),
        composedOrientation
    );
}

} // namespace metalrobo::semantic_transform
