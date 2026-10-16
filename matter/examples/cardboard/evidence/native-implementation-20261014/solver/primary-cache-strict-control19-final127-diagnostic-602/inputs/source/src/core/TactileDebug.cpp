#include "metalrobo/TactileDebug.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <string_view>

namespace metalrobo {
namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 add(const Vec3 left, const Vec3 right) {
    return {
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    };
}

Vec3 multiply(const Vec3 value, const double scale) {
    return {
        value.x * scale,
        value.y * scale,
        value.z * scale,
    };
}

Vec3 vector(const mr_float4 value) {
    return {value.x, value.y, value.z};
}

Vec3 rotate(const mr_float4 quaternion, const Vec3 value) {
    const Vec3 axis{
        quaternion.x,
        quaternion.y,
        quaternion.z,
    };
    const Vec3 twiceCross{
        2.0 * (axis.y * value.z - axis.z * value.y),
        2.0 * (axis.z * value.x - axis.x * value.z),
        2.0 * (axis.x * value.y - axis.y * value.x),
    };
    const Vec3 crossAgain{
        axis.y * twiceCross.z - axis.z * twiceCross.y,
        axis.z * twiceCross.x - axis.x * twiceCross.z,
        axis.x * twiceCross.y - axis.y * twiceCross.x,
    };
    return add(
        value,
        add(
            multiply(twiceCross, quaternion.w),
            crossAgain
        )
    );
}

Vec3 transform(
    const mr_float4 position,
    const mr_float4 orientation,
    const Vec3 local
) {
    return add(vector(position), rotate(orientation, local));
}

std::string safePrefix(std::string value) {
    for (char& character : value) {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '-' ||
            character == '_';
        if (!safe) {
            character = '_';
        }
    }
    return value.empty() ? "tactile" : value;
}

std::string escapedJSON(const std::string_view input) {
    std::string result;
    result.reserve(input.size() + 8u);
    constexpr char hexadecimal[] = "0123456789abcdef";
    for (const unsigned char value : input) {
        switch (value) {
        case '\\':
            result += "\\\\";
            break;
        case '"':
            result += "\\\"";
            break;
        case '\b':
            result += "\\b";
            break;
        case '\f':
            result += "\\f";
            break;
        case '\n':
            result += "\\n";
            break;
        case '\r':
            result += "\\r";
            break;
        case '\t':
            result += "\\t";
            break;
        default:
            if (value < 0x20u) {
                result += "\\u00";
                result += hexadecimal[value >> 4u];
                result += hexadecimal[value & 0x0fu];
            } else {
                result += static_cast<char>(value);
            }
            break;
        }
    }
    return result;
}

bool writeBigEndian16(
    std::ofstream& stream,
    const std::uint16_t value
) {
    const std::array bytes{
        static_cast<char>(value >> 8u),
        static_cast<char>(value & 0xffu),
    };
    stream.write(bytes.data(), bytes.size());
    return static_cast<bool>(stream);
}

} // namespace

TactileDebugExportResult exportTactileDebugFrame(
    const CookedTactileSystem& tactile,
    const TactileObservationBatch& observation,
    const TactileDebugExportConfig& config
) {
    TactileDebugExportResult result;
    const std::size_t denseCount =
        static_cast<std::size_t>(observation.environmentCount) *
        tactile.samples.size();
    const std::size_t summaryCount =
        static_cast<std::size_t>(observation.environmentCount) *
        tactile.sensors.size();
    if (config.directory.empty() ||
        config.environment >= observation.environmentCount ||
        config.sensor >= tactile.sensors.size() ||
        observation.sampleCount != tactile.samples.size() ||
        observation.sensorCount != tactile.sensors.size() ||
        observation.penetrationDepthMeters.size() != denseCount ||
        observation.validity.size() != denseCount ||
        observation.objectShapeIds.size() != denseCount ||
        observation.debugHits.size() != denseCount ||
        observation.summaries.size() != summaryCount ||
        !std::isfinite(config.forceVectorMetersPerNewton) ||
        config.forceVectorMetersPerNewton < 0.0f) {
        result.message =
            "tactile debug export extents or configuration are invalid";
        return result;
    }
    std::error_code error;
    std::filesystem::create_directories(config.directory, error);
    if (error) {
        result.message =
            "could not create tactile debug export directory";
        return result;
    }
    const std::string prefix = safePrefix(config.prefix);
    result.metricDepthCSV =
        config.directory / (prefix + "-depth-m.csv");
    result.depthPreviewPGM =
        config.directory / (prefix + "-depth-preview.pgm");
    result.validityPreviewPGM =
        config.directory / (prefix + "-validity-preview.pgm");
    result.geometryOBJ =
        config.directory / (prefix + "-geometry.obj");
    result.summaryJSON =
        config.directory / (prefix + "-summary.json");

    const MRTactileSensorGPU& sensor =
        tactile.sensors[config.sensor];
    const std::size_t denseBase =
        static_cast<std::size_t>(config.environment) *
            tactile.samples.size() +
        sensor.topology.z;
    const std::size_t summaryIndex =
        static_cast<std::size_t>(config.environment) *
            tactile.sensors.size() +
        config.sensor;
    const MRTactileSummaryGPU& summary =
        observation.summaries[summaryIndex];

    {
        std::ofstream stream(result.metricDepthCSV);
        stream << std::setprecision(9);
        for (std::uint32_t row = 0u;
             row < sensor.atlasAndTargets.y;
             ++row) {
            for (std::uint32_t column = 0u;
                 column < sensor.atlasAndTargets.x;
                 ++column) {
                if (column != 0u) {
                    stream << ',';
                }
                stream << observation.penetrationDepthMeters[
                    denseBase +
                    static_cast<std::size_t>(row) *
                        sensor.atlasAndTargets.x +
                    column
                ];
            }
            stream << '\n';
        }
        if (!stream) {
            result.message =
                "could not write metric tactile depth CSV";
            return result;
        }
    }
    {
        std::ofstream stream(
            result.depthPreviewPGM,
            std::ios::binary
        );
        stream
            << "P5\n"
            << sensor.atlasAndTargets.x << ' '
            << sensor.atlasAndTargets.y
            << "\n65535\n";
        for (std::uint32_t local = 0u;
             local < sensor.topology.w;
             ++local) {
            const float normalized = std::clamp(
                observation.penetrationDepthMeters[
                    denseBase + local
                ] / sensor.depth.x,
                0.0f,
                1.0f
            );
            if (!writeBigEndian16(
                    stream,
                    static_cast<std::uint16_t>(
                        std::lround(normalized * 65535.0f)
                    )
                )) {
                result.message =
                    "could not write tactile depth preview";
                return result;
            }
        }
    }
    {
        std::ofstream stream(
            result.validityPreviewPGM,
            std::ios::binary
        );
        stream
            << "P5\n"
            << sensor.atlasAndTargets.x << ' '
            << sensor.atlasAndTargets.y
            << "\n255\n";
        for (std::uint32_t local = 0u;
             local < sensor.topology.w;
             ++local) {
            const std::uint32_t flags =
                observation.validity[denseBase + local];
            const unsigned char value =
                (flags & MR_TACTILE_VALIDITY_SATURATED) != 0u
                ? 255u
                : (flags & MR_TACTILE_VALIDITY_CONTACT) != 0u
                    ? 192u
                    : (flags & MR_TACTILE_VALIDITY_SAMPLE) != 0u
                        ? 64u
                        : 0u;
            stream.write(
                reinterpret_cast<const char*>(&value),
                1
            );
        }
        if (!stream) {
            result.message =
                "could not write tactile validity preview";
            return result;
        }
    }
    {
        std::ofstream stream(result.geometryOBJ);
        stream << std::setprecision(9);
        std::uint32_t vertex = 1u;
        for (std::uint32_t local = 0u;
             local < sensor.topology.w;
             ++local) {
            const MRTactileSampleGPU& sample =
                tactile.samples[sensor.topology.z + local];
            const Vec3 rest = transform(
                summary.posePositionAndTimestamp,
                summary.poseOrientation,
                vector(sample.localPositionAndArea)
            );
            const Vec3 inward = multiply(
                rotate(
                    summary.poseOrientation,
                    vector(sample.localNormalAndMaximumDepth)
                ),
                -sample.localNormalAndMaximumDepth.w
            );
            const Vec3 volumeEnd = add(rest, inward);
            stream << "v " << rest.x << ' ' << rest.y << ' '
                   << rest.z << '\n';
            stream << "v " << volumeEnd.x << ' ' << volumeEnd.y
                   << ' ' << volumeEnd.z << '\n';
            stream << "l " << vertex << ' ' << vertex + 1u
                   << '\n';
            vertex += 2u;
            if ((observation.validity[denseBase + local] &
                 MR_TACTILE_VALIDITY_CONTACT) != 0u) {
                const mr_float4 hit =
                    observation.debugHits[denseBase + local].
                        worldPointAndDepth;
                stream << "v " << hit.x << ' ' << hit.y << ' '
                       << hit.z << '\n';
                stream << "p " << vertex << '\n';
                ++vertex;
            }
        }
        const Vec3 origin =
            vector(summary.posePositionAndTimestamp);
        const Vec3 forceEnd = add(
            origin,
            multiply(
                vector(summary.netForceAndContactArea),
                config.forceVectorMetersPerNewton
            )
        );
        stream << "v " << origin.x << ' ' << origin.y << ' '
               << origin.z << '\n';
        stream << "v " << forceEnd.x << ' ' << forceEnd.y << ' '
               << forceEnd.z << '\n';
        stream << "l " << vertex << ' ' << vertex + 1u << '\n';
        vertex += 2u;
        const Vec3 centroid =
            vector(summary.centroidWorldAndActiveCount);
        const Vec3 centerOfPressure =
            vector(
                summary.centerOfPressureWorldAndContactCount
            );
        stream << "v " << centroid.x << ' ' << centroid.y << ' '
               << centroid.z << "\np " << vertex << '\n';
        ++vertex;
        stream << "v " << centerOfPressure.x << ' '
               << centerOfPressure.y << ' '
               << centerOfPressure.z << "\np " << vertex << '\n';
        if (!stream) {
            result.message =
                "could not write tactile geometry OBJ";
            return result;
        }
    }
    {
        std::ofstream stream(result.summaryJSON);
        stream << std::setprecision(9)
               << "{\"schema\":\"metalrobo.tactile_debug\","
               << "\"environment\":" << config.environment << ','
               << "\"sensor\":" << config.sensor << ','
               << "\"sensor_id\":\""
               << escapedJSON(tactile.sensorIds[config.sensor]) << "\","
               << "\"maximum_depth_m\":"
               << summary.netTorqueAndMaximumDepth.w << ','
               << "\"mean_depth_m\":"
               << summary.centroidLocalAndMeanDepth.w << ','
               << "\"contact_area_m2\":"
               << summary.netForceAndContactArea.w << ','
               << "\"active_samples\":"
               << summary.centroidWorldAndActiveCount.w << ','
               << "\"saturated_samples\":"
               << summary.statisticsAndIdentity.x << ','
               << "\"object_shape_id\":"
               << summary.statisticsAndIdentity.w << ','
               << "\"net_force_n\":["
               << summary.netForceAndContactArea.x << ','
               << summary.netForceAndContactArea.y << ','
               << summary.netForceAndContactArea.z << "],"
               << "\"net_torque_nm\":["
               << summary.netTorqueAndMaximumDepth.x << ','
               << summary.netTorqueAndMaximumDepth.y << ','
               << summary.netTorqueAndMaximumDepth.z << "],"
               << "\"center_of_pressure_world_m\":["
               << summary.
                    centerOfPressureWorldAndContactCount.x << ','
               << summary.
                    centerOfPressureWorldAndContactCount.y << ','
               << summary.
                    centerOfPressureWorldAndContactCount.z << "]}\n";
        if (!stream) {
            result.message =
                "could not write tactile debug summary JSON";
            return result;
        }
    }
    result.message = "ok";
    return result;
}

} // namespace metalrobo
