#include "metalrobo/VisualPlatform.hpp"

#include "metalrobo/FrankaWorld.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numbers>
#include <ranges>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace metalrobo {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

bool fail(std::string* reason, std::string message) {
    if (reason != nullptr) {
        *reason = std::move(message);
    }
    return false;
}

bool finite(const float value) {
    return std::isfinite(value);
}

bool finite(const double value) {
    return std::isfinite(value);
}

bool finite(const mr_float4 value) {
    return finite(value.x) && finite(value.y) &&
        finite(value.z) && finite(value.w);
}

bool unitQuaternion(const mr_float4 value) {
    if (!finite(value)) {
        return false;
    }
    const double squared =
        static_cast<double>(value.x) * value.x +
        static_cast<double>(value.y) * value.y +
        static_cast<double>(value.z) * value.z +
        static_cast<double>(value.w) * value.w;
    return std::abs(squared - 1.0) <= 1.0e-4;
}

class HashBuilder {
public:
    void appendBytes(const void* data, const std::size_t size) {
        const auto* bytes =
            static_cast<const unsigned char*>(data);
        for (std::size_t index = 0u; index < size; ++index) {
            value_ ^= bytes[index];
            value_ *= kFnvPrime;
        }
    }

    template <typename T>
    void appendScalar(const T& value) {
        appendBytes(&value, sizeof(value));
    }

    template <typename T>
    void appendSpan(const std::span<const T> values) {
        const std::uint64_t count = values.size();
        appendScalar(count);
        if (!values.empty()) {
            appendBytes(values.data(), values.size_bytes());
        }
    }

    void appendString(const std::string_view value) {
        const std::uint64_t size = value.size();
        appendScalar(size);
        appendBytes(value.data(), value.size());
    }

    [[nodiscard]] std::uint64_t finish() const noexcept {
        return value_;
    }

private:
    std::uint64_t value_ = kFnvOffset;
};

mr_float4 quaternionProduct(
    const mr_float4 a,
    const mr_float4 b
) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

mr_float4 rotateVector(
    const mr_float4 quaternion,
    const mr_float4 vector
) {
    const mr_float4 conjugate{
        -quaternion.x,
        -quaternion.y,
        -quaternion.z,
        quaternion.w,
    };
    const mr_float4 rotated = quaternionProduct(
        quaternionProduct(
            quaternion,
            {vector.x, vector.y, vector.z, 0.0f}
        ),
        conjugate
    );
    return {rotated.x, rotated.y, rotated.z, vector.w};
}

mr_float4 add3(const mr_float4 a, const mr_float4 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w};
}

mr_float4 subtract3(const mr_float4 a, const mr_float4 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z, 0.0f};
}

std::uint64_t sensorProfileFingerprint(
    const VisualSensorProfileV2& profile
) {
    HashBuilder hash;
    hash.appendString(profile.id);
    hash.appendScalar(profile.nominalRateHz);
    hash.appendScalar(profile.exposureSeconds);
    hash.appendScalar(profile.shutterReadoutSeconds);
    hash.appendScalar(profile.shutterModel);
    hash.appendScalar(profile.shutterDirection);
    hash.appendScalar(profile.frameJitterSeconds);
    hash.appendScalar(profile.minimumDepthMeters);
    hash.appendScalar(profile.maximumDepthMeters);
    hash.appendScalar(profile.depthQuantumMeters);
    hash.appendScalar(profile.latencySeconds);
    return hash.finish();
}

std::uint64_t episodeFingerprint(
    const VisualEpisodeStreamV1& stream
) {
    HashBuilder hash;
    hash.appendScalar(stream.schemaVersion);
    hash.appendString(stream.id);
    hash.appendScalar(stream.source);
    hash.appendScalar(stream.episodeTwinFingerprint);
    hash.appendScalar(stream.worldFamilyFingerprint);
    hash.appendScalar(stream.scenarioFingerprint);
    hash.appendScalar(stream.rendererFingerprint);
    hash.appendScalar(stream.visualSceneFingerprint);
    hash.appendScalar(stream.visualPackFingerprint);
    hash.appendScalar(stream.environmentMapFingerprint);
    hash.appendScalar(stream.lightRigFingerprint);
    hash.appendScalar(stream.rendererProfileFingerprint);
    hash.appendScalar(stream.shutterProfileFingerprint);
    hash.appendScalar(stream.sensorProfileFingerprint);
    hash.appendScalar(stream.calibrationFingerprint);
    hash.appendScalar(stream.physicsFingerprint);
    for (const VisualEpisodeStepV1& step : stream.steps) {
        hash.appendScalar(step.frameIndex);
        hash.appendScalar(step.scenarioKey);
        hash.appendScalar(step.timestampSeconds);
        hash.appendScalar(step.reward);
        hash.appendSpan<float>(step.proprioception);
        hash.appendSpan<float>(step.action);
        hash.appendSpan<float>(step.taskCommand);
        hash.appendSpan<float>(step.privilegedState);
        hash.appendString(step.frameContentHash);
        hash.appendString(step.truthContentHash);
        hash.appendScalar(step.eventFlags);
    }
    return hash.finish();
}

bool checkedPixels(
    const std::uint32_t environments,
    const std::uint32_t views,
    const std::uint32_t width,
    const std::uint32_t height,
    std::size_t& output
) {
    std::size_t result = environments;
    for (const std::uint32_t factor : {views, width, height}) {
        if (factor != 0u &&
            result >
                std::numeric_limits<std::size_t>::max() /
                    factor) {
            return false;
        }
        result *= factor;
    }
    output = result;
    return true;
}

bool finiteValues(const std::span<const float> values) {
    return std::ranges::all_of(values, [](const float value) {
        return std::isfinite(value);
    });
}

struct NormalizedPoint {
    float x = 0.0f;
    float y = 0.0f;
    bool valid = false;
};

NormalizedPoint undistortNormalized(
    const float distortedX,
    const float distortedY,
    const mr_float4 distortion
) {
    if (!finite(distortedX) || !finite(distortedY) ||
        !finite(distortion)) {
        return {};
    }
    double x = distortedX;
    double y = distortedY;
    const double k1 = distortion.x;
    const double k2 = distortion.y;
    const double p1 = distortion.z;
    const double p2 = distortion.w;
    for (std::uint32_t iteration = 0u;
         iteration < 8u;
         ++iteration) {
        const double radiusSquared = x * x + y * y;
        const double radial =
            1.0 + k1 * radiusSquared +
            k2 * radiusSquared * radiusSquared;
        if (!std::isfinite(radial) ||
            std::abs(radial) <= 1.0e-12) {
            return {};
        }
        const double tangentialX =
            2.0 * p1 * x * y +
            p2 * (radiusSquared + 2.0 * x * x);
        const double tangentialY =
            p1 * (radiusSquared + 2.0 * y * y) +
            2.0 * p2 * x * y;
        x = (static_cast<double>(distortedX) - tangentialX) /
            radial;
        y = (static_cast<double>(distortedY) - tangentialY) /
            radial;
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return {};
        }
    }
    const double radiusSquared = x * x + y * y;
    const double radial =
        1.0 + k1 * radiusSquared +
        k2 * radiusSquared * radiusSquared;
    const double projectedX =
        x * radial +
        2.0 * p1 * x * y +
        p2 * (radiusSquared + 2.0 * x * x);
    const double projectedY =
        y * radial +
        p1 * (radiusSquared + 2.0 * y * y) +
        2.0 * p2 * x * y;
    if (!std::isfinite(projectedX) ||
        !std::isfinite(projectedY) ||
        std::abs(projectedX - distortedX) > 1.0e-4 ||
        std::abs(projectedY - distortedY) > 1.0e-4) {
        return {};
    }
    return {
        static_cast<float>(x),
        static_cast<float>(y),
        true,
    };
}

struct PixelProjection {
    float x = 0.0f;
    float y = 0.0f;
    float depth = 0.0f;
    bool valid = false;
};

PixelProjection projectBasePoint(
    const mr_float4 pointInBase,
    const VisualCameraFrameV1& camera
) {
    if (!finite(pointInBase) ||
        !finite(camera.baseFromCamera.position) ||
        !unitQuaternion(camera.baseFromCamera.orientation) ||
        !finite(camera.intrinsics) ||
        !finite(camera.distortion) ||
        !(camera.intrinsics.x > 0.0f) ||
        !(camera.intrinsics.y > 0.0f)) {
        return {};
    }
    const mr_float4 inverseCamera{
        -camera.baseFromCamera.orientation.x,
        -camera.baseFromCamera.orientation.y,
        -camera.baseFromCamera.orientation.z,
        camera.baseFromCamera.orientation.w,
    };
    const mr_float4 pointInCamera = rotateVector(
        inverseCamera,
        subtract3(pointInBase, camera.baseFromCamera.position)
    );
    if (!finite(pointInCamera) || !(pointInCamera.z > 1.0e-6f)) {
        return {};
    }
    const double normalizedX =
        static_cast<double>(pointInCamera.x) / pointInCamera.z;
    const double normalizedY =
        static_cast<double>(pointInCamera.y) / pointInCamera.z;
    const double radiusSquared =
        normalizedX * normalizedX + normalizedY * normalizedY;
    const double radial =
        1.0 +
        static_cast<double>(camera.distortion.x) * radiusSquared +
        static_cast<double>(camera.distortion.y) *
            radiusSquared * radiusSquared;
    const double distortedX =
        normalizedX * radial +
        2.0 * camera.distortion.z *
            normalizedX * normalizedY +
        camera.distortion.w *
            (radiusSquared + 2.0 * normalizedX * normalizedX);
    const double distortedY =
        normalizedY * radial +
        camera.distortion.z *
            (radiusSquared + 2.0 * normalizedY * normalizedY) +
        2.0 * camera.distortion.w *
            normalizedX * normalizedY;
    const double pixelX =
        distortedX * camera.intrinsics.x + camera.intrinsics.z;
    const double pixelY =
        distortedY * camera.intrinsics.y + camera.intrinsics.w;
    if (!std::isfinite(pixelX) || !std::isfinite(pixelY)) {
        return {};
    }
    return {
        static_cast<float>(pixelX),
        static_cast<float>(pixelY),
        pointInCamera.z,
        true,
    };
}

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) *
        0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) *
        0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

double signedUnit(const std::uint64_t value) {
    constexpr double inverse =
        1.0 / static_cast<double>(std::uint64_t{1} << 53u);
    const double unit = static_cast<double>(
        mix64(value) >> 11u
    ) * inverse;
    return 2.0 * unit - 1.0;
}

WorldPose composePose(
    const WorldPose& parent,
    const WorldPose& local
) {
    WorldPose result;
    result.position = add3(
        parent.position,
        rotateVector(parent.orientation, local.position)
    );
    result.position.w = 0.0f;
    result.orientation = quaternionProduct(
        parent.orientation,
        local.orientation
    );
    return result;
}

WorldPose bodyPose(const MRBodyStateGPU& body) {
    return {
        {
            body.position.x,
            body.position.y,
            body.position.z,
            0.0f,
        },
        body.orientation,
    };
}

WorldPose assetPose(const MRWorldAssetInstanceGPU& asset) {
    return {
        {
            asset.positionAndScale.x,
            asset.positionAndScale.y,
            asset.positionAndScale.z,
            0.0f,
        },
        asset.orientation,
    };
}

WorldPose relativePose(
    const WorldPose& baseInWorld,
    const WorldPose& poseInWorld
) {
    const mr_float4 inverseBase{
        -baseInWorld.orientation.x,
        -baseInWorld.orientation.y,
        -baseInWorld.orientation.z,
        baseInWorld.orientation.w,
    };
    WorldPose result;
    result.position = rotateVector(
        inverseBase,
        subtract3(
            poseInWorld.position,
            baseInWorld.position
        )
    );
    result.position.w = 0.0f;
    result.orientation = quaternionProduct(
        inverseBase,
        poseInWorld.orientation
    );
    return result;
}

bool hasDeviceModality(
    const std::span<const VisualDeviceBufferViewV1> buffers,
    const std::uint32_t modality
) {
    return std::ranges::any_of(
        buffers,
        [modality](const VisualDeviceBufferViewV1& buffer) {
            return buffer.valid() &&
                (buffer.modality & modality) != 0u;
        }
    );
}

bool hasDeviceStorage(
    const std::span<const VisualDeviceBufferViewV1> buffers,
    const std::uint32_t modality,
    const std::size_t elementCount,
    const std::span<const MRVisualPixelFormat> acceptedFormats
) {
    for (const VisualDeviceBufferViewV1& buffer : buffers) {
        if (!buffer.valid() || buffer.modality != modality ||
            std::ranges::find(
                acceptedFormats,
                buffer.format
            ) == acceptedFormats.end()) {
            continue;
        }
        std::size_t bytesPerElement = 0u;
        switch (buffer.format) {
        case MR_VISUAL_FORMAT_RGBA32_FLOAT:
        case MR_VISUAL_FORMAT_RGBA32_UINT:
            bytesPerElement = 16u;
            break;
        case MR_VISUAL_FORMAT_RG32_FLOAT:
            bytesPerElement = 8u;
            break;
        case MR_VISUAL_FORMAT_R32_FLOAT:
        case MR_VISUAL_FORMAT_R32_UINT:
            bytesPerElement = 4u;
            break;
        case MR_VISUAL_FORMAT_R8_UINT:
            bytesPerElement = 1u;
            break;
        case MR_VISUAL_FORMAT_UNKNOWN:
        default:
            continue;
        }
        if (elementCount <=
                std::numeric_limits<std::size_t>::max() /
                    bytesPerElement &&
            buffer.sizeBytes >= elementCount * bytesPerElement) {
            return true;
        }
    }
    return false;
}

std::string jsonEscape(const std::string_view value) {
    std::ostringstream stream;
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            stream << "\\\"";
            break;
        case '\\':
            stream << "\\\\";
            break;
        case '\b':
            stream << "\\b";
            break;
        case '\f':
            stream << "\\f";
            break;
        case '\n':
            stream << "\\n";
            break;
        case '\r':
            stream << "\\r";
            break;
        case '\t':
            stream << "\\t";
            break;
        default:
            if (character < 0x20u) {
                stream << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0')
                       << static_cast<unsigned int>(character)
                       << std::dec;
            } else {
                stream << static_cast<char>(character);
            }
            break;
        }
    }
    return stream.str();
}

template <typename Value>
void writeJSONFloatArray(
    std::ostream& stream,
    const std::vector<Value>& values
) {
    stream << '[';
    for (std::size_t index = 0u; index < values.size(); ++index) {
        if (index != 0u) {
            stream << ',';
        }
        stream << values[index];
    }
    stream << ']';
}

} // namespace

std::uint64_t computeVisualSensorProfileFingerprint(
    const VisualSensorProfileV2& profile
) {
    VisualSensorProfileV2 candidate = profile;
    candidate.fingerprint = 0u;
    return candidate.valid()
        ? sensorProfileFingerprint(candidate)
        : 0u;
}

bool VisualSensorCaptureV1::valid(std::string* reason) const {
    if (!finite(nominalTimestampSeconds) ||
        !finite(exposureOpenSeconds) ||
        !finite(exposureCloseSeconds) ||
        !finite(publishTimestampSeconds) ||
        exposureOpenSeconds > exposureCloseSeconds ||
        exposureCloseSeconds > publishTimestampSeconds) {
        return fail(reason, "visual sensor capture timing is invalid");
    }
    return true;
}

VisualSensorCaptureV1 makeVisualSensorCapture(
    const VisualSensorProfileV2& profile,
    const std::uint64_t scenarioIdentity,
    const std::uint64_t sensorIdentity,
    const std::uint64_t frameIndex,
    const double episodeStartSeconds
) {
    VisualSensorCaptureV1 result;
    if (!profile.valid() || !finite(episodeStartSeconds)) {
        return result;
    }
    const double nominal =
        episodeStartSeconds +
        static_cast<double>(frameIndex) /
            profile.nominalRateHz;
    const std::uint64_t identity =
        mix64(scenarioIdentity) ^
        std::rotl(mix64(sensorIdentity), 19) ^
        std::rotl(mix64(frameIndex), 41);
    const double jitter =
        profile.frameJitterSeconds * signedUnit(identity);
    const double center = nominal + jitter;
    result.frameIndex = frameIndex;
    result.sensorSequence =
        static_cast<std::uint32_t>(frameIndex);
    result.nominalTimestampSeconds = nominal;
    result.exposureOpenSeconds =
        center - 0.5 * profile.exposureSeconds;
    result.exposureCloseSeconds =
        center + 0.5 * profile.exposureSeconds +
        profile.shutterReadoutSeconds;
    result.publishTimestampSeconds =
        result.exposureCloseSeconds + profile.latencySeconds;
    return result;
}

bool VisualBatchProvenanceV1::valid(std::string* reason) const {
    if (source > MR_VISUAL_SOURCE_REPLAY ||
        episodeTwinFingerprint == 0u ||
        scenarioFingerprint == 0u ||
        rendererFingerprint == 0u ||
        visualPackFingerprint == 0u ||
        environmentMapFingerprint == 0u ||
        lightRigFingerprint == 0u ||
        rendererProfileFingerprint == 0u ||
        shutterProfileFingerprint == 0u ||
        sensorProfileFingerprint == 0u ||
        calibrationFingerprint == 0u) {
        return fail(reason, "visual batch provenance is incomplete");
    }
    return true;
}

bool assembleVisualBatches(
    const WorldTemplate& world,
    const WorldInstanceBatch& sampledWorlds,
    const VisualBatchAssemblyV1& input,
    VisualFrameBatchV1& frames,
    VisualTruthBatchV1& truth,
    std::string* reason
) {
    std::string worldReason;
    std::string sampledReason;
    std::string provenanceReason;
    if (!world.valid(&worldReason) ||
        !sampledWorlds.valid(&sampledReason) ||
        !input.provenance.valid(&provenanceReason)) {
        return fail(
            reason,
            "visual batch assembly input is invalid"
        );
    }
    if (sampledWorlds.instances.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        input.cameraIndices.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        world.engineModel.bodies.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return fail(
            reason,
            "visual batch dimensions exceed the ABI"
        );
    }
    const std::uint32_t environmentCount =
        static_cast<std::uint32_t>(
            sampledWorlds.instances.size()
        );
    const std::uint32_t viewCount =
        static_cast<std::uint32_t>(
            input.cameraIndices.size()
        );
    const std::uint32_t bodyCount =
        static_cast<std::uint32_t>(
            world.engineModel.bodies.size()
        );
    if (environmentCount == 0u || viewCount == 0u ||
        input.observations.size() != viewCount ||
        input.currentBodyStates.size() !=
            static_cast<std::size_t>(environmentCount) *
                bodyCount) {
        return fail(
            reason,
            "visual batch assembly dimensions are invalid"
        );
    }
    std::unordered_set<std::uint32_t> cameraSet;
    const std::uint32_t width =
        input.observations.front().width;
    const std::uint32_t height =
        input.observations.front().height;
    std::size_t pixelsPerView = 0u;
    std::size_t totalPixels = 0u;
    if (width == 0u || height == 0u ||
        !checkedPixels(
            1u,
            1u,
            width,
            height,
            pixelsPerView
        ) ||
        !checkedPixels(
            environmentCount,
            viewCount,
            width,
            height,
            totalPixels
        )) {
        return fail(reason, "visual batch resolution is invalid");
    }
    const MRVisualFrameMetadataGPU referenceMetadata =
        input.observations.front().metadata;
    constexpr std::uint32_t kRequiredRendererModalities =
        MR_VISUAL_MODALITY_RGB |
        MR_VISUAL_MODALITY_DEPTH |
        MR_VISUAL_MODALITY_DEPTH_VALIDITY |
        MR_VISUAL_MODALITY_NORMAL |
        MR_VISUAL_MODALITY_MOTION |
        MR_VISUAL_MODALITY_SEMANTIC |
        MR_VISUAL_MODALITY_INSTANCE |
        MR_VISUAL_MODALITY_LINK;
    for (std::uint32_t view = 0u; view < viewCount; ++view) {
        const std::uint32_t camera = input.cameraIndices[view];
        const HybridObservationBatch& observation =
            input.observations[view];
        const MRVisualFrameMetadataGPU metadata =
            observation.metadata;
        const std::size_t pixels =
            static_cast<std::size_t>(environmentCount) *
            pixelsPerView;
        if (camera >= world.sensors.size() ||
            !cameraSet.insert(camera).second ||
            observation.environmentCount != environmentCount ||
            observation.width != width ||
            observation.height != height ||
            observation.rgb.size() != pixels ||
            observation.depth.size() != pixels ||
            observation.segmentation.size() != pixels ||
            observation.identities.size() != pixels ||
            observation.normals.size() != pixels ||
            observation.motion.size() != pixels ||
            observation.validity.size() != pixels ||
            metadata.dimensions.x != environmentCount ||
            metadata.dimensions.y != 1u ||
            metadata.dimensions.z != width ||
            metadata.dimensions.w != height ||
            metadata.identity.w != input.provenance.source ||
            metadata.contract.y != MR_VISUAL_FRAME_CAMERA ||
            metadata.contract.z != MR_VISUAL_PLATFORM_ABI_VERSION ||
            metadata.contract.x != kRequiredRendererModalities ||
            !finite(metadata.timing) ||
            metadata.timing.y < 0.0f ||
            metadata.timing.z < 0.0f ||
            metadata.timing.w < 0.0f ||
            metadata.identity.x != referenceMetadata.identity.x ||
            metadata.identity.y != referenceMetadata.identity.y ||
            std::abs(
                (
                    metadata.timing.x +
                    metadata.timing.y
                ) -
                (
                    referenceMetadata.timing.x +
                    referenceMetadata.timing.y
                )
            ) > 1.0e-5f) {
            return fail(
                reason,
                "visual views are incomplete, stale, or unsynchronized"
            );
        }
    }

    std::uint32_t rootBody = MR_INVALID_INDEX;
    const std::uint32_t robotAsset =
        world.assetIndex(world.task.robotAssetId);
    if (robotAsset < world.assets.size()) {
        const std::uint32_t articulation =
            world.assets[robotAsset].articulationIndex;
        if (articulation < world.engineModel.articulations.size()) {
            rootBody =
                world.engineModel.articulations[articulation].
                    rootBody;
        }
    }
    if (rootBody == MR_INVALID_INDEX &&
        !world.engineModel.articulations.empty()) {
        rootBody = world.engineModel.articulations.front().rootBody;
    }
    if (rootBody >= bodyCount) {
        return fail(reason, "visual robot base body is unavailable");
    }

    VisualFrameBatchV1 frameCandidate;
    frameCandidate.source = input.provenance.source;
    frameCandidate.environmentCount = environmentCount;
    frameCandidate.viewCount = viewCount;
    frameCandidate.width = width;
    frameCandidate.height = height;
    frameCandidate.modalities =
        MR_VISUAL_MODALITY_RGB |
        MR_VISUAL_MODALITY_DEPTH |
        MR_VISUAL_MODALITY_DEPTH_VALIDITY;
    frameCandidate.episodeTwinFingerprint =
        input.provenance.episodeTwinFingerprint;
    frameCandidate.scenarioFingerprint =
        input.provenance.scenarioFingerprint;
    frameCandidate.rendererFingerprint =
        input.provenance.rendererFingerprint;
    frameCandidate.sensorProfileFingerprint =
        input.provenance.sensorProfileFingerprint;
    frameCandidate.calibrationFingerprint =
        input.provenance.calibrationFingerprint;
    frameCandidate.cameras.resize(
        static_cast<std::size_t>(environmentCount) *
        viewCount
    );
    frameCandidate.rgbLinear.resize(totalPixels);
    frameCandidate.depthMeters.resize(totalPixels);
    frameCandidate.depthValidity.resize(totalPixels);

    VisualTruthBatchV1 truthCandidate;
    truthCandidate.modalities =
        MR_VISUAL_MODALITY_NORMAL |
        MR_VISUAL_MODALITY_MOTION |
        MR_VISUAL_MODALITY_SEMANTIC |
        MR_VISUAL_MODALITY_INSTANCE |
        MR_VISUAL_MODALITY_LINK |
        MR_VISUAL_MODALITY_KEYPOINT |
        MR_VISUAL_MODALITY_OBJECT_POSE;
    truthCandidate.environmentCount = environmentCount;
    truthCandidate.viewCount = viewCount;
    truthCandidate.width = width;
    truthCandidate.height = height;
    truthCandidate.normals.resize(totalPixels);
    truthCandidate.motion.resize(totalPixels);
    truthCandidate.semanticIds.resize(totalPixels);
    truthCandidate.instanceIds.resize(totalPixels);
    truthCandidate.linkIds.resize(totalPixels);
    truthCandidate.visibility.resize(totalPixels);
    truthCandidate.occlusion.resize(totalPixels);
    truthCandidate.objectPoses.reserve(
        static_cast<std::size_t>(environmentCount) *
        world.assets.size()
    );
    truthCandidate.linkPoses.reserve(
        static_cast<std::size_t>(environmentCount) *
        bodyCount
    );

    std::vector<std::uint32_t> bodyAssets(
        bodyCount,
        MR_INVALID_INDEX
    );
    std::vector<std::uint32_t> assetSemanticIds(
        world.assets.size(),
        0u
    );
    std::unordered_map<std::string, std::uint32_t>
        semanticIds;
    for (std::uint32_t asset = 0u;
         asset < world.assets.size();
         ++asset) {
        const auto [semantic, inserted] =
            semanticIds.try_emplace(
                world.assets[asset].semanticClass,
                static_cast<std::uint32_t>(
                    semanticIds.size() + 1u
                )
            );
        static_cast<void>(inserted);
        assetSemanticIds[asset] = semantic->second;
        for (const std::uint32_t body :
             world.assets[asset].bodyIndices) {
            if (body < bodyAssets.size()) {
                bodyAssets[body] = asset;
            }
        }
    }

    for (std::uint32_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        const MRWorldInstanceHeaderGPU& instance =
            sampledWorlds.instances[environment];
        const WorldPose baseInWorld = bodyPose(
            input.currentBodyStates[
                static_cast<std::size_t>(environment) *
                    bodyCount +
                rootBody
            ]
        );
        for (std::uint32_t view = 0u; view < viewCount; ++view) {
            const std::uint32_t cameraIndex =
                input.cameraIndices[view];
            const SensorSpec& sensorSpec =
                world.sensors[cameraIndex];
            const MRWorldSensorInstanceGPU& sensor =
                sampledWorlds.sensors[
                    instance.ranges.z + cameraIndex
                ];
            WorldPose parentInWorld;
            switch (sensorSpec.parentKind) {
            case MR_WORLD_SENSOR_PARENT_WORLD:
                break;
            case MR_WORLD_SENSOR_PARENT_RIGID_BODY:
            case MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK:
                parentInWorld = bodyPose(
                    input.currentBodyStates[
                        static_cast<std::size_t>(environment) *
                            bodyCount +
                        sensorSpec.parentBodyIndex
                    ]
                );
                break;
            case MR_WORLD_SENSOR_PARENT_ASSET:
            default: {
                const std::uint32_t asset =
                    world.assetIndex(sensorSpec.parentAssetId);
                parentInWorld = assetPose(
                    sampledWorlds.assets[
                        instance.ranges.x + asset
                    ]
                );
                break;
            }
            }
            const WorldPose sensorLocal{
                {
                    sensor.positionAndFocalScale.x,
                    sensor.positionAndFocalScale.y,
                    sensor.positionAndFocalScale.z,
                    0.0f,
                },
                sensor.orientation,
            };
            const WorldPose cameraInWorld =
                composePose(parentInWorld, sensorLocal);
            VisualCameraFrameV1& camera =
                frameCandidate.cameras[
                    static_cast<std::size_t>(environment) *
                        viewCount +
                    view
                ];
            camera.sensorId = sensorSpec.id;
            camera.intrinsics = {
                sensor.intrinsics.x *
                    sensor.positionAndFocalScale.w,
                sensor.intrinsics.y *
                    sensor.positionAndFocalScale.w,
                sensor.intrinsics.z,
                sensor.intrinsics.w,
            };
            camera.distortion = sensor.distortion;
            camera.baseFromCamera =
                relativePose(baseInWorld, cameraInWorld);
            const MRVisualFrameMetadataGPU metadata =
                input.observations[view].metadata;
            camera.captureTimestampSeconds = metadata.timing.x;
            camera.frameAgeSeconds = metadata.timing.y;
            camera.exposureSeconds = metadata.timing.z;
            camera.shutterReadoutSeconds = metadata.timing.w;
            camera.frameIndex =
                static_cast<std::uint64_t>(
                    metadata.identity.x
                ) |
                (static_cast<std::uint64_t>(
                     metadata.identity.y
                 ) << 32u);
            camera.sensorSequence = metadata.identity.z;
            camera.valid =
                metadata.dimensions.x == environmentCount &&
                metadata.dimensions.z == width &&
                metadata.dimensions.w == height;

            const HybridObservationBatch& observation =
                input.observations[view];
            for (std::size_t local = 0u;
                 local < pixelsPerView;
                 ++local) {
                const std::size_t source =
                    static_cast<std::size_t>(environment) *
                        pixelsPerView +
                    local;
                const std::size_t destination =
                    (
                        static_cast<std::size_t>(environment) *
                            viewCount +
                        view
                    ) * pixelsPerView + local;
                frameCandidate.rgbLinear[destination] =
                    observation.rgb[source];
                const std::uint32_t validity =
                    observation.validity[source];
                const bool geometryValid =
                    (validity &
                     MR_VISUAL_VALIDITY_GEOMETRY) != 0u;
                const mr_uint4 identity =
                    observation.identities[source];
                if ((validity & MR_VISUAL_VALIDITY_FRAME) == 0u ||
                    (validity &
                     ~(
                         MR_VISUAL_VALIDITY_FRAME |
                         MR_VISUAL_VALIDITY_DEPTH |
                         MR_VISUAL_VALIDITY_GEOMETRY
                     )) != 0u ||
                    (
                        (validity &
                         MR_VISUAL_VALIDITY_DEPTH) != 0u &&
                        !geometryValid
                    ) ||
                    (
                        geometryValid &&
                        (
                            identity.x == MR_INVALID_INDEX ||
                            identity.y == MR_INVALID_INDEX
                        )
                    ) ||
                    (
                        !geometryValid &&
                        (
                            identity.x != MR_INVALID_INDEX ||
                            identity.y != MR_INVALID_INDEX ||
                            identity.z != MR_INVALID_INDEX ||
                            identity.w != MR_INVALID_INDEX
                        )
                    )) {
                    return fail(
                        reason,
                        "renderer validity and geometry truth disagree"
                    );
                }
                const bool depthValid =
                    (validity &
                     MR_VISUAL_VALIDITY_DEPTH) != 0u;
                frameCandidate.depthValidity[destination] =
                    depthValid ? 1u : 0u;
                frameCandidate.depthMeters[destination] =
                    depthValid
                    ? observation.depth[source]
                    : 0.0f;
                if (observation.segmentation[source] !=
                    observation.identities[source].x) {
                    return fail(
                        reason,
                        "semantic and identity buffers disagree"
                    );
                }
                truthCandidate.normals[destination] =
                    observation.normals[source];
                truthCandidate.motion[destination] =
                    observation.motion[source];
                truthCandidate.semanticIds[destination] =
                    observation.identities[source].x;
                truthCandidate.instanceIds[destination] =
                    observation.identities[source].y;
                truthCandidate.linkIds[destination] =
                    observation.identities[source].z;
                const float visibility =
                    geometryValid
                    ? std::clamp(
                          observation.motion[source].z,
                          0.0f,
                          1.0f
                      )
                    : 0.0f;
                truthCandidate.visibility[destination] =
                    visibility;
                truthCandidate.occlusion[destination] =
                    static_cast<std::uint8_t>(
                        std::lround(
                            (1.0f - visibility) * 255.0f
                        )
                    );
            }
        }

        for (std::uint32_t asset = 0u;
             asset < world.assets.size();
             ++asset) {
            const MRWorldAssetInstanceGPU& sampledAsset =
                sampledWorlds.assets[
                    instance.ranges.x + asset
                ];
            const WorldAsset& assetDefinition =
                world.assets[asset];
            std::uint32_t liveBody = MR_INVALID_INDEX;
            if (assetDefinition.articulationIndex <
                world.engineModel.articulations.size()) {
                liveBody =
                    world.engineModel.articulations[
                        assetDefinition.articulationIndex
                    ].rootBody;
            } else if (!assetDefinition.bodyIndices.empty()) {
                liveBody = assetDefinition.bodyIndices.front();
            }
            const WorldPose currentAssetInWorld =
                liveBody < bodyCount
                ? bodyPose(
                      input.currentBodyStates[
                          static_cast<std::size_t>(environment) *
                              bodyCount +
                          liveBody
                      ]
                  )
                : assetPose(sampledAsset);
            const WorldPose objectInBase = relativePose(
                baseInWorld,
                currentAssetInWorld
            );
            truthCandidate.objectPoses.push_back({
                {
                    objectInBase.position.x,
                    objectInBase.position.y,
                    objectInBase.position.z,
                    1.0f,
                },
                objectInBase.orientation,
                {
                    assetSemanticIds[asset],
                    asset + 1u,
                    MR_INVALID_INDEX,
                    sampledAsset.identity.y,
                },
            });
            for (std::uint32_t anchorIndex = 0u;
                 anchorIndex <
                    assetDefinition.anchors.size();
                 ++anchorIndex) {
                const SemanticAnchor& anchor =
                    assetDefinition.anchors[anchorIndex];
                const WorldPose anchorInBase = relativePose(
                    baseInWorld,
                    composePose(
                        currentAssetInWorld,
                        anchor.localPose
                    )
                );
                float keypointVisibility = 0.0f;
                for (std::uint32_t view = 0u;
                     view < viewCount;
                     ++view) {
                    const VisualCameraFrameV1& camera =
                        frameCandidate.cameras[
                            static_cast<std::size_t>(environment) *
                                viewCount +
                            view
                        ];
                    const PixelProjection projection =
                        projectBasePoint(
                            anchorInBase.position,
                            camera
                        );
                    if (!projection.valid ||
                        projection.x < 0.0f ||
                        projection.y < 0.0f ||
                        projection.x >=
                            static_cast<float>(width) ||
                        projection.y >=
                            static_cast<float>(height)) {
                        continue;
                    }
                    const std::uint32_t pixelX =
                        static_cast<std::uint32_t>(
                            std::floor(projection.x)
                        );
                    const std::uint32_t pixelY =
                        static_cast<std::uint32_t>(
                            std::floor(projection.y)
                        );
                    const std::size_t pixel =
                        (
                            static_cast<std::size_t>(environment) *
                                viewCount +
                            view
                        ) * pixelsPerView +
                        static_cast<std::size_t>(pixelY) * width +
                        pixelX;
                    if (truthCandidate.instanceIds[pixel] ==
                            asset + 1u &&
                        truthCandidate.visibility[pixel] > 0.0f) {
                        keypointVisibility +=
                            1.0f /
                            static_cast<float>(viewCount);
                    }
                }
                truthCandidate.keypoints.push_back({
                    {
                        anchorInBase.position.x,
                        anchorInBase.position.y,
                        anchorInBase.position.z,
                        keypointVisibility,
                    },
                    {
                        assetSemanticIds[asset],
                        asset + 1u,
                        assetDefinition.bodyIndices.empty()
                            ? MR_INVALID_INDEX
                            : assetDefinition.bodyIndices.front(),
                        anchorIndex,
                    },
                });
            }
        }
        for (std::uint32_t body = 0u;
             body < bodyCount;
             ++body) {
            const WorldPose linkInBase = relativePose(
                baseInWorld,
                bodyPose(
                    input.currentBodyStates[
                        static_cast<std::size_t>(environment) *
                            bodyCount +
                        body
                    ]
                )
            );
            const std::uint32_t asset = bodyAssets[body];
            truthCandidate.linkPoses.push_back({
                {
                    linkInBase.position.x,
                    linkInBase.position.y,
                    linkInBase.position.z,
                    1.0f,
                },
                linkInBase.orientation,
                {
                    asset == MR_INVALID_INDEX
                        ? 0u
                        : assetSemanticIds[asset],
                    asset == MR_INVALID_INDEX ? 0u : asset + 1u,
                    body,
                    0u,
                },
            });
        }
    }
    truthCandidate.frameIndex =
        frameCandidate.cameras.front().frameIndex;
    truthCandidate.timestampSeconds =
        frameCandidate.cameras.front().captureTimestampSeconds;

    std::string frameReason;
    std::string truthReason;
    if (!frameCandidate.valid(&frameReason) ||
        !truthCandidate.valid(&truthReason)) {
        return fail(
            reason,
            "assembled visual contracts are invalid"
        );
    }
    frames = std::move(frameCandidate);
    truth = std::move(truthCandidate);
    return true;
}

bool compileVisualSceneManifestV3(
    const WorldTemplate& world,
    const std::span<const VisualAssetReferenceV3>
        authoredAssets,
    const VisualEnvironmentReferenceV2& environment,
    const VisualLightRigV1& lightRig,
    VisualSceneManifestV3& output,
    std::string* reason
) {
    std::string worldReason;
    std::string presentationReason;
    if (!world.valid(&worldReason) || authoredAssets.empty() ||
        !environment.valid(&presentationReason) ||
        !lightRig.valid(&presentationReason)) {
        return fail(
            reason,
            "V3 visual scene source is invalid: " +
                (
                    !worldReason.empty()
                    ? worldReason
                    : authoredAssets.empty()
                        ? "authored visual packs are required"
                        : presentationReason
                )
        );
    }
    VisualSceneManifestV3 candidate;
    candidate.id = world.id + ".visual.v3";
    candidate.worldFingerprint = world.fingerprint;
    candidate.preprocessingProvenance =
        "compileVisualSceneManifestV3/native-pack-references-v3";
    candidate.renderScene.id = candidate.id + ".runtime";
    candidate.renderScene.assetCount =
        static_cast<std::uint32_t>(world.assets.size());
    candidate.renderScene.bodyCount =
        static_cast<std::uint32_t>(
            world.engineModel.bodies.size()
        );
    candidate.renderScene.environment = environment;
    candidate.renderScene.lightRig = lightRig;
    candidate.renderScene.sensorBindings.reserve(
        world.sensors.size()
    );
    for (const SensorSpec& sensor : world.sensors) {
        const std::uint32_t assetIndex =
            world.assetIndex(sensor.parentAssetId);
        MRVisualSensorBindingGPU binding{};
        binding.identity.z =
            assetIndex < world.assets.size()
            ? assetIndex
            : MR_INVALID_INDEX;
        switch (sensor.parentKind) {
        case MR_WORLD_SENSOR_PARENT_WORLD:
            binding.identity.x = MR_VISUAL_BINDING_WORLD;
            binding.identity.y = MR_INVALID_INDEX;
            break;
        case MR_WORLD_SENSOR_PARENT_RIGID_BODY:
            binding.identity.x = MR_VISUAL_BINDING_RIGID_BODY;
            binding.identity.y = sensor.parentBodyIndex;
            break;
        case MR_WORLD_SENSOR_PARENT_ARTICULATED_LINK:
            binding.identity.x =
                MR_VISUAL_BINDING_ARTICULATED_LINK;
            binding.identity.y = sensor.parentBodyIndex;
            break;
        case MR_WORLD_SENSOR_PARENT_ASSET:
        default:
            binding.identity.x = MR_VISUAL_BINDING_ASSET;
            binding.identity.y = assetIndex;
            break;
        }
        binding.timing = {
            sensor.nominalRateHz,
            sensor.exposureSeconds,
            sensor.shutterReadoutSeconds,
            sensor.frameJitterSeconds,
        };
        binding.rangeAndResponse = {
            sensor.minimumDepthMeters,
            sensor.maximumDepthMeters,
            sensor.depthQuantumMeters,
            sensor.motionBlurScale,
        };
        binding.shutter = {
            sensor.shutterModel,
            sensor.shutterDirection,
            0u,
            0u,
        };
        candidate.renderScene.sensorBindings.push_back(binding);
    }

    for (const VisualAssetReferenceV3& reference :
         authoredAssets) {
        if (reference.packPath.empty() ||
            reference.assetIndex >=
                candidate.renderScene.assetCount ||
            reference.semanticId == 0u ||
            reference.semanticId == MR_INVALID_INDEX ||
            reference.instanceId == 0u ||
            reference.instanceId == MR_INVALID_INDEX) {
            return fail(
                reason,
                "authored V3 visual reference is invalid"
            );
        }
        // Validate one pack at a time, but keep only its immutable path and
        // content hash in the live scene.
        VisualAssetPackV2 pack;
        if (!readVisualAssetPackIndex(
                reference.packPath,
                pack,
                &presentationReason
            )) {
            return fail(
                reason,
                "could not read authored visual pack: " +
                    presentationReason
            );
        }
        const std::string packContentHash = pack.contentHash;
        if (packContentHash != reference.contentHash) {
            return fail(
                reason,
                "authored visual pack hash does not match its V3 "
                "reference"
            );
        }
        candidate.renderScene.visualPacks.push_back(reference);
        if (std::ranges::find(
                candidate.visualPackHashes,
                packContentHash
            ) == candidate.visualPackHashes.end()) {
            candidate.visualPackHashes.push_back(
                packContentHash
            );
        }
    }
    candidate.environmentMapHash = environment.contentHash;
    candidate.lightRigHash = lightRig.contentHash;
    candidate.renderScene.fingerprint =
        computeVisualRenderSceneV3Fingerprint(
            candidate.renderScene
        );
    candidate.fingerprint =
        computeVisualSceneManifestV3Fingerprint(candidate);
    if (!candidate.valid(&presentationReason)) {
        return fail(
            reason,
            "compiled V3 visual scene is invalid: " +
                presentationReason
        );
    }
    output = std::move(candidate);
    return true;
}

bool makeFrankaPickPlaceVisualSceneManifestV3(
    const std::span<const VisualAssetReferenceV3>
        authoredAssets,
    VisualSceneManifestV3& output,
    std::string* reason
) {
    WorldTemplate world;
    const WorldCompileResult compiled = compileEpisodeTwin(
        makeFrankaPickPlaceEpisodeTwin(),
        makeFrankaPickPlaceEngineModel(),
        world
    );
    if (!compiled.succeeded()) {
        return fail(
            reason,
            "could not compile Franka visual reference world: " +
                compiled.message
        );
    }
    return compileVisualSceneManifestV3(
        world,
        authoredAssets,
        makeNeutralStudioEnvironmentV2(),
        makeIndoorAreaLightRigV1(),
        output,
        reason
    );
}

bool composeVisualBodyStates(
    const EngineModel& model,
    const std::uint32_t environmentCount,
    const std::span<const float> q,
    const std::span<const float> v,
    const std::span<const MRBodyStateGPU> sceneBodies,
    std::vector<MRBodyStateGPU>& output,
    std::string* reason
) {
    std::string modelReason;
    if (!model.valid(&modelReason) || environmentCount == 0u) {
        return fail(
            reason,
            "visual body composition requires a valid model and batch"
        );
    }
    const std::size_t nq = model.world.nq;
    const std::size_t nv = model.world.nv;
    if (q.size() != environmentCount * nq ||
        v.size() != environmentCount * nv ||
        !finiteValues(q) || !finiteValues(v)) {
        return fail(
            reason,
            "visual body q/v streams have invalid dimensions or values"
        );
    }
    std::vector<std::uint32_t> sceneIndices;
    for (std::uint32_t body = 0u; body < model.bodies.size(); ++body) {
        if (model.bodies[body].articulationIndex ==
            MR_INVALID_INDEX) {
            sceneIndices.push_back(body);
        }
    }
    if (sceneBodies.size() !=
        environmentCount * sceneIndices.size()) {
        return fail(
            reason,
            "visual scene-body stream has invalid dimensions"
        );
    }
    std::vector<MRBodyStateGPU> candidate(
        static_cast<std::size_t>(environmentCount) *
        model.bodies.size()
    );
    for (std::uint32_t environment = 0u;
         environment < environmentCount;
         ++environment) {
        for (std::uint32_t articulationIndex = 0u;
             articulationIndex < model.articulations.size();
             ++articulationIndex) {
            const MRArticulationGPU& articulation =
                model.articulations[articulationIndex];
            std::vector<double> localQ(articulation.nq);
            std::vector<double> localV(articulation.nv);
            for (std::uint32_t coordinate = 0u;
                 coordinate < articulation.nq;
                 ++coordinate) {
                localQ[coordinate] =
                    q[environment * nq +
                      articulation.qOffset + coordinate];
            }
            for (std::uint32_t coordinate = 0u;
                 coordinate < articulation.nv;
                 ++coordinate) {
                localV[coordinate] =
                    v[environment * nv +
                      articulation.vOffset + coordinate];
            }
            std::vector<ArticulatedBodyKinematics> kinematics(
                articulation.bodyCount
            );
            const ArticulatedDynamicsDiagnostics diagnostics =
                computeArticulatedBodyKinematics(
                    model,
                    articulationIndex,
                    localQ,
                    localV,
                    kinematics
                );
            if (!diagnostics.succeeded()) {
                return fail(
                    reason,
                    "articulated visual kinematics failed"
                );
            }
            for (const ArticulatedBodyKinematics& body :
                 kinematics) {
                MRBodyStateGPU state{};
                state.position = {
                    static_cast<float>(
                        body.centerOfMassPosition[0]
                    ),
                    static_cast<float>(
                        body.centerOfMassPosition[1]
                    ),
                    static_cast<float>(
                        body.centerOfMassPosition[2]
                    ),
                    1.0f,
                };
                state.orientation = {
                    static_cast<float>(body.orientation[0]),
                    static_cast<float>(body.orientation[1]),
                    static_cast<float>(body.orientation[2]),
                    static_cast<float>(body.orientation[3]),
                };
                state.linearVelocityAndInverseMass = {
                    static_cast<float>(body.linearVelocity[0]),
                    static_cast<float>(body.linearVelocity[1]),
                    static_cast<float>(body.linearVelocity[2]),
                    model.bodies[body.bodyIndex].
                        massAndInverseMass.y,
                };
                state.angularVelocity = {
                    static_cast<float>(body.angularVelocity[0]),
                    static_cast<float>(body.angularVelocity[1]),
                    static_cast<float>(body.angularVelocity[2]),
                    0.0f,
                };
                state.flagsAndIndices[0] =
                    model.bodies[body.bodyIndex].motionType;
                state.flagsAndIndices[1] =
                    model.bodies[body.bodyIndex].articulationIndex;
                state.flagsAndIndices[2] = body.bodyIndex;
                candidate[
                    environment * model.bodies.size() +
                    body.bodyIndex
                ] = state;
            }
        }
        for (std::size_t local = 0u;
             local < sceneIndices.size();
             ++local) {
            MRBodyStateGPU state =
                sceneBodies[
                    environment * sceneIndices.size() + local
                ];
            if (!finite(state.position) ||
                !unitQuaternion(state.orientation) ||
                !finite(state.linearVelocityAndInverseMass) ||
                !finite(state.angularVelocity)) {
                return fail(
                    reason,
                    "visual scene-body state is nonfinite"
                );
            }
            state.flagsAndIndices[2] = sceneIndices[local];
            candidate[
                environment * model.bodies.size() +
                sceneIndices[local]
            ] = state;
        }
    }
    output = std::move(candidate);
    return true;
}

bool VisualDeviceBufferViewV1::valid() const noexcept {
    constexpr std::uint32_t kKnownModalities =
        (MR_VISUAL_MODALITY_OBJECT_POSE << 1u) - 1u;
    std::size_t elementBytes = 0u;
    switch (format) {
    case MR_VISUAL_FORMAT_RGBA32_FLOAT:
    case MR_VISUAL_FORMAT_RGBA32_UINT:
        elementBytes = 16u;
        break;
    case MR_VISUAL_FORMAT_RG32_FLOAT:
        elementBytes = 8u;
        break;
    case MR_VISUAL_FORMAT_R32_FLOAT:
    case MR_VISUAL_FORMAT_R32_UINT:
        elementBytes = 4u;
        break;
    case MR_VISUAL_FORMAT_R8_UINT:
        elementBytes = 1u;
        break;
    case MR_VISUAL_FORMAT_UNKNOWN:
    default:
        return false;
    }
    return handle != 0u && sizeBytes != 0u &&
        offsetBytes <=
            std::numeric_limits<std::size_t>::max() - sizeBytes &&
        offsetBytes % elementBytes == 0u &&
        sizeBytes % elementBytes == 0u &&
        storage > MR_VISUAL_STORAGE_HOST &&
        storage <= MR_VISUAL_STORAGE_COREML_TENSOR &&
        format <= MR_VISUAL_FORMAT_RG32_FLOAT &&
        modality != 0u &&
        (modality & (modality - 1u)) == 0u &&
        (modality & ~kKnownModalities) == 0u;
}

std::size_t VisualFrameBatchV1::pixelCount() const noexcept {
    std::size_t result = 0u;
    return checkedPixels(
               environmentCount,
               viewCount,
               width,
               height,
               result
           )
        ? result
        : 0u;
}

bool VisualFrameBatchV1::valid(std::string* reason) const {
    const std::size_t pixels = pixelCount();
    if (schemaVersion != kVisualFrameBatchVersion ||
        source > MR_VISUAL_SOURCE_REPLAY ||
        environmentCount == 0u || viewCount == 0u ||
        width == 0u || height == 0u || pixels == 0u ||
        episodeTwinFingerprint == 0u ||
        scenarioFingerprint == 0u ||
        rendererFingerprint == 0u ||
        sensorProfileFingerprint == 0u ||
        calibrationFingerprint == 0u ||
        modalities !=
            (MR_VISUAL_MODALITY_RGB |
             MR_VISUAL_MODALITY_DEPTH |
             MR_VISUAL_MODALITY_DEPTH_VALIDITY) ||
        cameras.size() !=
            static_cast<std::size_t>(environmentCount) *
                viewCount) {
        return fail(reason, "visual frame dimensions are invalid");
    }
    const VisualCameraFrameV1& referenceCamera = cameras.front();
    std::unordered_set<std::string> sensorIds;
    for (std::uint32_t view = 0u; view < viewCount; ++view) {
        if (!sensorIds.insert(cameras[view].sensorId).second) {
            return fail(
                reason,
                "visual sensor ids must be unique across views"
            );
        }
    }
    for (std::size_t cameraIndex = 0u;
         cameraIndex < cameras.size();
         ++cameraIndex) {
        const VisualCameraFrameV1& camera = cameras[cameraIndex];
        const std::uint32_t view =
            static_cast<std::uint32_t>(cameraIndex % viewCount);
        const std::size_t environment =
            cameraIndex / viewCount;
        const VisualCameraFrameV1& environmentReference =
            cameras[environment * viewCount];
        if (camera.sensorId.empty() ||
            camera.sensorId != cameras[view].sensorId ||
            camera.frameIndex != referenceCamera.frameIndex ||
            !finite(camera.intrinsics) ||
            !(camera.intrinsics.x > 0.0f) ||
            !(camera.intrinsics.y > 0.0f) ||
            !finite(camera.distortion) ||
            !finite(camera.baseFromCamera.position) ||
            !unitQuaternion(camera.baseFromCamera.orientation) ||
            !finite(camera.captureTimestampSeconds) ||
            !finite(camera.frameAgeSeconds) ||
            camera.frameAgeSeconds < 0.0 ||
            std::abs(
                (
                    camera.captureTimestampSeconds +
                    camera.frameAgeSeconds
                ) -
                (
                    environmentReference.captureTimestampSeconds +
                    environmentReference.frameAgeSeconds
                )
            ) > 1.0e-6 ||
            !finite(camera.exposureSeconds) ||
            camera.exposureSeconds < 0.0 ||
            !finite(camera.shutterReadoutSeconds) ||
            camera.shutterReadoutSeconds < 0.0) {
            return fail(reason, "visual camera metadata is invalid");
        }
    }
    const auto validStorage =
        [this, pixels](
            const std::uint32_t modality,
            const std::size_t hostCount
        ) {
            if ((modalities & modality) == 0u) {
                return hostCount == 0u;
            }
            return hostCount == pixels ||
                (hostCount == 0u &&
                 hasDeviceModality(deviceBuffers, modality));
        };
    if (!validStorage(
            MR_VISUAL_MODALITY_RGB,
            rgbLinear.size()
        ) ||
        !validStorage(
            MR_VISUAL_MODALITY_DEPTH,
            depthMeters.size()
        ) ||
        !validStorage(
            MR_VISUAL_MODALITY_DEPTH_VALIDITY,
            depthValidity.size()
        )) {
        return fail(
            reason,
            "visual modality storage does not match the frame contract"
        );
    }
    constexpr std::array kRGBA32Float{
        MR_VISUAL_FORMAT_RGBA32_FLOAT,
    };
    constexpr std::array kR32Float{
        MR_VISUAL_FORMAT_R32_FLOAT,
    };
    constexpr std::array kDepthValidityFormats{
        MR_VISUAL_FORMAT_R8_UINT,
        MR_VISUAL_FORMAT_R32_UINT,
    };
    if ((rgbLinear.empty() &&
         !hasDeviceStorage(
             deviceBuffers,
             MR_VISUAL_MODALITY_RGB,
             pixels,
             kRGBA32Float
         )) ||
        (depthMeters.empty() &&
         !hasDeviceStorage(
             deviceBuffers,
             MR_VISUAL_MODALITY_DEPTH,
             pixels,
             kR32Float
         )) ||
        (depthValidity.empty() &&
         !hasDeviceStorage(
             deviceBuffers,
             MR_VISUAL_MODALITY_DEPTH_VALIDITY,
             pixels,
             kDepthValidityFormats
         ))) {
        return fail(
            reason,
            "visual device storage has the wrong format or byte size"
        );
    }
    if (!std::ranges::all_of(
            rgbLinear,
            [](const mr_float4 value) {
                return finite(value);
            }
        )) {
        return fail(reason, "visual RGB contains nonfinite values");
    }
    if (!std::ranges::all_of(
            depthMeters,
            [](const float value) {
                return std::isfinite(value);
            }
        )) {
        return fail(reason, "visual depth contains invalid values");
    }
    if (!std::ranges::all_of(
            deviceBuffers,
            [this](const VisualDeviceBufferViewV1& buffer) {
                return buffer.valid() &&
                    (modalities & buffer.modality) != 0u;
            }
        )) {
        return fail(reason, "visual device buffer view is invalid");
    }
    if (!std::ranges::all_of(
            depthValidity,
            [](const std::uint8_t value) {
                return value <= 1u;
            }
        )) {
        return fail(reason, "visual depth validity values are invalid");
    }
    if (!depthMeters.empty()) {
        for (std::size_t index = 0u;
             index < depthMeters.size();
             ++index) {
            if (!depthValidity.empty() &&
                depthValidity[index] != 0u &&
                (!std::isfinite(depthMeters[index]) ||
                 !(depthMeters[index] > 0.0f))) {
                return fail(
                    reason,
                    "valid visual depth must be finite and positive"
                );
            }
            if (!depthValidity.empty() &&
                depthValidity[index] == 0u &&
                depthMeters[index] != 0.0f) {
                return fail(
                    reason,
                    "invalid visual depth must use the zero sentinel"
                );
            }
        }
    }
    if (!depthValidity.empty()) {
        const std::size_t pixelsPerCamera =
            static_cast<std::size_t>(width) * height;
        for (std::size_t cameraIndex = 0u;
             cameraIndex < cameras.size();
             ++cameraIndex) {
            if (cameras[cameraIndex].valid) {
                continue;
            }
            if (std::ranges::any_of(
                    std::span<const std::uint8_t>{
                        depthValidity
                    }.subspan(
                        cameraIndex * pixelsPerCamera,
                        pixelsPerCamera
                    ),
                    [](const std::uint8_t value) {
                        return value != 0u;
                    }
                )) {
                return fail(
                    reason,
                    "invalid camera publishes valid depth samples"
                );
            }
        }
    }
    return true;
}

bool VisualTruthBatchV1::valid(std::string* reason) const {
    std::size_t pixels = 0u;
    constexpr std::uint32_t kKnownModalities =
        (MR_VISUAL_MODALITY_OBJECT_POSE << 1u) - 1u;
    constexpr std::uint32_t kRequiredModalities =
        MR_VISUAL_MODALITY_NORMAL |
        MR_VISUAL_MODALITY_MOTION |
        MR_VISUAL_MODALITY_SEMANTIC |
        MR_VISUAL_MODALITY_INSTANCE |
        MR_VISUAL_MODALITY_LINK;
    if (schemaVersion != kVisualFrameBatchVersion ||
        coordinateFrame > MR_VISUAL_FRAME_OBJECT ||
        (modalities & kRequiredModalities) !=
            kRequiredModalities ||
        (modalities & ~kKnownModalities) != 0u ||
        !finite(timestampSeconds) ||
        !checkedPixels(
            environmentCount,
            viewCount,
            width,
            height,
            pixels
        ) ||
        pixels == 0u) {
        return fail(reason, "visual truth dimensions are invalid");
    }
    if (objectPoses.size() % environmentCount != 0u ||
        linkPoses.size() % environmentCount != 0u ||
        keypoints.size() % environmentCount != 0u ||
        contacts.size() % environmentCount != 0u) {
        return fail(
            reason,
            "visual sparse truth is not environment-major"
        );
    }
    const auto validStorage =
        [this, pixels](
            const std::uint32_t modality,
            const std::size_t hostCount
        ) {
            if ((modalities & modality) == 0u) {
                return hostCount == 0u;
            }
            return hostCount == pixels ||
                (hostCount == 0u &&
                 hasDeviceModality(deviceBuffers, modality));
        };
    if (!validStorage(MR_VISUAL_MODALITY_NORMAL, normals.size()) ||
        !validStorage(MR_VISUAL_MODALITY_MOTION, motion.size()) ||
        !validStorage(
            MR_VISUAL_MODALITY_SEMANTIC,
            semanticIds.size()
        ) ||
        !validStorage(
            MR_VISUAL_MODALITY_INSTANCE,
            instanceIds.size()
        ) ||
        !validStorage(MR_VISUAL_MODALITY_LINK, linkIds.size()) ||
        visibility.size() != pixels ||
        occlusion.size() != pixels) {
        return fail(reason, "visual truth storage is incomplete");
    }
    constexpr std::array kRGBA32Float{
        MR_VISUAL_FORMAT_RGBA32_FLOAT,
    };
    constexpr std::array kR32Uint{
        MR_VISUAL_FORMAT_R32_UINT,
    };
    if ((normals.empty() &&
         (modalities & MR_VISUAL_MODALITY_NORMAL) != 0u &&
         !hasDeviceStorage(
             deviceBuffers,
             MR_VISUAL_MODALITY_NORMAL,
             pixels,
             kRGBA32Float
         )) ||
        (motion.empty() &&
         (modalities & MR_VISUAL_MODALITY_MOTION) != 0u &&
         !hasDeviceStorage(
             deviceBuffers,
             MR_VISUAL_MODALITY_MOTION,
             pixels,
             kRGBA32Float
         )) ||
        (semanticIds.empty() &&
         (modalities & MR_VISUAL_MODALITY_SEMANTIC) != 0u &&
         !hasDeviceStorage(
             deviceBuffers,
             MR_VISUAL_MODALITY_SEMANTIC,
             pixels,
             kR32Uint
         )) ||
        (instanceIds.empty() &&
         (modalities & MR_VISUAL_MODALITY_INSTANCE) != 0u &&
         !hasDeviceStorage(
             deviceBuffers,
             MR_VISUAL_MODALITY_INSTANCE,
             pixels,
             kR32Uint
         )) ||
        (linkIds.empty() &&
         (modalities & MR_VISUAL_MODALITY_LINK) != 0u &&
         !hasDeviceStorage(
             deviceBuffers,
             MR_VISUAL_MODALITY_LINK,
             pixels,
             kR32Uint
         ))) {
        return fail(
            reason,
            "visual truth device storage has the wrong format or byte size"
        );
    }
    if (!std::ranges::all_of(
            normals,
            [](const mr_float4 value) {
                return finite(value);
            }
        ) ||
        !std::ranges::all_of(
            motion,
            [](const mr_float4 value) {
                return finite(value);
            }
        )) {
        return fail(reason, "visual truth contains nonfinite values");
    }
    for (const MRVisualKeypointGPU& keypoint : keypoints) {
        if (!finite(keypoint.positionAndVisibility) ||
            keypoint.positionAndVisibility.w < 0.0f ||
            keypoint.positionAndVisibility.w > 1.0f ||
            keypoint.identity.x == 0u ||
            keypoint.identity.x == MR_INVALID_INDEX ||
            keypoint.identity.y == 0u ||
            keypoint.identity.y == MR_INVALID_INDEX) {
            return fail(reason, "visual keypoint is invalid");
        }
    }
    const auto validPose = [](const MRVisualPoseGPU& pose) {
        return finite(pose.position) &&
            pose.position.w == 1.0f &&
            unitQuaternion(pose.orientation);
    };
    if (!std::ranges::all_of(
            objectPoses,
            [&validPose](const MRVisualPoseGPU& pose) {
                return validPose(pose) &&
                    pose.identity.x != 0u &&
                    pose.identity.x != MR_INVALID_INDEX &&
                    pose.identity.y != 0u &&
                    pose.identity.y != MR_INVALID_INDEX;
            }
        ) ||
        !std::ranges::all_of(
            linkPoses,
            [&validPose](const MRVisualPoseGPU& pose) {
                return validPose(pose) &&
                    pose.identity.z != MR_INVALID_INDEX;
            }
        )) {
        return fail(reason, "visual object/link pose is invalid");
    }
    for (const MRVisualContactAnnotationGPU& contact : contacts) {
        if (!finite(contact.positionAndImpulse) ||
            !finite(contact.normalAndSeparation) ||
            contact.positionAndImpulse.w < 0.0f ||
            std::abs(
                (
                    contact.normalAndSeparation.x *
                        contact.normalAndSeparation.x +
                    contact.normalAndSeparation.y *
                        contact.normalAndSeparation.y +
                    contact.normalAndSeparation.z *
                        contact.normalAndSeparation.z
                ) -
                1.0f
            ) > 1.0e-4f ||
            contact.identity.x == 0u ||
            contact.identity.x == MR_INVALID_INDEX ||
            contact.identity.y == 0u ||
            contact.identity.y == MR_INVALID_INDEX) {
            return fail(reason, "visual contact annotation is invalid");
        }
    }
    if (!finiteValues(visibility) ||
        !std::ranges::all_of(
            visibility,
            [](const float value) {
                return value >= 0.0f && value <= 1.0f;
            }
        ) ||
        !std::ranges::all_of(
            deviceBuffers,
            [this](const VisualDeviceBufferViewV1& buffer) {
                return buffer.valid() &&
                    (modalities & buffer.modality) != 0u;
            }
        )) {
        return fail(reason, "visual truth visibility/storage is invalid");
    }
    for (std::size_t index = 0u;
         index < visibility.size();
         ++index) {
        const int expected = static_cast<int>(
            std::lround(
                (1.0f - visibility[index]) * 255.0f
            )
        );
        if (std::abs(
                static_cast<int>(occlusion[index]) -
                expected
            ) > 1) {
            return fail(
                reason,
                "visual visibility and occlusion disagree"
            );
        }
    }
    return true;
}

std::size_t PerceptionTensorV1::elementCount() const noexcept {
    std::size_t result = 1u;
    if (shape.empty()) {
        return 0u;
    }
    for (const std::uint32_t dimension : shape) {
        if (dimension == 0u) {
            return 0u;
        }
        if (result >
                std::numeric_limits<std::size_t>::max() /
                    dimension) {
            return 0u;
        }
        result *= dimension;
    }
    return result;
}

bool PerceptionTensorV1::validContract(std::string* reason) const {
    const std::size_t count = elementCount();
    constexpr std::uint32_t kKnownModalities =
        (MR_VISUAL_MODALITY_OBJECT_POSE << 1u) - 1u;
    bool shapeOverflow = shape.empty();
    if (!shape.empty()) {
        std::size_t checkedCount = 1u;
        shapeOverflow = false;
        for (const std::uint32_t dimension : shape) {
            if (dimension == 0u) {
                checkedCount = 0u;
                continue;
            }
            if (checkedCount != 0u &&
                checkedCount >
                    std::numeric_limits<std::size_t>::max() /
                        dimension) {
                shapeOverflow = true;
                break;
            }
            checkedCount *= dimension;
        }
    }
    if (id.empty() || modality == 0u ||
        (modality & (modality - 1u)) != 0u ||
        (modality & ~kKnownModalities) != 0u ||
        coordinateFrame > MR_VISUAL_FRAME_OBJECT ||
        elementType > PerceptionElementType::uint8 ||
        shapeOverflow || !finite(timestampSeconds) ||
        !finite(confidence) || confidence < 0.0f ||
        confidence > 1.0f) {
        return fail(reason, "perception tensor identity is invalid");
    }
    std::size_t hostCount = 0u;
    std::size_t elementBytes = 0u;
    MRVisualPixelFormat scalarFormat =
        MR_VISUAL_FORMAT_UNKNOWN;
    switch (elementType) {
    case PerceptionElementType::float32:
        if (!uintValues.empty() || !byteValues.empty()) {
            return fail(
                reason,
                "perception tensor has conflicting host storage"
            );
        }
        hostCount = floatValues.size();
        elementBytes = sizeof(float);
        scalarFormat = MR_VISUAL_FORMAT_R32_FLOAT;
        break;
    case PerceptionElementType::uint32:
        if (!floatValues.empty() || !byteValues.empty()) {
            return fail(
                reason,
                "perception tensor has conflicting host storage"
            );
        }
        hostCount = uintValues.size();
        elementBytes = sizeof(std::uint32_t);
        scalarFormat = MR_VISUAL_FORMAT_R32_UINT;
        break;
    case PerceptionElementType::uint8:
        if (!floatValues.empty() || !uintValues.empty()) {
            return fail(
                reason,
                "perception tensor has conflicting host storage"
            );
        }
        hostCount = byteValues.size();
        elementBytes = sizeof(std::uint8_t);
        scalarFormat = MR_VISUAL_FORMAT_R8_UINT;
        break;
    }
    if (hostCount != 0u && hostCount != count) {
        return fail(reason, "perception tensor host shape does not match");
    }
    if (count == 0u &&
        (hostCount != 0u || !deviceBuffers.empty())) {
        return fail(
            reason,
            "empty perception tensor must not carry storage"
        );
    }
    if (count != 0u && hostCount == 0u &&
        deviceBuffers.empty()) {
        return fail(reason, "perception tensor has no storage");
    }
    if (!finiteValues(floatValues)) {
        return fail(reason, "perception tensor contains nonfinite values");
    }
    if (!std::ranges::all_of(
            deviceBuffers,
            [&](const VisualDeviceBufferViewV1& buffer) {
                return buffer.valid() &&
                    buffer.modality == modality &&
                    buffer.format == scalarFormat &&
                    count <=
                        std::numeric_limits<std::size_t>::max() /
                            elementBytes &&
                    buffer.sizeBytes >= count * elementBytes;
            }
        )) {
        return fail(reason, "perception device buffer is invalid");
    }
    if (count != 0u && hostCount == 0u) {
        const auto storage = std::ranges::find_if(
            deviceBuffers,
            [&](const VisualDeviceBufferViewV1& buffer) {
                return buffer.modality == modality &&
                    buffer.format == scalarFormat &&
                    count <=
                        std::numeric_limits<std::size_t>::max() /
                            elementBytes &&
                    buffer.sizeBytes >= count * elementBytes;
            }
        );
        if (storage == deviceBuffers.end()) {
            return fail(
                reason,
                "perception device tensor format or byte size is invalid"
            );
        }
    }
    return true;
}

bool PerceptionProviderDescriptorV1::valid(
    std::string* reason
) const {
    constexpr std::uint32_t kKnownModalities =
        (MR_VISUAL_MODALITY_OBJECT_POSE << 1u) - 1u;
    constexpr std::uint32_t kFrameModalities =
        MR_VISUAL_MODALITY_RGB |
        MR_VISUAL_MODALITY_DEPTH |
        MR_VISUAL_MODALITY_DEPTH_VALIDITY;
    constexpr std::uint32_t kKnownCapabilities =
        (MR_PERCEPTION_CAP_EMBEDDING << 1u) - 1u;
    if (schemaVersion != kPerceptionProviderVersion ||
        id.empty() || contentHash.empty() ||
        inputModalities == 0u || capabilities == 0u ||
        temporalWindow == 0u ||
        (inputModalities & ~kKnownModalities) != 0u ||
        (inputModalities & ~kFrameModalities) != 0u ||
        (capabilities & ~kKnownCapabilities) != 0u) {
        return fail(reason, "perception provider descriptor is invalid");
    }
    return true;
}

const PerceptionTensorV1* PerceptionResultBatchV1::tensor(
    const std::uint32_t modality
) const noexcept {
    const auto found = std::ranges::find_if(
        tensors,
        [modality](const PerceptionTensorV1& value) {
            return value.modality == modality && value.valid;
        }
    );
    return found == tensors.end() ? nullptr : &*found;
}

bool PerceptionResultBatchV1::valid(std::string* reason) const {
    if (providerId.empty() || providerContentHash.empty() ||
        !finite(timestampSeconds)) {
        return fail(reason, "perception result identity is invalid");
    }
    std::unordered_set<std::string> ids;
    for (const PerceptionTensorV1& value : tensors) {
        if (!value.validContract(reason)) {
            return false;
        }
        if (!ids.insert(value.id).second) {
            return fail(reason, "perception tensor ids are not unique");
        }
        if (std::abs(value.timestampSeconds - timestampSeconds) >
            1.0e-6) {
            return fail(
                reason,
                "perception tensor timestamp does not match its batch"
            );
        }
    }
    return true;
}

bool PolicyObservationBatchV1::valid(std::string* reason) const {
    if (environmentCount == 0u || deployableWidth == 0u ||
        deployable.size() !=
            static_cast<std::size_t>(environmentCount) *
                deployableWidth ||
        privileged.size() !=
            static_cast<std::size_t>(environmentCount) *
                privilegedWidth ||
        !finite(timestampSeconds) ||
        !finiteValues(deployable) ||
        !finiteValues(privileged)) {
        return fail(reason, "policy observation batch is invalid");
    }
    return true;
}

bool PolicyObservationAssemblerV1::assemble(
    const VisualFrameBatchV1& frames,
    const PerceptionResultBatchV1* perception,
    const PolicyObservationRequestV1& request,
    PolicyObservationBatchV1& output,
    std::string* reason
) const {
    std::string frameReason;
    if (!frames.valid(&frameReason) ||
        request.environmentCount != frames.environmentCount ||
        request.profile > ObservationProfileV1::compactLatent) {
        return fail(
            reason,
            "policy frames or environment count are invalid"
        );
    }
    const auto validPacked =
        [count = request.environmentCount](
            const std::span<const float> values,
            const std::uint32_t width
        ) {
            return values.size() ==
                    static_cast<std::size_t>(count) * width &&
                finiteValues(values);
        };
    if (!validPacked(
            request.proprioception,
            request.proprioceptionWidth
        ) ||
        !validPacked(
            request.previousActions,
            request.previousActionWidth
        ) ||
        !validPacked(
            request.taskCommands,
            request.taskCommandWidth
        ) ||
        !validPacked(
            request.privilegedState,
            request.privilegedStateWidth
        )) {
        return fail(
            reason,
            "policy side-channel dimensions are invalid"
        );
    }

    const std::size_t pixelsPerEnvironment =
        static_cast<std::size_t>(frames.viewCount) *
        frames.width * frames.height;
    std::uint32_t visualWidth = 0u;
    const PerceptionTensorV1* selected = nullptr;
    if (request.profile == ObservationProfileV1::rawRGBD) {
        if (frames.rgbLinear.empty() ||
            frames.depthMeters.empty() ||
            frames.depthValidity.empty()) {
            return fail(
                reason,
                "raw RGB-D assembly requires host-visible frame storage"
            );
        }
        if (pixelsPerEnvironment >
            std::numeric_limits<std::uint32_t>::max() / 5u) {
            return fail(reason, "raw RGB-D observation width overflows");
        }
        visualWidth =
            static_cast<std::uint32_t>(pixelsPerEnvironment * 5u);
    } else if (request.profile == ObservationProfileV1::rgbXYZ) {
        if (frames.rgbLinear.empty() ||
            frames.depthMeters.empty() ||
            frames.depthValidity.empty()) {
            return fail(
                reason,
                "RGB-XYZ assembly requires host-visible frame storage"
            );
        }
        if (pixelsPerEnvironment >
            std::numeric_limits<std::uint32_t>::max() / 7u) {
            return fail(reason, "RGB-XYZ observation width overflows");
        }
        visualWidth =
            static_cast<std::uint32_t>(pixelsPerEnvironment * 7u);
    } else {
        if (perception == nullptr || !perception->valid()) {
            return fail(
                reason,
                "selected observation profile requires perception results"
            );
        }
        const VisualCameraFrameV1& reference =
            frames.cameras.front();
        if (perception->frameIndex != reference.frameIndex ||
            std::abs(
                perception->timestampSeconds -
                reference.captureTimestampSeconds
            ) > 1.0e-6) {
            return fail(
                reason,
                "perception result is stale or belongs to another frame"
            );
        }
        const std::uint32_t modality =
            request.profile == ObservationProfileV1::objectCentric
                ? MR_VISUAL_MODALITY_OBJECT_POSE
                : MR_VISUAL_MODALITY_FEATURE;
        selected = perception->tensor(modality);
        if (selected == nullptr ||
            selected->elementType !=
                PerceptionElementType::float32 ||
            (selected->elementCount() != 0u &&
             selected->floatValues.empty()) ||
            selected->shape.empty() ||
            selected->shape.front() !=
                request.environmentCount ||
            selected->floatValues.size() %
                    request.environmentCount !=
                0u) {
            return fail(
                reason,
                "perception result does not match observation profile"
            );
        }
        if (selected->floatValues.size() /
                request.environmentCount >
            std::numeric_limits<std::uint32_t>::max()) {
            return fail(
                reason,
                "perception observation width overflows"
            );
        }
        visualWidth = static_cast<std::uint32_t>(
            selected->floatValues.size() /
            request.environmentCount
        );
    }
    const std::uint64_t deployableWidth64 =
        static_cast<std::uint64_t>(visualWidth) +
        request.proprioceptionWidth +
        request.previousActionWidth +
        request.taskCommandWidth;
    if (deployableWidth64 >
        std::numeric_limits<std::uint32_t>::max()) {
        return fail(reason, "policy observation width overflows");
    }

    PolicyObservationBatchV1 candidate;
    candidate.profile = request.profile;
    candidate.environmentCount = request.environmentCount;
    candidate.deployableWidth =
        static_cast<std::uint32_t>(deployableWidth64);
    candidate.privilegedWidth =
        request.privilegedStateWidth;
    candidate.deployable.resize(
        static_cast<std::size_t>(candidate.environmentCount) *
        candidate.deployableWidth
    );
    candidate.privileged.assign(
        request.privilegedState.begin(),
        request.privilegedState.end()
    );
    candidate.frameIndex = frames.cameras.front().frameIndex;
    candidate.timestampSeconds =
        frames.cameras.front().captureTimestampSeconds;

    for (std::uint32_t environment = 0u;
         environment < candidate.environmentCount;
         ++environment) {
        std::size_t destination =
            static_cast<std::size_t>(environment) *
            candidate.deployableWidth;
        if (selected != nullptr) {
            if (visualWidth != 0u) {
                const std::size_t source =
                    static_cast<std::size_t>(environment) *
                    visualWidth;
                std::ranges::copy(
                    std::span{
                        selected->floatValues.data() + source,
                        visualWidth,
                    },
                    candidate.deployable.begin() + destination
                );
            }
            destination += visualWidth;
        } else {
            const std::size_t pixelBase =
                static_cast<std::size_t>(environment) *
                pixelsPerEnvironment;
            for (std::uint32_t view = 0u;
                 view < frames.viewCount;
                 ++view) {
                const VisualCameraFrameV1& camera =
                    frames.cameras[
                        static_cast<std::size_t>(environment) *
                            frames.viewCount +
                        view
                    ];
                for (std::uint32_t y = 0u; y < frames.height; ++y) {
                    for (std::uint32_t x = 0u;
                         x < frames.width;
                         ++x) {
                        const std::size_t pixel =
                            pixelBase +
                            static_cast<std::size_t>(view) *
                                frames.width * frames.height +
                            static_cast<std::size_t>(y) *
                                frames.width +
                            x;
                        const mr_float4 rgb = frames.rgbLinear[pixel];
                        candidate.deployable[destination++] = rgb.x;
                        candidate.deployable[destination++] = rgb.y;
                        candidate.deployable[destination++] = rgb.z;
                        const float depth = frames.depthMeters[pixel];
                        const float valid =
                            frames.depthValidity[pixel] != 0u
                                ? 1.0f
                                : 0.0f;
                        if (request.profile ==
                            ObservationProfileV1::rawRGBD) {
                            candidate.deployable[destination++] =
                                valid != 0.0f ? depth : 0.0f;
                            candidate.deployable[destination++] = valid;
                            continue;
                        }
                        mr_float4 cameraPoint{};
                        bool projectable = valid != 0.0f;
                        if (projectable) {
                            const NormalizedPoint normalized =
                                undistortNormalized(
                                    (
                                        (static_cast<float>(x) + 0.5f) -
                                        camera.intrinsics.z
                                    ) / camera.intrinsics.x,
                                    (
                                        (static_cast<float>(y) + 0.5f) -
                                        camera.intrinsics.w
                                    ) / camera.intrinsics.y,
                                    camera.distortion
                                );
                            projectable = normalized.valid;
                            cameraPoint = {
                                normalized.x * depth,
                                normalized.y * depth,
                                depth,
                                1.0f,
                            };
                        }
                        const mr_float4 basePoint = add3(
                            camera.baseFromCamera.position,
                            rotateVector(
                                camera.baseFromCamera.orientation,
                                cameraPoint
                            )
                        );
                        candidate.deployable[destination++] =
                            projectable ? basePoint.x : 0.0f;
                        candidate.deployable[destination++] =
                            projectable ? basePoint.y : 0.0f;
                        candidate.deployable[destination++] =
                            projectable ? basePoint.z : 0.0f;
                        candidate.deployable[destination++] =
                            projectable ? 1.0f : 0.0f;
                    }
                }
            }
        }
        const auto appendSide =
            [&candidate, &destination, environment](
                const std::span<const float> values,
                const std::uint32_t width
            ) {
                const std::size_t source =
                    static_cast<std::size_t>(environment) * width;
                std::ranges::copy(
                    values.subspan(source, width),
                    candidate.deployable.begin() + destination
                );
                destination += width;
            };
        appendSide(
            request.proprioception,
            request.proprioceptionWidth
        );
        appendSide(
            request.previousActions,
            request.previousActionWidth
        );
        appendSide(
            request.taskCommands,
            request.taskCommandWidth
        );
    }
    std::string candidateReason;
    if (!candidate.valid(&candidateReason)) {
        return fail(reason, candidateReason);
    }
    output = std::move(candidate);
    return true;
}

bool VisualEpisodeStreamV1::append(
    VisualEpisodeStepV1 step,
    std::string* reason
) {
    if (!finite(step.timestampSeconds) ||
        !finite(step.reward) ||
        !finiteValues(step.proprioception) ||
        !finiteValues(step.action) ||
        !finiteValues(step.taskCommand) ||
        !finiteValues(step.privilegedState) ||
        step.scenarioKey == 0u ||
        step.frameContentHash.empty()) {
        return fail(reason, "visual episode step is invalid");
    }
    if (!steps.empty() &&
        (step.frameIndex <= steps.back().frameIndex ||
         step.timestampSeconds < steps.back().timestampSeconds ||
         step.scenarioKey != steps.front().scenarioKey)) {
        return fail(
            reason,
            "visual episode steps must be monotonic and scenario-stable"
        );
    }
    steps.push_back(std::move(step));
    fingerprint = 0u;
    return true;
}

bool VisualEpisodeStreamV1::finalize(std::string* reason) {
    if (id.empty() || steps.empty() ||
        episodeTwinFingerprint == 0u ||
        worldFamilyFingerprint == 0u ||
        scenarioFingerprint == 0u ||
        rendererFingerprint == 0u ||
        visualSceneFingerprint == 0u ||
        visualPackFingerprint == 0u ||
        environmentMapFingerprint == 0u ||
        lightRigFingerprint == 0u ||
        rendererProfileFingerprint == 0u ||
        shutterProfileFingerprint == 0u ||
        sensorProfileFingerprint == 0u ||
        calibrationFingerprint == 0u ||
        physicsFingerprint == 0u) {
        return fail(reason, "visual episode identity is incomplete");
    }
    fingerprint = episodeFingerprint(*this);
    return valid(reason);
}

bool VisualEpisodeStreamV1::valid(std::string* reason) const {
    if (schemaVersion != kVisualEpisodeStreamVersion ||
        id.empty() || source > MR_VISUAL_SOURCE_REPLAY ||
        episodeTwinFingerprint == 0u ||
        worldFamilyFingerprint == 0u ||
        scenarioFingerprint == 0u ||
        rendererFingerprint == 0u ||
        visualSceneFingerprint == 0u ||
        visualPackFingerprint == 0u ||
        environmentMapFingerprint == 0u ||
        lightRigFingerprint == 0u ||
        rendererProfileFingerprint == 0u ||
        shutterProfileFingerprint == 0u ||
        sensorProfileFingerprint == 0u ||
        calibrationFingerprint == 0u ||
        physicsFingerprint == 0u ||
        fingerprint == 0u || steps.empty() ||
        fingerprint != episodeFingerprint(*this)) {
        return fail(reason, "visual episode stream identity is invalid");
    }
    for (std::size_t index = 0u; index < steps.size(); ++index) {
        const VisualEpisodeStepV1& step = steps[index];
        if (!finite(step.timestampSeconds) ||
            !finite(step.reward) ||
            !finiteValues(step.proprioception) ||
            !finiteValues(step.action) ||
            !finiteValues(step.taskCommand) ||
            !finiteValues(step.privilegedState) ||
            step.scenarioKey == 0u ||
            step.frameContentHash.empty() ||
            (index != 0u &&
             (step.frameIndex <= steps[index - 1u].frameIndex ||
              step.timestampSeconds <
                  steps[index - 1u].timestampSeconds ||
              step.scenarioKey != steps.front().scenarioKey))) {
            return fail(reason, "visual episode step is invalid");
        }
    }
    return true;
}

bool writeVisualEpisodeManifest(
    const VisualEpisodeStreamV1& stream,
    const std::filesystem::path& path,
    std::string* reason
) {
    std::string streamReason;
    if (!stream.valid(&streamReason) || path.empty()) {
        return fail(
            reason,
            "visual episode manifest input is invalid: " +
                streamReason
        );
    }
    const std::filesystem::path temporary =
        path.string() + ".tmp." + std::to_string(stream.fingerprint);
    std::ofstream output(
        temporary,
        std::ios::binary | std::ios::trunc
    );
    if (!output) {
        return fail(reason, "could not open visual episode manifest");
    }
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema_version\": " << stream.schemaVersion << ",\n"
           << "  \"id\": \"" << jsonEscape(stream.id) << "\",\n"
           << "  \"source\": " << static_cast<std::uint32_t>(stream.source)
           << ",\n"
           << "  \"episode_twin_fingerprint\": "
           << stream.episodeTwinFingerprint << ",\n"
           << "  \"world_family_fingerprint\": "
           << stream.worldFamilyFingerprint << ",\n"
           << "  \"scenario_fingerprint\": "
           << stream.scenarioFingerprint << ",\n"
           << "  \"renderer_fingerprint\": "
           << stream.rendererFingerprint << ",\n"
           << "  \"visual_scene_fingerprint\": "
           << stream.visualSceneFingerprint << ",\n"
           << "  \"visual_pack_fingerprint\": "
           << stream.visualPackFingerprint << ",\n"
           << "  \"environment_map_fingerprint\": "
           << stream.environmentMapFingerprint << ",\n"
           << "  \"light_rig_fingerprint\": "
           << stream.lightRigFingerprint << ",\n"
           << "  \"renderer_profile_fingerprint\": "
           << stream.rendererProfileFingerprint << ",\n"
           << "  \"shutter_profile_fingerprint\": "
           << stream.shutterProfileFingerprint << ",\n"
           << "  \"sensor_profile_fingerprint\": "
           << stream.sensorProfileFingerprint << ",\n"
           << "  \"calibration_fingerprint\": "
           << stream.calibrationFingerprint << ",\n"
           << "  \"physics_fingerprint\": "
           << stream.physicsFingerprint << ",\n"
           << "  \"fingerprint\": " << stream.fingerprint << ",\n"
           << "  \"steps\": [\n";
    for (std::size_t index = 0u;
         index < stream.steps.size();
         ++index) {
        const VisualEpisodeStepV1& step = stream.steps[index];
        output << "    {\"frame_index\":" << step.frameIndex
               << ",\"scenario_key\":" << step.scenarioKey
               << ",\"timestamp_seconds\":" << step.timestampSeconds
               << ",\"reward\":" << step.reward
               << ",\"proprioception\":";
        writeJSONFloatArray(output, step.proprioception);
        output << ",\"action\":";
        writeJSONFloatArray(output, step.action);
        output << ",\"task_command\":";
        writeJSONFloatArray(output, step.taskCommand);
        output << ",\"privileged_state\":";
        writeJSONFloatArray(output, step.privilegedState);
        output << ",\"frame_content_hash\":\""
               << jsonEscape(step.frameContentHash)
               << "\",\"truth_content_hash\":\""
               << jsonEscape(step.truthContentHash)
               << "\",\"event_flags\":" << step.eventFlags << '}';
        if (index + 1u != stream.steps.size()) {
            output << ',';
        }
        output << '\n';
    }
    output << "  ]\n}\n";
    output.close();
    if (!output) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return fail(reason, "could not write visual episode manifest");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return fail(reason, "could not publish visual episode manifest");
    }
    return true;
}

} // namespace metalrobo
