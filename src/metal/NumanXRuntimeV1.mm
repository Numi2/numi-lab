#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "NumanXBridgeV1Internal.hpp"

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalNumanXHumanMatter.hpp"
#include "metalrobo/VisualPresentation.hpp"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(sizeof(MRNumanXHumanSupportConsequenceGPU) ==
              sizeof(NMHumanSupportConsequenceGPU));
static_assert(alignof(MRNumanXHumanSupportConsequenceGPU) ==
              alignof(NMHumanSupportConsequenceGPU));
static_assert(sizeof(NMHumanSupportPointQueryGPU) ==
              sizeof(MRArticulatedPointImpulseGPU));
static_assert(alignof(NMHumanSupportPointQueryGPU) ==
              alignof(MRArticulatedPointImpulseGPU));

using metalrobo::numanx_bridge_v1::DomainPtr;
using metalrobo::numanx_bridge_v1::PreparedTerminalDisposition;

constexpr std::array<char, 8u> kRigidMagic{
    'N', 'H', 'R', 'I', 'G', 'I', 'D', '2'};
constexpr std::array<char, 8u> kLegacyMuscleMagic{
    'N', 'H', 'M', 'Y', 'O', '1', '\0', '\0'};
constexpr std::array<char, 8u> kMuscleMagic{
    'N', 'H', 'M', 'Y', 'O', '2', '\0', '\0'};
constexpr std::array<char, 8u> kSupportContactMagic{
    'N', 'H', 'C', 'N', 'T', '1', '\0', '\0'};
constexpr std::uint32_t kRigidABI = 1u;
constexpr std::uint32_t kLegacyMuscleABI = 1u;
constexpr std::uint32_t kMuscleABI = 2u;
constexpr std::uint32_t kSupportContactABI = 1u;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

#pragma pack(push, 1)
struct RigidHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadABI = 0u;
    std::uint32_t engineABI = 0u;
    std::uint32_t sourceBodyCount = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t jointCount = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t rootBodyIndex = 0u;
    std::uint32_t virtualBodyCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSHA256{};
};

struct SourcePoseRecord {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float quaternionX = 0.0f;
    float quaternionY = 0.0f;
    float quaternionZ = 0.0f;
    float quaternionW = 1.0f;
};

struct MuscleHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadABI = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t muscleCount = 0u;
    std::uint32_t siteCount = 0u;
    std::uint32_t wrapCount = 0u;
    std::uint32_t routeNodeCount = 0u;
    std::uint32_t sourceTendonCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::array<std::uint8_t, 32u> sourceSHA256{};
};

struct SiteRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct WrapRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t type = 0u;
    float radius = 0.0f;
    float reserved0 = 0.0f;
    float centerX = 0.0f;
    float centerY = 0.0f;
    float centerZ = 0.0f;
    float rotation[9]{};
};

struct RouteRecord {
    std::uint32_t type = 0u;
    std::uint32_t targetIndex = MR_INVALID_INDEX;
    std::uint32_t sideSiteIndex = MR_INVALID_INDEX;
    std::uint32_t reserved0 = 0u;
};

struct MuscleRecord {
    std::uint32_t sourceTendonIndex = 0u;
    std::uint32_t routeOffset = 0u;
    std::uint32_t routeCount = 0u;
    std::uint32_t reserved0 = 0u;
    float values[37]{};
};

struct MuscleArchitectureRecord {
    float optimalFiberLength = 0.0f;
    float tendonSlackLength = 0.0f;
    float tendonStrainAtOneNormalizedForce = 0.0f;
    float tendonStiffnessAtOneNormalizedForce = 0.0f;
    float tendonNormalizedForceAtToeEnd = 0.0f;
    float tendonCurviness = 0.0f;
    float normalizedFiberDamping = 0.0f;
    float fitNormalizedRmse = 0.0f;
};

struct SupportContactHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadABI = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSHA256{};
    float groundPoint[3]{};
    float groundNormal[3]{};
    float groundFriction = 0.0f;
};

struct SupportContactRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t sourceGeometryIndex = MR_INVALID_INDEX;
    float localPoint[3]{};
    float planeWitness[3]{};
    float friction = 0.0f;
    float defaultSignedDistance = 0.0f;
    float reserved0 = 0.0f;
    float reserved1 = 0.0f;
};
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SupportContactHeader) == 84u);
static_assert(sizeof(SupportContactRecord) == 48u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(MuscleHeader) == 76u);
static_assert(sizeof(SiteRecord) == 16u);
static_assert(sizeof(WrapRecord) == 64u);
static_assert(sizeof(RouteRecord) == 16u);
static_assert(sizeof(MuscleRecord) == 164u);
static_assert(sizeof(MuscleArchitectureRecord) == 32u);
static_assert(sizeof(mrnx_brain_joint_transaction_v1) ==
              sizeof(MRNumanXBrainJointTransactionToken));
static_assert(sizeof(mrnx_brain_joint_substep_v1) ==
              sizeof(MRNumanXBrainJointSubstepToken));
static_assert(sizeof(mrnx_brain_motor_candidate_v1) ==
              sizeof(MRNumanXBrainMotorCandidate));

class RuntimeBuildFailure final : public std::runtime_error {
public:
    RuntimeBuildFailure(
        const mrnx_runtime_status_v1 value,
        const std::string& message
    ) : std::runtime_error(message), status(value) {}
    mrnx_runtime_status_v1 status;
};

void requireBuild(
    const bool condition,
    const mrnx_runtime_status_v1 status,
    const std::string& message
) {
    if (!condition) throw RuntimeBuildFailure(status, message);
}

template <typename T>
void readObject(std::istream& input, T& value, const char* description) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    requireBuild(
        input.good(), MRNX_RUNTIME_ASSET_FAILURE_V1,
        std::string("truncated ") + description);
}

template <typename T>
std::vector<T> readVector(
    std::istream& input,
    const std::size_t count,
    const char* description
) {
    std::vector<T> result(count);
    if (count != 0u) {
        input.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(count * sizeof(T)));
        requireBuild(
            input.good(), MRNX_RUNTIME_ASSET_FAILURE_V1,
            std::string("truncated ") + description);
    }
    return result;
}

struct FullBodyAssets {
    metalrobo::EngineModel model;
    RigidHeader rigid{};
    MuscleHeader muscle{};
    std::vector<MRMujocoMuscleSiteGPU> sites;
    std::vector<MRMujocoMuscleWrapGPU> wraps;
    std::vector<MRMujocoMuscleRouteNodeGPU> routes;
    std::vector<MRMujocoMuscleGPU> muscles;
    std::vector<MRMujocoMuscleStateGPU> states;
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::vector<MRNumiHumanStandContactGPU> supportContacts;
    std::vector<NMHumanSupportContactGPU> matterSupportContacts;
    std::vector<NMHumanSupportPointQueryGPU> matterSupportPointQueries;
    mr_float4 groundPoint{};
    mr_float4 groundNormal{0.0f, 1.0f, 0.0f, 0.0f};
    std::uint32_t bodyJacobianPointOffset = 0u;
    std::uint64_t sourceFingerprint = 0u;
};

struct VisionProfile {
    std::uint32_t parentBodyIndex = MR_INVALID_INDEX;
    mr_float4 localPosition{};
    mr_float4 localOrientation{};
    mr_float4 intrinsics{};
    mr_float4 depthAndTimestep{};
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::uint64_t sourceFingerprint = 0u;
    std::vector<MRNumanXVisualBodyBoundsGPU> bodyBounds;
};

[[nodiscard]] mr_float4 quaternionMultiply(
    const mr_float4 left,
    const mr_float4 right
) noexcept {
    return {
        left.w * right.x + left.x * right.w + left.y * right.z -
            left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w +
            left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x +
            left.z * right.w,
        left.w * right.w - left.x * right.x - left.y * right.y -
            left.z * right.z};
}

[[nodiscard]] mr_float4 quaternionConjugate(const mr_float4 value) noexcept {
    return {-value.x, -value.y, -value.z, value.w};
}

[[nodiscard]] mr_float4 quaternionRotate(
    const mr_float4 rotation,
    const mr_float4 vector
) noexcept {
    const mr_float4 pure{vector.x, vector.y, vector.z, 0.0f};
    const mr_float4 rotated = quaternionMultiply(
        quaternionMultiply(rotation, pure), quaternionConjugate(rotation));
    return {rotated.x, rotated.y, rotated.z, 0.0f};
}

[[nodiscard]] std::uint64_t hashBytes(
    const void* raw,
    const std::size_t byteCount
) noexcept {
    if (raw == nullptr || byteCount == 0u) return 0u;
    const auto* bytes = static_cast<const std::uint8_t*>(raw);
    std::uint64_t hash = kFnvOffset;
    for (std::size_t index = 0u; index < byteCount; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash == 0u ? kFnvOffset : hash;
}

struct ImmutablePayload {
    std::string bytes;
};

[[nodiscard]] ImmutablePayload loadImmutablePayload(
    const std::string& path,
    const char* description
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    requireBuild(
        input.is_open(), MRNX_RUNTIME_ASSET_FAILURE_V1,
        std::string("cannot open ") + description);
    const std::streampos end = input.tellg();
    requireBuild(
        end > 0 && static_cast<std::uintmax_t>(end) <=
            static_cast<std::uintmax_t>(
                std::numeric_limits<std::streamsize>::max()),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        std::string(description) + " has an invalid byte length");
    ImmutablePayload result;
    result.bytes.resize(static_cast<std::size_t>(end));
    input.seekg(0, std::ios::beg);
    input.read(
        result.bytes.data(),
        static_cast<std::streamsize>(result.bytes.size()));
    requireBuild(
        input.good(), MRNX_RUNTIME_ASSET_FAILURE_V1,
        std::string("truncated ") + description);
    return result;
}

void appendFingerprintBytes(
    std::uint64_t& hash,
    const void* raw,
    const std::size_t byteCount
) noexcept {
    const auto* bytes = static_cast<const std::uint8_t*>(raw);
    for (std::size_t index = 0u; index < byteCount; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

void appendFingerprintU64(
    std::uint64_t& hash,
    const std::uint64_t value
) noexcept {
    for (std::uint32_t index = 0u; index < 8u; ++index) {
        const std::uint8_t byte = static_cast<std::uint8_t>(
            (value >> (index * 8u)) & 0xffu);
        appendFingerprintBytes(hash, &byte, sizeof(byte));
    }
}

[[nodiscard]] std::uint64_t fullBodySourceFingerprint(
    const ImmutablePayload& rigid,
    const ImmutablePayload& muscle,
    const ImmutablePayload& support
) noexcept {
    constexpr char domain[] = "mrnx.fullbody.source.v1";
    std::uint64_t hash = kFnvOffset;
    appendFingerprintBytes(hash, domain, sizeof(domain) - 1u);
    for (const ImmutablePayload* payload : {&rigid, &muscle, &support}) {
        appendFingerprintU64(
            hash, static_cast<std::uint64_t>(payload->bytes.size()));
        appendFingerprintBytes(
            hash, payload->bytes.data(), payload->bytes.size());
    }
    return hash == 0u ? kFnvOffset : hash;
}

[[nodiscard]] mr_float4 quaternionRotateHost(
    const mr_float4 quaternion,
    const mr_float4 point
) noexcept {
    const float ux = quaternion.x;
    const float uy = quaternion.y;
    const float uz = quaternion.z;
    const float scalar = quaternion.w;
    const float dotUV = ux * point.x + uy * point.y + uz * point.z;
    const float dotUU = ux * ux + uy * uy + uz * uz;
    const float crossX = uy * point.z - uz * point.y;
    const float crossY = uz * point.x - ux * point.z;
    const float crossZ = ux * point.y - uy * point.x;
    return {
        2.0f * dotUV * ux + (scalar * scalar - dotUU) * point.x +
            2.0f * scalar * crossX,
        2.0f * dotUV * uy + (scalar * scalar - dotUU) * point.y +
            2.0f * scalar * crossY,
        2.0f * dotUV * uz + (scalar * scalar - dotUU) * point.z +
            2.0f * scalar * crossZ,
        0.0f};
}

[[nodiscard]] bool finiteNumber(id value, double& output) noexcept {
    if (value == nil || ![value isKindOfClass:[NSNumber class]]) return false;
    output = [static_cast<NSNumber*>(value) doubleValue];
    return std::isfinite(output);
}

[[nodiscard]] bool float4Array(
    id value,
    mr_float4& output,
    const float w
) noexcept {
    if (value == nil || ![value isKindOfClass:[NSArray class]]) return false;
    NSArray* array = static_cast<NSArray*>(value);
    if (array.count != 3u && array.count != 4u) return false;
    double components[4]{0.0, 0.0, 0.0, static_cast<double>(w)};
    for (NSUInteger index = 0u; index < array.count; ++index) {
        if (!finiteNumber(array[index], components[index]) ||
            components[index] < -std::numeric_limits<float>::max() ||
            components[index] > std::numeric_limits<float>::max()) {
            return false;
        }
    }
    output = {
        static_cast<float>(components[0]),
        static_cast<float>(components[1]),
        static_cast<float>(components[2]),
        static_cast<float>(components[3])};
    return true;
}

VisionProfile loadVisionProfile(
    const std::string& packPath,
    const std::string& profilePath,
    const std::uint32_t bodyCount,
    const std::uint64_t timestepMicroseconds
) {
    metalrobo::VisualAssetPackV2 pack;
    std::string packReason;
    requireBuild(
        metalrobo::readVisualAssetPackIndex(
            std::filesystem::path(packPath), pack, &packReason) &&
            pack.schemaVersion == metalrobo::kVisualAssetPackVersion &&
            !pack.contentHash.empty() && !pack.instances.empty() &&
            !pack.primitives.empty(),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "visual pack index failed: " + packReason);

    NSString* profileNSString = [NSString stringWithUTF8String:profilePath.c_str()];
    requireBuild(
        profileNSString != nil, MRNX_RUNTIME_ASSET_FAILURE_V1,
        "vision profile path is not valid UTF-8");
    NSError* readError = nil;
    NSData* profileData = [NSData dataWithContentsOfFile:profileNSString
        options:NSDataReadingMappedIfSafe error:&readError];
    requireBuild(
        profileData != nil && profileData.length != 0u,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "vision profile is unreadable");
    NSError* jsonError = nil;
    id object = [NSJSONSerialization JSONObjectWithData:profileData
        options:0 error:&jsonError];
    requireBuild(
        object != nil && [object isKindOfClass:[NSDictionary class]],
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "vision profile is not a JSON object");
    NSDictionary* root = static_cast<NSDictionary*>(object);
    id schema = root[@"schema"];
    id contentHash = root[@"visual_pack_content_hash"];
    id cameraValue = root[@"camera"];
    requireBuild(
        [schema isKindOfClass:[NSString class]] &&
            [static_cast<NSString*>(schema) isEqualToString:
                @"numi.human.numanx-head-vision-profile.v1"] &&
            [contentHash isKindOfClass:[NSString class]] &&
            pack.contentHash == std::string(
                [static_cast<NSString*>(contentHash) UTF8String]) &&
            [cameraValue isKindOfClass:[NSDictionary class]],
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "vision profile identity does not match the source visual pack");
    NSDictionary* camera = static_cast<NSDictionary*>(cameraValue);
    double bodyIndex = 0.0;
    double width = 0.0;
    double height = 0.0;
    double minimumDepth = 0.0;
    double maximumDepth = 0.0;
    double depthQuantum = 0.0;
    VisionProfile result;
    requireBuild(
        finiteNumber(camera[@"parent_body_index"], bodyIndex) &&
            finiteNumber(camera[@"width"], width) &&
            finiteNumber(camera[@"height"], height) &&
            finiteNumber(camera[@"minimum_depth_metres"], minimumDepth) &&
            finiteNumber(camera[@"maximum_depth_metres"], maximumDepth) &&
            finiteNumber(camera[@"depth_quantum_metres"], depthQuantum) &&
            bodyIndex >= 0.0 && bodyIndex < bodyCount &&
            std::floor(bodyIndex) == bodyIndex &&
            width == MR_NUMANX_HUMAN_VISION_WIDTH &&
            height == MR_NUMANX_HUMAN_VISION_HEIGHT &&
            minimumDepth > 0.0 && maximumDepth > minimumDepth &&
            depthQuantum > 0.0 &&
            float4Array(camera[@"local_position_metres"],
                result.localPosition, 0.0f) &&
            float4Array(camera[@"local_orientation_xyzw"],
                result.localOrientation, 1.0f) &&
            float4Array(camera[@"intrinsics_fx_fy_cx_cy"],
                result.intrinsics, 0.0f),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "vision profile camera calibration is invalid");
    const float quaternionNorm = std::sqrt(
        result.localOrientation.x * result.localOrientation.x +
        result.localOrientation.y * result.localOrientation.y +
        result.localOrientation.z * result.localOrientation.z +
        result.localOrientation.w * result.localOrientation.w);
    requireBuild(
        std::isfinite(quaternionNorm) &&
            std::abs(quaternionNorm - 1.0f) <= 1.0e-4f &&
            result.intrinsics.x > 0.0f && result.intrinsics.y > 0.0f,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "vision profile quaternion or intrinsics are invalid");
    result.parentBodyIndex = static_cast<std::uint32_t>(bodyIndex);
    result.width = static_cast<std::uint32_t>(width);
    result.height = static_cast<std::uint32_t>(height);
    result.depthAndTimestep = {
        static_cast<float>(minimumDepth),
        static_cast<float>(maximumDepth),
        static_cast<float>(depthQuantum),
        static_cast<float>(static_cast<double>(timestepMicroseconds) /
            1'000'000.0)};
    result.bodyBounds.resize(bodyCount);
    const float infinity = std::numeric_limits<float>::infinity();
    for (auto& bounds : result.bodyBounds) {
        bounds.minimum = {infinity, infinity, infinity, 0.0f};
        bounds.maximum = {-infinity, -infinity, -infinity, 0.0f};
    }
    std::uint32_t boundBodyCount = 0u;
    for (std::size_t instanceIndex = 0u;
         instanceIndex < pack.instances.size(); ++instanceIndex) {
        const MRVisualInstanceGPUV2& instance = pack.instances[instanceIndex];
        const std::uint32_t body = instance.binding.y;
        if (body == MR_INVALID_INDEX) continue;
        requireBuild(
            body < bodyCount &&
                instance.geometry.x <= pack.primitives.size() &&
                instance.geometry.y <=
                    pack.primitives.size() - instance.geometry.x &&
                std::isfinite(instance.translationAndScale.w) &&
                instance.translationAndScale.w > 0.0f,
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "visual pack instance binding is outside the full body");
        bool bodyWasEmpty = !std::isfinite(result.bodyBounds[body].minimum.x);
        for (std::uint32_t primitiveOffset = 0u;
             primitiveOffset < instance.geometry.y; ++primitiveOffset) {
            const MRVisualPrimitiveGPUV2& primitive =
                pack.primitives[instance.geometry.x + primitiveOffset];
            for (std::uint32_t corner = 0u; corner < 8u; ++corner) {
                mr_float4 point{
                    (corner & 1u) != 0u ? primitive.boundsMaximum.x
                                        : primitive.boundsMinimum.x,
                    (corner & 2u) != 0u ? primitive.boundsMaximum.y
                                        : primitive.boundsMinimum.y,
                    (corner & 4u) != 0u ? primitive.boundsMaximum.z
                                        : primitive.boundsMinimum.z,
                    0.0f};
                point.x *= instance.translationAndScale.w;
                point.y *= instance.translationAndScale.w;
                point.z *= instance.translationAndScale.w;
                point = quaternionRotateHost(instance.orientation, point);
                point.x += instance.translationAndScale.x;
                point.y += instance.translationAndScale.y;
                point.z += instance.translationAndScale.z;
                auto& bounds = result.bodyBounds[body];
                bounds.minimum.x = std::min(bounds.minimum.x, point.x);
                bounds.minimum.y = std::min(bounds.minimum.y, point.y);
                bounds.minimum.z = std::min(bounds.minimum.z, point.z);
                bounds.maximum.x = std::max(bounds.maximum.x, point.x);
                bounds.maximum.y = std::max(bounds.maximum.y, point.y);
                bounds.maximum.z = std::max(bounds.maximum.z, point.z);
            }
        }
        if (bodyWasEmpty &&
            std::isfinite(result.bodyBounds[body].minimum.x)) {
            ++boundBodyCount;
        }
    }
    requireBuild(
        boundBodyCount != 0u &&
            std::isfinite(result.bodyBounds[result.parentBodyIndex].minimum.x),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "visual pack has no source-authored bounds for the calibrated head");
    std::uint64_t fingerprint = hashBytes(
        pack.contentHash.data(), pack.contentHash.size());
    const std::uint64_t profileFingerprint = hashBytes(
        profileData.bytes, profileData.length);
    fingerprint ^= profileFingerprint;
    fingerprint *= kFnvPrime;
    fingerprint ^= boundBodyCount;
    fingerprint *= kFnvPrime;
    result.sourceFingerprint = fingerprint == 0u ? kFnvOffset : fingerprint;
    return result;
}

[[nodiscard]] std::uint64_t timingFingerprint(
    const mrnx_candidate_timing_v1& timing
) noexcept {
    std::uint64_t hash = kFnvOffset;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&timing);
    for (std::size_t index = 0u;
         index < offsetof(mrnx_candidate_timing_v1, timing_fingerprint);
         ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash == 0u ? kFnvOffset : hash;
}

[[nodiscard]] bool validRouteType(const std::uint32_t value) noexcept {
    return value == MR_MUJOCO_MUSCLE_ROUTE_SITE ||
        value == MR_MUJOCO_MUSCLE_ROUTE_SPHERE ||
        value == MR_MUJOCO_MUSCLE_ROUTE_CYLINDER;
}

FullBodyAssets loadFullBodyAssets(
    const std::string& rigidPath,
    const std::string& musclePath,
    const std::string& supportContactPath
) {
    FullBodyAssets result;
    const ImmutablePayload rigidImage = loadImmutablePayload(
        rigidPath, "NHRIGID2 payload");
    const ImmutablePayload muscleImage = loadImmutablePayload(
        musclePath, "NHMYO payload");
    const ImmutablePayload supportImage = loadImmutablePayload(
        supportContactPath, "NHCNT1 support-contact payload");
    std::istringstream rigidInput(
        rigidImage.bytes, std::ios::in | std::ios::binary);
    readObject(rigidInput, result.rigid, "NHRIGID2 header");
    requireBuild(
        result.rigid.magic == kRigidMagic &&
            result.rigid.payloadABI == kRigidABI &&
            result.rigid.engineABI == MR_ENGINE_ABI_VERSION &&
            result.rigid.rootBodyIndex == 0u &&
            result.rigid.reserved0 == 0u &&
            result.rigid.nq == MRNX_FULL_BODY_NQ &&
            result.rigid.nv == MRNX_FULL_BODY_NV &&
            result.rigid.nq == result.rigid.nv + 1u &&
            result.rigid.engineBodyCount > 0u &&
            result.rigid.jointCount + 1u == result.rigid.engineBodyCount,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHRIGID2 is not the canonical 129/128 full body");
    result.model.name = "numanx_fullbody_runtime_v1";
    readObject(rigidInput, result.model.world, "NHRIGID2 world");
    MRArticulationGPU articulation{};
    readObject(rigidInput, articulation, "NHRIGID2 articulation");
    result.model.articulations.push_back(articulation);
    result.model.bodies = readVector<MRBodyPropertiesGPU>(
        rigidInput, result.rigid.engineBodyCount, "NHRIGID2 bodies");
    result.model.joints = readVector<MRJointDescriptorGPU>(
        rigidInput, result.rigid.jointCount, "NHRIGID2 joints");
    result.model.dofs = readVector<MRDofPropertiesGPU>(
        rigidInput, result.rigid.nv, "NHRIGID2 dofs");
    result.model.defaultQ = readVector<float>(
        rigidInput, result.rigid.nq, "NHRIGID2 default q");
    result.model.defaultV = readVector<float>(
        rigidInput, result.rigid.nv, "NHRIGID2 default v");
    const auto sourceMap = readVector<std::uint32_t>(
        rigidInput, result.rigid.sourceBodyCount, "NHRIGID2 source map");
    (void)readVector<SourcePoseRecord>(
        rigidInput, result.rigid.sourceBodyCount, "NHRIGID2 source poses");
    requireBuild(
        rigidInput.peek() == std::char_traits<char>::eof() &&
            result.model.world.bodyCount == result.rigid.engineBodyCount &&
            result.model.world.nq == result.rigid.nq &&
            result.model.world.nv == result.rigid.nv &&
            articulation.rootType == MR_ROOT_FLOATING &&
            articulation.rootBody == 0u &&
            articulation.bodyCount == result.rigid.engineBodyCount &&
            articulation.nq == result.rigid.nq &&
            articulation.nv == result.rigid.nv,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHRIGID2 model/header disagreement");
    for (const std::uint32_t body : sourceMap) {
        requireBuild(
            body < result.rigid.engineBodyCount,
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "NHRIGID2 source body map is out of range");
    }
    std::string modelReason;
    requireBuild(
        result.model.valid(&modelReason), MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHRIGID2 EngineModel invalid: " + modelReason);

    std::istringstream muscleInput(
        muscleImage.bytes, std::ios::in | std::ios::binary);
    readObject(muscleInput, result.muscle, "NHMYO header");
    const bool legacy = result.muscle.magic == kLegacyMuscleMagic &&
        result.muscle.payloadABI == kLegacyMuscleABI &&
        result.muscle.reserved0 == 0u && result.muscle.reserved1 == 0u;
    const bool compliant = result.muscle.magic == kMuscleMagic &&
        result.muscle.payloadABI == kMuscleABI &&
        result.muscle.reserved0 == result.muscle.muscleCount &&
        result.muscle.reserved1 == sizeof(MuscleArchitectureRecord);
    requireBuild(
        (legacy || compliant) &&
            result.muscle.engineBodyCount == result.rigid.engineBodyCount &&
            result.muscle.sourceSHA256 == result.rigid.sourceSHA256 &&
            result.muscle.muscleCount == MRNX_FULL_BODY_MUSCLE_COUNT,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHMYO is not the matching 416-muscle full body");
    const auto sourceSites = readVector<SiteRecord>(
        muscleInput, result.muscle.siteCount, "NHMYO sites");
    const auto sourceWraps = readVector<WrapRecord>(
        muscleInput, result.muscle.wrapCount, "NHMYO wraps");
    const auto sourceRoutes = readVector<RouteRecord>(
        muscleInput, result.muscle.routeNodeCount, "NHMYO routes");
    const auto sourceMuscles = readVector<MuscleRecord>(
        muscleInput, result.muscle.muscleCount, "NHMYO muscles");
    const auto architectures = compliant
        ? readVector<MuscleArchitectureRecord>(
              muscleInput, result.muscle.muscleCount,
              "NHMYO compliant architectures")
        : std::vector<MuscleArchitectureRecord>(
              result.muscle.muscleCount);
    requireBuild(
        muscleInput.peek() == std::char_traits<char>::eof(),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHMYO payload has trailing bytes");

    result.sites.reserve(sourceSites.size());
    for (const auto& source : sourceSites) {
        requireBuild(
            source.bodyIndex < result.rigid.engineBodyCount,
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "NHMYO site body is out of range");
        MRMujocoMuscleSiteGPU value{};
        value.bodyIndex = source.bodyIndex;
        value.localPoint = {source.x, source.y, source.z, 0.0f};
        result.sites.push_back(value);
    }
    result.wraps.reserve(sourceWraps.size());
    for (const auto& source : sourceWraps) {
        requireBuild(
            source.bodyIndex < result.rigid.engineBodyCount &&
                validRouteType(source.type),
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "NHMYO wrap is invalid");
        MRMujocoMuscleWrapGPU value{};
        value.bodyIndex = source.bodyIndex;
        value.type = source.type;
        value.localCenter = {
            source.centerX, source.centerY, source.centerZ, 0.0f};
        value.rotationRow0 = {
            source.rotation[0], source.rotation[1], source.rotation[2], 0.0f};
        value.rotationRow1 = {
            source.rotation[3], source.rotation[4], source.rotation[5], 0.0f};
        value.rotationRow2 = {
            source.rotation[6], source.rotation[7], source.rotation[8], 0.0f};
        value.radius = {source.radius, 0.0f, 0.0f, 0.0f};
        result.wraps.push_back(value);
    }
    result.routes.reserve(sourceRoutes.size());
    for (const auto& source : sourceRoutes) {
        requireBuild(
            validRouteType(source.type) && source.reserved0 == 0u &&
                ((source.type == MR_MUJOCO_MUSCLE_ROUTE_SITE &&
                  source.targetIndex < result.sites.size()) ||
                 (source.type != MR_MUJOCO_MUSCLE_ROUTE_SITE &&
                  source.targetIndex < result.wraps.size())) &&
                (source.sideSiteIndex == MR_INVALID_INDEX ||
                 source.sideSiteIndex < result.sites.size()),
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "NHMYO route is invalid");
        MRMujocoMuscleRouteNodeGPU value{};
        value.type = source.type;
        value.targetIndex = source.targetIndex;
        value.sideSiteIndex = source.sideSiteIndex;
        result.routes.push_back(value);
    }
    result.muscles.reserve(sourceMuscles.size());
    result.states.reserve(sourceMuscles.size());
    for (std::size_t index = 0u; index < sourceMuscles.size(); ++index) {
        const auto& source = sourceMuscles[index];
        const auto& architecture = architectures[index];
        requireBuild(
            source.reserved0 == 0u &&
                source.routeOffset <= result.routes.size() &&
                source.routeCount <=
                    result.routes.size() - source.routeOffset &&
                source.routeCount >= 2u,
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "NHMYO muscle route range is invalid");
        const auto& firstRoute = result.routes[source.routeOffset];
        const auto& terminalRoute = result.routes[
            source.routeOffset + source.routeCount - 1u];
        requireBuild(
            firstRoute.type == MR_MUJOCO_MUSCLE_ROUTE_SITE &&
                terminalRoute.type == MR_MUJOCO_MUSCLE_ROUTE_SITE &&
                firstRoute.targetIndex < result.sites.size() &&
                terminalRoute.targetIndex < result.sites.size(),
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "NHMYO muscle endpoints must be source sites");
        MRMujocoMuscleGPU value{};
        value.route = {source.routeOffset, source.routeCount, 0u, 0u};
        value.lengthRangeAndAcceleration = {
            source.values[0], source.values[1], source.values[2], 0.0f};
        value.controlRange = {
            source.values[3], source.values[4], 0.0f, 0.0f};
        for (std::size_t parameter = 0u; parameter < 10u; ++parameter) {
            (&value.gainParameters[parameter / 4u].x)[parameter % 4u] =
                source.values[5u + parameter];
            (&value.biasParameters[parameter / 4u].x)[parameter % 4u] =
                source.values[15u + parameter];
            (&value.dynamicParameters[parameter / 4u].x)[parameter % 4u] =
                source.values[25u + parameter];
        }
        value.compliantArchitecture0 = {
            architecture.optimalFiberLength,
            architecture.tendonSlackLength,
            architecture.tendonStrainAtOneNormalizedForce,
            architecture.tendonStiffnessAtOneNormalizedForce};
        value.compliantArchitecture1 = {
            architecture.tendonNormalizedForceAtToeEnd,
            architecture.tendonCurviness,
            architecture.normalizedFiberDamping,
            architecture.fitNormalizedRmse};
        result.muscles.push_back(value);
        MRMujocoMuscleStateGPU state{};
        state.excitationAndActivation = {0.0f, 0.0f, 0.0f, 0.0f};
        result.states.push_back(state);
    }

    std::istringstream supportInput(
        supportImage.bytes, std::ios::in | std::ios::binary);
    SupportContactHeader supportHeader{};
    readObject(supportInput, supportHeader, "NHCNT1 header");
    requireBuild(
        supportHeader.magic == kSupportContactMagic &&
            supportHeader.payloadABI == kSupportContactABI &&
            supportHeader.engineBodyCount == result.rigid.engineBodyCount &&
            supportHeader.contactCount > 0u &&
            supportHeader.contactCount <= MR_NUMI_HUMAN_STAND_MAX_CONTACTS &&
            supportHeader.reserved0 == 0u &&
            supportHeader.sourceSHA256 == result.rigid.sourceSHA256 &&
            std::isfinite(supportHeader.groundPoint[0]) &&
            std::isfinite(supportHeader.groundPoint[1]) &&
            std::isfinite(supportHeader.groundPoint[2]) &&
            std::isfinite(supportHeader.groundNormal[0]) &&
            std::isfinite(supportHeader.groundNormal[1]) &&
            std::isfinite(supportHeader.groundNormal[2]) &&
            std::isfinite(supportHeader.groundFriction) &&
            supportHeader.groundFriction >= 0.0f,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHCNT1 is not the matching full-body support-contact payload");
    const float normalSquared =
        supportHeader.groundNormal[0] * supportHeader.groundNormal[0] +
        supportHeader.groundNormal[1] * supportHeader.groundNormal[1] +
        supportHeader.groundNormal[2] * supportHeader.groundNormal[2];
    requireBuild(
        std::isfinite(normalSquared) && normalSquared >= 0.999f &&
            normalSquared <= 1.001f,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHCNT1 ground normal is not unit length");
    const auto supportRecords = readVector<SupportContactRecord>(
        supportInput, supportHeader.contactCount, "NHCNT1 contacts");
    requireBuild(
        supportInput.peek() == std::char_traits<char>::eof(),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHCNT1 payload has trailing bytes");
    result.groundPoint = {
        supportHeader.groundPoint[0], supportHeader.groundPoint[1],
        supportHeader.groundPoint[2], 0.0f};
    result.groundNormal = {
        supportHeader.groundNormal[0], supportHeader.groundNormal[1],
        supportHeader.groundNormal[2], 0.0f};

    result.points.reserve(
        static_cast<std::size_t>(result.rigid.engineBodyCount) * 5u +
            supportRecords.size());
    for (std::uint32_t body = 0u;
         body < result.rigid.engineBodyCount; ++body) {
        MRArticulatedPointImpulseGPU point{};
        point.bodyIndex = body;
        result.points.push_back(point);
    }
    result.supportContacts.reserve(supportRecords.size());
    for (const auto& source : supportRecords) {
        requireBuild(
            source.bodyIndex < result.rigid.engineBodyCount &&
                source.sourceGeometryIndex != MR_INVALID_INDEX &&
                std::isfinite(source.localPoint[0]) &&
                std::isfinite(source.localPoint[1]) &&
                std::isfinite(source.localPoint[2]) &&
                std::isfinite(source.planeWitness[0]) &&
                std::isfinite(source.planeWitness[1]) &&
                std::isfinite(source.planeWitness[2]) &&
                std::isfinite(source.friction) && source.friction >= 0.0f &&
                std::isfinite(source.defaultSignedDistance) &&
                source.reserved0 == 0.0f && source.reserved1 == 0.0f,
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "NHCNT1 contact record is malformed");
        MRArticulatedPointImpulseGPU point{};
        point.bodyIndex = source.bodyIndex;
        point.localPoint = {
            source.localPoint[0], source.localPoint[1],
            source.localPoint[2], 0.0f};
        const auto pointIndex = static_cast<std::uint32_t>(
            result.points.size());
        result.points.push_back(point);
        NMHumanSupportPointQueryGPU matterPoint{};
        matterPoint.bodyIndex = point.bodyIndex;
        matterPoint.flags = point.flags;
        matterPoint.localPoint = {
            point.localPoint.x, point.localPoint.y,
            point.localPoint.z, point.localPoint.w};
        matterPoint.worldImpulse = {
            point.worldImpulse.x, point.worldImpulse.y,
            point.worldImpulse.z, point.worldImpulse.w};
        result.matterSupportPointQueries.push_back(matterPoint);
        MRNumiHumanStandContactGPU contact{};
        contact.bodyIndex = source.bodyIndex;
        contact.pointQueryIndex = pointIndex;
        contact.sourceGeometryIndex = source.sourceGeometryIndex;
        contact.frictionSlopAndStabilization = {
            source.friction,
            std::max(source.defaultSignedDistance, 0.0f) + 0.001f,
            0.2f,
            0.0f};
        result.supportContacts.push_back(contact);
        NMHumanSupportContactGPU matterContact{};
        matterContact.identity = {
            source.bodyIndex,
            source.sourceGeometryIndex,
            static_cast<std::uint32_t>(
                result.matterSupportContacts.size()),
            0u};
        matterContact.localPoint = {
            point.localPoint.x, point.localPoint.y,
            point.localPoint.z, point.localPoint.w};
        matterContact.frictionSlopAndStabilization = {
            contact.frictionSlopAndStabilization.x,
            contact.frictionSlopAndStabilization.y,
            contact.frictionSlopAndStabilization.z,
            contact.frictionSlopAndStabilization.w};
        result.matterSupportContacts.push_back(matterContact);
    }
    result.bodyJacobianPointOffset =
        static_cast<std::uint32_t>(result.points.size());
    constexpr std::array<std::array<float, 3u>, 4u> probes{{
        {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}};
    for (std::uint32_t body = 0u;
         body < result.rigid.engineBodyCount; ++body) {
        for (const auto& probe : probes) {
            MRArticulatedPointImpulseGPU point{};
            point.bodyIndex = body;
            point.localPoint = {probe[0], probe[1], probe[2], 0.0f};
            result.points.push_back(point);
        }
    }
    result.sourceFingerprint = fullBodySourceFingerprint(
        rigidImage, muscleImage, supportImage);
    return result;
}

} // namespace

namespace {

[[nodiscard]] numi::matter::CompiledWorld compileAttachedWorld(
    const std::string& materialPath,
    const std::uint32_t attachmentBody,
    const std::array<double, 3u>& attachmentWorldPosition,
    const std::array<double, 4u>& attachmentBodyOrientation,
    const std::uint64_t timestepMicroseconds
) {
    const auto parsed = numi::matter::parseMatterFile(materialPath);
    requireBuild(
        parsed.succeeded(), MRNX_RUNTIME_MATTER_FAILURE_V1,
        "Matter material did not parse");
    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep =
        static_cast<double>(timestepMicroseconds) / 1'000'000.0;
    source.gravity = {0.0, 0.0, 0.0};
    source.articulatedDofCapacity =
        MR_NUMANX_COUPLED_HUMAN_MAX_DOFS;
    source.articulatedQCapacity = MR_NUMANX_COUPLED_HUMAN_MAX_Q;
    // The attached tetrahedron follows a moving articulated boundary. Give
    // the deterministic outer Newton solve enough reassembly steps and use a
    // bounded 0.5% post-step residual for this coupled acceptance fixture.
    source.mixedSolver.newtonIterations = 16u;
    source.mixedSolver.relativeResidual = 5.0e-3;
    source.materials.push_back(parsed.material);
    numi::matter::ObjectSource object;
    object.name = "numanx_fullbody_attached_fem_v1";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::fem;
    // Keep the proof-carrying attached patch small relative to the full body:
    // it is a coupled Matter witness, not an invented anatomical organ. Its
    // one-centimetre edge also avoids adding a large artificial inertial mass
    // to the otherwise source-authored articulated model.
    constexpr double elementEdgeMetres = 0.01;
    object.characteristicLength = elementEdgeMetres;
    object.mixedFEM = false;
    const double x = attachmentWorldPosition[0];
    const double y = attachmentWorldPosition[1];
    const double z = attachmentWorldPosition[2];
    object.femNodes = {
        {x, y, z}, {x + elementEdgeMetres, y, z},
        {x, y + elementEdgeMetres, z},
        {x, y, z + elementEdgeMetres}};
    object.tetrahedra.push_back({{0u, 1u, 2u, 3u}});
    // Constrain this proof-witness tetrahedron to the pelvis as one coherent
    // material sample. The original single-corner pin folded a one-centimetre
    // element under ordinary root motion and therefore measured a fixture
    // singularity rather than Human/Matter transaction correctness. Local
    // points are derived from the actual default body frame rather than
    // assuming that pelvis axes equal world axes.
    const auto inverseBodyRotate = [&](const std::array<double, 3u>& value) {
        const double ux = -attachmentBodyOrientation[0];
        const double uy = -attachmentBodyOrientation[1];
        const double uz = -attachmentBodyOrientation[2];
        const double scalar = attachmentBodyOrientation[3];
        const std::array<double, 3u> twiceCross{
            2.0 * (uy * value[2] - uz * value[1]),
            2.0 * (uz * value[0] - ux * value[2]),
            2.0 * (ux * value[1] - uy * value[0])};
        return std::array<double, 3u>{
            value[0] + scalar * twiceCross[0] +
                (uy * twiceCross[2] - uz * twiceCross[1]),
            value[1] + scalar * twiceCross[1] +
                (uz * twiceCross[0] - ux * twiceCross[2]),
            value[2] + scalar * twiceCross[2] +
                (ux * twiceCross[1] - uy * twiceCross[0])};
    };
    const std::array<std::array<double, 3u>, 4u> attachmentOffsets{{
        {0.0, 0.0, 0.0},
        {elementEdgeMetres, 0.0, 0.0},
        {0.0, elementEdgeMetres, 0.0},
        {0.0, 0.0, elementEdgeMetres}}};
    for (std::uint32_t node = 0u; node < attachmentOffsets.size(); ++node) {
        numi::matter::FEMHumanAttachmentSource attachment;
        attachment.node = node;
        attachment.bodyIndex = attachmentBody;
        attachment.stableIdentifier = 0x4e585246u + node; // NXRF..NXRH
        attachment.localPoint = inverseBodyRotate(attachmentOffsets[node]);
        object.femHumanAttachments.push_back(attachment);
    }
    source.objects.push_back(std::move(object));
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(source, options);
    requireBuild(
        compiled.succeeded(), MRNX_RUNTIME_MATTER_FAILURE_V1,
        "attached Matter world did not compile");
    std::string error;
    requireBuild(
        numi::matter::validateCompiledWorldLayout(compiled.world, &error),
        MRNX_RUNTIME_MATTER_FAILURE_V1,
        "attached Matter layout failed: " + error);
    return std::move(compiled.world);
}

bool encodeRuntimeProof(
    void* context,
    const metalrobo::MetalNumanXHumanMatterStateProofPass& source
) noexcept {
    auto* runtime = static_cast<numi::matter::Runtime*>(context);
    if (runtime == nullptr ||
        source.abiVersion !=
            MR_NUMANX_HUMAN_MATTER_ADAPTER_ABI_VERSION ||
        source.structSize != sizeof(source)) {
        return false;
    }
    numi::matter::AcceptedStateProofPass pass{};
    pass.environmentCount = source.environmentCount;
    pass.environmentIdentifierBase = source.environmentIdentifierBase;
    pass.commandBuffer = source.commandBuffer;
    pass.q = source.q;
    pass.v = source.v;
    pass.mujocoStates = source.mujocoStates;
    pass.matterGeneralizedReaction = source.matterGeneralizedReaction;
    pass.environmentStatuses = source.environmentStatuses;
    pass.matterStatuses = source.matterStatuses;
    pass.acceptedStateProofs = source.acceptedStateProofs;
    pass.qGPUAddress = source.qGPUAddress;
    pass.vGPUAddress = source.vGPUAddress;
    pass.mujocoStatesGPUAddress = source.mujocoStatesGPUAddress;
    pass.matterGeneralizedReactionGPUAddress =
        source.matterGeneralizedReactionGPUAddress;
    pass.environmentStatusesGPUAddress =
        source.environmentStatusesGPUAddress;
    pass.matterStatusesGPUAddress = source.matterStatusesGPUAddress;
    pass.acceptedStateProofsGPUAddress =
        source.acceptedStateProofsGPUAddress;
    pass.qElementCount = source.qElementCount;
    pass.vElementCount = source.vElementCount;
    pass.mujocoStateCount = source.mujocoStateCount;
    pass.matterGeneralizedReactionElementCount =
        source.matterGeneralizedReactionElementCount;
    pass.environmentStatusElementCount =
        source.environmentStatusElementCount;
    pass.matterStatusElementCount = source.matterStatusElementCount;
    pass.acceptedStateProofElementCount =
        source.acceptedStateProofElementCount;
    pass.qStride = source.qStride;
    pass.vStride = source.vStride;
    pass.mujocoStateStride = source.mujocoStateStride;
    pass.reactionStride = source.reactionStride;
    pass.environmentStatusStride = source.environmentStatusStride;
    pass.matterStatusStride = source.matterStatusStride;
    pass.acceptedStateProofStride = source.acceptedStateProofStride;
    pass.qCoordinateCount = source.qCoordinateCount;
    pass.dofCount = source.dofCount;
    pass.transactionSlot = source.transactionSlot;
    pass.programFingerprint = source.programFingerprint;
    pass.stateProofProgramFingerprint =
        source.stateProofProgramFingerprint;
    pass.transactionFingerprint = source.transactionFingerprint;
    pass.substepFingerprint = source.substepFingerprint;
    pass.acceptedTimestampMicroseconds =
        source.acceptedTimestampMicroseconds;
    pass.physicsGeneration = source.physicsGeneration;
    pass.linearizationEpoch = source.linearizationEpoch;
    pass.slotGeneration = source.slotGeneration;
    pass.matterSourcePhysicsFingerprint =
        source.matterSourcePhysicsFingerprint;
    pass.matterDeviceProgramFingerprint =
        source.matterDeviceProgramFingerprint;
    return runtime->encodeAcceptedStateProof(pass);
}

[[nodiscard]] bool bufferObject(
    void* raw,
    __unsafe_unretained id<MTLBuffer>& output
) noexcept {
    if (raw == nullptr) return false;
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLBuffer)]) return false;
    output = (__bridge id<MTLBuffer>)raw;
    return output != nil;
}

[[nodiscard]] bool eventObject(
    void* raw,
    __unsafe_unretained id<MTLSharedEvent>& output
) noexcept {
    if (raw == nullptr) return false;
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLSharedEvent)]) return false;
    output = (__bridge id<MTLSharedEvent>)raw;
    return output != nil;
}

[[nodiscard]] bool commandBufferObject(
    void* raw,
    __unsafe_unretained id<MTLCommandBuffer>& output
) noexcept {
    if (raw == nullptr) return false;
    __unsafe_unretained id object = (__bridge id)raw;
    if (![object conformsToProtocol:@protocol(MTLCommandBuffer)]) return false;
    output = (__bridge id<MTLCommandBuffer>)raw;
    return output != nil;
}

[[nodiscard]] bool importableSharedEvent(
    id<MTLDevice> device,
    id<MTLSharedEvent> event
) noexcept {
    if (device == nil || event == nil) return false;
    MTLSharedEventHandle* handle = event.newSharedEventHandle;
    if (handle == nil) return false;
    id<MTLSharedEvent> imported = [device newSharedEventWithHandle:handle];
    return imported != nil;
}

[[nodiscard]] bool checkedEnd(
    const std::uint64_t address,
    const std::uint64_t count,
    std::uint64_t& end
) noexcept {
    if (address == 0u || count == 0u ||
        address > std::numeric_limits<std::uint64_t>::max() - count) {
        return false;
    }
    end = address + count;
    return true;
}

[[nodiscard]] bool disjoint(
    const std::uint64_t firstAddress,
    const std::uint64_t firstCount,
    const std::uint64_t secondAddress,
    const std::uint64_t secondCount
) noexcept {
    std::uint64_t firstEnd = 0u;
    std::uint64_t secondEnd = 0u;
    return checkedEnd(firstAddress, firstCount, firstEnd) &&
        checkedEnd(secondAddress, secondCount, secondEnd) &&
        (firstEnd <= secondAddress || secondEnd <= firstAddress);
}

struct ImportedRange {
    __strong id<MTLBuffer> buffer = nil;
    std::uint64_t address = 0u;
    std::uint64_t byteCount = 0u;
};

[[nodiscard]] bool importExactRange(
    id<MTLDevice> device,
    const mrnx_metal_range_v1& range,
    const std::uint64_t expectedBytes,
    const mrnx_element_type_v1 expectedType,
    const std::uint32_t expectedElementBytes,
    ImportedRange& output
) noexcept {
    __unsafe_unretained id<MTLBuffer> buffer = nil;
    if (range.abi_version != MRNX_BRIDGE_ABI_V1 ||
        range.struct_size != sizeof(range) ||
        range.byte_count != expectedBytes ||
        range.element_type != expectedType ||
        range.element_byte_count != expectedElementBytes ||
        !bufferObject(range.metal_buffer, buffer) || buffer.device != device ||
        buffer.gpuAddress == 0u ||
        range.byte_offset > static_cast<std::uint64_t>(buffer.length) ||
        expectedBytes > static_cast<std::uint64_t>(buffer.length) -
            range.byte_offset ||
        buffer.gpuAddress > std::numeric_limits<std::uint64_t>::max() -
            range.byte_offset ||
        buffer.gpuAddress + range.byte_offset != range.gpu_address) {
        return false;
    }
    std::uint64_t end = 0u;
    if (!checkedEnd(range.gpu_address, expectedBytes, end)) return false;
    output.buffer = buffer;
    output.address = range.gpu_address;
    output.byteCount = expectedBytes;
    return true;
}

[[nodiscard]] bool sameKey(
    const metalrobo::MetalNumanXHumanIOTransactionKey& first,
    const metalrobo::MetalNumanXHumanIOTransactionKey& second
) noexcept {
    return first.transactionFingerprint == second.transactionFingerprint &&
        first.programFingerprint == second.programFingerprint &&
        first.sensorFingerprint == second.sensorFingerprint &&
        first.transactionInstanceFingerprint ==
            second.transactionInstanceFingerprint &&
        first.sensorGeneration == second.sensorGeneration &&
        first.commandBufferIdentity == second.commandBufferIdentity;
}

struct RuntimeState;

struct ActiveRoot final : std::enable_shared_from_this<ActiveRoot> {
    RuntimeState* runtime = nullptr;
    std::mutex mutex;
    mrnx_prepared_v1* prepared = nullptr;
    mrnx_candidate_v1* candidate = nullptr;
    metalrobo::MetalNumanXHumanIOTransactionKey candidateKey{};
    mrnx_physical_root_settled_callback_v1 completion = nullptr;
    void* completionContext = nullptr;
    std::uint64_t slotGeneration = 0u;
    std::uint64_t physicsGeneration = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint32_t transactionSlot = 0u;
    std::uint64_t brainGeneration = 0u;
    std::uint64_t controlStep = 0u;
    std::uint64_t acceptedTimestampMicroseconds = 0u;
    std::uint64_t receptorTimestampMicroseconds = 0u;
    std::uint64_t previousTransactionFingerprint = 0u;
    std::uint64_t previousPhysicsGeneration = 0u;
    ImportedRange motorHeader{};
    ImportedRange excitation{};
    ImportedRange autonomic{};
    ImportedRange activeSensing{};
    ImportedRange motorReadyGate{};
    __strong id<MTLSharedEvent> motorReadyEvent = nil;
    __strong id<MTLBuffer> kinesthesia = nil;
    __strong id<MTLBuffer> kinesthesiaValidity = nil;
    __strong id<MTLBuffer> vestibular = nil;
    __strong id<MTLBuffer> vestibularValidity = nil;
    __strong id<MTLBuffer> audition = nil;
    __strong id<MTLBuffer> auditionValidity = nil;
    __strong id<MTLBuffer> vision = nil;
    __strong id<MTLBuffer> visionValidity = nil;
    __strong id<MTLBuffer> touch = nil;
    __strong id<MTLBuffer> touchValidity = nil;
    MRNumanXHumanSupplementalDispatchGPU supplementalDispatch{};
    bool humanSettled = false;
    bool humanReady = false;
    bool physicalSettled = false;
    bool physicalReady = false;
    bool settlementStarted = false;
};

struct RuntimeState final : std::enable_shared_from_this<RuntimeState> {
    mutable std::mutex mutex;
    mutable std::shared_mutex aggregateGate;
    DomainPtr domain;
    __strong id<MTLDevice> device = nil;
    FullBodyAssets assets;
    VisionProfile visionProfile;
    __strong id<MTLLibrary> supplementalLibrary = nil;
    __strong id<MTLComputePipelineState> supplementalPipeline = nil;
    __strong id<MTLBuffer> visualBodyBounds = nil;
    std::uint64_t supplementalProgramFingerprint = 0u;
    std::uint64_t timestepMicroseconds = 0u;
    std::uint32_t transactionSlotCount = 0u;
    std::uint64_t nextSlotGeneration = 1u;
    std::uint64_t nextSensorGeneration = 1u;
    std::uint64_t nextLinearizationEpoch = 1u;
    bool beginInProgress = false;
    bool terminalQuarantine = false;
    bool publishedOnce = false;
    std::uint64_t publishedTransactionFingerprint = 0u;
    std::uint64_t publishedBrainGeneration = 0u;
    std::uint64_t publishedPhysicsGeneration = 0u;
    std::uint64_t publishedTimestampMicroseconds = 0u;
    std::uint64_t publishedControlStep = 0u;
    // Attempts advance even when the root is authoritatively rejected; public
    // generation/timestamp authority advances only on accepted publication.
    std::uint64_t lastAttemptedControlStep = 0u;
    std::shared_ptr<ActiveRoot> active;
    mrnx_runtime_info_v1 info{};
    mrnx_aggregate_snapshot_v1 aggregate{};
    mrnx_candidate_timing_v1 aggregateTiming{};
    mrnx_candidate_channel_v1 aggregateChannels[MRNX_MAX_SENSOR_CHANNELS_V2]{};
    std::uint32_t aggregateChannelCount = 0u;
    __strong id<MTLBuffer> publishedChannelValues[MRNX_MAX_SENSOR_CHANNELS_V2]{};
    __strong id<MTLBuffer> publishedChannelValidity[MRNX_MAX_SENSOR_CHANNELS_V2]{};
    __strong id<MTLBuffer> publishedProprioception = nil;
    __strong id<MTLBuffer> publishedProprioceptionValidity = nil;
    __strong id<MTLBuffer> publishedInteroception = nil;
    __strong id<MTLBuffer> publishedInteroceptionValidity = nil;
    std::unique_ptr<numi::matter::Runtime> matter;
    std::unique_ptr<metalrobo::MetalNumanXHumanMatterContext> adapter;
    std::unique_ptr<metalrobo::MetalNumanXHumanIOContext> humanIO;
    std::unique_ptr<metalrobo::MetalArticulatedOperatorContext> owner;
    std::unique_ptr<metalrobo::MetalArticulatedOperatorSubmission>
        quarantinedSubmission;
};

} // namespace

struct mrnx_runtime_v1 {
    std::atomic<std::uint32_t> references{1u};
    std::shared_ptr<RuntimeState> state;
};

namespace {

[[nodiscard]] mrnx_completion_v1 rootCompletion(
    std::uint32_t status,
    std::uint32_t metalStatus,
    std::uint64_t generation
) noexcept;
void runtimeTerminalCompletion(
    void* raw,
    PreparedTerminalDisposition disposition,
    const mrnx_root_v1& root,
    const mrnx_candidate_view_v1* candidate,
    const mrnx_candidate_channel_v1* channels,
    std::uint32_t channelCount
) noexcept;
void settleActiveRoot(const std::shared_ptr<ActiveRoot>& active) noexcept;
void humanCandidateCompletion(
    void* raw,
    metalrobo::MetalNumanXHumanIOCandidateCompletionStatus status,
    const metalrobo::MetalNumanXHumanIOTransactionKey& key,
    const metalrobo::MetalNumanXHumanIOSensorView& view
) noexcept;
void physicalCompletion(
    void* raw,
    bool ready,
    std::uint64_t slotGeneration
) noexcept;
[[nodiscard]] bool encodeSupplementalSensors(
    void* raw,
    const metalrobo::MetalNumanXTransactionPass& pass
) noexcept;
[[nodiscard]] bool validateRootRequest(
    const std::shared_ptr<RuntimeState>& runtime,
    const mrnx_physical_root_request_v1& request,
    std::shared_ptr<ActiveRoot>& active,
    MRNumanXBrainJointTransactionToken& root,
    MRNumanXBrainJointSubstepToken& substep,
    MRNumanXBrainMotorCandidate& candidate,
    std::uint32_t& failureStage
) noexcept;

[[nodiscard]] bool loadSupplementalProgram(
    RuntimeState& runtime,
    const char* metallibPath
) {
    NSString* path = metallibPath != nullptr
        ? [NSString stringWithUTF8String:metallibPath] : nil;
    requireBuild(
        path != nil, MRNX_RUNTIME_METAL_FAILURE_V1,
        "supplemental sensor metallib path is not valid UTF-8");
    NSError* readError = nil;
    NSData* image = [NSData dataWithContentsOfFile:path
        options:NSDataReadingMappedIfSafe error:&readError];
    requireBuild(
        image != nil && image.length != 0u,
        MRNX_RUNTIME_METAL_FAILURE_V1,
        "supplemental sensor metallib image is unreadable");
    const std::uint64_t imageFingerprint = hashBytes(image.bytes, image.length);
    requireBuild(
        imageFingerprint != 0u, MRNX_RUNTIME_METAL_FAILURE_V1,
        "supplemental sensor metallib identity is zero");
    dispatch_data_t libraryImage = dispatch_data_create(
        image.bytes,
        image.length,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    requireBuild(
        libraryImage != nullptr, MRNX_RUNTIME_METAL_FAILURE_V1,
        "failed to retain supplemental sensor metallib image");
    NSError* libraryError = nil;
    runtime.supplementalLibrary = [runtime.device
        newLibraryWithData:libraryImage error:&libraryError];
    requireBuild(
        runtime.supplementalLibrary != nil,
        MRNX_RUNTIME_METAL_FAILURE_V1,
        "failed to load supplemental sensor metallib");
    id<MTLFunction> function = [runtime.supplementalLibrary
        newFunctionWithName:@"numanx_human_write_supplemental_sensors"];
    requireBuild(
        function != nil, MRNX_RUNTIME_METAL_FAILURE_V1,
        "MetalRobo metallib is missing the supplemental sensor kernel");
    NSError* pipelineError = nil;
    runtime.supplementalPipeline = [runtime.device
        newComputePipelineStateWithFunction:function error:&pipelineError];
    requireBuild(
        runtime.supplementalPipeline != nil,
        MRNX_RUNTIME_METAL_FAILURE_V1,
        "failed to create supplemental sensor pipeline");
    std::uint64_t fingerprint = imageFingerprint;
    constexpr char functionName[] =
        "numanx_human_write_supplemental_sensors";
    for (const unsigned char byte : functionName) {
        fingerprint ^= byte;
        fingerprint *= kFnvPrime;
    }
    fingerprint ^= MR_NUMANX_HUMAN_IO_ABI_VERSION;
    fingerprint *= kFnvPrime;
    fingerprint ^= runtime.visionProfile.sourceFingerprint;
    fingerprint *= kFnvPrime;
    // The HumanIO program is proposal/publication authority. Bind the exact
    // rigid, muscle, and support-contact payload identity into it so a source
    // change cannot retain the prior accepted sensor/root identity.
    fingerprint ^= runtime.assets.sourceFingerprint;
    fingerprint *= kFnvPrime;
    runtime.supplementalProgramFingerprint =
        fingerprint == 0u ? kFnvOffset : fingerprint;
    return true;
}

[[nodiscard]] id<MTLBuffer> makePrivateBuffer(
    id<MTLDevice> device,
    const std::size_t byteCount,
    NSString* label
) noexcept {
    if (device == nil || byteCount == 0u) return nil;
    id<MTLBuffer> buffer = [device newBufferWithLength:byteCount
        options:MTLResourceStorageModePrivate];
    if (buffer == nil || buffer.gpuAddress == 0u ||
        buffer.length != byteCount) {
        return nil;
    }
    buffer.label = label;
    return buffer;
}

[[nodiscard]] bool allocateSupplementalBuffers(
    const std::shared_ptr<RuntimeState>& runtime,
    const std::shared_ptr<ActiveRoot>& active
) noexcept {
    if (runtime == nullptr || active == nullptr) return false;
    constexpr std::size_t kinesthesiaValueBytes =
        MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT *
        MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT * sizeof(float);
    constexpr std::size_t kinesthesiaValidityBytes =
        MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT * sizeof(std::uint32_t);
    constexpr std::size_t vestibularValueBytes =
        MR_NUMANX_HUMAN_VESTIBULAR_RECEPTOR_COUNT *
        MR_NUMANX_HUMAN_VESTIBULAR_FEATURE_COUNT * sizeof(float);
    constexpr std::size_t vestibularValidityBytes =
        MR_NUMANX_HUMAN_VESTIBULAR_RECEPTOR_COUNT * sizeof(std::uint32_t);
    constexpr std::size_t auditionValueBytes =
        MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT *
        MR_NUMANX_HUMAN_AUDITION_FEATURE_COUNT * sizeof(float);
    constexpr std::size_t auditionValidityBytes =
        MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT * sizeof(std::uint32_t);
    constexpr std::size_t visionValueBytes =
        MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT *
        MR_NUMANX_HUMAN_VISION_FEATURE_COUNT * sizeof(float);
    constexpr std::size_t visionValidityBytes =
        MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT * sizeof(std::uint32_t);
    constexpr std::size_t touchValueBytes =
        MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT *
        MR_NUMANX_HUMAN_TOUCH_FEATURE_COUNT * sizeof(float);
    constexpr std::size_t touchValidityBytes =
        MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT * sizeof(std::uint32_t);
    active->kinesthesia = makePrivateBuffer(runtime->device,
        kinesthesiaValueBytes, @"NumanX candidate kinesthesia");
    active->kinesthesiaValidity = makePrivateBuffer(runtime->device,
        kinesthesiaValidityBytes, @"NumanX candidate kinesthesia validity");
    active->vestibular = makePrivateBuffer(runtime->device,
        vestibularValueBytes, @"NumanX candidate vestibular");
    active->vestibularValidity = makePrivateBuffer(runtime->device,
        vestibularValidityBytes, @"NumanX candidate vestibular validity");
    active->audition = makePrivateBuffer(runtime->device,
        auditionValueBytes, @"NumanX candidate audition");
    active->auditionValidity = makePrivateBuffer(runtime->device,
        auditionValidityBytes, @"NumanX candidate audition validity");
    active->vision = makePrivateBuffer(runtime->device,
        visionValueBytes, @"NumanX candidate vision");
    active->visionValidity = makePrivateBuffer(runtime->device,
        visionValidityBytes, @"NumanX candidate vision validity");
    active->touch = makePrivateBuffer(runtime->device,
        touchValueBytes, @"NumanX candidate touch");
    active->touchValidity = makePrivateBuffer(runtime->device,
        touchValidityBytes, @"NumanX candidate touch validity");
    return active->kinesthesia != nil && active->kinesthesiaValidity != nil &&
        active->vestibular != nil && active->vestibularValidity != nil &&
        active->audition != nil && active->auditionValidity != nil &&
        active->vision != nil && active->visionValidity != nil &&
        active->touch != nil && active->touchValidity != nil;
}

[[nodiscard]] std::shared_ptr<RuntimeState> createRuntimeState(
    const mrnx_runtime_config_v1& config
) {
    requireBuild(
        config.abi_version == MRNX_BRIDGE_ABI_V1 &&
            config.struct_size == sizeof(config) &&
            config.metal_device != nullptr &&
            config.rigid_payload_path != nullptr &&
            config.muscle_payload_path != nullptr &&
            config.support_contact_payload_path != nullptr &&
            config.visual_pack_path != nullptr &&
            config.vision_profile_path != nullptr &&
            config.metalrobo_metallib_path != nullptr &&
            config.matter_metallib_path != nullptr &&
            config.matter_material_path != nullptr &&
            config.rigid_payload_path[0] != '\0' &&
            config.muscle_payload_path[0] != '\0' &&
            config.support_contact_payload_path[0] != '\0' &&
            config.visual_pack_path[0] != '\0' &&
            config.vision_profile_path[0] != '\0' &&
            config.metalrobo_metallib_path[0] != '\0' &&
            config.matter_metallib_path[0] != '\0' &&
            config.matter_material_path[0] != '\0' &&
            config.timestep_microseconds != 0u &&
            config.timestep_microseconds <= 1'000'000u &&
            config.maximum_retained_bytes != 0u &&
            config.transaction_slot_count != 0u &&
            config.transaction_slot_count <= 8u && config.reserved0 == 0u,
        MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
        "invalid NumanX runtime configuration");
    __unsafe_unretained id object = (__bridge id)config.metal_device;
    requireBuild(
        [object conformsToProtocol:@protocol(MTLDevice)],
        MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
        "runtime Metal device is not an MTLDevice");
    __unsafe_unretained id<MTLDevice> device =
        (__bridge id<MTLDevice>)config.metal_device;
    id<MTLDevice> defaultDevice = MTLCreateSystemDefaultDevice();
    requireBuild(
        device != nil && defaultDevice != nil && device.registryID != 0u &&
            device.registryID == defaultDevice.registryID,
        MRNX_RUNTIME_METAL_FAILURE_V1,
        "runtime device is not the owning MetalRobo default device");

    auto runtime = std::make_shared<RuntimeState>();
    runtime->device = device;
    runtime->domain = metalrobo::numanx_bridge_v1::makeDomain(
        config.metal_device);
    requireBuild(
        runtime->domain != nullptr, MRNX_RUNTIME_METAL_FAILURE_V1,
        "failed to create NumanX bridge domain");
    runtime->assets = loadFullBodyAssets(
        config.rigid_payload_path, config.muscle_payload_path,
        config.support_contact_payload_path);
    runtime->timestepMicroseconds = config.timestep_microseconds;
    runtime->transactionSlotCount = config.transaction_slot_count;
    runtime->visionProfile = loadVisionProfile(
        config.visual_pack_path,
        config.vision_profile_path,
        runtime->assets.rigid.engineBodyCount,
        config.timestep_microseconds);
    requireBuild(
        runtime->visionProfile.bodyBounds.size() ==
            runtime->assets.rigid.engineBodyCount,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "visual body-bound table does not cover the full-body capacity");
    runtime->visualBodyBounds = [runtime->device
        newBufferWithBytes:runtime->visionProfile.bodyBounds.data()
        length:runtime->visionProfile.bodyBounds.size() *
            sizeof(MRNumanXVisualBodyBoundsGPU)
        options:MTLResourceStorageModeShared];
    requireBuild(
        runtime->visualBodyBounds != nil &&
            runtime->visualBodyBounds.gpuAddress != 0u,
        MRNX_RUNTIME_METAL_FAILURE_V1,
        "failed to allocate source visual body bounds");
    runtime->visualBodyBounds.label = @"NumanX source visual body bounds";
    (void)loadSupplementalProgram(*runtime, config.metalrobo_metallib_path);

    // Bind the reference Matter patch to the articulated root/pelvis COM.
    // The prior use of the final imported body was topology-order dependent
    // and coupled the FEM to an arbitrary high-motion distal link.
    constexpr std::uint32_t attachmentBody = 0u;
    const std::vector<double> defaultQ(
        runtime->assets.model.defaultQ.begin(),
        runtime->assets.model.defaultQ.end());
    const std::vector<double> defaultV(
        runtime->assets.model.defaultV.begin(),
        runtime->assets.model.defaultV.end());
    std::vector<metalrobo::ArticulatedBodyKinematics> defaultBodies(
        runtime->assets.model.bodies.size());
    const auto defaultBodyDiagnostics =
        metalrobo::computeArticulatedBodyKinematics(
            runtime->assets.model, 0u, defaultQ, defaultV, defaultBodies);
    requireBuild(
        defaultBodyDiagnostics.succeeded() &&
            attachmentBody < defaultBodies.size(),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "full-body default attachment kinematics failed");
    const auto world = compileAttachedWorld(
        config.matter_material_path,
        attachmentBody,
        defaultBodies[attachmentBody].centerOfMassPosition,
        defaultBodies[attachmentBody].orientation,
        config.timestep_microseconds);
    runtime->matter = std::make_unique<numi::matter::Runtime>();
    numi::matter::RuntimeConfiguration matterConfig;
    matterConfig.metallib = config.matter_metallib_path;
    matterConfig.environmentCount = 1u;
    matterConfig.captureEvents = false;
    matterConfig.captureDiagnostics = true;
    matterConfig.adaptiveTransfer = false;
    matterConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
        static_cast<std::uint64_t>(MRNX_FULL_BODY_MUSCLE_COUNT) *
        sizeof(MRMujocoMuscleStateGPU);
    matterConfig.humanSupportContacts =
        runtime->assets.matterSupportContacts;
    matterConfig.humanSupportPointQueries =
        runtime->assets.matterSupportPointQueries;
    matterConfig.humanSupportGroundPoint = {
        runtime->assets.groundPoint.x, runtime->assets.groundPoint.y,
        runtime->assets.groundPoint.z, runtime->assets.groundPoint.w};
    matterConfig.humanSupportGroundNormal = {
        runtime->assets.groundNormal.x, runtime->assets.groundNormal.y,
        runtime->assets.groundNormal.z, runtime->assets.groundNormal.w};
    const auto matterDiagnostics = runtime->matter->initialize(
        world, matterConfig);
    requireBuild(
        matterDiagnostics.encoded, MRNX_RUNTIME_MATTER_FAILURE_V1,
        "Matter Runtime initialization failed: " +
            matterDiagnostics.message);
    metalrobo::MetalNumanXHumanMatterConfig adapterConfig;
    adapterConfig.matterRuntime = runtime->matter.get();
    adapterConfig.coupledHumanMetallibPath =
        config.metalrobo_metallib_path;
    adapterConfig.adapterMetallibPath = config.metalrobo_metallib_path;
    adapterConfig.environmentCapacity = 1u;
    adapterConfig.pointCapacity = std::max<std::uint32_t>(
        4u, static_cast<std::uint32_t>(
            runtime->assets.matterSupportContacts.size()));
    adapterConfig.transactionSlotCount = config.transaction_slot_count;
    adapterConfig.maximumRetainedBytes = config.maximum_retained_bytes;
    adapterConfig.stateProofProgram.context = runtime->matter.get();
    adapterConfig.stateProofProgram.encode = &encodeRuntimeProof;
    adapterConfig.stateProofProgram.fingerprint =
        runtime->matter->acceptedStateProofProgramFingerprint();
    runtime->adapter =
        std::make_unique<metalrobo::MetalNumanXHumanMatterContext>(
            std::move(adapterConfig));
    const auto adapterDiagnostics = runtime->adapter->initialize();
    requireBuild(
        adapterDiagnostics.succeeded() &&
            adapterDiagnostics.acceptedStateProofAvailable,
        MRNX_RUNTIME_MATTER_FAILURE_V1,
        "Human/Matter adapter initialization failed: " +
            adapterDiagnostics.message);

    metalrobo::MetalNumanXHumanIOConfig humanIOConfig;
    humanIOConfig.metallibPath = config.metalrobo_metallib_path;
    humanIOConfig.maximumRetainedBytes = config.maximum_retained_bytes;
    runtime->humanIO =
        std::make_unique<metalrobo::MetalNumanXHumanIOContext>(
            std::move(humanIOConfig));
    const metalrobo::MetalArticulatedOperatorConfig ownerConfig{
        .writeDiagnosticMassMatrix = false,
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = static_cast<float>(
            static_cast<double>(config.timestep_microseconds) /
            1'000'000.0),
        .metallibPath = config.metalrobo_metallib_path,
    };
    runtime->owner =
        std::make_unique<metalrobo::MetalArticulatedOperatorContext>(
            ownerConfig);
    runtime->info.abi_version = MRNX_BRIDGE_ABI_V1;
    runtime->info.struct_size = sizeof(runtime->info);
    runtime->info.status = MRNX_RUNTIME_READY_V1;
    runtime->info.body_count = runtime->assets.rigid.engineBodyCount;
    runtime->info.q_coordinate_count = runtime->assets.rigid.nq;
    runtime->info.dof_count = runtime->assets.rigid.nv;
    runtime->info.muscle_count = runtime->assets.muscle.muscleCount;
    runtime->info.transaction_slot_count = config.transaction_slot_count;
    runtime->info.device_registry_id = device.registryID;
    runtime->info.accepted_state_proof_program_fingerprint =
        runtime->matter->acceptedStateProofProgramFingerprint();
    runtime->info.model_source_fingerprint =
        runtime->assets.sourceFingerprint;
    return runtime;
}

void fillRuntimeInfoFailure(
    mrnx_runtime_info_v1* info,
    const mrnx_runtime_status_v1 status
) noexcept {
    if (info == nullptr) return;
    *info = {};
    info->abi_version = MRNX_BRIDGE_ABI_V1;
    info->struct_size = sizeof(*info);
    info->status = status;
}

[[nodiscard]] bool beginPhysicalRoot(
    const std::shared_ptr<RuntimeState>& runtime,
    const mrnx_physical_root_request_v1& request,
    void* completionContext,
    const mrnx_physical_root_settled_callback_v1 completion
) {
    if (runtime == nullptr || completion == nullptr) return false;
    {
        const std::lock_guard lock(runtime->mutex);
        if (runtime->beginInProgress || runtime->active != nullptr ||
            runtime->terminalQuarantine ||
            runtime->nextSlotGeneration == 0u ||
            runtime->nextSensorGeneration == 0u ||
            runtime->nextLinearizationEpoch == 0u) {
            return false;
        }
        runtime->beginInProgress = true;
    }
    const auto failBegin = [&](const mrnx_runtime_status_v1 status) noexcept {
        const std::lock_guard lock(runtime->mutex);
        runtime->beginInProgress = false;
        runtime->info.status = status;
        return false;
    };

    std::shared_ptr<ActiveRoot> active;
    MRNumanXBrainJointTransactionToken root{};
    MRNumanXBrainJointSubstepToken substep{};
    MRNumanXBrainMotorCandidate candidate{};
    std::uint32_t failureStage = 0u;
    if (!validateRootRequest(
            runtime, request, active, root, substep, candidate,
            failureStage)) {
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = failureStage;
        }
        return failBegin(MRNX_RUNTIME_INVALID_REQUEST_V1);
    }
    {
        const std::lock_guard lock(runtime->mutex);
        runtime->lastAttemptedControlStep = root.controlStepIdentifier;
    }

    std::uint64_t slotGeneration = 0u;
    std::uint64_t sensorGeneration = 0u;
    std::uint64_t linearizationEpoch = 0u;
    {
        const std::lock_guard lock(runtime->mutex);
        if (runtime->nextSlotGeneration ==
                std::numeric_limits<std::uint64_t>::max() ||
            runtime->nextSensorGeneration ==
                std::numeric_limits<std::uint64_t>::max() ||
            runtime->nextLinearizationEpoch ==
                std::numeric_limits<std::uint64_t>::max()) {
            runtime->beginInProgress = false;
            runtime->terminalQuarantine = true;
            return false;
        }
        slotGeneration = runtime->nextSlotGeneration++;
        sensorGeneration = runtime->nextSensorGeneration++;
        linearizationEpoch = runtime->nextLinearizationEpoch++;
    }
    active->slotGeneration = slotGeneration;
    active->physicsGeneration = root.basePhysicsGeneration + 1u;
    active->acceptedTimestampMicroseconds =
        substep.candidateTimestampMicroseconds;
    active->receptorTimestampMicroseconds =
        substep.startTimestampMicroseconds;
    active->completion = completion;
    active->completionContext = completionContext;
    if (!allocateSupplementalBuffers(runtime, active)) {
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = 650u;
        }
        return failBegin(MRNX_RUNTIME_METAL_FAILURE_V1);
    }
    MRNumanXHumanSupplementalDispatchGPU& supplemental =
        active->supplementalDispatch;
    supplemental.abiVersion = MR_NUMANX_HUMAN_IO_ABI_VERSION;
    supplemental.qCoordinateCount = MRNX_FULL_BODY_NQ;
    supplemental.dofCount = MRNX_FULL_BODY_NV;
    supplemental.bodyCount = runtime->assets.rigid.engineBodyCount;
    supplemental.pointCount = static_cast<std::uint32_t>(
        runtime->assets.points.size());
    supplemental.supportPointOffset = runtime->assets.rigid.engineBodyCount;
    supplemental.supportPointCount = static_cast<std::uint32_t>(
        runtime->assets.supportContacts.size());
    supplemental.headBodyIndex = runtime->visionProfile.parentBodyIndex;
    supplemental.visionWidth = runtime->visionProfile.width;
    supplemental.visionHeight = runtime->visionProfile.height;
    supplemental.bodyBoundsCount = static_cast<std::uint32_t>(
        runtime->visionProfile.bodyBounds.size());
    supplemental.sensorGeneration = sensorGeneration;
    supplemental.transactionFingerprint = root.transactionFingerprint;
    supplemental.substepFingerprint = substep.substepFingerprint;
    supplemental.expectedActiveSensingGPUAddress =
        active->activeSensing.address;
    supplemental.visualSourceFingerprint =
        runtime->visionProfile.sourceFingerprint;
    supplemental.programFingerprint =
        runtime->supplementalProgramFingerprint;
    const auto supportView =
        runtime->matter->humanSupportCandidateConsequences();
    supplemental.expectedSupportConsequencesGPUAddress =
        supportView.gpuAddress;
    supplemental.matterProgramFingerprint =
        runtime->matter->deviceProgramFingerprint();
    supplemental.groundPoint = runtime->assets.groundPoint;
    supplemental.groundNormal = runtime->assets.groundNormal;
    supplemental.cameraLocalPosition = runtime->visionProfile.localPosition;
    supplemental.cameraLocalOrientation =
        runtime->visionProfile.localOrientation;
    supplemental.visionIntrinsics = runtime->visionProfile.intrinsics;
    supplemental.visionDepthAndTimestep =
        runtime->visionProfile.depthAndTimestep;

    metalrobo::MetalNumanXHumanIOInput humanInput{};
    humanInput.root = root;
    humanInput.substep = substep;
    humanInput.candidate = candidate;
    humanInput.motorOutputHeaderMetalBuffer =
        (__bridge void*)active->motorHeader.buffer;
    humanInput.motorOutputHeaderByteOffset = 0u;
    humanInput.motorOutputHeaderByteCount = active->motorHeader.byteCount;
    humanInput.motorOutputHeaderEnvironmentStride =
        active->motorHeader.byteCount;
    humanInput.expectedMotorOutputHeaderGPUAddress =
        active->motorHeader.address;
    humanInput.excitationMetalBuffer =
        (__bridge void*)active->excitation.buffer;
    humanInput.excitationByteOffset = 0u;
    humanInput.excitationByteCount = active->excitation.byteCount;
    humanInput.excitationEnvironmentStride = MRNX_FULL_BODY_MUSCLE_COUNT;
    humanInput.expectedExcitationGPUAddress = active->excitation.address;
    humanInput.autonomicCommandMetalBuffer =
        (__bridge void*)active->autonomic.buffer;
    humanInput.autonomicCommandByteCount = active->autonomic.byteCount;
    humanInput.expectedAutonomicCommandGPUAddress =
        active->autonomic.address;
    humanInput.activeSensingCommandMetalBuffer =
        (__bridge void*)active->activeSensing.buffer;
    humanInput.activeSensingCommandByteCount =
        active->activeSensing.byteCount;
    humanInput.expectedActiveSensingCommandGPUAddress =
        active->activeSensing.address;
    humanInput.motorReadyGateMetalBuffer =
        (__bridge void*)active->motorReadyGate.buffer;
    humanInput.motorReadyGateByteCount = active->motorReadyGate.byteCount;
    humanInput.expectedMotorReadyGateGPUAddress =
        active->motorReadyGate.address;
    humanInput.motorReadySharedEvent =
        (__bridge void*)active->motorReadyEvent;
    humanInput.motorReadySharedEventValue = request.motor_ready.value;
    humanInput.environmentCount = 1u;
    humanInput.muscleCount = MRNX_FULL_BODY_MUSCLE_COUNT;
    humanInput.stepCount = 1u;
    humanInput.timestepSeconds = static_cast<float>(
        static_cast<double>(runtime->timestepMicroseconds) / 1'000'000.0);
    humanInput.receptorTimestampMicroseconds =
        substep.startTimestampMicroseconds;
    humanInput.candidateSensorGeneration = sensorGeneration;
    humanInput.supplementalProgram.context = active.get();
    humanInput.supplementalProgram.encode = &encodeSupplementalSensors;
    humanInput.supplementalProgram.fingerprint =
        runtime->supplementalProgramFingerprint;
    metalrobo::MetalNumanXTransactionProgram humanProgram{};
    metalrobo::MetalNumanXHumanIOSensorView candidateView{};
    const auto humanPrepared = runtime->humanIO->prepare(
        humanInput, humanProgram, candidateView);
    if (!humanPrepared.succeeded() || !humanProgram.valid()) {
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = 700u +
                static_cast<std::uint32_t>(humanPrepared.status);
        }
        return failBegin(MRNX_RUNTIME_INVALID_REQUEST_V1);
    }

    metalrobo::MetalNumanXHumanMatterTransaction transaction{};
    transaction.environmentCount = 1u;
    transaction.transactionSlot = static_cast<std::uint32_t>(
        (slotGeneration - 1u) % runtime->transactionSlotCount);
    active->transactionSlot = transaction.transactionSlot;
    transaction.controlStep = static_cast<std::uint32_t>(
        root.controlStepIdentifier);
    transaction.physicsSubstep = 0u;
    transaction.physicsSubsteps = 1u;
    transaction.expectedMatterCompletedMicrosteps = 1u;
    transaction.qCoordinateCount = MRNX_FULL_BODY_NQ;
    transaction.dofCount = MRNX_FULL_BODY_NV;
    // Physical retry/replay must not depend on the private reservation slot.
    // Slot generation is already bound independently throughout the owner and
    // proof lifecycle; the Matter stochastic seed belongs to the immutable
    // causal root so a restored retry reproduces the same accepted state.
    transaction.seed = root.transactionFingerprint;
    if (transaction.seed == 0u) transaction.seed = kFnvOffset;
    transaction.transactionFingerprint = root.transactionFingerprint;
    transaction.substepFingerprint = substep.substepFingerprint;
    transaction.acceptedTimestampMicroseconds =
        substep.candidateTimestampMicroseconds;
    transaction.physicsGeneration = active->physicsGeneration;
    transaction.linearizationEpoch = linearizationEpoch;
    transaction.slotGeneration = slotGeneration;
    const auto humanMatterProgram = runtime->adapter->program(transaction);
    if (!humanMatterProgram.valid()) {
        (void)runtime->humanIO->cancelPrepared(
            root.transactionFingerprint, humanProgram.fingerprint);
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = 800u;
        }
        return failBegin(MRNX_RUNTIME_SUBMISSION_FAILURE_V1);
    }

    metalrobo::MetalArticulatedOperatorInput ownerInput{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = runtime->assets.points.size(),
        .q = runtime->assets.model.defaultQ,
        .v = runtime->assets.model.defaultV,
        .points = runtime->assets.points,
        .mujoco = {
            .muscles = runtime->assets.muscles,
            .states = runtime->assets.states,
            .sites = runtime->assets.sites,
            .wraps = runtime->assets.wraps,
            .routeNodes = runtime->assets.routes,
            .bodyJacobianPointOffset =
                runtime->assets.bodyJacobianPointOffset,
        },
        .stand = {
            .v = runtime->assets.model.defaultV,
            // The exact source witnesses are present in the point stream for
            // causal sensing. Human/Matter ABI v1 still rejects constrained
            // dynamics until it owns a nullspace/KKT tangent, so they are not
            // yet offered as solver constraints here.
            .contacts = {},
            .jointEqualities = {},
            .tendonBindings = {},
            .tendonEnvelopes = {},
            .tendonLoadProgram = {},
            .numanXTransactionProgram = humanProgram,
            .numanXHumanMatterProgram = humanMatterProgram,
            .stepCount = 1u,
            .contactIterationCount = 12u,
            .enableContact = false,
            // The attached world has zero gravity and contact constraints are
            // outside the v1 tangent authority. Artificial root springs add a
            // stiff rigid residual without representing source contact, so
            // leave assistance disabled for this free coupled transaction.
            .enableRootAssistance = false,
            .groundPoint = runtime->assets.groundPoint,
            .groundNormal = runtime->assets.groundNormal,
            .targetRootPosition = {
                runtime->assets.model.defaultQ[0u],
                runtime->assets.model.defaultQ[1u],
                runtime->assets.model.defaultQ[2u], 0.0f},
            .targetRootOrientation = {
                runtime->assets.model.defaultQ[3u],
                runtime->assets.model.defaultQ[4u],
                runtime->assets.model.defaultQ[5u],
                runtime->assets.model.defaultQ[6u]},
            .assistanceGains = {0.0f, 0.0f, 0.0f, 0.0f},
        },
        .residentContinuation = {
            .previousTransactionFingerprint =
                active->previousTransactionFingerprint,
            .previousPhysicsGeneration =
                active->previousPhysicsGeneration,
        },
    };
    auto submission = std::make_unique<
        metalrobo::MetalArticulatedOperatorSubmission>();
    const auto submitted = runtime->owner->submit(
        runtime->assets.model, ownerInput, *submission);
    if (!submitted.succeeded() || !submitted.dispatched ||
        !submission->valid()) {
        if (std::getenv("MRNX_PHYSICAL_DIAGNOSTICS") != nullptr) {
            std::fprintf(
                stderr, "mrnx_owner_submit_failure status=%u message=%s\n",
                static_cast<unsigned>(submitted.status),
                submitted.message.c_str());
        }
        (void)runtime->humanIO->cancelPrepared(
            root.transactionFingerprint, humanProgram.fingerprint);
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = 900u +
                static_cast<std::uint32_t>(submitted.status);
        }
        return failBegin(MRNX_RUNTIME_SUBMISSION_FAILURE_V1);
    }

    metalrobo::MetalNumanXHumanIOTransactionKey key{};
    const auto pending = runtime->humanIO->pendingCandidate(
        key, candidateView);
    metalrobo::MetalNumanXHumanMatterPrepared nativePrepared;
    if (!pending.succeeded() || !key.valid() ||
        !submission->extractPreparedHumanMatter(nativePrepared) ||
        !nativePrepared.valid()) {
        const std::lock_guard lock(runtime->mutex);
        runtime->quarantinedSubmission = std::move(submission);
        runtime->beginInProgress = false;
        runtime->terminalQuarantine = true;
        const mrnx_completion_v1 failed = rootCompletion(
            MRNX_COMPLETION_TERMINAL_NO_TOUCH_V1,
            static_cast<std::uint32_t>(MTLCommandBufferStatusNotEnqueued),
            slotGeneration);
        completion(completionContext, nullptr, nullptr, &failed, nullptr);
        return true;
    }
    submission.reset();
    auto* prepared = metalrobo::numanx_bridge_v1::adoptPrepared(
        runtime->domain,
        std::move(nativePrepared),
        std::static_pointer_cast<void>(runtime),
        runtime.get(),
        &runtimeTerminalCompletion);
    if (prepared == nullptr) {
        const std::lock_guard lock(runtime->mutex);
        runtime->beginInProgress = false;
        runtime->terminalQuarantine = true;
        const mrnx_completion_v1 failed = rootCompletion(
            MRNX_COMPLETION_TERMINAL_NO_TOUCH_V1,
            static_cast<std::uint32_t>(MTLCommandBufferStatusNotEnqueued),
            slotGeneration);
        completion(completionContext, nullptr, nullptr, &failed, nullptr);
        return true;
    }
    active->prepared = prepared;
    active->candidateKey = key;
    {
        const std::lock_guard lock(runtime->mutex);
        runtime->active = active;
        runtime->beginInProgress = false;
        runtime->info.status = MRNX_RUNTIME_READY_V1;
        runtime->info.request_failure_stage = 0u;
    }

    const bool physicalArmed =
        metalrobo::numanx_bridge_v1::registerPreparedPhysicalCompletion(
            prepared, active.get(), &physicalCompletion);
    const auto humanArmed = runtime->humanIO->registerCandidateCompletion(
        key, active.get(), &humanCandidateCompletion);
    if (!physicalArmed || !humanArmed.succeeded()) {
        {
            const std::lock_guard lock(active->mutex);
            if (!active->physicalSettled) {
                active->physicalSettled = true;
                active->physicalReady = false;
            }
            if (!active->humanSettled) {
                active->humanSettled = true;
                active->humanReady = false;
            }
        }
        settleActiveRoot(active);
    }
    return true;
}

} // namespace

extern "C" {

mrnx_runtime_v1* mrnx_bridge_v1_runtime_create(
    const mrnx_runtime_config_v1* config,
    mrnx_runtime_info_v1* info
) {
    @autoreleasepool {
        if (config == nullptr) {
            fillRuntimeInfoFailure(
                info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
        try {
            auto state = createRuntimeState(*config);
            auto* runtime = new (std::nothrow) mrnx_runtime_v1;
            if (runtime == nullptr) {
                fillRuntimeInfoFailure(
                    info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
                return nullptr;
            }
            runtime->state = std::move(state);
            if (info != nullptr) *info = runtime->state->info;
            return runtime;
        } catch (const RuntimeBuildFailure& failure) {
            if (std::getenv("MRNX_RUNTIME_DIAGNOSTICS") != nullptr) {
                std::fprintf(
                    stderr, "mrnx runtime create failed: %s\n",
                    failure.what());
            }
            fillRuntimeInfoFailure(info, failure.status);
            return nullptr;
        } catch (...) {
            fillRuntimeInfoFailure(
                info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
    }
}

void mrnx_bridge_v1_runtime_retain(mrnx_runtime_v1* runtime) {
    if (runtime != nullptr) {
        runtime->references.fetch_add(1u, std::memory_order_relaxed);
    }
}

void mrnx_bridge_v1_runtime_drop(mrnx_runtime_v1* runtime) {
    if (runtime != nullptr && runtime->references.fetch_sub(
            1u, std::memory_order_acq_rel) == 1u) {
        delete runtime;
    }
}

bool mrnx_bridge_v1_runtime_copy_info(
    const mrnx_runtime_v1* runtime,
    mrnx_runtime_info_v1* info
) {
    if (runtime == nullptr || runtime->state == nullptr || info == nullptr ||
        info->abi_version != MRNX_BRIDGE_ABI_V1 ||
        info->struct_size != sizeof(*info)) {
        return false;
    }
    {
        const std::lock_guard lock(runtime->state->mutex);
        *info = runtime->state->info;
        if (runtime->state->terminalQuarantine) {
            info->status = MRNX_RUNTIME_TERMINAL_QUARANTINE_V1;
        } else if (runtime->state->active != nullptr ||
                   runtime->state->beginInProgress) {
            info->status = MRNX_RUNTIME_BUSY_V1;
        }
    }
    const auto ownerStats = runtime->state->owner->stats();
    info->resident_continuation_count = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(
            ownerStats.residentContinuationSubmissionCount,
            std::numeric_limits<std::uint32_t>::max()));
    if (info->resident_continuation_count !=
        ownerStats.residentContinuationSubmissionCount) {
        info->resident_continuation_count =
            std::numeric_limits<std::uint32_t>::max();
    }
    return true;
}

bool mrnx_bridge_v1_runtime_copy_joint_anatomy(
    const mrnx_runtime_v1* runtime,
    const std::uint32_t jointIndex,
    mrnx_joint_anatomy_v1* anatomy
) {
    if (runtime == nullptr || runtime->state == nullptr || anatomy == nullptr ||
        anatomy->abi_version != MRNX_BRIDGE_ABI_V1 ||
        anatomy->struct_size != sizeof(*anatomy)) {
        return false;
    }
    const std::lock_guard lock(runtime->state->mutex);
    const auto& model = runtime->state->assets.model;
    if (jointIndex >= model.joints.size()) {
        return false;
    }
    const MRJointDescriptorGPU& joint = model.joints[jointIndex];
    if (joint.parentBody >= model.bodies.size() ||
        joint.childBody >= model.bodies.size()) {
        return false;
    }
    mrnx_joint_anatomy_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.joint_identifier = jointIndex;
    result.parent_body_identifier = joint.parentBody;
    result.child_body_identifier = joint.childBody;
    result.coordinate_offset = joint.vOffset;
    result.coordinate_count = joint.nv;
    result.parent_local_anchor[0] = joint.parentAnchor.x;
    result.parent_local_anchor[1] = joint.parentAnchor.y;
    result.parent_local_anchor[2] = joint.parentAnchor.z;
    result.child_local_anchor[0] = joint.childAnchor.x;
    result.child_local_anchor[1] = joint.childAnchor.y;
    result.child_local_anchor[2] = joint.childAnchor.z;
    const mr_float4 relative = quaternionMultiply(
        joint.parentRotation, quaternionConjugate(joint.childRotation));
    result.rest_relative_orientation[0] = relative.x;
    result.rest_relative_orientation[1] = relative.y;
    result.rest_relative_orientation[2] = relative.z;
    result.rest_relative_orientation[3] = relative.w;
    *anatomy = result;
    return true;
}

bool mrnx_bridge_v1_runtime_copy_anatomy_info(
    const mrnx_runtime_v1* runtime,
    mrnx_runtime_anatomy_info_v1* info
) {
    if (runtime == nullptr || runtime->state == nullptr || info == nullptr ||
        info->abi_version != MRNX_BRIDGE_ABI_V1 ||
        info->struct_size != sizeof(*info)) {
        return false;
    }
    const std::lock_guard lock(runtime->state->mutex);
    const RuntimeState& state = *runtime->state;
    mrnx_runtime_anatomy_info_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.body_count = state.assets.rigid.engineBodyCount;
    result.joint_count = static_cast<std::uint32_t>(
        state.assets.model.joints.size());
    result.coordinate_count = static_cast<std::uint32_t>(
        state.assets.model.dofs.size());
    result.muscle_count = static_cast<std::uint32_t>(
        state.assets.muscles.size());
    result.head_body_identifier = state.visionProfile.parentBodyIndex;
    result.model_source_fingerprint = state.assets.sourceFingerprint;
    *info = result;
    return true;
}

bool mrnx_bridge_v1_runtime_copy_joint_coordinate_anatomy(
    const mrnx_runtime_v1* runtime,
    const std::uint32_t coordinateIndex,
    mrnx_joint_coordinate_anatomy_v1* anatomy
) {
    if (runtime == nullptr || runtime->state == nullptr || anatomy == nullptr ||
        anatomy->abi_version != MRNX_BRIDGE_ABI_V1 ||
        anatomy->struct_size != sizeof(*anatomy)) {
        return false;
    }
    const std::lock_guard lock(runtime->state->mutex);
    const auto& model = runtime->state->assets.model;
    if (coordinateIndex >= model.dofs.size()) {
        return false;
    }
    const MRDofPropertiesGPU& dof = model.dofs[coordinateIndex];
    if ((dof.flags & MR_DOF_FLAG_ROOT) != 0u ||
        dof.jointIndex >= model.joints.size() ||
        dof.vIndex != coordinateIndex || dof.reserved0 != 0u ||
        dof.reserved1 != 0u) {
        return false;
    }
    const MRJointDescriptorGPU& joint = model.joints[dof.jointIndex];
    if (dof.localDof >= joint.nv || dof.localDof >= 3u ||
        joint.jointType == MR_JOINT_FUNCTION_BASED ||
        joint.jointType == MR_JOINT_FREE) {
        return false;
    }
    const mr_float4 axis = dof.localDof == 0u
        ? joint.axis0
        : (dof.localDof == 1u ? joint.axis1 : joint.axis2);
    const mr_float4 parentAxis = quaternionRotate(joint.parentRotation, axis);
    const bool linear = joint.jointType == MR_JOINT_PRISMATIC ||
        (joint.jointType == MR_JOINT_PLANAR && dof.localDof < 2u);
    const bool hasPositionLimit =
        (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u;
    const float extent = std::numeric_limits<float>::max();
    float rest = 0.0f;
    if (dof.qIndex != MR_INVALID_INDEX && dof.qIndex < model.defaultQ.size()) {
        rest = model.defaultQ[dof.qIndex];
    }
    mrnx_joint_coordinate_anatomy_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.joint_identifier = dof.jointIndex;
    result.coordinate_identifier = dof.localDof;
    result.kind = linear ? MRNX_JOINT_COORDINATE_LINEAR_V1
                         : MRNX_JOINT_COORDINATE_ANGULAR_V1;
    result.q_index = dof.qIndex;
    result.v_index = dof.vIndex;
    result.flags = hasPositionLimit
        ? MRNX_JOINT_COORDINATE_POSITION_LIMIT_V1 : 0u;
    result.parent_local_axis[0] = parentAxis.x;
    result.parent_local_axis[1] = parentAxis.y;
    result.parent_local_axis[2] = parentAxis.z;
    result.minimum_position = hasPositionLimit ? dof.limits.x : -extent;
    result.maximum_position = hasPositionLimit ? dof.limits.y : extent;
    result.rest_position = rest;
    *anatomy = result;
    return true;
}

bool mrnx_bridge_v1_runtime_copy_muscle_attachment_anatomy(
    const mrnx_runtime_v1* runtime,
    const std::uint32_t muscleIndex,
    mrnx_muscle_attachment_anatomy_v1* anatomy
) {
    if (runtime == nullptr || runtime->state == nullptr || anatomy == nullptr ||
        anatomy->abi_version != MRNX_BRIDGE_ABI_V1 ||
        anatomy->struct_size != sizeof(*anatomy)) {
        return false;
    }
    const std::lock_guard lock(runtime->state->mutex);
    const FullBodyAssets& assets = runtime->state->assets;
    if (muscleIndex >= assets.muscles.size()) {
        return false;
    }
    const MRMujocoMuscleGPU& muscle = assets.muscles[muscleIndex];
    const std::uint32_t routeOffset = muscle.route.x;
    const std::uint32_t routeCount = muscle.route.y;
    if (routeCount < 2u || routeOffset > assets.routes.size() ||
        routeCount > assets.routes.size() - routeOffset) {
        return false;
    }
    const auto& firstRoute = assets.routes[routeOffset];
    const auto& terminalRoute = assets.routes[routeOffset + routeCount - 1u];
    if (firstRoute.type != MR_MUJOCO_MUSCLE_ROUTE_SITE ||
        terminalRoute.type != MR_MUJOCO_MUSCLE_ROUTE_SITE ||
        firstRoute.targetIndex >= assets.sites.size() ||
        terminalRoute.targetIndex >= assets.sites.size()) {
        return false;
    }
    const auto& first = assets.sites[firstRoute.targetIndex];
    const auto& terminal = assets.sites[terminalRoute.targetIndex];
    mrnx_muscle_attachment_anatomy_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.muscle_identifier = muscleIndex;
    result.route_node_count = routeCount;
    result.first_body_identifier = first.bodyIndex;
    result.terminal_body_identifier = terminal.bodyIndex;
    result.first_local_point[0] = first.localPoint.x;
    result.first_local_point[1] = first.localPoint.y;
    result.first_local_point[2] = first.localPoint.z;
    result.terminal_local_point[0] = terminal.localPoint.x;
    result.terminal_local_point[1] = terminal.localPoint.y;
    result.terminal_local_point[2] = terminal.localPoint.z;
    *anatomy = result;
    return true;
}

bool mrnx_bridge_v1_runtime_begin_physical_root(
    mrnx_runtime_v1* runtime,
    const mrnx_physical_root_request_v1* request,
    void* completionContext,
    const mrnx_physical_root_settled_callback_v1 completion
) {
    @autoreleasepool {
        if (runtime == nullptr || runtime->state == nullptr ||
            request == nullptr || completion == nullptr) {
            return false;
        }
        const auto state = runtime->state;
        try {
            return beginPhysicalRoot(
                state, *request, completionContext, completion);
        } catch (...) {
            const std::lock_guard lock(state->mutex);
            state->beginInProgress = false;
            return false;
        }
    }
}

bool mrnx_bridge_v1_runtime_copy_aggregate_snapshot(
    const mrnx_runtime_v1* runtime,
    mrnx_aggregate_snapshot_v1* snapshot
) {
    if (runtime == nullptr || runtime->state == nullptr ||
        snapshot == nullptr ||
        snapshot->abi_version != MRNX_BRIDGE_ABI_V1 ||
        snapshot->struct_size != sizeof(*snapshot)) {
        return false;
    }
    const std::shared_lock reader(runtime->state->aggregateGate);
    if (runtime->state->aggregate.publication_epoch == 0u ||
        runtime->state->publishedProprioception == nil ||
        runtime->state->publishedProprioceptionValidity == nil ||
        runtime->state->publishedInteroception == nil ||
        runtime->state->publishedInteroceptionValidity == nil) {
        *snapshot = {};
        return false;
    }
    *snapshot = runtime->state->aggregate;
    return true;
}

bool mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v2(
    const mrnx_runtime_v1* runtime,
    mrnx_aggregate_snapshot_v2* snapshot
) {
    if (runtime == nullptr || runtime->state == nullptr ||
        snapshot == nullptr ||
        snapshot->abi_version != MRNX_AGGREGATE_SNAPSHOT_ABI_V2 ||
        snapshot->struct_size != sizeof(*snapshot)) {
        return false;
    }
    const std::shared_lock reader(runtime->state->aggregateGate);
    bool channelsReady = runtime->state->aggregateChannelCount == 7u;
    for (std::uint32_t index = 0u;
         index < runtime->state->aggregateChannelCount; ++index) {
        channelsReady = channelsReady &&
            runtime->state->publishedChannelValues[index] != nil &&
            runtime->state->publishedChannelValidity[index] != nil;
    }
    if (runtime->state->aggregate.publication_epoch == 0u ||
        !channelsReady) {
        *snapshot = {};
        return false;
    }
    *snapshot = {};
    snapshot->abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V2;
    snapshot->struct_size = sizeof(*snapshot);
    snapshot->publication_epoch =
        runtime->state->aggregate.publication_epoch;
    snapshot->brain_generation =
        runtime->state->aggregate.brain_generation;
    snapshot->physics_generation =
        runtime->state->aggregate.physics_generation;
    snapshot->sensor_generation =
        runtime->state->aggregate.sensor_generation;
    snapshot->root = runtime->state->aggregate.root;
    snapshot->sensor = runtime->state->aggregate.sensor;
    snapshot->channel_count = runtime->state->aggregateChannelCount;
    snapshot->channel_capacity = MRNX_MAX_SENSOR_CHANNELS_V2;
    for (std::uint32_t index = 0u;
         index < runtime->state->aggregateChannelCount; ++index) {
        snapshot->channels[index] = runtime->state->aggregateChannels[index];
    }
    return true;
}

bool mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v3(
    const mrnx_runtime_v1* runtime,
    mrnx_aggregate_snapshot_v3* snapshot
) {
    if (runtime == nullptr || runtime->state == nullptr ||
        snapshot == nullptr ||
        snapshot->abi_version != MRNX_AGGREGATE_SNAPSHOT_ABI_V3 ||
        snapshot->struct_size != sizeof(*snapshot)) {
        return false;
    }
    const std::shared_lock reader(runtime->state->aggregateGate);
    bool channelsReady = runtime->state->aggregateChannelCount == 7u;
    for (std::uint32_t index = 0u;
         index < runtime->state->aggregateChannelCount; ++index) {
        channelsReady = channelsReady &&
            runtime->state->publishedChannelValues[index] != nil &&
            runtime->state->publishedChannelValidity[index] != nil;
    }
    if (runtime->state->aggregate.publication_epoch == 0u ||
        runtime->state->aggregateTiming.timing_fingerprint == 0u ||
        !channelsReady) {
        *snapshot = {};
        return false;
    }
    *snapshot = {};
    snapshot->abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V3;
    snapshot->struct_size = sizeof(*snapshot);
    snapshot->publication_epoch =
        runtime->state->aggregate.publication_epoch;
    snapshot->brain_generation =
        runtime->state->aggregate.brain_generation;
    snapshot->physics_generation =
        runtime->state->aggregate.physics_generation;
    snapshot->sensor_generation =
        runtime->state->aggregate.sensor_generation;
    snapshot->root = runtime->state->aggregate.root;
    snapshot->sensor = runtime->state->aggregate.sensor;
    snapshot->timing = runtime->state->aggregateTiming;
    snapshot->channel_count = runtime->state->aggregateChannelCount;
    snapshot->channel_capacity = MRNX_MAX_SENSOR_CHANNELS_V2;
    for (std::uint32_t index = 0u;
         index < runtime->state->aggregateChannelCount; ++index) {
        snapshot->channels[index] = runtime->state->aggregateChannels[index];
    }
    return true;
}

} // extern "C"

namespace {

[[nodiscard]] mrnx_completion_v1 rootCompletion(
    const std::uint32_t status,
    const std::uint32_t metalStatus,
    const std::uint64_t generation
) noexcept {
    mrnx_completion_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.status = status;
    result.metal_status = metalStatus;
    result.slot_generation = generation;
    return result;
}

bool encodeSupplementalSensors(
    void* raw,
    const metalrobo::MetalNumanXTransactionPass& pass
) noexcept {
    @autoreleasepool {
        auto* active = static_cast<ActiveRoot*>(raw);
        if (active == nullptr || active->runtime == nullptr) return false;
        if (pass.phase != metalrobo::MetalNumanXTransactionPhase::postDynamics) {
            return pass.phase == metalrobo::MetalNumanXTransactionPhase::beginStep ||
                pass.phase == metalrobo::MetalNumanXTransactionPhase::preDynamics;
        }
        RuntimeState& runtime = *active->runtime;
        __unsafe_unretained id<MTLCommandBuffer> commandBuffer = nil;
        __unsafe_unretained id<MTLBuffer> q = nil;
        __unsafe_unretained id<MTLBuffer> v = nil;
        __unsafe_unretained id<MTLBuffer> bodyPoses = nil;
        __unsafe_unretained id<MTLBuffer> pointWorld = nil;
        __unsafe_unretained id<MTLBuffer> standStatuses = nil;
        const auto supportView =
            runtime.matter->humanSupportCandidateConsequences();
        __unsafe_unretained id<MTLBuffer> supportConsequences =
            (__bridge id<MTLBuffer>)supportView.buffer;
        const bool passValid =
            pass.abiVersion == metalrobo::kMetalNumanXTransactionABIVersion &&
            pass.structSize == sizeof(pass) && pass.reserved0 == 0u &&
            pass.stepIndex == 0u && pass.stepCount == 1u &&
            pass.environmentCount == 1u &&
            pass.qCoordinateCount == MRNX_FULL_BODY_NQ &&
            pass.qStride == MRNX_FULL_BODY_NQ &&
            pass.qElementCount >= MRNX_FULL_BODY_NQ &&
            pass.dofCount == MRNX_FULL_BODY_NV &&
            pass.vStride == MRNX_FULL_BODY_NV &&
            pass.vElementCount >= MRNX_FULL_BODY_NV &&
            pass.bodyCount == runtime.assets.rigid.engineBodyCount &&
            pass.bodyPoseStride == runtime.assets.rigid.engineBodyCount &&
            pass.bodyPoseElementCount >= runtime.assets.rigid.engineBodyCount &&
            pass.pointCount == runtime.assets.points.size() &&
            pass.pointWorldStride == runtime.assets.points.size() &&
            pass.pointWorldElementCount >= runtime.assets.points.size() &&
            pass.standStatusStride == 1u &&
            pass.standStatusElementCount >= 1u &&
            pass.timestepSeconds ==
                runtime.visionProfile.depthAndTimestep.w &&
            pass.programFingerprint != 0u &&
            commandBufferObject(pass.commandBuffer, commandBuffer) &&
            bufferObject(pass.q, q) && bufferObject(pass.v, v) &&
            bufferObject(pass.bodyPoses, bodyPoses) &&
            bufferObject(pass.pointWorld, pointWorld) &&
            bufferObject(pass.standStatuses, standStatuses) &&
            q.device == runtime.device && v.device == runtime.device &&
            bodyPoses.device == runtime.device &&
            pointWorld.device == runtime.device &&
            standStatuses.device == runtime.device &&
            supportConsequences != nil &&
            supportConsequences.device == runtime.device &&
            supportView.gpuAddress ==
                active->supplementalDispatch
                    .expectedSupportConsequencesGPUAddress &&
            supportView.elementCount ==
                active->supplementalDispatch.supportPointCount &&
            supportView.stride ==
                active->supplementalDispatch.supportPointCount &&
            active->activeSensing.buffer.device == runtime.device &&
            runtime.visualBodyBounds.device == runtime.device &&
            runtime.supplementalPipeline != nil;
        if (!passValid) return false;

        struct Region {
            __unsafe_unretained id<MTLBuffer> buffer;
            std::uint64_t address;
            std::uint64_t bytes;
        };
        const Region regions[] = {
            {q, q.gpuAddress, MRNX_FULL_BODY_NQ * sizeof(float)},
            {v, v.gpuAddress, MRNX_FULL_BODY_NV * sizeof(float)},
            {bodyPoses, bodyPoses.gpuAddress,
             runtime.assets.rigid.engineBodyCount *
                sizeof(MRArticulatedBodyPoseGPU)},
            {pointWorld, pointWorld.gpuAddress,
             runtime.assets.points.size() * sizeof(MRArticulatedPointWorldGPU)},
            {standStatuses, standStatuses.gpuAddress,
             sizeof(MRNumiHumanStandStatusGPU)},
            {supportConsequences, supportView.gpuAddress,
             supportView.elementCount *
                sizeof(MRNumanXHumanSupportConsequenceGPU)},
            {active->activeSensing.buffer, active->activeSensing.address,
             active->activeSensing.byteCount},
            {runtime.visualBodyBounds, runtime.visualBodyBounds.gpuAddress,
             runtime.visualBodyBounds.length},
            {active->kinesthesia, active->kinesthesia.gpuAddress,
             active->kinesthesia.length},
            {active->kinesthesiaValidity,
             active->kinesthesiaValidity.gpuAddress,
             active->kinesthesiaValidity.length},
            {active->vestibular, active->vestibular.gpuAddress,
             active->vestibular.length},
            {active->vestibularValidity,
             active->vestibularValidity.gpuAddress,
             active->vestibularValidity.length},
            {active->audition, active->audition.gpuAddress,
             active->audition.length},
            {active->auditionValidity, active->auditionValidity.gpuAddress,
             active->auditionValidity.length},
            {active->vision, active->vision.gpuAddress, active->vision.length},
            {active->visionValidity, active->visionValidity.gpuAddress,
             active->visionValidity.length},
            {active->touch, active->touch.gpuAddress, active->touch.length},
            {active->touchValidity, active->touchValidity.gpuAddress,
             active->touchValidity.length},
        };
        for (const Region& region : regions) {
            if (region.buffer == nil || region.address == 0u ||
                region.bytes == 0u || region.buffer.device != runtime.device ||
                region.bytes > region.buffer.length) {
                return false;
            }
        }
        for (std::size_t first = 0u; first < std::size(regions); ++first) {
            for (std::size_t second = first + 1u;
                 second < std::size(regions); ++second) {
                if (regions[first].buffer == regions[second].buffer ||
                    !disjoint(
                        regions[first].address, regions[first].bytes,
                        regions[second].address, regions[second].bytes)) {
                    return false;
                }
            }
        }
        id<MTLComputeCommandEncoder> encoder =
            [commandBuffer computeCommandEncoder];
        if (encoder == nil) return false;
        encoder.label = @"NumanX physical supplemental sensors";
        [encoder setComputePipelineState:runtime.supplementalPipeline];
        [encoder setBuffer:q offset:0u atIndex:0u];
        [encoder setBuffer:v offset:0u atIndex:1u];
        [encoder setBuffer:bodyPoses offset:0u atIndex:2u];
        [encoder setBuffer:pointWorld offset:0u atIndex:3u];
        [encoder setBuffer:standStatuses offset:0u atIndex:4u];
        [encoder setBuffer:active->activeSensing.buffer offset:0u atIndex:5u];
        [encoder setBuffer:runtime.visualBodyBounds offset:0u atIndex:6u];
        [encoder setBuffer:active->kinesthesia offset:0u atIndex:7u];
        [encoder setBuffer:active->kinesthesiaValidity offset:0u atIndex:8u];
        [encoder setBuffer:active->vestibular offset:0u atIndex:9u];
        [encoder setBuffer:active->vestibularValidity offset:0u atIndex:10u];
        [encoder setBuffer:active->audition offset:0u atIndex:11u];
        [encoder setBuffer:active->auditionValidity offset:0u atIndex:12u];
        [encoder setBuffer:active->vision offset:0u atIndex:13u];
        [encoder setBuffer:active->visionValidity offset:0u atIndex:14u];
        [encoder setBuffer:active->touch offset:0u atIndex:15u];
        [encoder setBuffer:active->touchValidity offset:0u atIndex:16u];
        [encoder setBytes:&active->supplementalDispatch
            length:sizeof(active->supplementalDispatch) atIndex:17u];
        [encoder setBuffer:supportConsequences offset:0u atIndex:18u];
        const NSUInteger width = std::max<NSUInteger>(
            1u,
            std::min<NSUInteger>(
                256u, runtime.supplementalPipeline.maxTotalThreadsPerThreadgroup));
        [encoder dispatchThreads:
            MTLSizeMake(MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        [encoder endEncoding];
        return true;
    }
}

void runtimeTerminalCompletion(
    void* raw,
    const PreparedTerminalDisposition disposition,
    const mrnx_root_v1& root,
    const mrnx_candidate_view_v1* candidate,
    const mrnx_candidate_channel_v1* channels,
    const std::uint32_t channelCount
) noexcept {
    auto* runtime = static_cast<RuntimeState*>(raw);
    if (runtime == nullptr) return;
    std::shared_ptr<ActiveRoot> active;
    {
        const std::lock_guard lock(runtime->mutex);
        if (runtime->active == nullptr ||
            runtime->active->slotGeneration != root.slot_generation) {
            if (disposition != PreparedTerminalDisposition::rejected) {
                runtime->terminalQuarantine = true;
            }
            return;
        }
        active = runtime->active;
        if (disposition != PreparedTerminalDisposition::published) {
            runtime->active.reset();
        }
        if (disposition == PreparedTerminalDisposition::terminalNoTouch) {
            runtime->terminalQuarantine = true;
        }
    }
    if (disposition != PreparedTerminalDisposition::published) {
        return;
    }
    const bool commonIdentityValid = active != nullptr &&
        candidate != nullptr && channels != nullptr && channelCount == 7u &&
        candidate->channel_count == channelCount &&
        root.transaction_fingerprint == active->transactionFingerprint &&
        root.control_step == active->controlStep &&
        candidate->accepted_brain_generation == active->brainGeneration &&
        candidate->key.transaction_fingerprint ==
            active->transactionFingerprint &&
        candidate->key.sensor_generation != 0u;
    if (!commonIdentityValid) {
        const std::lock_guard lock(runtime->mutex);
        if (runtime->active == active) runtime->active.reset();
        runtime->terminalQuarantine = true;
        return;
    }
    const mrnx_candidate_channel_v1* proprioception = nullptr;
    const mrnx_candidate_channel_v1* interoception = nullptr;
    __unsafe_unretained id<MTLBuffer> channelValues[
        MRNX_MAX_SENSOR_CHANNELS_V2]{};
    __unsafe_unretained id<MTLBuffer> channelValidity[
        MRNX_MAX_SENSOR_CHANNELS_V2]{};
    bool channelSetValid = true;
    for (std::uint32_t index = 0u; index < channelCount; ++index) {
        const auto& channel = channels[index];
        std::uint32_t expectedReceptors = 0u;
        std::uint32_t expectedFeatures = 0u;
        switch (channel.modality) {
            case MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1:
                expectedReceptors = MRNX_FULL_BODY_MUSCLE_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1:
                expectedReceptors = MRNX_FULL_BODY_MUSCLE_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_KINESTHESIA_V1:
                expectedReceptors =
                    MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_VESTIBULAR_V1:
                expectedReceptors =
                    MR_NUMANX_HUMAN_VESTIBULAR_RECEPTOR_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_VESTIBULAR_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_AUDITION_V1:
                expectedReceptors =
                    MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_AUDITION_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_VISION_V1:
                expectedReceptors = MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT;
                expectedFeatures = MR_NUMANX_HUMAN_VISION_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_TOUCH_V1:
                expectedReceptors = MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT;
                expectedFeatures = MR_NUMANX_HUMAN_TOUCH_FEATURE_COUNT;
                break;
            default:
                break;
        }
        channelSetValid = channelSetValid && expectedReceptors != 0u &&
            channel.receptor_count == expectedReceptors &&
            channel.feature_dimension == expectedFeatures &&
            channel.receptor_timestamp_microseconds ==
                active->receptorTimestampMicroseconds &&
            channel.flags == MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1 &&
            bufferObject(channel.values.metal_buffer, channelValues[index]) &&
            bufferObject(
                channel.validity.metal_buffer, channelValidity[index]) &&
            channelValues[index].device == runtime->device &&
            channelValidity[index].device == runtime->device &&
            channel.values.gpu_address == channelValues[index].gpuAddress &&
            channel.validity.gpu_address ==
                channelValidity[index].gpuAddress &&
            channel.values.byte_offset == 0u &&
            channel.validity.byte_offset == 0u &&
            channel.values.byte_count == channelValues[index].length &&
            channel.validity.byte_count == channelValidity[index].length;
        if (channels[index].modality ==
            MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1) {
            proprioception = &channels[index];
        } else if (channels[index].modality ==
                   MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1) {
            interoception = &channels[index];
        }
    }
    __unsafe_unretained id<MTLBuffer> proprioceptionValues = nil;
    __unsafe_unretained id<MTLBuffer> proprioceptionValidity = nil;
    __unsafe_unretained id<MTLBuffer> interoceptionValues = nil;
    __unsafe_unretained id<MTLBuffer> interoceptionValidity = nil;
    if (!channelSetValid || proprioception == nullptr ||
        interoception == nullptr ||
        !bufferObject(
            proprioception->values.metal_buffer, proprioceptionValues) ||
        !bufferObject(
            proprioception->validity.metal_buffer,
            proprioceptionValidity) ||
        !bufferObject(
            interoception->values.metal_buffer, interoceptionValues) ||
        !bufferObject(
            interoception->validity.metal_buffer,
            interoceptionValidity) ||
        proprioceptionValues.device != runtime->device ||
        proprioceptionValidity.device != runtime->device ||
        interoceptionValues.device != runtime->device ||
        interoceptionValidity.device != runtime->device) {
        const std::lock_guard lock(runtime->mutex);
        if (runtime->active == active) runtime->active.reset();
        runtime->terminalQuarantine = true;
        return;
    }
    std::unique_lock runtimeLock(runtime->mutex);
    if (runtime->active != active || runtime->terminalQuarantine) {
        runtime->terminalQuarantine = true;
        if (runtime->active == active) runtime->active.reset();
        return;
    }
    std::unique_lock aggregateWriter(runtime->aggregateGate);
    const std::uint64_t priorEpoch = runtime->aggregate.publication_epoch;
    if (priorEpoch == std::numeric_limits<std::uint64_t>::max()) {
        runtime->active.reset();
        runtime->terminalQuarantine = true;
        return;
    }
    mrnx_aggregate_snapshot_v1 snapshot{};
    mrnx_candidate_timing_v1 timing{};
    if (runtime->timestepMicroseconds == 0u ||
        runtime->timestepMicroseconds >
            std::numeric_limits<std::uint32_t>::max() ||
        runtime->timestepMicroseconds >
            std::numeric_limits<std::uint64_t>::max() -
                active->receptorTimestampMicroseconds ||
        active->acceptedTimestampMicroseconds !=
            active->receptorTimestampMicroseconds +
                runtime->timestepMicroseconds) {
        runtime->active.reset();
        runtime->terminalQuarantine = true;
        return;
    }
    timing.abi_version = MRNX_BRIDGE_ABI_V1;
    timing.struct_size = sizeof(timing);
    timing.capture_timestamp_microseconds =
        active->receptorTimestampMicroseconds;
    timing.delivery_timestamp_microseconds =
        active->acceptedTimestampMicroseconds;
    timing.latency_microseconds = static_cast<std::uint32_t>(
        runtime->timestepMicroseconds);
    timing.sample_interval_microseconds = static_cast<std::uint32_t>(
        runtime->timestepMicroseconds);
    timing.timing_fingerprint = timingFingerprint(timing);
    snapshot.abi_version = MRNX_BRIDGE_ABI_V1;
    snapshot.struct_size = sizeof(snapshot);
    snapshot.publication_epoch = priorEpoch + 1u;
    snapshot.brain_generation = candidate->accepted_brain_generation;
    snapshot.physics_generation = active->physicsGeneration;
    snapshot.sensor_generation = candidate->key.sensor_generation;
    snapshot.root = root;
    snapshot.sensor = *candidate;
    snapshot.proprioception = *proprioception;
    snapshot.interoception = *interoception;
    runtime->publishedProprioception = proprioceptionValues;
    runtime->publishedProprioceptionValidity = proprioceptionValidity;
    runtime->publishedInteroception = interoceptionValues;
    runtime->publishedInteroceptionValidity = interoceptionValidity;
    runtime->aggregateChannelCount = channelCount;
    for (std::uint32_t index = 0u; index < channelCount; ++index) {
        runtime->aggregateChannels[index] = channels[index];
        runtime->publishedChannelValues[index] = channelValues[index];
        runtime->publishedChannelValidity[index] = channelValidity[index];
    }
    runtime->aggregate = snapshot;
    runtime->aggregateTiming = timing;
    runtime->publishedOnce = true;
    runtime->publishedTransactionFingerprint =
        active->transactionFingerprint;
    runtime->publishedBrainGeneration = active->brainGeneration;
    runtime->publishedPhysicsGeneration = active->physicsGeneration;
    runtime->publishedTimestampMicroseconds =
        active->acceptedTimestampMicroseconds;
    runtime->publishedControlStep = active->controlStep;
    runtime->active.reset();
}

void settleActiveRoot(const std::shared_ptr<ActiveRoot>& active) noexcept {
    if (active == nullptr || active->runtime == nullptr) return;
    mrnx_prepared_v1* prepared = nullptr;
    mrnx_candidate_v1* candidate = nullptr;
    mrnx_physical_root_settled_callback_v1 callback = nullptr;
    void* callbackContext = nullptr;
    std::uint64_t generation = 0u;
    bool ready = false;
    {
        const std::lock_guard lock(active->mutex);
        if (active->settlementStarted || !active->humanSettled ||
            !active->physicalSettled) {
            return;
        }
        active->settlementStarted = true;
        prepared = active->prepared;
        candidate = active->candidate;
        callback = active->completion;
        callbackContext = active->completionContext;
        generation = active->slotGeneration;
        ready = active->humanReady && active->physicalReady &&
            prepared != nullptr && candidate != nullptr;
    }

    mrnx_root_v1 root{};
    if (prepared != nullptr) {
        root.abi_version = MRNX_BRIDGE_ABI_V1;
        root.struct_size = sizeof(root);
        (void)mrnx_bridge_v1_prepared_copy_root(prepared, &root);
    }
    if (ready) {
        ready = mrnx_bridge_v1_bind_candidate(prepared, candidate);
    }
    if (!ready) {
        if (candidate != nullptr) {
            (void)mrnx_bridge_v1_reject_unbound_candidate(candidate);
        }
        if (prepared != nullptr) {
            metalrobo::numanx_bridge_v1::markPreparedPhysicalTerminal(
                prepared);
        }
    }

    const mrnx_completion_v1 completion = rootCompletion(
        ready ? MRNX_COMPLETION_READY_V1
              : MRNX_COMPLETION_TERMINAL_NO_TOUCH_V1,
        static_cast<std::uint32_t>(
            ready ? MTLCommandBufferStatusCompleted
                  : MTLCommandBufferStatusError),
        generation);
    if (callback != nullptr) {
        callback(
            callbackContext,
            ready ? prepared : nullptr,
            ready ? candidate : nullptr,
            &completion,
            ready ? &root : nullptr);
    }
    // adoptPrepared/adoptCandidate create one external reference for delivery
    // plus their lifecycle self-hold. The callback must explicitly retain the
    // borrowed handles if it wants to progress asynchronously after return.
    if (candidate != nullptr) mrnx_bridge_v1_candidate_drop(candidate);
    if (prepared != nullptr) mrnx_bridge_v1_prepared_drop(prepared);
}

[[nodiscard]] mrnx_metal_range_v1 outputRange(
    id<MTLBuffer> buffer,
    const mrnx_element_type_v1 type,
    const std::uint32_t elementBytes
) noexcept {
    mrnx_metal_range_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.metal_buffer = (__bridge void*)buffer;
    result.gpu_address = buffer != nil ? buffer.gpuAddress : 0u;
    result.byte_count = buffer != nil ? buffer.length : 0u;
    result.element_type = type;
    result.element_byte_count = elementBytes;
    return result;
}

[[nodiscard]] mrnx_candidate_channel_v1 supplementalChannel(
    const std::uint32_t modality,
    const std::uint64_t receptorTimestampMicroseconds,
    const std::uint32_t receptorCount,
    const std::uint32_t featureCount,
    id<MTLBuffer> values,
    id<MTLBuffer> validity
) noexcept {
    mrnx_candidate_channel_v1 result{};
    result.abi_version = MRNX_BRIDGE_ABI_V1;
    result.struct_size = sizeof(result);
    result.modality = modality;
    result.flags = MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1;
    result.receptor_timestamp_microseconds = receptorTimestampMicroseconds;
    result.receptor_count = receptorCount;
    result.feature_dimension = featureCount;
    result.values = outputRange(
        values, MRNX_ELEMENT_FLOAT32_V1, sizeof(float));
    result.validity = outputRange(
        validity, MRNX_ELEMENT_UINT32_V1, sizeof(std::uint32_t));
    return result;
}

void humanCandidateCompletion(
    void* raw,
    const metalrobo::MetalNumanXHumanIOCandidateCompletionStatus status,
    const metalrobo::MetalNumanXHumanIOTransactionKey& key,
    const metalrobo::MetalNumanXHumanIOSensorView&
) noexcept {
    auto* pointer = static_cast<ActiveRoot*>(raw);
    if (pointer == nullptr) return;
    const auto active = pointer->shared_from_this();
    mrnx_candidate_v1* candidate = nullptr;
    bool ready = status ==
        metalrobo::MetalNumanXHumanIOCandidateCompletionStatus::succeeded &&
        sameKey(key, active->candidateKey);
    if (!ready) {
        const std::lock_guard runtimeLock(active->runtime->mutex);
        active->runtime->info.request_failure_stage = 1000u +
            static_cast<std::uint32_t>(status);
    }
    if (ready) {
        metalrobo::MetalNumanXHumanIOCandidatePublicationLease lease;
        const auto reserved = active->runtime->humanIO->
            reserveCandidatePublication(key, lease);
        ready = reserved.succeeded() && lease.valid();
        if (ready) {
            candidate = metalrobo::numanx_bridge_v1::adoptCandidate(
                active->runtime->domain, std::move(lease));
            ready = candidate != nullptr;
        }
        if (ready) {
            const mrnx_candidate_channel_v1 supplemental[] = {
                supplementalChannel(
                    MRNX_CANDIDATE_MODALITY_KINESTHESIA_V1,
                    active->receptorTimestampMicroseconds,
                    MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT,
                    MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT,
                    active->kinesthesia, active->kinesthesiaValidity),
                supplementalChannel(
                    MRNX_CANDIDATE_MODALITY_VESTIBULAR_V1,
                    active->receptorTimestampMicroseconds,
                    MR_NUMANX_HUMAN_VESTIBULAR_RECEPTOR_COUNT,
                    MR_NUMANX_HUMAN_VESTIBULAR_FEATURE_COUNT,
                    active->vestibular, active->vestibularValidity),
                supplementalChannel(
                    MRNX_CANDIDATE_MODALITY_AUDITION_V1,
                    active->receptorTimestampMicroseconds,
                    MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT,
                    MR_NUMANX_HUMAN_AUDITION_FEATURE_COUNT,
                    active->audition, active->auditionValidity),
                supplementalChannel(
                    MRNX_CANDIDATE_MODALITY_VISION_V1,
                    active->receptorTimestampMicroseconds,
                    MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT,
                    MR_NUMANX_HUMAN_VISION_FEATURE_COUNT,
                    active->vision, active->visionValidity),
                supplementalChannel(
                    MRNX_CANDIDATE_MODALITY_TOUCH_V1,
                    active->receptorTimestampMicroseconds,
                    MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT,
                    MR_NUMANX_HUMAN_TOUCH_FEATURE_COUNT,
                    active->touch, active->touchValidity),
            };
            ready = metalrobo::numanx_bridge_v1::attachCandidateChannels(
                candidate, supplemental,
                static_cast<std::uint32_t>(std::size(supplemental)));
            if (!ready) {
                (void)mrnx_bridge_v1_reject_unbound_candidate(candidate);
                mrnx_bridge_v1_candidate_drop(candidate);
                candidate = nullptr;
            }
        }
    }
    {
        const std::lock_guard lock(active->mutex);
        if (active->humanSettled) {
            if (candidate != nullptr) {
                (void)mrnx_bridge_v1_reject_unbound_candidate(candidate);
                mrnx_bridge_v1_candidate_drop(candidate);
            }
            return;
        }
        active->humanSettled = true;
        active->humanReady = ready;
        active->candidate = candidate;
    }
    settleActiveRoot(active);
}

void physicalCompletion(
    void* raw,
    const bool ready,
    const std::uint64_t slotGeneration
) noexcept {
    auto* pointer = static_cast<ActiveRoot*>(raw);
    if (pointer == nullptr) return;
    const auto active = pointer->shared_from_this();
    {
        const std::lock_guard lock(active->mutex);
        if (active->physicalSettled) return;
        active->physicalSettled = true;
        active->physicalReady = ready &&
            slotGeneration == active->slotGeneration;
        metalrobo::MetalNumanXHumanMatterPhysicalOutcome outcome{};
        const bool hasOutcome = active->physicalReady &&
            active->runtime->adapter->physicalOutcome(
                active->transactionSlot,
                active->transactionFingerprint,
                active->slotGeneration,
                outcome);
        const std::lock_guard runtimeLock(active->runtime->mutex);
        if (!active->physicalReady || !hasOutcome) {
            active->runtime->info.request_failure_stage = 1100u;
        } else if (outcome.humanCode != MR_NUMI_HUMAN_STAND_SUCCESS) {
            active->runtime->info.request_failure_stage =
                2000u + outcome.humanCode;
        } else if (outcome.matterCode != 0u) {
            active->runtime->info.request_failure_stage =
                3000u + outcome.matterCode;
        } else if (outcome.worldCode != MR_STEP_SUCCESS) {
            active->runtime->info.request_failure_stage =
                4000u + outcome.worldCode;
        } else if (outcome.jointDecision !=
                   MR_NUMANX_COUPLED_HUMAN_ACCEPT) {
            active->runtime->info.request_failure_stage =
                5000u + outcome.jointDecision;
        }
        if (hasOutcome &&
            std::getenv("MRNX_PHYSICAL_DIAGNOSTICS") != nullptr) {
            std::fprintf(
                stderr,
                "mrnx_matter_status code=%u object=%u index=%u "
                "fgmres=%u contacts=%u diagnostics=[%.9g,%.9g,%.9g,%.9g]\n",
                outcome.matterCode, outcome.matterObjectIndex,
                outcome.matterFailingIndex,
                outcome.matterFGMRESIterations,
                outcome.matterContactCount,
                static_cast<double>(outcome.matterDiagnostics[0]),
                static_cast<double>(outcome.matterDiagnostics[1]),
                static_cast<double>(outcome.matterDiagnostics[2]),
                static_cast<double>(outcome.matterDiagnostics[3]));
            std::fprintf(
                stderr,
                "mrnx_human_status code=%u index=%u contacts=%u "
                "iterations=%u contact_accel=[%.9g,%.9g,%.9g,%.9g] "
                "factor_assist=[%.9g,%.9g,%.9g,%.9g]\n",
                outcome.humanCode, outcome.humanFailingIndex,
                outcome.humanActiveContactCount,
                outcome.humanContactIterations,
                static_cast<double>(outcome.humanContactAndAcceleration[0]),
                static_cast<double>(outcome.humanContactAndAcceleration[1]),
                static_cast<double>(outcome.humanContactAndAcceleration[2]),
                static_cast<double>(outcome.humanContactAndAcceleration[3]),
                static_cast<double>(outcome.humanFactorAndAssistance[0]),
                static_cast<double>(outcome.humanFactorAndAssistance[1]),
                static_cast<double>(outcome.humanFactorAndAssistance[2]),
                static_cast<double>(outcome.humanFactorAndAssistance[3]));
        }
    }
    settleActiveRoot(active);
}

[[nodiscard]] bool validateRootRequest(
    const std::shared_ptr<RuntimeState>& runtime,
    const mrnx_physical_root_request_v1& request,
    std::shared_ptr<ActiveRoot>& active,
    MRNumanXBrainJointTransactionToken& root,
    MRNumanXBrainJointSubstepToken& substep,
    MRNumanXBrainMotorCandidate& candidate,
    std::uint32_t& failureStage
) noexcept {
    failureStage = 1u;
    if (runtime == nullptr || request.abi_version != MRNX_BRIDGE_ABI_V1 ||
        request.struct_size != sizeof(request) ||
        request.root.control_step_identifier >
            std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    std::memcpy(&root, &request.root, sizeof(root));
    std::memcpy(&substep, &request.substep, sizeof(substep));
    std::memcpy(&candidate, &request.candidate, sizeof(candidate));
    failureStage = 10u;
    if (root.formatVersion != MR_NUMANX_BRAIN_JOINT_TRANSACTION_VERSION ||
        root.environmentIdentifier != 0u || root.reserved != 0u ||
        root.flags != 0u || root.parameterVersionFingerprint == 0u ||
        root.transactionFingerprint == 0u ||
        root.transactionFingerprint !=
            metalrobo::metalNumanXBrainJointTransactionFingerprint(root) ||
        root.baseBrainGeneration ==
            std::numeric_limits<std::uint64_t>::max() ||
        root.shadowGeneration != root.baseBrainGeneration + 1u ||
        root.basePhysicsGeneration ==
            std::numeric_limits<std::uint64_t>::max() ||
        root.committedTimestampMicroseconds >
            std::numeric_limits<std::uint64_t>::max() -
                runtime->timestepMicroseconds ||
        root.targetTimestampMicroseconds !=
            root.committedTimestampMicroseconds +
                runtime->timestepMicroseconds) {
        return false;
    }
    failureStage = 2u;
    if (substep.transactionFingerprint != root.transactionFingerprint ||
        substep.substepIndex != 0u || substep.attemptIndex != 0u ||
        substep.flags != 0u || substep.reserved != 0u ||
        substep.durationMicroseconds != runtime->timestepMicroseconds ||
        substep.startTimestampMicroseconds !=
            root.committedTimestampMicroseconds ||
        substep.startTimestampMicroseconds >
            std::numeric_limits<std::uint64_t>::max() -
                substep.durationMicroseconds ||
        substep.candidateTimestampMicroseconds !=
            substep.startTimestampMicroseconds +
                substep.durationMicroseconds ||
        substep.candidateTimestampMicroseconds !=
            root.targetTimestampMicroseconds ||
        substep.shadowGeneration != root.shadowGeneration ||
        substep.randomCounterGeneration != root.randomCounterGeneration ||
        substep.substepFingerprint == 0u ||
        substep.substepFingerprint !=
            metalrobo::metalNumanXBrainJointSubstepFingerprint(substep)) {
        return false;
    }
    failureStage = 3u;
    if (candidate.formatVersion !=
            MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VERSION ||
        candidate.flags !=
            (MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
             MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW) ||
        candidate.transactionFingerprint != root.transactionFingerprint ||
        candidate.substepFingerprint != substep.substepFingerprint ||
        candidate.acceptedBrainTimestampMicroseconds !=
            substep.startTimestampMicroseconds ||
        candidate.brainGeneration != root.shadowGeneration ||
        candidate.randomCounterGeneration !=
            root.randomCounterGeneration ||
        candidate.muscleCount != MRNX_FULL_BODY_MUSCLE_COUNT ||
        candidate.environmentIdentifier != 0u ||
        candidate.actuatorCommandKind !=
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION ||
        candidate.reserved != 0u ||
        candidate.motorOutputHeaderByteCount !=
            MR_NUMANX_BRAIN_MOTOR_OUTPUT_HEADER_BYTE_COUNT ||
        candidate.muscleExcitationByteCount !=
            MRNX_FULL_BODY_MUSCLE_COUNT * sizeof(float) ||
        candidate.autonomicCommandByteCount !=
            MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT ||
        candidate.activeSensingCommandByteCount !=
            MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT ||
        candidate.autonomicCommandCount == 0u ||
        candidate.activeSensingCommandCount == 0u ||
        candidate.candidateFingerprint == 0u ||
        candidate.candidateFingerprint !=
            metalrobo::metalNumanXBrainMotorCandidateFingerprint(candidate)) {
        return false;
    }
    failureStage = 4u;
    auto result = std::make_shared<ActiveRoot>();
    result->runtime = runtime.get();
    failureStage = 41u;
    if (!importExactRange(
            runtime->device, request.motor_header,
            MR_NUMANX_BRAIN_MOTOR_OUTPUT_HEADER_BYTE_COUNT,
            MRNX_ELEMENT_RAW_BYTES_V1, 1u, result->motorHeader)) {
        return false;
    }
    failureStage = 42u;
    if (!importExactRange(
            runtime->device, request.muscle_excitation,
            MRNX_FULL_BODY_MUSCLE_COUNT * sizeof(float),
            MRNX_ELEMENT_FLOAT32_V1, sizeof(float), result->excitation)) {
        return false;
    }
    failureStage = 43u;
    if (!importExactRange(
            runtime->device, request.autonomic_command,
            MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT,
            MRNX_ELEMENT_RAW_BYTES_V1, 1u, result->autonomic)) {
        return false;
    }
    failureStage = 44u;
    if (!importExactRange(
            runtime->device, request.active_sensing_command,
            MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT,
            MRNX_ELEMENT_RAW_BYTES_V1, 1u, result->activeSensing)) {
        return false;
    }
    failureStage = 45u;
    if (!importExactRange(
            runtime->device, request.motor_ready_gate,
            MR_NUMANX_BRAIN_MOTOR_READY_GATE_BYTE_COUNT,
            MRNX_ELEMENT_RAW_BYTES_V1, 1u, result->motorReadyGate)) {
        return false;
    }
    failureStage = 5u;
    const ImportedRange ranges[] = {
        result->motorHeader, result->excitation, result->autonomic,
        result->activeSensing, result->motorReadyGate};
    for (std::size_t first = 0u; first < std::size(ranges); ++first) {
        for (std::size_t second = first + 1u;
             second < std::size(ranges); ++second) {
            if (ranges[first].buffer == ranges[second].buffer ||
                !disjoint(
                    ranges[first].address, ranges[first].byteCount,
                    ranges[second].address, ranges[second].byteCount)) {
                return false;
            }
        }
    }
    failureStage = 6u;
    __unsafe_unretained id<MTLSharedEvent> event = nil;
    if (request.motor_ready.abi_version != MRNX_BRIDGE_ABI_V1 ||
        request.motor_ready.struct_size != sizeof(request.motor_ready) ||
        request.motor_ready.value == 0u ||
        request.motor_ready.device_registry_id !=
            runtime->device.registryID ||
        !eventObject(request.motor_ready.shared_event, event) ||
        !importableSharedEvent(runtime->device, event) ||
        candidate.motorOutputHeaderGPUAddress !=
            result->motorHeader.address ||
        candidate.muscleExcitationGPUAddress != result->excitation.address ||
        candidate.autonomicCommandGPUAddress != result->autonomic.address ||
        candidate.activeSensingCommandGPUAddress !=
            result->activeSensing.address) {
        return false;
    }
    result->motorReadyEvent = event;
    result->transactionFingerprint = root.transactionFingerprint;
    result->brainGeneration = root.shadowGeneration;
    result->controlStep = root.controlStepIdentifier;
    {
        const std::lock_guard lock(runtime->mutex);
        if (runtime->publishedOnce) {
            if (runtime->publishedTransactionFingerprint == 0u ||
                runtime->publishedBrainGeneration == 0u ||
                runtime->publishedPhysicsGeneration == 0u ||
                runtime->publishedTimestampMicroseconds == 0u ||
                root.baseBrainGeneration !=
                    runtime->publishedBrainGeneration ||
                root.basePhysicsGeneration !=
                    runtime->publishedPhysicsGeneration ||
                root.committedTimestampMicroseconds !=
                    runtime->publishedTimestampMicroseconds ||
                root.controlStepIdentifier <= runtime->publishedControlStep) {
                return false;
            }
            result->previousTransactionFingerprint =
                runtime->publishedTransactionFingerprint;
            result->previousPhysicsGeneration =
                runtime->publishedPhysicsGeneration;
        } else if (root.baseBrainGeneration != 0u ||
                   root.basePhysicsGeneration != 0u ||
                   runtime->aggregate.publication_epoch != 0u) {
            return false;
        }
        if (runtime->lastAttemptedControlStep ==
                std::numeric_limits<std::uint64_t>::max() ||
            root.controlStepIdentifier !=
                runtime->lastAttemptedControlStep + 1u) {
            return false;
        }
    }
    active = std::move(result);
    failureStage = 0u;
    return true;
}

} // namespace
