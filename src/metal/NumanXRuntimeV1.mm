#include "metalrobo/NumiHumanSupport.hpp"
#include "metalrobo/NumiHumanInitialState.hpp"
#include "metalrobo/NumiHumanProductionOwnerEvidenceWriter.hpp"
#include "metalrobo/NumiHumanProductionOwnerSnapshot.hpp"
#include "metalrobo/NumiHumanRuntimeIdentity.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "NumanXBridgeV1Internal.hpp"

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalNeuronCulture.hpp"
#include "metalrobo/MetalHumanBehaviorTelemetry.hpp"
#include "metalrobo/mrnx_human_behavior_v1.h"
#include "metalrobo/MetalNumanXHumanMatter.hpp"
#include "metalrobo/NeuronCultureArtifacts.hpp"
#include "metalrobo/NumiHumanTissueBinding.hpp"
#include "metalrobo/NumanXExactTransaction.hpp"
#include <CommonCrypto/CommonDigest.h>
#include "metalrobo/VisualPresentation.hpp"
#include "numi/matter/detail.hpp"
#include "numi/matter/matter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace metalrobo {

struct MetalNeuronCultureRuntimeBridgeAccess {
    [[nodiscard]] static MetalNeuronCultureAcceptedView preparedView(
        const MetalNeuronCultureRuntime& runtime
    ) noexcept {
        return runtime.preparedAcceptedView();
    }
};

} // namespace metalrobo

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
constexpr std::uint32_t kRigidABI = 1u;
constexpr std::uint32_t kLegacyMuscleABI = 1u;
constexpr std::uint32_t kMuscleABI = 2u;
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

struct JointEqualityHeader {
    std::array<char, 8u> magic{};
    std::uint32_t abi = 0u, nq = 0u, nv = 0u, count = 0u;
    std::uint32_t recordBytes = 0u, sourceCount = 0u;
    std::uint32_t policy = 0u, flags = 0u, reserved0 = 0u, reserved1 = 0u;
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

using SupportContactHeader = metalrobo::NumiHumanSupportHeader;
using SupportContactRecord = metalrobo::NumiHumanSupportContact;
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(JointEqualityHeader) == 80u);
static_assert(sizeof(SupportContactHeader) == 84u);

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
static_assert(sizeof(mrnx_brain_joint_transaction_v2) ==
              sizeof(MRNumanXBrainJointTransactionTokenV2));
static_assert(sizeof(mrnx_brain_joint_substep_v2) ==
              sizeof(MRNumanXBrainJointSubstepTokenV2));
static_assert(sizeof(mrnx_brain_motor_candidate_v2) ==
              sizeof(MRNumanXBrainMotorCandidateV2));
static_assert(sizeof(mrnx_brain_motor_output_header_v2) ==
              sizeof(MRNumanXBrainMotorOutputHeaderGPUV2));
static_assert(alignof(mrnx_brain_motor_output_header_v2) ==
              alignof(MRNumanXBrainMotorOutputHeaderGPUV2));
static_assert(sizeof(mrnx_brain_motor_ready_gate_v2) ==
              sizeof(MRNumanXBrainMotorReadyGateGPUV2));
static_assert(alignof(mrnx_brain_motor_ready_gate_v2) ==
              alignof(MRNumanXBrainMotorReadyGateGPUV2));

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
    std::array<std::uint8_t,32u> rigidSHA256{};
    std::vector<std::uint32_t> sourceMap;
    MuscleHeader muscle{};
    std::vector<MRMujocoMuscleSiteGPU> sites;
    std::vector<MRMujocoMuscleWrapGPU> wraps;
    std::vector<MRMujocoMuscleRouteNodeGPU> routes;
    std::vector<MRMujocoMuscleGPU> muscles;
    std::vector<MRMujocoMuscleStateGPU> states;
    std::vector<float> initialQ;
    std::vector<MRCompensatedRootTranslationGPU> initialRootTranslations;
    std::vector<float> initialV;
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::vector<MRNumiHumanStandContactGPU> supportContacts;
    std::vector<mr_uint4> touchSupportMapping;
    std::vector<NMHumanJointEqualityGPU> jointEqualities;
    NMHumanEqualityDispatchGPU equalityDispatch{};
    std::uint64_t equalityFingerprint = 0u;
    std::vector<NMHumanJointLimitGPU> jointLimits;
    NMHumanLimitDispatchGPU limitDispatch{};
    std::uint64_t limitFingerprint = 0u;
    std::vector<NMHumanSupportContactGPU> matterSupportContacts;
    std::vector<NMHumanSupportPointQueryGPU> matterSupportPointQueries;
    metalrobo::NumiHumanSupportPayloadIdentity supportIdentity{};
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

[[nodiscard]] std::uint64_t cultureReceiptFingerprint(
    mrnx_culture_prepared_view_v1 view
) noexcept {
    view.receipt_fingerprint = 0u;
    return hashBytes(&view, sizeof(view));
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
    return metalrobo::numiHumanRuntimeBaseSourceFingerprint(
        std::as_bytes(std::span(rigid.bytes)),
        std::as_bytes(std::span(muscle.bytes)),
        std::as_bytes(std::span(support.bytes)));
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

void loadJointEqualities(FullBodyAssets& assets,
                         const mrnx_runtime_config_v4& config) {
    const ImmutablePayload image = loadImmutablePayload(
        config.joint_equality_payload_path, "NHEQ2 source joint equalities");
    const std::uint64_t fingerprint = hashBytes(image.bytes.data(), image.bytes.size());
    requireBuild(fingerprint == config.expected_joint_equality_fingerprint,
        MRNX_RUNTIME_ASSET_FAILURE_V1, "NHEQ2 immutable payload fingerprint mismatch");
    std::istringstream input(image.bytes, std::ios::in | std::ios::binary);
    JointEqualityHeader header{};
    readObject(input, header, "NHEQ2 header");
    const std::array<char,8u> magic{'N','H','E','Q','2','\0','\0','\0'};
    requireBuild(header.magic == magic && header.abi == 2u &&
        header.nq == assets.rigid.nq && header.nv == assets.rigid.nv &&
        header.count != 0u && header.count <= header.nv &&
        header.count == header.sourceCount && header.recordBytes == sizeof(NMHumanJointEqualityGPU) &&
        header.policy == NM_HUMAN_EQUALITY_POLICY_MUJOCO_312_CLASSIC &&
        (header.flags & ~NM_HUMAN_EQUALITY_REFSAFE) == 0u &&
        header.reserved0 == 0u && header.reserved1 == 0u &&
        header.sourceSHA256 == assets.rigid.sourceSHA256 &&
        image.bytes.size() == sizeof(header) + header.count * sizeof(NMHumanJointEqualityGPU),
        MRNX_RUNTIME_ASSET_FAILURE_V1, "NHEQ2 header/source identity or policy mismatch");
    assets.jointEqualities = readVector<NMHumanJointEqualityGPU>(input, header.count, "NHEQ2 rows");
    assets.equalityDispatch.count = header.count;
    assets.equalityDispatch.qCount = header.nq;
    assets.equalityDispatch.dofCount = header.nv;
    assets.equalityDispatch.policy = header.policy;
    assets.equalityDispatch.flags = header.flags;
    assets.equalityFingerprint = fingerprint;
    // The base world admission was checked before adding this source owner.
    assets.sourceFingerprint = metalrobo::numiHumanRuntimeAppendPayloadOwner(
        assets.sourceFingerprint, "NHEQ2", fingerprint);
}

void loadJointLimits(FullBodyAssets& assets, const mrnx_runtime_config_v6& config) {
    const ImmutablePayload image = loadImmutablePayload(config.joint_limit_payload_path,
        "NHLIM1 source joint limits");
    const std::uint64_t fingerprint = hashBytes(image.bytes.data(), image.bytes.size());
    requireBuild(fingerprint == config.expected_joint_limit_fingerprint,
        MRNX_RUNTIME_ASSET_FAILURE_V1, "NHLIM1 immutable payload fingerprint mismatch");
    std::istringstream input(image.bytes, std::ios::in | std::ios::binary);
    // Scalar programs share the 80-byte source/policy envelope; records differ.
    JointEqualityHeader header{};
    readObject(input, header, "NHLIM1 header");
    const std::array<char,8u> magic{'N','H','L','I','M','1','\0','\0'};
    requireBuild(header.magic == magic && header.abi == NM_HUMAN_LIMIT_ABI_VERSION &&
        header.nq == assets.rigid.nq && header.nv == assets.rigid.nv && header.nv > 6u &&
        header.count != 0u && header.count <= header.nv - 6u && header.count == header.sourceCount &&
        header.recordBytes == sizeof(NMHumanJointLimitGPU) &&
        header.policy == assets.equalityDispatch.policy && header.flags == assets.equalityDispatch.flags &&
        assets.equalityDispatch.count != 0u && header.reserved0 == 0u && header.reserved1 == 0u &&
        header.sourceSHA256 == assets.rigid.sourceSHA256 &&
        image.bytes.size() == sizeof(header) + header.count * sizeof(NMHumanJointLimitGPU),
        MRNX_RUNTIME_ASSET_FAILURE_V1, "NHLIM1 header/source identity or NHEQ2 policy mismatch");
    assets.jointLimits = readVector<NMHumanJointLimitGPU>(input, header.count, "NHLIM1 rows");
    // Native hard-range flags omit source reset coordinates just beyond tiny
    // limits. NHLIM1 is authoritative for those too; do not require that flag.
    std::vector<bool> covered(header.nv, false);
    for (const auto& row : assets.jointLimits) {
        requireBuild(row.indices.y >= 6u && row.indices.y < header.nv &&
            row.indices.x == row.indices.y + 1u && !covered[row.indices.y],
            MRNX_RUNTIME_ASSET_FAILURE_V1, "NHLIM1 native coordinate binding is invalid");
        const auto& dof = assets.model.dofs[row.indices.y];
        requireBuild(dof.vIndex == row.indices.y && dof.qIndex == row.indices.x &&
            ((dof.flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
             (dof.limits.x == row.rangeMarginInverseWeight.x &&
              dof.limits.y == row.rangeMarginInverseWeight.y)),
            MRNX_RUNTIME_ASSET_FAILURE_V1, "NHLIM1 native coordinate range disagrees");
        covered[row.indices.y] = true;
    }
    for (std::size_t i = 6u; i < assets.model.dofs.size(); ++i)
        requireBuild((assets.model.dofs[i].flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u || covered[i],
            MRNX_RUNTIME_ASSET_FAILURE_V1, "NHLIM1 omitted a native authored range");
    assets.limitDispatch.count = header.count;
    assets.limitDispatch.qCount = header.nq;
    assets.limitDispatch.dofCount = header.nv;
    assets.limitDispatch.policy = header.policy;
    assets.limitDispatch.flags = header.flags;
    assets.limitFingerprint = fingerprint;
    assets.sourceFingerprint = metalrobo::numiHumanRuntimeAppendPayloadOwner(
        assets.sourceFingerprint, "NHLIM1", fingerprint);
}

VisionProfile loadVisionProfile(
    const std::string& packPath,
    const std::string& profilePath,
    const std::uint32_t bodyCount,
    const double timestepSeconds
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
        static_cast<float>(timestepSeconds)};
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
        supportContactPath, "NHCNT support-contact payload");
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
    result.sourceMap = readVector<std::uint32_t>(
        rigidInput, result.rigid.sourceBodyCount, "NHRIGID2 source map");
    requireBuild(rigidImage.bytes.size() <= std::numeric_limits<CC_LONG>::max(),
        MRNX_RUNTIME_ASSET_FAILURE_V1, "rigid image exceeds SHA256 input extent");
    CC_SHA256(rigidImage.bytes.data(), static_cast<CC_LONG>(rigidImage.bytes.size()), result.rigidSHA256.data());
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
    for (const std::uint32_t body : result.sourceMap) {
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
    const std::vector<char> supportRaw((std::istreambuf_iterator<char>(supportInput)), {});
    requireBuild(supportRaw.size() >= sizeof(SupportContactHeader) &&
        supportRaw.size() <= std::numeric_limits<CC_LONG>::max(),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "NHCNT payload extent is outside support-identity bounds");
    SupportContactHeader rawSupportHeader{};
    std::memcpy(&rawSupportHeader, supportRaw.data(), sizeof(rawSupportHeader));
    result.supportIdentity.byteCount = supportRaw.size();
    result.supportIdentity.payloadABI = rawSupportHeader.payloadAbi;
    result.supportIdentity.sourceRecordCount = rawSupportHeader.contactCount;
    CC_SHA256(supportRaw.data(), static_cast<CC_LONG>(supportRaw.size()),
        result.supportIdentity.sha256.data());
    metalrobo::NumiHumanSupportPayload supportPayload;
    std::string supportError;
    requireBuild(metalrobo::decodeNumiHumanSupportPayload(std::as_bytes(std::span(supportRaw)),
        result.rigid.engineBodyCount, result.rigid.sourceSHA256, supportPayload, supportError),
        MRNX_RUNTIME_ASSET_FAILURE_V1, supportError);
    const auto& supportHeader = supportPayload.header;
    const auto& supportRecords = supportPayload.contacts;
    requireBuild(supportRecords.size() <=
            std::numeric_limits<std::uint32_t>::max(),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "expanded NHCNT support rows exceed identity capacity");
    result.supportIdentity.expandedRowCount =
        static_cast<std::uint32_t>(supportRecords.size());
    result.groundPoint = {supportHeader.groundPointX, supportHeader.groundPointY,
        supportHeader.groundPointZ, 0.0f};
    result.groundNormal = {supportHeader.groundNormalX, supportHeader.groundNormalY,
        supportHeader.groundNormalZ, 0.0f};

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
        const MRArticulatedPointImpulseGPU point =
            metalrobo::compileNumiHumanSupportQuery(supportHeader, source);
        const auto pointIndex = static_cast<std::uint32_t>(
            result.points.size());
        result.points.push_back(point);
        NMHumanSupportPointQueryGPU matterPoint{};
        std::memcpy(&matterPoint, &point, sizeof(point));
        result.matterSupportPointQueries.push_back(matterPoint);
        MRNumiHumanStandContactGPU contact{};
        contact.bodyIndex = source.bodyIndex;
        contact.pointQueryIndex = pointIndex;
        contact.sourceGeometryIndex = source.sourceGeometryIndex;
        contact.frictionSlopAndStabilization = {
            source.friction,
            std::max(source.defaultSignedPlaneDistance, 0.0f) + 0.001f,
            0.2f,
            0.0f};
        result.supportContacts.push_back(contact);
        NMHumanSupportContactGPU matterContact{};
        matterContact.identity = {
            source.bodyIndex,
            source.sourceGeometryIndex,
            static_cast<std::uint32_t>(
                result.matterSupportContacts.size()),
            source.supportRadii[0] > 0 ? 2u : (source.supportRadius > 0 ? 1u : 0u)};
        matterContact.localPoint = {
            point.localPoint.x, point.localPoint.y,
            point.localPoint.z, source.supportRadius};
        matterContact.supportRadii = {point.supportRadii.x,point.supportRadii.y,point.supportRadii.z,0};
        matterContact.supportOrientation = {point.supportOrientation.x,point.supportOrientation.y,point.supportOrientation.z,point.supportOrientation.w};
        matterContact.frictionSlopAndStabilization = {
            contact.frictionSlopAndStabilization.x,
            contact.frictionSlopAndStabilization.y,
            contact.frictionSlopAndStabilization.z,
            contact.frictionSlopAndStabilization.w};
        result.matterSupportContacts.push_back(matterContact);
    }
    for (std::uint32_t row=0; row<result.supportContacts.size(); ++row) {
        const auto& contact=result.supportContacts[row];
        auto& groups=result.touchSupportMapping;
        if (!groups.empty() && groups.back().z==contact.bodyIndex && groups.back().w==contact.sourceGeometryIndex) {
            requireBuild(groups.back().y==1u, MRNX_RUNTIME_ASSET_FAILURE_V1, "too many rows per source touch geometry");
            ++groups.back().y;
        } else {
            requireBuild(std::none_of(groups.begin(),groups.end(),[&](const auto& group){
                return group.w==contact.sourceGeometryIndex;}), MRNX_RUNTIME_ASSET_FAILURE_V1,
                "source touch rows are not contiguous");
            groups.push_back({row,1u,contact.bodyIndex,contact.sourceGeometryIndex});
        }
    }
    requireBuild(result.touchSupportMapping.size()==MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT,
        MRNX_RUNTIME_ASSET_FAILURE_V1,"source support must map to all ten geometry receptors");
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
    const double timestepSeconds
) {
    const auto parsed = numi::matter::parseMatterFile(materialPath);
    requireBuild(
        parsed.succeeded(), MRNX_RUNTIME_MATTER_FAILURE_V1,
        "Matter material did not parse");
    numi::matter::WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = timestepSeconds;
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

template<class Point>
void shiftTissuePoint(Point& point, const std::uint32_t body,
                     const std::vector<std::array<double,3>>& offsets) {
    requireBuild(body<offsets.size(),MRNX_RUNTIME_ASSET_FAILURE_V1,
                 "tissue frame target escapes Human body table");
    point.x=static_cast<float>(point.x-offsets[body][0]);
    point.y=static_cast<float>(point.y-offsets[body][1]);
    point.z=static_cast<float>(point.z-offsets[body][2]);
}

// Construction-only mass ownership. Geometry and mass come from the final
// cooked package; all local points are then rebased together before allocation.
[[nodiscard]] std::vector<std::array<double,3>> prepareCostalMassOwnership(
    FullBodyAssets& assets, const mrnx_runtime_config_v5& config) {
    const auto& authored=config.runtime.runtime;
    const auto cartilageImage=loadImmutablePayload(config.costal_cartilage_payload_path,"costal cartilage");
    const auto bindingImage=loadImmutablePayload(config.costal_binding_payload_path,"costal binding");
    const auto rigidImage=loadImmutablePayload(authored.runtime.rigid_payload_path,"rigid tissue donor");
    requireBuild(hashBytes(bindingImage.bytes.data(),bindingImage.bytes.size())==config.expected_costal_binding_fingerprint,
                 MRNX_RUNTIME_ASSET_FAILURE_V1,"costal binding fingerprint mismatch");
    const auto span=[](const ImmutablePayload& p){return std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(p.bytes.data()),p.bytes.size());};
    const auto sha=[](const ImmutablePayload& p){
        requireBuild(p.bytes.size()<=std::numeric_limits<CC_LONG>::max(),MRNX_RUNTIME_ASSET_FAILURE_V1,"tissue input exceeds SHA input bound");
        std::array<std::uint8_t,32> result{};
        CC_SHA256(p.bytes.data(),static_cast<CC_LONG>(p.bytes.size()),result.data());return result;};
    metalrobo::NumiHumanCostalBinding binding;std::string error;
    requireBuild(metalrobo::decodeNumiHumanCostalBinding(span(bindingImage),sha(cartilageImage),sha(rigidImage),binding,error),
                 MRNX_RUNTIME_ASSET_FAILURE_V1,"costal binding admission failed: "+error);
    metalrobo::NumiHumanCostalCartilagePayload cartilage;
    requireBuild(metalrobo::decodeNumiHumanCostalCartilagePayload(span(cartilageImage),{},cartilage).succeeded(),
                 MRNX_RUNTIME_ASSET_FAILURE_V1,"invalid costal cartilage payload");
    numi::matter::CompiledWorld world;
    requireBuild(numi::matter::readPackage(authored.matter_world_package_path,world,nullptr,&error),
                 MRNX_RUNTIME_ASSET_FAILURE_V1,"costal mass package failed: "+error);
    requireBuild(world.fingerprint==authored.expected_matter_world_fingerprint && world.objects.size()==1 &&
                 world.fem.nodes.size()==cartilage.nodes.size() && world.fem.tetrahedra.size()==cartilage.tetrahedra.size() &&
                 binding.bodyCount==assets.model.bodies.size() && binding.nodeCount==cartilage.nodes.size() &&
                 binding.tetrahedronCount==cartilage.tetrahedra.size(),MRNX_RUNTIME_ASSET_FAILURE_V1,"costal mass package coverage mismatch");
    const std::vector<double> q(assets.model.defaultQ.begin(),assets.model.defaultQ.end()),v(assets.model.defaultV.begin(),assets.model.defaultV.end());
    std::vector<metalrobo::ArticulatedBodyKinematics> bodies(assets.model.bodies.size());
    requireBuild(metalrobo::computeArticulatedBodyKinematics(assets.model,0,q,v,bodies).succeeded(),
                 MRNX_RUNTIME_ASSET_FAILURE_V1,"costal donor kinematics failed");
    std::vector<metalrobo::NumiHumanTissueMassNode> nodes;
    const auto& a=binding.atlasToWorld;
    for(std::uint32_t i=0;i<cartilage.nodes.size();++i) {
        const auto& original=cartilage.nodes[i];const auto& cooked=world.fem.nodes[i];
        const std::array<float,3> actual{cooked.positionAndMass.x,cooked.positionAndMass.y,cooked.positionAndMass.z};
        for(unsigned axis=0;axis<3;++axis) {
            const float expected=static_cast<float>(a[4*axis]*original.restPosition[0]+a[4*axis+1]*original.restPosition[1]+a[4*axis+2]*original.restPosition[2]+a[4*axis+3]);
            requireBuild(actual[axis]==expected,MRNX_RUNTIME_ASSET_FAILURE_V1,"costal mass package changes registered rest geometry");
        }
        requireBuild(cooked.restAndFixed.w==(original.flags?2.0f:0.0f),MRNX_RUNTIME_ASSET_FAILURE_V1,"costal attachment mask drifted");
        const auto donor=binding.regions[original.regionIndex].donorBody;
        const auto& b=bodies[donor];const auto& rotation=b.orientation;
        const std::array<double,3> p{actual[0]-b.centerOfMassPosition[0],actual[1]-b.centerOfMassPosition[1],actual[2]-b.centerOfMassPosition[2]};
        const std::array<double,3> t{2*(-rotation[1]*p[2]+rotation[2]*p[1]),2*(-rotation[2]*p[0]+rotation[0]*p[2]),2*(-rotation[0]*p[1]+rotation[1]*p[0])};
        nodes.push_back({i,donor,cooked.positionAndMass.w,
                        {p[0]+rotation[3]*t[0]-rotation[1]*t[2]+rotation[2]*t[1],
                         p[1]+rotation[3]*t[1]-rotation[2]*t[0]+rotation[0]*t[2],
                         p[2]+rotation[3]*t[2]-rotation[0]*t[1]+rotation[1]*t[0]}});
    }
    for(unsigned i=0;i<cartilage.tetrahedra.size();++i) {
        const auto& t=world.fem.tetrahedra[i].nodes;
        requireBuild(cartilage.tetrahedra[i].node==std::array<std::uint32_t,4>{t.x,t.y,t.z,t.w},
                     MRNX_RUNTIME_ASSET_FAILURE_V1,"costal mass package topology drifted");
    }
    std::vector<bool> attached(nodes.size());std::size_t expectedAttachments=0;
    for(const auto& n:cartilage.nodes) expectedAttachments+=n.flags!=0;
    requireBuild(world.fem.humanAttachments.size()==expectedAttachments,MRNX_RUNTIME_ASSET_FAILURE_V1,"costal attachment coverage drifted");
    for(const auto& attachment:world.fem.humanAttachments) {
        const auto n=attachment.identity.x;
        requireBuild(n<nodes.size()&&!attached[n]&&cartilage.nodes[n].flags!=0&&
                     attachment.identity.y==binding.regions[cartilage.nodes[n].regionIndex].donorBody,
                     MRNX_RUNTIME_ASSET_FAILURE_V1,"costal source attachment ownership drifted");
        attached[n]=true;
    }
    const auto mass=metalrobo::compileNumiHumanTissueMassPartition(assets.model.bodies,nodes);
    requireBuild(mass.succeeded(),MRNX_RUNTIME_ASSET_FAILURE_V1,"costal mass partition failed: "+mass.error);
    metalrobo::EngineModel rebased;
    requireBuild(metalrobo::rebaseNumiHumanTissueMassPartition(assets.model,mass.partitions,rebased,error),
                 MRNX_RUNTIME_ASSET_FAILURE_V1,"costal COM rebase failed: "+error);
    std::vector<std::array<double,3>> offsets(bodies.size());
    for(const auto& p:mass.partitions) offsets[p.donorBody]=p.remainingCOMOffsetM;
    for(auto& site:assets.sites) shiftTissuePoint(site.localPoint,site.bodyIndex,offsets);
    for(auto& wrap:assets.wraps) shiftTissuePoint(wrap.localCenter,wrap.bodyIndex,offsets);
    // The final four-per-body points are basis probes at the NEW COM. Every
    // preceding query represents a physical source point and must be rebased.
    for(unsigned i=0;i<assets.bodyJacobianPointOffset;++i)
        shiftTissuePoint(assets.points[i].localPoint,assets.points[i].bodyIndex,offsets);
    for(auto& point:assets.matterSupportPointQueries) shiftTissuePoint(point.localPoint,point.bodyIndex,offsets);
    for(auto& contact:assets.matterSupportContacts) shiftTissuePoint(contact.localPoint,contact.identity.x,offsets);
    assets.model=std::move(rebased);
    return offsets;
}

// This is asset admission at construction, never a second stepping path.
[[nodiscard]] numi::matter::CompiledWorld loadAuthoredWorld(
    const mrnx_runtime_config_v3& config,
    const FullBodyAssets& assets,
    const std::vector<metalrobo::ArticulatedBodyKinematics>& bodies,
    const double expectedTimestepSeconds
) {
    requireBuild(
        config.expected_model_source_fingerprint == assets.sourceFingerprint,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "authored Matter world targets a different Human source");
    numi::matter::CompiledWorld world;
    std::string error;
    const bool loaded = numi::matter::readPackage(
        config.matter_world_package_path, world, nullptr, &error);
    requireBuild(
        loaded, MRNX_RUNTIME_ASSET_FAILURE_V1,
        "authored Matter package validation failed: " + error);
    const auto& dispatch = world.dispatch;
    const auto& humanGravity = assets.model.world.gravityAndTimestep;
    if (std::getenv("MRNX_RUNTIME_DIAGNOSTICS") != nullptr) {
        std::fprintf(stderr,
            "mrnx authored world=%llu expected=%llu physics=%llu env=%u rate=%u "
            "flags=%u nv=%u nq=%u attachments=%zu proxies=%zu "
            "dt=%.9g expected_dt=%.9g gravity=%.9g,%.9g,%.9g human=%.9g,%.9g,%.9g\n",
            static_cast<unsigned long long>(world.fingerprint),
            static_cast<unsigned long long>(config.expected_matter_world_fingerprint),
            static_cast<unsigned long long>(world.physicsFingerprint),
            dispatch.environmentCount, dispatch.maximumRateExponent, dispatch.flags,
            dispatch.rigidGeneralizedCapacity, dispatch.rigidQCapacity,
            world.fem.humanAttachments.size(), world.contact.rigidProxies.size(),
            dispatch.gravityAndTimestep.w,
            static_cast<float>(expectedTimestepSeconds),
            dispatch.gravityAndTimestep.x, dispatch.gravityAndTimestep.y, dispatch.gravityAndTimestep.z,
            humanGravity.x, humanGravity.y, humanGravity.z);
    }
    requireBuild(
        world.fingerprint == config.expected_matter_world_fingerprint &&
            world.physicsFingerprint != 0u &&
            dispatch.environmentCount == 1u &&
            dispatch.maximumRateExponent == 0u &&
            (dispatch.flags & NM_MATTER_DETERMINISTIC) != 0u &&
            (dispatch.flags & (NM_MATTER_ADAPTIVE | NM_MATTER_MUTATION)) == 0u &&
            dispatch.rigidGeneralizedCapacity ==
                MR_NUMANX_COUPLED_HUMAN_MAX_DOFS &&
            dispatch.rigidQCapacity == MR_NUMANX_COUPLED_HUMAN_MAX_Q &&
            !world.fem.humanAttachments.empty() &&
            world.contact.rigidProxies.empty() &&
            dispatch.gravityAndTimestep.w == static_cast<float>(
                static_cast<float>(expectedTimestepSeconds)) &&
            dispatch.gravityAndTimestep.x == humanGravity.x &&
            dispatch.gravityAndTimestep.y == humanGravity.y &&
            dispatch.gravityAndTimestep.z == humanGravity.z,
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "authored Matter package identity or joint-runtime contract mismatch");
    // The line-muscle owner still contributes every source J^T share.
    // Admitting additional active-fibre stress here would duplicate authority.
    // A future replacement map must remove that share before enabling it.
    for (const auto& material : world.mixedMaterials) {
        requireBuild(material.fibre.w == 0.0f,
                     MRNX_RUNTIME_ASSET_FAILURE_V1,
                     "active Matter fibre requires a muscle-force replacement map");
    }
    for (const auto& object : world.objects) {
        requireBuild((object.flags & NM_OBJECT_TWO_WAY_COUPLED) != 0u,
                     MRNX_RUNTIME_ASSET_FAILURE_V1,
                     "authored Matter object is not two-way coupled");
    }
    for (const auto& attachment : world.fem.humanAttachments) {
        requireBuild(attachment.identity.y < bodies.size(),
                     MRNX_RUNTIME_ASSET_FAILURE_V1,
                     "authored Matter attachment escapes Human body table");
        const auto& body = bodies[attachment.identity.y];
        const auto& q = body.orientation;
        const std::array<double, 3u> p{
            attachment.localPoint.x, attachment.localPoint.y,
            attachment.localPoint.z};
        const std::array<double, 3u> t{
            2.0 * (q[1] * p[2] - q[2] * p[1]),
            2.0 * (q[2] * p[0] - q[0] * p[2]),
            2.0 * (q[0] * p[1] - q[1] * p[0])};
        const std::array<double, 3u> expected{
            body.centerOfMassPosition[0] + p[0] + q[3] * t[0] +
                q[1] * t[2] - q[2] * t[1],
            body.centerOfMassPosition[1] + p[1] + q[3] * t[1] +
                q[2] * t[0] - q[0] * t[2],
            body.centerOfMassPosition[2] + p[2] + q[3] * t[2] +
                q[0] * t[1] - q[1] * t[0]};
        const std::array<double, 3u> offset{
            expected[0] - body.centerOfMassPosition[0],
            expected[1] - body.centerOfMassPosition[1],
            expected[2] - body.centerOfMassPosition[2]};
        const auto& omega = body.angularVelocity;
        const std::array<double, 3u> expectedVelocity{
            body.linearVelocity[0] + omega[1] * offset[2] - omega[2] * offset[1],
            body.linearVelocity[1] + omega[2] * offset[0] - omega[0] * offset[2],
            body.linearVelocity[2] + omega[0] * offset[1] - omega[1] * offset[0]};
        const auto& node = world.fem.nodes[attachment.identity.x];
        const std::array<double, 3u> actualVelocity{
            node.velocityAndInverseMass.x, node.velocityAndInverseMass.y,
            node.velocityAndInverseMass.z};
        const std::array<double, 3u> actual{
            node.positionAndMass.x, node.positionAndMass.y,
            node.positionAndMass.z};
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            // Only FP32 packing/kinematics roundoff is admitted. This is not
            // registration, prestrain relaxation or endpoint relocation.
            const double scale = std::max({1.0, std::abs(expected[axis]),
                                           std::abs(actual[axis])});
            requireBuild(std::isfinite(expected[axis]) &&
                std::abs(expected[axis] - actual[axis]) <=
                    16.0 * std::numeric_limits<float>::epsilon() * scale,
                MRNX_RUNTIME_ASSET_FAILURE_V1,
                "authored Matter attachment does not match initial Human frame");
            const double velocityScale = std::max({1.0,
                std::abs(expectedVelocity[axis]), std::abs(actualVelocity[axis])});
            requireBuild(std::isfinite(expectedVelocity[axis]) &&
                std::abs(expectedVelocity[axis] - actualVelocity[axis]) <=
                    16.0 * std::numeric_limits<float>::epsilon() * velocityScale,
                MRNX_RUNTIME_ASSET_FAILURE_V1,
                "authored Matter attachment velocity disagrees with Human");
        }
    }
    return world;
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
    pass.rootTranslation = source.rootTranslation;
    pass.rootTranslationGPUAddress = source.rootTranslationGPUAddress;
    pass.rootTranslationElementCount = source.rootTranslationElementCount;
    pass.rootTranslationStride = source.rootTranslationStride;
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

bool encodeRuntimeProofV2(
    void* context,
    const metalrobo::MetalNumanXHumanMatterStateProofPassV2& source
) noexcept {
    auto* runtime = static_cast<numi::matter::Runtime*>(context);
    if (runtime == nullptr ||
        source.abiVersion !=
            MR_NUMANX_HUMAN_MATTER_EXACT_ADAPTER_ABI_VERSION ||
        source.structSize != sizeof(source)) {
        return false;
    }
    numi::matter::AcceptedStateProofPassV2 pass{};
    pass.environmentCount = source.environmentCount;
    pass.environmentIdentifierBase = source.environmentIdentifierBase;
    pass.commandBuffer = source.commandBuffer;
    pass.q = source.q;
    pass.rootTranslation = source.rootTranslation;
    pass.rootTranslationGPUAddress = source.rootTranslationGPUAddress;
    pass.rootTranslationElementCount = source.rootTranslationElementCount;
    pass.rootTranslationStride = source.rootTranslationStride;
    pass.v = source.v;
    pass.mujocoStates = source.mujocoStates;
    pass.matterGeneralizedReaction = source.matterGeneralizedReaction;
    pass.environmentStatuses = source.environmentStatuses;
    pass.matterStatuses = source.matterStatuses;
    pass.acceptedStateProofs = source.acceptedStateProofs;
    pass.inboundAuthority = source.inboundAuthority;
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
    pass.inboundAuthorityGPUAddress = source.inboundAuthorityGPUAddress;
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
    pass.inboundAuthorityByteCount = source.inboundAuthorityByteCount;
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
    pass.clockDomain = source.clockDomain;
    pass.clockQuantumNanoseconds = source.clockQuantumNanoseconds;
    pass.reserved0 = source.reserved0;
    pass.programFingerprint = source.programFingerprint;
    pass.stateProofProgramFingerprint =
        source.stateProofProgramFingerprint;
    pass.transactionFingerprint = source.transactionFingerprint;
    pass.substepFingerprint = source.substepFingerprint;
    pass.acceptedTimestampNanoseconds =
        source.acceptedTimestampNanoseconds;
    pass.physicsGeneration = source.physicsGeneration;
    pass.linearizationEpoch = source.linearizationEpoch;
    pass.slotGeneration = source.slotGeneration;
    pass.matterSourcePhysicsFingerprint =
        source.matterSourcePhysicsFingerprint;
    pass.matterDeviceProgramFingerprint =
        source.matterDeviceProgramFingerprint;
    pass.motorCandidateFingerprint = source.motorCandidateFingerprint;
    return runtime->encodeAcceptedStateProofV2(pass);
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

// Scalar request-v3 admission only. This validates the caller's descriptor
// declaration without bridging, messaging, importing, or retaining the
// borrowed Metal object. The executable v3 lane follows with independent
// object/device/base-address authentication before any GPU submission.
[[nodiscard]] bool validateExactRangeDescriptorMetadata(
    const mrnx_metal_range_v1& range,
    const std::uint64_t expectedBytes,
    const mrnx_element_type_v1 expectedType,
    const std::uint32_t expectedElementBytes,
    const std::uint64_t expectedAlignment
) noexcept {
    std::uint64_t end = 0u;
    return range.abi_version == MRNX_BRIDGE_ABI_V1 &&
        range.struct_size == sizeof(range) &&
        range.metal_buffer != nullptr &&
        range.byte_count == expectedBytes &&
        range.element_type == expectedType &&
        range.element_byte_count == expectedElementBytes &&
        expectedAlignment != 0u &&
        range.gpu_address % expectedAlignment == 0u &&
        range.byte_offset % expectedAlignment == 0u &&
        range.byte_offset <=
            std::numeric_limits<std::uint64_t>::max() - expectedBytes &&
        checkedEnd(range.gpu_address, expectedBytes, end);
}

struct ImportedRange {
    __strong id<MTLBuffer> buffer = nil;
    std::uint64_t address = 0u;
    std::uint64_t byteOffset = 0u;
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
    output.byteOffset = range.byte_offset;
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

struct OwnerSnapshotCaptureLayout {
    std::uint64_t checkpointQ = 0u;
    std::uint64_t checkpointV = 0u;
    std::uint64_t checkpointRoot = 0u;
    std::uint64_t checkpointMuscles = 0u;
    std::uint64_t effectiveTangentFactorStorage = 0u;
    std::uint64_t sourceGeneralizedForce = 0u;
    std::uint64_t sourcePredictedVelocity = 0u;
    std::uint64_t matterGeneralizedReaction = 0u;
    std::uint64_t ownerStatus = 0u;
    std::uint64_t candidateQ = 0u;
    std::uint64_t candidateV = 0u;
    std::uint64_t candidateRoot = 0u;
    std::uint64_t candidateMuscles = 0u;
    std::uint64_t muscleResults = 0u;
    std::uint64_t muscleGeneralizedForces = 0u;
    std::uint64_t reducedMuscleGeneralizedForce = 0u;
    std::uint64_t standStatus = 0u;
    std::uint64_t candidateSupportConsequences = 0u;
    std::uint64_t tendonTransfers = 0u;
    std::uint64_t tendonGeneralizedCorrections = 0u;
    std::uint64_t totalBytes = 0u;
};

struct OwnerSnapshotCapture {
    __strong id<MTLBuffer> buffer = nil;
    OwnerSnapshotCaptureLayout layout{};
    std::uint64_t ownerProgramFingerprint = 0u;
    std::uint64_t linearizationEpoch = 0u;
    bool preDynamicsEncoded = false;
    bool humanMatterPostDynamicsEncoded = false;
    bool postDynamicsEncoded = false;
};

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
    std::uint64_t acceptedTimestampNanoseconds = 0u;
    std::uint64_t receptorTimestampNanoseconds = 0u;
    std::uint64_t previousTransactionFingerprint = 0u;
    std::uint64_t previousPhysicsGeneration = 0u;
    std::uint64_t previousAcceptedTokenFingerprint = 0u;
    std::uint64_t previousHumanIOProgramFingerprint = 0u;
    bool exactFamily = false;
    metalrobo::MetalNumanXHumanIOExactPreparedView exactHumanIO{};
    metalrobo::MetalNumanXHumanMatterExactPhysicalReceipt exactReceipt{};
    mrnx_candidate_timing_v2 exactTiming{};
    mrnx_exact_inbound_authority_v2 exactInboundAuthority{};
    mrnx_exact_sensor_packet_v2 exactSensorPacket{};
    mrnx_candidate_channel_v2
        exactChannels[MRNX_MAX_SENSOR_CHANNELS_V2]{};
    std::uint32_t exactChannelCount = 0u;
    ImportedRange motorHeader{};
    ImportedRange excitation{};
    ImportedRange autonomic{};
    ImportedRange activeSensing{};
    ImportedRange motorReadyGate{};
    __strong id<MTLSharedEvent> motorReadyEvent = nil;
    // Optional qualification copy, populated on the original physical command
    // buffer and exposed only after joint publication. Never a state owner.
    __strong id<MTLBuffer> rootTranslationTrace = nil;
    // Optional bounded evidence copy. It owns no simulation state and is read
    // only after the enclosing prepared root reaches a terminal disposition.
    std::optional<OwnerSnapshotCapture> ownerSnapshotCapture;
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
    __strong id<MTLBuffer> supportConsequences = nil;
    std::uint64_t supportConsequencesGPUAddress = 0u;
    std::optional<metalrobo::MetalNeuronCultureTicket> cultureTicket;
    mrnx_culture_prepared_view_v1 culturePrepared{};
    metalrobo::MetalNeuronCultureAcceptedView cultureAcceptedView;
    mrnx_culture_accepted_view_v1 cultureAccepted{};
    MRNumanXHumanSupplementalDispatchGPU supplementalDispatch{};
    bool humanSettled = false;
    bool humanReady = false;
    bool physicalSettled = false;
    bool physicalReady = false;
    bool cultureSettled = false;
    bool cultureReady = false;
    bool settlementStarted = false;
};

struct RuntimeState final : std::enable_shared_from_this<RuntimeState> {
    mutable std::mutex mutex;
    DomainPtr domain;
    __strong id<MTLDevice> device = nil;
    FullBodyAssets assets;
    VisionProfile visionProfile;
    __strong id<MTLLibrary> supplementalLibrary = nil;
    __strong id<MTLComputePipelineState> supplementalPipeline = nil;
    __strong id<MTLComputePipelineState> supportAggregationPipeline = nil;
    __strong id<MTLBuffer> touchSupportMapping = nil;
    __strong id<MTLBuffer> visualBodyBounds = nil;
    std::uint64_t supplementalProgramFingerprint = 0u;
    // The historical member name is retained for the fixed internal token
    // layout. In exact-clock mode it stores nanosecond ticks; the quantum and
    // seconds fields make the unit explicit at every conversion seam.
    std::uint64_t timestepMicroseconds = 0u;
    std::uint64_t timestepNanoseconds = 0u;
    std::uint64_t clockQuantumNanoseconds = 1000u;
    double timestepSeconds = 0.0;
    bool exactClock = false;
    std::uint32_t transactionSlotCount = 0u;
    std::uint64_t nextSlotGeneration = 1u;
    std::uint64_t nextSensorGeneration = 1u;
    std::uint64_t nextLinearizationEpoch = 1u;
    bool beginInProgress = false;
    std::atomic<bool> terminalQuarantine{false};
    bool publishedOnce = false;
    std::uint64_t publishedTransactionFingerprint = 0u;
    std::uint64_t publishedBrainGeneration = 0u;
    std::uint64_t publishedPhysicsGeneration = 0u;
    std::uint64_t publishedTimestampMicroseconds = 0u;
    std::uint64_t publishedTimestampNanoseconds = 0u;
    std::uint64_t publishedControlStep = 0u;
    // Opt-in, bounded production-owner evidence. At most the first published
    // root and one explicitly selected control root are persisted.
    std::filesystem::path ownerSnapshotDirectory;
    std::optional<std::uint64_t> ownerSnapshotSelectedControlStep;
    bool ownerSnapshotFirstPublishedCaptured = false;
    bool ownerSnapshotSelectedCaptured = false;
    std::uint64_t ownerSnapshotBaseStateFingerprint = 0u;
    std::uint64_t ownerSnapshotTreatmentHistoryFingerprint = 0u;
    std::uint64_t ownerSnapshotHumanSourceWithoutInitialHistory = 0u;
    std::uint64_t ownerSnapshotContactSampleCount = 0u;
    std::uint64_t ownerSnapshotMatterGeneralizedStateCount = 0u;
    std::uint64_t ownerSnapshotMatterReactionCount = 0u;
    metalrobo::NumiHumanProductionOwnerTreatmentV1 ownerSnapshotTreatment =
        metalrobo::NumiHumanProductionOwnerTreatmentV1::cold;
    std::vector<nm_float4> ownerSnapshotInitialSupportHistories;
    // Attempts advance even when the root is authoritatively rejected; public
    // generation/timestamp authority advances only on accepted publication.
    std::uint64_t lastAttemptedControlStep = 0u;
    std::shared_ptr<ActiveRoot> active;
    // Submit-time observer identity. The synchronous owner encoder borrows
    // this before prepared ownership becomes runtime->active.
    ActiveRoot* encodingActive = nullptr;
    mrnx_runtime_info_v1 info{};
    mrnx_runtime_world_info_v1 worldInfo{};
    mrnx_aggregate_snapshot_v1 aggregate{};
    mrnx_candidate_timing_v1 aggregateTiming{};
    mrnx_aggregate_snapshot_v5 exactAggregate{};
    mrnx_candidate_channel_v1 aggregateChannels[MRNX_MAX_SENSOR_CHANNELS_V2]{};
    std::uint32_t aggregateChannelCount = 0u;
    mrnx_culture_accepted_view_v1 aggregateCulture{};
    metalrobo::CompiledNeuronCulture culturePack;
    std::unique_ptr<metalrobo::MetalNeuronCultureRuntime> culture;
    metalrobo::MetalNeuronCultureAcceptedView publishedCultureView;
    std::uint32_t cultureWindowTicks = 0u;
    float cultureCurrentPerNewton = 0.0f;
    std::string cultureProtocolPath;
    __strong id<MTLBuffer> publishedChannelValues[MRNX_MAX_SENSOR_CHANNELS_V2]{};
    __strong id<MTLBuffer> publishedChannelValidity[MRNX_MAX_SENSOR_CHANNELS_V2]{};
    __strong id<MTLBuffer> publishedProprioception = nil;
    __strong id<MTLBuffer> publishedProprioceptionValidity = nil;
    __strong id<MTLBuffer> publishedInteroception = nil;
    __strong id<MTLBuffer> publishedInteroceptionValidity = nil;
    std::unique_ptr<metalrobo::MetalHumanBehaviorTelemetry> behavior;
    metalrobo::HumanBehaviorCompileBinding behaviorBinding;
    std::string behaviorMetallibPath;
    std::string behaviorError;
    std::string behaviorMetricSHA256;
    std::uint64_t behaviorInitialTimestampNanoseconds = 0u;
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
void recordRuntimeBehaviorTerminal(
    RuntimeState& runtime,
    const ActiveRoot& active,
    const mrnx_root_v1& root,
    bool accepted,
    const MRNumanXHumanMatterJointPublicationFenceGPU* fence
) noexcept;
[[nodiscard]] bool writeOwnerSnapshotEvidence(
    RuntimeState& runtime,
    const ActiveRoot& active,
    const mrnx_root_v1& terminalRoot,
    PreparedTerminalDisposition disposition,
    std::uint64_t publicationEpoch,
    const MRNumanXHumanMatterJointPublicationFenceGPU* committedFence,
    std::string& error
) noexcept;
[[nodiscard]] bool publishExactRuntimeTerminal(
    RuntimeState& runtime,
    const std::shared_ptr<ActiveRoot>& active,
    const mrnx_root_v1& root,
    const mrnx_candidate_view_v1* candidate,
    const mrnx_candidate_channel_v1* legacyChannels,
    const std::uint32_t legacyChannelCount,
    const MRNumanXHumanMatterJointPublicationFenceGPU* committedFence
) noexcept {
    if (active == nullptr || !active->exactFamily || candidate == nullptr ||
        legacyChannels != nullptr || legacyChannelCount != 0u ||
        committedFence == nullptr || active->exactChannelCount != 7u) {
        return false;
    }
    const auto& authority = active->exactInboundAuthority;
    const auto& proof = active->exactReceipt.acceptedStateProof;
    const auto& token = active->exactReceipt.acceptedPhysicsStateToken;
    const auto& timing = active->exactTiming;
    const auto& packet = active->exactSensorPacket;
    const auto* channels = active->exactChannels;
    const bool identityValid =
        root.abi_version == MRNX_BRIDGE_ABI_V1 &&
        root.struct_size == sizeof(root) &&
        root.owner_wire_abi_version == MRNX_OWNER_WIRE_ABI_V4 &&
        root.environment_count == 1u && root.environment == 0u &&
        root.transaction_slot == active->transactionSlot &&
        root.control_step == active->controlStep &&
        root.transaction_fingerprint == active->transactionFingerprint &&
        root.slot_generation == active->slotGeneration &&
        root.device_registry_id == runtime.device.registryID &&
        candidate->abi_version == MRNX_BRIDGE_ABI_V1 &&
        candidate->struct_size == sizeof(*candidate) &&
        candidate->key.abi_version == MRNX_BRIDGE_ABI_V1 &&
        candidate->key.struct_size == sizeof(candidate->key) &&
        candidate->channel_count == active->exactChannelCount &&
        candidate->reserved0 == 0u &&
        candidate->device_registry_id == runtime.device.registryID &&
        candidate->accepted_brain_generation == active->brainGeneration &&
        candidate->key.transaction_fingerprint ==
            active->candidateKey.transactionFingerprint &&
        candidate->key.program_fingerprint ==
            active->candidateKey.programFingerprint &&
        candidate->key.sensor_fingerprint ==
            active->candidateKey.sensorFingerprint &&
        candidate->key.transaction_instance_fingerprint ==
            active->candidateKey.transactionInstanceFingerprint &&
        candidate->key.sensor_generation ==
            active->candidateKey.sensorGeneration &&
        candidate->key.command_buffer_identity ==
            active->candidateKey.commandBufferIdentity &&
        candidate->key.fingerprint != 0u &&
        candidate->candidate_publication_fingerprint ==
            packet.candidate_publication_fingerprint &&
        candidate->candidate_identity_fingerprint != 0u &&
        timing.capture_timestamp_nanoseconds ==
            active->receptorTimestampNanoseconds &&
        timing.delivery_timestamp_nanoseconds ==
            active->acceptedTimestampNanoseconds &&
        timing.latency_nanoseconds == runtime.timestepNanoseconds &&
        timing.sample_interval_nanoseconds == runtime.timestepNanoseconds &&
        packet.sensor_generation == active->candidateKey.sensorGeneration &&
        packet.accepted_brain_generation == active->brainGeneration &&
        packet.device_registry_id == runtime.device.registryID &&
        committedFence->abiVersion ==
            MR_NUMANX_HUMAN_MATTER_PUBLICATION_FENCE_ABI_VERSION_V2 &&
        committedFence->structBytes == sizeof(*committedFence) &&
        committedFence->status ==
            MR_NUMANX_HUMAN_MATTER_PUBLICATION_COMMITTED &&
        committedFence->environment == 0u &&
        committedFence->controlStep == active->controlStep &&
        committedFence->transactionFingerprint ==
            active->transactionFingerprint &&
        committedFence->linearizationEpoch == root.linearization_epoch &&
        committedFence->slotGeneration == active->slotGeneration &&
        committedFence->physicsTokenFingerprint == token.tokenFingerprint &&
        committedFence->brainGeneration == active->brainGeneration &&
        committedFence->jointCommitFingerprint != 0u &&
        committedFence->fenceFingerprint != 0u;
    if (!identityValid) return false;

    mrnx_publication_v2 publication{};
    publication.abi_version = MRNX_PUBLICATION_ABI_V2;
    publication.struct_size = sizeof(publication);
    publication.clock_domain = MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    publication.clock_quantum_nanoseconds =
        MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    publication.transaction_fingerprint = active->transactionFingerprint;
    publication.accepted_physics_token_fingerprint = token.tokenFingerprint;
    publication.candidate_publication_fingerprint =
        packet.candidate_publication_fingerprint;
    publication.joint_commit_fingerprint =
        committedFence->jointCommitFingerprint;
    publication.brain_generation = committedFence->brainGeneration;
    publication.committed_timestamp_nanoseconds =
        token.acceptedTimestampNanoseconds;
    publication.publication_fingerprint =
        metalrobo::metalNumanXExactPublicationV2Fingerprint(publication);
    if (!metalrobo::metalNumanXExactOutboundFamilyV2Valid(
            authority, proof, token, timing, channels,
            active->exactChannelCount, packet, publication)) {
        return false;
    }

    __unsafe_unretained id<MTLBuffer> channelValues[
        MRNX_MAX_SENSOR_CHANNELS_V2]{};
    __unsafe_unretained id<MTLBuffer> channelValidity[
        MRNX_MAX_SENSOR_CHANNELS_V2]{};
    std::uint32_t proprioceptionIndex = MRNX_MAX_SENSOR_CHANNELS_V2;
    std::uint32_t interoceptionIndex = MRNX_MAX_SENSOR_CHANNELS_V2;
    for (std::uint32_t index = 0u;
         index < active->exactChannelCount; ++index) {
        const auto& channel = channels[index];
        std::uint32_t expectedReceptors = 0u;
        std::uint32_t expectedFeatures = 0u;
        switch (channel.modality) {
            case MRNX_CANDIDATE_MODALITY_VISION_V1:
                expectedReceptors = MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT;
                expectedFeatures = MR_NUMANX_HUMAN_VISION_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_AUDITION_V1:
                expectedReceptors = MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT;
                expectedFeatures = MR_NUMANX_HUMAN_AUDITION_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_TOUCH_V1:
                expectedReceptors = MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT;
                expectedFeatures = MR_NUMANX_HUMAN_TOUCH_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_PROPRIOCEPTION_V1:
                expectedReceptors = MRNX_FULL_BODY_MUSCLE_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_PROPRIOCEPTION_FEATURE_COUNT;
                proprioceptionIndex = index;
                break;
            case MRNX_CANDIDATE_MODALITY_VESTIBULAR_V1:
                expectedReceptors =
                    MR_NUMANX_HUMAN_VESTIBULAR_RECEPTOR_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_VESTIBULAR_FEATURE_COUNT;
                break;
            case MRNX_CANDIDATE_MODALITY_INTEROCEPTION_V1:
                expectedReceptors = MRNX_FULL_BODY_MUSCLE_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_INTEROCEPTION_FEATURE_COUNT;
                interoceptionIndex = index;
                break;
            case MRNX_CANDIDATE_MODALITY_KINESTHESIA_V1:
                expectedReceptors =
                    MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT;
                expectedFeatures =
                    MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT;
                break;
            default:
                return false;
        }
        if (channel.receptor_count != expectedReceptors ||
            channel.feature_dimension != expectedFeatures ||
            !bufferObject(channel.values.metal_buffer, channelValues[index]) ||
            !bufferObject(
                channel.validity.metal_buffer, channelValidity[index]) ||
            channelValues[index].device != runtime.device ||
            channelValidity[index].device != runtime.device ||
            channel.values.byte_offset > channelValues[index].length ||
            channel.validity.byte_offset > channelValidity[index].length ||
            channel.values.byte_count >
                channelValues[index].length - channel.values.byte_offset ||
            channel.validity.byte_count >
                channelValidity[index].length -
                    channel.validity.byte_offset ||
            channelValues[index].gpuAddress >
                std::numeric_limits<std::uint64_t>::max() -
                    channel.values.byte_offset ||
            channelValidity[index].gpuAddress >
                std::numeric_limits<std::uint64_t>::max() -
                    channel.validity.byte_offset ||
            channel.values.gpu_address !=
                channelValues[index].gpuAddress + channel.values.byte_offset ||
            channel.validity.gpu_address !=
                channelValidity[index].gpuAddress +
                    channel.validity.byte_offset) {
            return false;
        }
    }
    if (proprioceptionIndex >= active->exactChannelCount ||
        interoceptionIndex >= active->exactChannelCount) return false;

    std::unique_lock runtimeLock(runtime.mutex);
    if (runtime.active != active || runtime.terminalQuarantine ||
        runtime.exactAggregate.publication_epoch ==
            std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }
    if (runtime.culture != nullptr) {
        if (!active->cultureAcceptedView.valid() ||
            active->cultureAccepted.culture_fingerprint !=
                runtime.culturePack.fingerprint() ||
            active->cultureAccepted.generation == 0u ||
            runtime.culture->publishPrepared() !=
                metalrobo::MetalNeuronCultureStatus::success) {
            return false;
        }
        runtime.publishedCultureView = active->cultureAcceptedView;
        runtime.aggregateCulture = active->cultureAccepted;
    }

    mrnx_aggregate_snapshot_v5 snapshot{};
    snapshot.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V5;
    snapshot.struct_size = sizeof(snapshot);
    snapshot.publication_epoch =
        runtime.exactAggregate.publication_epoch + 1u;
    if (snapshot.publication_epoch !=
        metalrobo::numanx_bridge_v1::domainPublicationEpoch(
            runtime.domain)) return false;
    snapshot.brain_generation = active->brainGeneration;
    snapshot.physics_generation = active->physicsGeneration;
    snapshot.sensor_generation = active->candidateKey.sensorGeneration;
    snapshot.root = root;
    snapshot.sensor = *candidate;
    snapshot.timing = timing;
    snapshot.inbound_authority = authority;
    snapshot.sensor_packet = packet;
    snapshot.publication = publication;
    for (std::uint32_t index = 0u;
         index < active->exactChannelCount; ++index) {
        snapshot.channels[index] = channels[index];
        runtime.publishedChannelValues[index] = channelValues[index];
        runtime.publishedChannelValidity[index] = channelValidity[index];
    }
    if (runtime.culture != nullptr) snapshot.culture = runtime.aggregateCulture;
    runtime.publishedProprioception =
        channelValues[proprioceptionIndex];
    runtime.publishedProprioceptionValidity =
        channelValidity[proprioceptionIndex];
    runtime.publishedInteroception =
        channelValues[interoceptionIndex];
    runtime.publishedInteroceptionValidity =
        channelValidity[interoceptionIndex];
    runtime.aggregateChannelCount = active->exactChannelCount;
    runtime.exactAggregate = snapshot;
    runtime.publishedOnce = true;
    runtime.publishedTransactionFingerprint = active->transactionFingerprint;
    runtime.publishedBrainGeneration = active->brainGeneration;
    runtime.publishedPhysicsGeneration = active->physicsGeneration;
    runtime.publishedTimestampNanoseconds =
        active->acceptedTimestampNanoseconds;
    runtime.publishedTimestampMicroseconds = 0u;
    runtime.publishedControlStep = active->controlStep;
    recordRuntimeBehaviorTerminal(
        runtime, *active, root, true, committedFence);

    const bool selectedOwnerSnapshot =
        runtime.ownerSnapshotSelectedControlStep.has_value() &&
        !runtime.ownerSnapshotSelectedCaptured &&
        active->controlStep == *runtime.ownerSnapshotSelectedControlStep;
    const bool firstOwnerSnapshot =
        !runtime.ownerSnapshotFirstPublishedCaptured;
    if (active->ownerSnapshotCapture &&
        (firstOwnerSnapshot || selectedOwnerSnapshot)) {
        std::string evidenceError;
        if (writeOwnerSnapshotEvidence(
                runtime, *active, root,
                PreparedTerminalDisposition::published,
                snapshot.publication_epoch, committedFence, evidenceError)) {
            if (firstOwnerSnapshot)
                runtime.ownerSnapshotFirstPublishedCaptured = true;
            if (selectedOwnerSnapshot)
                runtime.ownerSnapshotSelectedCaptured = true;
        } else {
            std::fprintf(stderr,
                "mrnx_production_owner_snapshot_failure=%s\n",
                evidenceError.c_str());
            runtime.terminalQuarantine = true;
        }
    }
    runtime.active.reset();
    return !runtime.terminalQuarantine;
}

void runtimeTerminalCompletion(
    void* raw,
    PreparedTerminalDisposition disposition,
    const mrnx_root_v1& root,
    const mrnx_candidate_view_v1* candidate,
    const mrnx_candidate_channel_v1* channels,
    std::uint32_t channelCount,
    const MRNumanXHumanMatterJointPublicationFenceGPU* committedFence
) noexcept;
[[nodiscard]] bool encodeRuntimeBehaviorCandidate(void* raw,
    const metalrobo::MetalNumanXHumanMatterPass& pass) noexcept;
[[nodiscard]] mrnx_candidate_v1* finalizeExactCandidate(
    const std::shared_ptr<ActiveRoot>& active
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
void cultureCompletion(
    void* raw,
    metalrobo::MetalNeuronCultureStatus status
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
// Caller holds RuntimeState::mutex across this pure scalar admission so
// behavior attachment and legacy publication state cannot change mid-check.
[[nodiscard]] bool validateRootRequestV3CPUAdmissionLocked(
    const std::shared_ptr<RuntimeState>& runtime,
    const mrnx_physical_root_request_v3& request,
    MRNumanXBrainJointTransactionTokenV2& root,
    MRNumanXBrainJointSubstepTokenV2& substep,
    MRNumanXBrainMotorCandidateV2& candidate,
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
    id<MTLFunction> aggregate = [runtime.supplementalLibrary newFunctionWithName:@"numanx_human_aggregate_support"];
    requireBuild(aggregate != nil, MRNX_RUNTIME_METAL_FAILURE_V1, "support aggregation kernel unavailable");
    runtime.supportAggregationPipeline = [runtime.device newComputePipelineStateWithFunction:aggregate error:&pipelineError];
    runtime.touchSupportMapping = [runtime.device newBufferWithBytes:runtime.assets.touchSupportMapping.data()
        length:runtime.assets.touchSupportMapping.size()*sizeof(mr_uint4) options:MTLResourceStorageModeShared];
    requireBuild(runtime.supportAggregationPipeline != nil && runtime.touchSupportMapping != nil,
        MRNX_RUNTIME_METAL_FAILURE_V1, "support aggregation resources unavailable");
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
    bool captureOwnerSnapshot = false;
    {
        const std::lock_guard lock(runtime->mutex);
        captureOwnerSnapshot = !runtime->ownerSnapshotDirectory.empty() &&
            (!runtime->ownerSnapshotFirstPublishedCaptured ||
             (runtime->ownerSnapshotSelectedControlStep.has_value() &&
              !runtime->ownerSnapshotSelectedCaptured &&
              active->controlStep ==
                  *runtime->ownerSnapshotSelectedControlStep));
    }
    if (captureOwnerSnapshot) {
        OwnerSnapshotCapture capture;
        std::uint64_t cursor = 0u;
        const auto add = [&cursor](
            const std::uint64_t count,
            const std::uint64_t elementBytes,
            std::uint64_t& offset) noexcept {
            if (elementBytes == 0u || count >
                    std::numeric_limits<std::uint64_t>::max() /
                        elementBytes) return false;
            const std::uint64_t bytes = count * elementBytes;
            if (cursor > std::numeric_limits<std::uint64_t>::max() - 15u)
                return false;
            cursor = (cursor + 15u) & ~std::uint64_t{15u};
            offset = cursor;
            if (bytes > std::numeric_limits<std::uint64_t>::max() - cursor)
                return false;
            cursor += bytes;
            return true;
        };
        auto& layout = capture.layout;
        const std::uint64_t nq = runtime->assets.rigid.nq;
        const std::uint64_t nv = runtime->assets.rigid.nv;
        const std::uint64_t muscles = runtime->assets.muscle.muscleCount;
        const std::uint64_t tendonRows = 0u;
        if ((nv != 0u && nv >
                std::numeric_limits<std::uint64_t>::max() / nv) ||
            (nv != 0u && muscles >
                std::numeric_limits<std::uint64_t>::max() / nv) ||
            (nv != 0u && tendonRows >
                std::numeric_limits<std::uint64_t>::max() / nv)) {
            return false;
        }
        const std::uint64_t factorElements = nv * nv;
        const std::uint64_t muscleForceElements = muscles * nv;
        const std::uint64_t tendonCorrectionElements = tendonRows * nv;
        if (!add(nq, sizeof(float), layout.checkpointQ) ||
            !add(nv, sizeof(float), layout.checkpointV) ||
            !add(1u, sizeof(MRCompensatedRootTranslationGPU),
                layout.checkpointRoot) ||
            !add(muscles, sizeof(MRMujocoMuscleStateGPU),
                layout.checkpointMuscles) ||
            !add(factorElements, sizeof(float),
                layout.effectiveTangentFactorStorage) ||
            !add(nv, sizeof(float), layout.sourceGeneralizedForce) ||
            !add(nv, sizeof(float), layout.sourcePredictedVelocity) ||
            !add(nv, sizeof(float), layout.matterGeneralizedReaction) ||
            !add(1u, sizeof(MRNumanXHumanMatterOwnerStatusGPU),
                layout.ownerStatus) ||
            !add(nq, sizeof(float), layout.candidateQ) ||
            !add(nv, sizeof(float), layout.candidateV) ||
            !add(1u, sizeof(MRCompensatedRootTranslationGPU),
                layout.candidateRoot) ||
            !add(muscles, sizeof(MRMujocoMuscleStateGPU),
                layout.candidateMuscles) ||
            !add(muscles, sizeof(MRMujocoMuscleResultGPU),
                layout.muscleResults) ||
            !add(muscleForceElements, sizeof(float),
                layout.muscleGeneralizedForces) ||
            !add(nv, sizeof(float),
                layout.reducedMuscleGeneralizedForce) ||
            !add(1u, sizeof(MRNumiHumanStandStatusGPU),
                layout.standStatus) ||
            !add(runtime->assets.matterSupportContacts.size(),
                sizeof(NMHumanSupportConsequenceGPU),
                layout.candidateSupportConsequences) ||
            !add(tendonRows, sizeof(MRNumiHumanTendonTransferResultGPU),
                layout.tendonTransfers) ||
            !add(tendonCorrectionElements, sizeof(float),
                layout.tendonGeneralizedCorrections) ||
            cursor == 0u || cursor > std::numeric_limits<NSUInteger>::max()) {
            return false;
        }
        layout.totalBytes = cursor;
        capture.buffer = [runtime->device
            newBufferWithLength:static_cast<NSUInteger>(cursor)
            options:MTLResourceStorageModeShared];
        if (capture.buffer == nil || capture.buffer.gpuAddress == 0u ||
            capture.buffer.contents == nullptr ||
            capture.buffer.length != cursor) return false;
        capture.buffer.label =
            @"NumanX persistent production-owner snapshot v1";
        std::memset(capture.buffer.contents, 0,
            static_cast<std::size_t>(cursor));
        active->ownerSnapshotCapture.emplace(std::move(capture));
    }
    if (std::getenv("MRNX_PHYSICAL_BUFFER_TRACE") != nullptr) {
        active->rootTranslationTrace = [runtime->device
            newBufferWithLength:sizeof(MRCompensatedRootTranslationGPU)
            options:MTLResourceStorageModeShared];
        if (active->rootTranslationTrace == nil || active->rootTranslationTrace.gpuAddress == 0u)
            return false;
        active->rootTranslationTrace.label = @"NumanX root translation qualification copy";
    }
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
    active->supportConsequences = makePrivateBuffer(runtime->device,
        MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT*sizeof(MRNumanXHumanSupportConsequenceGPU), @"NumanX geometry touch consequences");
    if (active->supportConsequences == nil) return false;
    active->supportConsequencesGPUAddress = active->supportConsequences.gpuAddress;
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
    const mrnx_runtime_config_v1& config,
    const mrnx_runtime_config_v3* authored = nullptr,
    const mrnx_runtime_config_v4* equalityConfig = nullptr,
    const mrnx_runtime_config_v5* tissueConfig = nullptr,
    const mrnx_runtime_config_v6* limitConfig = nullptr,
    const mrnx_runtime_config_v7* initialConfig = nullptr,
    const std::uint64_t exactTimestepNanoseconds = 0u
) {
    const bool exactClock = exactTimestepNanoseconds != 0u;
    const std::uint64_t timestepNanoseconds = exactClock
        ? exactTimestepNanoseconds : config.timestep_microseconds * 1000ull;
    const std::uint64_t clockTicks = exactClock
        ? exactTimestepNanoseconds : config.timestep_microseconds;
    const double timestepSeconds = static_cast<double>(timestepNanoseconds) * 1.0e-9;
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
            (authored != nullptr || config.matter_material_path != nullptr) &&
            config.rigid_payload_path[0] != '\0' &&
            config.muscle_payload_path[0] != '\0' &&
            config.support_contact_payload_path[0] != '\0' &&
            config.visual_pack_path[0] != '\0' &&
            config.vision_profile_path[0] != '\0' &&
            config.metalrobo_metallib_path[0] != '\0' &&
            config.matter_metallib_path[0] != '\0' &&
            (authored != nullptr || config.matter_material_path[0] != '\0') &&
            ((exactClock && config.timestep_microseconds == 0u &&
              exactTimestepNanoseconds <= 1'000'000'000u) ||
             (!exactClock && config.timestep_microseconds != 0u &&
              config.timestep_microseconds <= 1'000'000u)) &&
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
    runtime->assets = loadFullBodyAssets(
        config.rigid_payload_path, config.muscle_payload_path,
        config.support_contact_payload_path);
    const auto tissueOffsets=tissueConfig!=nullptr
        ? prepareCostalMassOwnership(runtime->assets,*tissueConfig)
        : std::vector<std::array<double,3>>{};
    runtime->behaviorBinding.sourceArchiveSHA256 = runtime->assets.rigid.sourceSHA256;
    runtime->behaviorBinding.rigidSHA256 = runtime->assets.rigidSHA256;
    runtime->behaviorBinding.bodyCount = runtime->assets.rigid.engineBodyCount;
    runtime->behaviorBinding.nq = runtime->assets.rigid.nq;
    runtime->behaviorBinding.nv = runtime->assets.rigid.nv;
    runtime->behaviorBinding.timestepNanoseconds = timestepNanoseconds;
    for (std::uint32_t i = 0u; i < runtime->assets.sourceMap.size(); ++i)
        runtime->behaviorBinding.sourceToCore.push_back({i, runtime->assets.sourceMap[i]});
    runtime->behaviorBinding.cookedCOMOffset = tissueConfig != nullptr ? tissueOffsets
        : std::vector<std::array<double,3>>(runtime->assets.model.bodies.size(), {0.0, 0.0, 0.0});
    runtime->behaviorMetallibPath = config.metalrobo_metallib_path;
    // Decode after mass ownership establishes final body frames. The source
    // default pose remains authoritative for rest coordinates and ownership;
    // these vectors belong only to construction and the first resident submit.
    metalrobo::NumiHumanInitialState initialState;
    std::vector<nm_float4> initialSupportHistories;
    runtime->assets.initialQ = runtime->assets.model.defaultQ;
    runtime->assets.initialV = runtime->assets.model.defaultV;
    if (initialConfig != nullptr) {
        const auto image = loadImmutablePayload(initialConfig->initial_state_payload_path, "NHINIT initial state");
        requireBuild(hashBytes(image.bytes.data(), image.bytes.size()) == initialConfig->expected_initial_state_fingerprint,
            MRNX_RUNTIME_ASSET_FAILURE_V1, "initial-state fingerprint mismatch");
        std::string error;
        requireBuild(metalrobo::decodeNumiHumanInitialState(
            std::as_bytes(std::span(image.bytes.data(), image.bytes.size())),
            runtime->assets.rigid.nq, runtime->assets.rigid.nv, runtime->assets.muscle.muscleCount,
            runtime->assets.rigid.sourceSHA256, runtime->assets.supportIdentity,
            initialState, error),
            MRNX_RUNTIME_ASSET_FAILURE_V1, "initial-state admission failed: " + error);
        const auto initialNanoseconds =
            metalrobo::numiHumanInitialStateTimestepNanoseconds(initialState);
        // NHINIT2 carries the exact clock. Legacy construction still requires
        // the canonical microsecond representation; v8 compares the exact
        // nanosecond word without rounding.
        requireBuild(initialNanoseconds != 0u,
            MRNX_RUNTIME_ASSET_FAILURE_V1, "initial-state clock is missing");
        requireBuild(initialState.worldFingerprint == authored->expected_matter_world_fingerprint &&
            initialNanoseconds == timestepNanoseconds,
            MRNX_RUNTIME_ASSET_FAILURE_V1, "initial-state world/clock mismatch");
        double norm = 0.0;
        for (unsigned i = 3u; i < 7u; ++i) norm += double(initialState.q[i]) * initialState.q[i];
        requireBuild(std::abs(norm - 1.0) <= 16.0 * std::numeric_limits<float>::epsilon(),
            MRNX_RUNTIME_ASSET_FAILURE_V1, "initial-state root quaternion is not unit length");
        runtime->assets.initialQ = initialState.q;
        runtime->assets.initialV = initialState.v;
        runtime->assets.states = initialState.muscles;
        if (initialState.preparedSupportHistory) {
            const auto& prepared = *initialState.preparedSupportHistory;
            requireBuild(prepared.rows.size() ==
                    runtime->assets.matterSupportContacts.size(),
                MRNX_RUNTIME_ASSET_FAILURE_V1,
                "prepared support history row count changed after NHCNT admission");
            const auto normal = runtime->assets.groundNormal;
            initialSupportHistories.reserve(prepared.rows.size());
            for (std::size_t rowIndex = 0u;
                 rowIndex < prepared.rows.size(); ++rowIndex) {
                const auto& row = prepared.rows[rowIndex];
                const float friction = runtime->assets
                    .matterSupportContacts[rowIndex]
                    .frictionSlopAndStabilization.x;
                const nm_float4 history{
                    row.tangentImpulseWorldX,
                    row.tangentImpulseWorldY,
                    row.tangentImpulseWorldZ,
                    row.normalImpulse};
                requireBuild(numi::matter::detail::
                        humanSupportHistoryAdmissible(
                            history, friction,
                            {normal.x, normal.y, normal.z, normal.w}),
                    MRNX_RUNTIME_ASSET_FAILURE_V1,
                    "prepared support history violates its NHCNT tangent/Coulomb cone");
                initialSupportHistories.push_back(history);
            }
        }
    }
    requireBuild(runtime->assets.initialQ.size() >= 7u,
        MRNX_RUNTIME_ASSET_FAILURE_V1, "initial-state floating root is absent");
    const auto& q0 = runtime->assets.initialQ;
    const auto initialTranslation = initialState.rootTranslation.value_or(
        mrCompensatedTranslationFromProjection({q0[0], q0[1], q0[2], 0.0f}));
    requireBuild(mrCompensatedTranslationValid(initialTranslation),
        MRNX_RUNTIME_ASSET_FAILURE_V1, "invalid initial root translation expansion");
    const auto projectedTranslation = mrCompensatedTranslationProjection(initialTranslation);
    requireBuild(mrCompensatedBits(projectedTranslation.x) == mrCompensatedBits(q0[0]) &&
        mrCompensatedBits(projectedTranslation.y) == mrCompensatedBits(q0[1]) &&
        mrCompensatedBits(projectedTranslation.z) == mrCompensatedBits(q0[2]),
        MRNX_RUNTIME_ASSET_FAILURE_V1, "initial root projection disagrees with q");
    runtime->assets.initialRootTranslations = {initialTranslation};
    // Bind the reference Matter patch to the articulated root/pelvis COM.
    // The prior use of the final imported body was topology-order dependent
    // and coupled the FEM to an arbitrary high-motion distal link.
    constexpr std::uint32_t attachmentBody = 0u;
    std::vector<double> defaultQ(
        runtime->assets.initialQ.begin(), runtime->assets.initialQ.end());
    // Offline admission checks the authored physical pose, including the low
    // coordinate state. The production owner receives the original FP32 words.
    const auto rootCoordinate = [](float reference, float displacement, float correction) {
        return static_cast<double>(static_cast<long double>(reference) +
            static_cast<long double>(displacement) + static_cast<long double>(correction));
    };
    defaultQ[0] = rootCoordinate(initialTranslation.reference.x, initialTranslation.displacement.x, initialTranslation.correction.x);
    defaultQ[1] = rootCoordinate(initialTranslation.reference.y, initialTranslation.displacement.y, initialTranslation.correction.y);
    defaultQ[2] = rootCoordinate(initialTranslation.reference.z, initialTranslation.displacement.z, initialTranslation.correction.z);
    const std::vector<double> defaultV(
        runtime->assets.initialV.begin(), runtime->assets.initialV.end());
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
    const auto world = authored != nullptr
        ? loadAuthoredWorld(*authored, runtime->assets, defaultBodies,
            timestepSeconds)
        : compileAttachedWorld(
        config.matter_material_path,
        attachmentBody,
        defaultBodies[attachmentBody].centerOfMassPosition,
        defaultBodies[attachmentBody].orientation,
        timestepSeconds);
    if (equalityConfig != nullptr) loadJointEqualities(runtime->assets, *equalityConfig);
    if (limitConfig != nullptr) loadJointLimits(runtime->assets, *limitConfig);
    if(tissueConfig!=nullptr) {
        constexpr char marker[]="NHTMASS1";
        appendFingerprintBytes(runtime->assets.sourceFingerprint,marker,sizeof(marker)-1);
        appendFingerprintU64(runtime->assets.sourceFingerprint,tissueConfig->expected_costal_binding_fingerprint);
        appendFingerprintU64(runtime->assets.sourceFingerprint,world.fingerprint);
    }
    runtime->ownerSnapshotHumanSourceWithoutInitialHistory =
        runtime->assets.sourceFingerprint;
    if (initialConfig != nullptr) {
        requireBuild(initialState.humanSourceFingerprint == runtime->assets.sourceFingerprint,
            MRNX_RUNTIME_ASSET_FAILURE_V1, "initial-state composed Human source mismatch");
        runtime->assets.sourceFingerprint =
            metalrobo::numiHumanRuntimeAppendInitialState(
                runtime->assets.sourceFingerprint,
                initialConfig->expected_initial_state_fingerprint);
    }
    runtime->worldInfo = {
        MRNX_BRIDGE_ABI_V1, sizeof(mrnx_runtime_world_info_v1),
        authored != nullptr ? 1u : 0u,
        world.dispatch.objectCount, world.dispatch.femNodeCount,
        world.dispatch.femHumanAttachmentCount,
        world.fingerprint, world.physicsFingerprint};
    const std::uint64_t matterEnvironmentCount =
        world.dispatch.environmentCount;
    requireBuild(
        (world.dispatch.contactPairCount == 0u ||
         matterEnvironmentCount <=
            std::numeric_limits<std::uint64_t>::max() /
                world.dispatch.contactPairCount) &&
        (world.dispatch.rigidGeneralizedCapacity == 0u ||
         matterEnvironmentCount <=
            std::numeric_limits<std::uint64_t>::max() /
                world.dispatch.rigidGeneralizedCapacity) &&
        (world.dispatch.rigidProxyCount == 0u ||
         matterEnvironmentCount <=
            std::numeric_limits<std::uint64_t>::max() /
                world.dispatch.rigidProxyCount),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "production-owner Matter snapshot shape overflows uint64");
    runtime->ownerSnapshotContactSampleCount = matterEnvironmentCount *
        world.dispatch.contactPairCount;
    runtime->ownerSnapshotMatterGeneralizedStateCount =
        matterEnvironmentCount * world.dispatch.rigidGeneralizedCapacity;
    runtime->ownerSnapshotMatterReactionCount = matterEnvironmentCount *
        world.dispatch.rigidProxyCount;
    runtime->domain = metalrobo::numanx_bridge_v1::makeDomain(
        config.metal_device);
    requireBuild(
        runtime->domain != nullptr, MRNX_RUNTIME_METAL_FAILURE_V1,
        "failed to create NumanX bridge domain");
    runtime->timestepMicroseconds = clockTicks;
    runtime->timestepNanoseconds = timestepNanoseconds;
    runtime->clockQuantumNanoseconds = exactClock ? 1u : 1000u;
    runtime->timestepSeconds = timestepSeconds;
    runtime->exactClock = exactClock;
    runtime->ownerSnapshotInitialSupportHistories = initialSupportHistories;
    runtime->ownerSnapshotTreatment = initialSupportHistories.empty()
        ? metalrobo::NumiHumanProductionOwnerTreatmentV1::cold
        : metalrobo::NumiHumanProductionOwnerTreatmentV1::seeded;
    runtime->ownerSnapshotBaseStateFingerprint =
        metalrobo::numiHumanProductionOwnerBaseStateFingerprintV1(
            runtime->ownerSnapshotHumanSourceWithoutInitialHistory,
            world.fingerprint, timestepNanoseconds,
            std::as_bytes(std::span(runtime->assets.initialQ)),
            std::as_bytes(std::span(runtime->assets.initialV)),
            std::as_bytes(std::span(
                runtime->assets.initialRootTranslations)),
            std::as_bytes(std::span(runtime->assets.states)));
    std::vector<std::byte> supportIdentityBytes;
    supportIdentityBytes.reserve(52u);
    for (const auto value : runtime->assets.supportIdentity.sha256)
        supportIdentityBytes.push_back(std::byte{value});
    const auto appendLittleEndian = [&supportIdentityBytes](
        const std::uint64_t value, const std::uint32_t byteCount) {
        for (std::uint32_t index = 0u; index < byteCount; ++index) {
            supportIdentityBytes.push_back(std::byte{
                static_cast<std::uint8_t>(value >> (8u * index))});
        }
    };
    appendLittleEndian(runtime->assets.supportIdentity.byteCount, 8u);
    appendLittleEndian(runtime->assets.supportIdentity.payloadABI, 4u);
    appendLittleEndian(runtime->assets.supportIdentity.sourceRecordCount, 4u);
    appendLittleEndian(runtime->assets.supportIdentity.expandedRowCount, 4u);
    runtime->ownerSnapshotTreatmentHistoryFingerprint =
        metalrobo::numiHumanProductionOwnerTreatmentHistoryFingerprintV1(
            supportIdentityBytes,
            std::as_bytes(std::span(
                runtime->ownerSnapshotInitialSupportHistories)));
    const char* ownerSnapshotPath =
        std::getenv("MRNX_PRODUCTION_OWNER_SNAPSHOT_PATH");
    const char* ownerSnapshotControlStep =
        std::getenv("MRNX_PRODUCTION_OWNER_SNAPSHOT_CONTROL_STEP");
    requireBuild(
        ownerSnapshotControlStep == nullptr ||
            (ownerSnapshotPath != nullptr && ownerSnapshotPath[0] != '\0'),
        MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
        "production-owner selected control step requires a snapshot path");
    if (ownerSnapshotPath != nullptr) {
        requireBuild(ownerSnapshotPath[0] != '\0',
            MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
            "production-owner snapshot path is empty");
        runtime->ownerSnapshotDirectory = ownerSnapshotPath;
    }
    if (ownerSnapshotControlStep != nullptr) {
        std::uint64_t selected = 0u;
        const char* end = ownerSnapshotControlStep +
            std::strlen(ownerSnapshotControlStep);
        const auto parsed = std::from_chars(
            ownerSnapshotControlStep, end, selected);
        requireBuild(parsed.ec == std::errc{} && parsed.ptr == end,
            MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
            "production-owner selected control step is not an exact uint64");
        runtime->ownerSnapshotSelectedControlStep = selected;
    }
    runtime->transactionSlotCount = config.transaction_slot_count;
    runtime->visionProfile = loadVisionProfile(
        config.visual_pack_path,
        config.vision_profile_path,
        runtime->assets.rigid.engineBodyCount,
        timestepSeconds);
    if(tissueConfig!=nullptr) {
        shiftTissuePoint(runtime->visionProfile.localPosition,runtime->visionProfile.parentBodyIndex,tissueOffsets);
        for(unsigned i=0;i<runtime->visionProfile.bodyBounds.size();++i) {
            shiftTissuePoint(runtime->visionProfile.bodyBounds[i].minimum,i,tissueOffsets);
            shiftTissuePoint(runtime->visionProfile.bodyBounds[i].maximum,i,tissueOffsets);
        }
        appendFingerprintU64(runtime->visionProfile.sourceFingerprint,tissueConfig->expected_costal_binding_fingerprint);
    }
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

    runtime->matter = std::make_unique<numi::matter::Runtime>();
    numi::matter::RuntimeConfiguration matterConfig;
    matterConfig.metallib = config.matter_metallib_path;
    matterConfig.environmentCount = 1u;
    matterConfig.captureEvents = false;
    matterConfig.coupledCandidateCompensatedTranslation = true;
    matterConfig.captureDiagnostics = true;
    matterConfig.adaptiveTransfer = false;
    matterConfig.acceptedStateProofMujocoBytesPerEnvironmentCapacity =
        static_cast<std::uint64_t>(MRNX_FULL_BODY_MUSCLE_COUNT) *
        sizeof(MRMujocoMuscleStateGPU);
    matterConfig.humanJointLimits = runtime->assets.jointLimits;
    matterConfig.humanLimitDispatch = runtime->assets.limitDispatch;
    matterConfig.humanLimitSourceFingerprint = runtime->assets.limitFingerprint;
    matterConfig.humanJointEqualities = runtime->assets.jointEqualities;
    matterConfig.humanEqualityDispatch = runtime->assets.equalityDispatch;
    matterConfig.humanEqualitySourceFingerprint = runtime->assets.equalityFingerprint;
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
    matterConfig.humanSupportInitialHistories = initialSupportHistories;
    const auto matterDiagnostics = runtime->matter->initialize(
        world, matterConfig);
    requireBuild(
        matterDiagnostics.encoded, MRNX_RUNTIME_MATTER_FAILURE_V1,
        "Matter Runtime initialization failed: " +
            matterDiagnostics.message);
    metalrobo::MetalNumanXHumanMatterConfig adapterConfig;
    adapterConfig.matterRuntime = runtime->matter.get();
    adapterConfig.candidateObserverContext = runtime.get();
    adapterConfig.observeCandidate = &encodeRuntimeBehaviorCandidate;
    adapterConfig.coupledHumanMetallibPath =
        config.metalrobo_metallib_path;
    adapterConfig.adapterMetallibPath = config.metalrobo_metallib_path;
    adapterConfig.environmentCapacity = 1u;
    adapterConfig.pointCapacity = std::max<std::uint32_t>(
        world.dispatch.femHumanAttachmentCount,
        static_cast<std::uint32_t>(
            runtime->assets.matterSupportContacts.size()));
    adapterConfig.transactionSlotCount = config.transaction_slot_count;
    adapterConfig.maximumRetainedBytes = config.maximum_retained_bytes;
    adapterConfig.stateProofProgram.context = runtime->matter.get();
    adapterConfig.stateProofProgram.encode = &encodeRuntimeProof;
    adapterConfig.stateProofProgram.fingerprint =
        runtime->matter->acceptedStateProofProgramFingerprint();
    adapterConfig.stateProofProgramV2.context = runtime->matter.get();
    adapterConfig.stateProofProgramV2.encode = &encodeRuntimeProofV2;
    adapterConfig.stateProofProgramV2.fingerprint =
        runtime->matter->acceptedStateProofProgramFingerprintV2();
    runtime->adapter =
        std::make_unique<metalrobo::MetalNumanXHumanMatterContext>(
            std::move(adapterConfig));
    const auto adapterDiagnostics = runtime->adapter->initialize();
    requireBuild(
        adapterDiagnostics.succeeded() &&
            adapterDiagnostics.acceptedStateProofAvailable &&
            adapterDiagnostics.acceptedStateProofV2Available,
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
        .mujocoActivationTimestepSeconds = static_cast<float>(timestepSeconds),
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
        exactClock
        ? runtime->matter->acceptedStateProofProgramFingerprintV2()
        : runtime->matter->acceptedStateProofProgramFingerprint();
    runtime->info.model_source_fingerprint =
        runtime->assets.sourceFingerprint;
    return runtime;
}

[[nodiscard]] std::shared_ptr<RuntimeState> createRuntimeStateV2(
    const mrnx_runtime_config_v2& config,
    const mrnx_runtime_config_v3* authored = nullptr,
    const mrnx_runtime_config_v4* equalityConfig = nullptr,
    const mrnx_runtime_config_v5* tissueConfig = nullptr,
    const mrnx_runtime_config_v6* limitConfig = nullptr,
    const mrnx_runtime_config_v7* initialConfig = nullptr,
    const std::uint64_t exactTimestepNanoseconds = 0u
) {
    requireBuild(
        config.abi_version == MRNX_RUNTIME_CONFIG_ABI_V2 &&
            config.struct_size == sizeof(config),
        MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
        "invalid NumanX runtime configuration v2 header");
    mrnx_runtime_config_v1 base{};
    base.abi_version = MRNX_BRIDGE_ABI_V1;
    base.struct_size = sizeof(base);
    base.metal_device = config.metal_device;
    base.rigid_payload_path = config.rigid_payload_path;
    base.muscle_payload_path = config.muscle_payload_path;
    base.support_contact_payload_path = config.support_contact_payload_path;
    base.visual_pack_path = config.visual_pack_path;
    base.vision_profile_path = config.vision_profile_path;
    base.metalrobo_metallib_path = config.metalrobo_metallib_path;
    base.matter_metallib_path = config.matter_metallib_path;
    base.matter_material_path = config.matter_material_path;
    base.timestep_microseconds = config.timestep_microseconds;
    base.maximum_retained_bytes = config.maximum_retained_bytes;
    base.transaction_slot_count = config.transaction_slot_count;
    base.reserved0 = config.reserved0;
    auto runtime = createRuntimeState(base, authored, equalityConfig, tissueConfig,
        limitConfig, initialConfig, exactTimestepNanoseconds);
    const bool cultureEnabled = config.culture_pack_path != nullptr &&
        config.culture_pack_path[0] != '\0';
    const bool checkpointSupplied = config.culture_checkpoint_path != nullptr &&
        config.culture_checkpoint_path[0] != '\0';
    const bool protocolSupplied = config.culture_protocol_path != nullptr &&
        config.culture_protocol_path[0] != '\0';
    if (!cultureEnabled) {
        requireBuild(
            !checkpointSupplied && !protocolSupplied &&
                config.culture_window_ticks == 0u &&
                config.culture_current_per_newton == 0.0f,
            MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
            "culture options require a culture pack");
        return runtime;
    }
    requireBuild(
        config.culture_window_ticks != 0u &&
            config.culture_window_ticks <=
                metalrobo::kNeuronCultureMaximumWindowTicks &&
            std::isfinite(config.culture_current_per_newton) &&
            config.culture_current_per_newton >= 0.0f,
        MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
        "invalid culture runtime parameters");
    const auto packResult = metalrobo::readCompiledNeuronCulture(
        config.culture_pack_path, runtime->culturePack);
    requireBuild(
        packResult.succeeded() && runtime->culturePack.valid(),
        MRNX_RUNTIME_ASSET_FAILURE_V1,
        "culture pack validation failed: " + packResult.message);
    runtime->culture =
        std::make_unique<metalrobo::MetalNeuronCultureRuntime>(
            metalrobo::MetalNeuronCultureRuntime::create(
                runtime->culturePack, config.metal_device));
    requireBuild(
        runtime->culture != nullptr && runtime->culture->valid() &&
            runtime->culture->residentBytes() < 128ull * 1024ull * 1024ull,
        MRNX_RUNTIME_METAL_FAILURE_V1,
        "culture Metal runtime is unavailable or over budget");
    if (checkpointSupplied) {
        metalrobo::NeuronCultureState checkpoint;
        const auto checkpointResult = metalrobo::readNeuronCultureCheckpoint(
            runtime->culturePack, config.culture_checkpoint_path, checkpoint);
        requireBuild(
            checkpointResult.succeeded() &&
                runtime->culture->restoreAccepted(checkpoint) ==
                    metalrobo::MetalNeuronCultureStatus::success,
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "culture checkpoint validation failed: " + checkpointResult.message);
    }
    if (protocolSupplied) {
        const std::filesystem::path protocol(config.culture_protocol_path);
        const auto protocolResult =
            metalrobo::validateNeuronCultureRunManifest(
                protocol, runtime->culturePack.fingerprint(),
                "potter-switch-v1");
        requireBuild(
            protocolResult.succeeded(),
            MRNX_RUNTIME_ASSET_FAILURE_V1,
            "culture protocol validation failed: " + protocolResult.message);
        runtime->cultureProtocolPath = protocol.string();
    }
    runtime->cultureWindowTicks = config.culture_window_ticks;
    runtime->cultureCurrentPerNewton = config.culture_current_per_newton;
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
    std::optional<metalrobo::HumanBehaviorTelemetrySnapshot>
        behaviorBeforeSubmit;
    try {
        if (runtime->behavior != nullptr)
            behaviorBeforeSubmit = runtime->behavior->snapshot();
    } catch (...) {
        const std::lock_guard lock(runtime->mutex);
        runtime->behaviorError =
            "behavior pre-submit checkpoint allocation failed";
        runtime->info.request_failure_stage = 640u;
        runtime->beginInProgress = false;
        runtime->info.status = MRNX_RUNTIME_METAL_FAILURE_V1;
        return false;
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
    supplemental.supportPointCount = MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT;
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
    supplemental.expectedSupportConsequencesGPUAddress = active->supportConsequencesGPUAddress;
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
    humanInput.timestepSeconds = static_cast<float>(runtime->timestepSeconds);
    humanInput.timestampQuantumNanoseconds =
        runtime->clockQuantumNanoseconds;
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
        .q = runtime->assets.initialQ,
        .rootTranslations = runtime->assets.initialRootTranslations,
        .v = runtime->assets.initialV,
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
            .v = runtime->assets.initialV,
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
            // Support is owned by Matter's coupled system. Root assistance
            // cannot substitute for source contact or authored tissue.
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
    const auto submitted = [&] {
        struct EncodingScope {
            RuntimeState& runtime;
            ~EncodingScope() { runtime.encodingActive = nullptr; }
        } scope{*runtime};
        runtime->encodingActive = active.get();
        return runtime->owner->submit(runtime->assets.model, ownerInput, *submission);
    }();
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
        const bool behaviorRestored = !behaviorBeforeSubmit.has_value() ||
            (runtime->behavior != nullptr && runtime->behavior->restore(
                *behaviorBeforeSubmit, runtime->behaviorError));
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = 900u +
                static_cast<std::uint32_t>(submitted.status);
            if (!behaviorRestored) runtime->terminalQuarantine = true;
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

[[nodiscard]] bool validateRootRequestV3CPUAdmissionLocked(
    const std::shared_ptr<RuntimeState>& runtime,
    const mrnx_physical_root_request_v3& request,
    MRNumanXBrainJointTransactionTokenV2& root,
    MRNumanXBrainJointSubstepTokenV2& substep,
    MRNumanXBrainMotorCandidateV2& candidate,
    std::uint32_t& failureStage
) noexcept {
    failureStage = 1u;
    if (runtime == nullptr || !runtime->exactClock ||
        request.abi_version != MRNX_PHYSICAL_ROOT_REQUEST_ABI_V3 ||
        request.struct_size != sizeof(request) ||
        request.root.control_step_identifier >
            std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    std::memcpy(&root, &request.root, sizeof(root));
    std::memcpy(&substep, &request.substep, sizeof(substep));
    std::memcpy(&candidate, &request.candidate, sizeof(candidate));

    failureStage = 10u;
    if (!metalrobo::metalNumanXBrainJointTransactionV2Valid(root) ||
        root.environmentIdentifier != 0u ||
        root.clockDomain != MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS ||
        root.clockQuantumNanoseconds !=
            MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS ||
        root.basePhysicsGeneration ==
            std::numeric_limits<std::uint64_t>::max() ||
        root.committedTimestampNanoseconds >
            std::numeric_limits<std::uint64_t>::max() -
                runtime->timestepNanoseconds ||
        root.targetTimestampNanoseconds !=
            root.committedTimestampNanoseconds +
                runtime->timestepNanoseconds) {
        return false;
    }
    if (runtime->behavior != nullptr && !runtime->publishedOnce &&
        root.committedTimestampNanoseconds !=
            runtime->behaviorInitialTimestampNanoseconds) {
        return false;
    }

    failureStage = 2u;
    if (!metalrobo::metalNumanXBrainJointSubstepV2Valid(root, substep) ||
        substep.substepIndex != 0u || substep.attemptIndex != 0u ||
        substep.clockDomain != root.clockDomain ||
        substep.clockQuantumNanoseconds != root.clockQuantumNanoseconds ||
        substep.durationNanoseconds != runtime->timestepNanoseconds ||
        substep.startTimestampNanoseconds !=
            root.committedTimestampNanoseconds ||
        substep.candidateTimestampNanoseconds !=
            root.targetTimestampNanoseconds) {
        return false;
    }

    failureStage = 3u;
    if (!metalrobo::metalNumanXBrainMotorCandidateV2Valid(
            root, substep, candidate) ||
        candidate.flags !=
            (MR_NUMANX_BRAIN_MOTOR_CANDIDATE_VALID |
             MR_NUMANX_BRAIN_MOTOR_CANDIDATE_DECISION_SHADOW) ||
        candidate.clockDomain != root.clockDomain ||
        candidate.muscleCount != MRNX_FULL_BODY_MUSCLE_COUNT ||
        candidate.environmentIdentifier != 0u ||
        candidate.actuatorCommandKind !=
            MR_NUMANX_BRAIN_ACTUATOR_MUSCLE_EXCITATION ||
        candidate.motorOutputHeaderByteCount !=
            sizeof(MRNumanXBrainMotorOutputHeaderGPUV2) ||
        candidate.muscleExcitationByteCount !=
            MRNX_FULL_BODY_MUSCLE_COUNT * sizeof(float) ||
        candidate.autonomicCommandByteCount !=
            MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT ||
        candidate.activeSensingCommandByteCount !=
            MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT ||
        candidate.autonomicCommandCount != 1u ||
        candidate.activeSensingCommandCount != 1u) {
        return false;
    }

    failureStage = 41u;
    if (!validateExactRangeDescriptorMetadata(
            request.motor_header,
            sizeof(MRNumanXBrainMotorOutputHeaderGPUV2),
            MRNX_ELEMENT_BRAIN_MOTOR_OUTPUT_HEADER_V2,
            sizeof(MRNumanXBrainMotorOutputHeaderGPUV2),
            MR_NUMANX_BRAIN_EXACT_RECORD_ALIGNMENT)) {
        return false;
    }
    failureStage = 42u;
    if (!validateExactRangeDescriptorMetadata(
            request.muscle_excitation,
            MRNX_FULL_BODY_MUSCLE_COUNT * sizeof(float),
            MRNX_ELEMENT_FLOAT32_V1, sizeof(float), alignof(float))) {
        return false;
    }
    failureStage = 43u;
    if (!validateExactRangeDescriptorMetadata(
            request.autonomic_command,
            MR_NUMANX_BRAIN_AUTONOMIC_COMMAND_BYTE_COUNT,
            MRNX_ELEMENT_RAW_BYTES_V1, 1u, alignof(std::uint32_t))) {
        return false;
    }
    failureStage = 44u;
    if (!validateExactRangeDescriptorMetadata(
            request.active_sensing_command,
            MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT,
            MRNX_ELEMENT_RAW_BYTES_V1, 1u, alignof(std::uint32_t))) {
        return false;
    }
    failureStage = 45u;
    if (!validateExactRangeDescriptorMetadata(
            request.motor_ready_gate,
            sizeof(MRNumanXBrainMotorReadyGateGPUV2),
            MRNX_ELEMENT_BRAIN_MOTOR_READY_GATE_V2,
            sizeof(MRNumanXBrainMotorReadyGateGPUV2),
            MR_NUMANX_BRAIN_EXACT_RECORD_ALIGNMENT)) {
        return false;
    }

    failureStage = 5u;
    const mrnx_metal_range_v1* ranges[] = {
        &request.motor_header, &request.muscle_excitation,
        &request.autonomic_command, &request.active_sensing_command,
        &request.motor_ready_gate};
    for (std::size_t first = 0u; first < std::size(ranges); ++first) {
        for (std::size_t second = first + 1u;
             second < std::size(ranges); ++second) {
            if (ranges[first]->metal_buffer == ranges[second]->metal_buffer ||
                !disjoint(
                    ranges[first]->gpu_address, ranges[first]->byte_count,
                    ranges[second]->gpu_address, ranges[second]->byte_count)) {
                return false;
            }
        }
    }

    failureStage = 6u;
    if (request.motor_ready.abi_version != MRNX_BRIDGE_ABI_V1 ||
        request.motor_ready.struct_size != sizeof(request.motor_ready) ||
        request.motor_ready.shared_event == nullptr ||
        request.motor_ready.value == 0u ||
        request.motor_ready.device_registry_id !=
            runtime->info.device_registry_id) return false;
    failureStage = 62u;
    if (candidate.motorOutputHeaderGPUAddress !=
        request.motor_header.gpu_address)
        return false;
    failureStage = 63u;
    if (candidate.muscleExcitationGPUAddress !=
        request.muscle_excitation.gpu_address)
        return false;
    failureStage = 64u;
    if (candidate.autonomicCommandGPUAddress !=
        request.autonomic_command.gpu_address)
        return false;
    failureStage = 65u;
    if (candidate.activeSensingCommandGPUAddress !=
        request.active_sensing_command.gpu_address) return false;

    failureStage = 7u;
    if (runtime->publishedOnce) {
        if (runtime->publishedTransactionFingerprint == 0u ||
            runtime->publishedBrainGeneration == 0u ||
            runtime->publishedPhysicsGeneration == 0u ||
            runtime->publishedTimestampNanoseconds == 0u ||
            runtime->exactAggregate.publication_epoch == 0u ||
            runtime->exactAggregate.sensor_packet.
                    accepted_physics_token_fingerprint == 0u ||
            runtime->exactAggregate.sensor_packet.
                    human_io_program_fingerprint == 0u ||
            root.baseBrainGeneration != runtime->publishedBrainGeneration ||
            root.basePhysicsGeneration !=
                runtime->publishedPhysicsGeneration ||
            root.committedTimestampNanoseconds !=
                runtime->publishedTimestampNanoseconds ||
            runtime->publishedControlStep ==
                std::numeric_limits<std::uint64_t>::max() ||
            root.controlStepIdentifier !=
                runtime->publishedControlStep + 1u) {
            return false;
        }
    } else if (root.baseBrainGeneration != 0u ||
               root.basePhysicsGeneration != 0u ||
               runtime->exactAggregate.publication_epoch != 0u ||
               root.controlStepIdentifier != 1u) {
        return false;
    }
    failureStage = 0u;
    return true;
}

[[nodiscard]] bool importRootRequestV3Resources(
    const std::shared_ptr<RuntimeState>& runtime,
    const mrnx_physical_root_request_v3& request,
    const MRNumanXBrainJointTransactionTokenV2& root,
    const MRNumanXBrainJointSubstepTokenV2& substep,
    std::shared_ptr<ActiveRoot>& active,
    std::uint32_t& failureStage
) noexcept {
    failureStage = 40u;
    std::shared_ptr<ActiveRoot> result;
    try {
        result = std::make_shared<ActiveRoot>();
    } catch (...) {
        return false;
    }
    result->runtime = runtime.get();
    result->exactFamily = true;
    result->cultureSettled = runtime->culture == nullptr;
    result->cultureReady = runtime->culture == nullptr;

    failureStage = 41u;
    if (!importExactRange(
            runtime->device, request.motor_header,
            sizeof(MRNumanXBrainMotorOutputHeaderGPUV2),
            MRNX_ELEMENT_BRAIN_MOTOR_OUTPUT_HEADER_V2,
            sizeof(MRNumanXBrainMotorOutputHeaderGPUV2),
            result->motorHeader)) return false;
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
            MRNX_ELEMENT_RAW_BYTES_V1, 1u, result->autonomic)) return false;
    failureStage = 44u;
    if (!importExactRange(
            runtime->device, request.active_sensing_command,
            MR_NUMANX_BRAIN_ACTIVE_SENSING_COMMAND_BYTE_COUNT,
            MRNX_ELEMENT_RAW_BYTES_V1, 1u, result->activeSensing)) return false;
    failureStage = 45u;
    if (!importExactRange(
            runtime->device, request.motor_ready_gate,
            sizeof(MRNumanXBrainMotorReadyGateGPUV2),
            MRNX_ELEMENT_BRAIN_MOTOR_READY_GATE_V2,
            sizeof(MRNumanXBrainMotorReadyGateGPUV2),
            result->motorReadyGate)) return false;

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
        request.motor_ready.value == 0u) return false;
    failureStage = 61u;
    if (request.motor_ready.device_registry_id != runtime->device.registryID ||
        !eventObject(request.motor_ready.shared_event, event) ||
        !importableSharedEvent(runtime->device, event)) return false;
    result->motorReadyEvent = event;
    result->transactionFingerprint = root.transactionFingerprint;
    result->brainGeneration = root.shadowGeneration;
    result->controlStep = root.controlStepIdentifier;
    result->acceptedTimestampNanoseconds =
        substep.candidateTimestampNanoseconds;
    result->receptorTimestampNanoseconds = substep.startTimestampNanoseconds;
    {
        const std::lock_guard lock(runtime->mutex);
        if (runtime->publishedOnce) {
            result->previousTransactionFingerprint =
                runtime->publishedTransactionFingerprint;
            result->previousPhysicsGeneration =
                runtime->publishedPhysicsGeneration;
            result->previousAcceptedTokenFingerprint = runtime->exactAggregate.
                sensor_packet.accepted_physics_token_fingerprint;
            result->previousHumanIOProgramFingerprint = runtime->exactAggregate.
                sensor_packet.human_io_program_fingerprint;
        }
    }
    active = std::move(result);
    failureStage = 0u;
    return true;
}

[[nodiscard]] bool beginPhysicalRootV3(
    const std::shared_ptr<RuntimeState>& runtime,
    const mrnx_physical_root_request_v3& request,
    void* completionContext,
    const mrnx_physical_root_settled_callback_v1 completion
) {
    if (runtime == nullptr || completion == nullptr) return false;
    {
        const std::lock_guard lock(runtime->mutex);
        if (!runtime->exactClock || runtime->beginInProgress ||
            runtime->active != nullptr || runtime->terminalQuarantine ||
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

    MRNumanXBrainJointTransactionTokenV2 root{};
    MRNumanXBrainJointSubstepTokenV2 substep{};
    MRNumanXBrainMotorCandidateV2 candidate{};
    std::uint32_t failureStage = 0u;
    {
        const std::lock_guard lock(runtime->mutex);
        if (!validateRootRequestV3CPUAdmissionLocked(
                runtime, request, root, substep, candidate, failureStage)) {
            runtime->info.request_failure_stage = failureStage;
            runtime->beginInProgress = false;
            runtime->info.status = MRNX_RUNTIME_INVALID_REQUEST_V1;
            return false;
        }
        runtime->lastAttemptedControlStep = root.controlStepIdentifier;
    }

    std::shared_ptr<ActiveRoot> active;
    if (!importRootRequestV3Resources(
            runtime, request, root, substep, active, failureStage)) {
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = failureStage;
        }
        return failBegin(MRNX_RUNTIME_INVALID_REQUEST_V1);
    }
    std::optional<metalrobo::HumanBehaviorTelemetrySnapshot>
        behaviorBeforeSubmit;
    try {
        if (runtime->behavior != nullptr)
            behaviorBeforeSubmit = runtime->behavior->snapshot();
    } catch (...) {
        const std::lock_guard lock(runtime->mutex);
        runtime->behaviorError =
            "behavior pre-submit checkpoint allocation failed";
        runtime->info.request_failure_stage = 640u;
        runtime->beginInProgress = false;
        runtime->info.status = MRNX_RUNTIME_METAL_FAILURE_V1;
        return false;
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
    active->completion = completion;
    active->completionContext = completionContext;
    if (!allocateSupplementalBuffers(runtime, active)) {
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = 650u;
        }
        return failBegin(MRNX_RUNTIME_METAL_FAILURE_V1);
    }

    auto& supplemental = active->supplementalDispatch;
    supplemental.abiVersion = MR_NUMANX_HUMAN_IO_ABI_VERSION;
    supplemental.qCoordinateCount = MRNX_FULL_BODY_NQ;
    supplemental.dofCount = MRNX_FULL_BODY_NV;
    supplemental.bodyCount = runtime->assets.rigid.engineBodyCount;
    supplemental.pointCount = static_cast<std::uint32_t>(
        runtime->assets.points.size());
    supplemental.supportPointOffset = runtime->assets.rigid.engineBodyCount;
    supplemental.supportPointCount = MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT;
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
    supplemental.programFingerprint = runtime->supplementalProgramFingerprint;
    supplemental.expectedSupportConsequencesGPUAddress =
        active->supportConsequencesGPUAddress;
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

    metalrobo::MetalNumanXHumanIOInputV2 humanInput{};
    humanInput.root = root;
    humanInput.substep = substep;
    humanInput.candidate = candidate;
    humanInput.motorOutputHeaderMetalBuffer =
        (__bridge void*)active->motorHeader.buffer;
    humanInput.motorOutputHeaderByteOffset = active->motorHeader.byteOffset;
    humanInput.motorOutputHeaderByteCount = active->motorHeader.byteCount;
    humanInput.motorOutputHeaderEnvironmentStride =
        active->motorHeader.byteCount;
    humanInput.expectedMotorOutputHeaderGPUAddress =
        active->motorHeader.address;
    humanInput.excitationMetalBuffer =
        (__bridge void*)active->excitation.buffer;
    humanInput.excitationByteOffset = active->excitation.byteOffset;
    humanInput.excitationByteCount = active->excitation.byteCount;
    humanInput.excitationEnvironmentStride = MRNX_FULL_BODY_MUSCLE_COUNT;
    humanInput.expectedExcitationGPUAddress = active->excitation.address;
    humanInput.autonomicCommandMetalBuffer =
        (__bridge void*)active->autonomic.buffer;
    humanInput.autonomicCommandByteOffset = active->autonomic.byteOffset;
    humanInput.autonomicCommandByteCount = active->autonomic.byteCount;
    humanInput.expectedAutonomicCommandGPUAddress =
        active->autonomic.address;
    humanInput.activeSensingCommandMetalBuffer =
        (__bridge void*)active->activeSensing.buffer;
    humanInput.activeSensingCommandByteOffset =
        active->activeSensing.byteOffset;
    humanInput.activeSensingCommandByteCount =
        active->activeSensing.byteCount;
    humanInput.expectedActiveSensingCommandGPUAddress =
        active->activeSensing.address;
    humanInput.motorReadyGateMetalBuffer =
        (__bridge void*)active->motorReadyGate.buffer;
    humanInput.motorReadyGateByteOffset = active->motorReadyGate.byteOffset;
    humanInput.motorReadyGateByteCount = active->motorReadyGate.byteCount;
    humanInput.expectedMotorReadyGateGPUAddress =
        active->motorReadyGate.address;
    humanInput.motorReadySharedEvent =
        (__bridge void*)active->motorReadyEvent;
    humanInput.motorReadySharedEventValue = request.motor_ready.value;
    humanInput.environmentCount = 1u;
    humanInput.muscleCount = MRNX_FULL_BODY_MUSCLE_COUNT;
    humanInput.stepCount = 1u;
    humanInput.timestepNanoseconds = runtime->timestepNanoseconds;
    humanInput.receptorTimestampNanoseconds =
        substep.startTimestampNanoseconds;
    humanInput.candidateSensorGeneration = sensorGeneration;
    humanInput.supplementalProgram.context = active.get();
    humanInput.supplementalProgram.encode = &encodeSupplementalSensors;
    humanInput.supplementalProgram.fingerprint =
        runtime->supplementalProgramFingerprint;
    metalrobo::MetalNumanXTransactionProgram humanProgram{};
    metalrobo::MetalNumanXHumanIOExactPreparedView exactHumanIO{};
    const auto humanPrepared = runtime->humanIO->prepare(
        humanInput, humanProgram, exactHumanIO);
    if (!humanPrepared.succeeded() || !humanProgram.valid() ||
        !exactHumanIO.valid()) {
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = 700u +
                static_cast<std::uint32_t>(humanPrepared.status);
        }
        return failBegin(MRNX_RUNTIME_INVALID_REQUEST_V1);
    }
    active->exactHumanIO = exactHumanIO;

    metalrobo::MetalNumanXHumanMatterTransactionV2 transaction{};
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
    transaction.seed = root.transactionFingerprint;
    if (transaction.seed == 0u) transaction.seed = kFnvOffset;
    transaction.transactionFingerprint = root.transactionFingerprint;
    transaction.substepFingerprint = substep.substepFingerprint;
    transaction.physicsGeneration = active->physicsGeneration;
    transaction.linearizationEpoch = linearizationEpoch;
    transaction.slotGeneration = slotGeneration;
    const auto humanMatterProgram = runtime->adapter->program(
        transaction, exactHumanIO);
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
        .q = runtime->assets.initialQ,
        .rootTranslations = runtime->assets.initialRootTranslations,
        .v = runtime->assets.initialV,
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
            .v = runtime->assets.initialV,
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
            .previousPhysicsGeneration = active->previousPhysicsGeneration,
            .previousAcceptedTokenFingerprint =
                active->previousAcceptedTokenFingerprint,
            .previousHumanIOProgramFingerprint =
                active->previousHumanIOProgramFingerprint,
        },
    };
    auto submission = std::make_unique<
        metalrobo::MetalArticulatedOperatorSubmission>();
    const auto submitted = [&] {
        struct EncodingScope {
            RuntimeState& runtime;
            ~EncodingScope() { runtime.encodingActive = nullptr; }
        } scope{*runtime};
        runtime->encodingActive = active.get();
        return runtime->owner->submit(
            runtime->assets.model, ownerInput, *submission);
    }();
    if (!submitted.succeeded() || !submitted.dispatched ||
        !submission->valid()) {
        if (std::getenv("MRNX_PHYSICAL_DIAGNOSTICS") != nullptr) {
            std::fprintf(
                stderr, "mrnx_exact_owner_submit_failure status=%u message=%s\n",
                static_cast<unsigned>(submitted.status),
                submitted.message.c_str());
        }
        (void)runtime->humanIO->cancelPrepared(
            root.transactionFingerprint, humanProgram.fingerprint);
        const bool behaviorRestored = !behaviorBeforeSubmit.has_value() ||
            (runtime->behavior != nullptr && runtime->behavior->restore(
                *behaviorBeforeSubmit, runtime->behaviorError));
        {
            const std::lock_guard lock(runtime->mutex);
            runtime->info.request_failure_stage = 900u +
                static_cast<std::uint32_t>(submitted.status);
            if (!behaviorRestored) runtime->terminalQuarantine = true;
        }
        return failBegin(MRNX_RUNTIME_SUBMISSION_FAILURE_V1);
    }

    metalrobo::MetalNumanXHumanIOTransactionKey key{};
    metalrobo::MetalNumanXHumanIOSensorView candidateView{};
    const auto pending = runtime->humanIO->pendingCandidate(key, candidateView);
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
    auto* prepared = metalrobo::numanx_bridge_v1::adoptPreparedV2(
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

mrnx_runtime_v1* mrnx_bridge_v1_runtime_create_v2(
    const mrnx_runtime_config_v2* config,
    mrnx_runtime_info_v1* info
) {
    @autoreleasepool {
        if (config == nullptr) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
        try {
            auto state = createRuntimeStateV2(*config);
            auto* runtime = new (std::nothrow) mrnx_runtime_v1;
            if (runtime == nullptr) {
                fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
                return nullptr;
            }
            runtime->state = std::move(state);
            if (info != nullptr) *info = runtime->state->info;
            return runtime;
        } catch (const RuntimeBuildFailure& failure) {
            if (std::getenv("MRNX_RUNTIME_DIAGNOSTICS") != nullptr) {
                std::fprintf(
                    stderr, "mrnx runtime v2 create failed: %s\n",
                    failure.what());
            }
            fillRuntimeInfoFailure(info, failure.status);
            return nullptr;
        } catch (...) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
    }
}

mrnx_runtime_v1* mrnx_bridge_v1_runtime_create_v3(
    const mrnx_runtime_config_v3* config,
    mrnx_runtime_info_v1* info
) {
    @autoreleasepool {
        if (config == nullptr) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
        try {
            requireBuild(
                config->abi_version == MRNX_RUNTIME_CONFIG_ABI_V3 &&
                    config->struct_size == sizeof(*config) &&
                    config->matter_world_package_path != nullptr &&
                    config->matter_world_package_path[0] != '\0' &&
                    config->expected_model_source_fingerprint != 0u &&
                    config->expected_matter_world_fingerprint != 0u &&
                    config->runtime.matter_material_path == nullptr,
                MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
                "invalid authored-world configuration v3");
            auto state = createRuntimeStateV2(config->runtime, config);
            auto* runtime = new (std::nothrow) mrnx_runtime_v1;
            if (runtime == nullptr) {
                fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
                return nullptr;
            }
            runtime->state = std::move(state);
            if (info != nullptr) *info = runtime->state->info;
            return runtime;
        } catch (const RuntimeBuildFailure& failure) {
            if (std::getenv("MRNX_RUNTIME_DIAGNOSTICS") != nullptr) {
                std::fprintf(
                    stderr, "mrnx runtime v3 create failed: %s\n",
                    failure.what());
            }
            fillRuntimeInfoFailure(info, failure.status);
            return nullptr;
        } catch (...) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
    }
}

mrnx_runtime_v1* mrnx_bridge_v1_runtime_create_v4(
    const mrnx_runtime_config_v4* config,
    mrnx_runtime_info_v1* info
) {
    @autoreleasepool {
        if (config == nullptr) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
        try {
            requireBuild(
                config->abi_version == MRNX_RUNTIME_CONFIG_ABI_V4 &&
                    config->struct_size == sizeof(*config) &&
                    config->runtime.abi_version == MRNX_RUNTIME_CONFIG_ABI_V3 &&
                    config->runtime.struct_size == sizeof(config->runtime) &&
                    config->joint_equality_payload_path != nullptr &&
                    config->joint_equality_payload_path[0] != '\0' &&
                    config->expected_joint_equality_fingerprint != 0u &&
                    config->runtime.matter_world_package_path != nullptr &&
                    config->runtime.matter_world_package_path[0] != '\0' &&
                    config->runtime.expected_model_source_fingerprint != 0u &&
                    config->runtime.expected_matter_world_fingerprint != 0u &&
                    config->runtime.runtime.matter_material_path == nullptr,
                MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
                "invalid authored-world configuration v4");
            auto state = createRuntimeStateV2(config->runtime.runtime, &config->runtime, config);
            auto* runtime = new (std::nothrow) mrnx_runtime_v1;
            if (runtime == nullptr) {
                fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
                return nullptr;
            }
            runtime->state = std::move(state);
            if (info != nullptr) *info = runtime->state->info;
            return runtime;
        } catch (const RuntimeBuildFailure& failure) {
            if (std::getenv("MRNX_RUNTIME_DIAGNOSTICS") != nullptr) {
                std::fprintf(
                    stderr, "mrnx runtime v4 create failed: %s\n",
                    failure.what());
            }
            fillRuntimeInfoFailure(info, failure.status);
            return nullptr;
        } catch (...) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
    }
}

mrnx_runtime_v1* mrnx_bridge_v1_runtime_create_v5(
    const mrnx_runtime_config_v5* config,
    mrnx_runtime_info_v1* info
) {
    @autoreleasepool {
        if (config == nullptr) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
        try {
            requireBuild(
                config->abi_version == MRNX_RUNTIME_CONFIG_ABI_V5 &&
                    config->struct_size == sizeof(*config) &&
                    config->costal_cartilage_payload_path != nullptr &&
                    config->costal_cartilage_payload_path[0] != '\0' &&
                    config->costal_binding_payload_path != nullptr &&
                    config->costal_binding_payload_path[0] != '\0' &&
                    config->expected_costal_binding_fingerprint != 0u &&
                    config->runtime.abi_version == MRNX_RUNTIME_CONFIG_ABI_V4 &&
                    config->runtime.struct_size == sizeof(config->runtime) &&
                    config->runtime.runtime.abi_version == MRNX_RUNTIME_CONFIG_ABI_V3 &&
                    config->runtime.runtime.struct_size == sizeof(config->runtime.runtime) &&
                    config->runtime.joint_equality_payload_path != nullptr &&
                    config->runtime.joint_equality_payload_path[0] != '\0' &&
                    config->runtime.expected_joint_equality_fingerprint != 0u &&
                    config->runtime.runtime.matter_world_package_path != nullptr &&
                    config->runtime.runtime.matter_world_package_path[0] != '\0' &&
                    config->runtime.runtime.expected_model_source_fingerprint != 0u &&
                    config->runtime.runtime.expected_matter_world_fingerprint != 0u &&
                    config->runtime.runtime.runtime.matter_material_path == nullptr,
                MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
                "invalid authored-world configuration v5");
            auto state = createRuntimeStateV2(config->runtime.runtime.runtime, &config->runtime.runtime, &config->runtime, config);
            auto* runtime = new (std::nothrow) mrnx_runtime_v1;
            if (runtime == nullptr) {
                fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
                return nullptr;
            }
            runtime->state = std::move(state);
            if (info != nullptr) *info = runtime->state->info;
            return runtime;
        } catch (const RuntimeBuildFailure& failure) {
            if (std::getenv("MRNX_RUNTIME_DIAGNOSTICS") != nullptr) {
                std::fprintf(
                    stderr, "mrnx runtime v5 create failed: %s\n",
                    failure.what());
            }
            fillRuntimeInfoFailure(info, failure.status);
            return nullptr;
        } catch (...) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
    }
}

static mrnx_runtime_v1* createRuntimeV6OrV7(
    const mrnx_runtime_config_v6* config,
    mrnx_runtime_info_v1* info,
    const mrnx_runtime_config_v7* initialConfig,
    const std::uint64_t exactTimestepNanoseconds = 0u
) {
    @autoreleasepool {
        if (config == nullptr) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
        try {
            requireBuild(
                config->abi_version == MRNX_RUNTIME_CONFIG_ABI_V6 &&
                    config->struct_size == sizeof(*config) &&
                    config->joint_limit_payload_path != nullptr &&
                    config->joint_limit_payload_path[0] != '\0' &&
                    config->expected_joint_limit_fingerprint != 0u &&
                    ((config->costal_cartilage_payload_path == nullptr &&
                      config->costal_binding_payload_path == nullptr &&
                      config->expected_costal_binding_fingerprint == 0u) ||
                     (config->costal_cartilage_payload_path != nullptr &&
                      config->costal_cartilage_payload_path[0] != '\0' &&
                      config->costal_binding_payload_path != nullptr &&
                      config->costal_binding_payload_path[0] != '\0' &&
                      config->expected_costal_binding_fingerprint != 0u)) &&
                    config->runtime.abi_version == MRNX_RUNTIME_CONFIG_ABI_V4 &&
                    config->runtime.struct_size == sizeof(config->runtime) &&
                    config->runtime.runtime.abi_version == MRNX_RUNTIME_CONFIG_ABI_V3 &&
                    config->runtime.runtime.struct_size == sizeof(config->runtime.runtime) &&
                    config->runtime.joint_equality_payload_path != nullptr &&
                    config->runtime.joint_equality_payload_path[0] != '\0' &&
                    config->runtime.expected_joint_equality_fingerprint != 0u &&
                    config->runtime.runtime.matter_world_package_path != nullptr &&
                    config->runtime.runtime.matter_world_package_path[0] != '\0' &&
                    config->runtime.runtime.expected_model_source_fingerprint != 0u &&
                    config->runtime.runtime.expected_matter_world_fingerprint != 0u &&
                    config->runtime.runtime.runtime.matter_material_path == nullptr,
                MRNX_RUNTIME_INVALID_CONFIGURATION_V1,
                "invalid authored-world configuration v6");
            mrnx_runtime_config_v5 tissue{};
            tissue.abi_version = MRNX_RUNTIME_CONFIG_ABI_V5;
            tissue.struct_size = sizeof(tissue);
            tissue.runtime = config->runtime;
            tissue.costal_cartilage_payload_path = config->costal_cartilage_payload_path;
            tissue.costal_binding_payload_path = config->costal_binding_payload_path;
            tissue.expected_costal_binding_fingerprint = config->expected_costal_binding_fingerprint;
            auto state = createRuntimeStateV2(config->runtime.runtime.runtime, &config->runtime.runtime,
                &config->runtime, config->costal_binding_payload_path != nullptr ? &tissue : nullptr,
                config, initialConfig, exactTimestepNanoseconds);
            auto* runtime = new (std::nothrow) mrnx_runtime_v1;
            if (runtime == nullptr) {
                fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
                return nullptr;
            }
            runtime->state = std::move(state);
            if (info != nullptr) *info = runtime->state->info;
            return runtime;
        } catch (const RuntimeBuildFailure& failure) {
            if (std::getenv("MRNX_RUNTIME_DIAGNOSTICS") != nullptr) {
                std::fprintf(
                    stderr, "mrnx runtime v6 create failed: %s\n",
                    failure.what());
            }
            fillRuntimeInfoFailure(info, failure.status);
            return nullptr;
        } catch (...) {
            fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
            return nullptr;
        }
    }
}

mrnx_runtime_v1* mrnx_bridge_v1_runtime_create_v6(
    const mrnx_runtime_config_v6* config, mrnx_runtime_info_v1* info
) {
    return createRuntimeV6OrV7(config, info, nullptr);
}

mrnx_runtime_v1* mrnx_bridge_v1_runtime_create_v7(
    const mrnx_runtime_config_v7* config, mrnx_runtime_info_v1* info
) {
    if (config == nullptr || config->abi_version != MRNX_RUNTIME_CONFIG_ABI_V7 ||
        config->struct_size != sizeof(*config) || config->initial_state_payload_path == nullptr ||
        config->initial_state_payload_path[0] == '\0' || config->expected_initial_state_fingerprint == 0u) {
        fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
        return nullptr;
    }
    return createRuntimeV6OrV7(&config->runtime, info, config);
}

mrnx_runtime_v1* mrnx_bridge_v1_runtime_create_v8(
    const mrnx_runtime_config_v8* config, mrnx_runtime_info_v1* info
) {
    if (config == nullptr || config->abi_version != MRNX_RUNTIME_CONFIG_ABI_V8 ||
        config->struct_size != sizeof(*config) ||
        config->timestep_nanoseconds == 0u ||
        config->timestep_nanoseconds > 1'000'000'000u ||
        config->runtime.abi_version != MRNX_RUNTIME_CONFIG_ABI_V7 ||
        config->runtime.struct_size != sizeof(config->runtime) ||
        config->runtime.runtime.runtime.runtime.runtime.timestep_microseconds != 0u ||
        config->runtime.initial_state_payload_path == nullptr ||
        config->runtime.initial_state_payload_path[0] == '\0' ||
        config->runtime.expected_initial_state_fingerprint == 0u) {
        fillRuntimeInfoFailure(info, MRNX_RUNTIME_INVALID_CONFIGURATION_V1);
        return nullptr;
    }
    return createRuntimeV6OrV7(&config->runtime.runtime, info,
        &config->runtime, config->timestep_nanoseconds);
}

bool mrnx_bridge_v1_runtime_copy_world_info(
    const mrnx_runtime_v1* runtime,
    mrnx_runtime_world_info_v1* info
) {
    if (runtime == nullptr || runtime->state == nullptr || info == nullptr ||
        info->abi_version != MRNX_BRIDGE_ABI_V1 ||
        info->struct_size != sizeof(*info)) return false;
    *info = runtime->state->worldInfo;
    return true;
}

bool mrnx_bridge_v1_runtime_copy_exact_clock(
    const mrnx_runtime_v1* runtime,
    mrnx_exact_clock_info_v1* info
) {
    if (runtime == nullptr || runtime->state == nullptr || info == nullptr ||
        info->abi_version != MRNX_EXACT_CLOCK_INFO_ABI_V1 ||
        info->struct_size != sizeof(*info) || !runtime->state->exactClock) {
        return false;
    }
    struct Read {
        RuntimeState* state;
        mrnx_exact_clock_info_v1* output;
    } read{runtime->state.get(), info};
    *info = {};
    return metalrobo::numanx_bridge_v1::withDomainPublicReadGate(
        runtime->state->domain, &read, +[](void* raw) noexcept {
            auto& context = *static_cast<Read*>(raw);
            if (context.state->terminalQuarantine) return false;
            context.output->abi_version = MRNX_EXACT_CLOCK_INFO_ABI_V1;
            context.output->struct_size = sizeof(*context.output);
            context.output->timestep_nanoseconds =
                context.state->timestepNanoseconds;
            context.output->clock_quantum_nanoseconds =
                context.state->clockQuantumNanoseconds;
            context.output->published_timestamp_nanoseconds =
                context.state->publishedTimestampNanoseconds;
            context.output->publication_epoch =
                context.state->exactAggregate.publication_epoch;
            return context.output->publication_epoch ==
                metalrobo::numanx_bridge_v1::domainPublicationEpoch(
                    context.state->domain);
        });
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
    const auto& limitRows = runtime->state->assets.jointLimits;
    const auto sourceLimit = std::find_if(limitRows.begin(), limitRows.end(),
        [&](const auto& row) { return row.indices.y == coordinateIndex; });
    const bool sourceCompliant = sourceLimit != limitRows.end();
    const bool hasPositionLimit = sourceCompliant || (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u;
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
    if (sourceCompliant) result.flags |= MRNX_JOINT_COORDINATE_SOURCE_COMPLIANT_LIMIT_V1;
    result.parent_local_axis[0] = parentAxis.x;
    result.parent_local_axis[1] = parentAxis.y;
    result.parent_local_axis[2] = parentAxis.z;
    result.minimum_position = sourceCompliant ? sourceLimit->rangeMarginInverseWeight.x :
        (hasPositionLimit ? dof.limits.x : -extent);
    result.maximum_position = sourceCompliant ? sourceLimit->rangeMarginInverseWeight.y :
        (hasPositionLimit ? dof.limits.y : extent);
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
        if (state->exactClock) return false;
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

bool mrnx_bridge_v1_runtime_begin_physical_root_v2(
    mrnx_runtime_v1* runtime,
    const mrnx_physical_root_request_v2* request,
    void* completionContext,
    const mrnx_physical_root_settled_callback_v1 completion
) {
    @autoreleasepool {
        if (runtime == nullptr || runtime->state == nullptr ||
            request == nullptr || completion == nullptr ||
            !runtime->state->exactClock ||
            request->abi_version != MRNX_PHYSICAL_ROOT_REQUEST_ABI_V2 ||
            request->struct_size != sizeof(*request)) {
            return false;
        }
        const auto state = runtime->state;
        try {
            const std::lock_guard lock(state->mutex);
            if (state->beginInProgress || state->active != nullptr ||
                state->terminalQuarantine) return false;
            // ABI v2 was published with unit-ambiguous all-v1 nested records.
            // Preserve its symbol and layout, but never inspect its borrowed
            // resource descriptors or route it into an exact-clock runtime.
            (void)completionContext;
            state->info.status = MRNX_RUNTIME_INVALID_REQUEST_V1;
            state->info.request_failure_stage =
                MRNX_REQUEST_FAILURE_STAGE_LEGACY_EXACT_V2_UNROUTABLE;
            return false;
        } catch (...) {
            return false;
        }
    }
}

bool mrnx_bridge_v1_runtime_begin_physical_root_v3(
    mrnx_runtime_v1* runtime,
    const mrnx_physical_root_request_v3* request,
    void* completionContext,
    const mrnx_physical_root_settled_callback_v1 completion
) {
    @autoreleasepool {
        if (runtime == nullptr || runtime->state == nullptr ||
            request == nullptr || completion == nullptr ||
            !runtime->state->exactClock ||
            request->abi_version != MRNX_PHYSICAL_ROOT_REQUEST_ABI_V3 ||
            request->struct_size != sizeof(*request)) {
            return false;
        }
        const auto state = runtime->state;
        try {
            return beginPhysicalRootV3(
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
        snapshot->struct_size != sizeof(*snapshot) ||
        runtime->state->exactClock) {
        return false;
    }
    struct Read {
        RuntimeState* state;
        mrnx_aggregate_snapshot_v1* output;
    } read{runtime->state.get(), snapshot};
    *snapshot = {};
    return metalrobo::numanx_bridge_v1::withDomainPublicReadGate(
        runtime->state->domain, &read, +[](void* raw) noexcept {
            auto& context = *static_cast<Read*>(raw);
            if (context.state->terminalQuarantine) return false;
            if (context.state->aggregate.publication_epoch == 0u ||
                context.state->aggregate.publication_epoch !=
                    metalrobo::numanx_bridge_v1::domainPublicationEpoch(
                        context.state->domain) ||
                context.state->publishedProprioception == nil ||
                context.state->publishedProprioceptionValidity == nil ||
                context.state->publishedInteroception == nil ||
                context.state->publishedInteroceptionValidity == nil) {
                return false;
            }
            *context.output = context.state->aggregate;
            return true;
        });
}

bool mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v2(
    const mrnx_runtime_v1* runtime,
    mrnx_aggregate_snapshot_v2* snapshot
) {
    if (runtime == nullptr || runtime->state == nullptr ||
        snapshot == nullptr ||
        snapshot->abi_version != MRNX_AGGREGATE_SNAPSHOT_ABI_V2 ||
        snapshot->struct_size != sizeof(*snapshot) ||
        runtime->state->exactClock) {
        return false;
    }
    struct Read {
        RuntimeState* state;
        mrnx_aggregate_snapshot_v2* output;
    } read{runtime->state.get(), snapshot};
    *snapshot = {};
    return metalrobo::numanx_bridge_v1::withDomainPublicReadGate(
        runtime->state->domain, &read, +[](void* raw) noexcept {
            auto& context = *static_cast<Read*>(raw);
            if (context.state->terminalQuarantine) return false;
            bool channelsReady =
                context.state->aggregateChannelCount == 7u;
            for (std::uint32_t index = 0u;
                 index < context.state->aggregateChannelCount; ++index) {
                channelsReady = channelsReady &&
                    context.state->publishedChannelValues[index] != nil &&
                    context.state->publishedChannelValidity[index] != nil;
            }
            if (context.state->aggregate.publication_epoch == 0u ||
                context.state->aggregate.publication_epoch !=
                    metalrobo::numanx_bridge_v1::domainPublicationEpoch(
                        context.state->domain) ||
                !channelsReady) return false;
            auto& output = *context.output;
            output.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V2;
            output.struct_size = sizeof(output);
            output.publication_epoch =
                context.state->aggregate.publication_epoch;
            output.brain_generation =
                context.state->aggregate.brain_generation;
            output.physics_generation =
                context.state->aggregate.physics_generation;
            output.sensor_generation =
                context.state->aggregate.sensor_generation;
            output.root = context.state->aggregate.root;
            output.sensor = context.state->aggregate.sensor;
            output.channel_count = context.state->aggregateChannelCount;
            output.channel_capacity = MRNX_MAX_SENSOR_CHANNELS_V2;
            for (std::uint32_t index = 0u;
                 index < context.state->aggregateChannelCount; ++index) {
                output.channels[index] =
                    context.state->aggregateChannels[index];
            }
            return true;
        });
}

bool mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v3(
    const mrnx_runtime_v1* runtime,
    mrnx_aggregate_snapshot_v3* snapshot
) {
    if (runtime == nullptr || runtime->state == nullptr ||
        snapshot == nullptr ||
        snapshot->abi_version != MRNX_AGGREGATE_SNAPSHOT_ABI_V3 ||
        snapshot->struct_size != sizeof(*snapshot) ||
        runtime->state->exactClock) {
        return false;
    }
    struct Read {
        RuntimeState* state;
        mrnx_aggregate_snapshot_v3* output;
    } read{runtime->state.get(), snapshot};
    *snapshot = {};
    return metalrobo::numanx_bridge_v1::withDomainPublicReadGate(
        runtime->state->domain, &read, +[](void* raw) noexcept {
            auto& context = *static_cast<Read*>(raw);
            if (context.state->terminalQuarantine) return false;
            bool channelsReady =
                context.state->aggregateChannelCount == 7u;
            for (std::uint32_t index = 0u;
                 index < context.state->aggregateChannelCount; ++index) {
                channelsReady = channelsReady &&
                    context.state->publishedChannelValues[index] != nil &&
                    context.state->publishedChannelValidity[index] != nil;
            }
            if (context.state->aggregate.publication_epoch == 0u ||
                context.state->aggregate.publication_epoch !=
                    metalrobo::numanx_bridge_v1::domainPublicationEpoch(
                        context.state->domain) ||
                context.state->aggregateTiming.timing_fingerprint == 0u ||
                !channelsReady) return false;
            auto& output = *context.output;
            output.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V3;
            output.struct_size = sizeof(output);
            output.publication_epoch =
                context.state->aggregate.publication_epoch;
            output.brain_generation =
                context.state->aggregate.brain_generation;
            output.physics_generation =
                context.state->aggregate.physics_generation;
            output.sensor_generation =
                context.state->aggregate.sensor_generation;
            output.root = context.state->aggregate.root;
            output.sensor = context.state->aggregate.sensor;
            output.timing = context.state->aggregateTiming;
            output.channel_count = context.state->aggregateChannelCount;
            output.channel_capacity = MRNX_MAX_SENSOR_CHANNELS_V2;
            for (std::uint32_t index = 0u;
                 index < context.state->aggregateChannelCount; ++index) {
                output.channels[index] =
                    context.state->aggregateChannels[index];
            }
            return true;
        });
}

bool mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v4(
    const mrnx_runtime_v1* runtime,
    mrnx_aggregate_snapshot_v4* snapshot
) {
    if (runtime == nullptr || runtime->state == nullptr ||
        snapshot == nullptr ||
        snapshot->abi_version != MRNX_AGGREGATE_SNAPSHOT_ABI_V4 ||
        snapshot->struct_size != sizeof(*snapshot) ||
        runtime->state->exactClock) return false;
    struct Read {
        RuntimeState* state;
        mrnx_aggregate_snapshot_v4* output;
    } read{runtime->state.get(), snapshot};
    *snapshot = {};
    return metalrobo::numanx_bridge_v1::withDomainPublicReadGate(
        runtime->state->domain, &read, +[](void* raw) noexcept {
            auto& context = *static_cast<Read*>(raw);
            if (context.state->terminalQuarantine) return false;
            bool channelsReady =
                context.state->aggregateChannelCount == 7u;
            for (std::uint32_t index = 0u;
                 index < context.state->aggregateChannelCount; ++index) {
                channelsReady = channelsReady &&
                    context.state->publishedChannelValues[index] != nil &&
                    context.state->publishedChannelValidity[index] != nil;
            }
            if (context.state->culture == nullptr ||
                context.state->aggregate.publication_epoch == 0u ||
                context.state->aggregate.publication_epoch !=
                    metalrobo::numanx_bridge_v1::domainPublicationEpoch(
                        context.state->domain) ||
                context.state->aggregateTiming.timing_fingerprint == 0u ||
                context.state->aggregateCulture.culture_fingerprint == 0u ||
                context.state->aggregateCulture.generation == 0u ||
                context.state->aggregateCulture.source_root_fingerprint !=
                    context.state->aggregate.root.transaction_fingerprint ||
                context.state->aggregateCulture.receipt_fingerprint == 0u ||
                !context.state->publishedCultureView.valid() ||
                !channelsReady) return false;
            auto& output = *context.output;
            output.abi_version = MRNX_AGGREGATE_SNAPSHOT_ABI_V4;
            output.struct_size = sizeof(output);
            output.publication_epoch =
                context.state->aggregate.publication_epoch;
            output.brain_generation =
                context.state->aggregate.brain_generation;
            output.physics_generation =
                context.state->aggregate.physics_generation;
            output.sensor_generation =
                context.state->aggregate.sensor_generation;
            output.root = context.state->aggregate.root;
            output.sensor = context.state->aggregate.sensor;
            output.timing = context.state->aggregateTiming;
            output.channel_count = context.state->aggregateChannelCount;
            output.channel_capacity = MRNX_MAX_SENSOR_CHANNELS_V2;
            for (std::uint32_t index = 0u;
                 index < context.state->aggregateChannelCount; ++index) {
                output.channels[index] =
                    context.state->aggregateChannels[index];
            }
            output.culture = context.state->aggregateCulture;
            return true;
        });
}

bool mrnx_bridge_v1_runtime_copy_aggregate_snapshot_v5(
    const mrnx_runtime_v1* runtime,
    mrnx_aggregate_snapshot_v5* snapshot
) {
    if (runtime == nullptr || runtime->state == nullptr ||
        snapshot == nullptr ||
        snapshot->abi_version != MRNX_AGGREGATE_SNAPSHOT_ABI_V5 ||
        snapshot->struct_size != sizeof(*snapshot) ||
        !runtime->state->exactClock) return false;
    struct Read {
        RuntimeState* state;
        mrnx_aggregate_snapshot_v5* output;
    } read{runtime->state.get(), snapshot};
    *snapshot = {};
    return metalrobo::numanx_bridge_v1::withDomainPublicReadGate(
        runtime->state->domain, &read, +[](void* raw) noexcept {
            auto& context = *static_cast<Read*>(raw);
            if (context.state->terminalQuarantine) return false;
            bool channelsReady =
                context.state->aggregateChannelCount == 7u;
            for (std::uint32_t index = 0u;
                 index < context.state->aggregateChannelCount; ++index) {
                channelsReady = channelsReady &&
                    context.state->publishedChannelValues[index] != nil &&
                    context.state->publishedChannelValidity[index] != nil;
            }
            const auto& aggregate = context.state->exactAggregate;
            const auto& packet = aggregate.sensor_packet;
            const auto& publication = aggregate.publication;
            bool channelsValid = packet.channel_count == 7u &&
                packet.channel_capacity == MRNX_MAX_SENSOR_CHANNELS_V2;
            std::uint32_t previousModality = 0u;
            for (std::uint32_t index = 0u;
                 channelsValid && index < packet.channel_count; ++index) {
                channelsValid =
                    aggregate.channels[index].modality > previousModality &&
                    metalrobo::metalNumanXExactCandidateChannelV2Valid(
                        aggregate.channels[index], aggregate.timing);
                previousModality = aggregate.channels[index].modality;
            }
            const bool publicFamilyValid =
                aggregate.publication_epoch != 0u &&
                aggregate.publication_epoch ==
                    metalrobo::numanx_bridge_v1::domainPublicationEpoch(
                        context.state->domain) &&
                channelsReady &&
                metalrobo::metalNumanXExactInboundAuthorityV2Valid(
                    aggregate.inbound_authority) &&
                metalrobo::metalNumanXExactCandidateTimingV2Valid(
                    aggregate.timing) &&
                channelsValid &&
                packet.abi_version == MRNX_EXACT_SENSOR_PACKET_ABI_V2 &&
                packet.struct_size == sizeof(packet) &&
                packet.clock_domain ==
                    MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS &&
                packet.clock_quantum_nanoseconds ==
                    MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS &&
                packet.transaction_fingerprint ==
                    aggregate.root.transaction_fingerprint &&
                packet.transaction_fingerprint ==
                    aggregate.inbound_authority.transaction_fingerprint &&
                packet.substep_fingerprint ==
                    aggregate.inbound_authority.substep_fingerprint &&
                packet.inbound_authority_fingerprint ==
                    aggregate.inbound_authority.
                        inbound_authority_fingerprint &&
                packet.sensor_generation == aggregate.sensor_generation &&
                packet.accepted_brain_generation ==
                    aggregate.brain_generation &&
                packet.device_registry_id ==
                    context.state->device.registryID &&
                packet.timing_fingerprint ==
                    aggregate.timing.timing_fingerprint &&
                packet.channel_set_fingerprint ==
                    metalrobo::
                        metalNumanXExactCandidateChannelSetV2Fingerprint(
                            aggregate.channels, packet.channel_count) &&
                packet.candidate_publication_fingerprint != 0u &&
                packet.candidate_publication_fingerprint ==
                    metalrobo::metalNumanXExactSensorPacketV2Fingerprint(
                        packet) &&
                publication.abi_version == MRNX_PUBLICATION_ABI_V2 &&
                publication.struct_size == sizeof(publication) &&
                publication.clock_domain == packet.clock_domain &&
                publication.clock_quantum_nanoseconds ==
                    packet.clock_quantum_nanoseconds &&
                publication.transaction_fingerprint ==
                    packet.transaction_fingerprint &&
                publication.accepted_physics_token_fingerprint ==
                    packet.accepted_physics_token_fingerprint &&
                publication.candidate_publication_fingerprint ==
                    packet.candidate_publication_fingerprint &&
                publication.joint_commit_fingerprint != 0u &&
                publication.brain_generation == aggregate.brain_generation &&
                publication.committed_timestamp_nanoseconds ==
                    aggregate.timing.delivery_timestamp_nanoseconds &&
                publication.publication_fingerprint != 0u &&
                publication.publication_fingerprint ==
                    metalrobo::metalNumanXExactPublicationV2Fingerprint(
                        publication) &&
                aggregate.sensor.candidate_publication_fingerprint ==
                    packet.candidate_publication_fingerprint &&
                aggregate.sensor.key.sensor_generation ==
                    aggregate.sensor_generation;
            if (!publicFamilyValid) return false;
            if (context.state->culture != nullptr) {
                if (aggregate.culture.culture_fingerprint == 0u ||
                    aggregate.culture.generation == 0u ||
                    aggregate.culture.source_root_fingerprint !=
                        aggregate.root.transaction_fingerprint ||
                    aggregate.culture.receipt_fingerprint == 0u ||
                    !context.state->publishedCultureView.valid()) return false;
            } else if (aggregate.culture.culture_fingerprint != 0u) {
                return false;
            }
            *context.output = aggregate;
            return true;
        });
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
        __unsafe_unretained id<MTLBuffer> rootTranslation = nil;
        __unsafe_unretained id<MTLBuffer> bodyPositionLow = nil;
        __unsafe_unretained id<MTLBuffer> pointPositionLow = nil;
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
            pass.rootTranslationElementCount == 1u &&
            pass.bodyPositionLowElementCount >= pass.bodyPoseElementCount &&
            pass.pointPositionLowElementCount >= pass.pointWorldElementCount &&
            pass.standStatusStride == 1u &&
            pass.standStatusElementCount >= 1u &&
            pass.timestepSeconds ==
                runtime.visionProfile.depthAndTimestep.w &&
            pass.programFingerprint != 0u &&
            commandBufferObject(pass.commandBuffer, commandBuffer) &&
            bufferObject(pass.q, q) && bufferObject(pass.v, v) &&
            bufferObject(pass.bodyPoses, bodyPoses) &&
            bufferObject(pass.pointWorld, pointWorld) &&
            bufferObject(pass.rootTranslation, rootTranslation) &&
            bufferObject(pass.bodyPositionLow, bodyPositionLow) &&
            bufferObject(pass.pointPositionLow, pointPositionLow) &&
            rootTranslation.gpuAddress == pass.rootTranslationGPUAddress &&
            bodyPositionLow.gpuAddress == pass.bodyPositionLowGPUAddress &&
            pointPositionLow.gpuAddress == pass.pointPositionLowGPUAddress &&
            bufferObject(pass.standStatuses, standStatuses) &&
            q.device == runtime.device && v.device == runtime.device &&
            bodyPoses.device == runtime.device &&
            pointWorld.device == runtime.device &&
            standStatuses.device == runtime.device &&
            supportConsequences != nil &&
            supportConsequences.device == runtime.device &&
            supportView.gpuAddress == supportConsequences.gpuAddress &&
            supportView.elementCount == runtime.assets.supportContacts.size() &&
            supportView.stride == runtime.assets.supportContacts.size() &&
            active->supportConsequencesGPUAddress == active->supplementalDispatch.expectedSupportConsequencesGPUAddress &&
            runtime.supportAggregationPipeline != nil && runtime.touchSupportMapping != nil &&
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
            {rootTranslation, pass.rootTranslationGPUAddress, sizeof(MRCompensatedRootTranslationGPU)},
            {bodyPositionLow, pass.bodyPositionLowGPUAddress,
             runtime.assets.rigid.engineBodyCount * sizeof(mr_float4)},
            {pointPositionLow, pass.pointPositionLowGPUAddress,
             runtime.assets.points.size() * sizeof(mr_float4)},
            {standStatuses, standStatuses.gpuAddress,
             sizeof(MRNumiHumanStandStatusGPU)},
            {supportConsequences, supportView.gpuAddress,
             supportView.elementCount *
                sizeof(MRNumanXHumanSupportConsequenceGPU)},
            {active->supportConsequences, active->supportConsequencesGPUAddress, active->supportConsequences.length},
            {runtime.touchSupportMapping, runtime.touchSupportMapping.gpuAddress, runtime.touchSupportMapping.length},
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
        if (active->rootTranslationTrace != nil) {
            if (active->rootTranslationTrace.device != runtime.device ||
                active->rootTranslationTrace.length != sizeof(MRCompensatedRootTranslationGPU)) return false;
            for (const auto& region : regions) {
                if (active->rootTranslationTrace == region.buffer ||
                    !disjoint(active->rootTranslationTrace.gpuAddress, active->rootTranslationTrace.length,
                        region.address, region.bytes)) return false;
            }
            id<MTLBlitCommandEncoder> copy = [commandBuffer blitCommandEncoder];
            if (copy == nil) return false;
            [copy copyFromBuffer:rootTranslation sourceOffset:0u
                toBuffer:active->rootTranslationTrace destinationOffset:0u
                size:sizeof(MRCompensatedRootTranslationGPU)];
            [copy endEncoding];
        }
        if (active->ownerSnapshotCapture) {
            auto& capture = *active->ownerSnapshotCapture;
            __unsafe_unretained id<MTLBuffer> mujocoStates = nil;
            __unsafe_unretained id<MTLBuffer> mujocoResults = nil;
            __unsafe_unretained id<MTLBuffer> forceArena = nil;
            const std::uint64_t muscleCount =
                runtime.assets.muscle.muscleCount;
            const std::uint64_t dofCount = runtime.assets.rigid.nv;
            if (dofCount != 0u && muscleCount >
                    std::numeric_limits<std::uint64_t>::max() / dofCount) {
                return false;
            }
            const std::uint64_t perMuscleForceCount =
                muscleCount * dofCount;
            const bool captureValid = !capture.postDynamicsEncoded &&
                capture.buffer != nil &&
                capture.buffer.device == runtime.device &&
                capture.buffer.contents != nullptr &&
                capture.layout.totalBytes == capture.buffer.length &&
                pass.mujocoMuscleCount == muscleCount &&
                pass.mujocoStateElementCount == muscleCount &&
                pass.mujocoStateStride == muscleCount &&
                pass.mujocoResultElementCount == muscleCount &&
                pass.mujocoResultStride == muscleCount &&
                pass.mujocoMuscleGeneralizedForceElementCount ==
                    perMuscleForceCount &&
                pass.mujocoMuscleGeneralizedForceRowStride == dofCount &&
                pass.mujocoMuscleGeneralizedForceEnvironmentStride ==
                    perMuscleForceCount &&
                pass.mujocoGeneralizedForceElementCount == dofCount &&
                pass.mujocoGeneralizedForceStride == dofCount &&
                pass.mujocoGeneralizedForceOffset == perMuscleForceCount &&
                pass.mujocoGeneralizedForceOffset <=
                    pass.mujocoGeneralizedForceArenaElementCount &&
                dofCount <= pass.mujocoGeneralizedForceArenaElementCount -
                    pass.mujocoGeneralizedForceOffset &&
                pass.tendonBindingCount == 0u &&
                pass.tendonEnvelopeCount == 0u &&
                pass.tendonTransferElementCount == 0u &&
                pass.tendonCorrectionElementCount == 0u &&
                bufferObject(pass.mujocoStates, mujocoStates) &&
                bufferObject(pass.mujocoResults, mujocoResults) &&
                bufferObject(pass.mujocoGeneralizedForceArena, forceArena) &&
                mujocoStates.device == runtime.device &&
                mujocoResults.device == runtime.device &&
                forceArena.device == runtime.device &&
                mujocoStates != capture.buffer &&
                mujocoResults != capture.buffer &&
                forceArena != capture.buffer;
            if (!captureValid) return false;
            struct Copy {
                __unsafe_unretained id<MTLBuffer> source = nil;
                std::uint64_t sourceOffset = 0u;
                std::uint64_t destinationOffset = 0u;
                std::uint64_t bytes = 0u;
            };
            const Copy copies[] = {
                {mujocoStates, 0u, capture.layout.candidateMuscles,
                    muscleCount * sizeof(MRMujocoMuscleStateGPU)},
                {mujocoResults, 0u, capture.layout.muscleResults,
                    muscleCount * sizeof(MRMujocoMuscleResultGPU)},
                {forceArena, 0u, capture.layout.muscleGeneralizedForces,
                    perMuscleForceCount * sizeof(float)},
                {standStatuses, 0u, capture.layout.standStatus,
                    sizeof(MRNumiHumanStandStatusGPU)},
                {supportConsequences, 0u,
                    capture.layout.candidateSupportConsequences,
                    supportView.elementCount *
                        sizeof(NMHumanSupportConsequenceGPU)},
            };
            for (const auto& copy : copies) {
                if (copy.sourceOffset > copy.source.length ||
                    copy.bytes > copy.source.length - copy.sourceOffset ||
                    copy.destinationOffset > capture.buffer.length ||
                    copy.bytes > capture.buffer.length -
                        copy.destinationOffset) return false;
            }
            id<MTLBlitCommandEncoder> copy =
                [commandBuffer blitCommandEncoder];
            if (copy == nil) return false;
            copy.label = @"NumanX production-owner MyoSim snapshot";
            for (const auto& region : copies) {
                [copy copyFromBuffer:region.source
                    sourceOffset:static_cast<NSUInteger>(region.sourceOffset)
                    toBuffer:capture.buffer
                    destinationOffset:static_cast<NSUInteger>(
                        region.destinationOffset)
                    size:static_cast<NSUInteger>(region.bytes)];
            }
            [copy endEncoding];
            capture.postDynamicsEncoded = true;
        }
        id<MTLComputeCommandEncoder> reduction = [commandBuffer computeCommandEncoder];
        if (reduction == nil) return false;
        [reduction setComputePipelineState:runtime.supportAggregationPipeline];
        [reduction setBuffer:supportConsequences offset:0u atIndex:0u];
        [reduction setBuffer:runtime.touchSupportMapping offset:0u atIndex:1u];
        [reduction setBuffer:active->supportConsequences offset:0u atIndex:2u];
        const mr_uint4 mappingDispatch{static_cast<std::uint32_t>(supportView.elementCount),10u,0u,0u};
        [reduction setBytes:&mappingDispatch length:sizeof(mappingDispatch) atIndex:3u];
        [reduction dispatchThreads:MTLSizeMake(10u,1u,1u) threadsPerThreadgroup:MTLSizeMake(10u,1u,1u)];
        [reduction endEncoding];
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
        [encoder setBuffer:active->supportConsequences offset:0u atIndex:18u];
        [encoder setBuffer:bodyPositionLow offset:0u atIndex:19u];
        [encoder setBuffer:pointPositionLow offset:0u atIndex:20u];
        [encoder setBuffer:rootTranslation offset:0u atIndex:21u];
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

bool encodeRuntimeBehaviorCandidate(void* raw,
    const metalrobo::MetalNumanXHumanMatterPass& pass) noexcept {
    auto* runtime = static_cast<RuntimeState*>(raw);
    if (runtime == nullptr) return false;
    const auto encodeOwnerSnapshot = [&]() noexcept {
        if (pass.phase ==
            metalrobo::MetalNumanXHumanMatterPhase::beginStep) return true;
        auto* active = runtime->encodingActive;
        if (active == nullptr || !active->ownerSnapshotCapture) return true;
        auto& capture = *active->ownerSnapshotCapture;
        const bool preDynamics = pass.phase ==
            metalrobo::MetalNumanXHumanMatterPhase::preDynamics;
        const bool postDynamics = pass.phase ==
            metalrobo::MetalNumanXHumanMatterPhase::postDynamics;
        if ((!preDynamics && !postDynamics) ||
            (preDynamics && capture.preDynamicsEncoded) ||
            (postDynamics && (!capture.preDynamicsEncoded ||
                capture.humanMatterPostDynamicsEncoded)) ||
            capture.buffer == nil ||
            capture.buffer.device != runtime->device ||
            capture.buffer.contents == nullptr ||
            capture.layout.totalBytes != capture.buffer.length ||
            active->slotGeneration != pass.slotGeneration ||
            active->transactionFingerprint != pass.transactionFingerprint ||
            pass.abiVersion !=
                metalrobo::kMetalNumanXHumanMatterPassABIVersion ||
            pass.structSize != sizeof(pass) || pass.environmentCount != 1u ||
            pass.qCoordinateCount != runtime->assets.rigid.nq ||
            pass.dofCount != runtime->assets.rigid.nv ||
            pass.mujocoStateCount != runtime->assets.muscle.muscleCount ||
            pass.qStride < pass.qCoordinateCount ||
            pass.vStride < pass.dofCount ||
            pass.mujocoStateStride < pass.mujocoStateCount ||
            pass.generalizedForceStride < pass.dofCount ||
            pass.generalizedForceOffset >
                pass.generalizedForceArenaElementCount ||
            pass.generalizedForceOffset >
                std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
            pass.dofCount > pass.generalizedForceArenaElementCount -
                pass.generalizedForceOffset ||
            pass.reactionStride < pass.dofCount ||
            pass.rootTranslationElementCount < 1u ||
            pass.rootTranslationCheckpointElementCount < 1u) return false;
        if (pass.dofCount != 0u && pass.dofCount >
                std::numeric_limits<std::uint64_t>::max() /
                    pass.dofCount) return false;
        const std::uint64_t factorElements =
            pass.dofCount * pass.dofCount;
        if (pass.factorStride < factorElements ||
            pass.qCoordinateCount >
                std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
            pass.dofCount >
                std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
            pass.mujocoStateCount >
                std::numeric_limits<std::uint64_t>::max() /
                    sizeof(MRMujocoMuscleStateGPU) ||
            factorElements >
                std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
            return false;
        }
        if (postDynamics &&
            (capture.ownerProgramFingerprint != pass.programFingerprint ||
             capture.linearizationEpoch != pass.linearizationEpoch)) {
            return false;
        }

        __unsafe_unretained id<MTLCommandBuffer> commandBuffer = nil;
        if (!commandBufferObject(pass.commandBuffer, commandBuffer))
            return false;
        struct Copy {
            void* raw = nullptr;
            std::uint64_t sourceOffset = 0u;
            std::uint64_t destinationOffset = 0u;
            std::uint64_t bytes = 0u;
        };
        const std::uint64_t nqBytes =
            pass.qCoordinateCount * sizeof(float);
        const std::uint64_t nvBytes = pass.dofCount * sizeof(float);
        const std::uint64_t muscleBytes = pass.mujocoStateCount *
            sizeof(MRMujocoMuscleStateGPU);
        const std::uint64_t factorBytes = factorElements * sizeof(float);
        const Copy preDynamicsCopies[] = {
            {pass.qCheckpoint, 0u, capture.layout.checkpointQ, nqBytes},
            {pass.vCheckpoint, 0u, capture.layout.checkpointV, nvBytes},
            {pass.rootTranslationCheckpoint, 0u,
                capture.layout.checkpointRoot,
                sizeof(MRCompensatedRootTranslationGPU)},
            {pass.mujocoStateCheckpoint, 0u,
                capture.layout.checkpointMuscles, muscleBytes},
            {pass.sourceEffectiveTangentFactor, 0u,
                capture.layout.effectiveTangentFactorStorage, factorBytes},
            {pass.mujocoGeneralizedForceArena,
                pass.generalizedForceOffset * sizeof(float),
                capture.layout.sourceGeneralizedForce, nvBytes},
            {pass.mujocoGeneralizedForceArena,
                pass.generalizedForceOffset * sizeof(float),
                capture.layout.reducedMuscleGeneralizedForce, nvBytes},
            {pass.sourcePredictedVelocity, 0u,
                capture.layout.sourcePredictedVelocity, nvBytes},
            {pass.matterGeneralizedReaction, 0u,
                capture.layout.matterGeneralizedReaction, nvBytes},
        };
        const Copy postDynamicsCopies[] = {
            {pass.ownerStatuses, 0u, capture.layout.ownerStatus,
                sizeof(MRNumanXHumanMatterOwnerStatusGPU)},
            {pass.q, 0u, capture.layout.candidateQ, nqBytes},
            {pass.v, 0u, capture.layout.candidateV, nvBytes},
            {pass.rootTranslation, 0u, capture.layout.candidateRoot,
                sizeof(MRCompensatedRootTranslationGPU)},
        };
        const auto encodeCopies = [&] (
            const std::span<const Copy> copies,
            NSString* label) noexcept {
            if (copies.size() > std::size(preDynamicsCopies)) return false;
            struct BorrowedSource {
                __unsafe_unretained id<MTLBuffer> buffer = nil;
            };
            std::array<BorrowedSource,
                std::size(preDynamicsCopies)> sources{};
            for (std::size_t index = 0u; index < copies.size(); ++index) {
                __unsafe_unretained id<MTLBuffer> source = nil;
                const auto& copy = copies[index];
                if (!bufferObject(copy.raw, source) ||
                    source == capture.buffer ||
                    source.device != runtime->device ||
                    copy.sourceOffset > source.length ||
                    copy.bytes > source.length - copy.sourceOffset ||
                    copy.destinationOffset > capture.buffer.length ||
                    copy.bytes > capture.buffer.length -
                        copy.destinationOffset) return false;
                sources[index].buffer = source;
            }
            id<MTLBlitCommandEncoder> encoder =
                [commandBuffer blitCommandEncoder];
            if (encoder == nil) return false;
            encoder.label = label;
            for (std::size_t index = 0u; index < copies.size(); ++index) {
                const auto& copy = copies[index];
                [encoder copyFromBuffer:sources[index].buffer
                    sourceOffset:static_cast<NSUInteger>(copy.sourceOffset)
                    toBuffer:capture.buffer
                    destinationOffset:static_cast<NSUInteger>(
                        copy.destinationOffset)
                    size:static_cast<NSUInteger>(copy.bytes)];
            }
            [encoder endEncoding];
            return true;
        };
        if (preDynamics) {
            if (!encodeCopies(preDynamicsCopies,
                    @"NumanX production-owner source snapshot")) {
                return false;
            }
            capture.ownerProgramFingerprint = pass.programFingerprint;
            capture.linearizationEpoch = pass.linearizationEpoch;
            capture.preDynamicsEncoded = true;
        } else {
            if (!encodeCopies(postDynamicsCopies,
                    @"NumanX production-owner candidate snapshot")) {
                return false;
            }
            capture.humanMatterPostDynamicsEncoded = true;
        }
        return true;
    };
    if (!encodeOwnerSnapshot()) return false;
    if (runtime->behavior == nullptr) return true;
    if (pass.phase == metalrobo::MetalNumanXHumanMatterPhase::beginStep) {
        return runtime->behavior->encodeFlush(
            pass.commandBuffer, runtime->behaviorError);
    }
    if (pass.phase == metalrobo::MetalNumanXHumanMatterPhase::preDynamics) {
        if (!runtime->behavior->initialObservationEncoded())
            return runtime->behavior->encodeInitial(pass, runtime->behaviorError);
        return true;
    }
    if (pass.phase != metalrobo::MetalNumanXHumanMatterPhase::postDynamics)
        return false;
    const auto* active = runtime->encodingActive;
    if (active == nullptr || active->slotGeneration != pass.slotGeneration ||
        active->transactionFingerprint != pass.transactionFingerprint ||
        (!active->exactFamily &&
         active->acceptedTimestampMicroseconds >
             std::numeric_limits<std::uint64_t>::max() /
                 runtime->clockQuantumNanoseconds))
        return false;
    const std::uint64_t acceptedTimestampNanoseconds = active->exactFamily
        ? active->acceptedTimestampNanoseconds
        : active->acceptedTimestampMicroseconds *
            runtime->clockQuantumNanoseconds;
    return runtime->behavior->encodeCandidate(pass, active->physicsGeneration,
        acceptedTimestampNanoseconds, runtime->behaviorError);
}

void recordRuntimeBehaviorTerminal(RuntimeState& runtime, const ActiveRoot& active,
    const mrnx_root_v1& root, bool accepted,
    const MRNumanXHumanMatterJointPublicationFenceGPU* fence) noexcept {
    if (runtime.behavior == nullptr) return;
    if ((!active.exactFamily &&
         active.acceptedTimestampMicroseconds >
             std::numeric_limits<std::uint64_t>::max() /
                 runtime.clockQuantumNanoseconds) ||
        runtime.behavior->completedAttempts() == std::numeric_limits<std::uint64_t>::max()) {
        runtime.behaviorError = "behavior clock or attempt count overflow";
        return;
    }
    MRHumanBehaviorReleaseGPU release{};
    release.programFingerprint = runtime.behavior->fingerprint();
    release.transactionFingerprint = root.transaction_fingerprint;
    release.linearizationEpoch = root.linearization_epoch;
    release.slotGeneration = root.slot_generation;
    release.physicsGeneration = active.physicsGeneration;
    release.acceptedTimestampNanoseconds = active.exactFamily
        ? active.acceptedTimestampNanoseconds
        : active.acceptedTimestampMicroseconds *
            runtime.clockQuantumNanoseconds;
    release.publicationSerial = runtime.behavior->completedAttempts() + 1u;
    release.jointFenceFingerprint = fence != nullptr ? fence->fenceFingerprint : 0u;
    release.released = accepted ? 1u : 2u;
    (void)runtime.behavior->terminal(release, fence, runtime.behaviorError);
}

template <typename T>
[[nodiscard]] metalrobo::NumiHumanProductionOwnerArrayV1 ownerHostArray(
    const std::span<const T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    metalrobo::NumiHumanProductionOwnerArrayV1 result;
    result.available = true;
    result.expectedElementCount = values.size();
    result.elementBytes = sizeof(T);
    const auto bytes = std::as_bytes(values);
    result.bytes.assign(bytes.begin(), bytes.end());
    return result;
}

[[nodiscard]] bool ownerCapturedArray(
    const OwnerSnapshotCapture& capture,
    const std::uint64_t offset,
    const std::uint64_t count,
    const std::uint32_t elementBytes,
    metalrobo::NumiHumanProductionOwnerArrayV1& output
) {
    if (capture.buffer == nil || capture.buffer.contents == nullptr ||
        elementBytes == 0u || count >
            std::numeric_limits<std::uint64_t>::max() / elementBytes) {
        return false;
    }
    const std::uint64_t bytes = count * elementBytes;
    if (offset > capture.buffer.length ||
        bytes > capture.buffer.length - offset ||
        bytes > std::numeric_limits<std::size_t>::max()) return false;
    metalrobo::NumiHumanProductionOwnerArrayV1 result;
    result.available = true;
    result.expectedElementCount = count;
    result.elementBytes = elementBytes;
    const auto* begin = static_cast<const std::byte*>(
        capture.buffer.contents) + offset;
    result.bytes.assign(begin, begin + static_cast<std::size_t>(bytes));
    output = std::move(result);
    return true;
}

[[nodiscard]] bool ownerFloatValues(
    const metalrobo::NumiHumanProductionOwnerArrayV1& source,
    const std::size_t expectedCount,
    std::vector<float>& output
) {
    if (!source.available || source.elementBytes != sizeof(float) ||
        source.expectedElementCount != expectedCount ||
        expectedCount > std::numeric_limits<std::size_t>::max() /
            sizeof(float) ||
        source.bytes.size() != expectedCount * sizeof(float)) {
        return false;
    }
    output.resize(expectedCount);
    if (!output.empty()) {
        std::memcpy(output.data(), source.bytes.data(), source.bytes.size());
    }
    return true;
}

[[nodiscard]] bool deriveOwnerSnapshotDynamics(
    metalrobo::NumiHumanProductionOwnerSnapshotV1& snapshot,
    std::string& error
) {
    const std::size_t nv = snapshot.dofCount;
    std::vector<float> v0;
    std::vector<float> freeVelocity;
    std::vector<float> sourceForce;
    std::vector<float> reaction;
    std::vector<float> candidateVelocity;
    std::vector<float> lower;
    if (!ownerFloatValues(snapshot.checkpointV, nv, v0) ||
        !ownerFloatValues(
            snapshot.sourcePredictedVelocity, nv, freeVelocity) ||
        !ownerFloatValues(
            snapshot.sourceGeneralizedForce, nv, sourceForce) ||
        !ownerFloatValues(snapshot.matterGeneralizedReaction, nv, reaction) ||
        !ownerFloatValues(snapshot.candidateV, nv, candidateVelocity) ||
        !ownerFloatValues(snapshot.effectiveTangentFactorStorage,
            nv * nv, lower) ||
        snapshot.timestepNanoseconds == 0u) {
        error = "production-owner dynamic derivation inputs are incomplete";
        return false;
    }
    const double timestep =
        static_cast<double>(snapshot.timestepNanoseconds) * 1.0e-9;
    std::vector<float> acceleration(nv);
    std::vector<double> transposeAction(nv, 0.0);
    std::vector<float> rhs(nv);
    std::vector<float> bias(nv);
    std::vector<double> candidateDelta(nv, 0.0);
    for (std::size_t row = 0u; row < nv; ++row) {
        if (!std::isfinite(v0[row]) || !std::isfinite(freeVelocity[row]) ||
            !std::isfinite(sourceForce[row]) ||
            !std::isfinite(reaction[row]) ||
            !std::isfinite(candidateVelocity[row])) {
            error = "production-owner dynamic vector is nonfinite";
            return false;
        }
        const double value =
            (static_cast<double>(freeVelocity[row]) - v0[row]) / timestep;
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<float>::max()) {
            error = "production-owner acceleration is not FP32 representable";
            return false;
        }
        acceleration[row] = static_cast<float>(value);
        candidateDelta[row] =
            static_cast<double>(candidateVelocity[row]) - v0[row];
    }
    // A0 = L L^T. Preserve the exact device factor bytes in the record and
    // derive the missing RHS/bias in host FP64 before emitting FP32 witnesses.
    for (std::size_t column = 0u; column < nv; ++column) {
        double value = 0.0;
        for (std::size_t row = column; row < nv; ++row) {
            const float coefficient = lower[row * nv + column];
            if (!std::isfinite(coefficient)) {
                error = "production-owner effective tangent is nonfinite";
                return false;
            }
            value += static_cast<double>(coefficient) * acceleration[row];
        }
        transposeAction[column] = value;
    }
    for (std::size_t row = 0u; row < nv; ++row) {
        double value = 0.0;
        for (std::size_t column = 0u; column <= row; ++column) {
            value += static_cast<double>(lower[row * nv + column]) *
                transposeAction[column];
        }
        const double biasValue = static_cast<double>(sourceForce[row]) - value;
        if (!std::isfinite(value) || !std::isfinite(biasValue) ||
            std::abs(value) > std::numeric_limits<float>::max() ||
            std::abs(biasValue) > std::numeric_limits<float>::max()) {
            error = "production-owner RHS/bias is not FP32 representable";
            return false;
        }
        rhs[row] = static_cast<float>(value);
        bias[row] = static_cast<float>(biasValue);
    }
    double sourceWork = 0.0;
    double reactionWork = 0.0;
    for (std::size_t index = 0u; index < nv; ++index) {
        sourceWork += 0.5 * timestep *
            (static_cast<double>(v0[index]) + freeVelocity[index]) *
            sourceForce[index];
        reactionWork += 0.5 * timestep *
            (static_cast<double>(freeVelocity[index]) +
             candidateVelocity[index]) * reaction[index];
    }
    std::vector<double> deltaTranspose(nv, 0.0);
    for (std::size_t column = 0u; column < nv; ++column) {
        for (std::size_t row = column; row < nv; ++row) {
            deltaTranspose[column] +=
                static_cast<double>(lower[row * nv + column]) *
                candidateDelta[row];
        }
    }
    double effectiveEnergy = 0.0;
    for (const double value : deltaTranspose)
        effectiveEnergy += 0.5 * value * value;
    if (!std::isfinite(sourceWork) || !std::isfinite(reactionWork) ||
        !std::isfinite(effectiveEnergy) ||
        std::abs(sourceWork) > std::numeric_limits<float>::max() ||
        std::abs(reactionWork) > std::numeric_limits<float>::max() ||
        effectiveEnergy > std::numeric_limits<float>::max()) {
        error = "production-owner work component is not FP32 representable";
        return false;
    }
    const std::array<float, 3u> work{
        static_cast<float>(sourceWork), static_cast<float>(reactionWork),
        static_cast<float>(effectiveEnergy)};
    snapshot.acceleration = ownerHostArray<float>(acceleration);
    snapshot.sourceRHS = ownerHostArray<float>(rhs);
    snapshot.sourceBias = ownerHostArray<float>(bias);
    snapshot.workEnergyComponents = ownerHostArray<float>(work);
    return true;
}

[[nodiscard]] float ownerScalarImpedance(
    const nm_float4 solimp0,
    const nm_float4 solimp1,
    const float phi
) noexcept {
    const float d0 = std::clamp(solimp0.x, 0.0001f, 0.9999f);
    const float dw = std::clamp(solimp0.y, 0.0001f, 0.9999f);
    const float width = std::max(solimp0.z, 0.0f);
    const float midpoint = std::clamp(solimp0.w, 0.0001f, 0.9999f);
    const float power = std::max(solimp1.x, 1.0f);
    if (d0 == dw || width <= 1.0e-15f) return 0.5f * (d0 + dw);
    const float x = std::clamp(std::abs(phi) / width, 0.0f, 1.0f);
    const float y = power == 1.0f ? x :
        (x <= midpoint
            ? std::pow(x, power) /
                std::pow(midpoint, power - 1.0f)
            : 1.0f - std::pow(1.0f - x, power) /
                std::pow(1.0f - midpoint, power - 1.0f));
    return d0 + y * (dw - d0);
}

[[nodiscard]] std::array<float, 2u> ownerScalarStiffnessDamping(
    const nm_float4 solref,
    const float dw,
    const std::uint32_t flags,
    const float timestep
) noexcept {
    const bool positive = solref.x > 0.0f;
    const float timeConstant =
        (flags & NM_HUMAN_EQUALITY_REFSAFE) != 0u
        ? std::max(solref.x, 2.0f * timestep) : solref.x;
    const float stiffness = positive
        ? 1.0f / std::max(1.0e-15f,
            dw * dw * timeConstant * timeConstant * solref.y * solref.y)
        : -solref.x / std::max(1.0e-15f, dw * dw);
    const float damping = positive
        ? 2.0f / std::max(1.0e-15f, dw * timeConstant)
        : -solref.y / std::max(1.0e-15f, dw);
    return {stiffness, damping};
}

[[nodiscard]] bool deriveOwnerConstraintWitnesses(
    const RuntimeState& runtime,
    metalrobo::NumiHumanProductionOwnerSnapshotV1& snapshot,
    std::string& error
) {
    using Witness = std::array<float, 8u>;
    const std::size_t nq = snapshot.qCoordinateCount;
    const std::size_t nv = snapshot.dofCount;
    std::vector<float> q0;
    std::vector<float> v0;
    std::vector<float> vFree;
    std::vector<float> candidateV;
    if (!ownerFloatValues(snapshot.checkpointQ, nq, q0) ||
        !ownerFloatValues(snapshot.checkpointV, nv, v0) ||
        !ownerFloatValues(snapshot.sourcePredictedVelocity, nv, vFree) ||
        !ownerFloatValues(snapshot.candidateV, nv, candidateV) ||
        snapshot.timestepNanoseconds == 0u) {
        error = "production-owner constraint reconstruction inputs are incomplete";
        return false;
    }
    const float timestep = static_cast<float>(
        static_cast<double>(snapshot.timestepNanoseconds) * 1.0e-9);
    std::vector<float> deltaVelocity(nv);
    for (std::size_t index = 0u; index < nv; ++index)
        deltaVelocity[index] = candidateV[index] - vFree[index];

    std::vector<Witness> equalities;
    equalities.reserve(runtime.assets.jointEqualities.size());
    for (const auto& row : runtime.assets.jointEqualities) {
        const bool fixed = row.indices.z == NM_INVALID_INDEX;
        if (row.indices.x >= nq || row.indices.y >= nv ||
            (!fixed && (row.indices.z >= nq || row.indices.w >= nv))) {
            error = "production-owner equality row escapes captured state";
            return false;
        }
        const float x = fixed ? 0.0f :
            q0[row.indices.z] - row.referencesAndCoefficients0.y;
        const float a0 = row.referencesAndCoefficients0.z;
        const float a1 = row.referencesAndCoefficients0.w;
        const float a2 = row.coefficients1.x;
        const float a3 = row.coefficients1.y;
        const float a4 = row.coefficients1.z;
        const float polynomial =
            (((a4 * x + a3) * x + a2) * x + a1) * x + a0;
        const float derivative = fixed ? 0.0f :
            ((4.0f * a4 * x + 3.0f * a3) * x + 2.0f * a2) * x + a1;
        const float phi = q0[row.indices.x] -
            row.referencesAndCoefficients0.x - polynomial;
        const float sourceVelocity = v0[row.indices.y] -
            (fixed ? 0.0f : derivative * v0[row.indices.w]);
        const float freeIncrement =
            vFree[row.indices.y] - v0[row.indices.y] -
            (fixed ? 0.0f : derivative *
                (vFree[row.indices.w] - v0[row.indices.w]));
        const float dw = std::clamp(
            row.solimp0.y, 0.0001f, 0.9999f);
        const float impedance = ownerScalarImpedance(
            row.solimp0, row.solimp1, phi);
        const auto kb = ownerScalarStiffnessDamping(
            row.solref, dw, runtime.assets.equalityDispatch.flags,
            timestep);
        const float referenceAcceleration =
            -kb[1] * sourceVelocity - kb[0] * impedance * phi;
        const float regularizer = std::max(1.0e-15f,
            (1.0f - impedance) / impedance *
            (row.sourceInverseWeights.x + row.sourceInverseWeights.y));
        const float bDelta = timestep * referenceAcceleration - freeIncrement;
        const float inverseRegularizer = 1.0f / regularizer;
        const float rowDeltaVelocity = deltaVelocity[row.indices.y] -
            (fixed ? 0.0f : derivative * deltaVelocity[row.indices.w]);
        const float impulse =
            (rowDeltaVelocity - bDelta) * inverseRegularizer;
        const Witness witness{derivative, bDelta, inverseRegularizer, phi,
            impulse, rowDeltaVelocity, 1.0f, 0.0f};
        if (!std::all_of(witness.begin(), witness.end(),
                [](const float value) { return std::isfinite(value); })) {
            error = "production-owner equality witness is nonfinite";
            return false;
        }
        equalities.push_back(witness);
    }

    std::vector<Witness> limits;
    limits.reserve(2u * runtime.assets.jointLimits.size());
    for (const auto& row : runtime.assets.jointLimits) {
        if (row.indices.x >= nq || row.indices.y >= nv) {
            error = "production-owner limit row escapes captured state";
            return false;
        }
        for (std::uint32_t side = 0u; side < 2u; ++side) {
            const bool lower = side == 0u;
            const float position = q0[row.indices.x];
            const float direction = lower ? 1.0f : -1.0f;
            const float distance = lower
                ? position - row.rangeMarginInverseWeight.x
                : row.rangeMarginInverseWeight.y - position;
            const float margin = row.rangeMarginInverseWeight.z;
            if (!std::isfinite(distance)) {
                error = "production-owner limit distance is nonfinite";
                return false;
            }
            Witness witness{};
            if (distance < margin) {
                const float phi = distance - margin;
                const float sourceVelocity = direction * v0[row.indices.y];
                const float freeIncrement = direction *
                    (vFree[row.indices.y] - v0[row.indices.y]);
                const float dw = std::clamp(
                    row.solimp0.y, 0.0001f, 0.9999f);
                const float impedance = ownerScalarImpedance(
                    row.solimp0, row.solimp1, phi);
                const auto kb = ownerScalarStiffnessDamping(
                    row.solref, dw, runtime.assets.limitDispatch.flags,
                    timestep);
                const float referenceAcceleration =
                    -kb[1] * sourceVelocity - kb[0] * impedance * phi;
                const float regularizer = std::max(1.0e-15f,
                    (1.0f - impedance) / impedance *
                    row.rangeMarginInverseWeight.w);
                const float bDelta =
                    timestep * referenceAcceleration - freeIncrement;
                const float inverseRegularizer = 1.0f / regularizer;
                const float violation = direction *
                    deltaVelocity[row.indices.y] - bDelta;
                const float impulse =
                    std::min(0.0f, violation) * inverseRegularizer;
                witness = {direction, bDelta, inverseRegularizer, phi,
                    impulse, violation, 1.0f, 0.0f};
            }
            if (!std::all_of(witness.begin(), witness.end(),
                    [](const float value) { return std::isfinite(value); })) {
                error = "production-owner limit witness is nonfinite";
                return false;
            }
            limits.push_back(witness);
        }
    }
    snapshot.equalityLinearizationImpulses =
        ownerHostArray<Witness>(equalities);
    snapshot.limitLinearizationImpulses = ownerHostArray<Witness>(limits);
    return true;
}

[[nodiscard]] bool writeOwnerSnapshotEvidence(
    RuntimeState& runtime,
    const ActiveRoot& active,
    const mrnx_root_v1& terminalRoot,
    const PreparedTerminalDisposition disposition,
    const std::uint64_t publicationEpoch,
    const MRNumanXHumanMatterJointPublicationFenceGPU* committedFence,
    std::string& error
) noexcept {
    try {
        if (!active.ownerSnapshotCapture ||
            !active.ownerSnapshotCapture->preDynamicsEncoded ||
            !active.ownerSnapshotCapture->humanMatterPostDynamicsEncoded ||
            !active.ownerSnapshotCapture->postDynamicsEncoded) {
            error = "production-owner GPU capture phases are incomplete";
            return false;
        }
        const auto& capture = *active.ownerSnapshotCapture;
        const auto matter = runtime.matter->snapshot();
        if (!matter.available) {
            error = "production-owner Matter terminal snapshot unavailable: " +
                matter.message;
            return false;
        }
        const bool published =
            disposition == PreparedTerminalDisposition::published;
        const bool rejected =
            disposition == PreparedTerminalDisposition::rejected;
        if (!published && !rejected) {
            error = "production-owner terminal disposition is not final";
            return false;
        }
        const bool terminalRootIdentityValid =
            terminalRoot.abi_version == MRNX_BRIDGE_ABI_V1 &&
            terminalRoot.struct_size == sizeof(terminalRoot) &&
            terminalRoot.owner_wire_abi_version == MRNX_OWNER_WIRE_ABI_V4 &&
            terminalRoot.environment_count == 1u &&
            terminalRoot.environment == 0u &&
            terminalRoot.transaction_slot == active.transactionSlot &&
            terminalRoot.step_index == 0u &&
            terminalRoot.control_step == active.controlStep &&
            terminalRoot.substep_index == 0u &&
            terminalRoot.physics_substep_count == 1u &&
            terminalRoot.q_coordinate_count == runtime.assets.rigid.nq &&
            terminalRoot.dof_count == runtime.assets.rigid.nv &&
            terminalRoot.dof_layout_version ==
                metalrobo::kMetalNumanXHumanMatterDofLayoutVersion &&
            terminalRoot.reserved0 == 0u &&
            terminalRoot.program_fingerprint ==
                capture.ownerProgramFingerprint &&
            terminalRoot.transaction_fingerprint ==
                active.transactionFingerprint &&
            terminalRoot.linearization_epoch == capture.linearizationEpoch &&
            terminalRoot.slot_generation == active.slotGeneration &&
            terminalRoot.device_registry_id == runtime.device.registryID;
        const bool publicationIdentityValid = published &&
            terminalRootIdentityValid &&
            committedFence != nullptr &&
            committedFence->fenceFingerprint != 0u &&
            committedFence->transactionFingerprint ==
                active.transactionFingerprint &&
            committedFence->linearizationEpoch ==
                capture.linearizationEpoch &&
            committedFence->slotGeneration == active.slotGeneration &&
            committedFence->controlStep == active.controlStep &&
            publicationEpoch != 0u &&
            matter.controlStep == active.controlStep;
        const bool rollbackIdentityValid = rejected &&
            terminalRootIdentityValid &&
            matter.controlStep == active.controlStep;
        if ((published && !publicationIdentityValid) ||
            (rejected && !rollbackIdentityValid)) {
            error = "production-owner terminal publication/rollback identity mismatch";
            return false;
        }

        metalrobo::NumiHumanProductionOwnerSnapshotV1 snapshot;
        snapshot.treatment = runtime.ownerSnapshotTreatment;
        snapshot.disposition = published
            ? metalrobo::NumiHumanProductionOwnerDispositionV1::published
            : metalrobo::NumiHumanProductionOwnerDispositionV1::rejected;
        snapshot.baseStateFingerprint =
            runtime.ownerSnapshotBaseStateFingerprint;
        snapshot.treatmentHistoryFingerprint =
            runtime.ownerSnapshotTreatmentHistoryFingerprint;
        snapshot.humanSourceFingerprint = runtime.assets.sourceFingerprint;
        snapshot.matterSourcePhysicsFingerprint =
            matter.sourcePhysicsFingerprint;
        snapshot.matterDeviceProgramFingerprint =
            matter.deviceProgramFingerprint;
        snapshot.ownerProgramFingerprint = capture.ownerProgramFingerprint;
        snapshot.transactionFingerprint = active.transactionFingerprint;
        snapshot.previousTransactionFingerprint =
            active.previousTransactionFingerprint;
        snapshot.linearizationEpoch = capture.linearizationEpoch;
        snapshot.slotGeneration = active.slotGeneration;
        snapshot.controlStep = active.controlStep;
        snapshot.physicsGeneration = active.physicsGeneration;
        snapshot.previousPhysicsGeneration = active.previousPhysicsGeneration;
        snapshot.brainGeneration = active.brainGeneration;
        snapshot.sensorGeneration = active.candidateKey.sensorGeneration;
        snapshot.humanIOProgramFingerprint =
            active.candidateKey.programFingerprint;
        snapshot.sensorFingerprint = active.candidateKey.sensorFingerprint;
        snapshot.transactionInstanceFingerprint =
            active.candidateKey.transactionInstanceFingerprint;
        if (!active.exactFamily &&
            active.acceptedTimestampMicroseconds >
                std::numeric_limits<std::uint64_t>::max() /
                    runtime.clockQuantumNanoseconds) {
            error = "production-owner accepted timestamp overflow";
            return false;
        }
        snapshot.candidateTimestampNanoseconds = active.exactFamily
            ? active.acceptedTimestampNanoseconds
            : active.acceptedTimestampMicroseconds *
                runtime.clockQuantumNanoseconds;
        snapshot.publicationEpoch = published ? publicationEpoch : 0u;
        snapshot.jointFenceFingerprint = published
            ? committedFence->fenceFingerprint : 0u;
        snapshot.timestepNanoseconds = runtime.timestepNanoseconds;
        snapshot.equalityProgramFingerprint =
            runtime.assets.equalityFingerprint;
        snapshot.limitProgramFingerprint = runtime.assets.limitFingerprint;
        snapshot.supportPayloadByteCount =
            runtime.assets.supportIdentity.byteCount;
        snapshot.supportPayloadABI = runtime.assets.supportIdentity.payloadABI;
        snapshot.supportPayloadSHA256.reserve(
            runtime.assets.supportIdentity.sha256.size());
        for (const auto byte : runtime.assets.supportIdentity.sha256)
            snapshot.supportPayloadSHA256.push_back(std::byte{byte});
        snapshot.qCoordinateCount = runtime.assets.rigid.nq;
        snapshot.dofCount = runtime.assets.rigid.nv;
        snapshot.muscleCount = runtime.assets.muscle.muscleCount;
        snapshot.muscleSiteCount = runtime.assets.sites.size();
        snapshot.muscleWrapCount = runtime.assets.wraps.size();
        snapshot.muscleRouteNodeCount = runtime.assets.routes.size();
        snapshot.supportRowCount = runtime.assets.matterSupportContacts.size();
        snapshot.equalityRowCount = runtime.assets.jointEqualities.size();
        snapshot.limitRowCount = runtime.assets.jointLimits.size();
        snapshot.tendonRowCount = 0u;
        snapshot.matterControlStep = matter.controlStep;
        snapshot.publicationIdentityAvailable = publicationIdentityValid;
        snapshot.rollbackIdentityAvailable = rollbackIdentityValid;

        snapshot.initialQ = ownerHostArray<float>(runtime.assets.initialQ);
        snapshot.initialV = ownerHostArray<float>(runtime.assets.initialV);
        snapshot.initialRoot =
            ownerHostArray<MRCompensatedRootTranslationGPU>(
                runtime.assets.initialRootTranslations);
        snapshot.initialMuscles = ownerHostArray<MRMujocoMuscleStateGPU>(
            runtime.assets.states);
        snapshot.muscleRecords = ownerHostArray<MRMujocoMuscleGPU>(
            runtime.assets.muscles);
        snapshot.muscleSites = ownerHostArray<MRMujocoMuscleSiteGPU>(
            runtime.assets.sites);
        snapshot.muscleWraps = ownerHostArray<MRMujocoMuscleWrapGPU>(
            runtime.assets.wraps);
        snapshot.muscleRouteNodes =
            ownerHostArray<MRMujocoMuscleRouteNodeGPU>(
                runtime.assets.routes);
        snapshot.supportRows = ownerHostArray<NMHumanSupportContactGPU>(
            runtime.assets.matterSupportContacts);
        const std::array<mr_float4, 2u> supportPlane{
            runtime.assets.groundPoint, runtime.assets.groundNormal};
        snapshot.supportPlane = ownerHostArray<mr_float4>(supportPlane);
        std::vector<nm_float4> realizedInitialSupportHistories =
            runtime.ownerSnapshotInitialSupportHistories;
        if (realizedInitialSupportHistories.empty()) {
            realizedInitialSupportHistories.resize(
                runtime.assets.matterSupportContacts.size());
        }
        snapshot.initialSupportHistories = ownerHostArray<nm_float4>(
            realizedInitialSupportHistories);
        snapshot.terminalAcceptedSupportHistories = ownerHostArray<nm_float4>(
            matter.humanSupportHistories);
        snapshot.terminalAcceptedSupportConsequences =
            ownerHostArray<NMHumanSupportConsequenceGPU>(
                matter.humanSupportConsequences);
        snapshot.equalityRows = ownerHostArray<NMHumanJointEqualityGPU>(
            runtime.assets.jointEqualities);
        snapshot.limitRows = ownerHostArray<NMHumanJointLimitGPU>(
            runtime.assets.jointLimits);
        snapshot.contactSampleCount =
            runtime.ownerSnapshotContactSampleCount;
        snapshot.contactSamples = ownerHostArray<NMContactSampleGPU>(
            matter.contactSamples);
        snapshot.terminalAcceptedMatterRigidGeneralizedStateCount =
            runtime.ownerSnapshotMatterGeneralizedStateCount;
        snapshot.terminalAcceptedMatterRigidGeneralizedState =
            ownerHostArray<float>(matter.rigidGeneralizedCandidate);
        snapshot.terminalAcceptedMatterRigidReactionCount =
            runtime.ownerSnapshotMatterReactionCount;
        if (runtime.ownerSnapshotMatterReactionCount >
                std::numeric_limits<std::size_t>::max()) {
            error = "production-owner Matter reaction count exceeds host size";
            return false;
        }
        const std::size_t logicalMatterReactionCount =
            static_cast<std::size_t>(
                runtime.ownerSnapshotMatterReactionCount);
        // Matter deliberately retains one fully typed allocation sentinel for
        // a zero-width Metal binding. It is physical storage, not a logical
        // reaction, and must never be promoted into production-owner evidence.
        // Every nonzero logical range remains exact and fail-closed.
        const bool exactMatterReactionShape =
            matter.reactions.size() == logicalMatterReactionCount;
        const bool typedZeroMatterReactionSentinel =
            logicalMatterReactionCount == 0u &&
            matter.reactions.size() == 1u;
        if (!exactMatterReactionShape &&
            !typedZeroMatterReactionSentinel) {
            std::ostringstream detail;
            detail << "production-owner Matter reaction logical shape mismatch"
                   << " (expected_elements=" << logicalMatterReactionCount
                   << ", physical_elements=" << matter.reactions.size()
                   << ')';
            error = detail.str();
            return false;
        }
        snapshot.terminalAcceptedMatterRigidReactions =
            ownerHostArray<NMRigidReactionGPU>(
                std::span<const NMRigidReactionGPU>(matter.reactions)
                    .first(logicalMatterReactionCount));
        snapshot.tendonTransfers.available = true;
        snapshot.tendonTransfers.expectedElementCount = 0u;
        snapshot.tendonTransfers.elementBytes =
            sizeof(MRNumiHumanTendonTransferResultGPU);
        snapshot.tendonGeneralizedCorrections.available = true;
        snapshot.tendonGeneralizedCorrections.expectedElementCount = 0u;
        snapshot.tendonGeneralizedCorrections.elementBytes = sizeof(float);

        const std::uint64_t nq = snapshot.qCoordinateCount;
        const std::uint64_t nv = snapshot.dofCount;
        const std::uint64_t muscleCount = snapshot.muscleCount;
        if (!ownerCapturedArray(capture, capture.layout.checkpointQ,
                nq, sizeof(float), snapshot.checkpointQ) ||
            !ownerCapturedArray(capture, capture.layout.checkpointV,
                nv, sizeof(float), snapshot.checkpointV) ||
            !ownerCapturedArray(capture, capture.layout.checkpointRoot,
                1u, sizeof(MRCompensatedRootTranslationGPU),
                snapshot.checkpointRoot) ||
            !ownerCapturedArray(capture, capture.layout.checkpointMuscles,
                muscleCount, sizeof(MRMujocoMuscleStateGPU),
                snapshot.checkpointMuscles) ||
            !ownerCapturedArray(capture,
                capture.layout.effectiveTangentFactorStorage, nv * nv,
                sizeof(float), snapshot.effectiveTangentFactorStorage) ||
            !ownerCapturedArray(capture,
                capture.layout.sourceGeneralizedForce, nv, sizeof(float),
                snapshot.sourceGeneralizedForce) ||
            !ownerCapturedArray(capture,
                capture.layout.sourcePredictedVelocity, nv, sizeof(float),
                snapshot.sourcePredictedVelocity) ||
            !ownerCapturedArray(capture,
                capture.layout.matterGeneralizedReaction, nv, sizeof(float),
                snapshot.matterGeneralizedReaction) ||
            !ownerCapturedArray(capture, capture.layout.candidateQ,
                nq, sizeof(float), snapshot.candidateQ) ||
            !ownerCapturedArray(capture, capture.layout.candidateV,
                nv, sizeof(float), snapshot.candidateV) ||
            !ownerCapturedArray(capture, capture.layout.candidateRoot,
                1u, sizeof(MRCompensatedRootTranslationGPU),
                snapshot.candidateRoot) ||
            !ownerCapturedArray(capture, capture.layout.candidateMuscles,
                muscleCount, sizeof(MRMujocoMuscleStateGPU),
                snapshot.candidateMuscles) ||
            !ownerCapturedArray(capture, capture.layout.muscleResults,
                muscleCount, sizeof(MRMujocoMuscleResultGPU),
                snapshot.muscleResults) ||
            !ownerCapturedArray(capture,
                capture.layout.muscleGeneralizedForces,
                muscleCount * nv, sizeof(float),
                snapshot.muscleGeneralizedForces) ||
            !ownerCapturedArray(capture,
                capture.layout.reducedMuscleGeneralizedForce, nv,
                sizeof(float), snapshot.reducedMuscleGeneralizedForce) ||
            !ownerCapturedArray(capture, capture.layout.standStatus,
                1u, sizeof(MRNumiHumanStandStatusGPU),
                snapshot.standStatus) ||
            !ownerCapturedArray(capture, capture.layout.ownerStatus,
                1u, sizeof(MRNumanXHumanMatterOwnerStatusGPU),
                snapshot.humanMatterOwnerStatus) ||
            !ownerCapturedArray(capture,
                capture.layout.candidateSupportConsequences,
                snapshot.supportRowCount,
                sizeof(NMHumanSupportConsequenceGPU),
                snapshot.candidateSupportConsequences)) {
            error = "production-owner captured Metal range is incomplete";
            return false;
        }
        std::vector<nm_float4> candidateSupportHistories(
            snapshot.supportRowCount);
        for (std::size_t index = 0u;
             index < candidateSupportHistories.size(); ++index) {
            NMHumanSupportConsequenceGPU consequence{};
            std::memcpy(&consequence,
                snapshot.candidateSupportConsequences.bytes.data() +
                    index * sizeof(consequence),
                sizeof(consequence));
            const auto normal = runtime.assets.groundNormal;
            const float normalImpulse =
                consequence.impulseAndNormal.w;
            candidateSupportHistories[index] = {
                consequence.impulseAndNormal.x -
                    normal.x * normalImpulse,
                consequence.impulseAndNormal.y -
                    normal.y * normalImpulse,
                consequence.impulseAndNormal.z -
                    normal.z * normalImpulse,
                normalImpulse};
            const auto& history = candidateSupportHistories[index];
            if (!std::isfinite(history.x) || !std::isfinite(history.y) ||
                !std::isfinite(history.z) || !std::isfinite(history.w)) {
                error = "production-owner candidate support history is nonfinite";
                return false;
            }
        }
        snapshot.candidateSupportHistories = ownerHostArray<nm_float4>(
            candidateSupportHistories);
        if (!deriveOwnerSnapshotDynamics(snapshot, error) ||
            !deriveOwnerConstraintWitnesses(runtime, snapshot, error)) {
            return false;
        }
        snapshot.requiredCoverageMask =
            metalrobo::NumiHumanOwnerInitialStateV1 |
            metalrobo::NumiHumanOwnerCheckpointStateV1 |
            metalrobo::NumiHumanOwnerEffectiveTangentFactorV1 |
            metalrobo::NumiHumanOwnerSourceGeneralizedForceV1 |
            metalrobo::NumiHumanOwnerFreeVelocityV1 |
            metalrobo::NumiHumanOwnerCandidateStateV1 |
            metalrobo::NumiHumanOwnerMuscleStateV1 |
            metalrobo::NumiHumanOwnerMuscleResultsV1 |
            metalrobo::NumiHumanOwnerMuscleGeneralizedForcesV1 |
            metalrobo::NumiHumanOwnerMuscleProgramV1 |
            metalrobo::NumiHumanOwnerTendonV1 |
            metalrobo::NumiHumanOwnerMatterReactionV1 |
            metalrobo::NumiHumanOwnerSupportRowsV1 |
            metalrobo::NumiHumanOwnerSupportHistoryV1 |
            metalrobo::NumiHumanOwnerEqualityRowsV1 |
            metalrobo::NumiHumanOwnerLimitRowsV1 |
            metalrobo::NumiHumanOwnerRHSBiasAccelerationV1 |
            metalrobo::NumiHumanOwnerWorkEnergyV1 |
            metalrobo::NumiHumanOwnerContactSamplesV1 |
            metalrobo::NumiHumanOwnerStatusRecordsV1 |
            (published
                ? metalrobo::NumiHumanOwnerMatterIntegrationUpdateV1
                : 0u) |
            (published
                ? metalrobo::NumiHumanOwnerPublicationIdentityV1
                : metalrobo::NumiHumanOwnerRollbackIdentityV1);
        snapshot.coverageMask =
            metalrobo::numiHumanProductionOwnerCoverageMaskV1(snapshot);
        std::string envelope;
        std::string payloadSHA256;
        if (!metalrobo::
                serializeNumiHumanProductionOwnerSnapshotEvidenceV1(
                    snapshot, envelope, payloadSHA256, error)) {
            return false;
        }
        envelope.push_back('\n');

        std::ostringstream name;
        name << "persistent-production-owner-snapshot.v1.root-"
             << active.controlStep << '.' << std::hex << std::nouppercase
             << std::setfill('0') << std::setw(16)
             << active.transactionFingerprint << '.'
             << (published ? "published" : "rejected") << ".json";
        const auto target = runtime.ownerSnapshotDirectory / name.str();
        if (!metalrobo::publishNumiHumanProductionOwnerEvidenceNoReplace(
                target, envelope, error))
            return false;
        std::fprintf(stderr,
            "mrnx_production_owner_snapshot={\"path\":\"%s\","
            "\"payload_sha256\":\"%s\",\"root\":%llu,"
            "\"disposition\":\"%s\"}\n",
            target.string().c_str(), payloadSHA256.c_str(),
            static_cast<unsigned long long>(active.controlStep),
            published ? "published" : "rejected");
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        try {
            error = std::string("production-owner evidence exception: ") +
                exception.what();
        } catch (...) {
            error.clear();
        }
        return false;
    } catch (...) {
        try {
            error = "production-owner evidence unknown exception";
        } catch (...) {
            error.clear();
        }
        return false;
    }
}

void runtimeTerminalCompletion(
    void* raw,
    const PreparedTerminalDisposition disposition,
    const mrnx_root_v1& root,
    const mrnx_candidate_view_v1* candidate,
    const mrnx_candidate_channel_v1* channels,
    const std::uint32_t channelCount,
    const MRNumanXHumanMatterJointPublicationFenceGPU* committedFence
) noexcept {
    auto* runtime = static_cast<RuntimeState*>(raw);
    if (runtime == nullptr) return;
    std::shared_ptr<ActiveRoot> active;
    bool persistRejectedOwnerSnapshot = false;
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
        persistRejectedOwnerSnapshot =
            disposition == PreparedTerminalDisposition::rejected &&
            active->ownerSnapshotCapture.has_value() &&
            runtime->ownerSnapshotSelectedControlStep.has_value() &&
            !runtime->ownerSnapshotSelectedCaptured &&
            active->controlStep ==
                *runtime->ownerSnapshotSelectedControlStep;
        if (disposition == PreparedTerminalDisposition::rejected)
            recordRuntimeBehaviorTerminal(*runtime, *active, root, false, nullptr);
        // Keep a selected rejected root reserved until its completion-boundary
        // evidence has been read and durably published. Clearing active here
        // would admit a new Matter transaction that could race the synchronous
        // terminal snapshot below and mix two roots in one record.
        if (disposition != PreparedTerminalDisposition::published &&
            !persistRejectedOwnerSnapshot) {
            runtime->active.reset();
        }
        if (disposition == PreparedTerminalDisposition::terminalNoTouch) {
            if (runtime->behavior != nullptr) runtime->behaviorError = "physical attempt has no committed terminal outcome";
            runtime->terminalQuarantine = true;
        }
    }
    if (disposition != PreparedTerminalDisposition::published) {
        if (disposition == PreparedTerminalDisposition::rejected &&
            runtime->culture != nullptr) {
            runtime->culture->rejectPrepared();
        }
        if (persistRejectedOwnerSnapshot) {
            std::string evidenceError;
            const bool persisted = writeOwnerSnapshotEvidence(
                    *runtime, *active, root, disposition, 0u, nullptr,
                    evidenceError);
            const std::lock_guard lock(runtime->mutex);
            if (runtime->active == active) runtime->active.reset();
            if (persisted) {
                runtime->ownerSnapshotSelectedCaptured = true;
            } else {
                std::fprintf(stderr,
                    "mrnx_production_owner_snapshot_failure=%s\n",
                    evidenceError.c_str());
                runtime->terminalQuarantine = true;
            }
        }
        return;
    }
    if (active != nullptr && active->exactFamily) {
        if (!publishExactRuntimeTerminal(
                *runtime, active, root, candidate, channels, channelCount,
                committedFence)) {
            const std::lock_guard lock(runtime->mutex);
            if (runtime->active == active) runtime->active.reset();
            runtime->terminalQuarantine = true;
        }
        return;
    }
    const bool commonIdentityValid = active != nullptr &&
        candidate != nullptr && channels != nullptr && channelCount == 7u &&
        candidate->abi_version == MRNX_BRIDGE_ABI_V1 &&
        candidate->struct_size == sizeof(*candidate) &&
        candidate->key.abi_version == MRNX_BRIDGE_ABI_V1 &&
        candidate->key.struct_size == sizeof(candidate->key) &&
        candidate->channel_count == channelCount &&
        candidate->reserved0 == 0u &&
        candidate->device_registry_id == runtime->device.registryID &&
        root.transaction_fingerprint == active->transactionFingerprint &&
        root.control_step == active->controlStep &&
        candidate->accepted_brain_generation == active->brainGeneration &&
        candidate->key.transaction_fingerprint ==
            active->transactionFingerprint &&
        candidate->key.program_fingerprint ==
            active->candidateKey.programFingerprint &&
        candidate->key.sensor_fingerprint ==
            active->candidateKey.sensorFingerprint &&
        candidate->key.transaction_instance_fingerprint ==
            active->candidateKey.transactionInstanceFingerprint &&
        candidate->key.sensor_generation ==
            active->candidateKey.sensorGeneration &&
        candidate->key.command_buffer_identity ==
            active->candidateKey.commandBufferIdentity &&
        candidate->key.fingerprint != 0u &&
        candidate->candidate_publication_fingerprint != 0u &&
        candidate->candidate_identity_fingerprint != 0u;
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
    if (runtime->culture != nullptr) {
        if (!active->cultureAcceptedView.valid() ||
            active->cultureAccepted.culture_fingerprint !=
                runtime->culturePack.fingerprint() ||
            active->cultureAccepted.generation == 0u ||
            runtime->culture->publishPrepared() !=
                metalrobo::MetalNeuronCultureStatus::success) {
            runtime->active.reset();
            runtime->terminalQuarantine = true;
            return;
        }
        runtime->publishedCultureView = active->cultureAcceptedView;
        runtime->aggregateCulture = active->cultureAccepted;
    }
    snapshot.abi_version = MRNX_BRIDGE_ABI_V1;
    snapshot.struct_size = sizeof(snapshot);
    snapshot.publication_epoch = priorEpoch + 1u;
    if (snapshot.publication_epoch !=
        metalrobo::numanx_bridge_v1::domainPublicationEpoch(
            runtime->domain)) {
        runtime->active.reset();
        runtime->terminalQuarantine = true;
        return;
    }
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
    recordRuntimeBehaviorTerminal(*runtime, *active, root, true, committedFence);
    if (active->rootTranslationTrace != nil && active->rootTranslationTrace.contents != nullptr) {
        MRCompensatedRootTranslationGPU translation{};
        std::memcpy(&translation, active->rootTranslationTrace.contents, sizeof(translation));
        std::array<std::uint32_t, 12u> words{};
        std::memcpy(words.data(), &translation, sizeof(translation));
        // One bounded record after the actual joint release. Readers join this
        // to the subsequent per-root Brain trace, not a candidate-only event.
        std::fprintf(stderr,
            "mrnx_accepted_root_translation={\"schema\":\"numi.human.accepted-root-translation.v1\","
            "\"root\":%llu,\"physics_generation\":%llu,\"timestamp_microseconds\":%llu,"
            "\"transaction_fingerprint\":\"%016llx\",\"valid\":%s,\"words\":["
            "\"%08x\",\"%08x\",\"%08x\",\"%08x\",\"%08x\",\"%08x\","
            "\"%08x\",\"%08x\",\"%08x\",\"%08x\",\"%08x\",\"%08x\"]}\n",
            static_cast<unsigned long long>(active->controlStep),
            static_cast<unsigned long long>(active->physicsGeneration),
            static_cast<unsigned long long>(active->acceptedTimestampMicroseconds),
            static_cast<unsigned long long>(active->transactionFingerprint),
            mrCompensatedTranslationValid(translation) ? "true" : "false",
            words[0], words[1], words[2], words[3], words[4], words[5],
            words[6], words[7], words[8], words[9], words[10], words[11]);
    }
    const bool selectedOwnerSnapshot =
        runtime->ownerSnapshotSelectedControlStep.has_value() &&
        !runtime->ownerSnapshotSelectedCaptured &&
        active->controlStep ==
            *runtime->ownerSnapshotSelectedControlStep;
    const bool firstOwnerSnapshot =
        !runtime->ownerSnapshotFirstPublishedCaptured;
    if (active->ownerSnapshotCapture &&
        (firstOwnerSnapshot || selectedOwnerSnapshot)) {
        std::string evidenceError;
        if (writeOwnerSnapshotEvidence(
                *runtime, *active, root, disposition,
                runtime->aggregate.publication_epoch, committedFence,
                evidenceError)) {
            if (firstOwnerSnapshot)
                runtime->ownerSnapshotFirstPublishedCaptured = true;
            if (selectedOwnerSnapshot)
                runtime->ownerSnapshotSelectedCaptured = true;
        } else {
            std::fprintf(stderr,
                "mrnx_production_owner_snapshot_failure=%s\n",
                evidenceError.c_str());
            runtime->terminalQuarantine = true;
        }
    }
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
    bool exactFamily = false;
    {
        const std::lock_guard lock(active->mutex);
        if (active->settlementStarted || !active->humanSettled ||
            !active->physicalSettled || !active->cultureSettled) {
            return;
        }
        active->settlementStarted = true;
        prepared = active->prepared;
        candidate = active->candidate;
        exactFamily = active->exactFamily;
        callback = active->completion;
        callbackContext = active->completionContext;
        generation = active->slotGeneration;
        ready = active->humanReady && active->physicalReady &&
            active->cultureReady &&
            prepared != nullptr && candidate != nullptr;
        if (exactFamily) {
            ready = active->humanReady && active->physicalReady &&
                active->cultureReady && prepared != nullptr;
        }
    }

    if (ready && exactFamily) {
        candidate = finalizeExactCandidate(active);
        ready = candidate != nullptr;
        if (ready) {
            const std::lock_guard lock(active->mutex);
            active->candidate = candidate;
        }
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
    if (ready && active->runtime->culture != nullptr) {
        ready = metalrobo::numanx_bridge_v1::installPreparedCultureView(
            prepared, active->culturePrepared);
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

[[nodiscard]] mrnx_candidate_channel_v2 supplementalChannelV2(
    const std::uint32_t modality,
    const std::uint64_t receptorTimestampNanoseconds,
    const std::uint32_t receptorCount,
    const std::uint32_t featureCount,
    id<MTLBuffer> values,
    id<MTLBuffer> validity
) noexcept {
    mrnx_candidate_channel_v2 result{};
    result.abi_version = MRNX_CANDIDATE_CHANNEL_ABI_V2;
    result.struct_size = sizeof(result);
    result.modality = modality;
    result.flags = MRNX_CANDIDATE_CHANNEL_HAS_VALIDITY_V1;
    result.receptor_timestamp_nanoseconds = receptorTimestampNanoseconds;
    result.clock_domain = MRNX_PHYSICAL_CLOCK_DOMAIN_EXACT_NANOSECONDS;
    result.clock_quantum_nanoseconds = MRNX_EXACT_CLOCK_QUANTUM_NANOSECONDS;
    result.receptor_count = receptorCount;
    result.feature_dimension = featureCount;
    result.values = outputRange(
        values, MRNX_ELEMENT_FLOAT32_V1, sizeof(float));
    result.validity = outputRange(
        validity, MRNX_ELEMENT_UINT32_V1, sizeof(std::uint32_t));
    result.channel_fingerprint =
        metalrobo::metalNumanXExactCandidateChannelV2Fingerprint(result);
    return result;
}

[[nodiscard]] mrnx_candidate_v1* finalizeExactCandidate(
    const std::shared_ptr<ActiveRoot>& active
) noexcept {
    if (active == nullptr || active->runtime == nullptr ||
        !active->exactFamily || !active->candidateKey.valid() ||
        !active->exactHumanIO.valid()) {
        return nullptr;
    }
    mrnx_exact_inbound_authority_v2 authority{};
    static_assert(sizeof(authority) ==
                  sizeof(active->exactReceipt.inboundAuthority));
    std::memcpy(
        &authority, &active->exactReceipt.inboundAuthority,
        sizeof(authority));
    const auto& proof = active->exactReceipt.acceptedStateProof;
    const auto& token = active->exactReceipt.acceptedPhysicsStateToken;
    if (!metalrobo::metalNumanXExactInboundAuthorityV2Valid(authority) ||
        !metalrobo::metalNumanXExactAcceptedStateProofV2Valid(proof) ||
        !metalrobo::metalNumanXExactAcceptedPhysicsTokenV2Valid(
            proof, token)) {
        return nullptr;
    }

    metalrobo::MetalNumanXHumanIOCandidatePublicationLease lease;
    const auto reserved = active->runtime->humanIO->
        reserveCandidatePublication(
            active->candidateKey, active->exactHumanIO.authority, lease);
    if (!reserved.succeeded() || !lease.valid()) return nullptr;
    auto* candidate = metalrobo::numanx_bridge_v1::adoptCandidateV2(
        active->runtime->domain, std::move(lease), authority, proof, token);
    if (candidate == nullptr) {
        const std::lock_guard lock(active->runtime->mutex);
        active->runtime->terminalQuarantine = true;
        active->runtime->info.request_failure_stage = 1010u;
        return nullptr;
    }

    const mrnx_candidate_channel_v2 supplemental[] = {
        supplementalChannelV2(
            MRNX_CANDIDATE_MODALITY_KINESTHESIA_V1,
            active->receptorTimestampNanoseconds,
            MR_NUMANX_HUMAN_KINESTHESIA_RECEPTOR_COUNT,
            MR_NUMANX_HUMAN_KINESTHESIA_FEATURE_COUNT,
            active->kinesthesia, active->kinesthesiaValidity),
        supplementalChannelV2(
            MRNX_CANDIDATE_MODALITY_VESTIBULAR_V1,
            active->receptorTimestampNanoseconds,
            MR_NUMANX_HUMAN_VESTIBULAR_RECEPTOR_COUNT,
            MR_NUMANX_HUMAN_VESTIBULAR_FEATURE_COUNT,
            active->vestibular, active->vestibularValidity),
        supplementalChannelV2(
            MRNX_CANDIDATE_MODALITY_AUDITION_V1,
            active->receptorTimestampNanoseconds,
            MR_NUMANX_HUMAN_AUDITION_RECEPTOR_COUNT,
            MR_NUMANX_HUMAN_AUDITION_FEATURE_COUNT,
            active->audition, active->auditionValidity),
        supplementalChannelV2(
            MRNX_CANDIDATE_MODALITY_VISION_V1,
            active->receptorTimestampNanoseconds,
            MR_NUMANX_HUMAN_VISION_RECEPTOR_COUNT,
            MR_NUMANX_HUMAN_VISION_FEATURE_COUNT,
            active->vision, active->visionValidity),
        supplementalChannelV2(
            MRNX_CANDIDATE_MODALITY_TOUCH_V1,
            active->receptorTimestampNanoseconds,
            MR_NUMANX_HUMAN_TOUCH_RECEPTOR_COUNT,
            MR_NUMANX_HUMAN_TOUCH_FEATURE_COUNT,
            active->touch, active->touchValidity),
    };
    if (!metalrobo::numanx_bridge_v1::attachCandidateChannelsV2(
            candidate, supplemental,
            static_cast<std::uint32_t>(std::size(supplemental)))) {
        (void)mrnx_bridge_v1_reject_unbound_candidate(candidate);
        mrnx_bridge_v1_candidate_drop(candidate);
        return nullptr;
    }
    mrnx_candidate_timing_v2 timing{};
    timing.abi_version = MRNX_CANDIDATE_TIMING_ABI_V2;
    timing.struct_size = sizeof(timing);
    mrnx_exact_inbound_authority_v2 copiedAuthority{};
    copiedAuthority.abi_version = MRNX_EXACT_INBOUND_AUTHORITY_ABI_V2;
    copiedAuthority.struct_size = sizeof(copiedAuthority);
    mrnx_exact_sensor_packet_v2 packet{};
    packet.abi_version = MRNX_EXACT_SENSOR_PACKET_ABI_V2;
    packet.struct_size = sizeof(packet);
    mrnx_candidate_channel_v2 channels[MRNX_MAX_SENSOR_CHANNELS_V2]{};
    bool copied = mrnx_bridge_v1_candidate_copy_timing_v2(
            candidate, &timing) &&
        mrnx_bridge_v1_candidate_copy_inbound_authority_v2(
            candidate, &copiedAuthority) &&
        mrnx_bridge_v1_candidate_copy_sensor_packet_v2(candidate, &packet) &&
        packet.channel_count == 7u &&
        packet.channel_count <= MRNX_MAX_SENSOR_CHANNELS_V2;
    for (std::uint32_t index = 0u;
         copied && index < packet.channel_count; ++index) {
        channels[index].abi_version = MRNX_CANDIDATE_CHANNEL_ABI_V2;
        channels[index].struct_size = sizeof(channels[index]);
        copied = mrnx_bridge_v1_candidate_copy_channel_v2(
            candidate, index, &channels[index]);
    }
    copied = copied && copiedAuthority.inbound_authority_fingerprint ==
            authority.inbound_authority_fingerprint &&
        metalrobo::metalNumanXExactSensorPacketV2Valid(
            copiedAuthority, proof, token, timing, channels,
            packet.channel_count, packet);
    if (!copied) {
        (void)mrnx_bridge_v1_reject_unbound_candidate(candidate);
        mrnx_bridge_v1_candidate_drop(candidate);
        return nullptr;
    }
    active->exactTiming = timing;
    active->exactInboundAuthority = copiedAuthority;
    active->exactSensorPacket = packet;
    active->exactChannelCount = packet.channel_count;
    for (std::uint32_t index = 0u; index < packet.channel_count; ++index) {
        active->exactChannels[index] = channels[index];
    }
    return candidate;
}

[[nodiscard]] bool bridgeCultureAcceptedView(
    const metalrobo::MetalNeuronCultureAcceptedView& source,
    const std::uint64_t deviceRegistryID,
    mrnx_culture_accepted_view_v1& output
) noexcept {
    if (!source.valid() || source.cultureFingerprint() == 0u ||
        source.generation() == 0u || source.completionEvent() == nullptr ||
        source.completionValue() != source.generation() ||
        source.buffers().size() != MRNX_CULTURE_ACCEPTED_BUFFER_COUNT_V1) return false;
    mrnx_culture_accepted_view_v1 result{};
    result.abi_version = MRNX_CULTURE_ACCEPTED_VIEW_ABI_V1;
    result.struct_size = sizeof(result);
    result.culture_fingerprint = source.cultureFingerprint();
    result.generation = source.generation();
    result.tick = source.tick();
    result.growth_generation = source.growthIteration();
    result.ready.abi_version = MRNX_BRIDGE_ABI_V1;
    result.ready.struct_size = sizeof(result.ready);
    result.ready.shared_event = source.completionEvent();
    result.ready.value = source.completionValue();
    result.ready.device_registry_id = deviceRegistryID;
    result.buffer_count = MRNX_CULTURE_ACCEPTED_BUFFER_COUNT_V1;
    std::array<bool, MRNX_CULTURE_ACCEPTED_BUFFER_COUNT_V1> seen{};
    for (const auto& buffer : source.buffers()) {
        const auto index = static_cast<std::uint32_t>(buffer.kind);
        if (index >= seen.size() || seen[index] || buffer.metalBuffer == nullptr ||
            buffer.gpuAddress == 0u || buffer.byteLength == 0u) return false;
        seen[index] = true;
        const bool unsignedBuffer =
            buffer.kind == metalrobo::MetalNeuronCultureAcceptedBuffer::spikes ||
            buffer.kind == metalrobo::MetalNeuronCultureAcceptedBuffer::spikeHistory ||
            buffer.kind ==
                metalrobo::MetalNeuronCultureAcceptedBuffer::electrodeSpikeCounts;
        __unsafe_unretained id<MTLBuffer> object =
            (__bridge id<MTLBuffer>)buffer.metalBuffer;
        if (object == nil || object.gpuAddress != buffer.gpuAddress ||
            object.length != buffer.byteLength ||
            object.device.registryID != deviceRegistryID) return false;
        result.buffers[index] = outputRange(
            object,
            unsignedBuffer ? MRNX_ELEMENT_UINT32_V1 : MRNX_ELEMENT_FLOAT32_V1,
            sizeof(std::uint32_t));
    }
    if (!std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })) {
        return false;
    }
    output = result;
    return true;
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
    if (ready && !active->exactFamily) {
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
        metalrobo::MetalNumanXHumanMatterExactPhysicalReceipt exactReceipt{};
        const bool hasExactReceipt = !active->exactFamily ||
            (active->physicalReady &&
             active->runtime->adapter->exactPhysicalReceipt(
                 active->transactionSlot,
                 active->transactionFingerprint,
                 active->slotGeneration,
                 exactReceipt));
        const std::uint64_t exactProofProgramFingerprint =
            active->exactFamily && active->runtime->matter != nullptr
            ? active->runtime->matter
                  ->acceptedStateProofProgramFingerprintV2()
            : 0u;
        const std::lock_guard runtimeLock(active->runtime->mutex);
        const bool exactProvenanceValid = !active->exactFamily ||
            (exactProofProgramFingerprint != 0u &&
             exactReceipt.acceptedStateProof.stateProofProgramFingerprint ==
                 exactProofProgramFingerprint &&
             active->runtime->info.
                     accepted_state_proof_program_fingerprint ==
                 exactProofProgramFingerprint);
        active->physicalReady = active->physicalReady && hasOutcome &&
            hasExactReceipt && exactProvenanceValid;
        if (active->physicalReady && active->exactFamily) {
            active->exactReceipt = exactReceipt;
        }
        if (!active->physicalReady) {
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
    if (active->runtime->culture != nullptr) {
        bool physicalReady = false;
        __strong id<MTLBuffer> supportConsequences = nil;
        std::uint64_t supportAddress = 0u;
        {
            const std::lock_guard lock(active->mutex);
            physicalReady = active->physicalReady;
            supportConsequences = active->supportConsequences;
            supportAddress = active->supportConsequencesGPUAddress;
        }
        if (physicalReady && supportConsequences != nil && supportAddress != 0u) {
            const metalrobo::MetalNeuronCultureSupportRequest request{
                .cultureFingerprint = active->runtime->culturePack.fingerprint(),
                .rootFingerprint = active->transactionFingerprint,
                .supportConsequencesBuffer = (__bridge void*)supportConsequences,
                .supportConsequencesGPUAddress = supportAddress,
                .supportCount = 10u,
                .supportStride = 10u,
                .tickCount = active->runtime->cultureWindowTicks,
                .physicsTimestepSeconds = static_cast<float>(
                    active->runtime->timestepSeconds),
                .currentPerNewton = active->runtime->cultureCurrentPerNewton,
            };
            auto ticket = active->runtime->culture->prepareSupportWindow(request);
            mrnx_root_v1 root{};
            root.abi_version = MRNX_BRIDGE_ABI_V1;
            root.struct_size = sizeof(root);
            const auto accepted = active->runtime->culture->acceptedView();
            __unsafe_unretained id<MTLSharedEvent> event =
                (__bridge id<MTLSharedEvent>)ticket.completionEvent();
            mrnx_culture_prepared_view_v1 view{};
            const bool viewReady = ticket.valid() && accepted.valid() &&
                accepted.generation() != std::numeric_limits<std::uint64_t>::max() &&
                event != nil && ticket.completionValue() != 0u &&
                active->prepared != nullptr &&
                mrnx_bridge_v1_prepared_copy_root(active->prepared, &root);
            if (viewReady) {
                view.abi_version = MRNX_CULTURE_PREPARED_VIEW_ABI_V1;
                view.struct_size = sizeof(view);
                view.root = root;
                view.culture_fingerprint = accepted.cultureFingerprint();
                view.accepted_generation = accepted.generation();
                view.prepared_generation = accepted.generation() + 1u;
                view.source_root_fingerprint = active->transactionFingerprint;
                view.ready.abi_version = MRNX_BRIDGE_ABI_V1;
                view.ready.struct_size = sizeof(view.ready);
                view.ready.shared_event = (__bridge void*)event;
                view.ready.value = ticket.completionValue();
                view.ready.device_registry_id = active->runtime->device.registryID;
                view.status = 1u;
                view.receipt_fingerprint = cultureReceiptFingerprint(view);
            }
            if (viewReady && view.receipt_fingerprint != 0u) {
                {
                    const std::lock_guard lock(active->mutex);
                    active->culturePrepared = view;
                    active->cultureTicket.emplace(std::move(ticket));
                }
                if (!active->cultureTicket->onCompleted(
                        active.get(), &cultureCompletion)) {
                    active->runtime->culture->rejectPrepared();
                    const std::lock_guard lock(active->mutex);
                    active->cultureSettled = true;
                    active->cultureReady = false;
                }
            } else {
                active->runtime->culture->rejectPrepared();
                const std::lock_guard lock(active->mutex);
                active->cultureSettled = true;
                active->cultureReady = false;
            }
        } else {
            const std::lock_guard lock(active->mutex);
            active->cultureSettled = true;
            active->cultureReady = false;
        }
    }
    settleActiveRoot(active);
}

void cultureCompletion(
    void* raw,
    const metalrobo::MetalNeuronCultureStatus status
) noexcept {
    auto* pointer = static_cast<ActiveRoot*>(raw);
    if (pointer == nullptr) return;
    const auto active = pointer->shared_from_this();
    auto preparedView = status == metalrobo::MetalNeuronCultureStatus::success
        ? metalrobo::MetalNeuronCultureRuntimeBridgeAccess::preparedView(
            *active->runtime->culture)
        : metalrobo::MetalNeuronCultureAcceptedView{};
    mrnx_culture_accepted_view_v1 bridgeView{};
    const bool ready = status == metalrobo::MetalNeuronCultureStatus::success &&
        active->culturePrepared.source_root_fingerprint ==
            active->transactionFingerprint &&
        active->culturePrepared.receipt_fingerprint != 0u &&
        bridgeCultureAcceptedView(
            preparedView, active->runtime->device.registryID, bridgeView);
    if (ready) {
        bridgeView.source_root_fingerprint =
            active->culturePrepared.source_root_fingerprint;
        bridgeView.receipt_fingerprint =
            active->culturePrepared.receipt_fingerprint;
    }
    {
        const std::lock_guard lock(active->mutex);
        if (active->cultureSettled) return;
        active->cultureSettled = true;
        active->cultureReady = ready;
        if (ready) {
            active->cultureAcceptedView = std::move(preparedView);
            active->cultureAccepted = bridgeView;
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
    if (runtime->behavior != nullptr) {
        if (runtime->exactClock) {
            if (!runtime->publishedOnce &&
                root.committedTimestampMicroseconds !=
                    runtime->behaviorInitialTimestampNanoseconds) return false;
        } else if (
            root.committedTimestampMicroseconds >
                std::numeric_limits<std::uint64_t>::max() / 1000ull ||
            root.targetTimestampMicroseconds >
                std::numeric_limits<std::uint64_t>::max() / 1000ull ||
            (!runtime->publishedOnce &&
             root.committedTimestampMicroseconds * 1000ull !=
                 runtime->behaviorInitialTimestampNanoseconds)) return false;
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
    result->cultureSettled = runtime->culture == nullptr;
    result->cultureReady = runtime->culture == nullptr;
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
        request.motor_ready.value == 0u) return false;
    failureStage = 61u;
    if (request.motor_ready.device_registry_id != runtime->device.registryID ||
        !eventObject(request.motor_ready.shared_event, event) ||
        !importableSharedEvent(runtime->device, event)) return false;
    failureStage = 62u;
    if (candidate.motorOutputHeaderGPUAddress != result->motorHeader.address)
        return false;
    failureStage = 63u;
    if (candidate.muscleExcitationGPUAddress != result->excitation.address)
        return false;
    failureStage = 64u;
    if (candidate.autonomicCommandGPUAddress != result->autonomic.address)
        return false;
    failureStage = 65u;
    if (candidate.activeSensingCommandGPUAddress != result->activeSensing.address) {
        if (std::getenv("MRNX_PHYSICAL_DIAGNOSTICS") != nullptr) {
            std::fprintf(stderr, "mrnx_active_sensing candidate=%llu range=%llu\n",
                static_cast<unsigned long long>(candidate.activeSensingCommandGPUAddress),
                static_cast<unsigned long long>(result->activeSensing.address));
        }
        return false;
    }
    result->motorReadyEvent = event;
    result->transactionFingerprint = root.transactionFingerprint;
    result->brainGeneration = root.shadowGeneration;
    result->controlStep = root.controlStepIdentifier;
    failureStage = 7u;
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
                runtime->publishedControlStep ==
                    std::numeric_limits<std::uint64_t>::max() ||
                root.controlStepIdentifier != runtime->publishedControlStep + 1u) {
                return false;
            }
            result->previousTransactionFingerprint =
                runtime->publishedTransactionFingerprint;
            result->previousPhysicsGeneration =
                runtime->publishedPhysicsGeneration;
        } else if (root.baseBrainGeneration != 0u ||
                   root.basePhysicsGeneration != 0u ||
                   runtime->aggregate.publication_epoch != 0u ||
                   root.controlStepIdentifier != 1u) {
            return false;
        }
    }
    active = std::move(result);
    failureStage = 0u;
    return true;
}

} // namespace

#include "RuntimeHumanBehaviorAPI.inc"
