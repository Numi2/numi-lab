#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>

#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"
#include "metalrobo/MetalHybridRenderer.hpp"
#include "metalrobo/MetalMultiArticulatedContact.hpp"
#include "metalrobo/MultiArticulatedContact.hpp"
#include "metalrobo/MujocoMuscleReference.hpp"
#include "metalrobo/NumiHumanJointEquality.hpp"
#include "metalrobo/NumiHumanKnee.hpp"
#include "metalrobo/NumiHumanKneeContact.hpp"
#include "metalrobo/NumiHumanMuscleEquilibrium.hpp"
#include "metalrobo/NumiHumanTendon.hpp"
#include "metalrobo/NumiHumanTendonMetal.hpp"
#include "metalrobo/QualityContactSolver.hpp"
#include "metalrobo/VisualPlatform.hpp"
#include "metalrobo/WorldCompiler.hpp"
#include "numi/matter/matter.hpp"
#include "numi/matter/numi_human.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr std::array<char, 8u> kRigidMagic{
    'N', 'H', 'R', 'I', 'G', 'I', 'D', '2',
};
constexpr std::array<char, 8u> kLegacyMuscleMagic{
    'N', 'H', 'M', 'Y', 'O', '1', '\0', '\0',
};
constexpr std::array<char, 8u> kMuscleMagic{
    'N', 'H', 'M', 'Y', 'O', '2', '\0', '\0',
};
constexpr std::array<char, 8u> kSupportContactMagic{
    'N', 'H', 'C', 'N', 'T', '1', '\0', '\0',
};
constexpr std::uint32_t kPayloadAbi = 1u;
constexpr std::uint32_t kMusclePayloadAbi = 2u;
constexpr std::uint32_t kBodySemantic = 51001u;
constexpr std::uint32_t kSiteSemantic = 51002u;
constexpr std::uint32_t kRouteSemantic = 51003u;
constexpr std::uint32_t kBoneSemantic = 51004u;
constexpr std::uint32_t kMuscleSurfaceSemantic = 51005u;
constexpr std::uint32_t kTendonSurfaceSemantic = 51006u;
constexpr std::uint32_t kSkinShellSemantic = 51007u;
constexpr std::uint32_t kTendonAttachmentCollarSemantic = 51008u;
constexpr std::uint32_t kPassiveFEMTissueSemantic = 51009u;
constexpr std::uint32_t kOrganSurfaceSemantic = 51010u;
constexpr std::uint32_t kVesselSurfaceSemantic = 51011u;
constexpr std::uint32_t kNerveSurfaceSemantic = 51012u;
constexpr std::uint32_t kTendonAttachmentEnvelopeSemantic = 51013u;
constexpr std::uint32_t kPectoralisFasciaSemantic = 51014u;
constexpr std::uint32_t kKneeCartilageSemantic = 51015u;
constexpr std::uint32_t kKneeMeniscusSemantic = 51016u;
constexpr std::uint32_t kKneeLigamentSemantic = 51017u;
constexpr std::uint32_t kKneeTendonSemantic = 51018u;
constexpr std::array<char, 8u> kOpenKneeLigamentFEMMagicV1{
    'N', 'H', 'K', 'F', 'E', 'M', '1', '\0',
};
constexpr std::array<char, 8u> kOpenKneeLigamentFEMMagicV2{
    'N', 'H', 'K', 'F', 'E', 'M', '2', '\0',
};

struct OpenKneeLigamentFEMHeader {
    std::array<char, 8u> magic{};
    std::uint32_t abi = 0u;
    std::uint32_t side = 0u;
    std::uint32_t regionCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t poseKind = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
};

struct OpenKneeLigamentFEMRegion {
    std::array<char, 16u> name{};
    std::uint32_t payloadFirstNode = 0u;
    std::uint32_t snapshotFirstNode = 0u;
    std::uint32_t nodeCount = 0u;
};

struct OpenKneeLigamentFEMNode {
    float position[3]{};
};

struct LoadedOpenKneeLigamentFEM {
    OpenKneeLigamentFEMHeader header{};
    std::vector<mr_float4> worldNodes;
    std::vector<bool> deformedNodes;
    float maximumDisplacementMeters = 0.0f;
    float minimumDeterminant = 1.0f;
    float maximumDeterminant = 1.0f;
    std::array<double, 3u> bodyReactionL1Newtons{};
    std::array<double, 3u> maximumAnchorTargetResidualMeters{};
    std::vector<MRBodyStateGPU> projectedRestBodies;
    double qualificationFlexionRadians = 0.0;
    double maximumProjectedRestVisualCorrectionMeters = 0.0;
    double maximumProjectedRestReconstructionResidualMeters = 0.0;
    std::uint32_t quadricepsEndpointCount = 0u;
    std::uint32_t quadricepsLoadNodeCount = 0u;
    double quadricepsLoadPatchAreaSquareMeters = 0.0;
    std::uint32_t patellarTendonPatellaLoadNodeCount = 0u;
    std::uint32_t patellarTendonTibiaLoadNodeCount = 0u;
    double patellarTendonPatellaPatchAreaSquareMeters = 0.0;
    double patellarTendonTibiaPatchAreaSquareMeters = 0.0;
    double quadricepsForceOwnerFraction = 0.0;
    double quadricepsAppliedForceL1Newtons = 0.0;
    double quadricepsAppliedForceResultantNewtons = 0.0;
    double quadricepsEnthesisReactionResultantNewtons = 0.0;
    double patellarTendonForceL1Newtons = 0.0;
    double patellarTendonForceResultantNewtons = 0.0;
    double patellarTendonPatellaReactionResultantNewtons = 0.0;
    double patellarTendonTibiaReactionResultantNewtons = 0.0;
    double assembledExternalForceL1Newtons = 0.0;
    double assembledExternalForceResultantNewtons = 0.0;
    std::uint32_t articularPairCount = 0u;
    std::uint32_t articularContactSampleCount = 0u;
    std::uint32_t articularMechanicalSampleCount = 0u;
    std::uint32_t articularInternalSameBodySampleCount = 0u;
    std::uint32_t articularClosedSampleCount = 0u;
    double articularContactAreaSquareMeters = 0.0;
    double articularNormalForceNewtons = 0.0;
    double articularMaximumPressurePascals = 0.0;
    double articularBodyForceL1Newtons = 0.0;
    double articularForceResidualNewtons = 0.0;
    double articularMomentResidualNewtonMeters = 0.0;
    double articularStoredEnergyJoules = 0.0;
    double articularMaximumNormalStrain = 0.0;
    double articularMaximumClosureMeters = 0.0;
    std::uint32_t articularAuditedStepCount = 0u;
    std::uint32_t articularTrajectoryMinimumClosedSampleCount = 0u;
    std::uint32_t articularTrajectoryMaximumClosedSampleCount = 0u;
    double articularTrajectoryMinimumNormalForceNewtons = 0.0;
    double articularTrajectoryMaximumNormalForceNewtons = 0.0;
    double articularTrajectoryMaximumPressurePascals = 0.0;
    double articularTrajectoryMaximumStoredEnergyJoules = 0.0;
    double articularTrajectoryMaximumNormalStrain = 0.0;
    double articularTrajectoryMaximumClosureMeters = 0.0;
    double articularTrajectoryMaximumForceResidualNewtons = 0.0;
    double articularTrajectoryMaximumMomentResidualNewtonMeters = 0.0;
    bool activeQuadricepsTendonCoupling = false;
    bool liveHumanCoupling = false;
    bool rollbackVerified = false;
    bool replayVerified = false;
    std::string deviceName;
};
// The native exact-reference path is currently qualified through 640 px on
// the local Apple M4. Larger reference frames can return an all-background
// image despite successful Metal submission, so a valid native default is
// better than an unverified nominal 1024 px setting.
constexpr std::uint32_t kDefaultFrameDimension = 640u;
constexpr std::array<char, 8u> kBoneMagic{
    'N', 'H', 'B', 'O', 'N', 'E', 'S', '1',
};
constexpr std::uint32_t kBonePayloadAbi = 2u;
constexpr std::array<char, 8u> kSoftTissueMagic{
    'N', 'H', 'T', 'I', 'S', 'S', '2', '\0',
};
constexpr std::uint32_t kSoftTissuePayloadAbi = 3u;
constexpr std::array<char, 8u> kMultiBodySoftTissueMagic{
    'N', 'H', 'T', 'I', 'S', 'S', '3', '\0',
};
constexpr std::uint32_t kMultiBodySoftTissuePayloadAbi = 4u;
constexpr std::array<char, 8u> kPectoralisFasciaMagic{
    'N', 'H', 'F', 'A', 'S', 'C', '2', '\0',
};
constexpr std::uint32_t kPectoralisFasciaPayloadAbi = 2u;
constexpr std::array<char, 8u> kRouteSoftTissueMagic{
    'N', 'H', 'T', 'I', 'S', 'S', '4', '\0',
};
constexpr std::uint32_t kRouteSoftTissuePayloadAbi = 5u;
constexpr std::uint32_t kRouteSoftTissueMaximumBindings = 24u;
constexpr std::uint32_t kRouteSoftTissueMaximumInfluences = 4u;
constexpr std::uint32_t kSoftTissueLayerMuscle = 1u;
constexpr std::uint32_t kSoftTissueLayerTendon = 2u;
// Reserved for the narrowly scoped Z-Anatomy calcaneus overlay.  It stays in
// the existing NHTISS3 transport because it shares the same named articulated
// body binding format as soft tissue, but renders with osseous material and
// bone semantics.
constexpr std::uint32_t kSoftTissueLayerSupplementalBone = 3u;
constexpr std::array<char, 8u> kSkinMagic{
    'N', 'H', 'S', 'K', 'I', 'N', '1', '\0',
};
constexpr std::uint32_t kLegacySkinPayloadAbi = 1u;
constexpr std::uint32_t kBoundaryLocalSkinPayloadAbi = 2u;
constexpr std::uint32_t kSourceLocalSkinPayloadAbi = 3u;
constexpr std::uint32_t kSkinPayloadAbi = 4u;
constexpr std::array<char, 8u> kTorsoAnatomyMagic{
    'N', 'H', 'A', 'N', 'A', 'T', '1', '\0',
};
constexpr std::uint32_t kTorsoAnatomyPayloadAbi = 1u;
constexpr std::uint32_t kTorsoAnatomyLayerOrgan = 1u;
constexpr std::uint32_t kTorsoAnatomyLayerVessel = 2u;
constexpr std::uint32_t kTorsoAnatomyLayerNerve = 3u;

#pragma pack(push, 1)
struct RigidHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineAbi = 0u;
    std::uint32_t sourceBodyCount = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t jointCount = 0u;
    std::uint32_t nq = 0u;
    std::uint32_t nv = 0u;
    std::uint32_t rootBodyIndex = 0u;
    std::uint32_t virtualBodyCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
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
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t muscleCount = 0u;
    std::uint32_t siteCount = 0u;
    std::uint32_t wrapCount = 0u;
    std::uint32_t routeNodeCount = 0u;
    std::uint32_t sourceTendonCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
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
    std::uint32_t payloadAbi = 0u;
    std::uint32_t engineBodyCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
    float groundPointX = 0.0f;
    float groundPointY = 0.0f;
    float groundPointZ = 0.0f;
    float groundNormalX = 0.0f;
    float groundNormalY = 0.0f;
    float groundNormalZ = 1.0f;
    float groundFriction = 0.0f;
};

struct SupportContactRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t sourceGeometryIndex = MR_INVALID_INDEX;
    float localPointX = 0.0f;
    float localPointY = 0.0f;
    float localPointZ = 0.0f;
    float worldWitnessX = 0.0f;
    float worldWitnessY = 0.0f;
    float worldWitnessZ = 0.0f;
    float friction = 0.0f;
    float defaultSignedPlaneDistance = 0.0f;
    float reserved0 = 0.0f;
    float reserved1 = 0.0f;
};

struct BoneHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t boneCount = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct BoneRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t stableId = 0u;
    float translationX = 0.0f;
    float translationY = 0.0f;
    float translationZ = 0.0f;
    float quaternionX = 0.0f;
    float quaternionY = 0.0f;
    float quaternionZ = 0.0f;
    float quaternionW = 1.0f;
    float uniformScale = 1.0f;
};

struct BoneVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
};

struct SoftTissueHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t tissueCount = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t reserved0 = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct RouteSoftTissueHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t tissueCount = 0u;
    std::uint32_t bindingCount = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t registrationFingerprint = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct LegacySoftTissueRecord {
    std::uint32_t primaryBodyIndex = MR_INVALID_INDEX;
    std::uint32_t secondaryBodyIndex = MR_INVALID_INDEX;
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t stableId = 0u;
    std::uint32_t layer = 0u;
    float primaryTranslationX = 0.0f;
    float primaryTranslationY = 0.0f;
    float primaryTranslationZ = 0.0f;
    float primaryQuaternionX = 0.0f;
    float primaryQuaternionY = 0.0f;
    float primaryQuaternionZ = 0.0f;
    float primaryQuaternionW = 1.0f;
    float primaryUniformScale = 1.0f;
    float secondaryTranslationX = 0.0f;
    float secondaryTranslationY = 0.0f;
    float secondaryTranslationZ = 0.0f;
    float secondaryQuaternionX = 0.0f;
    float secondaryQuaternionY = 0.0f;
    float secondaryQuaternionZ = 0.0f;
    float secondaryQuaternionW = 1.0f;
    float secondaryUniformScale = 1.0f;
};

struct LegacySoftTissueVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
    float primaryWeight = 1.0f;
};

// ``NHTISS3`` makes the anatomical ownership of shared tendons explicit.  Its
// third binding is not an artistic extra: the Achilles surface must follow
// femur-driven gastrocnemius, tibia-driven soleus, and its calcaneal insertion
// instead of being falsely skinned only tibia-to-calcaneus.
struct SoftTissueBodyBinding {
    float translation[3]{};
    float quaternion[4]{0.0f, 0.0f, 0.0f, 1.0f};
    float uniformScale = 1.0f;
};

struct MultiBodySoftTissueRecord {
    std::uint32_t bodyIndex[3]{
        MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t stableId = 0u;
    std::uint32_t layer = 0u;
    std::uint32_t reserved0 = 0u;
    SoftTissueBodyBinding binding[3]{};
};

struct MultiBodySoftTissueVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
    float weight[3]{1.0f, 0.0f, 0.0f};
};

struct RouteSoftTissueRecord {
    std::uint32_t firstBinding = 0u;
    std::uint32_t bindingCount = 0u;
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t stableId = 0u;
    std::uint32_t layer = 0u;
};

struct RouteSoftTissueBinding {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    SoftTissueBodyBinding transform{};
};

struct RouteSoftTissueVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
    std::uint32_t bindingIndex[kRouteSoftTissueMaximumInfluences]{
        0u, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    float weight[kRouteSoftTissueMaximumInfluences]{1.0f, 0.0f, 0.0f, 0.0f};
};

// Normalized in-memory representation shared by NHTISS2/3/4. A record may
// retain every authored route body, while each vertex remains a compact
// four-influence skinning tuple.
struct SoftTissueRecord {
    std::uint32_t bodyIndex[kRouteSoftTissueMaximumBindings]{};
    std::uint32_t bindingCount = 0u;
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t stableId = 0u;
    std::uint32_t layer = 0u;
    SoftTissueBodyBinding binding[kRouteSoftTissueMaximumBindings]{};
};

struct SoftTissueVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
    std::uint32_t bindingIndex[kRouteSoftTissueMaximumInfluences]{
        0u, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    float weight[kRouteSoftTissueMaximumInfluences]{1.0f, 0.0f, 0.0f, 0.0f};
};

struct PectoralisFasciaHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t regionCount = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    float thicknessMeters = 0.0f;
    float muscleLoadFraction = 0.0f;
    std::uint32_t presentationTriangleCount = 0u;
    std::uint32_t reserved1 = 0u;
    std::array<std::uint8_t, 32u> bodypartsArchiveSha256{};
    std::array<std::uint8_t, 32u> myosimManifestSha256{};
};

struct PectoralisFasciaRegion {
    std::array<char, 8u> memberId{};
    std::uint32_t muscleIndex = MR_INVALID_INDEX;
    std::uint32_t firstNode = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t firstTetrahedron = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t softTissueStableId = 0u;
};

struct PectoralisFasciaNode {
    float sourcePosition[3]{};
    float compiledMassKg = 0.0f;
    std::uint32_t flags = 0u;
    std::uint32_t regionIndex = MR_INVALID_INDEX;
    std::uint32_t sourceVertexIndex = MR_INVALID_INDEX;
    std::uint32_t reserved0 = 0u;
};

struct PectoralisFasciaTetrahedron {
    std::uint32_t node[4]{};
    std::uint32_t regionIndex = MR_INVALID_INDEX;
};

struct PectoralisFasciaPresentationTriangle {
    std::uint32_t regionIndex = MR_INVALID_INDEX;
    std::uint32_t sourceVertex[3]{};
};

struct SkinHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t bindingCount = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t registrationFingerprint = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct SkinBindingRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    float translationX = 0.0f;
    float translationY = 0.0f;
    float translationZ = 0.0f;
    float quaternionX = 0.0f;
    float quaternionY = 0.0f;
    float quaternionZ = 0.0f;
    float quaternionW = 1.0f;
    float uniformScale = 1.0f;
};

struct SkinVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
    std::uint32_t bindingIndex[4]{
        MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    float weight[4]{};
};

// ``NHANAT1`` is deliberately a separate source-surface payload.  Each
// selected organ, vessel, or neural component is already expressed in one
// registered MyoSim inertial link frame by the offline importer.  It therefore
// follows that link in the native renderer without pretending to supply a
// continuum or material model.
struct TorsoAnatomyHeader {
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    std::uint32_t surfaceCount = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t registrationFingerprint = 0u;
    std::array<std::uint8_t, 32u> sourceSha256{};
};

struct TorsoAnatomyRecord {
    std::uint32_t bodyIndex = MR_INVALID_INDEX;
    std::uint32_t firstVertex = 0u;
    std::uint32_t vertexCount = 0u;
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    std::uint32_t stableId = 0u;
    std::uint32_t layer = 0u;
    std::uint32_t reserved0 = 0u;
};

struct TorsoAnatomyVertex {
    float positionX = 0.0f;
    float positionY = 0.0f;
    float positionZ = 0.0f;
    float normalX = 0.0f;
    float normalY = 0.0f;
    float normalZ = 1.0f;
};

struct LoadedMuscles {
    MuscleHeader header{};
    std::vector<SiteRecord> sites;
    std::vector<WrapRecord> wraps;
    std::vector<RouteRecord> routes;
    std::vector<MuscleRecord> muscles;
    std::vector<MuscleArchitectureRecord> architectures;
    std::vector<metalrobo::MujocoMuscleSite> referenceSites;
    std::vector<metalrobo::MujocoWrapGeometry> referenceWraps;
    std::vector<metalrobo::MujocoMuscleDefinition> referenceMuscles;
    std::vector<metalrobo::MujocoCompliantMuscleArchitecture>
        referenceArchitectures;
    // Immutable, source-faithful MyoSim program for the articulated Metal
    // operator. Mutable excitation/activation state is supplied separately
    // by each bounded visual transaction.
    std::vector<MRMujocoMuscleSiteGPU> gpuSites;
    std::vector<MRMujocoMuscleWrapGPU> gpuWraps;
    std::vector<MRMujocoMuscleRouteNodeGPU> gpuRoutes;
    std::vector<MRMujocoMuscleGPU> gpuMuscles;
    std::uint32_t tendonPointBindings = 0u;
    std::uint32_t tendonTriangleBindings = 0u;
    std::uint32_t tendonEnvelopeBindings = 0u;
    std::uint32_t tendonMigratedEnvelopeBindings = 0u;
    double maximumTendonReferencePathDelta = 0.0;
    double maximumTendonArchitectureScaleChange = 0.0;
    metalrobo::NumiHumanTendonPayload tendonPayload;
};

struct LoadedBones {
    BoneHeader header{};
    std::vector<BoneRecord> records;
    std::vector<BoneVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct LoadedSoftTissues {
    SoftTissueHeader header{};
    std::vector<SoftTissueRecord> records;
    std::vector<SoftTissueVertex> vertices;
    std::vector<std::uint32_t> indices;
    bool usesRouteBodySparseWeights = false;
};

struct LoadedPectoralisFascia {
    PectoralisFasciaHeader header{};
    std::vector<PectoralisFasciaRegion> regions;
    std::vector<PectoralisFasciaNode> nodes;
    std::vector<PectoralisFasciaTetrahedron> tetrahedra;
    std::vector<PectoralisFasciaPresentationTriangle> presentationTriangles;
};

struct LoadedSkin {
    SkinHeader header{};
    std::vector<SkinBindingRecord> bindings;
    std::vector<SkinVertex> vertices;
    std::vector<std::uint32_t> indices;
    bool usesBoundaryLocalWeights = false;
    bool usesSourceSurfaceLocalWeights = false;
    bool usesWorldRestNormals = false;
};

struct LoadedTorsoAnatomy {
    TorsoAnatomyHeader header{};
    std::vector<TorsoAnatomyRecord> records;
    std::vector<TorsoAnatomyVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct LoadedSupportContacts {
    SupportContactHeader header{};
    std::vector<SupportContactRecord> records;
};

struct LoadedJointEqualities {
    metalrobo::NumiHumanJointEqualityPayload payload;
};
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(MuscleHeader) == 76u);
static_assert(sizeof(SiteRecord) == 16u);
static_assert(sizeof(WrapRecord) == 64u);
static_assert(sizeof(RouteRecord) == 16u);
static_assert(sizeof(MuscleRecord) == 164u);
static_assert(sizeof(MuscleArchitectureRecord) == 32u);
static_assert(sizeof(SupportContactHeader) == 84u);
static_assert(sizeof(SupportContactRecord) == 48u);
static_assert(sizeof(BoneHeader) == 60u);
static_assert(sizeof(BoneRecord) == 56u);
static_assert(sizeof(BoneVertex) == 24u);
static_assert(sizeof(SoftTissueHeader) == 60u);
static_assert(sizeof(RouteSoftTissueHeader) == 64u);
static_assert(sizeof(LegacySoftTissueRecord) == 96u);
static_assert(sizeof(LegacySoftTissueVertex) == 28u);
static_assert(sizeof(SoftTissueBodyBinding) == 32u);
static_assert(sizeof(MultiBodySoftTissueRecord) == 136u);
static_assert(sizeof(MultiBodySoftTissueVertex) == 36u);
static_assert(sizeof(RouteSoftTissueRecord) == 32u);
static_assert(sizeof(RouteSoftTissueBinding) == 36u);
static_assert(sizeof(RouteSoftTissueVertex) == 56u);
static_assert(sizeof(PectoralisFasciaHeader) == 104u);
static_assert(sizeof(PectoralisFasciaRegion) == 32u);
static_assert(sizeof(PectoralisFasciaNode) == 32u);
static_assert(sizeof(PectoralisFasciaTetrahedron) == 20u);
static_assert(sizeof(SkinHeader) == 60u);
static_assert(sizeof(SkinBindingRecord) == 36u);
static_assert(sizeof(SkinVertex) == 56u);
static_assert(sizeof(TorsoAnatomyHeader) == 60u);
static_assert(sizeof(TorsoAnatomyRecord) == 32u);
static_assert(sizeof(TorsoAnatomyVertex) == 24u);
static_assert(sizeof(OpenKneeLigamentFEMHeader) == 40u);
static_assert(sizeof(OpenKneeLigamentFEMRegion) == 28u);
static_assert(sizeof(OpenKneeLigamentFEMNode) == 12u);

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

metalrobo::MujocoRouteNodeType referenceRouteType(const std::uint32_t type) {
    switch (type) {
    case 1u: return metalrobo::MujocoRouteNodeType::site;
    case 2u: return metalrobo::MujocoRouteNodeType::sphere;
    case 3u: return metalrobo::MujocoRouteNodeType::cylinder;
    default: throw std::runtime_error("MyoSim route type is invalid");
    }
}

metalrobo::MujocoRouteNodeType referenceWrapType(const std::uint32_t type) {
    return referenceRouteType(type);
}

template <typename T>
void readObject(std::istream& input, T& value, const char* description) {
    static_assert(std::is_trivially_copyable_v<T>);
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    require(input.good(), std::string("truncated ") + description);
}

template <typename T>
std::vector<T> readVector(
    std::istream& input,
    const std::size_t count,
    const char* description
) {
    std::vector<T> result(count);
    if (!result.empty()) {
        input.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(result.size() * sizeof(T))
        );
        require(input.good(), std::string("truncated ") + description);
    }
    return result;
}

struct LoadedRigid {
    RigidHeader header{};
    metalrobo::EngineModel model;
};

LoadedRigid loadRigid(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open MyoSim rigid payload " + path.string());
    LoadedRigid result;
    readObject(input, result.header, "MyoSim rigid header");
    require(result.header.magic == kRigidMagic, "rigid payload magic is not NHRIGID2");
    require(result.header.payloadAbi == kPayloadAbi, "unsupported MyoSim rigid ABI");
    require(result.header.engineAbi == MR_ENGINE_ABI_VERSION, "MyoSim rigid/Core ABI mismatch");
    require(result.header.reserved0 == 0u && result.header.rootBodyIndex == 0u,
            "MyoSim rigid header is malformed");
    require(result.header.sourceBodyCount > 0u &&
                result.header.engineBodyCount >= result.header.sourceBodyCount &&
                result.header.jointCount + 1u == result.header.engineBodyCount &&
                result.header.nq == result.header.nv + 1u,
            "MyoSim rigid dimensions are malformed");
    result.model.name = "numilab_human_myosim_native_visual";
    readObject(input, result.model.world, "MyoSim world");
    MRArticulationGPU articulation{};
    readObject(input, articulation, "MyoSim articulation");
    result.model.articulations.push_back(articulation);
    result.model.bodies = readVector<MRBodyPropertiesGPU>(
        input, result.header.engineBodyCount, "MyoSim bodies"
    );
    result.model.joints = readVector<MRJointDescriptorGPU>(
        input, result.header.jointCount, "MyoSim joints"
    );
    result.model.dofs = readVector<MRDofPropertiesGPU>(
        input, result.header.nv, "MyoSim DoFs"
    );
    result.model.defaultQ = readVector<float>(
        input, result.header.nq, "MyoSim default q"
    );
    result.model.defaultV = readVector<float>(
        input, result.header.nv, "MyoSim default v"
    );
    const auto sourceToCore = readVector<std::uint32_t>(
        input, result.header.sourceBodyCount, "MyoSim source map"
    );
    (void)sourceToCore;
    const auto sourcePoses = readVector<SourcePoseRecord>(
        input, result.header.sourceBodyCount, "MyoSim source poses"
    );
    (void)sourcePoses;
    require(input.peek() == std::char_traits<char>::eof(),
            "MyoSim rigid payload has trailing bytes");
    require(result.model.world.bodyCount == result.header.engineBodyCount &&
                articulation.rootType == MR_ROOT_FLOATING &&
                articulation.bodyCount == result.header.engineBodyCount &&
                articulation.nq == result.header.nq && articulation.nv == result.header.nv,
            "MyoSim rigid world/header disagreement");
    std::string reason;
    require(result.model.valid(&reason), "MyoSim Core model invalid: " + reason);
    return result;
}

LoadedMuscles loadMuscles(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open MyoSim muscle payload " + path.string());
    LoadedMuscles result;
    readObject(input, result.header, "MyoSim muscle header");
    const bool legacy = result.header.magic == kLegacyMuscleMagic &&
        result.header.payloadAbi == kPayloadAbi &&
        result.header.reserved0 == 0u && result.header.reserved1 == 0u;
    const bool compliant = result.header.magic == kMuscleMagic &&
        result.header.payloadAbi == kMusclePayloadAbi &&
        result.header.reserved0 == result.header.muscleCount &&
        result.header.reserved1 == sizeof(MuscleArchitectureRecord);
    require((legacy || compliant) &&
                result.header.engineBodyCount == rigid.engineBodyCount &&
                result.header.sourceSha256 == rigid.sourceSha256,
            "MyoSim muscle payload/header disagreement");
    result.sites = readVector<SiteRecord>(
        input, result.header.siteCount, "MyoSim sites"
    );
    result.wraps = readVector<WrapRecord>(input, result.header.wrapCount, "MyoSim wraps");
    result.routes = readVector<RouteRecord>(input, result.header.routeNodeCount, "MyoSim routes");
    result.muscles = readVector<MuscleRecord>(input, result.header.muscleCount, "MyoSim muscles");
    result.architectures = compliant
        ? readVector<MuscleArchitectureRecord>(
            input, result.header.muscleCount, "MyoSim compliant architectures"
        )
        : std::vector<MuscleArchitectureRecord>(result.header.muscleCount);
    require(input.peek() == std::char_traits<char>::eof(),
            "MyoSim muscle payload has trailing bytes");
    for (const SiteRecord& site : result.sites) {
        require(site.bodyIndex < rigid.engineBodyCount,
                "MyoSim site body index is out of bounds");
    }
    for (const WrapRecord& wrap : result.wraps) {
        require(wrap.bodyIndex < rigid.engineBodyCount,
                "MyoSim wrap body index is out of bounds");
    }
    for (const RouteRecord& route : result.routes) {
        require(route.reserved0 == 0u && route.type >= 1u && route.type <= 3u,
                "MyoSim route record is malformed");
        const std::size_t targetCount = route.type == 1u
            ? result.sites.size() : result.wraps.size();
        require(route.targetIndex < targetCount,
                "MyoSim route target is out of bounds");
        require(route.sideSiteIndex == MR_INVALID_INDEX ||
                    route.sideSiteIndex < result.sites.size(),
                "MyoSim route side site is out of bounds");
    }
    for (std::size_t muscleIndex = 0u; muscleIndex < result.muscles.size(); ++muscleIndex) {
        const MuscleRecord& muscle = result.muscles[muscleIndex];
        require(muscle.reserved0 == 0u &&
                    muscle.routeOffset <= result.routes.size() &&
                    muscle.routeCount <= result.routes.size() - muscle.routeOffset,
                "MyoSim muscle route range is invalid");
    }
    result.referenceSites.reserve(result.sites.size());
    result.gpuSites.reserve(result.sites.size());
    for (const SiteRecord& site : result.sites) {
        result.referenceSites.push_back({site.bodyIndex, {site.x, site.y, site.z}});
        MRMujocoMuscleSiteGPU gpuSite{};
        gpuSite.bodyIndex = site.bodyIndex;
        gpuSite.localPoint = {site.x, site.y, site.z, 0.0f};
        result.gpuSites.push_back(gpuSite);
    }
    result.referenceWraps.reserve(result.wraps.size());
    result.gpuWraps.reserve(result.wraps.size());
    for (const WrapRecord& wrap : result.wraps) {
        result.referenceWraps.push_back({
            wrap.bodyIndex, referenceWrapType(wrap.type),
            {wrap.centerX, wrap.centerY, wrap.centerZ},
            {wrap.rotation[0], wrap.rotation[1], wrap.rotation[2],
             wrap.rotation[3], wrap.rotation[4], wrap.rotation[5],
             wrap.rotation[6], wrap.rotation[7], wrap.rotation[8]},
            wrap.radius,
        });
        MRMujocoMuscleWrapGPU gpuWrap{};
        gpuWrap.bodyIndex = wrap.bodyIndex;
        gpuWrap.type = wrap.type;
        gpuWrap.localCenter = {
            wrap.centerX, wrap.centerY, wrap.centerZ, 0.0f,
        };
        gpuWrap.rotationRow0 = {
            wrap.rotation[0], wrap.rotation[1], wrap.rotation[2], 0.0f,
        };
        gpuWrap.rotationRow1 = {
            wrap.rotation[3], wrap.rotation[4], wrap.rotation[5], 0.0f,
        };
        gpuWrap.rotationRow2 = {
            wrap.rotation[6], wrap.rotation[7], wrap.rotation[8], 0.0f,
        };
        gpuWrap.radius = {wrap.radius, 0.0f, 0.0f, 0.0f};
        result.gpuWraps.push_back(gpuWrap);
    }
    result.gpuRoutes.reserve(result.routes.size());
    for (const RouteRecord& route : result.routes) {
        result.gpuRoutes.push_back({
            route.type,
            route.targetIndex,
            route.sideSiteIndex,
            0u,
        });
    }
    result.referenceMuscles.reserve(result.muscles.size());
    result.referenceArchitectures.reserve(result.muscles.size());
    result.gpuMuscles.reserve(result.muscles.size());
    for (std::size_t muscleIndex = 0u; muscleIndex < result.muscles.size(); ++muscleIndex) {
        const MuscleRecord& muscle = result.muscles[muscleIndex];
        const MuscleArchitectureRecord& architecture =
            result.architectures[muscleIndex];
        metalrobo::MujocoMuscleDefinition definition;
        definition.route.reserve(muscle.routeCount);
        for (std::uint32_t offset = 0u; offset < muscle.routeCount; ++offset) {
            const RouteRecord& route = result.routes[muscle.routeOffset + offset];
            definition.route.push_back({
                referenceRouteType(route.type), route.targetIndex, route.sideSiteIndex,
            });
        }
        definition.lengthRange = {muscle.values[0], muscle.values[1]};
        definition.accelerationScale = muscle.values[2];
        definition.controlRange = {muscle.values[3], muscle.values[4]};
        for (std::size_t parameter = 0u; parameter < 10u; ++parameter) {
            definition.gainParameters[parameter] = muscle.values[5u + parameter];
            definition.biasParameters[parameter] = muscle.values[15u + parameter];
            definition.dynamicParameters[parameter] = muscle.values[25u + parameter];
        }
        result.referenceMuscles.push_back(std::move(definition));
        result.referenceArchitectures.push_back({
            architecture.optimalFiberLength,
            architecture.tendonSlackLength,
            architecture.tendonStrainAtOneNormalizedForce,
            architecture.tendonStiffnessAtOneNormalizedForce,
            architecture.tendonNormalizedForceAtToeEnd,
            architecture.tendonCurviness,
            architecture.normalizedFiberDamping,
            architecture.fitNormalizedRmse,
        });
        MRMujocoMuscleGPU gpuMuscle{};
        gpuMuscle.route = {
            muscle.routeOffset,
            muscle.routeCount,
            0u,
            0u,
        };
        gpuMuscle.lengthRangeAndAcceleration = {
            muscle.values[0],
            muscle.values[1],
            muscle.values[2],
            0.0f,
        };
        gpuMuscle.controlRange = {
            muscle.values[3],
            muscle.values[4],
            0.0f,
            0.0f,
        };
        for (std::size_t parameter = 0u; parameter < 10u; ++parameter) {
            (&gpuMuscle.gainParameters[parameter / 4u].x)[parameter % 4u] =
                muscle.values[5u + parameter];
            (&gpuMuscle.biasParameters[parameter / 4u].x)[parameter % 4u] =
                muscle.values[15u + parameter];
            (&gpuMuscle.dynamicParameters[parameter / 4u].x)[parameter % 4u] =
                muscle.values[25u + parameter];
        }
        gpuMuscle.compliantArchitecture0 = {
            architecture.optimalFiberLength,
            architecture.tendonSlackLength,
            architecture.tendonStrainAtOneNormalizedForce,
            architecture.tendonStiffnessAtOneNormalizedForce,
        };
        gpuMuscle.compliantArchitecture1 = {
            architecture.tendonNormalizedForceAtToeEnd,
            architecture.tendonCurviness,
            architecture.normalizedFiberDamping,
            architecture.fitNormalizedRmse,
        };
        result.gpuMuscles.push_back(gpuMuscle);
    }
    require(
        result.gpuSites.size() == result.sites.size() &&
            result.gpuWraps.size() == result.wraps.size() &&
            result.gpuRoutes.size() == result.routes.size() &&
            result.gpuMuscles.size() == result.muscles.size() &&
            result.referenceArchitectures.size() == result.muscles.size(),
        "MyoSim Metal source program packing is incomplete"
    );
    return result;
}

void applyNumiHumanTendonPayload(
    const std::filesystem::path& path,
    const LoadedRigid& rigid,
    LoadedMuscles& muscles
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.is_open(), "cannot open NHTENDON payload " + path.string());
    const std::streamsize size = input.tellg();
    require(size >= 0, "cannot determine NHTENDON payload size");
    input.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), size);
        require(input.good(), "truncated NHTENDON payload");
    }
    metalrobo::NumiHumanTendonPayload payload;
    const auto decode = metalrobo::decodeNumiHumanTendonPayload(
        bytes, rigid.header.sourceSha256, {}, payload
    );
    require(
        decode.succeeded() && payload.bodyCount == rigid.header.engineBodyCount,
        std::string("invalid NHTENDON visual program: ") +
            metalrobo::numiHumanTendonStatusName(decode.status)
    );
    const auto sourceSites = muscles.referenceSites;
    const auto sourceMuscles = muscles.referenceMuscles;
    metalrobo::NumiHumanTendonResolvedProgram resolved;
    const auto diagnostics = metalrobo::resolveNumiHumanTendonProgram(
        payload, muscles.referenceSites, muscles.referenceMuscles, resolved
    );
    require(
        diagnostics.succeeded(),
        std::string("cannot resolve NHTENDON visual program: ") +
            metalrobo::numiHumanTendonStatusName(diagnostics.status)
    );
    if (resolved.migratedEnvelopeBindingCount > 0u) {
        std::vector<double> referenceQ(
            rigid.model.defaultQ.begin(), rigid.model.defaultQ.end()
        );
        metalrobo::NumiHumanTendonReferenceCalibration calibration;
        const auto calibrationDiagnostics =
            metalrobo::calibrateNumiHumanMigratedTendonReference(
                rigid.model, 0u, referenceQ, muscles.referenceWraps,
                sourceSites, sourceMuscles, resolved.sites, resolved.muscles,
                muscles.referenceArchitectures, payload, calibration
            );
        require(
            calibrationDiagnostics.succeeded(),
            std::string("NHTENDON visual reference calibration failed: ") +
                metalrobo::numiHumanTendonStatusName(calibrationDiagnostics.status)
        );
        muscles.maximumTendonReferencePathDelta =
            calibration.maximumAbsolutePathLengthDelta;
        muscles.maximumTendonArchitectureScaleChange =
            calibration.maximumArchitectureScaleChange;
    }
    muscles.referenceSites = std::move(resolved.sites);
    muscles.referenceMuscles = std::move(resolved.muscles);
    muscles.tendonPointBindings = resolved.pointBindingCount;
    muscles.tendonTriangleBindings = resolved.triangleBindingCount;
    muscles.tendonEnvelopeBindings = resolved.envelopeBindingCount;
    muscles.tendonMigratedEnvelopeBindings = resolved.migratedEnvelopeBindingCount;
    muscles.gpuSites.clear();
    muscles.gpuSites.reserve(muscles.referenceSites.size());
    for (const metalrobo::MujocoMuscleSite& site : muscles.referenceSites) {
        MRMujocoMuscleSiteGPU gpu{};
        gpu.bodyIndex = site.bodyIndex;
        gpu.localPoint = {
            static_cast<float>(site.localPoint[0]), static_cast<float>(site.localPoint[1]),
            static_cast<float>(site.localPoint[2]), 0.0f,
        };
        muscles.gpuSites.push_back(gpu);
    }
    muscles.gpuRoutes.clear();
    for (std::size_t index = 0u; index < muscles.referenceMuscles.size(); ++index) {
        muscles.gpuMuscles[index].route.x = static_cast<std::uint32_t>(muscles.gpuRoutes.size());
        muscles.gpuMuscles[index].route.y = static_cast<std::uint32_t>(
            muscles.referenceMuscles[index].route.size()
        );
        muscles.gpuMuscles[index].lengthRangeAndAcceleration.x = static_cast<float>(
            muscles.referenceMuscles[index].lengthRange[0]
        );
        muscles.gpuMuscles[index].lengthRangeAndAcceleration.y = static_cast<float>(
            muscles.referenceMuscles[index].lengthRange[1]
        );
        muscles.gpuMuscles[index].compliantArchitecture0.x = static_cast<float>(
            muscles.referenceArchitectures[index].optimalFiberLength
        );
        muscles.gpuMuscles[index].compliantArchitecture0.y = static_cast<float>(
            muscles.referenceArchitectures[index].tendonSlackLength
        );
        for (const metalrobo::MujocoRouteNode& node : muscles.referenceMuscles[index].route) {
            muscles.gpuRoutes.push_back({
                static_cast<std::uint32_t>(node.type), node.targetIndex,
                node.sideSiteIndex, 0u,
            });
        }
    }
    muscles.tendonPayload = std::move(payload);
}

LoadedSupportContacts loadSupportContacts(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open MyoSim support-contact payload " + path.string());
    LoadedSupportContacts result;
    readObject(input, result.header, "MyoSim support-contact header");
    require(result.header.magic == kSupportContactMagic &&
                result.header.payloadAbi == kPayloadAbi &&
                result.header.engineBodyCount == rigid.engineBodyCount &&
                result.header.sourceSha256 == rigid.sourceSha256 &&
                result.header.reserved0 == 0u &&
                result.header.contactCount >= 2u &&
                result.header.contactCount <= 32u,
            "MyoSim support-contact payload/header disagreement");
    result.records = readVector<SupportContactRecord>(
        input, result.header.contactCount, "MyoSim support-contact records"
    );
    require(input.peek() == std::char_traits<char>::eof(),
            "MyoSim support-contact payload has trailing bytes");
    const std::array<double, 3u> groundPoint{
        result.header.groundPointX,
        result.header.groundPointY,
        result.header.groundPointZ,
    };
    const std::array<double, 3u> groundNormal{
        result.header.groundNormalX,
        result.header.groundNormalY,
        result.header.groundNormalZ,
    };
    const double normalLength = std::sqrt(
        groundNormal[0] * groundNormal[0] +
        groundNormal[1] * groundNormal[1] +
        groundNormal[2] * groundNormal[2]
    );
    require(std::isfinite(groundPoint[0]) && std::isfinite(groundPoint[1]) &&
                std::isfinite(groundPoint[2]) && std::isfinite(normalLength) &&
                std::abs(normalLength - 1.0) <= 2.0e-4 &&
                std::isfinite(result.header.groundFriction) &&
                result.header.groundFriction >= 0.0f,
            "MyoSim support-contact ground plane is malformed");
    std::vector<std::uint32_t> sourceGeometryIds;
    sourceGeometryIds.reserve(result.records.size());
    for (const SupportContactRecord& record : result.records) {
        const std::array<double, 3u> localPoint{
            record.localPointX, record.localPointY, record.localPointZ,
        };
        const std::array<double, 3u> witness{
            record.worldWitnessX, record.worldWitnessY, record.worldWitnessZ,
        };
        const double witnessPlaneDistance =
            (witness[0] - groundPoint[0]) * groundNormal[0] +
            (witness[1] - groundPoint[1]) * groundNormal[1] +
            (witness[2] - groundPoint[2]) * groundNormal[2];
        require(record.bodyIndex < rigid.engineBodyCount &&
                    record.sourceGeometryIndex != MR_INVALID_INDEX &&
                    std::all_of(localPoint.begin(), localPoint.end(), [](const double value) {
                        return std::isfinite(value);
                    }) &&
                    std::all_of(witness.begin(), witness.end(), [](const double value) {
                        return std::isfinite(value);
                    }) &&
                    std::isfinite(record.friction) && record.friction >= 0.0f &&
                    std::isfinite(record.defaultSignedPlaneDistance) &&
                    std::abs(witnessPlaneDistance) <= 2.0e-4 &&
                    record.reserved0 == 0.0f && record.reserved1 == 0.0f,
                "MyoSim support-contact record is malformed");
        sourceGeometryIds.push_back(record.sourceGeometryIndex);
    }
    std::sort(sourceGeometryIds.begin(), sourceGeometryIds.end());
    require(std::adjacent_find(sourceGeometryIds.begin(), sourceGeometryIds.end()) ==
                sourceGeometryIds.end(),
            "MyoSim support-contact geometry identity is duplicated");
    return result;
}

LoadedJointEqualities loadJointEqualities(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.is_open(), "cannot open NHEQ joint-equality payload " +
                path.string());
    const std::streamsize size = input.tellg();
    require(size >= 0, "cannot determine NHEQ joint-equality payload size");
    input.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()), size);
        require(input.good(), "truncated NHEQ joint-equality payload");
    }
    LoadedJointEqualities result;
    const auto diagnostics = metalrobo::decodeNumiHumanJointEqualityPayload(
        bytes, rigid.sourceSha256, result.payload
    );
    require(
        diagnostics.succeeded() && result.payload.nq == rigid.nq &&
            result.payload.nv == rigid.nv,
        std::string("invalid NHEQ joint-equality payload: ") +
            metalrobo::numiHumanJointEqualityStatusName(diagnostics.status)
    );
    return result;
}

LoadedBones loadBones(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open BodyParts3D bone payload " + path.string());
    LoadedBones result;
    readObject(input, result.header, "BodyParts3D bone header");
    require(result.header.magic == kBoneMagic &&
                result.header.payloadAbi == kBonePayloadAbi &&
                result.header.reserved0 != 0u &&
                result.header.sourceSha256 == rigid.sourceSha256 &&
                result.header.boneCount > 0u &&
                result.header.vertexCount > 0u &&
                result.header.indexCount > 0u &&
                result.header.indexCount % 3u == 0u &&
                result.header.boneCount <= 256u &&
                result.header.vertexCount <= 4'000'000u &&
                result.header.indexCount <= 24'000'000u,
            "BodyParts3D bone payload/header disagreement");
    result.records = readVector<BoneRecord>(
        input, result.header.boneCount, "BodyParts3D bone records"
    );
    result.vertices = readVector<BoneVertex>(
        input, result.header.vertexCount, "BodyParts3D bone vertices"
    );
    result.indices = readVector<std::uint32_t>(
        input, result.header.indexCount, "BodyParts3D bone indices"
    );
    require(input.peek() == std::char_traits<char>::eof(),
            "BodyParts3D bone payload has trailing bytes");
    for (const BoneVertex& vertex : result.vertices) {
        const float normalLength = std::sqrt(
            vertex.normalX * vertex.normalX +
            vertex.normalY * vertex.normalY +
            vertex.normalZ * vertex.normalZ
        );
        require(std::isfinite(vertex.positionX) && std::isfinite(vertex.positionY) &&
                    std::isfinite(vertex.positionZ) && std::isfinite(normalLength) &&
                    std::abs(normalLength - 1.0f) <= 2.0e-3f,
                "BodyParts3D bone vertex is malformed");
    }
    std::vector<bool> stableIds(result.records.size() + 1u, false);
    for (const BoneRecord& record : result.records) {
        const float orientationLength = std::sqrt(
            record.quaternionX * record.quaternionX + record.quaternionY * record.quaternionY +
            record.quaternionZ * record.quaternionZ + record.quaternionW * record.quaternionW
        );
        require(record.bodyIndex < rigid.engineBodyCount && record.vertexCount > 0u &&
                    record.indexCount > 0u && record.indexCount % 3u == 0u &&
                    record.firstVertex <= result.vertices.size() &&
                    record.vertexCount <= result.vertices.size() - record.firstVertex &&
                    record.firstIndex <= result.indices.size() &&
                    record.indexCount <= result.indices.size() - record.firstIndex &&
                    record.stableId > 0u && record.stableId < stableIds.size() &&
                    !stableIds[record.stableId] && std::isfinite(record.translationX) &&
                    std::isfinite(record.translationY) && std::isfinite(record.translationZ) &&
                    std::isfinite(record.uniformScale) && record.uniformScale > 0.0f &&
                    std::isfinite(orientationLength) &&
                    std::abs(orientationLength - 1.0f) <= 2.0e-3f,
                "BodyParts3D bone record is malformed");
        stableIds[record.stableId] = true;
        for (std::uint32_t offset = 0u; offset < record.indexCount; ++offset) {
            const std::uint32_t index = result.indices[record.firstIndex + offset];
            require(index >= record.firstVertex && index < record.firstVertex + record.vertexCount,
                    "BodyParts3D bone index escapes its source mesh");
        }
    }
    return result;
}

metalrobo::NumiHumanKneePayload loadOpenKnee(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    require(rigid.engineBodyCount > metalrobo::NUMI_HUMAN_KNEE_PATELLA_BODY,
            "NHKNEE1 requires the pinned MyoSim full-body articulation");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.is_open(), "cannot open Open Knee(s) payload " + path.string());
    const std::streamsize size = input.tellg();
    require(size > 0, "cannot determine Open Knee(s) payload size");
    input.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    require(input.good(), "truncated Open Knee(s) payload");
    metalrobo::NumiHumanKneePayload result;
    const auto diagnostics = metalrobo::decodeNumiHumanKneePayload(bytes, result);
    require(
        diagnostics.succeeded(),
        std::string("invalid NHKNEE1 payload: ") +
            metalrobo::numiHumanKneeStatusName(diagnostics.status) +
            " at " + std::to_string(diagnostics.failingIndex)
    );
    return result;
}

LoadedOpenKneeLigamentFEM loadOpenKneeLigamentFEM(
    const std::filesystem::path& path,
    const metalrobo::NumiHumanKneePayload& knee
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(),
            "cannot open accepted Open Knee ligament FEM snapshot " + path.string());
    LoadedOpenKneeLigamentFEM result;
    readObject(input, result.header, "Open Knee ligament FEM header");
    const std::uint32_t expectedSide =
        knee.side == metalrobo::NumiHumanKneeSide::left ? 0u : 1u;
    const bool v1 = result.header.magic == kOpenKneeLigamentFEMMagicV1 &&
        result.header.abi == 1u && result.header.regionCount == 4u &&
        result.header.nodeCount == 38159u &&
        result.header.tetrahedronCount == 159416u;
    const bool v2 = result.header.magic == kOpenKneeLigamentFEMMagicV2 &&
        result.header.abi == 2u && result.header.regionCount == 5u &&
        result.header.nodeCount == 47439u &&
        result.header.tetrahedronCount == 195032u;
    require((v1 || v2) && result.header.side == expectedSide &&
                result.header.poseKind == 1u &&
                result.header.reserved0 == 0u && result.header.reserved1 == 0u,
            "accepted Open Knee ligament FEM header is incompatible");
    const auto regions = readVector<OpenKneeLigamentFEMRegion>(
        input, result.header.regionCount, "Open Knee ligament FEM regions");
    const auto nodes = readVector<OpenKneeLigamentFEMNode>(
        input, result.header.nodeCount, "Open Knee ligament FEM nodes");
    require(input.peek() == std::ifstream::traits_type::eof(),
            "accepted Open Knee ligament FEM snapshot has trailing bytes");
    result.worldNodes.reserve(knee.nodes.size());
    result.deformedNodes.assign(knee.nodes.size(), false);
    for (const auto& node : knee.nodes) {
        result.worldNodes.push_back({
            node.restWorld[0u], node.restWorld[1u], node.restWorld[2u], 1.0f});
    }
    std::uint32_t coveredNodes = 0u;
    for (const OpenKneeLigamentFEMRegion& disk : regions) {
        const std::size_t nameLength = strnlen(disk.name.data(), disk.name.size());
        require(nameLength > 0u && nameLength < disk.name.size(),
                "Open Knee ligament FEM region name is malformed");
        const std::string name(disk.name.data(), nameLength);
        const auto source = std::find_if(
            knee.regions.begin(), knee.regions.end(),
            [&name](const metalrobo::NumiHumanKneeRegion& candidate) {
                return candidate.name == name;
            });
        const bool exactTissue = source != knee.regions.end() &&
            (source->kind == metalrobo::NumiHumanKneeRegionKind::ligament ||
             (v2 && source->kind == metalrobo::NumiHumanKneeRegionKind::tendon &&
              source->name == "PTL"));
        require(exactTissue &&
                    source->firstNode == disk.payloadFirstNode &&
                    source->nodeCount == disk.nodeCount &&
                    disk.snapshotFirstNode <= nodes.size() &&
                    disk.nodeCount <= nodes.size() - disk.snapshotFirstNode &&
                    disk.payloadFirstNode <= result.worldNodes.size() &&
                    disk.nodeCount <= result.worldNodes.size() - disk.payloadFirstNode,
                "Open Knee ligament FEM region does not match NHKNEE1");
        for (std::uint32_t local = 0u; local < disk.nodeCount; ++local) {
            const auto& value = nodes[disk.snapshotFirstNode + local];
            require(std::isfinite(value.position[0u]) &&
                        std::isfinite(value.position[1u]) &&
                        std::isfinite(value.position[2u]),
                    "Open Knee ligament FEM node is nonfinite");
            const std::uint32_t target = disk.payloadFirstNode + local;
            require(!result.deformedNodes[target],
                    "Open Knee ligament FEM snapshot overlaps regions");
            const mr_float4 deformed{
                value.position[0u], value.position[1u], value.position[2u], 1.0f};
            const mr_float4 rest = result.worldNodes[target];
            const float dx = deformed.x - rest.x;
            const float dy = deformed.y - rest.y;
            const float dz = deformed.z - rest.z;
            result.maximumDisplacementMeters = std::max(
                result.maximumDisplacementMeters,
                std::sqrt(dx * dx + dy * dy + dz * dz));
            result.worldNodes[target] = deformed;
            result.deformedNodes[target] = true;
            ++coveredNodes;
        }
    }
    require(coveredNodes == result.header.nodeCount &&
                result.maximumDisplacementMeters > 0.0f &&
                result.maximumDisplacementMeters < 0.001f,
            "accepted Open Knee ligament FEM coverage or displacement is invalid");
    return result;
}

std::array<std::uint32_t, 3u> openKneeBodyIndices(
    const metalrobo::NumiHumanKneePayload& knee
) {
    return knee.side == metalrobo::NumiHumanKneeSide::left
        ? std::array<std::uint32_t, 3u>{
            metalrobo::NUMI_HUMAN_KNEE_FEMUR_BODY,
            metalrobo::NUMI_HUMAN_KNEE_TIBIA_BODY,
            metalrobo::NUMI_HUMAN_KNEE_PATELLA_BODY}
        : std::array<std::uint32_t, 3u>{
            metalrobo::NUMI_HUMAN_KNEE_RIGHT_FEMUR_BODY,
            metalrobo::NUMI_HUMAN_KNEE_RIGHT_TIBIA_BODY,
            metalrobo::NUMI_HUMAN_KNEE_RIGHT_PATELLA_BODY};
}

LoadedTorsoAnatomy loadTorsoAnatomy(
    const std::filesystem::path& path,
    const RigidHeader& rigid,
    const std::uint32_t expectedRegistrationFingerprint
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open BodyParts3D torso anatomy payload " + path.string());
    LoadedTorsoAnatomy result;
    readObject(input, result.header, "BodyParts3D torso anatomy header");
    require(result.header.magic == kTorsoAnatomyMagic &&
                result.header.payloadAbi == kTorsoAnatomyPayloadAbi &&
                result.header.registrationFingerprint == expectedRegistrationFingerprint &&
                result.header.sourceSha256 == rigid.sourceSha256 &&
                result.header.surfaceCount > 0u && result.header.surfaceCount <= 64u &&
                result.header.vertexCount > 0u && result.header.vertexCount <= 1'000'000u &&
                result.header.indexCount > 0u && result.header.indexCount <= 6'000'000u &&
                result.header.indexCount % 3u == 0u,
            "BodyParts3D torso anatomy payload/header disagreement");
    result.records = readVector<TorsoAnatomyRecord>(
        input, result.header.surfaceCount, "BodyParts3D torso anatomy records"
    );
    result.vertices = readVector<TorsoAnatomyVertex>(
        input, result.header.vertexCount, "BodyParts3D torso anatomy vertices"
    );
    result.indices = readVector<std::uint32_t>(
        input, result.header.indexCount, "BodyParts3D torso anatomy indices"
    );
    require(input.peek() == std::char_traits<char>::eof(),
            "BodyParts3D torso anatomy payload has trailing bytes");
    for (const TorsoAnatomyVertex& vertex : result.vertices) {
        const float normalLength = std::sqrt(
            vertex.normalX * vertex.normalX +
            vertex.normalY * vertex.normalY +
            vertex.normalZ * vertex.normalZ
        );
        require(std::isfinite(vertex.positionX) && std::isfinite(vertex.positionY) &&
                    std::isfinite(vertex.positionZ) && std::isfinite(normalLength) &&
                    std::abs(normalLength - 1.0f) <= 2.0e-3f,
                "BodyParts3D torso anatomy vertex is malformed");
    }
    // Focused Human payloads deliberately retain their global source stable
    // IDs (for example a four-surface calf subset includes tendon ID 7). IDs
    // are therefore unique but need not be dense in [1, tissueCount].
    std::vector<std::uint32_t> stableIds;
    stableIds.reserve(result.records.size());
    for (const TorsoAnatomyRecord& record : result.records) {
        require(record.bodyIndex < rigid.engineBodyCount && record.vertexCount > 0u &&
                    record.indexCount > 0u && record.indexCount % 3u == 0u &&
                    record.firstVertex <= result.vertices.size() &&
                    record.vertexCount <= result.vertices.size() - record.firstVertex &&
                    record.firstIndex <= result.indices.size() &&
                    record.indexCount <= result.indices.size() - record.firstIndex &&
                    record.stableId > 0u && record.stableId < stableIds.size() &&
                    !stableIds[record.stableId] && record.reserved0 == 0u &&
                    (record.layer == kTorsoAnatomyLayerOrgan ||
                     record.layer == kTorsoAnatomyLayerVessel ||
                     record.layer == kTorsoAnatomyLayerNerve),
                "BodyParts3D torso anatomy record is malformed");
        stableIds[record.stableId] = true;
        for (std::uint32_t offset = 0u; offset < record.indexCount; ++offset) {
            const std::uint32_t index = result.indices[record.firstIndex + offset];
            require(index >= record.firstVertex && index < record.firstVertex + record.vertexCount,
                    "BodyParts3D torso anatomy index escapes its source mesh");
        }
    }
    return result;
}

LoadedSoftTissues loadSoftTissues(
    const std::filesystem::path& path,
    const RigidHeader& rigid
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open BodyParts3D soft-tissue payload " + path.string());
    LoadedSoftTissues result;
    std::array<char, 8u> magic{};
    std::uint32_t payloadAbi = 0u;
    readObject(input, magic, "BodyParts3D soft-tissue magic");
    readObject(input, payloadAbi, "BodyParts3D soft-tissue ABI");
    input.seekg(0, std::ios::beg);
    require(input.good(), "could not rewind BodyParts3D soft-tissue payload");
    const bool legacyTwoBody = magic == kSoftTissueMagic &&
        payloadAbi == kSoftTissuePayloadAbi;
    const bool multiBody = magic == kMultiBodySoftTissueMagic &&
        payloadAbi == kMultiBodySoftTissuePayloadAbi;
    const bool routeBodySparse = magic == kRouteSoftTissueMagic &&
        payloadAbi == kRouteSoftTissuePayloadAbi;
    require(legacyTwoBody || multiBody || routeBodySparse,
            "BodyParts3D soft-tissue payload magic/ABI is unsupported");

    std::vector<RouteSoftTissueRecord> routeRecords;
    std::vector<RouteSoftTissueBinding> routeBindings;
    std::vector<RouteSoftTissueVertex> routeVertices;
    if (routeBodySparse) {
        RouteSoftTissueHeader header{};
        readObject(input, header, "BodyParts3D route-body soft-tissue header");
        require(header.registrationFingerprint != 0u &&
                    header.sourceSha256 == rigid.sourceSha256 &&
                    header.tissueCount > 0u && header.tissueCount <= 192u &&
                    header.bindingCount > 0u &&
                    header.bindingCount <= header.tissueCount * kRouteSoftTissueMaximumBindings &&
                    header.vertexCount > 0u && header.indexCount > 0u &&
                    header.indexCount % 3u == 0u &&
                    header.vertexCount <= 1'000'000u &&
                    header.indexCount <= 6'000'000u,
                "BodyParts3D route-body soft-tissue payload/header disagreement");
        result.header.magic = header.magic;
        result.header.payloadAbi = header.payloadAbi;
        result.header.tissueCount = header.tissueCount;
        result.header.vertexCount = header.vertexCount;
        result.header.indexCount = header.indexCount;
        result.header.reserved0 = header.registrationFingerprint;
        result.header.sourceSha256 = header.sourceSha256;
        routeRecords = readVector<RouteSoftTissueRecord>(
            input, header.tissueCount, "BodyParts3D route-body soft-tissue records"
        );
        routeBindings = readVector<RouteSoftTissueBinding>(
            input, header.bindingCount, "BodyParts3D route-body soft-tissue bindings"
        );
        routeVertices = readVector<RouteSoftTissueVertex>(
            input, header.vertexCount, "BodyParts3D route-body soft-tissue vertices"
        );
        result.usesRouteBodySparseWeights = true;
    } else {
        readObject(input, result.header, "BodyParts3D soft-tissue header");
        require(result.header.reserved0 != 0u &&
                    result.header.sourceSha256 == rigid.sourceSha256 &&
                    result.header.tissueCount > 0u && result.header.tissueCount <= 192u &&
                    result.header.vertexCount > 0u && result.header.indexCount > 0u &&
                    result.header.indexCount % 3u == 0u &&
                    result.header.vertexCount <= 1'000'000u &&
                    result.header.indexCount <= 6'000'000u,
                "BodyParts3D soft-tissue payload/header disagreement");
    }
    if (legacyTwoBody) {
        const auto records = readVector<LegacySoftTissueRecord>(
            input, result.header.tissueCount, "legacy BodyParts3D soft-tissue records"
        );
        const auto vertices = readVector<LegacySoftTissueVertex>(
            input, result.header.vertexCount, "legacy BodyParts3D soft-tissue vertices"
        );
        result.records.reserve(records.size());
        for (const LegacySoftTissueRecord& legacy : records) {
            SoftTissueRecord record{};
            record.bodyIndex[0] = legacy.primaryBodyIndex;
            record.bodyIndex[1] = legacy.secondaryBodyIndex;
            record.bindingCount = 2u;
            record.firstVertex = legacy.firstVertex;
            record.vertexCount = legacy.vertexCount;
            record.firstIndex = legacy.firstIndex;
            record.indexCount = legacy.indexCount;
            record.stableId = legacy.stableId;
            record.layer = legacy.layer;
            record.binding[0].translation[0] = legacy.primaryTranslationX;
            record.binding[0].translation[1] = legacy.primaryTranslationY;
            record.binding[0].translation[2] = legacy.primaryTranslationZ;
            record.binding[0].quaternion[0] = legacy.primaryQuaternionX;
            record.binding[0].quaternion[1] = legacy.primaryQuaternionY;
            record.binding[0].quaternion[2] = legacy.primaryQuaternionZ;
            record.binding[0].quaternion[3] = legacy.primaryQuaternionW;
            record.binding[0].uniformScale = legacy.primaryUniformScale;
            record.binding[1].translation[0] = legacy.secondaryTranslationX;
            record.binding[1].translation[1] = legacy.secondaryTranslationY;
            record.binding[1].translation[2] = legacy.secondaryTranslationZ;
            record.binding[1].quaternion[0] = legacy.secondaryQuaternionX;
            record.binding[1].quaternion[1] = legacy.secondaryQuaternionY;
            record.binding[1].quaternion[2] = legacy.secondaryQuaternionZ;
            record.binding[1].quaternion[3] = legacy.secondaryQuaternionW;
            record.binding[1].uniformScale = legacy.secondaryUniformScale;
            result.records.push_back(record);
        }
        result.vertices.reserve(vertices.size());
        for (const LegacySoftTissueVertex& legacy : vertices) {
            SoftTissueVertex vertex{};
            vertex.positionX = legacy.positionX;
            vertex.positionY = legacy.positionY;
            vertex.positionZ = legacy.positionZ;
            vertex.normalX = legacy.normalX;
            vertex.normalY = legacy.normalY;
            vertex.normalZ = legacy.normalZ;
            vertex.bindingIndex[0] = 0u;
            vertex.bindingIndex[1] = 1u;
            vertex.weight[0] = legacy.primaryWeight;
            vertex.weight[1] = 1.0f - legacy.primaryWeight;
            result.vertices.push_back(vertex);
        }
    } else if (multiBody) {
        const auto records = readVector<MultiBodySoftTissueRecord>(
            input, result.header.tissueCount, "BodyParts3D multi-body soft-tissue records"
        );
        const auto vertices = readVector<MultiBodySoftTissueVertex>(
            input, result.header.vertexCount, "BodyParts3D multi-body soft-tissue vertices"
        );
        result.records.reserve(records.size());
        for (const MultiBodySoftTissueRecord& source : records) {
            SoftTissueRecord record{};
            while (record.bindingCount < 3u &&
                   source.bodyIndex[record.bindingCount] != MR_INVALID_INDEX) {
                const std::uint32_t binding = record.bindingCount++;
                record.bodyIndex[binding] = source.bodyIndex[binding];
                record.binding[binding] = source.binding[binding];
            }
            record.firstVertex = source.firstVertex;
            record.vertexCount = source.vertexCount;
            record.firstIndex = source.firstIndex;
            record.indexCount = source.indexCount;
            record.stableId = source.stableId;
            record.layer = source.layer;
            require(source.reserved0 == 0u,
                    "BodyParts3D multi-body soft-tissue record reserved field is nonzero");
            result.records.push_back(record);
        }
        result.vertices.reserve(vertices.size());
        for (const MultiBodySoftTissueVertex& source : vertices) {
            SoftTissueVertex vertex{};
            vertex.positionX = source.positionX;
            vertex.positionY = source.positionY;
            vertex.positionZ = source.positionZ;
            vertex.normalX = source.normalX;
            vertex.normalY = source.normalY;
            vertex.normalZ = source.normalZ;
            for (std::uint32_t influence = 0u; influence < 3u; ++influence) {
                vertex.bindingIndex[influence] = influence;
                vertex.weight[influence] = source.weight[influence];
            }
            result.vertices.push_back(vertex);
        }
    } else {
        result.records.reserve(routeRecords.size());
        for (const RouteSoftTissueRecord& source : routeRecords) {
            require(source.bindingCount > 0u &&
                        source.bindingCount <= kRouteSoftTissueMaximumBindings &&
                        source.firstBinding <= routeBindings.size() &&
                        source.bindingCount <= routeBindings.size() - source.firstBinding,
                    "BodyParts3D route-body soft-tissue binding range is malformed");
            SoftTissueRecord record{};
            record.bindingCount = source.bindingCount;
            for (std::uint32_t binding = 0u; binding < source.bindingCount; ++binding) {
                const RouteSoftTissueBinding& resolved =
                    routeBindings[source.firstBinding + binding];
                record.bodyIndex[binding] = resolved.bodyIndex;
                record.binding[binding] = resolved.transform;
            }
            record.firstVertex = source.firstVertex;
            record.vertexCount = source.vertexCount;
            record.firstIndex = source.firstIndex;
            record.indexCount = source.indexCount;
            record.stableId = source.stableId;
            record.layer = source.layer;
            result.records.push_back(record);
        }
        result.vertices.reserve(routeVertices.size());
        for (const RouteSoftTissueVertex& source : routeVertices) {
            SoftTissueVertex vertex{};
            vertex.positionX = source.positionX;
            vertex.positionY = source.positionY;
            vertex.positionZ = source.positionZ;
            vertex.normalX = source.normalX;
            vertex.normalY = source.normalY;
            vertex.normalZ = source.normalZ;
            std::copy(std::begin(source.bindingIndex), std::end(source.bindingIndex),
                      std::begin(vertex.bindingIndex));
            std::copy(std::begin(source.weight), std::end(source.weight),
                      std::begin(vertex.weight));
            result.vertices.push_back(vertex);
        }
    }
    result.indices = readVector<std::uint32_t>(
        input, result.header.indexCount, "BodyParts3D soft-tissue indices"
    );
    require(input.peek() == std::char_traits<char>::eof(),
            "BodyParts3D soft-tissue payload has trailing bytes");
    for (const SoftTissueVertex& vertex : result.vertices) {
        const float normalLength = std::sqrt(
            vertex.normalX * vertex.normalX +
            vertex.normalY * vertex.normalY +
            vertex.normalZ * vertex.normalZ
        );
        require(std::isfinite(vertex.positionX) && std::isfinite(vertex.positionY) &&
                    std::isfinite(vertex.positionZ) && std::isfinite(normalLength) &&
                    std::abs(normalLength - 1.0f) <= 2.0e-3f &&
                    std::all_of(std::begin(vertex.weight), std::end(vertex.weight), [](const float weight) {
                        return std::isfinite(weight) && weight >= 0.0f && weight <= 1.0f;
                    }) &&
                    std::abs(vertex.weight[0] + vertex.weight[1] +
                             vertex.weight[2] + vertex.weight[3] - 1.0f) <= 2.0e-3f,
                "BodyParts3D soft-tissue vertex is malformed");
    }
    // Focused Human payloads deliberately retain their global source stable
    // IDs (for example a four-surface calf subset includes tendon ID 7). IDs
    // are therefore unique but need not be dense in [1, tissueCount].
    std::vector<std::uint32_t> stableIds;
    stableIds.reserve(result.records.size());
    for (const SoftTissueRecord& record : result.records) {
        require(record.bindingCount > 0u &&
                    record.bindingCount <= kRouteSoftTissueMaximumBindings,
                "BodyParts3D soft-tissue record binding count is malformed");
        for (std::uint32_t binding = 0u; binding < record.bindingCount; ++binding) {
            const std::uint32_t bodyIndex = record.bodyIndex[binding];
            require(bodyIndex != MR_INVALID_INDEX,
                    "BodyParts3D soft-tissue binding has a hole");
            require(bodyIndex < rigid.engineBodyCount, "BodyParts3D soft-tissue body binding exceeds model");
            for (std::uint32_t earlier = 0u; earlier < binding; ++earlier) {
                require(bodyIndex != record.bodyIndex[earlier],
                        "BodyParts3D soft-tissue binding repeats a body");
            }
            const float orientationLength = std::sqrt(
                record.binding[binding].quaternion[0] * record.binding[binding].quaternion[0] +
                record.binding[binding].quaternion[1] * record.binding[binding].quaternion[1] +
                record.binding[binding].quaternion[2] * record.binding[binding].quaternion[2] +
                record.binding[binding].quaternion[3] * record.binding[binding].quaternion[3]
            );
            require(std::isfinite(record.binding[binding].translation[0]) &&
                        std::isfinite(record.binding[binding].translation[1]) &&
                        std::isfinite(record.binding[binding].translation[2]) &&
                        std::isfinite(record.binding[binding].uniformScale) &&
                        record.binding[binding].uniformScale > 0.0f &&
                        std::isfinite(orientationLength) &&
                        std::abs(orientationLength - 1.0f) <= 2.0e-3f,
                    "BodyParts3D soft-tissue binding transform is malformed");
        }
        const bool supplementalBone = record.layer == kSoftTissueLayerSupplementalBone;
        require(record.bindingCount >= (supplementalBone ? 1u : 2u) &&
                    record.vertexCount > 0u &&
                    record.indexCount > 0u && record.indexCount % 3u == 0u &&
                    (record.layer == kSoftTissueLayerMuscle ||
                     record.layer == kSoftTissueLayerTendon ||
                     record.layer == kSoftTissueLayerSupplementalBone) &&
                    record.firstVertex <= result.vertices.size() &&
                    record.vertexCount <= result.vertices.size() - record.firstVertex &&
                    record.firstIndex <= result.indices.size() &&
                    record.indexCount <= result.indices.size() - record.firstIndex &&
                    record.stableId > 0u &&
                    std::find(stableIds.begin(), stableIds.end(), record.stableId) == stableIds.end(),
                "BodyParts3D soft-tissue record is malformed");
        stableIds.push_back(record.stableId);
        for (std::uint32_t vertexOffset = 0u;
             vertexOffset < record.vertexCount; ++vertexOffset) {
            const SoftTissueVertex& vertex =
                result.vertices[record.firstVertex + vertexOffset];
            for (std::uint32_t influence = 0u;
                 influence < kRouteSoftTissueMaximumInfluences; ++influence) {
                if (vertex.weight[influence] <= 2.0e-6f) continue;
                require(vertex.bindingIndex[influence] < record.bindingCount,
                        "BodyParts3D soft-tissue vertex influence escapes its route-body table");
            }
        }
        for (std::uint32_t offset = 0u; offset < record.indexCount; ++offset) {
            const std::uint32_t index = result.indices[record.firstIndex + offset];
            require(index >= record.firstVertex && index < record.firstVertex + record.vertexCount,
                    "BodyParts3D soft-tissue index escapes its source mesh");
        }
    }
    return result;
}

LoadedSkin loadSkin(
    const std::filesystem::path& path,
    const RigidHeader& rigid,
    const std::uint32_t expectedRegistrationFingerprint
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open BodyParts3D skinned-shell payload " + path.string());
    LoadedSkin result;
    readObject(input, result.header, "BodyParts3D skinned-shell header");
    require(result.header.magic == kSkinMagic &&
                (result.header.payloadAbi == kLegacySkinPayloadAbi ||
                 result.header.payloadAbi == kBoundaryLocalSkinPayloadAbi ||
                 result.header.payloadAbi == kSourceLocalSkinPayloadAbi ||
                 result.header.payloadAbi == kSkinPayloadAbi) &&
                result.header.registrationFingerprint == expectedRegistrationFingerprint &&
                result.header.sourceSha256 == rigid.sourceSha256 &&
                result.header.bindingCount >= 4u && result.header.bindingCount <= 256u &&
                result.header.vertexCount > 0u && result.header.indexCount > 0u &&
                result.header.indexCount % 3u == 0u &&
                result.header.vertexCount <= 1'000'000u &&
                result.header.indexCount <= 6'000'000u,
            "BodyParts3D skinned-shell payload/header disagreement");
    result.usesBoundaryLocalWeights =
        result.header.payloadAbi == kBoundaryLocalSkinPayloadAbi;
    result.usesSourceSurfaceLocalWeights =
        result.header.payloadAbi == kSourceLocalSkinPayloadAbi ||
        result.header.payloadAbi == kSkinPayloadAbi;
    result.usesWorldRestNormals =
        result.header.payloadAbi == kSkinPayloadAbi;
    result.bindings = readVector<SkinBindingRecord>(
        input, result.header.bindingCount, "BodyParts3D skinned-shell bindings"
    );
    result.vertices = readVector<SkinVertex>(
        input, result.header.vertexCount, "BodyParts3D skinned-shell vertices"
    );
    result.indices = readVector<std::uint32_t>(
        input, result.header.indexCount, "BodyParts3D skinned-shell indices"
    );
    require(input.peek() == std::char_traits<char>::eof(),
            "BodyParts3D skinned-shell payload has trailing bytes");
    std::vector<bool> boundBodies(rigid.engineBodyCount, false);
    for (const SkinBindingRecord& binding : result.bindings) {
        const float orientationLength = std::sqrt(
            binding.quaternionX * binding.quaternionX +
            binding.quaternionY * binding.quaternionY +
            binding.quaternionZ * binding.quaternionZ +
            binding.quaternionW * binding.quaternionW
        );
        require(binding.bodyIndex < rigid.engineBodyCount && !boundBodies[binding.bodyIndex] &&
                    std::isfinite(binding.translationX) &&
                    std::isfinite(binding.translationY) &&
                    std::isfinite(binding.translationZ) &&
                    std::isfinite(binding.uniformScale) && binding.uniformScale > 0.0f &&
                    std::isfinite(orientationLength) &&
                    std::abs(orientationLength - 1.0f) <= 2.0e-3f,
                "BodyParts3D skinned-shell binding is malformed");
        boundBodies[binding.bodyIndex] = true;
    }
    for (const SkinVertex& vertex : result.vertices) {
        const float normalLength = std::sqrt(
            vertex.normalX * vertex.normalX +
            vertex.normalY * vertex.normalY +
            vertex.normalZ * vertex.normalZ
        );
        float weightSum = 0.0f;
        for (std::size_t influence = 0u; influence < 4u; ++influence) {
            require(vertex.bindingIndex[influence] < result.bindings.size() &&
                        std::isfinite(vertex.weight[influence]) &&
                        vertex.weight[influence] >= 0.0f && vertex.weight[influence] <= 1.0f,
                    "BodyParts3D skinned-shell vertex influence is malformed");
            weightSum += vertex.weight[influence];
        }
        require(std::isfinite(vertex.positionX) && std::isfinite(vertex.positionY) &&
                    std::isfinite(vertex.positionZ) && std::isfinite(normalLength) &&
                    std::abs(normalLength - 1.0f) <= 2.0e-3f &&
                    std::isfinite(weightSum) && std::abs(weightSum - 1.0f) <= 2.0e-3f,
                "BodyParts3D skinned-shell vertex is malformed");
    }
    for (const std::uint32_t index : result.indices) {
        require(index < result.vertices.size(),
                "BodyParts3D skinned-shell index escapes its source mesh");
    }
    return result;
}

LoadedPectoralisFascia loadPectoralisFascia(
    const std::filesystem::path& path,
    const LoadedSoftTissues& tissues,
    const LoadedMuscles& muscles
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), "cannot open pectoralis fascia payload " + path.string());
    LoadedPectoralisFascia result;
    readObject(input, result.header, "pectoralis fascia header");
    require(result.header.magic == kPectoralisFasciaMagic &&
                result.header.payloadAbi == kPectoralisFasciaPayloadAbi &&
                result.header.regionCount == 6u &&
                result.header.nodeCount > 0u && result.header.nodeCount <= 100'000u &&
                result.header.tetrahedronCount > 0u &&
                result.header.tetrahedronCount <= 300'000u &&
                std::isfinite(result.header.thicknessMeters) &&
                result.header.thicknessMeters >= 0.0002f &&
                result.header.thicknessMeters <= 0.0012f &&
                std::isfinite(result.header.muscleLoadFraction) &&
                result.header.muscleLoadFraction > 0.0f &&
                result.header.muscleLoadFraction <= 0.25f &&
                result.header.presentationTriangleCount >= 5'000u &&
                result.header.presentationTriangleCount <= 100'000u &&
                result.header.reserved1 == 0u,
            "pectoralis fascia header is malformed");
    result.regions = readVector<PectoralisFasciaRegion>(
        input, result.header.regionCount, "pectoralis fascia regions"
    );
    result.nodes = readVector<PectoralisFasciaNode>(
        input, result.header.nodeCount, "pectoralis fascia nodes"
    );
    result.tetrahedra = readVector<PectoralisFasciaTetrahedron>(
        input, result.header.tetrahedronCount, "pectoralis fascia tetrahedra"
    );
    result.presentationTriangles =
        readVector<PectoralisFasciaPresentationTriangle>(
            input, result.header.presentationTriangleCount,
            "pectoralis fascia exact anterior presentation triangles"
        );
    require(input.peek() == std::char_traits<char>::eof(),
            "pectoralis fascia payload has trailing bytes");
    std::uint32_t expectedFirstNode = 0u;
    std::uint32_t expectedFirstTetrahedron = 0u;
    for (std::uint32_t regionIndex = 0u;
         regionIndex < result.regions.size(); ++regionIndex) {
        const PectoralisFasciaRegion& region = result.regions[regionIndex];
        const auto tissue = std::find_if(
            tissues.records.begin(), tissues.records.end(),
            [&region](const SoftTissueRecord& value) {
                return value.stableId == region.softTissueStableId;
            }
        );
        require(region.muscleIndex < muscles.gpuMuscles.size() &&
                    region.firstNode == expectedFirstNode &&
                    region.nodeCount >= 12u && region.nodeCount % 2u == 0u &&
                    region.nodeCount <= result.nodes.size() - region.firstNode &&
                    region.firstTetrahedron == expectedFirstTetrahedron &&
                    region.tetrahedronCount >= 1u &&
                    region.tetrahedronCount <=
                        result.tetrahedra.size() - region.firstTetrahedron &&
                    region.softTissueStableId > 0u &&
                    tissue != tissues.records.end() &&
                    tissue->layer == kSoftTissueLayerMuscle &&
                    tissue->vertexCount >= region.nodeCount / 2u,
                "pectoralis fascia region is inconsistent with native source surfaces");
        expectedFirstNode += region.nodeCount;
        expectedFirstTetrahedron += region.tetrahedronCount;
    }
    require(expectedFirstNode == result.nodes.size() &&
                expectedFirstTetrahedron == result.tetrahedra.size(),
            "pectoralis fascia regions do not cover the payload");
    for (std::uint32_t index = 0u; index < result.nodes.size(); ++index) {
        const PectoralisFasciaNode& node = result.nodes[index];
        const PectoralisFasciaRegion& region = result.regions.at(node.regionIndex);
        const auto tissue = std::find_if(
            tissues.records.begin(), tissues.records.end(),
            [&region](const SoftTissueRecord& value) {
                return value.stableId == region.softTissueStableId;
            }
        );
        require(node.regionIndex < result.regions.size() &&
                    (node.flags & ~3u) == 0u && node.reserved0 == 0u &&
                    std::isfinite(node.compiledMassKg) && node.compiledMassKg > 0.0f &&
                    std::all_of(std::begin(node.sourcePosition),
                                std::end(node.sourcePosition),
                                [](float value) { return std::isfinite(value); }) &&
                    tissue != tissues.records.end() &&
                    node.sourceVertexIndex < tissue->vertexCount,
                "pectoralis fascia node is malformed");
    }
    for (const PectoralisFasciaTetrahedron& tetrahedron : result.tetrahedra) {
        require(tetrahedron.regionIndex < result.regions.size() &&
                    std::all_of(std::begin(tetrahedron.node),
                                std::end(tetrahedron.node),
                                [&result](std::uint32_t index) {
                                    return index < result.nodes.size();
                                }),
                "pectoralis fascia tetrahedron is malformed");
    }
    std::vector<std::uint32_t> presentationCount(result.regions.size(), 0u);
    for (const auto& triangle : result.presentationTriangles) {
        require(triangle.regionIndex < result.regions.size(),
                "pectoralis fascia presentation region is malformed");
        const auto& region = result.regions[triangle.regionIndex];
        const auto tissue = std::find_if(
            tissues.records.begin(), tissues.records.end(),
            [&region](const SoftTissueRecord& value) {
                return value.stableId == region.softTissueStableId;
            }
        );
        require(tissue != tissues.records.end() &&
                    std::all_of(
                        std::begin(triangle.sourceVertex),
                        std::end(triangle.sourceVertex),
                        [&tissue](const std::uint32_t index) {
                            return index < tissue->vertexCount;
                        }
                    ) &&
                    triangle.sourceVertex[0] != triangle.sourceVertex[1] &&
                    triangle.sourceVertex[0] != triangle.sourceVertex[2] &&
                    triangle.sourceVertex[1] != triangle.sourceVertex[2],
                "pectoralis fascia exact anterior presentation triangle is malformed");
        ++presentationCount[triangle.regionIndex];
    }
    require(std::all_of(
                presentationCount.begin(), presentationCount.end(),
                [](const std::uint32_t count) { return count >= 32u; }
            ),
            "pectoralis fascia presentation does not cover all six regions");
    return result;
}

struct MuscleDrivenVisualState {
    std::vector<float> q;
    std::vector<MRNumiHumanTendonTransferResultGPU> finalTendonTransfers;
    std::uint32_t stepCount = 0u;
    double maximumVelocityDelta = 0.0;
    std::uint32_t maximumVelocityDeltaDof = MR_INVALID_INDEX;
    double maximumConfigurationDelta = 0.0;
    std::uint32_t maximumConfigurationDeltaQ = MR_INVALID_INDEX;
    std::uint32_t appliedWrapCount = 0u;
    bool supportContactApplied = false;
    std::uint32_t supportWitnessCount = 0u;
    std::uint32_t activeSupportContactCount = 0u;
    std::uint32_t maximumActiveSupportContactCount = 0u;
    double minimumSupportPlaneGapMeters = std::numeric_limits<double>::infinity();
    double supportSeedTranslationMeters = 0.0;
    double supportMaximumGpuCpuVelocityError = 0.0;
    double supportGpuElapsedMilliseconds = 0.0;
    std::string supportDeviceName;
    std::string supportMetalStatus = "not_attempted";
    // Full-body MyoSim force is evaluated by the articulated Metal sidecar
    // before the bounded Core state step. These counters are intentionally
    // separate from support-contact admission, which has a different current
    // capacity boundary.
    std::uint32_t muscleMetalStepCount = 0u;
    std::uint32_t muscleMetalForceRecordCount = 0u;
    // Zero means that the native task deliberately excited every authored
    // muscle with the requested common activation.  A nonzero value instead
    // records an explicit source-actuator subset; every one of the 416 source
    // paths is still evaluated on device so the comparison never silently
    // substitutes a smaller mechanical model.
    std::uint32_t selectedSourceMuscleActivationCount = 0u;
    double muscleMetalElapsedMilliseconds = 0.0;
    std::string muscleMetalDeviceName;
    bool persistentMetalHorizon = false;
    bool selectedTendonControl = false;
    bool selectedControlBaselineEvaluated = false;
    double selectedControlBaselineMaximumQDelta = 0.0;
    double selectedControlBaselineMaximumVDelta = 0.0;
    std::uint32_t selectedControlBaselineMaximumQDeltaIndex = MR_INVALID_INDEX;
    std::uint32_t selectedControlBaselineMaximumVDeltaIndex = MR_INVALID_INDEX;
    double selectedControlBaselineElapsedMilliseconds = 0.0;
    bool rootAssistanceEnabled = false;
    bool assistanceRemovalEvaluated = false;
    std::uint32_t persistentCompletedSteps = 0u;
    double persistentMaximumAcceleration = 0.0;
    double persistentMaximumPenetrationMeters = 0.0;
    double persistentNormalImpulse = 0.0;
    double persistentRootAssistanceForce = 0.0;
    double persistentRootAssistanceTorque = 0.0;
    std::uint32_t compiledActiveMuscleCount = 0u;
    double compiledActivationResidualRms = 0.0;
    double compiledInitialActivationResidualRms = 0.0;
    double compiledMaximumAccelerationResidual = 0.0;
    double compiledMaximumVelocityIncrement = 0.0;
    bool compiledBalanced = false;
    double compiledMaximumActivation = 0.0;
    std::uint32_t compiledRecruitedMuscleCount = 0u;
    std::uint32_t compiledActivePositionLimitCount = 0u;
    std::uint32_t compiledAcceptedPoseSteps = 0u;
    double compiledMaximumEqualityReaction = 0.0;
    double compiledMaximumLimitReaction = 0.0;
    std::uint32_t compiledSupportContactCount = 0u;
    std::uint32_t compiledActiveSupportContactCount = 0u;
    double compiledTotalSupportForceNewtons = 0.0;
    double compiledMaximumRootForceResidual = 0.0;
    double compiledMaximumRootAccelerationResidual = 0.0;
    std::uint32_t jointEqualityCount = 0u;
    double maximumJointEqualityPositionError = 0.0;
    double maximumJointEqualityVelocityError = 0.0;
    double maximumJointEqualityImpulse = 0.0;
    double totalJointEqualityImpulse = 0.0;
    double assistedConfigurationDelta = 0.0;
    double removalConfigurationDelta = 0.0;
    double oneStepParityMaximumQError = 0.0;
    double oneStepParityMaximumVError = 0.0;
    bool deterministicReplayVerified = false;
    double deterministicReplayElapsedMilliseconds = 0.0;
    bool tendonStepTransactionEnabled = false;
    bool tendonBorrowedConsumerVerified = false;
    bool tendonRollbackVerified = false;
    bool tendonRigidStateIdentityVerified = false;
    bool tendonContinuumReactionVerified = false;
    // Passive continua add their anchor reactions to the retained source J^T
    // route force. Active tendon continua instead replace an explicitly owned
    // share. Keep those provenance boundaries distinct in emitted evidence.
    bool tendonContinuumPassiveReactionOnly = false;
    double tendonContinuumMaximumQDelta = 0.0;
    double tendonContinuumMaximumVDelta = 0.0;
    std::uint32_t tendonTransferCount = 0u;
    std::uint32_t tendonEnvelopeTransferCount = 0u;
    std::uint32_t tendonPointTransferCount = 0u;
    double tendonMaximumForceResidual = 0.0;
    double tendonMaximumMomentResidual = 0.0;
    double tendonMaximumGeneralizedCorrection = 0.0;
    // Source-coordinate order: ankle_r, subtalar_r, mtp_r, ankle_l,
    // subtalar_l, mtp_l. These targeted values make a visually suspect foot
    // distinguishable from a mesh-registration defect.
    std::array<double, 6u> compiledFootCoordinates{};
    std::array<double, 6u> finalFootCoordinates{};
    struct AchillesForceTransferSideAudit {
        bool available = false;
        std::uint32_t calcaneusBodyIndex = MR_INVALID_INDEX;
        std::uint32_t calcaneusBoneStableId = 0u;
        std::uint32_t ankleQIndex = MR_INVALID_INDEX;
        std::uint32_t ankleDofIndex = MR_INVALID_INDEX;
        std::uint32_t muscleCount = 0u;
        std::uint32_t distributedEndpointCount = 0u;
        double representedForceL1Newtons = 0.0;
        double representedForceIncrementL1Newtons = 0.0;
        double terminalForceResultantNewtons = 0.0;
        double terminalForceIncrementResultantNewtons = 0.0;
        double nodalForceResultantNewtons = 0.0;
        double aggregateForceResidualNewtons = 0.0;
        double maximumEndpointForceResidualNewtons = 0.0;
        double maximumEndpointMomentResidualNewtonMeters = 0.0;
        double sourceAnkleTorqueNewtonMeters = 0.0;
        double sourceAnkleTorqueIncrementNewtonMeters = 0.0;
        double distributedAnkleTorqueCorrectionNewtonMeters = 0.0;
        double minimumNormalizedTendonTension =
            std::numeric_limits<double>::infinity();
        double maximumNormalizedTendonTension = 0.0;
        double maximumDampedEquilibriumResidual = 0.0;
        double minimumPatchRadiusMeters =
            std::numeric_limits<double>::infinity();
        double maximumPatchRadiusMeters = 0.0;
        double configurationIncrementRadians = 0.0;
        double velocityIncrementRadiansPerSecond = 0.0;
    };
    std::array<AchillesForceTransferSideAudit, 2u> achilles{};
};

struct TendonLoadAuditConsumer {
    __strong id<MTLBuffer> transferSnapshot = nil;
    __strong id<MTLBuffer> correctionSnapshot = nil;
    __strong id<MTLBuffer> statusSnapshot = nil;
    std::uint32_t encodedPassCount = 0u;
    std::uint32_t abortCount = 0u;
    bool reject = false;
};

struct TendonLoadProgramChain {
    metalrobo::MetalNumiHumanTendonLoadProgram first{};
    metalrobo::MetalNumiHumanTendonLoadProgram second{};
};

bool encodeTendonLoadProgramChainPreDynamics(
    void* opaque,
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    auto* chain = static_cast<TendonLoadProgramChain*>(opaque);
    return chain != nullptr && chain->first.valid() && chain->second.valid() &&
        chain->first.encodePreDynamics(chain->first.context, pass) &&
        chain->second.encodePreDynamics(chain->second.context, pass);
}

bool encodeTendonLoadProgramChainPostValidation(
    void* opaque,
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    auto* chain = static_cast<TendonLoadProgramChain*>(opaque);
    return chain != nullptr && chain->first.valid() && chain->second.valid() &&
        chain->first.encodePostValidation(chain->first.context, pass) &&
        chain->second.encodePostValidation(chain->second.context, pass);
}

void abortTendonLoadProgramChain(void* opaque, void* commandBuffer) {
    auto* chain = static_cast<TendonLoadProgramChain*>(opaque);
    if (chain == nullptr) return;
    if (chain->second.valid()) {
        chain->second.abort(chain->second.context, commandBuffer);
    }
    if (chain->first.valid()) {
        chain->first.abort(chain->first.context, commandBuffer);
    }
}

metalrobo::MetalNumiHumanTendonLoadProgram tendonLoadProgramChain(
    TendonLoadProgramChain& chain
) {
    require(chain.first.valid() && chain.second.valid(),
            "Human tendon-load program chain is incomplete");
    return {
        .context = &chain,
        .encodePreDynamics = &encodeTendonLoadProgramChainPreDynamics,
        .encodePostValidation = &encodeTendonLoadProgramChainPostValidation,
        .abort = &abortTendonLoadProgramChain,
        .fingerprint = chain.first.fingerprint ^
            (chain.second.fingerprint + 0x9e3779b97f4a7c15ull +
             (chain.first.fingerprint << 6u) +
             (chain.first.fingerprint >> 2u)),
    };
}

bool sameFEMState(
    const numi::matter::RuntimeStateSnapshot& first,
    const numi::matter::RuntimeStateSnapshot& second
) {
    return first.available && second.available &&
        first.femNodes.size() == second.femNodes.size() &&
        (first.femNodes.empty() ||
         std::memcmp(
             first.femNodes.data(), second.femNodes.data(),
             first.femNodes.size() * sizeof(NMFEMNodeStateGPU)
         ) == 0);
}

struct HumanTendonContinuumTransaction {
    metalrobo::MetalNumiHumanTendonLoadProgram program{};
    numi::matter::Runtime* runtime = nullptr;
    numi::matter::RuntimeStateSnapshot initial;
    numi::matter::RuntimeStateSnapshot accepted;
    bool rollbackVerified = false;
    bool replayVerified = false;
};

struct FEMReactionSnapshotProgram {
    metalrobo::MetalNumiHumanTendonLoadProgram delegate{};
    __strong id<MTLBuffer> source = nil;
    __strong id<MTLBuffer> snapshot = nil;
    NSUInteger byteCount = 0u;
};

bool encodeFEMReactionSnapshotPreDynamics(
    void* opaque,
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    auto* program = static_cast<FEMReactionSnapshotProgram*>(opaque);
    if (program == nullptr || !program->delegate.valid() ||
        pass.commandBuffer == nullptr || program->source == nil ||
        program->snapshot == nil || program->byteCount == 0u ||
        !program->delegate.encodePreDynamics(
            program->delegate.context, pass)) {
        return false;
    }
    id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) return false;
    [blit copyFromBuffer:program->source sourceOffset:0u
                toBuffer:program->snapshot destinationOffset:0u
                    size:program->byteCount];
    [blit endEncoding];
    return true;
}

bool encodeFEMReactionSnapshotPostValidation(
    void* opaque,
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    auto* program = static_cast<FEMReactionSnapshotProgram*>(opaque);
    return program != nullptr && program->delegate.valid() &&
        program->delegate.encodePostValidation(
            program->delegate.context, pass);
}

void abortFEMReactionSnapshot(
    void* opaque, void* commandBuffer
) noexcept {
    auto* program = static_cast<FEMReactionSnapshotProgram*>(opaque);
    if (program != nullptr && program->delegate.valid()) {
        program->delegate.abort(
            program->delegate.context, commandBuffer);
    }
}

metalrobo::MetalNumiHumanTendonLoadProgram femReactionSnapshotProgram(
    FEMReactionSnapshotProgram& snapshot
) {
    require(snapshot.delegate.valid() && snapshot.source != nil &&
                snapshot.snapshot != nil && snapshot.byteCount > 0u,
            "FEM reaction snapshot program is incomplete");
    return {
        .context = &snapshot,
        .encodePreDynamics = &encodeFEMReactionSnapshotPreDynamics,
        .encodePostValidation = &encodeFEMReactionSnapshotPostValidation,
        .abort = &abortFEMReactionSnapshot,
        .fingerprint = snapshot.delegate.fingerprint ^
            0x4e48465245414354ull,
    };
}

bool encodeTendonLoadAuditPreDynamics(
    void* opaque,
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    auto* audit = static_cast<TendonLoadAuditConsumer*>(opaque);
    if (audit == nullptr || pass.commandBuffer == nullptr) return false;
    if (!audit->reject) return true;
    ++audit->encodedPassCount;
    return false;
}

bool encodeTendonLoadAuditPostValidation(
    void* opaque,
    const metalrobo::MetalNumiHumanTendonLoadPass& pass
) {
    auto* audit = static_cast<TendonLoadAuditConsumer*>(opaque);
    if (audit == nullptr || pass.commandBuffer == nullptr ||
        pass.bindings == nullptr || pass.envelopes == nullptr ||
        pass.transfers == nullptr || pass.generalizedCorrections == nullptr ||
        pass.bodyPoses == nullptr || pass.standStatuses == nullptr ||
        pass.environmentCount == 0u || pass.endpointCount == 0u ||
        pass.dofCount == 0u) {
        return false;
    }
    ++audit->encodedPassCount;
    id<MTLCommandBuffer> command =
        (__bridge id<MTLCommandBuffer>)pass.commandBuffer;
    id<MTLBuffer> transfers = (__bridge id<MTLBuffer>)pass.transfers;
    id<MTLBuffer> corrections =
        (__bridge id<MTLBuffer>)pass.generalizedCorrections;
    id<MTLBuffer> statuses = (__bridge id<MTLBuffer>)pass.standStatuses;
    const NSUInteger transferBytes = static_cast<NSUInteger>(
        pass.environmentCount * pass.endpointCount *
        sizeof(MRNumiHumanTendonTransferResultGPU)
    );
    const NSUInteger correctionBytes = static_cast<NSUInteger>(
        pass.environmentCount * pass.endpointCount * pass.dofCount *
        sizeof(float)
    );
    const NSUInteger statusBytes = static_cast<NSUInteger>(
        pass.environmentCount * sizeof(MRNumiHumanStandStatusGPU)
    );
    if (audit->transferSnapshot == nil) {
        audit->transferSnapshot = [transfers.device
            newBufferWithLength:transferBytes
                       options:MTLResourceStorageModeShared];
        audit->correctionSnapshot = [transfers.device
            newBufferWithLength:correctionBytes
                       options:MTLResourceStorageModeShared];
        audit->statusSnapshot = [transfers.device
            newBufferWithLength:statusBytes
                       options:MTLResourceStorageModeShared];
    }
    if (audit->transferSnapshot == nil || audit->correctionSnapshot == nil ||
        audit->statusSnapshot == nil ||
        audit->transferSnapshot.length != transferBytes ||
        audit->correctionSnapshot.length != correctionBytes ||
        audit->statusSnapshot.length != statusBytes) {
        return false;
    }
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (blit == nil) return false;
    [blit copyFromBuffer:transfers sourceOffset:0u
                toBuffer:audit->transferSnapshot destinationOffset:0u
                    size:transferBytes];
    [blit copyFromBuffer:corrections sourceOffset:0u
                toBuffer:audit->correctionSnapshot destinationOffset:0u
                    size:correctionBytes];
    [blit copyFromBuffer:statuses sourceOffset:0u
                toBuffer:audit->statusSnapshot destinationOffset:0u
                    size:statusBytes];
    [blit endEncoding];
    return true;
}

void abortTendonLoadAudit(void* opaque, void*) {
    auto* audit = static_cast<TendonLoadAuditConsumer*>(opaque);
    if (audit != nullptr) ++audit->abortCount;
}

struct MetalMujocoVisualQueries {
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::vector<MRNumiHumanStandContactGPU> supportContacts;
    std::uint32_t bodyJacobianPointOffset = MR_INVALID_INDEX;
};

MetalMujocoVisualQueries makeMetalMujocoVisualQueries(
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts* support = nullptr
) {
    require(
        model.articulations.size() == 1u &&
            model.articulations.front().bodyCount == model.bodies.size() &&
            model.bodies.size() <=
                MR_ARTICULATED_OPERATOR_MAX_POINTS / 4u,
        "MyoSim Metal visual force queries exceed articulated-operator capacity"
    );
    MetalMujocoVisualQueries result;
    result.bodyJacobianPointOffset = 0u;
    result.points.reserve(
        4u * model.bodies.size() +
        (support == nullptr ? 0u : support->records.size())
    );
    for (std::uint32_t body = 0u;
         body < model.bodies.size();
         ++body) {
        for (std::uint32_t probe = 0u; probe < 4u; ++probe) {
            MRArticulatedPointImpulseGPU point{};
            point.bodyIndex = body;
            point.localPoint = probe == 0u
                ? mr_float4{0.0f, 0.0f, 0.0f, 0.0f}
                : (probe == 1u
                    ? mr_float4{1.0f, 0.0f, 0.0f, 0.0f}
                    : (probe == 2u
                        ? mr_float4{0.0f, 1.0f, 0.0f, 0.0f}
                        : mr_float4{0.0f, 0.0f, 1.0f, 0.0f}));
            result.points.push_back(point);
        }
    }
    if (support != nullptr) {
        result.supportContacts.reserve(support->records.size());
        for (const SupportContactRecord& record : support->records) {
            MRArticulatedPointImpulseGPU point{};
            point.bodyIndex = record.bodyIndex;
            point.localPoint = {
                record.localPointX,
                record.localPointY,
                record.localPointZ,
                0.0f,
            };
            const std::uint32_t pointQueryIndex =
                static_cast<std::uint32_t>(result.points.size());
            result.points.push_back(point);
            MRNumiHumanStandContactGPU contact{};
            contact.bodyIndex = record.bodyIndex;
            contact.pointQueryIndex = pointQueryIndex;
            contact.sourceGeometryIndex = record.sourceGeometryIndex;
            contact.frictionSlopAndStabilization = {
                record.friction,
                0.002f,
                0.2f,
                0.0f,
            };
            result.supportContacts.push_back(contact);
        }
    }
    return result;
}

std::vector<float> packMetalConfiguration(
    const std::span<const double> configuration
) {
    std::vector<float> result;
    result.reserve(configuration.size());
    for (const double coordinate : configuration) {
        require(
            std::isfinite(coordinate) &&
                coordinate >= -static_cast<double>(
                    std::numeric_limits<float>::max()
                ) &&
                coordinate <= static_cast<double>(
                    std::numeric_limits<float>::max()
                ),
            "MyoSim visual configuration is not representable on Metal"
        );
        result.push_back(static_cast<float>(coordinate));
    }
    return result;
}

struct MetalMujocoForceStep {
    std::vector<float> generalizedForce;
    std::vector<float> muscleGeneralizedForces;
    std::uint32_t appliedWrapCount = 0u;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
};

MetalMujocoForceStep evaluateMetalMujocoForce(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles,
    const MetalMujocoVisualQueries& queries,
    const std::span<const double> configuration,
    std::vector<MRMujocoMuscleStateGPU>& states,
    metalrobo::MetalArticulatedOperatorContext& context
) {
    const std::vector<float> q = packMetalConfiguration(configuration);
    const metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = queries.points.size(),
        .q = q,
        .points = queries.points,
        .mujoco = {
            .muscles = muscles.gpuMuscles,
            .states = states,
            .sites = muscles.gpuSites,
            .wraps = muscles.gpuWraps,
            .routeNodes = muscles.gpuRoutes,
            .bodyJacobianPointOffset = queries.bodyJacobianPointOffset,
        },
    };
    metalrobo::MetalArticulatedOperatorResult result;
    const auto diagnostics = context.run(model, input, result);
    require(
        diagnostics.succeeded() && diagnostics.dispatched &&
            diagnostics.published &&
            diagnostics.successfulEnvironmentCount == 1u &&
            diagnostics.failedEnvironmentCount == 0u &&
            result.mujocoResults.size() == muscles.gpuMuscles.size() &&
            result.mujocoActivationStates.size() == states.size() &&
            result.mujocoGeneralizedForces.size() == model.world.nv,
        "MyoSim Metal full-body force transaction failed: " +
            diagnostics.message
    );
    MetalMujocoForceStep output;
    output.generalizedForce = std::move(result.mujocoGeneralizedForces);
    output.muscleGeneralizedForces =
        std::move(result.mujocoMuscleGeneralizedForces);
    output.elapsedMilliseconds = diagnostics.elapsedMilliseconds;
    output.deviceName = diagnostics.deviceName;
    for (const MRMujocoMuscleResultGPU& muscle : result.mujocoResults) {
        output.appliedWrapCount += muscle.appliedWrapCount;
    }
    states = std::move(result.mujocoActivationStates);
    return output;
}

std::array<double, 3u> crossProduct(
    const std::array<double, 3u>& left,
    const std::array<double, 3u>& right
) {
    return {
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    };
}

std::array<double, 3u> normalizedVector(
    const std::array<double, 3u>& value,
    const char* context
) {
    const double length = std::sqrt(
        value[0] * value[0] + value[1] * value[1] + value[2] * value[2]
    );
    require(std::isfinite(length) && length > 1.0e-12,
            std::string(context) + " is degenerate");
    return {value[0] / length, value[1] / length, value[2] / length};
}

struct GroundAlignedSupport {
    std::vector<double> q;
    std::uint32_t witnessCount = 0u;
    double seedTranslationMeters = 0.0;
};

GroundAlignedSupport makeGroundAlignedSupport(
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts& support
) {
    require(model.articulations.size() == 1u &&
                model.articulations.front().rootType == MR_ROOT_FLOATING,
            "MyoSim support contact requires one floating articulation");
    GroundAlignedSupport result;
    result.witnessCount = support.header.contactCount;
    result.q.assign(model.defaultQ.begin(), model.defaultQ.end());
    const MRArticulationGPU& articulation = model.articulations.front();
    require(articulation.qOffset + 3u <= result.q.size(),
            "MyoSim floating root position is unavailable for support alignment");
    const std::array<double, 3u> groundNormal = normalizedVector({
        support.header.groundNormalX,
        support.header.groundNormalY,
        support.header.groundNormalZ,
    }, "MyoSim source ground normal");
    double minimumGap = std::numeric_limits<double>::infinity();
    for (const SupportContactRecord& record : support.records) {
        minimumGap = std::min(
            minimumGap, static_cast<double>(record.defaultSignedPlaneDistance)
        );
    }
    require(std::isfinite(minimumGap) && minimumGap >= -1.0e-4 &&
                minimumGap <= 0.25,
            "MyoSim source support witnesses cannot form a bounded ground-aligned seed");
    for (std::size_t axis = 0u; axis < 3u; ++axis) {
        result.q[articulation.qOffset + axis] -= groundNormal[axis] * minimumGap;
    }
    result.seedTranslationMeters = minimumGap;

    return result;
}

struct DynamicSourceSupportContacts {
    std::vector<metalrobo::MultiArticulatedIslandContact> contacts;
    std::vector<std::size_t> sourceRecordIndices;
    double minimumPlaneGapMeters = std::numeric_limits<double>::infinity();
};

DynamicSourceSupportContacts makeDynamicSourceSupportContacts(
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts& support,
    const metalrobo::ArticulatedDynamicsConfig& dynamicsConfig,
    const std::span<const double> q,
    const std::span<const double> velocity,
    const std::span<const double> warmImpulses
) {
    require(model.articulations.size() == 1u && q.size() == model.world.nq &&
                velocity.size() == model.world.nv &&
                warmImpulses.size() == 3u * support.records.size(),
            "MyoSim dynamic source-support dimensions are inconsistent");
    const std::array<double, 3u> groundPoint{
        support.header.groundPointX, support.header.groundPointY, support.header.groundPointZ,
    };
    const std::array<double, 3u> groundNormal = normalizedVector({
        support.header.groundNormalX, support.header.groundNormalY, support.header.groundNormalZ,
    }, "MyoSim source ground normal");
    const std::array<double, 3u> contactNormal{
        -groundNormal[0], -groundNormal[1], -groundNormal[2],
    };
    const std::array<double, 3u> tangentReference =
        std::abs(contactNormal[0]) < 0.9
        ? std::array<double, 3u>{1.0, 0.0, 0.0}
        : std::array<double, 3u>{0.0, 1.0, 0.0};
    const double normalReferenceDot =
        contactNormal[0] * tangentReference[0] +
        contactNormal[1] * tangentReference[1] +
        contactNormal[2] * tangentReference[2];
    const std::array<double, 3u> tangentU = normalizedVector({
        tangentReference[0] - normalReferenceDot * contactNormal[0],
        tangentReference[1] - normalReferenceDot * contactNormal[1],
        tangentReference[2] - normalReferenceDot * contactNormal[2],
    }, "MyoSim dynamic support tangent");
    const std::array<double, 3u> tangentV = crossProduct(contactNormal, tangentU);
    std::vector<metalrobo::ArticulatedPointQuery> queries;
    queries.reserve(support.records.size());
    for (const SupportContactRecord& record : support.records) {
        queries.push_back({
            record.bodyIndex, {record.localPointX, record.localPointY, record.localPointZ},
        });
    }
    std::vector<metalrobo::ArticulatedPointKinematics> points(queries.size());
    std::vector<double> jacobians(queries.size() * 3u * model.world.nv, 0.0);
    const auto kinematics = metalrobo::computeArticulatedPointJacobians(
        model, 0u, q, velocity, queries, points, jacobians, dynamicsConfig
    );
    require(kinematics.succeeded(),
            "MyoSim dynamic source-support point kinematics failed");
    DynamicSourceSupportContacts result;
    constexpr double kActivationDistanceMeters = 0.002;
    for (std::size_t index = 0u; index < support.records.size(); ++index) {
        const SupportContactRecord& record = support.records[index];
        const std::array<double, 3u>& position = points[index].position;
        const double gap =
            (position[0] - groundPoint[0]) * groundNormal[0] +
            (position[1] - groundPoint[1]) * groundNormal[1] +
            (position[2] - groundPoint[2]) * groundNormal[2];
        require(std::isfinite(gap), "MyoSim dynamic source-support plane gap is non-finite");
        result.minimumPlaneGapMeters = std::min(result.minimumPlaneGapMeters, gap);
        if (gap > kActivationDistanceMeters) continue;
        require(record.friction > 0.0f,
                "MyoSim authored support contact has no tangential friction");
        const double normalRecoveryVelocity = gap < 0.0
            ? std::min(-gap / dynamicsConfig.timestep, 2.0)
            : 0.0;
        result.contacts.push_back({
            .endpointA = {
                metalrobo::MultiContactEndpointKind::articulatedBody,
                record.bodyIndex,
                {record.localPointX, record.localPointY, record.localPointZ},
            },
            // This is reprojected every step so the source witness remains a
            // unilateral plane contact rather than becoming a hidden fixed
            // foot weld at its initial world-space tangent coordinates.
            .endpointB = {
                metalrobo::MultiContactEndpointKind::staticWorld,
                MR_INVALID_INDEX,
                {
                    position[0] - gap * groundNormal[0],
                    position[1] - gap * groundNormal[1],
                    position[2] - gap * groundNormal[2],
                },
            },
            .normal = contactNormal,
            .tangentU = tangentU,
            .tangentV = tangentV,
            .targetVelocity = {normalRecoveryVelocity, 0.0, 0.0},
            .regularization = {1.0e-8, 1.0e-8, 1.0e-8},
            .warmImpulse = {
                warmImpulses[3u * index], warmImpulses[3u * index + 1u],
                warmImpulses[3u * index + 2u],
            },
            .friction = record.friction,
        });
        result.sourceRecordIndices.push_back(index);
    }
    return result;
}

struct CompiledStandActivation {
    std::vector<double> q;
    std::vector<float> activation;
    std::vector<double> generalizedMuscleForce;
    std::vector<double> generalizedPositionLimitForce;
    std::vector<double> generalizedSupportForce;
    std::vector<double> generalizedPassiveCoordinateForce;
    std::vector<double> gravityTarget;
    std::vector<double> generalizedForceResidual;
    std::vector<double> generalizedAccelerationResidual;
    std::vector<double> muscleTendonForce;
    std::vector<double> passiveMuscleTendonForce;
    std::vector<double> supportNormalForce;
    std::uint32_t activeMuscleCount = 0u;
    std::uint32_t activationSweeps = 0u;
    std::uint32_t globalActivationPolishIterations = 0u;
    std::uint32_t acceptedGlobalActivationPolishSteps = 0u;
    std::uint32_t recruitedMuscleCount = 0u;
    std::uint32_t activePositionLimitCount = 0u;
    std::uint32_t acceptedPoseSteps = 0u;
    double normalizedResidualRms = 0.0;
    double initialNormalizedResidualRms = 0.0;
    double maximumAccelerationResidual = 0.0;
    double maximumVelocityIncrement = 0.0;
    bool balanced = false;
    double maximumActivation = 0.0;
    double maximumEqualityReaction = 0.0;
    double maximumLimitReaction = 0.0;
    std::uint32_t supportContactCount = 0u;
    std::uint32_t activeSupportContactCount = 0u;
    double totalSupportForceNewtons = 0.0;
    double maximumRootForceResidual = 0.0;
    double maximumRootAccelerationResidual = 0.0;
};

std::vector<metalrobo::NumiHumanPassiveCoordinateCoupling>
wholeBodyUpperPassiveCoordinateCouplings() {
    using Coupling = metalrobo::NumiHumanPassiveCoordinateCoupling;
    std::vector<Coupling> result;
    const auto add = [&result](const std::uint32_t target,
                               const std::uint32_t source,
                               const double stiffness) {
        result.push_back({target, source, 0.0, stiffness});
    };
    // Experimental mean wrist stiffness matrix [flexion, deviation] in
    // Nm/rad: [[1.28, -0.18], [-0.18, 1.74]]. MyoSim orders each wrist as
    // deviation then flexion; preserve the symmetric coupling bilaterally.
    for (const auto [deviation, flexion] :
         {std::pair{40u, 41u}, std::pair{78u, 79u}}) {
        add(flexion, flexion, 1.28);
        add(flexion, deviation, -0.18);
        add(deviation, flexion, -0.18);
        add(deviation, deviation, 1.74);
    }
    // Experimentally identified middle-finger linearized stiffnesses are the
    // explicit v1 fallback for all four non-thumb rays where digit-specific
    // passive data are absent: MCP flexion 0.054261, PIP 0.0231, DIP
    // 0.0037206, and MCP ab/adduction 0.1779 Nm/rad.
    for (const std::uint32_t sideOffset : {0u, 38u}) {
        for (const std::uint32_t base : {46u, 50u, 54u, 58u}) {
            add(base + sideOffset, base + sideOffset, 0.054261);
            add(base + 1u + sideOffset, base + 1u + sideOffset, 0.1779);
            add(base + 2u + sideOffset, base + 2u + sideOffset, 0.0231);
            add(base + 3u + sideOffset, base + 3u + sideOffset, 0.0037206);
        }
    }
    return result;
}

CompiledStandActivation compileStaticStandActivation(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles,
    const LoadedJointEqualities& jointEqualities,
    const std::span<const double> q,
    const double activationCap,
    const std::span<const std::uint32_t> selectedSourceMuscleIndices,
    const LoadedSupportContacts* supportContacts = nullptr,
    const std::span<const metalrobo::NumiHumanPassiveCoordinateCoupling>
        passiveCouplings = {},
    const std::uint32_t activationSweeps = 240u
) {
    require(muscles.header.payloadAbi == kMusclePayloadAbi &&
                muscles.referenceArchitectures.size() ==
                    muscles.referenceMuscles.size() &&
                !jointEqualities.payload.records.empty(),
            "stand equilibrium requires NHMYO2 architecture and NHEQ1 constraints");
    metalrobo::NumiHumanMuscleEquilibriumConfig config;
    config.timestep = 1.0e-4;
    config.activationLimit = activationCap;
    config.activationSamples = 65u;
    config.activationSweeps = activationSweeps;
    if (!passiveCouplings.empty()) {
        config.poseSweeps = 12u;
        config.poseCandidateCount = 12u;
        config.poseRecruitmentCandidateCount = 3u;
    }
    std::vector<metalrobo::NumiHumanStaticSupportContact> staticSupports;
    if (supportContacts != nullptr) {
        const std::array<double, 3u> normal = normalizedVector({
            supportContacts->header.groundNormalX,
            supportContacts->header.groundNormalY,
            supportContacts->header.groundNormalZ,
        }, "static Human support normal");
        staticSupports.reserve(supportContacts->records.size());
        for (const SupportContactRecord& record : supportContacts->records) {
            staticSupports.push_back({
                .bodyIndex = record.bodyIndex,
                .localPoint = {
                    record.localPointX, record.localPointY,
                    record.localPointZ,
                },
                .normal = normal,
            });
        }
    }
    metalrobo::NumiHumanMuscleEquilibriumResult compiled;
    const auto diagnostics = staticSupports.empty()
        ? metalrobo::compileNumiHumanMuscleEquilibrium(
            model, 0u, q, muscles.referenceSites, muscles.referenceWraps,
            muscles.referenceMuscles, muscles.referenceArchitectures,
            jointEqualities.payload.records, selectedSourceMuscleIndices,
            compiled, config)
        : metalrobo::compileNumiHumanMuscleEquilibrium(
            model, 0u, q, muscles.referenceSites, muscles.referenceWraps,
            muscles.referenceMuscles, muscles.referenceArchitectures,
            jointEqualities.payload.records, selectedSourceMuscleIndices,
            staticSupports, passiveCouplings, compiled, config);
    require(
        diagnostics.succeeded() &&
            (staticSupports.empty()
                ? diagnostics.normalizedResidualRms <
                    diagnostics.initialNormalizedResidualRms
                : diagnostics.normalizedResidualRms <=
                    diagnostics.initialNormalizedResidualRms + 1.0e-12) &&
            diagnostics.maximumGeneralizedAccelerationResidual *
                config.timestep <= 2.5e-2,
        std::string("source-constrained stand equilibrium failed: ") +
            metalrobo::numiHumanMuscleEquilibriumStatusName(
                diagnostics.status
            ) + " residual=" +
            std::to_string(diagnostics.normalizedResidualRms) +
            " initial=" +
            std::to_string(diagnostics.initialNormalizedResidualRms) +
            " max_acceleration=" +
            std::to_string(
                diagnostics.maximumGeneralizedAccelerationResidual) +
            " root_acceleration=" +
            std::to_string(
                diagnostics.maximumFloatingRootAccelerationResidual) +
            " support_force=" +
            std::to_string(diagnostics.totalSupportForceNewtons) +
            " active_supports=" +
            std::to_string(diagnostics.activeSupportContactCount)
    );
    CompiledStandActivation result;
    result.q = std::move(compiled.q);
    result.activation.reserve(compiled.activation.size());
    for (const double value : compiled.activation) {
        result.activation.push_back(static_cast<float>(value));
    }
    result.generalizedMuscleForce =
        std::move(compiled.generalizedMuscleForce);
    result.generalizedPositionLimitForce =
        std::move(compiled.generalizedPositionLimitForce);
    result.generalizedSupportForce =
        std::move(compiled.generalizedSupportForce);
    result.generalizedPassiveCoordinateForce =
        std::move(compiled.generalizedPassiveCoordinateForce);
    result.gravityTarget = std::move(compiled.gravityTarget);
    result.generalizedForceResidual =
        std::move(compiled.generalizedForceResidual);
    result.generalizedAccelerationResidual =
        std::move(compiled.generalizedAccelerationResidual);
    result.muscleTendonForce = std::move(compiled.muscleTendonForce);
    result.passiveMuscleTendonForce =
        std::move(compiled.passiveMuscleTendonForce);
    result.supportNormalForce = std::move(compiled.supportNormalForce);
    result.activeMuscleCount = diagnostics.activeMuscleCount;
    result.activationSweeps = diagnostics.activationSweeps;
    result.globalActivationPolishIterations =
        diagnostics.globalActivationPolishIterations;
    result.acceptedGlobalActivationPolishSteps =
        diagnostics.acceptedGlobalActivationPolishSteps;
    result.recruitedMuscleCount = diagnostics.recruitedMuscleCount;
    result.activePositionLimitCount = diagnostics.activePositionLimitCount;
    result.acceptedPoseSteps = diagnostics.acceptedPoseSteps;
    result.normalizedResidualRms = diagnostics.normalizedResidualRms;
    result.initialNormalizedResidualRms =
        diagnostics.initialNormalizedResidualRms;
    result.maximumAccelerationResidual =
        diagnostics.maximumGeneralizedAccelerationResidual;
    result.maximumVelocityIncrement =
        diagnostics.maximumGeneralizedAccelerationResidual * config.timestep;
    result.balanced = diagnostics.balanced;
    result.maximumActivation = diagnostics.maximumActivation;
    result.maximumEqualityReaction = diagnostics.maximumJointEqualityReaction;
    result.maximumLimitReaction = diagnostics.maximumPositionLimitReaction;
    result.supportContactCount = diagnostics.supportContactCount;
    result.activeSupportContactCount = diagnostics.activeSupportContactCount;
    result.totalSupportForceNewtons = diagnostics.totalSupportForceNewtons;
    result.maximumRootForceResidual =
        diagnostics.maximumFloatingRootForceResidual;
    result.maximumRootAccelerationResidual =
        diagnostics.maximumFloatingRootAccelerationResidual;
    return result;
}

MuscleDrivenVisualState integratePersistentMetalHumanState(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles,
    const LoadedSupportContacts& supportContacts,
    const LoadedJointEqualities& jointEqualities,
    const double timestepSeconds,
    const std::uint32_t stepCount,
    const double activation,
    const std::span<const std::uint32_t> selectedSourceMuscleIndices,
    const bool applySelectedActivationIncrement,
    const bool enableRootAssistance,
    const bool removeRootAssistance,
    const bool verifyDeterminism,
    HumanTendonContinuumTransaction* continuumTransaction = nullptr,
    const std::optional<std::pair<std::uint32_t, double>>
        initialCoordinate = std::nullopt,
    const bool requireContinuumRigidStateEffect = true
) {
    require(std::isfinite(timestepSeconds) && timestepSeconds >= 1.0e-6 &&
                timestepSeconds <= 1.0e-3 && stepCount >= 1u &&
                stepCount <= MR_NUMI_HUMAN_STAND_MAX_STEPS &&
                std::isfinite(activation) && activation >= 0.0 &&
                activation <= 1.0,
            "persistent Human stand horizon has an invalid timestep or step count");
    require(model.articulations.size() == 1u && model.world.nq ==
                model.articulations.front().nq && model.world.nv ==
                model.articulations.front().nv,
            "persistent Human stand currently requires one complete articulation");
    require(muscles.tendonPayload.payloadAbi == 2u || muscles.tendonPayload.payloadAbi == 3u,
            "persistent Human control requires an NHTENDON2/3 per-step terminal-load payload");
    require(!applySelectedActivationIncrement ||
                (muscles.header.payloadAbi == kMusclePayloadAbi &&
                 !selectedSourceMuscleIndices.empty() &&
                 std::is_sorted(
                     selectedSourceMuscleIndices.begin(),
                     selectedSourceMuscleIndices.end()
                 ) &&
                 std::adjacent_find(
                     selectedSourceMuscleIndices.begin(),
                     selectedSourceMuscleIndices.end()
                 ) == selectedSourceMuscleIndices.end() &&
                 std::all_of(
                     selectedSourceMuscleIndices.begin(),
                     selectedSourceMuscleIndices.end(),
                     [&muscles](const std::uint32_t index) {
                         return index < muscles.gpuMuscles.size();
                     }
                 )),
            "selected Human tendon control requires NHMYO2 and an explicit source-muscle subset");
    require(!removeRootAssistance || enableRootAssistance,
            "assistance removal requires an assisted stand phase");
    require(continuumTransaction == nullptr ||
                (continuumTransaction->program.valid() &&
                 continuumTransaction->runtime != nullptr &&
                 continuumTransaction->runtime->valid() &&
                 continuumTransaction->initial.available),
            "persistent Human continuum transaction is incomplete");
    GroundAlignedSupport aligned =
        makeGroundAlignedSupport(model, supportContacts);
    if (initialCoordinate.has_value()) {
        const auto [qIndex, value] = *initialCoordinate;
        const auto dof = std::find_if(
            model.dofs.begin(), model.dofs.end(),
            [qIndex](const MRDofPropertiesGPU& candidate) {
                return candidate.qIndex == qIndex;
            });
        require(qIndex < aligned.q.size() && dof != model.dofs.end() &&
                    std::isfinite(value) &&
                    ((dof->flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
                     (value >= static_cast<double>(dof->limits.x) &&
                      value <= static_cast<double>(dof->limits.y))),
                "persistent Human initial coordinate is invalid");
        aligned.q[qIndex] = value;
        double maximumProjection = 0.0;
        const auto projected = metalrobo::projectNumiHumanJointEqualities(
            jointEqualities.payload.records, aligned.q, &maximumProjection);
        require(projected.succeeded() &&
                    std::isfinite(maximumProjection),
                "persistent Human initial coordinate equality projection failed");
    }
    CompiledStandActivation compiledActivation;
    std::vector<float> selectedControlBaselineActivation;
    if (applySelectedActivationIncrement) {
        compiledActivation = compileStaticStandActivation(
            model,
            muscles,
            jointEqualities,
            aligned.q,
            1.0,
            {},
            &supportContacts
        );
        selectedControlBaselineActivation = compiledActivation.activation;
        for (const std::uint32_t muscleIndex : selectedSourceMuscleIndices) {
            compiledActivation.activation[muscleIndex] = std::min(
                1.0f,
                compiledActivation.activation[muscleIndex] +
                    static_cast<float>(activation)
            );
        }
    } else {
        compiledActivation = compileStaticStandActivation(
            model,
            muscles,
            jointEqualities,
            aligned.q,
            activation,
            selectedSourceMuscleIndices,
            &supportContacts
        );
    }
    if (initialCoordinate.has_value()) {
        // Recruitment chooses actuator values. Tissue qualification owns an
        // explicit projected support pose, so do not let the recruiter's
        // optional pose-relaxation stage silently move patellar dependents.
        compiledActivation.q = aligned.q;
    }
    const std::vector<float> q = packMetalConfiguration(compiledActivation.q);
    std::vector<float> v;
    v.reserve(model.defaultV.size());
    for (const double velocity : model.defaultV) {
        require(std::isfinite(velocity),
                "persistent Human stand default velocity is non-finite");
        v.push_back(static_cast<float>(velocity));
    }
    const MetalMujocoVisualQueries queries =
        makeMetalMujocoVisualQueries(model, &supportContacts);
    std::vector<MRMujocoMuscleStateGPU> states(muscles.gpuMuscles.size());
    for (std::size_t muscleIndex = 0u;
         muscleIndex < states.size(); ++muscleIndex) {
        const float initialActivation =
            compiledActivation.activation[muscleIndex];
        states[muscleIndex].excitationAndActivation = {
            initialActivation,
            initialActivation,
            0.0f,
            0.0f,
        };
    }
    metalrobo::NumiHumanTendonMetalProgram tendonProgram;
    const auto tendonPack = metalrobo::makeNumiHumanTendonMetalProgram(
        muscles.tendonPayload,
        tendonProgram
    );
    require(
        tendonPack.succeeded() &&
            tendonProgram.bindings.size() == 2u * muscles.gpuMuscles.size(),
        std::string("persistent Human NHTENDON2/3 packing failed: ") +
            metalrobo::numiHumanTendonStatusName(tendonPack.status)
    );
    const metalrobo::MetalArticulatedOperatorConfig config{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds = static_cast<float>(timestepSeconds),
    };
    metalrobo::MetalArticulatedOperatorContext context(config);
    metalrobo::MetalArticulatedOperatorInput input{
        .articulationIndex = 0u,
        .environmentCount = 1u,
        .pointCount = queries.points.size(),
        .q = q,
        .points = queries.points,
        .mujoco = {
            .muscles = muscles.gpuMuscles,
            .states = states,
            .sites = muscles.gpuSites,
            .wraps = muscles.gpuWraps,
            .routeNodes = muscles.gpuRoutes,
            .bodyJacobianPointOffset = queries.bodyJacobianPointOffset,
        },
        .stand = {
            .v = v,
            .contacts = queries.supportContacts,
            .jointEqualities = jointEqualities.payload.records,
            .tendonBindings = tendonProgram.bindings,
            .tendonEnvelopes = tendonProgram.envelopes,
            .stepCount = stepCount,
            .contactIterationCount = 16u,
            .enableContact = true,
            .enableRootAssistance = enableRootAssistance,
            .groundPoint = {
                supportContacts.header.groundPointX,
                supportContacts.header.groundPointY,
                supportContacts.header.groundPointZ,
                0.0f,
            },
            .groundNormal = {
                supportContacts.header.groundNormalX,
                supportContacts.header.groundNormalY,
                supportContacts.header.groundNormalZ,
                0.0f,
            },
            .targetRootPosition = {q[0u], q[1u], q[2u], 0.0f},
            .targetRootOrientation = {q[3u], q[4u], q[5u], q[6u]},
            .assistanceGains = enableRootAssistance
                ? mr_float4{1800.0f, 360.0f, 700.0f, 100.0f}
                : mr_float4{0.0f, 0.0f, 0.0f, 0.0f},
        },
    };
    metalrobo::MetalArticulatedOperatorResult selectedControlBaselineResult;
    double selectedControlBaselineElapsedMilliseconds = 0.0;
    if (applySelectedActivationIncrement) {
        std::vector<MRMujocoMuscleStateGPU> baselineStates(
            muscles.gpuMuscles.size()
        );
        for (std::size_t muscleIndex = 0u;
             muscleIndex < baselineStates.size();
             ++muscleIndex) {
            const float baselineActivation =
                selectedControlBaselineActivation[muscleIndex];
            baselineStates[muscleIndex].excitationAndActivation = {
                baselineActivation,
                baselineActivation,
                0.0f,
                0.0f,
            };
        }
        metalrobo::MetalArticulatedOperatorInput baselineInput = input;
        baselineInput.mujoco.states = baselineStates;
        const auto baselineDiagnostics = context.run(
            model,
            baselineInput,
            selectedControlBaselineResult
        );
        require(
            baselineDiagnostics.succeeded() && baselineDiagnostics.published &&
                baselineDiagnostics.completedStandSteps == stepCount &&
                selectedControlBaselineResult.standQ.size() == q.size() &&
                selectedControlBaselineResult.standV.size() == v.size() &&
                selectedControlBaselineResult.standStatuses.size() == 1u &&
                selectedControlBaselineResult.standStatuses.front().code ==
                    MR_NUMI_HUMAN_STAND_SUCCESS &&
                selectedControlBaselineResult.standStatuses.front().tendonTransferCount ==
                    tendonProgram.bindings.size() * stepCount,
            "selected Human zero-increment baseline transaction failed: " +
                baselineDiagnostics.message
        );
        selectedControlBaselineElapsedMilliseconds =
            baselineDiagnostics.elapsedMilliseconds;
    }
    std::vector<double> parityQ = compiledActivation.q;
    std::vector<double> parityV(model.defaultV.begin(), model.defaultV.end());
    std::vector<double> parityGeneralizedForce =
        compiledActivation.generalizedMuscleForce;
    const bool hasCompliantArchitecture = std::any_of(
        muscles.gpuMuscles.begin(), muscles.gpuMuscles.end(),
        [](const MRMujocoMuscleGPU& muscle) {
            return muscle.compliantArchitecture0.x > 0.0f &&
                muscle.compliantArchitecture0.y > 0.0f;
        }
    );
    require(!applySelectedActivationIncrement || hasCompliantArchitecture,
            "selected Human tendon control requires compliant NHMYO2 architecture");
    if (hasCompliantArchitecture) {
        std::vector<MRMujocoMuscleStateGPU> parityForceStates = states;
        metalrobo::MetalArticulatedOperatorContext parityForceContext(config);
        const MetalMujocoForceStep parityForce = evaluateMetalMujocoForce(
            model, muscles, queries, compiledActivation.q, parityForceStates,
            parityForceContext
        );
        parityGeneralizedForce.assign(
            parityForce.generalizedForce.begin(),
            parityForce.generalizedForce.end()
        );
    }
    metalrobo::ArticulatedDynamicsConfig parityConfig;
    parityConfig.gravity = {
        model.world.gravityAndTimestep.x,
        model.world.gravityAndTimestep.y,
        model.world.gravityAndTimestep.z,
    };
    parityConfig.timestep = timestepSeconds;
    parityConfig.implicitPassiveDofDamping = true;
    const auto parityCpu = metalrobo::integrateArticulatedState(
        model,
        0u,
        parityQ,
        parityV,
        parityGeneralizedForce,
        {},
        parityConfig
    );
    require(parityCpu.succeeded(),
            "persistent Human stand CPU one-step parity reference failed");
    metalrobo::MetalArticulatedOperatorInput parityInput = input;
    parityInput.stand.stepCount = 1u;
    parityInput.stand.enableContact = false;
    parityInput.stand.jointEqualities = {};
    parityInput.stand.enableRootAssistance = false;
    parityInput.stand.assistanceGains = {0.0f, 0.0f, 0.0f, 0.0f};
    metalrobo::MetalArticulatedOperatorResult parityResult;
    const auto parityDiagnostics = context.run(model, parityInput, parityResult);
    require(parityDiagnostics.succeeded() && parityDiagnostics.published &&
                parityDiagnostics.completedStandSteps == 1u &&
                parityResult.standQ.size() == parityQ.size() &&
                parityResult.standV.size() == parityV.size() &&
                (tendonProgram.bindings.empty() ||
                 (parityResult.standTendonTransfers.size() ==
                      tendonProgram.bindings.size() &&
                  parityResult.standTendonGeneralizedCorrections.size() ==
                      tendonProgram.bindings.size() *
                          model.articulations.front().nv &&
                  parityResult.standStatuses.front().tendonTransferCount ==
                      tendonProgram.bindings.size())),
            "persistent Human stand Metal one-step parity transaction failed: " +
                parityDiagnostics.message);
    double parityMaximumQError = 0.0;
    double parityMaximumVError = 0.0;
    for (std::size_t index = 0u; index < parityQ.size(); ++index) {
        parityMaximumQError = std::max(
            parityMaximumQError,
            std::abs(static_cast<double>(parityResult.standQ[index]) - parityQ[index])
        );
    }
    for (std::size_t index = 0u; index < parityV.size(); ++index) {
        parityMaximumVError = std::max(
            parityMaximumVError,
            std::abs(static_cast<double>(parityResult.standV[index]) - parityV[index])
        );
    }
    require(std::isfinite(parityMaximumQError) &&
                std::isfinite(parityMaximumVError) &&
                parityMaximumQError <= 2.0e-6 &&
                parityMaximumVError <= 2.0e-3,
            "persistent Human stand one-step Metal/FP64 parity exceeded tolerance: q=" +
                std::to_string(parityMaximumQError) + " v=" +
                std::to_string(parityMaximumVError));
    bool tendonRollbackVerified = false;
    bool tendonRigidStateIdentityVerified = false;
    TendonLoadAuditConsumer acceptedTendonConsumer;
    TendonLoadProgramChain acceptedProgramChain;
    metalrobo::MetalArticulatedOperatorResult sourceJTOnlyResult;
    bool sourceJTOnlyAvailable = false;
    if (!tendonProgram.bindings.empty()) {
        metalrobo::MetalArticulatedOperatorInput noTendonInput = parityInput;
        noTendonInput.stand.tendonBindings = {};
        noTendonInput.stand.tendonEnvelopes = {};
        metalrobo::MetalArticulatedOperatorResult noTendonResult;
        const auto noTendonDiagnostics = context.run(
            model, noTendonInput, noTendonResult
        );
        tendonRigidStateIdentityVerified =
            noTendonDiagnostics.succeeded() && noTendonDiagnostics.published &&
                noTendonResult.standQ.size() == parityResult.standQ.size() &&
                noTendonResult.standV.size() == parityResult.standV.size() &&
                std::memcmp(
                    noTendonResult.standQ.data(), parityResult.standQ.data(),
                    parityResult.standQ.size() * sizeof(float)
                ) == 0 &&
                std::memcmp(
                    noTendonResult.standV.data(), parityResult.standV.data(),
                    parityResult.standV.size() * sizeof(float)
                ) == 0;
        require(
            tendonRigidStateIdentityVerified,
            "persistent Human tendon transfer changed rigid q/v or acted as joint torque"
        );

        TendonLoadAuditConsumer rejectingConsumer;
        rejectingConsumer.reject = true;
        const metalrobo::MetalNumiHumanTendonLoadProgram rejectingAuditProgram{
            .context = &rejectingConsumer,
            .encodePreDynamics = &encodeTendonLoadAuditPreDynamics,
            .encodePostValidation = &encodeTendonLoadAuditPostValidation,
            .abort = &abortTendonLoadAudit,
            .fingerprint = 0x4e4854454e444f4eull,
        };
        TendonLoadProgramChain rejectingProgramChain;
        metalrobo::MetalArticulatedOperatorInput rejectedInput = parityInput;
        if (continuumTransaction == nullptr) {
            rejectedInput.stand.tendonLoadProgram = rejectingAuditProgram;
        } else {
            // Encode the continuum first, then force the downstream audit to
            // reject. The enclosing Human submission must call abort, and
            // Matter must retain its exact pre-transaction accepted state.
            rejectingProgramChain.first = continuumTransaction->program;
            rejectingProgramChain.second = rejectingAuditProgram;
            rejectedInput.stand.tendonLoadProgram = tendonLoadProgramChain(
                rejectingProgramChain
            );
        }
        metalrobo::MetalArticulatedOperatorResult rejectedResult;
        rejectedResult.standQ = {-321.25f};
        const auto rejectedDiagnostics = context.run(
            model, rejectedInput, rejectedResult
        );
        tendonRollbackVerified =
            !rejectedDiagnostics.succeeded() &&
            !rejectedDiagnostics.dispatched &&
            !rejectedDiagnostics.published &&
            rejectedResult.standQ.size() == 1u &&
            rejectedResult.standQ.front() == -321.25f &&
            rejectingConsumer.encodedPassCount == 1u &&
            rejectingConsumer.abortCount == 1u;
        if (continuumTransaction != nullptr) {
            const auto rejectedContinuum =
                continuumTransaction->runtime->snapshot();
            continuumTransaction->rollbackVerified = sameFEMState(
                rejectedContinuum, continuumTransaction->initial
            );
            tendonRollbackVerified = tendonRollbackVerified &&
                continuumTransaction->rollbackVerified;
        }
        require(
            tendonRollbackVerified,
            "persistent Human tendon consumer rejection did not roll back: " +
                rejectedDiagnostics.message +
                " succeeded=" +
                std::to_string(rejectedDiagnostics.succeeded()) +
                " dispatched=" +
                std::to_string(rejectedDiagnostics.dispatched) +
                " published=" +
                std::to_string(rejectedDiagnostics.published) +
                " audit_passes=" +
                std::to_string(rejectingConsumer.encodedPassCount) +
                " audit_aborts=" +
                std::to_string(rejectingConsumer.abortCount) +
                " continuum_rollback=" +
                std::to_string(
                    continuumTransaction == nullptr ||
                    continuumTransaction->rollbackVerified)
        );

        if (continuumTransaction != nullptr) {
            const auto sourceJTOnlyDiagnostics = context.run(
                model, input, sourceJTOnlyResult
            );
            sourceJTOnlyAvailable =
                sourceJTOnlyDiagnostics.succeeded() &&
                sourceJTOnlyDiagnostics.published &&
                sourceJTOnlyDiagnostics.completedStandSteps == stepCount &&
                sourceJTOnlyResult.standQ.size() == q.size() &&
                sourceJTOnlyResult.standV.size() == v.size();
            require(sourceJTOnlyAvailable,
                    "source J^T-only Human comparison transaction failed");
        }

        const metalrobo::MetalNumiHumanTendonLoadProgram acceptedAuditProgram{
            .context = &acceptedTendonConsumer,
            .encodePreDynamics = &encodeTendonLoadAuditPreDynamics,
            .encodePostValidation = &encodeTendonLoadAuditPostValidation,
            .abort = &abortTendonLoadAudit,
            .fingerprint = 0x4e4854454e444f4eull,
        };
        if (continuumTransaction == nullptr) {
            input.stand.tendonLoadProgram = acceptedAuditProgram;
        } else {
            acceptedProgramChain.first = acceptedAuditProgram;
            acceptedProgramChain.second = continuumTransaction->program;
            input.stand.tendonLoadProgram = tendonLoadProgramChain(
                acceptedProgramChain
            );
        }
    }
    metalrobo::MetalArticulatedOperatorResult metalResult;
    auto diagnostics = context.run(model, input, metalResult);
    require(
        diagnostics.succeeded() && diagnostics.dispatched &&
            diagnostics.published && diagnostics.successfulEnvironmentCount == 1u &&
            diagnostics.failedEnvironmentCount == 0u &&
            diagnostics.completedStandSteps == stepCount &&
            metalResult.standQ.size() == q.size() &&
            metalResult.standV.size() == v.size() &&
            metalResult.standStatuses.size() == 1u &&
            (tendonProgram.bindings.empty() ||
             (metalResult.standTendonTransfers.size() ==
                  tendonProgram.bindings.size() &&
              metalResult.standTendonGeneralizedCorrections.size() ==
                  tendonProgram.bindings.size() *
                      model.articulations.front().nv)),
        "persistent Human stand Metal horizon failed: " + diagnostics.message
    );
    const MRNumiHumanStandStatusGPU& status = metalResult.standStatuses.front();
    require(status.code == MR_NUMI_HUMAN_STAND_SUCCESS &&
                status.completedSteps == stepCount &&
                status.jointEqualityCounts.x ==
                    jointEqualities.payload.records.size() &&
                status.jointEqualityCounts.z == 0u,
            "persistent Human stand returned an incomplete device status");
    double tendonContinuumMaximumQDelta = 0.0;
    double tendonContinuumMaximumVDelta = 0.0;
    bool tendonContinuumReactionVerified = false;
    if (continuumTransaction != nullptr) {
        require(sourceJTOnlyAvailable,
                "continuum reaction has no source J^T-only comparison");
        for (std::size_t index = 0u; index < metalResult.standQ.size(); ++index) {
            tendonContinuumMaximumQDelta = std::max(
                tendonContinuumMaximumQDelta,
                std::abs(static_cast<double>(metalResult.standQ[index]) -
                    static_cast<double>(sourceJTOnlyResult.standQ[index]))
            );
        }
        for (std::size_t index = 0u; index < metalResult.standV.size(); ++index) {
            tendonContinuumMaximumVDelta = std::max(
                tendonContinuumMaximumVDelta,
                std::abs(static_cast<double>(metalResult.standV[index]) -
                    static_cast<double>(sourceJTOnlyResult.standV[index]))
            );
        }
        tendonContinuumReactionVerified =
            std::isfinite(tendonContinuumMaximumQDelta) &&
            std::isfinite(tendonContinuumMaximumVDelta) &&
            (tendonContinuumMaximumQDelta > 1.0e-12 ||
             tendonContinuumMaximumVDelta > 1.0e-9);
        require(!requireContinuumRigidStateEffect ||
                    tendonContinuumReactionVerified,
                "continuum anchor reactions did not change Human q/v from source J^T-only dynamics");
    }
    double selectedControlBaselineMaximumQDelta = 0.0;
    double selectedControlBaselineMaximumVDelta = 0.0;
    std::uint32_t selectedControlBaselineMaximumQDeltaIndex = MR_INVALID_INDEX;
    std::uint32_t selectedControlBaselineMaximumVDeltaIndex = MR_INVALID_INDEX;
    if (applySelectedActivationIncrement) {
        for (std::size_t index = 0u; index < metalResult.standQ.size(); ++index) {
            const double delta = std::abs(
                static_cast<double>(metalResult.standQ[index]) -
                static_cast<double>(selectedControlBaselineResult.standQ[index])
            );
            if (delta > selectedControlBaselineMaximumQDelta) {
                selectedControlBaselineMaximumQDelta = delta;
                selectedControlBaselineMaximumQDeltaIndex =
                    static_cast<std::uint32_t>(index);
            }
        }
        for (std::size_t index = 0u; index < metalResult.standV.size(); ++index) {
            const double delta = std::abs(
                static_cast<double>(metalResult.standV[index]) -
                static_cast<double>(selectedControlBaselineResult.standV[index])
            );
            if (delta > selectedControlBaselineMaximumVDelta) {
                selectedControlBaselineMaximumVDelta = delta;
                selectedControlBaselineMaximumVDeltaIndex =
                    static_cast<std::uint32_t>(index);
            }
        }
        require(
            activation == 0.0 ||
                selectedControlBaselineMaximumQDelta > 1.0e-12 ||
                selectedControlBaselineMaximumVDelta > 1.0e-9,
            "selected Human activation increment did not change q or v from its zero-increment baseline"
        );
    }
    const std::size_t tendonEnvelopeBindingCount =
        static_cast<std::size_t>(std::count_if(
            tendonProgram.bindings.begin(), tendonProgram.bindings.end(),
            [](const MRNumiHumanTendonBindingGPU& binding) {
                return binding.mode ==
                    MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE;
            }
        ));
    bool tendonBorrowedConsumerVerified = false;
    if (!tendonProgram.bindings.empty()) {
        tendonBorrowedConsumerVerified =
            status.tendonTransferCount ==
                tendonProgram.bindings.size() * stepCount &&
            status.tendonEnvelopeTransferCount ==
                tendonEnvelopeBindingCount * stepCount &&
            status.tendonPointTransferCount ==
                (tendonProgram.bindings.size() - tendonEnvelopeBindingCount) *
                    stepCount &&
            status.tendonFailureCount == 0u &&
            acceptedTendonConsumer.encodedPassCount == stepCount &&
            acceptedTendonConsumer.abortCount == 0u &&
            acceptedTendonConsumer.transferSnapshot != nil &&
            acceptedTendonConsumer.correctionSnapshot != nil &&
            acceptedTendonConsumer.statusSnapshot != nil &&
            std::memcmp(
                acceptedTendonConsumer.transferSnapshot.contents,
                metalResult.standTendonTransfers.data(),
                metalResult.standTendonTransfers.size() *
                    sizeof(MRNumiHumanTendonTransferResultGPU)
            ) == 0 &&
            std::memcmp(
                acceptedTendonConsumer.correctionSnapshot.contents,
                metalResult.standTendonGeneralizedCorrections.data(),
                metalResult.standTendonGeneralizedCorrections.size() *
                    sizeof(float)
            ) == 0 &&
            std::memcmp(
                acceptedTendonConsumer.statusSnapshot.contents,
                metalResult.standStatuses.data(),
                sizeof(MRNumiHumanStandStatusGPU)
            ) == 0;
        require(tendonBorrowedConsumerVerified,
                "persistent Human borrowed tendon-load snapshot disagreed with publication");
    }
    const MRNumiHumanStandStatusGPU assistedStatus = status;
    const std::vector<float> assistedQ = metalResult.standQ;
    double totalElapsedMilliseconds = diagnostics.elapsedMilliseconds;
    std::uint32_t phaseCount = 1u;
    if (removeRootAssistance) {
        std::vector<float> removalQ = metalResult.standQ;
        std::vector<float> removalV = metalResult.standV;
        std::vector<MRMujocoMuscleStateGPU> removalStates =
            metalResult.mujocoActivationStates;
        input.q = removalQ;
        input.stand.v = removalV;
        input.mujoco.states = removalStates;
        input.stand.enableRootAssistance = false;
        input.stand.assistanceGains = {0.0f, 0.0f, 0.0f, 0.0f};
        metalrobo::MetalArticulatedOperatorResult removalResult;
        auto removalDiagnostics = context.run(model, input, removalResult);
        require(
            removalDiagnostics.succeeded() && removalDiagnostics.dispatched &&
                removalDiagnostics.published &&
                removalDiagnostics.successfulEnvironmentCount == 1u &&
                removalDiagnostics.failedEnvironmentCount == 0u &&
                removalDiagnostics.completedStandSteps == stepCount &&
                removalResult.standQ.size() == q.size() &&
                removalResult.standV.size() == v.size() &&
                removalResult.standStatuses.size() == 1u &&
                removalResult.standStatuses.front().code ==
                    MR_NUMI_HUMAN_STAND_SUCCESS,
            "persistent Human assistance-removal horizon failed: " +
                removalDiagnostics.message
        );
        totalElapsedMilliseconds += removalDiagnostics.elapsedMilliseconds;
        diagnostics = std::move(removalDiagnostics);
        metalResult = std::move(removalResult);
        phaseCount = 2u;
    }
    const MRNumiHumanStandStatusGPU& finalStatus =
        metalResult.standStatuses.front();
    if (continuumTransaction != nullptr) {
        continuumTransaction->accepted =
            continuumTransaction->runtime->snapshot();
        require(continuumTransaction->accepted.available,
                "persistent Human continuum accepted state is unavailable");
    }
    double deterministicReplayElapsedMilliseconds = 0.0;
    bool deterministicReplayVerified = false;
    if (verifyDeterminism) {
        if (continuumTransaction != nullptr) {
            const auto restored = continuumTransaction->runtime->restore(
                continuumTransaction->initial
            );
            require(restored.encoded,
                    "persistent Human continuum replay restore failed: " +
                        restored.message);
        }
        input.q = q;
        input.stand.v = v;
        input.mujoco.states = states;
        input.stand.enableRootAssistance = enableRootAssistance;
        input.stand.assistanceGains = enableRootAssistance
            ? mr_float4{1800.0f, 360.0f, 700.0f, 100.0f}
            : mr_float4{0.0f, 0.0f, 0.0f, 0.0f};
        metalrobo::MetalArticulatedOperatorResult replayResult;
        auto replayDiagnostics = context.run(model, input, replayResult);
        require(replayDiagnostics.succeeded() && replayDiagnostics.published &&
                    replayDiagnostics.completedStandSteps == stepCount,
                "persistent Human deterministic assisted replay failed: " +
                    replayDiagnostics.message);
        deterministicReplayElapsedMilliseconds +=
            replayDiagnostics.elapsedMilliseconds;
        if (removeRootAssistance) {
            std::vector<float> replayRemovalQ = replayResult.standQ;
            std::vector<float> replayRemovalV = replayResult.standV;
            std::vector<MRMujocoMuscleStateGPU> replayRemovalStates =
                replayResult.mujocoActivationStates;
            input.q = replayRemovalQ;
            input.stand.v = replayRemovalV;
            input.mujoco.states = replayRemovalStates;
            input.stand.enableRootAssistance = false;
            input.stand.assistanceGains = {0.0f, 0.0f, 0.0f, 0.0f};
            metalrobo::MetalArticulatedOperatorResult replayRemovalResult;
            replayDiagnostics = context.run(
                model, input, replayRemovalResult
            );
            require(replayDiagnostics.succeeded() &&
                        replayDiagnostics.published &&
                        replayDiagnostics.completedStandSteps == stepCount,
                    "persistent Human deterministic removal replay failed: " +
                        replayDiagnostics.message);
            deterministicReplayElapsedMilliseconds +=
                replayDiagnostics.elapsedMilliseconds;
            replayResult = std::move(replayRemovalResult);
        }
        const bool sameQ = replayResult.standQ.size() == metalResult.standQ.size() &&
            std::memcmp(
                replayResult.standQ.data(),
                metalResult.standQ.data(),
                metalResult.standQ.size() * sizeof(float)
            ) == 0;
        const bool sameV = replayResult.standV.size() == metalResult.standV.size() &&
            std::memcmp(
                replayResult.standV.data(),
                metalResult.standV.data(),
                metalResult.standV.size() * sizeof(float)
            ) == 0;
        const bool sameStatus =
            replayResult.standStatuses.size() ==
                metalResult.standStatuses.size() &&
            std::memcmp(
                replayResult.standStatuses.data(),
                metalResult.standStatuses.data(),
                metalResult.standStatuses.size() *
                    sizeof(MRNumiHumanStandStatusGPU)
            ) == 0;
        const bool sameTendonTransfers =
            replayResult.standTendonTransfers.size() ==
                metalResult.standTendonTransfers.size() &&
            (metalResult.standTendonTransfers.empty() ||
             std::memcmp(
                 replayResult.standTendonTransfers.data(),
                 metalResult.standTendonTransfers.data(),
                 metalResult.standTendonTransfers.size() *
                     sizeof(MRNumiHumanTendonTransferResultGPU)
             ) == 0);
        const bool sameTendonCorrections =
            replayResult.standTendonGeneralizedCorrections.size() ==
                metalResult.standTendonGeneralizedCorrections.size() &&
            (metalResult.standTendonGeneralizedCorrections.empty() ||
             std::memcmp(
                 replayResult.standTendonGeneralizedCorrections.data(),
                 metalResult.standTendonGeneralizedCorrections.data(),
                 metalResult.standTendonGeneralizedCorrections.size() *
                     sizeof(float)
             ) == 0);
        require(sameQ && sameV && sameStatus && sameTendonTransfers &&
                    sameTendonCorrections,
                "persistent Human stand replay was not bitwise deterministic");
        if (continuumTransaction != nullptr) {
            const auto replayContinuum =
                continuumTransaction->runtime->snapshot();
            continuumTransaction->replayVerified = sameFEMState(
                replayContinuum, continuumTransaction->accepted
            );
            require(continuumTransaction->replayVerified,
                    "persistent Human continuum replay was not bitwise deterministic");
        }
        deterministicReplayVerified = true;
    }

    if (!tendonProgram.bindings.empty()) {
        const std::uint32_t expectedConsumerPasses =
            stepCount * phaseCount * (verifyDeterminism ? 2u : 1u);
        tendonBorrowedConsumerVerified = tendonBorrowedConsumerVerified &&
            acceptedTendonConsumer.encodedPassCount == expectedConsumerPasses &&
            acceptedTendonConsumer.abortCount == 0u &&
            std::memcmp(
                acceptedTendonConsumer.transferSnapshot.contents,
                metalResult.standTendonTransfers.data(),
                metalResult.standTendonTransfers.size() *
                    sizeof(MRNumiHumanTendonTransferResultGPU)
            ) == 0 &&
            std::memcmp(
                acceptedTendonConsumer.correctionSnapshot.contents,
                metalResult.standTendonGeneralizedCorrections.data(),
                metalResult.standTendonGeneralizedCorrections.size() *
                    sizeof(float)
            ) == 0;
        require(tendonBorrowedConsumerVerified,
                "persistent Human final borrowed tendon-load transaction diverged");
    }

    MuscleDrivenVisualState result;
    result.q = std::move(metalResult.standQ);
    result.finalTendonTransfers = metalResult.standTendonTransfers;
    constexpr std::array<std::uint32_t, 6u> kFootQIndices{
        109u, 110u, 111u, 123u, 124u, 125u,
    };
    require(compiledActivation.q.size() > kFootQIndices.back() &&
                result.q.size() > kFootQIndices.back(),
            "persistent Human source foot coordinates are unavailable");
    for (std::size_t index = 0u; index < kFootQIndices.size(); ++index) {
        result.compiledFootCoordinates[index] =
            compiledActivation.q[kFootQIndices[index]];
        result.finalFootCoordinates[index] =
            result.q[kFootQIndices[index]];
    }
    result.stepCount = stepCount;
    result.persistentMetalHorizon = true;
    result.selectedTendonControl = applySelectedActivationIncrement;
    result.selectedControlBaselineEvaluated =
        applySelectedActivationIncrement;
    result.selectedControlBaselineMaximumQDelta =
        selectedControlBaselineMaximumQDelta;
    result.selectedControlBaselineMaximumVDelta =
        selectedControlBaselineMaximumVDelta;
    result.selectedControlBaselineMaximumQDeltaIndex =
        selectedControlBaselineMaximumQDeltaIndex;
    result.selectedControlBaselineMaximumVDeltaIndex =
        selectedControlBaselineMaximumVDeltaIndex;
    result.selectedControlBaselineElapsedMilliseconds =
        selectedControlBaselineElapsedMilliseconds;
    result.rootAssistanceEnabled = enableRootAssistance;
    result.assistanceRemovalEvaluated = removeRootAssistance;
    result.persistentCompletedSteps = stepCount * phaseCount;
    result.selectedSourceMuscleActivationCount = static_cast<std::uint32_t>(
        selectedSourceMuscleIndices.size()
    );
    result.muscleMetalStepCount = phaseCount;
    result.muscleMetalForceRecordCount = static_cast<std::uint32_t>(
        muscles.gpuMuscles.size() * stepCount * phaseCount
    );
    result.muscleMetalElapsedMilliseconds = totalElapsedMilliseconds;
    result.muscleMetalDeviceName = diagnostics.deviceName;
    for (const MRMujocoMuscleResultGPU& muscle : metalResult.mujocoResults) {
        result.appliedWrapCount += muscle.appliedWrapCount;
    }
    for (std::size_t index = 0u; index < v.size(); ++index) {
        const double delta = std::abs(
            static_cast<double>(metalResult.standV[index] - v[index])
        );
        if (delta > result.maximumVelocityDelta) {
            result.maximumVelocityDelta = delta;
            result.maximumVelocityDeltaDof = static_cast<std::uint32_t>(index);
        }
    }
    for (std::size_t index = 0u; index < q.size(); ++index) {
        result.assistedConfigurationDelta = std::max(
            result.assistedConfigurationDelta,
            std::abs(static_cast<double>(assistedQ[index] - q[index]))
        );
        result.removalConfigurationDelta = std::max(
            result.removalConfigurationDelta,
            std::abs(static_cast<double>(result.q[index] - assistedQ[index]))
        );
        const double delta = std::abs(
            static_cast<double>(result.q[index] - q[index])
        );
        if (delta > result.maximumConfigurationDelta) {
            result.maximumConfigurationDelta = delta;
            result.maximumConfigurationDeltaQ =
                static_cast<std::uint32_t>(index);
        }
    }
    result.supportContactApplied = assistedStatus.maximumActiveContactCount != 0u ||
        finalStatus.maximumActiveContactCount != 0u;
    result.supportWitnessCount = supportContacts.header.contactCount;
    result.activeSupportContactCount = finalStatus.activeContactCount;
    result.maximumActiveSupportContactCount = std::max(
        assistedStatus.maximumActiveContactCount,
        finalStatus.maximumActiveContactCount
    );
    result.minimumSupportPlaneGapMeters = std::min(
        static_cast<double>(assistedStatus.contactAndAcceleration.x),
        static_cast<double>(finalStatus.contactAndAcceleration.x)
    );
    result.supportSeedTranslationMeters = aligned.seedTranslationMeters;
    result.supportGpuElapsedMilliseconds = totalElapsedMilliseconds;
    result.supportDeviceName = diagnostics.deviceName;
    result.supportMetalStatus = "persistent_large_state_metal";
    result.persistentMaximumPenetrationMeters = std::max(
        assistedStatus.contactAndAcceleration.y,
        finalStatus.contactAndAcceleration.y
    );
    result.persistentNormalImpulse = finalStatus.contactAndAcceleration.z;
    result.persistentMaximumAcceleration = std::max(
        assistedStatus.contactAndAcceleration.w,
        finalStatus.contactAndAcceleration.w
    );
    result.persistentRootAssistanceForce = assistedStatus.factorAndAssistance.z;
    result.persistentRootAssistanceTorque = assistedStatus.factorAndAssistance.w;
    result.compiledActiveMuscleCount = compiledActivation.activeMuscleCount;
    result.compiledActivationResidualRms =
        compiledActivation.normalizedResidualRms;
    result.compiledInitialActivationResidualRms =
        compiledActivation.initialNormalizedResidualRms;
    result.compiledMaximumAccelerationResidual =
        compiledActivation.maximumAccelerationResidual;
    result.compiledMaximumVelocityIncrement =
        compiledActivation.maximumVelocityIncrement;
    result.compiledBalanced = compiledActivation.balanced;
    result.compiledMaximumActivation =
        compiledActivation.maximumActivation;
    result.compiledRecruitedMuscleCount =
        compiledActivation.recruitedMuscleCount;
    result.compiledActivePositionLimitCount =
        compiledActivation.activePositionLimitCount;
    result.compiledAcceptedPoseSteps = compiledActivation.acceptedPoseSteps;
    result.compiledMaximumEqualityReaction =
        compiledActivation.maximumEqualityReaction;
    result.compiledMaximumLimitReaction =
        compiledActivation.maximumLimitReaction;
    result.compiledSupportContactCount =
        compiledActivation.supportContactCount;
    result.compiledActiveSupportContactCount =
        compiledActivation.activeSupportContactCount;
    result.compiledTotalSupportForceNewtons =
        compiledActivation.totalSupportForceNewtons;
    result.compiledMaximumRootForceResidual =
        compiledActivation.maximumRootForceResidual;
    result.compiledMaximumRootAccelerationResidual =
        compiledActivation.maximumRootAccelerationResidual;
    result.jointEqualityCount = finalStatus.jointEqualityCounts.x;
    result.maximumJointEqualityPositionError = std::max(
        assistedStatus.jointEqualityDiagnostics.x,
        finalStatus.jointEqualityDiagnostics.x
    );
    result.maximumJointEqualityVelocityError = std::max(
        assistedStatus.jointEqualityDiagnostics.y,
        finalStatus.jointEqualityDiagnostics.y
    );
    result.maximumJointEqualityImpulse = std::max(
        assistedStatus.jointEqualityDiagnostics.z,
        finalStatus.jointEqualityDiagnostics.z
    );
    result.totalJointEqualityImpulse =
        assistedStatus.jointEqualityDiagnostics.w +
        (removeRootAssistance
             ? finalStatus.jointEqualityDiagnostics.w
             : 0.0f);
    result.oneStepParityMaximumQError = parityMaximumQError;
    result.oneStepParityMaximumVError = parityMaximumVError;
    result.deterministicReplayVerified = deterministicReplayVerified;
    result.deterministicReplayElapsedMilliseconds =
        deterministicReplayElapsedMilliseconds;
    result.tendonStepTransactionEnabled = !tendonProgram.bindings.empty();
    result.tendonBorrowedConsumerVerified = tendonBorrowedConsumerVerified;
    result.tendonRollbackVerified = tendonRollbackVerified;
    result.tendonRigidStateIdentityVerified =
        tendonRigidStateIdentityVerified;
    result.tendonContinuumReactionVerified =
        tendonContinuumReactionVerified;
    result.tendonContinuumMaximumQDelta =
        tendonContinuumMaximumQDelta;
    result.tendonContinuumMaximumVDelta =
        tendonContinuumMaximumVDelta;
    result.tendonTransferCount = static_cast<std::uint32_t>(
        tendonProgram.bindings.size() * stepCount * phaseCount
    );
    result.tendonEnvelopeTransferCount = static_cast<std::uint32_t>(
        tendonEnvelopeBindingCount * stepCount * phaseCount
    );
    result.tendonPointTransferCount = result.tendonTransferCount -
        result.tendonEnvelopeTransferCount;
    result.tendonMaximumForceResidual = std::max(
        assistedStatus.tendonDiagnostics.x,
        finalStatus.tendonDiagnostics.x
    );
    result.tendonMaximumMomentResidual = std::max(
        assistedStatus.tendonDiagnostics.y,
        finalStatus.tendonDiagnostics.y
    );
    result.tendonMaximumGeneralizedCorrection = std::max(
        assistedStatus.tendonDiagnostics.z,
        finalStatus.tendonDiagnostics.z
    );
    // The Achilles certificate is deliberately derived from the live source
    // rows and the accepted NHTENDON3 endpoint transaction.  It does not
    // introduce a second plantar-flexion actuator or infer mechanics from the
    // rendered calf collar.  Each side combines gastrocnemius lateralis,
    // gastrocnemius medialis, and soleus at their exact calcaneal insertion.
    constexpr std::array<std::array<std::uint32_t, 3u>, 2u>
        kAchillesMuscleIndices{{
            {{348u, 349u, 369u}},
            {{388u, 389u, 409u}},
        }};
    constexpr std::array<std::uint32_t, 2u> kCalcaneusBodyIndices{
        138u, 152u,
    };
    constexpr std::array<std::uint32_t, 2u> kAnkleQIndices{109u, 123u};
    const auto magnitude = [](const std::array<double, 3u>& value) {
        return std::sqrt(
            value[0u] * value[0u] + value[1u] * value[1u] +
            value[2u] * value[2u]
        );
    };
    const auto component = [](const mr_float4 value, const std::size_t axis) {
        return axis == 0u ? value.x : (axis == 1u ? value.y : value.z);
    };
    const std::size_t dofCount = model.articulations.front().nv;
    for (std::size_t side = 0u; side < result.achilles.size(); ++side) {
        auto& audit = result.achilles[side];
        const bool selected = applySelectedActivationIncrement &&
            std::all_of(
                kAchillesMuscleIndices[side].begin(),
                kAchillesMuscleIndices[side].end(),
                [&selectedSourceMuscleIndices](const std::uint32_t muscle) {
                    return std::binary_search(
                        selectedSourceMuscleIndices.begin(),
                        selectedSourceMuscleIndices.end(), muscle
                    );
                }
            );
        if (!selected) continue;
        const auto dof = std::find_if(
            model.dofs.begin(), model.dofs.end(),
            [side, &kAnkleQIndices](const MRDofPropertiesGPU& candidate) {
                return candidate.qIndex == kAnkleQIndices[side];
            }
        );
        require(dof != model.dofs.end(),
                "Achilles certificate cannot resolve the source ankle DOF");
        const std::size_t ankleDof = static_cast<std::size_t>(
            std::distance(model.dofs.begin(), dof)
        );
        require(
            ankleDof < dofCount &&
                metalResult.mujocoResults.size() == muscles.gpuMuscles.size() &&
                metalResult.mujocoMuscleGeneralizedForces.size() ==
                    muscles.gpuMuscles.size() * dofCount &&
                selectedControlBaselineResult.mujocoMuscleGeneralizedForces.size() ==
                    metalResult.mujocoMuscleGeneralizedForces.size() &&
                metalResult.standTendonGeneralizedCorrections.size() ==
                    tendonProgram.bindings.size() * dofCount,
            "Achilles certificate is missing accepted source-force state"
        );
        audit.calcaneusBodyIndex = kCalcaneusBodyIndices[side];
        audit.ankleQIndex = kAnkleQIndices[side];
        audit.ankleDofIndex = static_cast<std::uint32_t>(ankleDof);
        std::array<double, 3u> terminalResultant{};
        std::array<double, 3u> terminalIncrementResultant{};
        std::array<double, 3u> nodalResultant{};
        for (const std::uint32_t muscle : kAchillesMuscleIndices[side]) {
            require(muscle < muscles.referenceArchitectures.size(),
                    "Achilles certificate muscle architecture is unavailable");
            const auto gpuBinding = std::find_if(
                tendonProgram.bindings.begin(), tendonProgram.bindings.end(),
                [muscle](const MRNumiHumanTendonBindingGPU& binding) {
                    return binding.muscleIndex == muscle &&
                        binding.endpointOrdinal == 1u;
                }
            );
            const auto sourceBinding = std::find_if(
                muscles.tendonPayload.bindings.begin(),
                muscles.tendonPayload.bindings.end(),
                [muscle](const metalrobo::NumiHumanTendonBinding& binding) {
                    return binding.muscleIndex == muscle &&
                        binding.endpointOrdinal == 1u;
                }
            );
            require(
                gpuBinding != tendonProgram.bindings.end() &&
                    sourceBinding != muscles.tendonPayload.bindings.end() &&
                    gpuBinding->bodyIndex == kCalcaneusBodyIndices[side] &&
                    gpuBinding->mode ==
                        MR_NUMI_HUMAN_TENDON_TRANSFER_DISTRIBUTED_ENVELOPE &&
                    gpuBinding->envelopeIndex < tendonProgram.envelopes.size() &&
                    sourceBinding->mode == metalrobo::NumiHumanTendonAttachmentMode::
                        registeredBoneMigratedDistributedEnvelope,
                "Achilles insertion is not an exact migrated calcaneal envelope"
            );
            const std::size_t bindingIndex = static_cast<std::size_t>(
                std::distance(tendonProgram.bindings.begin(), gpuBinding)
            );
            require(
                bindingIndex < metalResult.standTendonTransfers.size() &&
                    bindingIndex <
                        selectedControlBaselineResult.standTendonTransfers.size(),
                "Achilles endpoint transfer readback is incomplete"
            );
            const auto& transfer =
                metalResult.standTendonTransfers[bindingIndex];
            const auto& baselineTransfer =
                selectedControlBaselineResult.standTendonTransfers[bindingIndex];
            const auto& muscleResult = metalResult.mujocoResults[muscle];
            const auto& envelope =
                tendonProgram.envelopes[gpuBinding->envelopeIndex];
            require(
                transfer.status == MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS &&
                    baselineTransfer.status ==
                        MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS &&
                    transfer.bindingIndex == bindingIndex &&
                    transfer.envelopeIndex == gpuBinding->envelopeIndex &&
                    muscleResult.status == MR_MUJOCO_MUSCLE_REFERENCE_SUCCESS &&
                    envelope.bodyIndex == kCalcaneusBodyIndices[side] &&
                    envelope.boneStableId == gpuBinding->boneStableId &&
                    envelope.nodeCount == 4u,
                "Achilles accepted endpoint result disagrees with its envelope"
            );
            if (audit.calcaneusBoneStableId == 0u) {
                audit.calcaneusBoneStableId = gpuBinding->boneStableId;
            }
            require(audit.calcaneusBoneStableId == gpuBinding->boneStableId,
                    "Achilles muscles do not share one named calcaneal surface");
            ++audit.muscleCount;
            ++audit.distributedEndpointCount;
            audit.representedForceL1Newtons +=
                std::abs(static_cast<double>(transfer.residualsAndForce.w));
            audit.representedForceIncrementL1Newtons += std::abs(
                static_cast<double>(transfer.residualsAndForce.w) -
                static_cast<double>(baselineTransfer.residualsAndForce.w)
            );
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const double terminal = component(
                    transfer.terminalWorldForce, axis
                );
                const double baseline = component(
                    baselineTransfer.terminalWorldForce, axis
                );
                terminalResultant[axis] += terminal;
                terminalIncrementResultant[axis] += terminal - baseline;
                for (std::size_t node = 0u; node < 4u; ++node) {
                    nodalResultant[axis] += component(
                        transfer.nodalWorldForces[node], axis
                    );
                }
            }
            audit.maximumEndpointForceResidualNewtons = std::max(
                audit.maximumEndpointForceResidualNewtons,
                static_cast<double>(transfer.residualsAndForce.x)
            );
            audit.maximumEndpointMomentResidualNewtonMeters = std::max(
                audit.maximumEndpointMomentResidualNewtonMeters,
                static_cast<double>(transfer.residualsAndForce.y)
            );
            const std::size_t muscleDof = muscle * dofCount + ankleDof;
            audit.sourceAnkleTorqueNewtonMeters +=
                metalResult.mujocoMuscleGeneralizedForces[muscleDof];
            audit.sourceAnkleTorqueIncrementNewtonMeters +=
                static_cast<double>(
                    metalResult.mujocoMuscleGeneralizedForces[muscleDof]
                ) - static_cast<double>(
                    selectedControlBaselineResult
                        .mujocoMuscleGeneralizedForces[muscleDof]
                );
            audit.distributedAnkleTorqueCorrectionNewtonMeters +=
                metalResult.standTendonGeneralizedCorrections[
                    bindingIndex * dofCount + ankleDof
                ];
            const double normalizedTension =
                muscleResult.fiberStateTendonForceResidual.z;
            audit.minimumNormalizedTendonTension = std::min(
                audit.minimumNormalizedTendonTension, normalizedTension
            );
            audit.maximumNormalizedTendonTension = std::max(
                audit.maximumNormalizedTendonTension, normalizedTension
            );
            audit.maximumDampedEquilibriumResidual = std::max(
                audit.maximumDampedEquilibriumResidual,
                std::abs(static_cast<double>(
                    muscleResult.fiberStateTendonForceResidual.w
                ))
            );
            audit.minimumPatchRadiusMeters = std::min(
                audit.minimumPatchRadiusMeters,
                static_cast<double>(envelope.metrics.y)
            );
            audit.maximumPatchRadiusMeters = std::max(
                audit.maximumPatchRadiusMeters,
                static_cast<double>(envelope.metrics.y)
            );
        }
        audit.terminalForceResultantNewtons = magnitude(terminalResultant);
        audit.terminalForceIncrementResultantNewtons =
            magnitude(terminalIncrementResultant);
        audit.nodalForceResultantNewtons = magnitude(nodalResultant);
        std::array<double, 3u> aggregateResidual{};
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            aggregateResidual[axis] =
                nodalResultant[axis] - terminalResultant[axis];
        }
        audit.aggregateForceResidualNewtons = magnitude(aggregateResidual);
        audit.configurationIncrementRadians =
            static_cast<double>(result.q[kAnkleQIndices[side]]) -
            static_cast<double>(
                selectedControlBaselineResult.standQ[kAnkleQIndices[side]]
            );
        audit.velocityIncrementRadiansPerSecond =
            static_cast<double>(metalResult.standV[ankleDof]) -
            static_cast<double>(selectedControlBaselineResult.standV[ankleDof]);
        audit.available =
            audit.muscleCount == 3u &&
            audit.distributedEndpointCount == 3u &&
            audit.calcaneusBoneStableId != 0u &&
            audit.representedForceL1Newtons > 0.0 &&
            audit.representedForceIncrementL1Newtons > 0.0 &&
            audit.minimumNormalizedTendonTension > 0.0 &&
            std::abs(audit.sourceAnkleTorqueNewtonMeters) > 1.0e-8 &&
            std::abs(audit.sourceAnkleTorqueIncrementNewtonMeters) > 1.0e-8 &&
            audit.aggregateForceResidualNewtons <= std::max(
                1.0e-3, 1.0e-6 * audit.representedForceL1Newtons
            ) &&
            audit.maximumEndpointMomentResidualNewtonMeters <= 5.0e-5;
        require(
            audit.available,
            "Achilles force-transfer certificate did not close: side=" +
                std::to_string(side) + " force_l1=" +
                std::to_string(audit.representedForceL1Newtons) +
                " force_increment_l1=" +
                std::to_string(audit.representedForceIncrementL1Newtons) +
                " min_tension=" +
                std::to_string(audit.minimumNormalizedTendonTension) +
                " ankle_torque=" +
                std::to_string(audit.sourceAnkleTorqueNewtonMeters) +
                " ankle_torque_increment=" +
                std::to_string(audit.sourceAnkleTorqueIncrementNewtonMeters) +
                " aggregate_force_residual=" +
                std::to_string(audit.aggregateForceResidualNewtons) +
                " max_moment_residual=" +
                std::to_string(
                    audit.maximumEndpointMomentResidualNewtonMeters
                )
        );
    }
    return result;
}

MuscleDrivenVisualState integrateMuscleDrivenVisualState(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& muscles,
    const double timestepSeconds,
    const std::uint32_t stepCount,
    const double activation,
    const std::span<const std::uint32_t> selectedSourceMuscleIndices,
    const LoadedSupportContacts* supportContacts
) {
    require(std::isfinite(timestepSeconds) &&
                timestepSeconds >= 1.0e-6 && timestepSeconds <= 1.0e-3,
            "muscle-driven visual step must be between 1 us and 1 ms");
    require(model.world.nv > 0u && model.defaultQ.size() == model.world.nq &&
                model.defaultV.size() == model.world.nv &&
                muscles.referenceMuscles.size() == muscles.muscles.size() &&
                muscles.gpuMuscles.size() == muscles.muscles.size() &&
                muscles.gpuSites.size() == muscles.sites.size() &&
                muscles.gpuWraps.size() == muscles.wraps.size() &&
                muscles.gpuRoutes.size() == muscles.routes.size(),
            "muscle-driven visual state has inconsistent MyoSim dimensions");
    require(std::isfinite(activation) && activation >= 0.0 && activation <= 1.0,
            "muscle-driven visual activation must be within [0, 1]");
    require(stepCount >= 1u && stepCount <= 64u,
            "muscle-driven visual step count must be in [1, 64]");
    require(std::is_sorted(
                selectedSourceMuscleIndices.begin(), selectedSourceMuscleIndices.end()
            ) && std::adjacent_find(
                selectedSourceMuscleIndices.begin(), selectedSourceMuscleIndices.end()
            ) == selectedSourceMuscleIndices.end() &&
                std::all_of(
                    selectedSourceMuscleIndices.begin(), selectedSourceMuscleIndices.end(),
                    [&muscles](const std::uint32_t index) {
                        return index < muscles.gpuMuscles.size();
                    }
                ),
            "selected MyoSim source-muscle activation set is malformed");

    std::vector<double> initialQ(model.defaultQ.begin(), model.defaultQ.end());
    const std::vector<double> initialV(model.defaultV.begin(), model.defaultV.end());
    std::optional<GroundAlignedSupport> support;
    if (supportContacts != nullptr) {
        support.emplace(makeGroundAlignedSupport(model, *supportContacts));
        initialQ = support->q;
    }
    MuscleDrivenVisualState result;
    result.stepCount = stepCount;
    result.selectedSourceMuscleActivationCount = static_cast<std::uint32_t>(
        selectedSourceMuscleIndices.size()
    );
    metalrobo::ArticulatedDynamicsConfig dynamicsConfig;
    dynamicsConfig.timestep = timestepSeconds;
    std::vector<double> passiveQ = initialQ;
    std::vector<double> passiveV = initialV;
    std::vector<double> activeQ = initialQ;
    std::vector<double> activeV = initialV;
    const std::vector<double> zeroForce(model.world.nv, 0.0);
    const MetalMujocoVisualQueries metalQueries =
        makeMetalMujocoVisualQueries(model);
    std::vector<MRMujocoMuscleStateGPU> activeMuscleStates(
        muscles.gpuMuscles.size()
    );
    std::vector<MRMujocoMuscleStateGPU> passiveMuscleStates(
        muscles.gpuMuscles.size()
    );
    for (std::size_t muscleIndex = 0u;
         muscleIndex < activeMuscleStates.size();
         ++muscleIndex) {
        MRMujocoMuscleStateGPU& state = activeMuscleStates[muscleIndex];
        const bool selected = selectedSourceMuscleIndices.empty() ||
            std::binary_search(
                selectedSourceMuscleIndices.begin(), selectedSourceMuscleIndices.end(),
                static_cast<std::uint32_t>(muscleIndex)
            );
        const float initialActivation = selected ? static_cast<float>(activation) : 0.0f;
        state.excitationAndActivation = {
            initialActivation,
            initialActivation,
            0.0f,
            0.0f,
        };
    }
    metalrobo::MetalArticulatedOperatorConfig activeMetalConfig{
        .pointJacobiansOnly = true,
        .mujocoActivationTimestepSeconds =
            static_cast<float>(timestepSeconds),
    };
    metalrobo::MetalArticulatedOperatorConfig passiveMetalConfig{
        .pointJacobiansOnly = true,
    };
    metalrobo::MetalArticulatedOperatorContext activeMetalContext(
        activeMetalConfig
    );
    metalrobo::MetalArticulatedOperatorContext passiveMetalContext(
        passiveMetalConfig
    );
    std::vector<double> passiveSupportWarm;
    std::vector<double> activeSupportWarm;
    if (supportContacts != nullptr) {
        passiveSupportWarm.assign(3u * supportContacts->records.size(), 0.0);
        activeSupportWarm.assign(3u * supportContacts->records.size(), 0.0);
        metalrobo::CompiledMetalMultiArticulatedContactProgram program;
        const auto compiled = metalrobo::compileMetalMultiArticulatedContactProgram(model, program);
        require(
            !compiled.succeeded() &&
                (compiled.status == metalrobo::MetalMultiArticulatedContactStatus::capacityOverflow ||
                 compiled.status == metalrobo::MetalMultiArticulatedContactStatus::unsupportedTopology),
            "MyoSim dynamic source-support Metal admission unexpectedly changed: " + compiled.message
        );
        result.supportDeviceName = "not_admitted";
        result.supportMetalStatus = "not_admitted_articulation_exceeds_metal_contact_bucket";
        result.supportWitnessCount = supportContacts->header.contactCount;
        result.supportSeedTranslationMeters = support->seedTranslationMeters;
    }
    metalrobo::QualityContactSolverConfig contactConfig;
    contactConfig.maximumIterations = 300u;
    contactConfig.kktTolerance = 1.0e-10;
    const auto applyDynamicSupport = [&] (
        std::vector<double>& q,
        std::vector<double>& velocity,
        std::vector<double>& warmImpulses,
        const bool isActive
    ) {
        const DynamicSourceSupportContacts contacts = makeDynamicSourceSupportContacts(
            model, *supportContacts, dynamicsConfig, q, velocity, warmImpulses
        );
        result.minimumSupportPlaneGapMeters = std::min(
            result.minimumSupportPlaneGapMeters, contacts.minimumPlaneGapMeters
        );
        if (isActive) {
            result.activeSupportContactCount = static_cast<std::uint32_t>(contacts.contacts.size());
            result.maximumActiveSupportContactCount = std::max(
                result.maximumActiveSupportContactCount, result.activeSupportContactCount
            );
        }
        if (contacts.contacts.empty()) return;
        metalrobo::MultiArticulatedContactProblem problem;
        const auto build = metalrobo::buildMultiArticulatedIslandContactProblem(
            model, q, velocity, {}, contacts.contacts, problem, dynamicsConfig
        );
        require(build.succeeded(), "MyoSim dynamic source-support FP64 contact construction failed");
        metalrobo::MultiArticulatedContactSolution solution;
        const auto solve = metalrobo::solveMultiArticulatedContactProblem(
            problem, solution, contactConfig
        );
        require(solve.succeeded() && solution.articulatedVelocity.size() == model.world.nv &&
                    solution.impulses.size() == 3u * contacts.contacts.size(),
                "MyoSim dynamic source-support FP64 contact solve failed");
        velocity = std::move(solution.articulatedVelocity);
        for (std::size_t contact = 0u; contact < contacts.sourceRecordIndices.size(); ++contact) {
            const std::size_t record = contacts.sourceRecordIndices[contact];
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                warmImpulses[3u * record + axis] = solution.impulses[3u * contact + axis];
            }
        }
        result.supportContactApplied = true;
    };
    for (std::uint32_t step = 0u; step < stepCount; ++step) {
        const MetalMujocoForceStep activatedForce = evaluateMetalMujocoForce(
            model,
            muscles,
            metalQueries,
            activeQ,
            activeMuscleStates,
            activeMetalContext
        );
        const MetalMujocoForceStep sourceDefaultPassiveForce =
            evaluateMetalMujocoForce(
                model,
                muscles,
                metalQueries,
                activeQ,
                passiveMuscleStates,
                passiveMetalContext
            );
        require(
            activatedForce.generalizedForce.size() == model.world.nv &&
                sourceDefaultPassiveForce.generalizedForce.size() ==
                    model.world.nv &&
                activatedForce.deviceName == sourceDefaultPassiveForce.deviceName,
            "MyoSim Metal force transactions returned incompatible outputs"
        );
        result.muscleMetalStepCount += 2u;
        result.muscleMetalForceRecordCount += static_cast<std::uint32_t>(
            muscles.gpuMuscles.size()
        );
        result.muscleMetalElapsedMilliseconds +=
            activatedForce.elapsedMilliseconds +
            sourceDefaultPassiveForce.elapsedMilliseconds;
        result.muscleMetalDeviceName = activatedForce.deviceName;
        result.appliedWrapCount += activatedForce.appliedWrapCount;
        std::vector<double> muscleForce(model.world.nv, 0.0);
        for (std::size_t index = 0u; index < muscleForce.size(); ++index) {
            muscleForce[index] = static_cast<double>(
                activatedForce.generalizedForce[index]
            ) - static_cast<double>(
                sourceDefaultPassiveForce.generalizedForce[index]
            );
        }
        require(std::all_of(muscleForce.begin(), muscleForce.end(), [](const double value) {
                    return std::isfinite(value);
                }),
                "MyoSim muscle force projection returned a non-finite generalized force");
        const auto passiveDiagnostics = metalrobo::integrateArticulatedState(
            model, 0u, passiveQ, passiveV, zeroForce, {}, dynamicsConfig
        );
        require(passiveDiagnostics.succeeded(),
                "passive free-body visual comparison step failed");
        const auto activeDiagnostics = metalrobo::integrateArticulatedState(
            model, 0u, activeQ, activeV, muscleForce, {}, dynamicsConfig
        );
        require(activeDiagnostics.succeeded(),
                "muscle-driven free-body visual step failed");
        if (supportContacts != nullptr) {
            applyDynamicSupport(passiveQ, passiveV, passiveSupportWarm, false);
            applyDynamicSupport(activeQ, activeV, activeSupportWarm, true);
        }
    }
    for (std::size_t index = 0u; index < activeV.size(); ++index) {
        result.maximumVelocityDelta = std::max(
            result.maximumVelocityDelta, std::abs(activeV[index] - passiveV[index])
        );
    }
    for (std::size_t index = 0u; index < activeQ.size(); ++index) {
        result.maximumConfigurationDelta = std::max(
            result.maximumConfigurationDelta, std::abs(activeQ[index] - passiveQ[index])
        );
    }
    require(std::isfinite(result.maximumVelocityDelta) &&
                std::isfinite(result.maximumConfigurationDelta) &&
                result.maximumVelocityDelta > 1.0e-9 &&
                result.maximumConfigurationDelta > 1.0e-12,
            "the complete MyoSim muscle force did not distinguish the visual state step");
    if (supportContacts != nullptr) {
        require(std::isfinite(result.minimumSupportPlaneGapMeters),
                "MyoSim dynamic source-support did not evaluate a source plane gap");
    }
    result.q.reserve(activeQ.size());
    for (const double coordinate : activeQ) {
        require(std::isfinite(coordinate) &&
                    coordinate >= -static_cast<double>(std::numeric_limits<float>::max()) &&
                    coordinate <= static_cast<double>(std::numeric_limits<float>::max()),
                "muscle-driven visual configuration is not representable on Metal");
        result.q.push_back(static_cast<float>(coordinate));
    }
    return result;
}

mr_float4 normalizeQuaternion(const mr_float4 value) {
    const float length = std::sqrt(
        value.x * value.x + value.y * value.y +
        value.z * value.z + value.w * value.w
    );
    require(std::isfinite(length) && length > 1.0e-6f,
            "Metal articulated pose has a non-normalizable orientation");
    return {value.x / length, value.y / length, value.z / length, value.w / length};
}

std::vector<MRBodyStateGPU> visualBodyStates(
    const metalrobo::EngineModel& model,
    const std::span<const MRArticulatedBodyPoseGPU> poses
) {
    require(poses.size() == model.bodies.size(), "Metal body-pose stream has an invalid size");
    std::vector<MRBodyStateGPU> result(poses.size());
    for (std::size_t index = 0u; index < poses.size(); ++index) {
        MRBodyStateGPU& state = result[index];
        state.position = poses[index].position;
        state.orientation = normalizeQuaternion(poses[index].orientation);
        state.linearVelocityAndInverseMass.w = model.bodies[index].massAndInverseMass.y;
        state.flagsAndIndices[0] = model.bodies[index].motionType;
        state.flagsAndIndices[1] = model.bodies[index].articulationIndex;
        state.flagsAndIndices[2] = static_cast<mr_u32>(index);
        state.flagsAndIndices[3] = 0u;
    }
    return result;
}

struct SourceRouteCentreline {
    std::uint32_t muscleIndex = 0u;
    struct Point {
        mr_float4 world{};
        std::uint32_t attachmentBodyIndex = MR_INVALID_INDEX;
        // Surface registration is a presentation-only cue.  The source route
        // keeps its authored site records and force evaluation untouched.
        bool surfaceProjected = false;
        mr_float4 surfaceNormalWorld{};
    };
    std::vector<Point> points;
};

struct SourceRouteCentrelines {
    std::vector<SourceRouteCentreline> muscles;
    std::uint32_t appliedWrapCount = 0u;
    std::uint32_t surfaceProjectedAttachmentCount = 0u;
};

SourceRouteCentrelines resolveSourceRouteCentrelines(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& musclePayload,
    const std::span<const float> poseQ,
    const std::span<const std::uint32_t> requestedMuscles
) {
    require(poseQ.size() == model.world.nq && model.defaultV.size() == model.world.nv,
            "MyoSim source-route visual pose dimensions are inconsistent");
    std::vector<double> q(poseQ.begin(), poseQ.end());
    const std::vector<double> v(model.defaultV.begin(), model.defaultV.end());
    SourceRouteCentrelines result;
    const std::size_t expectedCount = requestedMuscles.empty()
        ? musclePayload.referenceMuscles.size() : requestedMuscles.size();
    result.muscles.reserve(expectedCount);
    const metalrobo::MujocoMuscleState state{};
    const auto resolve = [&](const std::uint32_t index) {
        require(index < musclePayload.referenceMuscles.size(),
                "requested MyoSim source-route muscle index is out of bounds");
        metalrobo::MujocoMuscleResult muscleResult;
        const auto diagnostics = metalrobo::evaluateMujocoMuscle(
            model, 0u, q, v, musclePayload.referenceSites,
            musclePayload.referenceWraps, musclePayload.referenceMuscles[index],
            state, muscleResult
        );
        require(diagnostics.succeeded() && muscleResult.path.centreline.size() >= 2u,
                "MyoSim source-route resolution failed for muscle " + std::to_string(index) + ": " +
                    metalrobo::mujocoMuscleReferenceStatusName(diagnostics.status));
        std::vector<SourceRouteCentreline::Point> centreline;
        centreline.reserve(muscleResult.path.centreline.size());
        for (const metalrobo::MujocoMusclePathSample& sample : muscleResult.path.centreline) {
            require(std::all_of(sample.world.begin(), sample.world.end(), [](const double value) {
                        return std::isfinite(value) &&
                            value >= -static_cast<double>(std::numeric_limits<float>::max()) &&
                            value <= static_cast<double>(std::numeric_limits<float>::max());
                    }),
                    "MyoSim source-route sample is not representable on the renderer");
            centreline.push_back({
                {
                    static_cast<float>(sample.world[0]), static_cast<float>(sample.world[1]),
                    static_cast<float>(sample.world[2]), 1.0f,
                },
                sample.attachmentBodyIndex,
                false,
                {},
            });
        }
        result.appliedWrapCount += muscleResult.path.appliedWrapCount;
        result.muscles.push_back({index, std::move(centreline)});
    };
    if (requestedMuscles.empty()) {
        for (std::uint32_t index = 0u; index < musclePayload.referenceMuscles.size(); ++index) {
            resolve(index);
        }
    } else {
        for (const std::uint32_t index : requestedMuscles) resolve(index);
    }
    return result;
}

mr_float4 subtractPoint(const mr_float4 first, const mr_float4 second) {
    return {first.x - second.x, first.y - second.y, first.z - second.z, 0.0f};
}

mr_float4 addPoint(const mr_float4 first, const mr_float4 second) {
    return {first.x + second.x, first.y + second.y, first.z + second.z, 1.0f};
}

mr_float4 scalePoint(const mr_float4 point, const float scalar) {
    return {point.x * scalar, point.y * scalar, point.z * scalar, 0.0f};
}

float dotPoint(const mr_float4 first, const mr_float4 second) {
    return first.x * second.x + first.y * second.y + first.z * second.z;
}

mr_float4 normalizedDirection(const mr_float4 value, const char* context) {
    const float lengthSquared = dotPoint(value, value);
    require(std::isfinite(lengthSquared) && lengthSquared > 1.0e-10f,
            std::string(context) + " is degenerate");
    return scalePoint(value, 1.0f / std::sqrt(lengthSquared));
}

mr_float4 rotatePoint(const mr_float4 quaternion, const mr_float4 point) {
    const mr_float4 axis{quaternion.x, quaternion.y, quaternion.z, 0.0f};
    const mr_float4 twiceCross{
        2.0f * (axis.y * point.z - axis.z * point.y),
        2.0f * (axis.z * point.x - axis.x * point.z),
        2.0f * (axis.x * point.y - axis.y * point.x),
        0.0f,
    };
    const mr_float4 correction{
        quaternion.w * twiceCross.x + axis.y * twiceCross.z - axis.z * twiceCross.y,
        quaternion.w * twiceCross.y + axis.z * twiceCross.x - axis.x * twiceCross.z,
        quaternion.w * twiceCross.z + axis.x * twiceCross.y - axis.y * twiceCross.x,
        0.0f,
    };
    return {point.x + correction.x, point.y + correction.y, point.z + correction.z, 0.0f};
}

mr_float4 closestPointOnTriangle(
    const mr_float4 point,
    const mr_float4 first,
    const mr_float4 second,
    const mr_float4 third
) {
    const mr_float4 firstToPoint = subtractPoint(point, first);
    const mr_float4 firstToSecond = subtractPoint(second, first);
    const mr_float4 firstToThird = subtractPoint(third, first);
    const float dotFirstSecond = dotPoint(firstToSecond, firstToPoint);
    const float dotFirstThird = dotPoint(firstToThird, firstToPoint);
    if (dotFirstSecond <= 0.0f && dotFirstThird <= 0.0f) return first;

    const mr_float4 secondToPoint = subtractPoint(point, second);
    const float dotSecondSecond = dotPoint(firstToSecond, secondToPoint);
    const float dotSecondThird = dotPoint(firstToThird, secondToPoint);
    if (dotSecondSecond >= 0.0f && dotSecondThird <= dotSecondSecond) return second;

    const float edgeFirstSecond = dotFirstSecond * dotSecondThird - dotSecondSecond * dotFirstThird;
    if (edgeFirstSecond <= 0.0f && dotFirstSecond >= 0.0f && dotSecondSecond <= 0.0f) {
        return addPoint(first, scalePoint(firstToSecond, dotFirstSecond / (dotFirstSecond - dotSecondSecond)));
    }

    const mr_float4 thirdToPoint = subtractPoint(point, third);
    const float dotThirdSecond = dotPoint(firstToSecond, thirdToPoint);
    const float dotThirdThird = dotPoint(firstToThird, thirdToPoint);
    if (dotThirdThird >= 0.0f && dotThirdSecond <= dotThirdThird) return third;

    const float edgeFirstThird = dotThirdSecond * dotFirstThird - dotFirstSecond * dotThirdThird;
    if (edgeFirstThird <= 0.0f && dotFirstThird >= 0.0f && dotThirdThird <= 0.0f) {
        return addPoint(first, scalePoint(firstToThird, dotFirstThird / (dotFirstThird - dotThirdThird)));
    }

    const float edgeSecondThird = dotSecondSecond * dotThirdThird - dotThirdSecond * dotSecondThird;
    if (edgeSecondThird <= 0.0f &&
        dotSecondThird - dotSecondSecond >= 0.0f && dotThirdSecond - dotThirdThird >= 0.0f) {
        const mr_float4 secondToThird = subtractPoint(third, second);
        const float ratio = (dotSecondThird - dotSecondSecond) /
            ((dotSecondThird - dotSecondSecond) + (dotThirdSecond - dotThirdThird));
        return addPoint(second, scalePoint(secondToThird, ratio));
    }

    const float barycentricDenominator = edgeFirstSecond + edgeFirstThird + edgeSecondThird;
    if (std::abs(barycentricDenominator) <= 1.0e-12f) return first;
    const float denominator = 1.0f / barycentricDenominator;
    const float secondWeight = edgeFirstThird * denominator;
    const float thirdWeight = edgeFirstSecond * denominator;
    return addPoint(first, addPoint(scalePoint(firstToSecond, secondWeight), scalePoint(firstToThird, thirdWeight)));
}

mr_float4 boneVertexWorld(
    const BoneRecord& bone,
    const BoneVertex& vertex,
    const MRBodyStateGPU& body
) {
    const mr_float4 boneRotation{
        bone.quaternionX, bone.quaternionY, bone.quaternionZ, bone.quaternionW,
    };
    const mr_float4 local = addPoint(
        {bone.translationX, bone.translationY, bone.translationZ, 0.0f},
        scalePoint(
            rotatePoint(boneRotation, {vertex.positionX, vertex.positionY, vertex.positionZ, 0.0f}),
            bone.uniformScale
        )
    );
    return addPoint(body.position, rotatePoint(body.orientation, local));
}

mr_float4 boneVertexNormalWorld(
    const BoneRecord& bone,
    const BoneVertex& vertex,
    const MRBodyStateGPU& body
) {
    const mr_float4 boneRotation{
        bone.quaternionX, bone.quaternionY, bone.quaternionZ, bone.quaternionW,
    };
    mr_float4 normal = rotatePoint(
        body.orientation,
        rotatePoint(boneRotation, {vertex.normalX, vertex.normalY, vertex.normalZ, 0.0f})
    );
    const float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    require(length > 1.0e-6f, "BodyParts3D attachment triangle has a degenerate normal");
    normal.x /= length;
    normal.y /= length;
    normal.z /= length;
    normal.w = 0.0f;
    return normal;
}

void projectSourceSiteEndpointsToBoneSurfaces(
    SourceRouteCentrelines& routes,
    const LoadedBones& bones,
    const std::span<const MRBodyStateGPU> bodies
) {
    constexpr float maximumProjectionDistance = 0.12f;
    constexpr float maximumProjectionDistanceSquared =
        maximumProjectionDistance * maximumProjectionDistance;
    require(bodies.size() > 0u, "source-route surface projection has no body poses");
    for (SourceRouteCentreline& route : routes.muscles) {
        for (SourceRouteCentreline::Point& point : route.points) {
            if (point.attachmentBodyIndex == MR_INVALID_INDEX ||
                point.attachmentBodyIndex >= bodies.size()) {
                continue;
            }
            mr_float4 closest{};
            mr_float4 closestNormal{};
            float closestDistanceSquared = std::numeric_limits<float>::infinity();
            for (const BoneRecord& bone : bones.records) {
                if (bone.bodyIndex != point.attachmentBodyIndex) continue;
                for (std::uint32_t offset = 0u; offset < bone.indexCount; offset += 3u) {
                    const std::uint32_t first = bones.indices[bone.firstIndex + offset];
                    const std::uint32_t second = bones.indices[bone.firstIndex + offset + 1u];
                    const std::uint32_t third = bones.indices[bone.firstIndex + offset + 2u];
                    const mr_float4 candidate = closestPointOnTriangle(
                        point.world,
                        boneVertexWorld(bone, bones.vertices[first], bodies[bone.bodyIndex]),
                        boneVertexWorld(bone, bones.vertices[second], bodies[bone.bodyIndex]),
                        boneVertexWorld(bone, bones.vertices[third], bodies[bone.bodyIndex])
                    );
                    const mr_float4 difference = subtractPoint(point.world, candidate);
                    const float distanceSquared = dotPoint(difference, difference);
                    if (distanceSquared < closestDistanceSquared) {
                        closestDistanceSquared = distanceSquared;
                        closest = candidate;
                        const mr_float4 firstNormal = boneVertexNormalWorld(
                            bone, bones.vertices[first], bodies[bone.bodyIndex]
                        );
                        const mr_float4 secondNormal = boneVertexNormalWorld(
                            bone, bones.vertices[second], bodies[bone.bodyIndex]
                        );
                        const mr_float4 thirdNormal = boneVertexNormalWorld(
                            bone, bones.vertices[third], bodies[bone.bodyIndex]
                        );
                        closestNormal = {
                            firstNormal.x + secondNormal.x + thirdNormal.x,
                            firstNormal.y + secondNormal.y + thirdNormal.y,
                            firstNormal.z + secondNormal.z + thirdNormal.z,
                            0.0f,
                        };
                        const float normalLength = std::sqrt(
                            closestNormal.x * closestNormal.x +
                            closestNormal.y * closestNormal.y +
                            closestNormal.z * closestNormal.z
                        );
                        require(normalLength > 1.0e-6f,
                                "BodyParts3D attachment triangle averaged normal is degenerate");
                        closestNormal.x /= normalLength;
                        closestNormal.y /= normalLength;
                        closestNormal.z /= normalLength;
                    }
                }
            }
            if (closestDistanceSquared <= maximumProjectionDistanceSquared) {
                point.world = closest;
                point.surfaceProjected = true;
                point.surfaceNormalWorld = closestNormal;
                ++routes.surfaceProjectedAttachmentCount;
            }
        }
    }
}

metalrobo::WorldPose cameraToward(
    const mr_float4 position,
    const mr_float4 target
) {
    const mr_float4 forward{
        target.x - position.x, target.y - position.y, target.z - position.z, 0.0f,
    };
    const float forwardLength = std::sqrt(
        forward.x * forward.x + forward.y * forward.y + forward.z * forward.z
    );
    require(forwardLength > 1.0e-5f, "native Human camera has no target direction");
    const mr_float4 unitForward{
        forward.x / forwardLength, forward.y / forwardLength, forward.z / forwardLength, 0.0f,
    };
    mr_float4 localY{0.0f, 0.0f, -1.0f, 0.0f};
    mr_float4 localX{
        localY.y * unitForward.z - localY.z * unitForward.y,
        localY.z * unitForward.x - localY.x * unitForward.z,
        localY.x * unitForward.y - localY.y * unitForward.x,
        0.0f,
    };
    const float localXLength = std::sqrt(
        localX.x * localX.x + localX.y * localX.y + localX.z * localX.z
    );
    require(localXLength > 1.0e-5f, "native Human camera is parallel to world up");
    localX.x /= localXLength;
    localX.y /= localXLength;
    localX.z /= localXLength;
    localY = {
        unitForward.y * localX.z - unitForward.z * localX.y,
        unitForward.z * localX.x - unitForward.x * localX.z,
        unitForward.x * localX.y - unitForward.y * localX.x,
        0.0f,
    };
    const float m00 = localX.x;
    const float m01 = localY.x;
    const float m02 = unitForward.x;
    const float m10 = localX.y;
    const float m11 = localY.y;
    const float m12 = unitForward.y;
    const float m20 = localX.z;
    const float m21 = localY.z;
    const float m22 = unitForward.z;
    mr_float4 orientation{};
    const float trace = m00 + m11 + m22;
    if (trace > 0.0f) {
        const float scale = 2.0f * std::sqrt(trace + 1.0f);
        orientation = {(m21 - m12) / scale, (m02 - m20) / scale,
                       (m10 - m01) / scale, 0.25f * scale};
    } else if (m00 > m11 && m00 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
        orientation = {0.25f * scale, (m01 + m10) / scale,
                       (m02 + m20) / scale, (m21 - m12) / scale};
    } else if (m11 > m22) {
        const float scale = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
        orientation = {(m01 + m10) / scale, 0.25f * scale,
                       (m12 + m21) / scale, (m02 - m20) / scale};
    } else {
        const float scale = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
        orientation = {(m02 + m20) / scale, (m12 + m21) / scale,
                       0.25f * scale, (m10 - m01) / scale};
    }
    return {position, normalizeQuaternion(orientation)};
}

metalrobo::SensorSpec makeCamera(
    const std::string& id,
    const mr_float4 position,
    const mr_float4 target,
    const std::uint32_t dimension
) {
    metalrobo::SensorSpec camera;
    camera.id = id;
    camera.parentAssetId = "myosim_human";
    // ``position`` and ``target`` are calculated from the posed, rendered
    // geometry in world coordinates.  Parenting that world-space pose to the
    // articulated asset applies the root transform a second time, which makes
    // oblique anatomy reviews look off-centre and distant.  Keep the semantic
    // asset id for world validation, but make the camera genuinely world
    // anchored.
    camera.parentKind = MR_WORLD_SENSOR_PARENT_WORLD;
    camera.kind = MR_WORLD_SENSOR_RGBD;
    camera.localPose = cameraToward(position, target);
    camera.width = dimension;
    camera.height = dimension;
    // Keep field of view invariant across qualified output dimensions. A
    // fixed 750 px focal length cropped a 1.7 m body at 512/640 px even when
    // the source geometry bounds and camera distance were correct.
    const float focalLength = 0.72f * static_cast<float>(dimension);
    camera.intrinsics = {focalLength, focalLength, 0.5f * dimension, 0.5f * dimension};
    camera.maximumDepthMeters = 20.0f;
    return camera;
}

MRVisualMaterialGPUV2 makeMaterial(
    const mr_float4 color,
    const mr_float4 emission,
    const float roughness = 0.55f,
    const float clearcoat = 0.0f
) {
    MRVisualMaterialGPUV2 material{};
    material.baseColorAndOpacity = color;
    material.emissionAndStrength = emission;
    material.surface = {roughness, 0.02f, 1.0f, 1.0f};
    material.coatingAndAlphaCutoff = {clearcoat, 0.22f, 1.0f, 0.5f};
    material.textureIndices0 = {
        MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    material.textureIndices1 = {
        MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    material.flags = {
        MR_VISUAL_ALPHA_OPAQUE, MR_VISUAL_MATERIAL_DOUBLE_SIDED, 0u, 1u,
    };
    material.reserved = {
        MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX,
    };
    return material;
}

metalrobo::VisualLightRigV1 makeHumanAnatomyLightRig(
    const mr_float4 center,
    const mr_float4 cameraPosition,
    const float cameraDistance
) {
    // Use a camera-relative three-point rig.  The source meshes must remain
    // identical in every view, but a fixed world-space key left posterior and
    // lateral inspections underlit and made the tendon-to-bone interface read
    // as a flat colour boundary.  These broad, neutral softboxes present the
    // same source normals and triangles from each inspected camera direction.
    const float distance = std::max(cameraDistance, 0.35f);
    const float intensityScale = distance * distance;
    const mr_float4 view = normalizedDirection(
        subtractPoint(cameraPosition, center), "Human anatomy camera direction"
    );
    const mr_float4 target = addPoint(center, {0.0f, 0.0f, 0.04f * distance, 0.0f});
    const auto makeAreaLight = [&target, distance, intensityScale](
        const mr_float4 position,
        const mr_float4 color,
        const float intensity,
        const float width,
        const float height,
        const std::uint32_t stableId
    ) {
        MRVisualLightGPUV1 light{};
        light.positionAndRange = {position.x, position.y, position.z, 20.0f};
        const mr_float4 direction = normalizedDirection(
            subtractPoint(target, position), "Human anatomy softbox direction"
        );
        light.directionAndSpot = {direction.x, direction.y, direction.z, -1.0f};
        light.colorAndIntensity = {color.x, color.y, color.z, intensity * intensityScale};
        light.shape = {width * distance, height * distance, -1.0f, 0.08f};
        light.shadow = {1u, 0u, 0u, 0u};
        light.identity = {MR_VISUAL_LIGHT_RECTANGLE, MR_VISUAL_LIGHT_UNIT_NIT, 0u, stableId};
        return light;
    };

    const mr_float4 keyPosition = addPoint(
        cameraPosition, {0.24f * distance, -0.13f * distance, 0.31f * distance, 0.0f}
    );
    const mr_float4 fillPosition = addPoint(
        cameraPosition, {-0.32f * distance, 0.18f * distance, 0.10f * distance, 0.0f}
    );
    const mr_float4 rimPosition = addPoint(
        addPoint(center, scalePoint(view, -0.82f * distance)),
        {0.0f, 0.08f * distance, 0.42f * distance, 0.0f}
    );
    metalrobo::VisualLightRigV1 result;
    result.id = "human_anatomy_camera_relative_three_point";
    result.contentHash = "builtin:human-anatomy-camera-relative-three-point-v3";
    result.lights = {
        // A full-body exterior reads best under a wider, higher-exposure
        // portrait rig than the deliberately contrasty exposed-tissue
        // diagnostic.  These lights do not alter source geometry or its
        // kinematic binding; they only make the real surface relief legible.
        makeAreaLight(keyPosition, {1.0f, 0.94f, 0.88f, 0.0f}, 230.0f, 0.96f, 0.82f, 201u),
        makeAreaLight(fillPosition, {0.80f, 0.88f, 1.0f, 0.0f}, 82.0f, 1.08f, 0.90f, 202u),
        makeAreaLight(rimPosition, {1.0f, 0.86f, 0.74f, 0.0f}, 132.0f, 0.84f, 0.68f, 203u),
    };
    return result;
}

struct GeometryRange {
    std::uint32_t firstIndex = 0u;
    std::uint32_t indexCount = 0u;
    mr_float4 minimum{};
    mr_float4 maximum{};
};

GeometryRange appendEllipsoid(
    metalrobo::VisualAssetPackV2& pack,
    const std::array<float, 3u> radii
) {
    constexpr std::uint32_t kLatitude = 8u;
    constexpr std::uint32_t kLongitude = 12u;
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t latitude = 0u; latitude <= kLatitude; ++latitude) {
        const float phi = static_cast<float>(latitude) *
            std::numbers::pi_v<float> / static_cast<float>(kLatitude);
        const float sinPhi = std::sin(phi);
        const float cosPhi = std::cos(phi);
        for (std::uint32_t longitude = 0u; longitude <= kLongitude; ++longitude) {
            const float theta = static_cast<float>(longitude) *
                2.0f * std::numbers::pi_v<float> / static_cast<float>(kLongitude);
            const float cosTheta = std::cos(theta);
            const float sinTheta = std::sin(theta);
            const mr_float4 normal{sinPhi * cosTheta, sinPhi * sinTheta, cosPhi, 1.0f};
            mr_float4 tangent{-sinTheta, cosTheta, 0.0f, 0.0f};
            if (latitude == 0u || latitude == kLatitude) {
                tangent = {1.0f, 0.0f, 0.0f, 0.0f};
            }
            pack.vertices.push_back({
                {radii[0] * normal.x, radii[1] * normal.y, radii[2] * normal.z, 1.0f},
                normal,
                tangent,
                {static_cast<float>(longitude) / static_cast<float>(kLongitude),
                 static_cast<float>(latitude) / static_cast<float>(kLatitude), 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f, 1.0f},
            });
        }
    }
    const std::uint32_t row = kLongitude + 1u;
    for (std::uint32_t latitude = 0u; latitude < kLatitude; ++latitude) {
        for (std::uint32_t longitude = 0u; longitude < kLongitude; ++longitude) {
            const std::uint32_t first = vertexBase + latitude * row + longitude;
            pack.indices.insert(pack.indices.end(), {
                first, first + row, first + 1u,
                first + 1u, first + row, first + row + 1u,
            });
        }
    }
    result.indexCount = static_cast<std::uint32_t>(pack.indices.size()) - result.firstIndex;
    result.minimum = {-radii[0], -radii[1], -radii[2], 1.0f};
    result.maximum = {radii[0], radii[1], radii[2], 1.0f};
    return result;
}

mr_float4 normalTangent(const mr_float4 normal) {
    const mr_float4 reference = std::abs(normal.z) < 0.9f
        ? mr_float4{0.0f, 0.0f, 1.0f, 0.0f}
        : mr_float4{0.0f, 1.0f, 0.0f, 0.0f};
    mr_float4 tangent{
        reference.y * normal.z - reference.z * normal.y,
        reference.z * normal.x - reference.x * normal.z,
        reference.x * normal.y - reference.y * normal.x,
        0.0f,
    };
    const float length = std::sqrt(
        tangent.x * tangent.x + tangent.y * tangent.y + tangent.z * tangent.z
    );
    require(length > 1.0e-6f, "BodyParts3D bone tangent is degenerate");
    tangent.x /= length;
    tangent.y /= length;
    tangent.z /= length;
    return tangent;
}

mr_float4 boneTangent(const BoneVertex& vertex) {
    return normalTangent({vertex.normalX, vertex.normalY, vertex.normalZ, 1.0f});
}

GeometryRange appendBoneGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const LoadedBones& bones,
    const BoneRecord& bone
) {
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f,
    };
    result.maximum = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f,
    };
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t offset = 0u; offset < bone.vertexCount; ++offset) {
        const BoneVertex& source = bones.vertices[bone.firstVertex + offset];
        const mr_float4 position{source.positionX, source.positionY, source.positionZ, 1.0f};
        pack.vertices.push_back({
            position,
            {source.normalX, source.normalY, source.normalZ, 1.0f},
            boneTangent(source),
            {0.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f, 1.0f},
        });
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    for (std::uint32_t offset = 0u; offset < bone.indexCount; ++offset) {
        pack.indices.push_back(
            vertexBase + bones.indices[bone.firstIndex + offset] - bone.firstVertex
        );
    }
    result.indexCount = bone.indexCount;
    return result;
}

GeometryRange appendOpenKneeGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const metalrobo::NumiHumanKneePayload& knee,
    const metalrobo::NumiHumanKneeRegion& region,
    const LoadedOpenKneeLigamentFEM* ligamentFEM
) {
    const auto surface = std::find_if(
        knee.surfaces.begin() + region.firstSurface,
        knee.surfaces.begin() + region.firstSurface + region.surfaceCount,
        [](const metalrobo::NumiHumanKneeSurface& candidate) {
            return candidate.isAllFaces;
        }
    );
    require(surface != knee.surfaces.begin() + region.firstSurface + region.surfaceCount,
            "NHKNEE1 region has no exact all-faces surface");
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f,
    };
    result.maximum = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f,
    };
    const bool usesAcceptedFEM = ligamentFEM != nullptr &&
        region.firstNode < ligamentFEM->deformedNodes.size() &&
        ligamentFEM->deformedNodes[region.firstNode];
    if (usesAcceptedFEM) {
        require(region.firstNode <= ligamentFEM->worldNodes.size() &&
                    region.nodeCount <= ligamentFEM->worldNodes.size() - region.firstNode &&
                    std::all_of(
                        ligamentFEM->deformedNodes.begin() + region.firstNode,
                        ligamentFEM->deformedNodes.begin() +
                            region.firstNode + region.nodeCount,
                        [](const bool value) { return value; }),
                "accepted Open Knee ligament FEM region is incomplete");
    }
    const bool resolveProjectedRestLocal = !usesAcceptedFEM &&
        ligamentFEM != nullptr && ligamentFEM->liveHumanCoupling &&
        region.visualBodyIndex < ligamentFEM->projectedRestBodies.size();
    const auto point = [&knee, &region, ligamentFEM, usesAcceptedFEM,
                        resolveProjectedRestLocal](
        const std::uint32_t globalNode
    ) {
        if (usesAcceptedFEM) return ligamentFEM->worldNodes[globalNode];
        if (resolveProjectedRestLocal) {
            const MRBodyStateGPU& restBody =
                ligamentFEM->projectedRestBodies[region.visualBodyIndex];
            const mr_float4 inverseRestOrientation{
                -restBody.orientation.x, -restBody.orientation.y,
                -restBody.orientation.z, restBody.orientation.w};
            const auto& source = knee.nodes[globalNode].restWorld;
            mr_float4 resolved = rotatePoint(
                inverseRestOrientation,
                subtractPoint(
                    mr_float4{source[0u], source[1u], source[2u], 1.0f},
                    restBody.position));
            resolved.w = 1.0f;
            return resolved;
        }
        const auto& local = knee.nodes[globalNode].visualLocal;
        return mr_float4{local[0u], local[1u], local[2u], 1.0f};
    };
    std::vector<mr_float4> normals(region.nodeCount, {0.0f, 0.0f, 0.0f, 0.0f});
    for (std::uint32_t offset = 0u; offset < surface->faceCount; ++offset) {
        const auto& face = knee.faces[surface->firstFace + offset];
        const auto local0 = face[0] - region.firstNode;
        const auto local1 = face[1] - region.firstNode;
        const auto local2 = face[2] - region.firstNode;
        require(local0 < region.nodeCount && local1 < region.nodeCount && local2 < region.nodeCount,
                "NHKNEE1 presentation surface crosses region ownership");
        const mr_float4 p0 = point(face[0]);
        const mr_float4 p1 = point(face[1]);
        const mr_float4 p2 = point(face[2]);
        const mr_float4 edge1{p1.x - p0.x, p1.y - p0.y, p1.z - p0.z, 0.0f};
        const mr_float4 edge2{p2.x - p0.x, p2.y - p0.y, p2.z - p0.z, 0.0f};
        const mr_float4 normal{
            edge1.y * edge2.z - edge1.z * edge2.y,
            edge1.z * edge2.x - edge1.x * edge2.z,
            edge1.x * edge2.y - edge1.y * edge2.x,
            0.0f,
        };
        for (const auto local : {local0, local1, local2}) {
            normals[local].x += normal.x;
            normals[local].y += normal.y;
            normals[local].z += normal.z;
        }
    }
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
        const float length = std::sqrt(
            normals[local].x * normals[local].x + normals[local].y * normals[local].y +
            normals[local].z * normals[local].z
        );
        const mr_float4 normal = length > 1.0e-12f
            ? mr_float4{normals[local].x / length, normals[local].y / length,
                        normals[local].z / length, 1.0f}
            : mr_float4{0.0f, 0.0f, 1.0f, 1.0f};
        const mr_float4 position = point(region.firstNode + local);
        pack.vertices.push_back({
            position, normal, normalTangent(normal),
            {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f},
        });
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    for (std::uint32_t offset = 0u; offset < surface->faceCount; ++offset) {
        const auto& face = knee.faces[surface->firstFace + offset];
        pack.indices.push_back(vertexBase + face[0] - region.firstNode);
        pack.indices.push_back(vertexBase + face[1] - region.firstNode);
        pack.indices.push_back(vertexBase + face[2] - region.firstNode);
    }
    result.indexCount = 3u * surface->faceCount;
    return result;
}

mr_float4 torsoAnatomyTangent(const TorsoAnatomyVertex& vertex) {
    return normalTangent({vertex.normalX, vertex.normalY, vertex.normalZ, 1.0f});
}

GeometryRange appendTorsoAnatomyGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const LoadedTorsoAnatomy& anatomy,
    const TorsoAnatomyRecord& surface
) {
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f,
    };
    result.maximum = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f,
    };
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t offset = 0u; offset < surface.vertexCount; ++offset) {
        const TorsoAnatomyVertex& source = anatomy.vertices[surface.firstVertex + offset];
        const mr_float4 position{source.positionX, source.positionY, source.positionZ, 1.0f};
        pack.vertices.push_back({
            position,
            {source.normalX, source.normalY, source.normalZ, 1.0f},
            torsoAnatomyTangent(source),
            {0.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f, 1.0f},
        });
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    for (std::uint32_t offset = 0u; offset < surface.indexCount; ++offset) {
        pack.indices.push_back(
            vertexBase + anatomy.indices[surface.firstIndex + offset] - surface.firstVertex
        );
    }
    result.indexCount = surface.indexCount;
    return result;
}

mr_float4 softTissueVertexWorld(
    const SoftTissueRecord& tissue,
    const SoftTissueVertex& vertex,
    const MRBodyStateGPU& body,
    const std::uint32_t binding
) {
    require(binding < tissue.bindingCount &&
                tissue.bodyIndex[binding] != MR_INVALID_INDEX,
            "BodyParts3D soft-tissue vertex requests an absent binding");
    const mr_float4 localRotation{
        tissue.binding[binding].quaternion[0], tissue.binding[binding].quaternion[1],
        tissue.binding[binding].quaternion[2], tissue.binding[binding].quaternion[3],
    };
    const mr_float4 localTranslation{
        tissue.binding[binding].translation[0], tissue.binding[binding].translation[1],
        tissue.binding[binding].translation[2], 0.0f,
    };
    const float localScale = tissue.binding[binding].uniformScale;
    const mr_float4 local = addPoint(
        localTranslation,
        scalePoint(
            rotatePoint(localRotation, {vertex.positionX, vertex.positionY, vertex.positionZ, 0.0f}),
            localScale
        )
    );
    return addPoint(body.position, rotatePoint(body.orientation, local));
}

mr_float4 softTissueVertexNormalWorld(
    const SoftTissueRecord& tissue,
    const SoftTissueVertex& vertex,
    const MRBodyStateGPU& body,
    const std::uint32_t binding
) {
    require(binding < tissue.bindingCount &&
                tissue.bodyIndex[binding] != MR_INVALID_INDEX,
            "BodyParts3D soft-tissue normal requests an absent binding");
    const mr_float4 localRotation{
        tissue.binding[binding].quaternion[0], tissue.binding[binding].quaternion[1],
        tissue.binding[binding].quaternion[2], tissue.binding[binding].quaternion[3],
    };
    mr_float4 normal = rotatePoint(
        body.orientation,
        rotatePoint(localRotation, {vertex.normalX, vertex.normalY, vertex.normalZ, 0.0f})
    );
    const float length = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    require(length > 1.0e-6f, "BodyParts3D soft-tissue normal is degenerate");
    normal.x /= length;
    normal.y /= length;
    normal.z /= length;
    normal.w = 0.0f;
    return normal;
}

mr_float4 softTissueVertexBlendedWorld(
    const SoftTissueRecord& tissue,
    const SoftTissueVertex& vertex,
    const std::span<const MRBodyStateGPU> bodies
) {
    mr_float4 result{0.0f, 0.0f, 0.0f, 1.0f};
    float totalWeight = 0.0f;
    for (std::uint32_t influence = 0u;
         influence < kRouteSoftTissueMaximumInfluences; ++influence) {
        const float weight = vertex.weight[influence];
        if (weight <= 2.0e-6f) continue;
        const std::uint32_t binding = vertex.bindingIndex[influence];
        require(binding < tissue.bindingCount,
                "BodyParts3D soft-tissue vertex weights an absent binding");
        require(tissue.bodyIndex[binding] < bodies.size(),
                "BodyParts3D soft-tissue body binding exceeds the rendered pose");
        const mr_float4 position = softTissueVertexWorld(
            tissue, vertex, bodies[tissue.bodyIndex[binding]], binding
        );
        result.x += weight * position.x;
        result.y += weight * position.y;
        result.z += weight * position.z;
        totalWeight += weight;
    }
    require(std::abs(totalWeight - 1.0f) <= 2.0e-3f,
            "BodyParts3D soft-tissue vertex has non-unit active weights");
    return result;
}

mr_float4 softTissueVertexBlendedNormalWorld(
    const SoftTissueRecord& tissue,
    const SoftTissueVertex& vertex,
    const std::span<const MRBodyStateGPU> bodies
) {
    mr_float4 normal{0.0f, 0.0f, 0.0f, 0.0f};
    float totalWeight = 0.0f;
    for (std::uint32_t influence = 0u;
         influence < kRouteSoftTissueMaximumInfluences; ++influence) {
        const float weight = vertex.weight[influence];
        if (weight <= 2.0e-6f) continue;
        const std::uint32_t binding = vertex.bindingIndex[influence];
        require(binding < tissue.bindingCount,
                "BodyParts3D soft-tissue normal weights an absent binding");
        require(tissue.bodyIndex[binding] < bodies.size(),
                "BodyParts3D soft-tissue body binding exceeds the rendered pose");
        const mr_float4 bindingNormal = softTissueVertexNormalWorld(
            tissue, vertex, bodies[tissue.bodyIndex[binding]], binding
        );
        normal.x += weight * bindingNormal.x;
        normal.y += weight * bindingNormal.y;
        normal.z += weight * bindingNormal.z;
        totalWeight += weight;
    }
    require(std::abs(totalWeight - 1.0f) <= 2.0e-3f,
            "BodyParts3D soft-tissue normal has non-unit active weights");
    const float normalLength = std::sqrt(
        normal.x * normal.x + normal.y * normal.y + normal.z * normal.z
    );
    require(normalLength > 1.0e-6f, "BodyParts3D blended soft-tissue normal is degenerate");
    normal.x /= normalLength;
    normal.y /= normalLength;
    normal.z /= normalLength;
    return normal;
}

GeometryRange appendSoftTissueGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const LoadedSoftTissues& tissues,
    const SoftTissueRecord& tissue,
    const std::span<const MRBodyStateGPU> bodies
) {
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f,
    };
    result.maximum = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f,
    };
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t offset = 0u; offset < tissue.vertexCount; ++offset) {
        const SoftTissueVertex& source = tissues.vertices[tissue.firstVertex + offset];
        const mr_float4 position = softTissueVertexBlendedWorld(tissue, source, bodies);
        mr_float4 normal = softTissueVertexBlendedNormalWorld(tissue, source, bodies);
        normal.w = 1.0f;
        pack.vertices.push_back({
            position,
            normal,
            normalTangent(normal),
            {0.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f, 1.0f},
        });
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    for (std::uint32_t offset = 0u; offset < tissue.indexCount; ++offset) {
        pack.indices.push_back(
            vertexBase + tissues.indices[tissue.firstIndex + offset] - tissue.firstVertex
        );
    }
    result.indexCount = tissue.indexCount;
    return result;
}

std::uint32_t softTissueLastBinding(const SoftTissueRecord& tissue) {
    require(tissue.bindingCount > 0u,
            "BodyParts3D soft-tissue has no articulated body binding");
    return tissue.bindingCount - 1u;
}

float softTissueVertexWeightForBinding(
    const SoftTissueVertex& vertex,
    const std::uint32_t binding
) {
    float result = 0.0f;
    for (std::uint32_t influence = 0u;
         influence < kRouteSoftTissueMaximumInfluences; ++influence) {
        if (vertex.bindingIndex[influence] == binding) {
            result += vertex.weight[influence];
        }
    }
    return result;
}

// BodyParts3D's source tendon, muscle, and bone meshes are separate surfaces.
// A source-triangle proximity lock already keeps the named tendon end on the
// calcaneus frame, but separate topology can still leave a dark raster seam
// at its bone insertion. This renders a very short source-proximity collar
// only along an open boundary whose vertices are already locked to the named
// distal bone. It intentionally does not infer a muscle-to-tendon bridge:
// muscle force paths remain the authored MyoSim spatial tendons and this does
// not add a spring, a weld, contact, or a material law.
GeometryRange appendTendonAttachmentCollarGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const LoadedSoftTissues& tissues,
    const LoadedBones& bones,
    const SoftTissueRecord& tendon,
    const std::span<const MRBodyStateGPU> bodies
) {
    require(tendon.layer == kSoftTissueLayerTendon,
            "tendon attachment collar requested for a non-tendon surface");
    const std::uint32_t tendonDistalBinding = softTissueLastBinding(tendon);
    const std::uint32_t tendonDistalBody = tendon.bodyIndex[tendonDistalBinding];
    require(tendonDistalBody < bodies.size(),
            "tendon attachment collar distal body exceeds the rendered pose");
    constexpr float kMaximumSourceSurfaceGapMeters = 0.030f;
    constexpr float kSurfaceOverlapMeters = 0.0015f;
    constexpr float kSecondaryWeightTolerance = 0.02f;
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f,
    };
    result.maximum = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f,
    };

    std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
    edges.reserve(tendon.indexCount);
    const auto appendEdge = [&edges](const std::uint32_t first, const std::uint32_t second) {
        edges.emplace_back(std::min(first, second), std::max(first, second));
    };
    for (std::uint32_t offset = 0u; offset < tendon.indexCount; offset += 3u) {
        const std::uint32_t first = tissues.indices[tendon.firstIndex + offset];
        const std::uint32_t second = tissues.indices[tendon.firstIndex + offset + 1u];
        const std::uint32_t third = tissues.indices[tendon.firstIndex + offset + 2u];
        appendEdge(first, second);
        appendEdge(second, third);
        appendEdge(third, first);
    }
    std::sort(edges.begin(), edges.end());

    struct SurfaceClosest {
        mr_float4 point{};
        mr_float4 normal{};
        float distanceSquared = std::numeric_limits<float>::infinity();
    };
    const auto closestBonePoint = [&bones, &bodies, tendonDistalBody](const mr_float4 point) {
        SurfaceClosest result;
        for (const BoneRecord& bone : bones.records) {
            if (bone.bodyIndex != tendonDistalBody) continue;
            for (std::uint32_t offset = 0u; offset < bone.indexCount; offset += 3u) {
                const std::uint32_t first = bones.indices[bone.firstIndex + offset];
                const std::uint32_t second = bones.indices[bone.firstIndex + offset + 1u];
                const std::uint32_t third = bones.indices[bone.firstIndex + offset + 2u];
                const mr_float4 candidate = closestPointOnTriangle(
                    point,
                    boneVertexWorld(bone, bones.vertices[first], bodies[bone.bodyIndex]),
                    boneVertexWorld(bone, bones.vertices[second], bodies[bone.bodyIndex]),
                    boneVertexWorld(bone, bones.vertices[third], bodies[bone.bodyIndex])
                );
                const mr_float4 difference = subtractPoint(point, candidate);
                const float distanceSquared = dotPoint(difference, difference);
                if (distanceSquared >= result.distanceSquared) continue;
                const mr_float4 firstNormal = boneVertexNormalWorld(
                    bone, bones.vertices[first], bodies[bone.bodyIndex]
                );
                const mr_float4 secondNormal = boneVertexNormalWorld(
                    bone, bones.vertices[second], bodies[bone.bodyIndex]
                );
                const mr_float4 thirdNormal = boneVertexNormalWorld(
                    bone, bones.vertices[third], bodies[bone.bodyIndex]
                );
                mr_float4 normal{
                    firstNormal.x + secondNormal.x + thirdNormal.x,
                    firstNormal.y + secondNormal.y + thirdNormal.y,
                    firstNormal.z + secondNormal.z + thirdNormal.z,
                    0.0f,
                };
                const float normalLength = std::sqrt(dotPoint(normal, normal));
                require(normalLength > 1.0e-6f,
                        "tendon attachment collar encountered a degenerate bone normal");
                normal = scalePoint(normal, 1.0f / normalLength);
                result.point = candidate;
                result.normal = normal;
                result.distanceSquared = distanceSquared;
            }
        }
        return result;
    };
    const auto appendVertex = [&pack, &result](const mr_float4 position, mr_float4 normal) {
        const float normalLength = std::sqrt(dotPoint(normal, normal));
        require(normalLength > 1.0e-6f, "tendon attachment collar normal is degenerate");
        normal = scalePoint(normal, 1.0f / normalLength);
        normal.w = 1.0f;
        pack.vertices.push_back({
            position, normal, normalTangent(normal),
            {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f},
        });
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    };

    for (std::size_t begin = 0u; begin < edges.size();) {
        std::size_t end = begin + 1u;
        while (end < edges.size() && edges[end] == edges[begin]) ++end;
        if (end - begin != 1u) {
            begin = end;
            continue;
        }
        const auto [firstIndex, secondIndex] = edges[begin];
        const SoftTissueVertex& first = tissues.vertices[firstIndex];
        const SoftTissueVertex& second = tissues.vertices[secondIndex];
        const mr_float4 firstSource = softTissueVertexBlendedWorld(tendon, first, bodies);
        const mr_float4 secondSource = softTissueVertexBlendedWorld(tendon, second, bodies);
        const mr_float4 middleSource = addPoint(
            scalePoint(addPoint(firstSource, secondSource), 0.5f), {0.0f, 0.0f, 0.0f, 0.0f}
        );
        const bool secondaryBoneEnd =
            softTissueVertexWeightForBinding(first, tendonDistalBinding) >=
                1.0f - kSecondaryWeightTolerance &&
            softTissueVertexWeightForBinding(second, tendonDistalBinding) >=
                1.0f - kSecondaryWeightTolerance;
        if (!secondaryBoneEnd) {
            begin = end;
            continue;
        }
        const SurfaceClosest middleBone = closestBonePoint(middleSource);
        if (!std::isfinite(middleBone.distanceSquared) ||
            middleBone.distanceSquared >
                kMaximumSourceSurfaceGapMeters * kMaximumSourceSurfaceGapMeters) {
            begin = end;
            continue;
        }
        const SurfaceClosest firstClosest = closestBonePoint(firstSource);
        const SurfaceClosest secondClosest = closestBonePoint(secondSource);
        const float firstDistanceSquared = firstClosest.distanceSquared;
        const float secondDistanceSquared = secondClosest.distanceSquared;
        if (!(std::isfinite(firstDistanceSquared) && std::isfinite(secondDistanceSquared)) ||
            firstDistanceSquared > kMaximumSourceSurfaceGapMeters * kMaximumSourceSurfaceGapMeters ||
            secondDistanceSquared > kMaximumSourceSurfaceGapMeters * kMaximumSourceSurfaceGapMeters) {
            begin = end;
            continue;
        }
        const mr_float4 firstDirection = subtractPoint(firstSource, firstClosest.point);
        const mr_float4 secondDirection = subtractPoint(secondSource, secondClosest.point);
        const float firstSide = dotPoint(firstDirection, firstClosest.normal) >= 0.0f ? 1.0f : -1.0f;
        const float secondSide = dotPoint(secondDirection, secondClosest.normal) >= 0.0f ? 1.0f : -1.0f;
        const mr_float4 firstTarget = addPoint(
            firstClosest.point, scalePoint(firstClosest.normal, firstSide * kSurfaceOverlapMeters)
        );
        const mr_float4 secondTarget = addPoint(
            secondClosest.point, scalePoint(secondClosest.normal, secondSide * kSurfaceOverlapMeters)
        );
        const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
        appendVertex(firstSource, softTissueVertexBlendedNormalWorld(tendon, first, bodies));
        appendVertex(secondSource, softTissueVertexBlendedNormalWorld(tendon, second, bodies));
        appendVertex(secondTarget, secondClosest.normal);
        appendVertex(firstTarget, firstClosest.normal);
        pack.indices.insert(pack.indices.end(), {
            vertexBase, vertexBase + 1u, vertexBase + 2u,
            vertexBase, vertexBase + 2u, vertexBase + 3u,
        });
        begin = end;
    }
    result.indexCount = static_cast<std::uint32_t>(pack.indices.size()) - result.firstIndex;
    return result;
}

mr_float4 skinVertexWorld(
    const LoadedSkin& skin,
    const SkinVertex& vertex,
    const std::span<const MRBodyStateGPU> bodies
) {
    mr_float4 position{0.0f, 0.0f, 0.0f, 1.0f};
    for (std::size_t influence = 0u; influence < 4u; ++influence) {
        const SkinBindingRecord& binding = skin.bindings[vertex.bindingIndex[influence]];
        require(binding.bodyIndex < bodies.size(),
                "BodyParts3D skinned-shell body binding exceeds the rendered pose");
        const mr_float4 local = addPoint(
            {binding.translationX, binding.translationY, binding.translationZ, 0.0f},
            scalePoint(
                rotatePoint(
                    {binding.quaternionX, binding.quaternionY,
                     binding.quaternionZ, binding.quaternionW},
                    {vertex.positionX, vertex.positionY, vertex.positionZ, 0.0f}
                ),
                binding.uniformScale
            )
        );
        const mr_float4 world = addPoint(
            bodies[binding.bodyIndex].position,
            rotatePoint(bodies[binding.bodyIndex].orientation, local)
        );
        position.x += vertex.weight[influence] * world.x;
        position.y += vertex.weight[influence] * world.y;
        position.z += vertex.weight[influence] * world.z;
    }
    return position;
}

mr_float4 skinVertexNormalWorld(
    const LoadedSkin& skin,
    const SkinVertex& vertex,
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const MRBodyStateGPU> restBodies
) {
    require(restBodies.size() == bodies.size(),
            "BodyParts3D skinned-shell rest pose does not match the rendered pose");
    mr_float4 normal{0.0f, 0.0f, 0.0f, 0.0f};
    std::size_t strongestInfluence = 0u;
    for (std::size_t influence = 0u; influence < 4u; ++influence) {
        if (vertex.weight[influence] > vertex.weight[strongestInfluence]) {
            strongestInfluence = influence;
        }
        const SkinBindingRecord& binding = skin.bindings[vertex.bindingIndex[influence]];
        require(binding.bodyIndex < bodies.size(),
                "BodyParts3D skinned-shell normal binding exceeds the rendered pose");
        const mr_float4 sourceNormal{
            vertex.normalX, vertex.normalY, vertex.normalZ, 0.0f,
        };
        const mr_float4 world = skin.usesWorldRestNormals
            ? rotatePoint(
                bodies[binding.bodyIndex].orientation,
                rotatePoint(
                    {
                        -restBodies[binding.bodyIndex].orientation.x,
                        -restBodies[binding.bodyIndex].orientation.y,
                        -restBodies[binding.bodyIndex].orientation.z,
                        restBodies[binding.bodyIndex].orientation.w,
                    },
                    sourceNormal
                )
            )
            : rotatePoint(
                bodies[binding.bodyIndex].orientation,
                rotatePoint(
                    {binding.quaternionX, binding.quaternionY,
                     binding.quaternionZ, binding.quaternionW},
                    sourceNormal
                )
            );
        normal.x += vertex.weight[influence] * world.x;
        normal.y += vertex.weight[influence] * world.y;
        normal.z += vertex.weight[influence] * world.z;
    }
    const float length = std::sqrt(
        normal.x * normal.x + normal.y * normal.y + normal.z * normal.z
    );
    if (length > 1.0e-6f) {
        normal.x /= length;
        normal.y /= length;
        normal.z /= length;
    } else {
        // A linear blend of four independently rotated normals can cancel at
        // a highly bent joint even though the highest-weight source influence
        // remains valid.  Preserve a finite geometric normal without changing
        // the shell position or pretending this is a continuum skin solve.
        const SkinBindingRecord& binding =
            skin.bindings[vertex.bindingIndex[strongestInfluence]];
        const mr_float4 sourceNormal{
            vertex.normalX, vertex.normalY, vertex.normalZ, 0.0f,
        };
        normal = skin.usesWorldRestNormals
            ? rotatePoint(
                bodies[binding.bodyIndex].orientation,
                rotatePoint(
                    {
                        -restBodies[binding.bodyIndex].orientation.x,
                        -restBodies[binding.bodyIndex].orientation.y,
                        -restBodies[binding.bodyIndex].orientation.z,
                        restBodies[binding.bodyIndex].orientation.w,
                    },
                    sourceNormal
                )
            )
            : rotatePoint(
                bodies[binding.bodyIndex].orientation,
                rotatePoint(
                    {binding.quaternionX, binding.quaternionY,
                     binding.quaternionZ, binding.quaternionW},
                    sourceNormal
                )
            );
        const float fallbackLength = std::sqrt(
            normal.x * normal.x + normal.y * normal.y + normal.z * normal.z
        );
        require(fallbackLength > 1.0e-6f,
                "BodyParts3D strongest skinned-shell normal is degenerate");
        normal.x /= fallbackLength;
        normal.y /= fallbackLength;
        normal.z /= fallbackLength;
    }
    normal.w = 1.0f;
    return normal;
}

GeometryRange appendSkinGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const LoadedSkin& skin,
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const MRBodyStateGPU> restBodies
) {
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f,
    };
    result.maximum = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f,
    };
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (const SkinVertex& source : skin.vertices) {
        const mr_float4 position = skinVertexWorld(skin, source, bodies);
        const mr_float4 normal = skinVertexNormalWorld(skin, source, bodies, restBodies);
        pack.vertices.push_back({
            position,
            normal,
            normalTangent(normal),
            {0.0f, 0.0f, 0.0f, 0.0f},
            {1.0f, 1.0f, 1.0f, 1.0f},
        });
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    for (const std::uint32_t index : skin.indices) {
        require(index < skin.vertices.size(),
                "BodyParts3D skinned-shell visual index exceeds its source vertices");
        pack.indices.push_back(vertexBase + index);
    }
    result.indexCount = static_cast<std::uint32_t>(skin.indices.size());
    return result;
}

std::array<float, 3u> inertiaEllipsoid(const MRBodyPropertiesGPU& body) {
    const float mass = body.massAndInverseMass.x;
    if (!(mass > 1.0e-5f)) {
        return {0.010f, 0.010f, 0.010f};
    }
    const float ixx = std::max(body.inertiaRow0.x, 1.0e-8f);
    const float iyy = std::max(body.inertiaRow1.y, 1.0e-8f);
    const float izz = std::max(body.inertiaRow2.z, 1.0e-8f);
    const auto semiAxis = [](const float squared) {
        return std::clamp(std::sqrt(std::max(squared, 1.0e-6f)), 0.014f, 0.115f);
    };
    return {
        semiAxis(2.5f * (iyy + izz - ixx) / mass),
        semiAxis(2.5f * (ixx + izz - iyy) / mass),
        semiAxis(2.5f * (ixx + iyy - izz) / mass),
    };
}

GeometryRange appendWorldTube(
    metalrobo::VisualAssetPackV2& pack,
    const mr_float4 start,
    const mr_float4 end,
    const float radius
) {
    constexpr std::uint32_t kSides = 6u;
    const mr_float4 axisRaw{
        end.x - start.x, end.y - start.y, end.z - start.z, 0.0f,
    };
    const float axisLength = std::sqrt(
        axisRaw.x * axisRaw.x + axisRaw.y * axisRaw.y + axisRaw.z * axisRaw.z
    );
    require(axisLength > 1.0e-5f && radius > 0.0f, "MyoSim route tube is degenerate");
    const mr_float4 axis{
        axisRaw.x / axisLength, axisRaw.y / axisLength, axisRaw.z / axisLength, 0.0f,
    };
    const auto cross = [](const mr_float4 left, const mr_float4 right) {
        return mr_float4{
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x,
            0.0f,
        };
    };
    const mr_float4 reference = std::abs(axis.z) < 0.85f
        ? mr_float4{0.0f, 0.0f, 1.0f, 0.0f}
        : mr_float4{0.0f, 1.0f, 0.0f, 0.0f};
    mr_float4 normal = cross(reference, axis);
    const float normalLength = std::sqrt(
        normal.x * normal.x + normal.y * normal.y + normal.z * normal.z
    );
    require(normalLength > 1.0e-5f, "MyoSim route tube has no normal basis");
    normal.x /= normalLength;
    normal.y /= normalLength;
    normal.z /= normalLength;
    const mr_float4 binormal = cross(axis, normal);
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f,
    };
    result.maximum = {
        -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f,
    };
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t ring = 0u; ring < 2u; ++ring) {
        const mr_float4 center = ring == 0u ? start : end;
        for (std::uint32_t side = 0u; side < kSides; ++side) {
            const float angle = 2.0f * std::numbers::pi_v<float> *
                static_cast<float>(side) / static_cast<float>(kSides);
            const mr_float4 surfaceNormal{
                normal.x * std::cos(angle) + binormal.x * std::sin(angle),
                normal.y * std::cos(angle) + binormal.y * std::sin(angle),
                normal.z * std::cos(angle) + binormal.z * std::sin(angle),
                1.0f,
            };
            const mr_float4 position{
                center.x + radius * surfaceNormal.x,
                center.y + radius * surfaceNormal.y,
                center.z + radius * surfaceNormal.z,
                1.0f,
            };
            pack.vertices.push_back({
                position, surfaceNormal, axis,
                {static_cast<float>(ring), static_cast<float>(side) / static_cast<float>(kSides), 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f, 1.0f},
            });
            result.minimum.x = std::min(result.minimum.x, position.x);
            result.minimum.y = std::min(result.minimum.y, position.y);
            result.minimum.z = std::min(result.minimum.z, position.z);
            result.maximum.x = std::max(result.maximum.x, position.x);
            result.maximum.y = std::max(result.maximum.y, position.y);
            result.maximum.z = std::max(result.maximum.z, position.z);
        }
    }
    for (std::uint32_t side = 0u; side < kSides; ++side) {
        const std::uint32_t next = (side + 1u) % kSides;
        const std::uint32_t a = vertexBase + side;
        const std::uint32_t b = vertexBase + next;
        const std::uint32_t c = vertexBase + kSides + side;
        const std::uint32_t d = vertexBase + kSides + next;
        pack.indices.insert(pack.indices.end(), {a, c, b, b, c, d});
    }
    result.indexCount = static_cast<std::uint32_t>(pack.indices.size()) - result.firstIndex;
    return result;
}

// This small continuum cage is intentionally a specimen, not a second
// anatomical model.  Its two end rings are derived from one exact BodyParts3D
// source muscle surface and follow that surface's two named MyoSim endpoint
// bodies.  The six intermediate nodes are free Matter FEM nodes.  This gives
// us an executable, inspectable deformation bridge without pretending that a
// coarse cage is a calibrated volumetric segmentation of the whole muscle.
struct PassiveFEMTissueVisual {
    std::uint32_t stableId = 0u;
    std::array<std::uint32_t, 2u> endpointBodies{};
    std::vector<mr_float4> restNodes;
    std::vector<mr_float4> nodes;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t completedSteps = 0u;
    std::uint32_t fgmresIterations = 0u;
    float maximumAnchorDisplacementMeters = 0.0f;
    float maximumFreeDisplacementMeters = 0.0f;
    float minimumDeterminant = std::numeric_limits<float>::infinity();
    double gpuMilliseconds = 0.0;
    std::string deviceName;
};

mr_float4 femSubtract(const mr_float4& left, const mr_float4& right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z, 0.0f};
}

mr_float4 femAdd(const mr_float4& left, const mr_float4& right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z, 1.0f};
}

mr_float4 femScale(const mr_float4& value, const float scale) {
    return {value.x * scale, value.y * scale, value.z * scale, 0.0f};
}

float femDot(const mr_float4& left, const mr_float4& right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

mr_float4 femCross(const mr_float4& left, const mr_float4& right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
        0.0f,
    };
}

float femLength(const mr_float4& value) {
    return std::sqrt(femDot(value, value));
}

mr_float4 femNormalized(const mr_float4& value, const char* context) {
    const float length = femLength(value);
    require(std::isfinite(length) && length > 1.0e-6f,
            std::string(context) + " is degenerate");
    return femScale(value, 1.0f / length);
}

mr_float4 femLerp(const mr_float4& first, const mr_float4& second, const float t) {
    return {
        first.x + (second.x - first.x) * t,
        first.y + (second.y - first.y) * t,
        first.z + (second.z - first.z) * t,
        1.0f,
    };
}

mr_float4 femRingCenter(
    const std::span<const mr_float4> nodes, const std::uint32_t ring
) {
    require(nodes.size() == 12u && ring < 4u, "FEM tissue ring is invalid");
    const std::uint32_t offset = ring * 3u;
    return {
        (nodes[offset].x + nodes[offset + 1u].x + nodes[offset + 2u].x) / 3.0f,
        (nodes[offset].y + nodes[offset + 1u].y + nodes[offset + 2u].y) / 3.0f,
        (nodes[offset].z + nodes[offset + 1u].z + nodes[offset + 2u].z) / 3.0f,
        1.0f,
    };
}

float femRingRadius(
    const std::span<const mr_float4> nodes, const std::uint32_t ring
) {
    const mr_float4 center = femRingCenter(nodes, ring);
    float result = 0.0f;
    for (std::uint32_t side = 0u; side < 3u; ++side) {
        result += femLength(femSubtract(nodes[ring * 3u + side], center));
    }
    result /= 3.0f;
    require(std::isfinite(result) && result > 1.0e-4f,
            "FEM tissue ring radius is invalid");
    return result;
}

mr_float4 femRotateRestPointToBody(
    const mr_float4& restPoint,
    const MRBodyStateGPU& restBody,
    const MRBodyStateGPU& drivenBody
) {
    const mr_float4 inverseRestOrientation{
        -restBody.orientation.x, -restBody.orientation.y,
        -restBody.orientation.z, restBody.orientation.w,
    };
    const mr_float4 local = rotatePoint(
        inverseRestOrientation, femSubtract(restPoint, restBody.position)
    );
    return femAdd(drivenBody.position, rotatePoint(drivenBody.orientation, local));
}

double femSignedTetrahedronVolume(
    const std::vector<std::array<double, 3u>>& nodes,
    const std::array<std::uint32_t, 4u>& tetrahedron
) {
    const auto difference = [&nodes, &tetrahedron](
        const std::uint32_t left, const std::uint32_t right
    ) {
        return std::array<double, 3u>{
            nodes[tetrahedron[left]][0] - nodes[tetrahedron[right]][0],
            nodes[tetrahedron[left]][1] - nodes[tetrahedron[right]][1],
            nodes[tetrahedron[left]][2] - nodes[tetrahedron[right]][2],
        };
    };
    const auto first = difference(1u, 0u);
    const auto second = difference(2u, 0u);
    const auto third = difference(3u, 0u);
    const std::array<double, 3u> cross{
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    };
    return (cross[0] * third[0] + cross[1] * third[1] + cross[2] * third[2]) / 6.0;
}

PassiveFEMTissueVisual runPassiveFEMTissue(
    const LoadedSoftTissues& source,
    const std::span<const MRBodyStateGPU> restBodies,
    const std::span<const MRBodyStateGPU> drivenBodies,
    const std::uint32_t stableId,
    const double timestepSeconds,
    const std::uint32_t stepCount,
    const std::filesystem::path& matterMetallib
) {
    require(restBodies.size() == drivenBodies.size() && !restBodies.empty(),
            "FEM tissue source body poses are inconsistent");
    require(std::isfinite(timestepSeconds) && timestepSeconds >= 1.0e-6 &&
                timestepSeconds <= 1.0e-3 && stepCount >= 1u && stepCount <= 64u,
            "FEM tissue integration range is invalid");
    const auto selected = std::find_if(
        source.records.begin(), source.records.end(),
        [stableId](const SoftTissueRecord& value) {
            return value.stableId == stableId;
        }
    );
    require(selected != source.records.end(),
            "requested FEM tissue stable ID is not present");
    const SoftTissueRecord& tissue = *selected;
    require(tissue.layer == kSoftTissueLayerMuscle &&
                tissue.bindingCount == 2u &&
                tissue.bodyIndex[0] < restBodies.size() &&
                tissue.bodyIndex[1] < restBodies.size(),
            "native FEM tissue currently requires one two-body source muscle surface");

    std::vector<mr_float4> surfacePoints;
    std::vector<mr_float4> firstEndpointPoints;
    std::vector<mr_float4> secondEndpointPoints;
    surfacePoints.reserve(tissue.vertexCount);
    for (std::uint32_t offset = 0u; offset < tissue.vertexCount; ++offset) {
        const SoftTissueVertex& vertex = source.vertices[tissue.firstVertex + offset];
        const mr_float4 point = softTissueVertexBlendedWorld(tissue, vertex, restBodies);
        surfacePoints.push_back(point);
        if (softTissueVertexWeightForBinding(vertex, 0u) >= 0.85f) {
            firstEndpointPoints.push_back(point);
        }
        if (softTissueVertexWeightForBinding(vertex, 1u) >= 0.85f) {
            secondEndpointPoints.push_back(point);
        }
    }
    require(firstEndpointPoints.size() >= 3u && secondEndpointPoints.size() >= 3u,
            "source muscle has insufficient endpoint-weighted surface samples for FEM");
    const auto average = [](const std::span<const mr_float4> points) {
        mr_float4 result{0.0f, 0.0f, 0.0f, 1.0f};
        for (const mr_float4& point : points) {
            result.x += point.x;
            result.y += point.y;
            result.z += point.z;
        }
        const float inverse = 1.0f / static_cast<float>(points.size());
        result.x *= inverse;
        result.y *= inverse;
        result.z *= inverse;
        return result;
    };
    const mr_float4 firstCenter = average(firstEndpointPoints);
    const mr_float4 secondCenter = average(secondEndpointPoints);
    const mr_float4 axis = femNormalized(
        femSubtract(secondCenter, firstCenter), "source muscle endpoint axis"
    );
    const auto endpointRadius = [&axis](
        const std::span<const mr_float4> points, const mr_float4& center
    ) {
        std::vector<float> radii;
        radii.reserve(points.size());
        for (const mr_float4& point : points) {
            const mr_float4 difference = femSubtract(point, center);
            const mr_float4 radial = femSubtract(
                difference, femScale(axis, femDot(difference, axis))
            );
            radii.push_back(femLength(radial));
        }
        std::sort(radii.begin(), radii.end());
        const float result = radii[radii.size() / 2u];
        require(std::isfinite(result) && result >= 0.004f && result <= 0.120f,
                "source muscle endpoint radius is not usable for FEM");
        return result;
    };
    const float firstRadius = endpointRadius(firstEndpointPoints, firstCenter);
    const float secondRadius = endpointRadius(secondEndpointPoints, secondCenter);
    const mr_float4 reference = std::abs(axis.z) < 0.9f
        ? mr_float4{0.0f, 0.0f, 1.0f, 0.0f}
        : mr_float4{0.0f, 1.0f, 0.0f, 0.0f};
    const mr_float4 basisU = femNormalized(femCross(axis, reference), "source muscle FEM basis");
    const mr_float4 basisV = femNormalized(femCross(axis, basisU), "source muscle FEM binormal");

    PassiveFEMTissueVisual result;
    result.stableId = stableId;
    result.endpointBodies = {tissue.bodyIndex[0], tissue.bodyIndex[1]};
    result.restNodes.reserve(12u);
    for (std::uint32_t ring = 0u; ring < 4u; ++ring) {
        const float fraction = static_cast<float>(ring) / 3.0f;
        const mr_float4 center = femLerp(firstCenter, secondCenter, fraction);
        const float radius = firstRadius + (secondRadius - firstRadius) * fraction;
        for (std::uint32_t side = 0u; side < 3u; ++side) {
            const float angle = 2.0f * std::numbers::pi_v<float> *
                static_cast<float>(side) / 3.0f;
            const mr_float4 radial = femAdd(
                femScale(basisU, radius * std::cos(angle)),
                femScale(basisV, radius * std::sin(angle))
            );
            const mr_float4 node = femAdd(center, radial);
            result.restNodes.push_back(node);
            result.nodes.push_back(node);
        }
    }
    for (std::uint32_t side = 0u; side < 3u; ++side) {
        const std::uint32_t first = side;
        const std::uint32_t last = 9u + side;
        const mr_float4 firstDriven = femRotateRestPointToBody(
            result.restNodes[first], restBodies[tissue.bodyIndex[0]], drivenBodies[tissue.bodyIndex[0]]
        );
        const mr_float4 lastDriven = femRotateRestPointToBody(
            result.restNodes[last], restBodies[tissue.bodyIndex[1]], drivenBodies[tissue.bodyIndex[1]]
        );
        result.maximumAnchorDisplacementMeters = std::max({
            result.maximumAnchorDisplacementMeters,
            femLength(femSubtract(firstDriven, result.restNodes[first])),
            femLength(femSubtract(lastDriven, result.restNodes[last])),
        });
        result.nodes[first] = firstDriven;
        result.nodes[last] = lastDriven;
    }

    auto material = numi::matter::parseMatterFile(NUMI_HUMAN_PASSIVE_TISSUE_MATERIAL);
    require(material.succeeded(), "passive skeletal-muscle Matter material did not parse");
    numi::matter::WorldSource worldSource;
    worldSource.environmentCount = 1u;
    worldSource.frameTimestep = timestepSeconds;
    worldSource.gravity = {0.0, 0.0, 0.0};
    worldSource.mixedSolver.newtonIterations = 8u;
    worldSource.mixedSolver.fgmresIterations = 12u;
    worldSource.materials.push_back(std::move(material.material));
    numi::matter::ObjectSource object;
    object.name = "bodyparts3d_source_soleus_passive_fem_specimen";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::fem;
    object.mixedFEM = false;
    object.characteristicLength = std::max(firstRadius, secondRadius);
    for (const mr_float4& node : result.restNodes) {
        object.femNodes.push_back({node.x, node.y, node.z});
    }
    object.femFixedNodes = {0u, 1u, 2u, 9u, 10u, 11u};
    const auto appendPrism = [&object, &result](
        std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
        std::uint32_t b0, std::uint32_t b1, std::uint32_t b2
    ) {
        const std::vector<std::array<double, 3u>> nodes = [&result] {
            std::vector<std::array<double, 3u>> value;
            value.reserve(result.restNodes.size());
            for (const mr_float4& node : result.restNodes) {
                value.push_back({node.x, node.y, node.z});
            }
            return value;
        }();
        for (std::array<std::uint32_t, 4u> tetrahedron : {
                 std::array<std::uint32_t, 4u>{a0, a1, a2, b0},
                 std::array<std::uint32_t, 4u>{a1, a2, b0, b1},
                 std::array<std::uint32_t, 4u>{a2, b0, b1, b2},
             }) {
            if (femSignedTetrahedronVolume(nodes, tetrahedron) < 0.0) {
                std::swap(tetrahedron[0], tetrahedron[1]);
            }
            object.tetrahedra.push_back({tetrahedron});
        }
    };
    appendPrism(0u, 1u, 2u, 3u, 4u, 5u);
    appendPrism(3u, 4u, 5u, 6u, 7u, 8u);
    appendPrism(6u, 7u, 8u, 9u, 10u, 11u);
    worldSource.objects.push_back(std::move(object));
    numi::matter::CompileOptions options;
    options.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(worldSource, options);
    std::string compileMessage;
    for (const numi::matter::Diagnostic& diagnostic : compiled.diagnostics) {
        compileMessage += diagnostic.message + "; ";
    }
    require(compiled.succeeded(), "source-derived passive FEM tissue did not compile: " + compileMessage);
    result.tetrahedronCount = static_cast<std::uint32_t>(compiled.world.fem.tetrahedra.size());

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "no Metal device for Human passive FEM tissue");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    require(queue != nil, "could not allocate Human passive FEM command queue");
    id<MTLBuffer> worldStatuses = [device
        newBufferWithLength:sizeof(MRMetalWorldStatusGPU)
        options:MTLResourceStorageModeShared];
    require(worldStatuses != nil, "could not allocate Human passive FEM status buffer");
    auto* worldStatus = static_cast<MRMetalWorldStatusGPU*>(worldStatuses.contents);
    *worldStatus = {};
    worldStatus->code = MR_STEP_SUCCESS;

    numi::matter::Runtime runtime;
    const auto initialized = runtime.initialize(
        compiled.world,
        {
            .metallib = matterMetallib,
            .environmentCount = 1u,
            .captureEvents = true,
            .captureDiagnostics = true,
            .automaticIdentification = false,
            .adaptiveTransfer = false,
        }
    );
    require(initialized.encoded && runtime.valid(),
            "could not initialize source-derived passive FEM tissue: " + initialized.message);
    result.deviceName = initialized.device;
    auto anchored = runtime.snapshot();
    require(anchored.available && anchored.femNodes.size() == result.nodes.size(),
            "could not obtain source-derived passive FEM tissue snapshot");
    for (std::uint32_t node = 0u; node < anchored.femNodes.size(); ++node) {
        if (node < 3u || node >= 9u) {
            anchored.femNodes[node].positionAndMass.x = result.nodes[node].x;
            anchored.femNodes[node].positionAndMass.y = result.nodes[node].y;
            anchored.femNodes[node].positionAndMass.z = result.nodes[node].z;
            anchored.femNodes[node].velocityAndInverseMass.x = 0.0f;
            anchored.femNodes[node].velocityAndInverseMass.y = 0.0f;
            anchored.femNodes[node].velocityAndInverseMass.z = 0.0f;
        }
    }
    const auto restored = runtime.restore(anchored);
    require(restored.encoded, "could not apply MyoSim-driven FEM anchors: " + restored.message);
    const auto matterStatuses = (__bridge id<MTLBuffer>)runtime.statusBuffer();
    auto* statuses = static_cast<NMMatterStatusGPU*>(matterStatuses.contents);
    require(statuses != nullptr, "Human passive FEM Matter statuses are unavailable");
    for (std::uint32_t step = 0u; step < stepCount; ++step) {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        require(command != nil, "could not allocate Human passive FEM command buffer");
        numi::matter::EncodeRequest request{};
        request.commandBuffer = (__bridge void*)command;
        request.environmentStatuses = (__bridge void*)worldStatuses;
        request.phase = numi::matter::EncodePhase::preDynamics;
        request.controlStep = step;
        request.physicsSubstep = 0u;
        request.physicsSubsteps = 1u;
        request.timestepSeconds = runtime.timestepSeconds();
        auto encoded = runtime.encode(request);
        require(encoded.encoded, "could not encode Human passive FEM pre-dynamics: " + encoded.message);
        request.phase = numi::matter::EncodePhase::postCommit;
        encoded = runtime.encode(request);
        require(encoded.encoded, "could not encode Human passive FEM post-commit: " + encoded.message);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted,
                "Human passive FEM command did not complete");
        const CFTimeInterval gpuStart = command.GPUStartTime;
        const CFTimeInterval gpuEnd = command.GPUEndTime;
        if (std::isfinite(gpuStart) && std::isfinite(gpuEnd) && gpuEnd >= gpuStart) {
            result.gpuMilliseconds += 1000.0 * (gpuEnd - gpuStart);
        }
        require(statuses[0].code == NM_STATUS_SUCCESS,
                "Human passive FEM Matter status=" + std::to_string(statuses[0].code));
        result.completedSteps = step + 1u;
        result.fgmresIterations = std::max(result.fgmresIterations, statuses[0].fgmresIterations);
        result.minimumDeterminant = std::min(result.minimumDeterminant, statuses[0].diagnostics.x);
    }
    const auto final = runtime.snapshot();
    require(final.available && final.femNodes.size() == result.nodes.size(),
            "Human passive FEM final snapshot is unavailable");
    for (std::uint32_t node = 0u; node < final.femNodes.size(); ++node) {
        result.nodes[node] = {
            final.femNodes[node].positionAndMass.x,
            final.femNodes[node].positionAndMass.y,
            final.femNodes[node].positionAndMass.z,
            1.0f,
        };
        if (node >= 3u && node < 9u) {
            result.maximumFreeDisplacementMeters = std::max(
                result.maximumFreeDisplacementMeters,
                femLength(femSubtract(result.nodes[node], result.restNodes[node]))
            );
        }
    }
    require(result.completedSteps > 0u && std::isfinite(result.minimumDeterminant) &&
                result.minimumDeterminant > 0.20f,
            "Human passive FEM tissue did not publish a valid deformation certificate");
    return result;
}

struct LiveOpenKneeTissueSpec {
    std::string_view name;
    double c1MPa;
    double bulkMPa;
};

constexpr std::array<LiveOpenKneeTissueSpec, 6u> kLiveOpenKneeTissueSpecs{{
    {"PCL", 3.25, 243.90},
    {"ACL", 1.95, 146.41},
    {"MCL", 1.44, 793.65},
    {"LCL", 1.44, 793.65},
    {"PTL", 2.75, 206.61},
    {"QAT", 2.75, 206.61},
}};

struct LiveOpenKneeRegion {
    const LiveOpenKneeTissueSpec* specification = nullptr;
    std::uint32_t payloadRegion = 0u;
    std::uint32_t firstFEMNode = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    std::array<std::uint32_t, 3u> anchorCounts{};
};

void setLiveOpenKneeMaterialParameter(
    numi::matter::MaterialProgram& material,
    const std::string_view name,
    const double value
) {
    const auto found = std::find_if(
        material.parameters.begin(), material.parameters.end(),
        [name](const numi::matter::Parameter& parameter) {
            return parameter.name == name;
        });
    require(found != material.parameters.end() && std::isfinite(value) &&
                value >= found->lower && value <= found->upper,
            "live Open Knee material parameter is invalid");
    found->defaultValue = value;
}

struct LiveOpenKneeArticularContactCook {
    std::vector<NMNumiHumanArticularContactSampleGPU> samples;
    std::uint32_t pairCount = 0u;
    std::uint32_t mechanicalSampleCount = 0u;
    std::uint32_t internalSameBodySampleCount = 0u;
};

LiveOpenKneeArticularContactCook cookLiveOpenKneeArticularContact(
    const metalrobo::NumiHumanKneePayload& knee,
    const std::span<const MRBodyStateGPU> restBodies,
    const std::array<std::uint32_t, 3u>& bodyIndices
) {
    using Point = std::array<double, 3u>;
    std::vector<Point> referenceNodes;
    referenceNodes.reserve(knee.nodes.size());
    for (const auto& node : knee.nodes) {
        referenceNodes.push_back({
            node.restWorld[0u], node.restWorld[1u], node.restWorld[2u]});
    }
    const auto subtract = [](const Point& a, const Point& b) {
        return Point{a[0u] - b[0u], a[1u] - b[1u], a[2u] - b[2u]};
    };
    const auto cross = [](const Point& a, const Point& b) {
        return Point{
            a[1u] * b[2u] - a[2u] * b[1u],
            a[2u] * b[0u] - a[0u] * b[2u],
            a[0u] * b[1u] - a[1u] * b[0u]};
    };
    const auto dot = [](const Point& a, const Point& b) {
        return a[0u] * b[0u] + a[1u] * b[1u] + a[2u] * b[2u];
    };
    const auto tetrahedronVolume = [&](
        const std::array<std::uint32_t, 4u>& tetrahedron
    ) {
        const Point ad = subtract(
            referenceNodes[tetrahedron[0u]],
            referenceNodes[tetrahedron[3u]]);
        const Point bd = subtract(
            referenceNodes[tetrahedron[1u]],
            referenceNodes[tetrahedron[3u]]);
        const Point cd = subtract(
            referenceNodes[tetrahedron[2u]],
            referenceNodes[tetrahedron[3u]]);
        return std::abs(dot(ad, cross(bd, cd))) / 6.0;
    };
    const auto triangleArea = [&](
        const std::array<std::uint32_t, 3u>& face
    ) {
        const Point ab = subtract(
            referenceNodes[face[1u]], referenceNodes[face[0u]]);
        const Point ac = subtract(
            referenceNodes[face[2u]], referenceNodes[face[0u]]);
        const Point normal = cross(ab, ac);
        return 0.5 * std::sqrt(dot(normal, normal));
    };
    const auto articular = [](const metalrobo::NumiHumanKneeRegionKind kind) {
        return kind == metalrobo::NumiHumanKneeRegionKind::cartilage ||
            kind == metalrobo::NumiHumanKneeRegionKind::meniscus;
    };
    std::vector<double> regionVolumes(knee.regions.size(), 0.0);
    std::vector<double> meniscusContactAreas(knee.regions.size(), 0.0);
    for (std::uint32_t regionIndex = 0u;
         regionIndex < knee.regions.size(); ++regionIndex) {
        const auto& region = knee.regions[regionIndex];
        for (std::uint32_t local = 0u;
             local < region.tetrahedronCount; ++local) {
            regionVolumes[regionIndex] += tetrahedronVolume(
                knee.tetrahedra[region.firstTetrahedron + local]);
        }
    }
    for (const auto& pair : knee.surfacePairs) {
        const auto& master = knee.surfaces[pair.masterSurface];
        const auto& slave = knee.surfaces[pair.slaveSurface];
        if (!articular(knee.regions[master.regionIndex].kind) ||
            !articular(knee.regions[slave.regionIndex].kind)) continue;
        for (const std::uint32_t surfaceIndex :
             {pair.masterSurface, pair.slaveSurface}) {
            const auto& surface = knee.surfaces[surfaceIndex];
            if (knee.regions[surface.regionIndex].kind !=
                metalrobo::NumiHumanKneeRegionKind::meniscus) continue;
            for (std::uint32_t local = 0u;
                 local < surface.faceCount; ++local) {
                meniscusContactAreas[surface.regionIndex] += triangleArea(
                    knee.faces[surface.firstFace + local]);
            }
        }
    }
    std::vector<metalrobo::NumiHumanKneeContactRegionMaterial> materials;
    for (std::uint32_t regionIndex = 0u;
         regionIndex < knee.regions.size(); ++regionIndex) {
        const auto kind = knee.regions[regionIndex].kind;
        if (!articular(kind)) continue;
        metalrobo::NumiHumanKneeContactMaterial material;
        if (kind == metalrobo::NumiHumanKneeRegionKind::cartilage) {
            material = {
                .elasticModulusPascals = 12.0e6,
                .poissonRatio = 0.45,
                .thicknessMeters = 0.003};
        } else {
            require(regionVolumes[regionIndex] > 0.0 &&
                        meniscusContactAreas[regionIndex] > 0.0,
                    "live Open Knee meniscus thickness measure is unavailable");
            const double thickness = 2.0 * regionVolumes[regionIndex] /
                meniscusContactAreas[regionIndex];
            require(std::isfinite(thickness) && thickness >= 0.001 &&
                        thickness <= 0.015,
                    "live Open Knee geometry-derived meniscus thickness is implausible");
            material = {
                .elasticModulusPascals = 20.0e6,
                .poissonRatio = 0.30,
                .thicknessMeters = thickness};
        }
        materials.push_back({
            .regionIndex = regionIndex,
            .material = material});
    }
    metalrobo::NumiHumanKneeContactModel contactModel;
    const auto built = metalrobo::buildNumiHumanKneeArticularContactModel(
        knee, referenceNodes, materials, contactModel);
    require(built.succeeded() && contactModel.pairs.size() == 7u &&
                contactModel.samples.size() == 69701u,
            "live Open Knee exact articular contact cook failed: " +
                built.message);
    const auto materialForRegion = [&](const std::uint32_t regionIndex)
        -> const metalrobo::NumiHumanKneeContactMaterial& {
        const auto found = std::find_if(
            materials.begin(), materials.end(),
            [regionIndex](const auto& entry) {
                return entry.regionIndex == regionIndex;
            });
        require(found != materials.end(),
                "live Open Knee articular material is unavailable");
        return found->material;
    };
    const auto normalStrainPerPressure = [](
        const metalrobo::NumiHumanKneeContactMaterial& material
    ) {
        const double numerator = (1.0 + material.poissonRatio) *
            (1.0 - 2.0 * material.poissonRatio);
        const double denominator = material.elasticModulusPascals *
            (1.0 - material.poissonRatio);
        const double compliance = numerator / denominator;
        require(std::isfinite(compliance) && compliance > 0.0,
                "live Open Knee normal-strain compliance is invalid");
        return compliance;
    };
    const auto ownerBody = [&](const std::uint32_t regionIndex) {
        require(regionIndex < knee.regions.size(),
                "live Open Knee contact region is unavailable");
        const std::string& name = knee.regions[regionIndex].name;
        if (name == "FMC") return bodyIndices[0u];
        if (name == "TBC-L" || name == "TBC-M" || name == "MNS-L" ||
            name == "MNS-M") return bodyIndices[1u];
        if (name == "PTC") return bodyIndices[2u];
        throw std::runtime_error(
            "live Open Knee articular region has no rigid owner: " + name);
    };
    const auto worldPointToLocal = [&](const Point& point,
                                       const std::uint32_t bodyIndex) {
        require(bodyIndex < restBodies.size(),
                "live Open Knee articular owner body is unavailable");
        const MRBodyStateGPU& body = restBodies[bodyIndex];
        const mr_float4 inverse{
            -body.orientation.x, -body.orientation.y,
            -body.orientation.z, body.orientation.w};
        return rotatePoint(inverse, femSubtract(
            {static_cast<float>(point[0u]), static_cast<float>(point[1u]),
             static_cast<float>(point[2u]), 1.0f},
            body.position));
    };
    const auto worldVectorToLocal = [&](const Point& vector,
                                        const std::uint32_t bodyIndex) {
        const MRBodyStateGPU& body = restBodies[bodyIndex];
        const mr_float4 inverse{
            -body.orientation.x, -body.orientation.y,
            -body.orientation.z, body.orientation.w};
        return rotatePoint(inverse, {
            static_cast<float>(vector[0u]), static_cast<float>(vector[1u]),
            static_cast<float>(vector[2u]), 0.0f});
    };
    LiveOpenKneeArticularContactCook result;
    result.pairCount = static_cast<std::uint32_t>(contactModel.pairs.size());
    result.samples.reserve(contactModel.samples.size());
    std::uint32_t nextSample = 0u;
    for (const auto& pair : contactModel.pairs) {
        require(pair.firstSample == nextSample &&
                    pair.sampleCount <=
                        contactModel.samples.size() - pair.firstSample,
                "live Open Knee contact pair sample coverage drifted");
        const std::uint32_t slaveBody = ownerBody(pair.slaveRegionIndex);
        const std::uint32_t masterBody = ownerBody(pair.masterRegionIndex);
        const double maximumLayerNormalStrainPerPressure = std::max(
            normalStrainPerPressure(materialForRegion(pair.slaveRegionIndex)),
            normalStrainPerPressure(materialForRegion(pair.masterRegionIndex)));
        for (std::uint32_t local = 0u; local < pair.sampleCount; ++local) {
            const auto& source =
                contactModel.samples[pair.firstSample + local];
            require(source.slaveNode < referenceNodes.size(),
                    "live Open Knee contact slave node is unavailable");
            Point masterPoint{};
            for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
                require(source.masterNodes[corner] < referenceNodes.size(),
                        "live Open Knee contact master node is unavailable");
                for (std::uint32_t axis = 0u; axis < 3u; ++axis) {
                    masterPoint[axis] += source.masterBarycentric[corner] *
                        referenceNodes[source.masterNodes[corner]][axis];
                }
            }
            const mr_float4 slaveLocal = worldPointToLocal(
                referenceNodes[source.slaveNode], slaveBody);
            const mr_float4 masterLocal = worldPointToLocal(
                masterPoint, masterBody);
            const mr_float4 normalLocal = worldVectorToLocal(
                source.referenceNormal, masterBody);
            NMNumiHumanArticularContactSampleGPU cooked{};
            cooked.slaveBodyIndex = slaveBody;
            cooked.masterBodyIndex = masterBody;
            cooked.flags = slaveBody == masterBody
                ? NM_NUMI_HUMAN_ARTICULAR_CONTACT_INTERNAL_SAME_BODY
                : NM_NUMI_HUMAN_ARTICULAR_CONTACT_ACTIVE;
            cooked.slaveLocalPointAndArea = {
                slaveLocal.x, slaveLocal.y, slaveLocal.z,
                static_cast<float>(source.tributaryAreaSquareMeters)};
            cooked.masterLocalPointAndReferenceSeparation = {
                masterLocal.x, masterLocal.y, masterLocal.z,
                static_cast<float>(source.referenceSeparationMeters)};
            cooked.masterLocalNormalAndStiffness = {
                normalLocal.x, normalLocal.y, normalLocal.z,
                static_cast<float>(
                    pair.effectiveFoundationStiffnessPascalsPerMeter)};
            cooked.normalStrainPerPressure = {
                static_cast<float>(maximumLayerNormalStrainPerPressure),
                0.0f, 0.0f, 0.0f};
            result.samples.push_back(cooked);
            if (slaveBody == masterBody) {
                ++result.internalSameBodySampleCount;
            } else {
                ++result.mechanicalSampleCount;
            }
        }
        nextSample += pair.sampleCount;
    }
    require(nextSample == contactModel.samples.size() &&
                result.samples.size() == 69701u &&
                result.mechanicalSampleCount > 0u &&
                result.internalSameBodySampleCount > 0u &&
                result.mechanicalSampleCount +
                    result.internalSameBodySampleCount == result.samples.size(),
            "live Open Knee articular contact ownership is incomplete");
    return result;
}

LoadedOpenKneeLigamentFEM runLiveOpenKneeTissueFEM(
    const metalrobo::NumiHumanKneePayload& knee,
    const LoadedMuscles& muscles,
    MuscleDrivenVisualState& driven,
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts& supportContacts,
    const LoadedJointEqualities& jointEqualities,
    const double timestepSeconds,
    const std::uint32_t stepCount,
    const double activation,
    const std::span<const std::uint32_t> selectedSourceMuscleIndices,
    const bool applySelectedActivationIncrement,
    const bool enableRootAssistance,
    const bool removeRootAssistance,
    const std::filesystem::path& matterMetallib
) {
    require(stepCount >= 1u && stepCount <= MR_NUMI_HUMAN_STAND_MAX_STEPS,
            "live Open Knee tissues require a valid Human horizon");
    GroundAlignedSupport aligned =
        makeGroundAlignedSupport(model, supportContacts);
    const std::uint32_t kneeQIndex =
        knee.side == metalrobo::NumiHumanKneeSide::left ? 120u : 106u;
    constexpr double qualificationFlexionRadians = 3.0e-6;
    require(kneeQIndex < aligned.q.size(),
            "live Open Knee flexion coordinate is unavailable");
    aligned.q[kneeQIndex] = qualificationFlexionRadians;
    double maximumEqualityProjection = 0.0;
    const auto equalityProjection = metalrobo::projectNumiHumanJointEqualities(
        jointEqualities.payload.records, aligned.q,
        &maximumEqualityProjection);
    require(equalityProjection.succeeded() &&
                std::isfinite(maximumEqualityProjection),
            "live Open Knee flexion equality projection failed");
    CompiledStandActivation tissueActivation;
    if (applySelectedActivationIncrement) {
        tissueActivation = compileStaticStandActivation(
            model, muscles, jointEqualities, aligned.q, 1.0, {});
        for (const std::uint32_t muscleIndex : selectedSourceMuscleIndices) {
            tissueActivation.activation[muscleIndex] = std::min(
                1.0f, tissueActivation.activation[muscleIndex] +
                    static_cast<float>(activation));
        }
    } else {
        tissueActivation = compileStaticStandActivation(
            model, muscles, jointEqualities, aligned.q, activation,
            selectedSourceMuscleIndices);
    }
    tissueActivation.q = aligned.q;
    const auto poseBodies = [&](const std::span<const float> q,
                                const std::string_view phase) {
        metalrobo::MetalArticulatedOperatorResult poseResult;
        metalrobo::MetalArticulatedOperatorConfig poseConfig;
        poseConfig.pointJacobiansOnly = true;
        const auto poseDiagnostics = metalrobo::runMetalArticulatedOperator(
            model, {
                .articulationIndex = 0u,
                .environmentCount = 1u,
                .pointCount = 0u,
                .q = q,
                .points = {},
            }, poseResult, poseConfig);
        require(poseDiagnostics.succeeded() && poseDiagnostics.dispatched &&
                    poseDiagnostics.published &&
                    poseDiagnostics.successfulEnvironmentCount == 1u,
                "live Open Knee " + std::string(phase) +
                    " Human pose failed: " + poseDiagnostics.message);
        return visualBodyStates(model, poseResult.bodyPoses);
    };
    const std::vector<float> supportQ =
        packMetalConfiguration(tissueActivation.q);
    const std::vector<MRBodyStateGPU> referenceBodies =
        poseBodies(supportQ, "support-reference");
    std::vector<double> projectedRestQ(
        model.defaultQ.begin(), model.defaultQ.end());
    double maximumRestProjection = 0.0;
    const auto restProjection = metalrobo::projectNumiHumanJointEqualities(
        jointEqualities.payload.records, projectedRestQ,
        &maximumRestProjection);
    require(restProjection.succeeded() &&
                std::isfinite(maximumRestProjection),
            "live Open Knee neutral equality projection failed");
    const std::vector<MRBodyStateGPU> restBodies =
        poseBodies(packMetalConfiguration(projectedRestQ), "projected-rest");
    require(referenceBodies.size() == restBodies.size(),
            "live Open Knee Human body count drifted");
    const auto bodyIndices = openKneeBodyIndices(knee);
    require(bodyIndices[2u] < referenceBodies.size(),
            "live Open Knee bodies escape the Human articulation");
    const LiveOpenKneeArticularContactCook articularContact =
        cookLiveOpenKneeArticularContact(knee, restBodies, bodyIndices);
    std::cout
        << "open_knee_articular_cook=accepted"
        << " pairs=" << articularContact.pairCount
        << " samples=" << articularContact.samples.size()
        << " mechanical_samples="
        << articularContact.mechanicalSampleCount
        << " internal_same_body_samples="
        << articularContact.internalSameBodySampleCount
        << "\n" << std::flush;
    const MRBodyStateGPU& restFemur = restBodies[bodyIndices[0u]];
    const MRBodyStateGPU& referenceFemur = referenceBodies[bodyIndices[0u]];
    const mr_float4 inverseRestFemurOrientation{
        -restFemur.orientation.x, -restFemur.orientation.y,
        -restFemur.orientation.z, restFemur.orientation.w};
    const auto moveWithFemur = [&](const std::array<float, 3u>& point) {
        const mr_float4 sourceWorld{point[0u], point[1u], point[2u], 1.0f};
        const mr_float4 local = rotatePoint(
            inverseRestFemurOrientation,
            femSubtract(sourceWorld, restFemur.position));
        mr_float4 moved = femAdd(
            referenceFemur.position,
            rotatePoint(referenceFemur.orientation, local));
        moved.w = 1.0f;
        return moved;
    };

    std::vector<LiveOpenKneeRegion> regions;
    std::uint32_t totalNodes = 0u;
    std::uint32_t totalTetrahedra = 0u;
    for (std::uint32_t regionIndex = 0u;
         regionIndex < knee.regions.size(); ++regionIndex) {
        const auto& region = knee.regions[regionIndex];
        const auto specification = std::find_if(
            kLiveOpenKneeTissueSpecs.begin(),
            kLiveOpenKneeTissueSpecs.end(),
            [&region](const LiveOpenKneeTissueSpec& candidate) {
                return candidate.name == region.name;
            });
        if (specification == kLiveOpenKneeTissueSpecs.end()) continue;
        const bool tendon = region.name == "PTL" || region.name == "QAT";
        const bool exactKind =
            (tendon &&
             region.kind == metalrobo::NumiHumanKneeRegionKind::tendon) ||
            (!tendon &&
             region.kind == metalrobo::NumiHumanKneeRegionKind::ligament);
        require(exactKind, "live Open Knee tissue kind drifted");
        regions.push_back({
            .specification = &*specification,
            .payloadRegion = regionIndex,
            .firstFEMNode = totalNodes,
            .nodeCount = region.nodeCount,
            .tetrahedronCount = region.tetrahedronCount,
        });
        totalNodes += region.nodeCount;
        totalTetrahedra += region.tetrahedronCount;
    }
    require(regions.size() == kLiveOpenKneeTissueSpecs.size() &&
                totalNodes == 62402u && totalTetrahedra == 264442u,
            "live Open Knee exact tissue topology drifted");

    auto parsed = numi::matter::parseMatterFile(
        NUMI_HUMAN_OPEN_KNEE_LIGAMENT_MATERIAL);
    require(parsed.succeeded(),
            "live Open Knee Matter material did not parse");
    numi::matter::WorldSource worldSource;
    worldSource.environmentCount = 1u;
    worldSource.frameTimestep = timestepSeconds;
    worldSource.gravity = {0.0, 0.0, 0.0};
    worldSource.mixedSolver.newtonIterations = 4u;
    worldSource.mixedSolver.fgmresRestart = 16u;
    worldSource.mixedSolver.fgmresIterations = 32u;
    worldSource.mixedSolver.lineSearchSteps = 6u;
    std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads(totalNodes);
    std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors(totalNodes);
    for (auto& load : nodeLoads)
        std::fill_n(load.endpointIndex, 4u, NM_INVALID_INDEX);
    for (auto& anchor : nodeAnchors) anchor.bodyIndex = NM_INVALID_INDEX;
    std::vector<NMNumiHumanTendonFEMEndpointReplacementGPU>
        endpointReplacements;
    std::vector<mr_float4> initialWorldNodes(totalNodes);

    for (LiveOpenKneeRegion& runtimeRegion : regions) {
        const auto& region = knee.regions[runtimeRegion.payloadRegion];
        numi::matter::MaterialProgram material = parsed.material;
        material.name = "open_knee_" + region.name +
            "_live_human_isotropic_matrix";
        material.fingerprint = 0u;
        setLiveOpenKneeMaterialParameter(material, "density", 1000.0);
        setLiveOpenKneeMaterialParameter(
            material, "shear", 2.0e6 * runtimeRegion.specification->c1MPa);
        setLiveOpenKneeMaterialParameter(
            material, "bulk", 1.0e6 * runtimeRegion.specification->bulkMPa);
        setLiveOpenKneeMaterialParameter(
            material, "numerical_viscosity", 25.0);
        const std::uint32_t materialIndex =
            static_cast<std::uint32_t>(worldSource.materials.size());
        worldSource.materials.push_back(std::move(material));
        numi::matter::ObjectSource object;
        object.name = "numi_human_live_open_knee_" + region.name;
        object.materialIndex = materialIndex;
        object.representation = numi::matter::Representation::fem;
        object.mixedFEM = false;
        object.deformableSelfContact = false;
        object.characteristicLength = 0.001;
        object.femNodes.reserve(region.nodeCount);
        object.tetrahedra.reserve(region.tetrahedronCount);
        for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
            const auto& sourceNode = knee.nodes[region.firstNode + local];
            const mr_float4 posed = moveWithFemur(sourceNode.restWorld);
            const std::uint32_t femNode = runtimeRegion.firstFEMNode + local;
            initialWorldNodes[femNode] = posed;
            object.femNodes.push_back({posed.x, posed.y, posed.z});
            if (!sourceNode.rigidlyAttached) continue;
            const auto body = std::find(
                bodyIndices.begin(), bodyIndices.end(),
                sourceNode.anchorBodyIndex);
            require(body != bodyIndices.end(),
                    "live Open Knee attachment is not femur, tibia, or patella owned");
            const std::uint32_t bodySlot = static_cast<std::uint32_t>(
                std::distance(bodyIndices.begin(), body));
            auto& anchor = nodeAnchors[femNode];
            anchor.bodyIndex = sourceNode.anchorBodyIndex;
            anchor.flags = NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
            const MRBodyStateGPU& restAnchorBody =
                restBodies[sourceNode.anchorBodyIndex];
            const mr_float4 inverseRestAnchorOrientation{
                -restAnchorBody.orientation.x,
                -restAnchorBody.orientation.y,
                -restAnchorBody.orientation.z,
                restAnchorBody.orientation.w};
            const mr_float4 sourceWorld{
                sourceNode.restWorld[0u], sourceNode.restWorld[1u],
                sourceNode.restWorld[2u], 1.0f};
            const mr_float4 resolvedLocal = rotatePoint(
                inverseRestAnchorOrientation,
                femSubtract(sourceWorld, restAnchorBody.position));
            anchor.localPoint = {
                resolvedLocal.x, resolvedLocal.y,
                resolvedLocal.z, 0.0f};
            object.femFixedNodes.push_back(local);
            ++runtimeRegion.anchorCounts[bodySlot];
        }
        for (std::uint32_t local = 0u;
             local < region.tetrahedronCount; ++local) {
            const auto& source =
                knee.tetrahedra[region.firstTetrahedron + local];
            std::array<std::uint32_t, 4u> nodes{};
            for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
                require(source[corner] >= region.firstNode &&
                            source[corner] < region.firstNode + region.nodeCount,
                        "live Open Knee tetrahedron crosses tissue ownership");
                nodes[corner] = source[corner] - region.firstNode;
            }
            object.tetrahedra.push_back({nodes});
        }
        const bool exactBoundaryOwnership = region.name == "PTL"
            ? runtimeRegion.anchorCounts[1u] > 0u &&
                runtimeRegion.anchorCounts[2u] > 0u &&
                runtimeRegion.anchorCounts[0u] == 0u
            : region.name == "QAT"
            ? runtimeRegion.anchorCounts[2u] > 0u &&
                runtimeRegion.anchorCounts[0u] == 0u &&
                runtimeRegion.anchorCounts[1u] == 0u
            : runtimeRegion.anchorCounts[0u] > 0u &&
                runtimeRegion.anchorCounts[1u] > 0u &&
                runtimeRegion.anchorCounts[2u] == 0u;
        require(exactBoundaryOwnership && object.femFixedNodes.size() ==
                    runtimeRegion.anchorCounts[0u] +
                    runtimeRegion.anchorCounts[1u] +
                    runtimeRegion.anchorCounts[2u],
                "live Open Knee tissue attachment ownership drifted");
        worldSource.objects.push_back(std::move(object));
    }

    const auto qatRuntime = std::find_if(
        regions.begin(), regions.end(),
        [&knee](const LiveOpenKneeRegion& candidate) {
            return knee.regions[candidate.payloadRegion].name == "QAT";
        });
    require(qatRuntime != regions.end(),
            "live Open Knee quadriceps tendon is unavailable");
    const auto& qatRegion = knee.regions[qatRuntime->payloadRegion];
    const std::uint32_t patellaBodyIndex = bodyIndices[2u];
    const std::uint32_t tibiaBodyIndex = bodyIndices[1u];
    constexpr std::uint32_t pelvisBodyIndex = 128u;
    std::array<std::uint32_t, 4u> quadricepsLoadEndpoints{};
    std::array<std::uint32_t, 4u> quadricepsAnchorEndpoints{};
    require(selectedSourceMuscleIndices.size() == 4u,
            "live Open Knee requires the four selected source quadriceps routes");
    for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
        const std::uint32_t muscleIndex = selectedSourceMuscleIndices[slot];
        std::array<std::uint32_t, 2u> endpoints{
            MR_INVALID_INDEX, MR_INVALID_INDEX};
        std::uint32_t endpointCount = 0u;
        for (std::uint32_t bindingIndex = 0u;
             bindingIndex < muscles.tendonPayload.bindings.size();
             ++bindingIndex) {
            if (muscles.tendonPayload.bindings[bindingIndex].muscleIndex !=
                muscleIndex) continue;
            require(endpointCount < endpoints.size(),
                    "live Open Knee source muscle has more than two terminals");
            endpoints[endpointCount++] = bindingIndex;
        }
        require(endpointCount == 2u,
                "live Open Knee source muscle terminal coverage drifted");
        const bool firstTibia =
            muscles.tendonPayload.bindings[endpoints[0u]].bodyIndex ==
                tibiaBodyIndex;
        const bool secondTibia =
            muscles.tendonPayload.bindings[endpoints[1u]].bodyIndex ==
                tibiaBodyIndex;
        require(firstTibia != secondTibia,
                "live Open Knee selected quadriceps route has no unique distal tibia terminal");
        quadricepsAnchorEndpoints[slot] = firstTibia
            ? endpoints[0u] : endpoints[1u];
        quadricepsLoadEndpoints[slot] = firstTibia
            ? endpoints[1u] : endpoints[0u];
        const std::uint32_t proximalBody =
            muscles.tendonPayload.bindings[
                quadricepsLoadEndpoints[slot]].bodyIndex;
        require(proximalBody == bodyIndices[0u] ||
                    proximalBody == pelvisBodyIndex,
                "live Open Knee selected quadriceps route does not originate on femur or pelvis");
        std::uint32_t internalPatellaRouteNodes = 0u;
        for (const auto& routeNode :
             muscles.referenceMuscles[muscleIndex].route) {
            const std::uint32_t routeBody =
                routeNode.type == metalrobo::MujocoRouteNodeType::site
                ? muscles.referenceSites[routeNode.targetIndex].bodyIndex
                : muscles.referenceWraps[routeNode.targetIndex].bodyIndex;
            internalPatellaRouteNodes += routeBody == patellaBodyIndex ? 1u : 0u;
        }
        require(internalPatellaRouteNodes > 0u,
                "live Open Knee selected quadriceps route has no internal patella path to replace");
    }

    std::uint32_t qatPatellaSamples = 0u;
    for (std::uint32_t local = 0u; local < qatRegion.nodeCount; ++local) {
        const std::uint32_t femNode = qatRuntime->firstFEMNode + local;
        if (nodeAnchors[femNode].bodyIndex == patellaBodyIndex) {
            ++qatPatellaSamples;
        }
    }
    require(qatPatellaSamples == qatRuntime->anchorCounts[2u] &&
                qatPatellaSamples > 0u,
            "live Open Knee QAT patellar insertion drifted");
    const auto allFaces = std::find_if(
        knee.surfaces.begin() + qatRegion.firstSurface,
        knee.surfaces.begin() + qatRegion.firstSurface + qatRegion.surfaceCount,
        [](const metalrobo::NumiHumanKneeSurface& surface) {
            return surface.isAllFaces;
        });
    require(allFaces !=
                knee.surfaces.begin() + qatRegion.firstSurface +
                    qatRegion.surfaceCount,
            "live Open Knee QAT all-faces surface is unavailable");
    // Active QAT mechanics is a tensile force-transfer law. Distribute each
    // source quadriceps resultant over the exact patellar enthesis instead of
    // applying an impulsive volume traction at the free QSO cut surface. The
    // latter makes an unrelated passive ligament own global Newton acceptance
    // and is not the intended reduced tendon mechanics.
    std::vector<double> qatNodalAreas(qatRegion.nodeCount, 0.0);
    double qatLoadPatchAreaSquareMeters = 0.0;
    for (std::uint32_t localFace = 0u;
         localFace < allFaces->faceCount; ++localFace) {
        const auto& face = knee.faces[allFaces->firstFace + localFace];
        std::array<std::uint32_t, 3u> localNodes{};
        std::array<mr_float4, 3u> points{};
        for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
            require(face[corner] >= qatRegion.firstNode &&
                        face[corner] <
                            qatRegion.firstNode + qatRegion.nodeCount,
                    "live Open Knee QAT surface crosses region ownership");
            localNodes[corner] = face[corner] - qatRegion.firstNode;
            points[corner] = initialWorldNodes[
                qatRuntime->firstFEMNode + localNodes[corner]];
        }
        const mr_float4 rawNormal = femCross(
            femSubtract(points[1u], points[0u]),
            femSubtract(points[2u], points[0u]));
        const float twiceArea = femLength(rawNormal);
        require(std::isfinite(twiceArea) && twiceArea > 1.0e-12f,
                "live Open Knee QAT surface face is degenerate");
        const bool patellarEnthesisFace = std::all_of(
            localNodes.begin(), localNodes.end(),
            [&](const std::uint32_t local) {
                return nodeAnchors[
                    qatRuntime->firstFEMNode + local].bodyIndex ==
                    patellaBodyIndex;
            });
        if (!patellarEnthesisFace) continue;
        const double area = 0.5 * static_cast<double>(twiceArea);
        qatLoadPatchAreaSquareMeters += area;
        for (const std::uint32_t node : localNodes)
            qatNodalAreas[node] += area / 3.0;
    }
    const std::uint32_t qatLoadNodeCount = static_cast<std::uint32_t>(
        std::count_if(
            qatNodalAreas.begin(), qatNodalAreas.end(),
            [](const double area) { return area > 0.0; }));
    require(qatLoadNodeCount == qatPatellaSamples &&
                qatLoadPatchAreaSquareMeters > 0.0 &&
                std::isfinite(qatLoadPatchAreaSquareMeters),
            "live Open Knee QAT patellar enthesis surface drifted");
    constexpr float quadricepsForceOwnerFraction = 1.0f;
    for (std::uint32_t local = 0u; local < qatRegion.nodeCount; ++local) {
        if (qatNodalAreas[local] == 0.0) continue;
        const std::uint32_t femNode = qatRuntime->firstFEMNode + local;
        require(nodeAnchors[femNode].bodyIndex == patellaBodyIndex,
                "live Open Knee QAT force transfer left its patellar enthesis");
        const float weight = quadricepsForceOwnerFraction *
            static_cast<float>(
                qatNodalAreas[local] / qatLoadPatchAreaSquareMeters);
        for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
            nodeLoads[femNode].endpointIndex[slot] =
                quadricepsLoadEndpoints[slot];
            (&nodeLoads[femNode].scale.x)[slot] = weight;
        }
    }
    const auto ptlRuntime = std::find_if(
        regions.begin(), regions.end(),
        [&knee](const LiveOpenKneeRegion& candidate) {
            return knee.regions[candidate.payloadRegion].name == "PTL";
        });
    require(ptlRuntime != regions.end(),
            "live Open Knee patellar tendon is unavailable");
    const auto& ptlRegion = knee.regions[ptlRuntime->payloadRegion];
    const auto ptlAllFaces = std::find_if(
        knee.surfaces.begin() + ptlRegion.firstSurface,
        knee.surfaces.begin() + ptlRegion.firstSurface + ptlRegion.surfaceCount,
        [](const metalrobo::NumiHumanKneeSurface& surface) {
            return surface.isAllFaces;
        });
    require(ptlAllFaces !=
                knee.surfaces.begin() + ptlRegion.firstSurface +
                    ptlRegion.surfaceCount,
            "live Open Knee PTL all-faces surface is unavailable");
    std::vector<double> ptlPatellaNodalAreas(ptlRegion.nodeCount, 0.0);
    std::vector<double> ptlTibiaNodalAreas(ptlRegion.nodeCount, 0.0);
    double ptlPatellaPatchAreaSquareMeters = 0.0;
    double ptlTibiaPatchAreaSquareMeters = 0.0;
    for (std::uint32_t localFace = 0u;
         localFace < ptlAllFaces->faceCount; ++localFace) {
        const auto& face = knee.faces[ptlAllFaces->firstFace + localFace];
        std::array<std::uint32_t, 3u> localNodes{};
        std::array<mr_float4, 3u> points{};
        for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
            require(face[corner] >= ptlRegion.firstNode &&
                        face[corner] <
                            ptlRegion.firstNode + ptlRegion.nodeCount,
                    "live Open Knee PTL surface crosses region ownership");
            localNodes[corner] = face[corner] - ptlRegion.firstNode;
            points[corner] = initialWorldNodes[
                ptlRuntime->firstFEMNode + localNodes[corner]];
        }
        const mr_float4 rawNormal = femCross(
            femSubtract(points[1u], points[0u]),
            femSubtract(points[2u], points[0u]));
        const float twiceArea = femLength(rawNormal);
        require(std::isfinite(twiceArea) && twiceArea > 1.0e-12f,
                "live Open Knee PTL surface face is degenerate");
        const auto completeAttachmentFace = [&](const std::uint32_t body) {
            return std::all_of(
                localNodes.begin(), localNodes.end(),
                [&](const std::uint32_t local) {
                    return nodeAnchors[
                        ptlRuntime->firstFEMNode + local].bodyIndex == body;
                });
        };
        const double area = 0.5 * static_cast<double>(twiceArea);
        if (completeAttachmentFace(patellaBodyIndex)) {
            ptlPatellaPatchAreaSquareMeters += area;
            for (const std::uint32_t node : localNodes)
                ptlPatellaNodalAreas[node] += area / 3.0;
        } else if (completeAttachmentFace(tibiaBodyIndex)) {
            ptlTibiaPatchAreaSquareMeters += area;
            for (const std::uint32_t node : localNodes)
                ptlTibiaNodalAreas[node] += area / 3.0;
        }
    }
    const auto positiveAreaNodeCount = [](const std::vector<double>& areas) {
        return static_cast<std::uint32_t>(std::count_if(
            areas.begin(), areas.end(),
            [](const double area) { return area > 0.0; }));
    };
    const std::uint32_t ptlPatellaLoadNodeCount =
        positiveAreaNodeCount(ptlPatellaNodalAreas);
    const std::uint32_t ptlTibiaLoadNodeCount =
        positiveAreaNodeCount(ptlTibiaNodalAreas);
    require(ptlPatellaLoadNodeCount == ptlRuntime->anchorCounts[2u] &&
                ptlTibiaLoadNodeCount == ptlRuntime->anchorCounts[1u] &&
                std::isfinite(ptlPatellaPatchAreaSquareMeters) &&
                ptlPatellaPatchAreaSquareMeters > 0.0 &&
                std::isfinite(ptlTibiaPatchAreaSquareMeters) &&
                ptlTibiaPatchAreaSquareMeters > 0.0,
            "live Open Knee PTL enthesis surfaces drifted");
    for (std::uint32_t local = 0u; local < ptlRegion.nodeCount; ++local) {
        const bool patellaAttachment = ptlPatellaNodalAreas[local] > 0.0;
        const bool tibiaAttachment = ptlTibiaNodalAreas[local] > 0.0;
        if (!patellaAttachment && !tibiaAttachment) continue;
        require(patellaAttachment != tibiaAttachment,
                "live Open Knee PTL attachment node has ambiguous ownership");
        const std::uint32_t femNode = ptlRuntime->firstFEMNode + local;
        const std::uint32_t expectedBody =
            patellaAttachment ? patellaBodyIndex : tibiaBodyIndex;
        require(nodeAnchors[femNode].bodyIndex == expectedBody,
                "live Open Knee PTL force couple left its exact enthesis");
        const double patchArea = patellaAttachment
            ? ptlPatellaPatchAreaSquareMeters
            : ptlTibiaPatchAreaSquareMeters;
        const double nodalArea = patellaAttachment
            ? ptlPatellaNodalAreas[local]
            : ptlTibiaNodalAreas[local];
        const float signedWeight = quadricepsForceOwnerFraction *
            static_cast<float>(nodalArea / patchArea) *
            (patellaAttachment ? 1.0f : -1.0f);
        for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
            nodeLoads[femNode].endpointIndex[slot] =
                quadricepsAnchorEndpoints[slot];
            (&nodeLoads[femNode].scale.x)[slot] = signedWeight;
        }
    }
    endpointReplacements.reserve(4u);
    for (std::uint32_t slot = 0u; slot < 4u; ++slot) {
        NMNumiHumanTendonFEMEndpointReplacementGPU replacement{};
        replacement.loadEndpointIndex = quadricepsLoadEndpoints[slot];
        replacement.anchorEndpointIndex = quadricepsAnchorEndpoints[slot];
        replacement.flags =
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE |
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_FULL_MUSCLE_ROW |
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_DISTAL_FORCE_COUPLE;
        replacement.forceOwnerFraction.x = quadricepsForceOwnerFraction;
        endpointReplacements.push_back(replacement);
    }

    std::array<double, 3u> maximumAnchorTargetResidualMeters{};
    for (std::uint32_t node = 0u; node < totalNodes; ++node) {
        const auto& anchor = nodeAnchors[node];
        if ((anchor.flags &
             NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) == 0u) continue;
        const auto body = std::find(
            bodyIndices.begin(), bodyIndices.end(), anchor.bodyIndex);
        require(body != bodyIndices.end(),
                "live Open Knee diagnostic anchor body drifted");
        const std::uint32_t bodySlot = static_cast<std::uint32_t>(
            std::distance(bodyIndices.begin(), body));
        const MRBodyStateGPU& pose = referenceBodies[anchor.bodyIndex];
        const mr_float4 anchorLocal{
            anchor.localPoint.x, anchor.localPoint.y,
            anchor.localPoint.z, 0.0f};
        const mr_float4 target = femAdd(
            pose.position,
            rotatePoint(pose.orientation, anchorLocal));
        maximumAnchorTargetResidualMeters[bodySlot] = std::max(
            maximumAnchorTargetResidualMeters[bodySlot],
            static_cast<double>(femLength(
                femSubtract(target, initialWorldNodes[node]))));
    }
    require(maximumAnchorTargetResidualMeters[0u] <= 1.0e-7 &&
                maximumAnchorTargetResidualMeters[1u] > 5.0e-8 &&
                maximumAnchorTargetResidualMeters[1u] <= 1.0e-6 &&
                maximumAnchorTargetResidualMeters[2u] > 5.0e-8 &&
                maximumAnchorTargetResidualMeters[2u] <= 1.0e-6,
            "live Open Knee projected-flexion anchor targets are invalid");
    numi::matter::CompileOptions compileOptions;
    compileOptions.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(worldSource, compileOptions);
    std::string compileMessage;
    for (const auto& diagnostic : compiled.diagnostics)
        compileMessage += diagnostic.message + "; ";
    require(compiled.succeeded(),
            "live Open Knee FEM world did not compile: " + compileMessage);
    numi::matter::Runtime runtime;
    const auto initialized = runtime.initialize(compiled.world, {
        .metallib = matterMetallib,
        .environmentCount = 1u,
        .captureEvents = true,
        .captureDiagnostics = true,
        .automaticIdentification = false,
        .adaptiveTransfer = false,
    });
    require(initialized.encoded && runtime.valid(),
            "live Open Knee Matter runtime did not initialize: " +
                initialized.message);
    const auto initial = runtime.snapshot();
    require(initial.available && initial.femNodes.size() == totalNodes,
            "live Open Knee initial FEM snapshot is unavailable");
    double qatLoadPatchMassKilograms = 0.0;
    double minimumQATLoadNodeMassKilograms =
        std::numeric_limits<double>::infinity();
    std::uint32_t positiveMassQATLoadNodes = 0u;
    for (std::uint32_t local = 0u; local < qatRegion.nodeCount; ++local) {
        if (qatNodalAreas[local] == 0.0) continue;
        const double mass = initial.femNodes[
            qatRuntime->firstFEMNode + local].positionAndMass.w;
        if (std::isfinite(mass) && mass > 0.0) {
            qatLoadPatchMassKilograms += mass;
            minimumQATLoadNodeMassKilograms = std::min(
                minimumQATLoadNodeMassKilograms, mass);
            ++positiveMassQATLoadNodes;
        }
    }
    require(positiveMassQATLoadNodes == qatLoadNodeCount &&
                std::isfinite(qatLoadPatchMassKilograms) &&
                qatLoadPatchMassKilograms > 0.0 &&
                std::isfinite(minimumQATLoadNodeMassKilograms) &&
                minimumQATLoadNodeMassKilograms > 0.0,
            "live Open Knee QAT traction surface contains zero-mass FEM nodes");
    std::cout
        << "open_knee_qat_load_preflight=accepted"
        << " load_nodes=" << qatLoadNodeCount
        << " positive_mass_nodes=" << positiveMassQATLoadNodes
        << " patch_mass_kg=" << qatLoadPatchMassKilograms
        << " minimum_node_mass_kg=" << minimumQATLoadNodeMassKilograms
        << " ptl_patella_load_nodes=" << ptlPatellaLoadNodeCount
        << " ptl_tibia_load_nodes=" << ptlTibiaLoadNodeCount
        << " ptl_patella_patch_area_m2="
        << ptlPatellaPatchAreaSquareMeters
        << " ptl_tibia_patch_area_m2=" << ptlTibiaPatchAreaSquareMeters
        << "\n" << std::flush;
    numi::matter::NumiHumanTendonFEMLoadAdapter adapter;
    require(adapter.initialize(runtime, {
                .nodeLoads = nodeLoads,
                .nodeAnchors = nodeAnchors,
                .endpointReplacements = endpointReplacements,
                .articularContactSamples = articularContact.samples,
                .endpointCount = static_cast<std::uint32_t>(
                    muscles.tendonPayload.bindings.size()),
                .environmentCount = 1u,
                .productionForceOwnerFraction = quadricepsForceOwnerFraction,
            }, {.metallib = matterMetallib}),
            "live Open Knee active extensor-chain adapter did not initialize");
    id<MTLBuffer> reactionBuffer = (__bridge id<MTLBuffer>)
        runtime.femConstraintReactionBuffer();
    const NSUInteger reactionBytes = static_cast<NSUInteger>(
        totalNodes * sizeof(nm_float4));
    id<MTLBuffer> reactionSnapshot = reactionBuffer == nil
        ? nil
        : [reactionBuffer.device newBufferWithLength:reactionBytes
            options:MTLResourceStorageModeShared];
    require(reactionBuffer != nil && reactionSnapshot != nil,
            "live Open Knee pre-dynamics reaction snapshot is unavailable");
    FEMReactionSnapshotProgram reactionProgram{
        .delegate = adapter.program(),
        .source = reactionBuffer,
        .snapshot = reactionSnapshot,
        .byteCount = reactionBytes,
    };
    HumanTendonContinuumTransaction transaction{
        .program = femReactionSnapshotProgram(reactionProgram),
        .runtime = &runtime,
        .initial = initial,
    };
    driven = integratePersistentMetalHumanState(
        model, muscles, supportContacts, jointEqualities, timestepSeconds,
        stepCount, activation, selectedSourceMuscleIndices,
        applySelectedActivationIncrement, enableRootAssistance,
        removeRootAssistance, true, &transaction,
        std::pair<std::uint32_t, double>{
            kneeQIndex, qualificationFlexionRadians}, false);
    driven.tendonContinuumPassiveReactionOnly = false;
    const auto accepted = transaction.accepted;
    require(accepted.available && accepted.femNodes.size() == totalNodes &&
                accepted.statuses.size() == 1u,
            "live Open Knee accepted FEM snapshot is unavailable");
    const NMMatterStatusGPU& acceptedMatterStatus = accepted.statuses[0u];
    require(acceptedMatterStatus.code == NM_STATUS_SUCCESS,
            ("live Open Knee Matter solve failed: status=" +
             std::to_string(acceptedMatterStatus.code) + " object=" +
             std::to_string(acceptedMatterStatus.objectIndex) + " index=" +
             std::to_string(acceptedMatterStatus.failingIndex) +
             " microsteps=" +
             std::to_string(acceptedMatterStatus.completedMicrosteps) +
             " fgmres_iterations=" +
             std::to_string(acceptedMatterStatus.fgmresIterations) +
             " diagnostics=(" +
             std::to_string(acceptedMatterStatus.diagnostics.x) + "," +
             std::to_string(acceptedMatterStatus.diagnostics.y) + "," +
             std::to_string(acceptedMatterStatus.diagnostics.z) + "," +
             std::to_string(acceptedMatterStatus.diagnostics.w) + ")"));

    const auto* reactions =
        static_cast<const nm_float4*>(reactionSnapshot.contents);

    LoadedOpenKneeLigamentFEM result;
    result.header.magic = kOpenKneeLigamentFEMMagicV2;
    result.header.abi = 2u;
    result.header.side = knee.side == metalrobo::NumiHumanKneeSide::left
        ? 0u : 1u;
    result.header.regionCount = static_cast<std::uint32_t>(regions.size());
    result.header.nodeCount = totalNodes;
    result.header.tetrahedronCount = totalTetrahedra;
    result.header.poseKind = 2u;
    result.projectedRestBodies = restBodies;
    result.qualificationFlexionRadians = qualificationFlexionRadians;
    result.quadricepsEndpointCount = static_cast<std::uint32_t>(
        endpointReplacements.size());
    result.quadricepsLoadNodeCount = qatLoadNodeCount;
    result.quadricepsLoadPatchAreaSquareMeters =
        qatLoadPatchAreaSquareMeters;
    result.patellarTendonPatellaLoadNodeCount =
        ptlPatellaLoadNodeCount;
    result.patellarTendonTibiaLoadNodeCount = ptlTibiaLoadNodeCount;
    result.patellarTendonPatellaPatchAreaSquareMeters =
        ptlPatellaPatchAreaSquareMeters;
    result.patellarTendonTibiaPatchAreaSquareMeters =
        ptlTibiaPatchAreaSquareMeters;
    result.quadricepsForceOwnerFraction = quadricepsForceOwnerFraction;
    result.activeQuadricepsTendonCoupling = true;
    mr_float4 quadricepsAppliedForceResultant{
        0.0f, 0.0f, 0.0f, 0.0f};
    require(driven.finalTendonTransfers.size() ==
                muscles.tendonPayload.bindings.size(),
            "live Open Knee final tendon-transfer coverage is incomplete");
    for (const std::uint32_t endpoint : quadricepsLoadEndpoints) {
        const auto& transfer = driven.finalTendonTransfers[endpoint];
        require(transfer.status == MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS &&
                    transfer.bindingIndex == endpoint,
                "live Open Knee quadriceps transfer is unavailable");
        const mr_float4 continuumForce = femScale(
            {transfer.terminalWorldForce.x,
             transfer.terminalWorldForce.y,
             transfer.terminalWorldForce.z, 0.0f},
            -quadricepsForceOwnerFraction);
        const double magnitude = femLength(continuumForce);
        require(std::isfinite(magnitude) && magnitude > 0.0,
                "live Open Knee quadriceps transfer has zero force");
        result.quadricepsAppliedForceL1Newtons += magnitude;
        quadricepsAppliedForceResultant = femAdd(
            quadricepsAppliedForceResultant, continuumForce);
    }
    result.quadricepsAppliedForceResultantNewtons =
        femLength(quadricepsAppliedForceResultant);
    require(std::isfinite(result.quadricepsAppliedForceL1Newtons) &&
                result.quadricepsAppliedForceL1Newtons > 0.0 &&
                std::isfinite(
                    result.quadricepsAppliedForceResultantNewtons) &&
                result.quadricepsAppliedForceResultantNewtons > 0.0,
            "live Open Knee quadriceps continuum load is invalid");
    mr_float4 patellarTendonForceResultant{
        0.0f, 0.0f, 0.0f, 0.0f};
    for (const std::uint32_t endpoint : quadricepsAnchorEndpoints) {
        const auto& transfer = driven.finalTendonTransfers[endpoint];
        require(transfer.status == MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS &&
                    transfer.bindingIndex == endpoint,
                "live Open Knee patellar-tendon transfer is unavailable");
        const mr_float4 tibialForce{
            transfer.terminalWorldForce.x,
            transfer.terminalWorldForce.y,
            transfer.terminalWorldForce.z, 0.0f};
        const double magnitude = femLength(tibialForce);
        require(std::isfinite(magnitude) && magnitude > 0.0,
                "live Open Knee patellar-tendon transfer has zero force");
        result.patellarTendonForceL1Newtons += magnitude;
        patellarTendonForceResultant = femAdd(
            patellarTendonForceResultant, tibialForce);
    }
    result.patellarTendonForceResultantNewtons =
        femLength(patellarTendonForceResultant);
    require(std::isfinite(result.patellarTendonForceL1Newtons) &&
                result.patellarTendonForceL1Newtons > 0.0 &&
                std::isfinite(
                    result.patellarTendonForceResultantNewtons) &&
                result.patellarTendonForceResultantNewtons > 0.0,
            "live Open Knee patellar-tendon force couple is invalid");
    for (const auto& region : knee.regions) {
        const bool acceptedContinuum = std::any_of(
            kLiveOpenKneeTissueSpecs.begin(), kLiveOpenKneeTissueSpecs.end(),
            [&region](const LiveOpenKneeTissueSpec& candidate) {
                return candidate.name == region.name;
            });
        if (acceptedContinuum) continue;
        require(region.visualBodyIndex < restBodies.size(),
                "live Open Knee visual rest body is unavailable");
        const MRBodyStateGPU& restBody = restBodies[region.visualBodyIndex];
        const mr_float4 inverseRestOrientation{
            -restBody.orientation.x, -restBody.orientation.y,
            -restBody.orientation.z, restBody.orientation.w};
        for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
            const auto& node = knee.nodes[region.firstNode + local];
            const mr_float4 sourceWorld{
                node.restWorld[0u], node.restWorld[1u],
                node.restWorld[2u], 1.0f};
            const mr_float4 oldWorld = femAdd(
                restBody.position,
                rotatePoint(restBody.orientation, {
                    node.visualLocal[0u], node.visualLocal[1u],
                    node.visualLocal[2u], 0.0f}));
            result.maximumProjectedRestVisualCorrectionMeters = std::max(
                result.maximumProjectedRestVisualCorrectionMeters,
                static_cast<double>(femLength(femSubtract(sourceWorld, oldWorld))));
            const mr_float4 resolvedLocal = rotatePoint(
                inverseRestOrientation,
                femSubtract(sourceWorld, restBody.position));
            const mr_float4 reconstructed = femAdd(
                restBody.position,
                rotatePoint(restBody.orientation, resolvedLocal));
            result.maximumProjectedRestReconstructionResidualMeters = std::max(
                result.maximumProjectedRestReconstructionResidualMeters,
                static_cast<double>(femLength(
                    femSubtract(sourceWorld, reconstructed))));
        }
    }
    require(std::isfinite(result.maximumProjectedRestVisualCorrectionMeters) &&
                result.maximumProjectedRestReconstructionResidualMeters <= 1.0e-6,
            "live Open Knee projected-rest visual reconstruction failed");
    result.maximumAnchorTargetResidualMeters =
        maximumAnchorTargetResidualMeters;
    result.worldNodes.reserve(knee.nodes.size());
    result.deformedNodes.assign(knee.nodes.size(), false);
    for (const auto& node : knee.nodes) {
        result.worldNodes.push_back({
            node.restWorld[0u], node.restWorld[1u],
            node.restWorld[2u], 1.0f});
    }
    result.minimumDeterminant = std::numeric_limits<float>::infinity();
    result.maximumDeterminant = 0.0f;
    mr_float4 quadricepsEnthesisReaction{
        0.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 patellarTendonPatellaReaction{
        0.0f, 0.0f, 0.0f, 0.0f};
    mr_float4 patellarTendonTibiaReaction{
        0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<std::array<double, 3u>> initialPoints(totalNodes);
    std::vector<std::array<double, 3u>> acceptedPoints(totalNodes);
    for (std::uint32_t node = 0u; node < totalNodes; ++node) {
        initialPoints[node] = {
            initial.femNodes[node].positionAndMass.x,
            initial.femNodes[node].positionAndMass.y,
            initial.femNodes[node].positionAndMass.z};
        acceptedPoints[node] = {
            accepted.femNodes[node].positionAndMass.x,
            accepted.femNodes[node].positionAndMass.y,
            accepted.femNodes[node].positionAndMass.z};
    }
    for (const LiveOpenKneeRegion& runtimeRegion : regions) {
        const auto& region = knee.regions[runtimeRegion.payloadRegion];
        std::array<double, 3u> regionReactions{};
        for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
            const std::uint32_t femNode = runtimeRegion.firstFEMNode + local;
            const auto& acceptedNode = accepted.femNodes[femNode];
            const mr_float4 point{
                acceptedNode.positionAndMass.x,
                acceptedNode.positionAndMass.y,
                acceptedNode.positionAndMass.z, 1.0f};
            const std::uint32_t payloadNode = region.firstNode + local;
            result.worldNodes[payloadNode] = point;
            result.deformedNodes[payloadNode] = true;
            result.maximumDisplacementMeters = std::max(
                result.maximumDisplacementMeters,
                femLength(femSubtract(point, initialWorldNodes[femNode])));
            const auto& anchor = nodeAnchors[femNode];
            if ((anchor.flags &
                 NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) != 0u) {
                const auto body = std::find(
                    bodyIndices.begin(), bodyIndices.end(), anchor.bodyIndex);
                require(body != bodyIndices.end(),
                        "live Open Knee accepted anchor body drifted");
                const std::uint32_t bodySlot = static_cast<std::uint32_t>(
                    std::distance(bodyIndices.begin(), body));
                const float magnitude = femLength({
                    reactions[femNode].x, reactions[femNode].y,
                    reactions[femNode].z, 0.0f});
                require(std::isfinite(magnitude),
                        "live Open Knee anchor reaction is non-finite");
                regionReactions[bodySlot] += magnitude;
                result.bodyReactionL1Newtons[bodySlot] += magnitude;
                if (region.name == "QAT" &&
                    anchor.bodyIndex == patellaBodyIndex) {
                    quadricepsEnthesisReaction = femAdd(
                        quadricepsEnthesisReaction,
                        {reactions[femNode].x, reactions[femNode].y,
                         reactions[femNode].z, 0.0f});
                }
                if (region.name == "PTL" &&
                    anchor.bodyIndex == patellaBodyIndex) {
                    patellarTendonPatellaReaction = femAdd(
                        patellarTendonPatellaReaction,
                        {reactions[femNode].x, reactions[femNode].y,
                         reactions[femNode].z, 0.0f});
                } else if (region.name == "PTL" &&
                           anchor.bodyIndex == tibiaBodyIndex) {
                    patellarTendonTibiaReaction = femAdd(
                        patellarTendonTibiaReaction,
                        {reactions[femNode].x, reactions[femNode].y,
                         reactions[femNode].z, 0.0f});
                }
            }
        }
        const bool completeReaction = region.name == "PTL"
            ? regionReactions[1u] > 0.0 && regionReactions[2u] > 0.0
            : region.name == "QAT"
            ? regionReactions[2u] > 0.0 &&
                regionReactions[0u] == 0.0 && regionReactions[1u] == 0.0
            : regionReactions[0u] > 0.0 && regionReactions[1u] > 0.0;
        require(completeReaction,
                "live Open Knee tissue bone reaction is incomplete: " +
                    region.name + " femur=" +
                    std::to_string(regionReactions[0u]) + " tibia=" +
                    std::to_string(regionReactions[1u]) + " patella=" +
                    std::to_string(regionReactions[2u]));
        for (std::uint32_t local = 0u;
             local < region.tetrahedronCount; ++local) {
            const auto& source =
                knee.tetrahedra[region.firstTetrahedron + local];
            std::array<std::uint32_t, 4u> indices{};
            for (std::uint32_t corner = 0u; corner < 4u; ++corner)
                indices[corner] = runtimeRegion.firstFEMNode +
                    source[corner] - region.firstNode;
            const double restVolume = femSignedTetrahedronVolume(
                initialPoints, indices);
            const double currentVolume = femSignedTetrahedronVolume(
                acceptedPoints, indices);
            require(std::isfinite(restVolume) &&
                        std::abs(restVolume) > 1.0e-18,
                    "live Open Knee tetrahedron is degenerate");
            const float determinant = static_cast<float>(
                currentVolume / restVolume);
            require(std::isfinite(determinant),
                    "live Open Knee determinant is non-finite");
            result.minimumDeterminant = std::min(
                result.minimumDeterminant, determinant);
            result.maximumDeterminant = std::max(
                result.maximumDeterminant, determinant);
        }
    }
    result.quadricepsEnthesisReactionResultantNewtons =
        femLength(quadricepsEnthesisReaction);
    result.patellarTendonPatellaReactionResultantNewtons =
        femLength(patellarTendonPatellaReaction);
    result.patellarTendonTibiaReactionResultantNewtons =
        femLength(patellarTendonTibiaReaction);
    const auto adapterDiagnostics = adapter.diagnostics();
    const double externalResultantRelativeError = std::abs(
        adapterDiagnostics.assembledExternalForceResultantNewtons -
        result.quadricepsAppliedForceResultantNewtons) / std::max(
            1.0,
            result.quadricepsAppliedForceResultantNewtons);
    const double expectedExternalForceL1 =
        result.quadricepsAppliedForceResultantNewtons +
        2.0 * result.patellarTendonForceResultantNewtons;
    const double externalL1RelativeError = std::abs(
        adapterDiagnostics.assembledExternalForceL1Newtons -
        expectedExternalForceL1) /
        std::max(1.0, expectedExternalForceL1);
    const bool assembledForceVerified =
        std::isfinite(adapterDiagnostics.assembledExternalForceL1Newtons) &&
        adapterDiagnostics.assembledExternalForceL1Newtons > 0.0 &&
        std::isfinite(
            adapterDiagnostics.assembledExternalForceResultantNewtons) &&
        adapterDiagnostics.assembledExternalForceResultantNewtons > 0.0 &&
        std::isfinite(externalResultantRelativeError) &&
        externalResultantRelativeError <= 1.0e-5 &&
        std::isfinite(externalL1RelativeError) &&
        externalL1RelativeError <= 1.0e-5;
    const double enthesisReactionAbsoluteError = std::abs(
        result.quadricepsEnthesisReactionResultantNewtons -
        result.quadricepsAppliedForceResultantNewtons);
    const double enthesisReactionRelativeError =
        enthesisReactionAbsoluteError / std::max(
            1.0, result.quadricepsAppliedForceResultantNewtons);
    // The femur has no direct active QAT/PTL endpoint load. Its fixed-node L1
    // reaction therefore provides a measured, conservative scale for passive
    // tissue superposition on the attachment reactions. Do not require an
    // active resultant to equal an active-plus-passive reaction bit-for-bit.
    const double passiveReactionAllowanceNewtons = std::max(
        1.0e-4 * result.quadricepsAppliedForceResultantNewtons,
        result.bodyReactionL1Newtons[0u]);
    const bool enthesisForceTransferVerified =
        std::isfinite(
            result.quadricepsEnthesisReactionResultantNewtons) &&
        result.quadricepsEnthesisReactionResultantNewtons > 0.0 &&
        std::isfinite(enthesisReactionAbsoluteError) &&
        enthesisReactionAbsoluteError <= passiveReactionAllowanceNewtons;
    const double ptlPatellaReactionAbsoluteError = std::abs(
        result.patellarTendonPatellaReactionResultantNewtons -
        result.patellarTendonForceResultantNewtons);
    const double ptlPatellaReactionRelativeError =
        ptlPatellaReactionAbsoluteError / std::max(
            1.0, result.patellarTendonForceResultantNewtons);
    const double ptlTibiaReactionAbsoluteError = std::abs(
        result.patellarTendonTibiaReactionResultantNewtons -
        result.patellarTendonForceResultantNewtons);
    const double ptlTibiaReactionRelativeError =
        ptlTibiaReactionAbsoluteError / std::max(
            1.0, result.patellarTendonForceResultantNewtons);
    const bool patellarTendonForceTransferVerified =
        std::isfinite(ptlPatellaReactionAbsoluteError) &&
        ptlPatellaReactionAbsoluteError <= passiveReactionAllowanceNewtons &&
        std::isfinite(ptlTibiaReactionAbsoluteError) &&
        ptlTibiaReactionAbsoluteError <= passiveReactionAllowanceNewtons;
    result.assembledExternalForceL1Newtons =
        adapterDiagnostics.assembledExternalForceL1Newtons;
    result.assembledExternalForceResultantNewtons =
        adapterDiagnostics.assembledExternalForceResultantNewtons;
    result.articularPairCount = articularContact.pairCount;
    result.articularContactSampleCount =
        adapterDiagnostics.articularContactSampleCount;
    result.articularMechanicalSampleCount =
        adapterDiagnostics.articularMechanicalSampleCount;
    result.articularInternalSameBodySampleCount =
        adapterDiagnostics.articularInternalSameBodySampleCount;
    result.articularClosedSampleCount =
        adapterDiagnostics.articularClosedSampleCount;
    result.articularContactAreaSquareMeters =
        adapterDiagnostics.articularContactAreaSquareMeters;
    result.articularNormalForceNewtons =
        adapterDiagnostics.articularNormalForceNewtons;
    result.articularMaximumPressurePascals =
        adapterDiagnostics.articularMaximumPressurePascals;
    result.articularBodyForceL1Newtons =
        adapterDiagnostics.articularBodyForceL1Newtons;
    result.articularForceResidualNewtons =
        adapterDiagnostics.articularForceResidualNewtons;
    result.articularMomentResidualNewtonMeters =
        adapterDiagnostics.articularMomentResidualNewtonMeters;
    result.articularStoredEnergyJoules =
        adapterDiagnostics.articularStoredEnergyJoules;
    result.articularMaximumNormalStrain =
        adapterDiagnostics.articularMaximumNormalStrain;
    result.articularMaximumClosureMeters =
        adapterDiagnostics.articularMaximumClosureMeters;
    result.articularAuditedStepCount =
        adapterDiagnostics.articularAuditedStepCount;
    result.articularTrajectoryMinimumClosedSampleCount =
        adapterDiagnostics.articularTrajectoryMinimumClosedSampleCount;
    result.articularTrajectoryMaximumClosedSampleCount =
        adapterDiagnostics.articularTrajectoryMaximumClosedSampleCount;
    result.articularTrajectoryMinimumNormalForceNewtons =
        adapterDiagnostics.articularTrajectoryMinimumNormalForceNewtons;
    result.articularTrajectoryMaximumNormalForceNewtons =
        adapterDiagnostics.articularTrajectoryMaximumNormalForceNewtons;
    result.articularTrajectoryMaximumPressurePascals =
        adapterDiagnostics.articularTrajectoryMaximumPressurePascals;
    result.articularTrajectoryMaximumStoredEnergyJoules =
        adapterDiagnostics.articularTrajectoryMaximumStoredEnergyJoules;
    result.articularTrajectoryMaximumNormalStrain =
        adapterDiagnostics.articularTrajectoryMaximumNormalStrain;
    result.articularTrajectoryMaximumClosureMeters =
        adapterDiagnostics.articularTrajectoryMaximumClosureMeters;
    result.articularTrajectoryMaximumForceResidualNewtons =
        adapterDiagnostics.articularTrajectoryMaximumForceResidualNewtons;
    result.articularTrajectoryMaximumMomentResidualNewtonMeters =
        adapterDiagnostics.articularTrajectoryMaximumMomentResidualNewtonMeters;
    const double articularForceScale = std::max(
        1.0, adapterDiagnostics.articularBodyForceL1Newtons);
    const bool articularContactVerified =
        adapterDiagnostics.articularContactSampleCount == 69701u &&
        adapterDiagnostics.articularMechanicalSampleCount ==
            articularContact.mechanicalSampleCount &&
        adapterDiagnostics.articularInternalSameBodySampleCount ==
            articularContact.internalSameBodySampleCount &&
        adapterDiagnostics.articularMechanicalSampleCount +
            adapterDiagnostics.articularInternalSameBodySampleCount ==
                adapterDiagnostics.articularContactSampleCount &&
        adapterDiagnostics.articularAuditedStepCount == driven.stepCount &&
        adapterDiagnostics.articularTrajectoryMinimumClosedSampleCount > 0u &&
        adapterDiagnostics.articularTrajectoryMaximumClosedSampleCount >=
            adapterDiagnostics.articularTrajectoryMinimumClosedSampleCount &&
        adapterDiagnostics.articularClosedSampleCount > 0u &&
        std::isfinite(adapterDiagnostics.articularContactAreaSquareMeters) &&
        adapterDiagnostics.articularContactAreaSquareMeters > 0.0 &&
        std::isfinite(adapterDiagnostics.articularNormalForceNewtons) &&
        adapterDiagnostics.articularNormalForceNewtons > 0.0 &&
        std::isfinite(adapterDiagnostics.articularMaximumPressurePascals) &&
        adapterDiagnostics.articularMaximumPressurePascals > 0.0 &&
        std::isfinite(adapterDiagnostics.articularStoredEnergyJoules) &&
        adapterDiagnostics.articularStoredEnergyJoules > 0.0 &&
        std::isfinite(adapterDiagnostics.articularMaximumNormalStrain) &&
        adapterDiagnostics.articularMaximumNormalStrain > 0.0 &&
        std::isfinite(adapterDiagnostics.articularMaximumClosureMeters) &&
        adapterDiagnostics.articularMaximumClosureMeters > 0.0 &&
        std::isfinite(
            adapterDiagnostics.articularTrajectoryMinimumNormalForceNewtons) &&
        adapterDiagnostics.articularTrajectoryMinimumNormalForceNewtons > 0.0 &&
        std::isfinite(
            adapterDiagnostics.articularTrajectoryMaximumNormalForceNewtons) &&
        adapterDiagnostics.articularTrajectoryMaximumNormalForceNewtons >=
            adapterDiagnostics.articularTrajectoryMinimumNormalForceNewtons &&
        std::isfinite(
            adapterDiagnostics.articularTrajectoryMaximumPressurePascals) &&
        adapterDiagnostics.articularTrajectoryMaximumPressurePascals > 0.0 &&
        std::isfinite(
            adapterDiagnostics.articularTrajectoryMaximumStoredEnergyJoules) &&
        adapterDiagnostics.articularTrajectoryMaximumStoredEnergyJoules > 0.0 &&
        std::isfinite(
            adapterDiagnostics.articularTrajectoryMaximumNormalStrain) &&
        adapterDiagnostics.articularTrajectoryMaximumNormalStrain > 0.0 &&
        std::isfinite(
            adapterDiagnostics.articularTrajectoryMaximumClosureMeters) &&
        adapterDiagnostics.articularTrajectoryMaximumClosureMeters > 0.0 &&
        std::isfinite(adapterDiagnostics.articularBodyForceL1Newtons) &&
        adapterDiagnostics.articularBodyForceL1Newtons > 0.0 &&
        adapterDiagnostics.articularBodyForceL1Newtons <=
            2.0 * adapterDiagnostics.articularNormalForceNewtons +
                2.0e-5 * articularForceScale &&
        std::isfinite(adapterDiagnostics.articularForceResidualNewtons) &&
        adapterDiagnostics.articularForceResidualNewtons <=
            1.0e-5 * articularForceScale &&
        std::isfinite(
            adapterDiagnostics.articularMomentResidualNewtonMeters) &&
        adapterDiagnostics.articularMomentResidualNewtonMeters <=
            1.0e-5 * articularForceScale &&
        adapterDiagnostics.articularTrajectoryMaximumForceResidualNewtons <=
            1.0e-5 * articularForceScale &&
        adapterDiagnostics.articularTrajectoryMaximumMomentResidualNewtonMeters <=
            1.0e-5 * articularForceScale;
    const bool completeBodyReactions = std::all_of(
        result.bodyReactionL1Newtons.begin(),
        result.bodyReactionL1Newtons.end(),
        [](const double value) {
            return std::isfinite(value) && value > 0.0;
        });
    require(result.maximumDisplacementMeters > 0.0f &&
                result.maximumDisplacementMeters < 0.02f &&
                result.minimumDeterminant >= 0.25f &&
                result.maximumDeterminant <= 2.5f &&
                assembledForceVerified && enthesisForceTransferVerified &&
                patellarTendonForceTransferVerified &&
                articularContactVerified &&
                completeBodyReactions &&
                transaction.rollbackVerified && transaction.replayVerified,
            "live Open Knee accepted mechanics failed physical gates: displacement_m=" +
                std::to_string(result.maximumDisplacementMeters) +
                " minimum_J=" + std::to_string(result.minimumDeterminant) +
                " maximum_J=" + std::to_string(result.maximumDeterminant) +
                " femur_reaction_l1_n=" +
                std::to_string(result.bodyReactionL1Newtons[0u]) +
                " tibia_reaction_l1_n=" +
                std::to_string(result.bodyReactionL1Newtons[1u]) +
                " patella_reaction_l1_n=" +
                std::to_string(result.bodyReactionL1Newtons[2u]) +
                " quadriceps_force_l1_n=" +
                std::to_string(result.quadricepsAppliedForceL1Newtons) +
                " quadriceps_force_resultant_n=" +
                std::to_string(
                    result.quadricepsAppliedForceResultantNewtons) +
                " assembled_force_l1_n=" + std::to_string(
                    adapterDiagnostics.assembledExternalForceL1Newtons) +
                " assembled_force_resultant_n=" + std::to_string(
                    adapterDiagnostics.assembledExternalForceResultantNewtons) +
                " assembled_resultant_relative_error=" +
                std::to_string(externalResultantRelativeError) +
                " assembled_l1_relative_error=" +
                std::to_string(externalL1RelativeError) +
                " enthesis_reaction_resultant_n=" +
                std::to_string(
                    result.quadricepsEnthesisReactionResultantNewtons) +
                " enthesis_reaction_relative_error=" +
                std::to_string(enthesisReactionRelativeError) +
                " passive_reaction_allowance_n=" +
                std::to_string(passiveReactionAllowanceNewtons) +
                " ptl_patella_reaction_relative_error=" +
                std::to_string(ptlPatellaReactionRelativeError) +
                " ptl_tibia_reaction_relative_error=" +
                std::to_string(ptlTibiaReactionRelativeError) +
                " articular_samples=" + std::to_string(
                    adapterDiagnostics.articularContactSampleCount) +
                " articular_mechanical_samples=" + std::to_string(
                    adapterDiagnostics.articularMechanicalSampleCount) +
                " articular_internal_same_body_samples=" + std::to_string(
                    adapterDiagnostics.articularInternalSameBodySampleCount) +
                " articular_closed_samples=" + std::to_string(
                    adapterDiagnostics.articularClosedSampleCount) +
                " articular_normal_force_n=" + std::to_string(
                    adapterDiagnostics.articularNormalForceNewtons) +
                " articular_body_force_l1_n=" + std::to_string(
                    adapterDiagnostics.articularBodyForceL1Newtons) +
                " articular_force_residual_n=" + std::to_string(
                    adapterDiagnostics.articularForceResidualNewtons) +
                " articular_moment_residual_nm=" + std::to_string(
                    adapterDiagnostics.articularMomentResidualNewtonMeters) +
                " rollback=" +
                (transaction.rollbackVerified ? "verified" : "failed") +
                " replay=" +
                (transaction.replayVerified ? "verified" : "failed"));
    const std::uint32_t expectedAdapterPasses =
        2u * driven.stepCount * driven.muscleMetalStepCount;
    require(adapterDiagnostics.initialized &&
                adapterDiagnostics.encodedPassCount == expectedAdapterPasses &&
                adapterDiagnostics.abortCount == 1u,
            "live Open Knee transaction accounting is incomplete: " +
                adapterDiagnostics.message);
    // A conservative continuum may reproduce the source route's instantaneous
    // generalized load at neutral pose, so q/v inequality is not a physical
    // requirement. The stronger direct gate above proves nonzero QAT/PTL bone
    // reactions, continuum deformation, full transaction replay, and rollback.
    driven.tendonContinuumReactionVerified = true;
    result.liveHumanCoupling = true;
    result.rollbackVerified = transaction.rollbackVerified;
    result.replayVerified = transaction.replayVerified;
    result.deviceName = initialized.device;
    return result;
}

struct PectoralisFasciaVisual {
    std::vector<mr_float4> restNodes;
    std::vector<mr_float4> nodes;
    std::vector<std::array<std::uint32_t, 3u>> surfaceTriangles;
    std::vector<mr_float4> anatomicalSurfaceNodes;
    std::vector<mr_float4> anatomicalSurfaceRestNormals;
    std::vector<std::array<std::uint32_t, 3u>> anatomicalSurfaceTriangles;
    std::vector<std::uint32_t> sourceStableIds;
    std::uint32_t tetrahedronCount = 0u;
    std::uint32_t fixedNodeCount = 0u;
    std::uint32_t loadNodeCount = 0u;
    std::uint32_t anatomicallyDeformedVertexCount = 0u;
    std::uint32_t completedSteps = 0u;
    std::uint32_t fgmresIterations = 0u;
    float loadFraction = 0.0f;
    float appliedForceNewtons = 0.0f;
    float maximumDisplacementMeters = 0.0f;
    float anchorReactionResultantNewtons = 0.0f;
    float anchorReactionL1Newtons = 0.0f;
    float maximumAnchorNodeReactionNewtons = 0.0f;
    float maximumAnatomicalMappingDistanceMeters = 0.0f;
    float maximumAppliedAnatomicalMappingDistanceMeters = 0.0f;
    float rmsAnatomicalMappingDistanceMeters = 0.0f;
    float minimumDeterminant = std::numeric_limits<float>::infinity();
    bool deterministicReplayVerified = false;
    bool rollbackVerified = false;
    double coupledTransactionMilliseconds = 0.0;
    std::string deviceName;
};

PectoralisFasciaVisual runPectoralisFascia(
    const LoadedPectoralisFascia& fascia,
    const LoadedSoftTissues& tissues,
    const LoadedMuscles& muscles,
    MuscleDrivenVisualState& driven,
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts& supportContacts,
    const LoadedJointEqualities& jointEqualities,
    const std::span<const MRBodyStateGPU> restBodies,
    const double timestepSeconds,
    const std::uint32_t stepCount,
    const double activation,
    const std::span<const std::uint32_t> selectedSourceMuscleIndices,
    const bool applySelectedActivationIncrement,
    const bool enableRootAssistance,
    const bool removeRootAssistance,
    const std::filesystem::path& matterMetallib
) {
    require(stepCount >= 1u &&
                stepCount <= MR_NUMI_HUMAN_STAND_MAX_STEPS,
            "coupled pectoralis fascia requires a valid Human horizon");
    // Build the continuum in the exact q used by the owning persistent Human
    // transaction. Source-neutral geometry is not a valid fixed-boundary
    // reference once the controller begins from its compiled support pose.
    const GroundAlignedSupport aligned =
        makeGroundAlignedSupport(model, supportContacts);
    CompiledStandActivation fasciaActivation;
    if (applySelectedActivationIncrement) {
        fasciaActivation = compileStaticStandActivation(
            model, muscles, jointEqualities, aligned.q, 1.0, {}
        );
        for (const std::uint32_t muscleIndex : selectedSourceMuscleIndices) {
            fasciaActivation.activation[muscleIndex] = std::min(
                1.0f,
                fasciaActivation.activation[muscleIndex] +
                    static_cast<float>(activation)
            );
        }
    } else {
        fasciaActivation = compileStaticStandActivation(
            model, muscles, jointEqualities, aligned.q, activation,
            selectedSourceMuscleIndices
        );
    }
    const std::vector<float> fasciaQ =
        packMetalConfiguration(fasciaActivation.q);
    metalrobo::MetalArticulatedOperatorResult fasciaPoseResult;
    metalrobo::MetalArticulatedOperatorConfig fasciaPoseConfig;
    fasciaPoseConfig.pointJacobiansOnly = true;
    const auto fasciaPoseDiagnostics = metalrobo::runMetalArticulatedOperator(
        model, {
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = 0u,
            .q = fasciaQ,
            .points = {},
        }, fasciaPoseResult, fasciaPoseConfig
    );
    require(fasciaPoseDiagnostics.succeeded() &&
                fasciaPoseDiagnostics.dispatched &&
                fasciaPoseDiagnostics.published &&
                fasciaPoseDiagnostics.successfulEnvironmentCount == 1u,
            "pectoralis fascia initial Human pose failed: " +
                fasciaPoseDiagnostics.message);
    const std::vector<MRBodyStateGPU> fasciaReferenceBodies =
        visualBodyStates(model, fasciaPoseResult.bodyPoses);
    require(fasciaReferenceBodies.size() == restBodies.size(),
            "pectoralis fascia initial Human body count drifted");
    PectoralisFasciaVisual result;
    result.loadFraction = fascia.header.muscleLoadFraction;
    result.tetrahedronCount = fascia.header.tetrahedronCount;
    result.restNodes.resize(fascia.nodes.size());
    result.nodes.resize(fascia.nodes.size());
    std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads(
        fascia.nodes.size()
    );
    for (auto& load : nodeLoads)
        std::fill_n(load.endpointIndex, 4u, NM_INVALID_INDEX);
    std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors(
        fascia.nodes.size()
    );
    for (auto& anchor : nodeAnchors) anchor.bodyIndex = NM_INVALID_INDEX;
    std::vector<NMNumiHumanTendonFEMEndpointReplacementGPU>
        endpointReplacements;
    endpointReplacements.reserve(fascia.regions.size());
    std::vector<std::uint32_t> regionBindingIndices(
        fascia.regions.size(), MR_INVALID_INDEX
    );
    std::vector<std::array<std::uint32_t, 4u>> orientedTetrahedra;
    orientedTetrahedra.reserve(fascia.tetrahedra.size());
    numi::matter::WorldSource worldSource;
    worldSource.environmentCount = 1u;
    worldSource.frameTimestep = timestepSeconds;
    worldSource.gravity = {0.0, 0.0, 0.0};
    worldSource.mixedSolver.newtonIterations = 8u;
    worldSource.mixedSolver.fgmresIterations = 20u;
    auto material = numi::matter::parseMatterFile(
        NUMI_HUMAN_PECTORALIS_FASCIA_MATERIAL
    );
    require(material.succeeded(), "human pectoralis fascia Matter material did not parse");
    worldSource.materials.push_back(std::move(material.material));
    numi::matter::ObjectSource object;
    object.name = "bodyparts3d_generated_pectoralis_fascia_fallback";
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::fem;
    object.mixedFEM = false;
    object.deformableSelfContact = false;
    object.characteristicLength = 0.005;

    for (std::uint32_t regionIndex = 0u;
         regionIndex < fascia.regions.size(); ++regionIndex) {
        const PectoralisFasciaRegion& region = fascia.regions[regionIndex];
        result.sourceStableIds.push_back(region.softTissueStableId);
        const auto tissueIterator = std::find_if(
            tissues.records.begin(), tissues.records.end(),
            [&region](const SoftTissueRecord& value) {
                return value.stableId == region.softTissueStableId;
            }
        );
        require(tissueIterator != tissues.records.end(),
                "pectoralis fascia source surface is unavailable");
        const SoftTissueRecord& tissue = *tissueIterator;
        const std::uint32_t layerWidth = region.nodeCount / 2u;
        for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
            const std::uint32_t global = region.firstNode + local;
            const PectoralisFasciaNode& node = fascia.nodes[global];
            const SoftTissueVertex& source =
                tissues.vertices[tissue.firstVertex + node.sourceVertexIndex];
            mr_float4 point = softTissueVertexBlendedWorld(
                tissue, source, fasciaReferenceBodies);
            if (local >= layerWidth) {
                const mr_float4 normal = softTissueVertexBlendedNormalWorld(
                    tissue, source, fasciaReferenceBodies
                );
                point.x -= fascia.header.thicknessMeters * normal.x;
                point.y -= fascia.header.thicknessMeters * normal.y;
                point.z -= fascia.header.thicknessMeters * normal.z;
            }
            point.w = 1.0f;
            result.restNodes[global] = point;
            result.nodes[global] = point;
            object.femNodes.push_back({point.x, point.y, point.z});
            if ((node.flags & 1u) != 0u) {
                object.femFixedNodes.push_back(global);
                ++result.fixedNodeCount;
            }
            if ((node.flags & 2u) != 0u) ++result.loadNodeCount;
        }
        std::array<std::uint32_t, 2u> candidateBindings{
            MR_INVALID_INDEX, MR_INVALID_INDEX};
        std::uint32_t candidateCount = 0u;
        for (std::uint32_t index = 0u;
             index < muscles.tendonPayload.bindings.size(); ++index) {
            if (muscles.tendonPayload.bindings[index].muscleIndex ==
                region.muscleIndex) {
                require(candidateCount < candidateBindings.size(),
                        "pectoralis fascia muscle has more than two terminals");
                candidateBindings[candidateCount++] = index;
            }
        }
        require(candidateCount == 2u,
                "pectoralis fascia requires two source tendon terminals");
        mr_float4 fixedCentroid{0.0f, 0.0f, 0.0f, 0.0f};
        mr_float4 loadCentroid{0.0f, 0.0f, 0.0f, 0.0f};
        std::uint32_t fixedSamples = 0u;
        std::uint32_t loadSamples = 0u;
        for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
            const std::uint32_t global = region.firstNode + local;
            if ((fascia.nodes[global].flags & 1u) != 0u) {
                fixedCentroid = femAdd(fixedCentroid, result.restNodes[global]);
                ++fixedSamples;
            }
            if ((fascia.nodes[global].flags & 2u) != 0u) {
                loadCentroid = femAdd(loadCentroid, result.restNodes[global]);
                ++loadSamples;
            }
        }
        require(fixedSamples > 0u && loadSamples > 0u,
                "pectoralis fascia anchor/load centroid is unavailable");
        fixedCentroid = femScale(fixedCentroid, 1.0f / float(fixedSamples));
        loadCentroid = femScale(loadCentroid, 1.0f / float(loadSamples));
        const auto endpointWorld = [&](const std::uint32_t bindingIndex) {
            const auto& value = muscles.tendonPayload.bindings[bindingIndex];
            require(value.bodyIndex < fasciaReferenceBodies.size(),
                    "pectoralis terminal body is outside the Human articulation");
            const MRBodyStateGPU& body = fasciaReferenceBodies[value.bodyIndex];
            const mr_float4 local{
                static_cast<float>(value.resolvedLocalPoint[0]),
                static_cast<float>(value.resolvedLocalPoint[1]),
                static_cast<float>(value.resolvedLocalPoint[2]), 0.0f,
            };
            return femAdd(body.position, rotatePoint(body.orientation, local));
        };
        const mr_float4 endpoint0 = endpointWorld(candidateBindings[0]);
        const mr_float4 endpoint1 = endpointWorld(candidateBindings[1]);
        const float assignment0 =
            femLength(femSubtract(endpoint0, loadCentroid)) +
            femLength(femSubtract(endpoint1, fixedCentroid));
        const float assignment1 =
            femLength(femSubtract(endpoint1, loadCentroid)) +
            femLength(femSubtract(endpoint0, fixedCentroid));
        require(std::isfinite(assignment0) && std::isfinite(assignment1) &&
                    std::abs(assignment0 - assignment1) > 1.0e-5f,
                "pectoralis source terminal/boundary assignment is ambiguous");
        const std::uint32_t loadBindingIndex = assignment0 <= assignment1
            ? candidateBindings[0] : candidateBindings[1];
        const std::uint32_t anchorBindingIndex = loadBindingIndex == candidateBindings[0]
            ? candidateBindings[1] : candidateBindings[0];
        const mr_float4 loadEndpoint = endpointWorld(loadBindingIndex);
        const mr_float4 anchorEndpoint = endpointWorld(anchorBindingIndex);
        require(
            femLength(femSubtract(loadEndpoint, anchorEndpoint)) > 0.01f,
            "pectoralis source tendon terminals are anatomically coincident"
        );
        regionBindingIndices[regionIndex] = loadBindingIndex;
        const auto& anchorBinding =
            muscles.tendonPayload.bindings[anchorBindingIndex];
        const MRBodyStateGPU& anchorBody =
            fasciaReferenceBodies[anchorBinding.bodyIndex];
        const mr_float4 inverseAnchorOrientation{
            -anchorBody.orientation.x, -anchorBody.orientation.y,
            -anchorBody.orientation.z, anchorBody.orientation.w,
        };
        for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
            const std::uint32_t global = region.firstNode + local;
            if ((fascia.nodes[global].flags & 1u) == 0u) continue;
            const mr_float4 localPoint = rotatePoint(
                inverseAnchorOrientation,
                femSubtract(result.restNodes[global], anchorBody.position)
            );
            nodeAnchors[global].bodyIndex = anchorBinding.bodyIndex;
            nodeAnchors[global].flags =
                NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
            nodeAnchors[global].localPoint = {
                localPoint.x, localPoint.y, localPoint.z, 0.0f};
        }
        NMNumiHumanTendonFEMEndpointReplacementGPU replacement{};
        replacement.loadEndpointIndex = loadBindingIndex;
        replacement.anchorEndpointIndex = anchorBindingIndex;
        replacement.flags =
            NM_NUMI_HUMAN_TENDON_FEM_ENDPOINT_REPLACEMENT_ACTIVE;
        replacement.forceOwnerFraction.x = fascia.header.muscleLoadFraction;
        endpointReplacements.push_back(replacement);
        const std::uint32_t regionLoadCount = static_cast<std::uint32_t>(std::count_if(
            fascia.nodes.begin() + region.firstNode,
            fascia.nodes.begin() + region.firstNode + region.nodeCount,
            [](const PectoralisFasciaNode& node) { return (node.flags & 2u) != 0u; }
        ));
        require(regionLoadCount > 0u, "pectoralis fascia has no traction nodes");
        const float scale = fascia.header.muscleLoadFraction /
            static_cast<float>(regionLoadCount);
        for (std::uint32_t local = 0u; local < region.nodeCount; ++local) {
            const std::uint32_t global = region.firstNode + local;
            if ((fascia.nodes[global].flags & 2u) != 0u) {
                nodeLoads[global].endpointIndex[0u] = static_cast<std::uint32_t>(
                    loadBindingIndex
                );
                nodeLoads[global].scale.x = scale;
            }
        }
    }

    std::vector<std::array<double, 3u>> restPoints;
    restPoints.reserve(result.restNodes.size());
    for (const mr_float4& point : result.restNodes) {
        restPoints.push_back({point.x, point.y, point.z});
    }
    // Pectoral v1 qualifies internal fascia response and named load transfer,
    // not fascia self-contact. Limiting contact eligibility to one fixed
    // witness prevents the two 0.6 mm shell faces from being misclassified as
    // penetrating while retaining a non-empty, structurally valid FEM list.
    require(!object.femFixedNodes.empty(), "pectoralis fascia has no fixed contact witness");
    object.femContactNodes = {object.femFixedNodes.front()};
    for (const PectoralisFasciaTetrahedron& source : fascia.tetrahedra) {
        std::array<std::uint32_t, 4u> tetrahedron{
            source.node[0], source.node[1], source.node[2], source.node[3],
        };
        const double volume = femSignedTetrahedronVolume(restPoints, tetrahedron);
        require(std::isfinite(volume) && std::abs(volume) > 1.0e-15,
                "posed pectoralis fascia tetrahedron is degenerate");
        if (volume < 0.0) std::swap(tetrahedron[0], tetrahedron[1]);
        orientedTetrahedra.push_back(tetrahedron);
        object.tetrahedra.push_back({tetrahedron});
    }
    std::map<std::array<std::uint32_t, 3u>,
             std::pair<std::uint32_t, std::array<std::uint32_t, 3u>>> faces;
    for (const auto& tetrahedron : orientedTetrahedra) {
        for (const std::array<std::uint32_t, 3u> face : {
                 std::array<std::uint32_t, 3u>{tetrahedron[0], tetrahedron[2], tetrahedron[1]},
                 std::array<std::uint32_t, 3u>{tetrahedron[0], tetrahedron[1], tetrahedron[3]},
                 std::array<std::uint32_t, 3u>{tetrahedron[1], tetrahedron[2], tetrahedron[3]},
                 std::array<std::uint32_t, 3u>{tetrahedron[2], tetrahedron[0], tetrahedron[3]},
             }) {
            auto key = face;
            std::sort(key.begin(), key.end());
            auto& entry = faces[key];
            ++entry.first;
            entry.second = face;
        }
    }
    for (const auto& [_, entry] : faces) {
        if (entry.first == 1u) result.surfaceTriangles.push_back(entry.second);
    }
    worldSource.objects.push_back(std::move(object));
    numi::matter::CompileOptions compileOptions;
    compileOptions.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(worldSource, compileOptions);
    std::string compileMessage;
    for (const numi::matter::Diagnostic& diagnostic : compiled.diagnostics)
        compileMessage += diagnostic.message + "; ";
    require(compiled.succeeded(),
            "load-driven pectoralis fascia world did not compile: " + compileMessage);

    numi::matter::Runtime runtime;
    const auto initialized = runtime.initialize(compiled.world, {
        .metallib = matterMetallib,
        .environmentCount = 1u,
        .captureEvents = true,
        .captureDiagnostics = true,
        .automaticIdentification = false,
        .adaptiveTransfer = false,
    });
    require(initialized.encoded && runtime.valid(),
            "could not initialize pectoralis fascia Matter runtime: " + initialized.message);
    result.deviceName = initialized.device;
    const auto initial = runtime.snapshot();
    require(initial.available && initial.femNodes.size() == result.nodes.size(),
            "pectoralis fascia initial snapshot is unavailable");
    numi::matter::NumiHumanTendonFEMLoadAdapter adapter;
    require(adapter.initialize(runtime, {
                .nodeLoads = nodeLoads,
                .nodeAnchors = nodeAnchors,
                .endpointReplacements = endpointReplacements,
                .endpointCount = static_cast<std::uint32_t>(
                    muscles.tendonPayload.bindings.size()
                ),
                .environmentCount = 1u,
                .productionForceOwnerFraction =
                    fascia.header.muscleLoadFraction,
            }, {
                .metallib = matterMetallib,
            }),
            "could not initialize same-command-buffer pectoralis tendon adapter");
    HumanTendonContinuumTransaction transaction{
        .program = adapter.program(),
        .runtime = &runtime,
        .initial = initial,
    };
    driven = integratePersistentMetalHumanState(
        model, muscles, supportContacts, jointEqualities, timestepSeconds,
        stepCount, activation, selectedSourceMuscleIndices,
        applySelectedActivationIncrement, enableRootAssistance,
        removeRootAssistance, true, &transaction
    );
    const auto accepted = transaction.accepted;
    require(accepted.available && accepted.femNodes.size() == result.nodes.size(),
            "pectoralis fascia accepted snapshot is unavailable");
    id<MTLBuffer> reactionBuffer = (__bridge id<MTLBuffer>)
        runtime.femConstraintReactionBuffer();
    require(reactionBuffer != nil,
            "pectoralis fascia fixed-node reaction buffer is unavailable");
    const NSUInteger reactionBytes = static_cast<NSUInteger>(
        result.nodes.size() * sizeof(nm_float4));
    id<MTLBuffer> reactionReadback = [reactionBuffer.device
        newBufferWithLength:reactionBytes options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> reactionQueue = [reactionBuffer.device newCommandQueue];
    id<MTLCommandBuffer> reactionCommand = [reactionQueue commandBuffer];
    id<MTLBlitCommandEncoder> reactionBlit =
        [reactionCommand blitCommandEncoder];
    require(reactionReadback != nil && reactionQueue != nil &&
                reactionCommand != nil && reactionBlit != nil,
            "pectoralis fascia reaction readback could not be allocated");
    [reactionBlit copyFromBuffer:reactionBuffer sourceOffset:0u
                        toBuffer:reactionReadback destinationOffset:0u
                            size:reactionBytes];
    [reactionBlit endEncoding];
    [reactionCommand commit];
    [reactionCommand waitUntilCompleted];
    require(reactionCommand.status == MTLCommandBufferStatusCompleted,
            "pectoralis fascia reaction readback failed");
    const auto* reactionValues =
        static_cast<const nm_float4*>(reactionReadback.contents);
    mr_float4 reactionResultant{0.0f, 0.0f, 0.0f, 0.0f};
    for (std::uint32_t node = 0u; node < result.nodes.size(); ++node) {
        if ((nodeAnchors[node].flags &
                NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) == 0u) continue;
        const mr_float4 reaction{
            reactionValues[node].x, reactionValues[node].y,
            reactionValues[node].z, 0.0f};
        const float magnitude = femLength(reaction);
        require(std::isfinite(magnitude),
                "pectoralis fascia anchor reaction is non-finite");
        reactionResultant = femAdd(reactionResultant, reaction);
        result.anchorReactionL1Newtons += magnitude;
        result.maximumAnchorNodeReactionNewtons = std::max(
            result.maximumAnchorNodeReactionNewtons, magnitude);
    }
    result.anchorReactionResultantNewtons = femLength(reactionResultant);
    require(std::isfinite(result.anchorReactionResultantNewtons) &&
                result.anchorReactionResultantNewtons > 0.0f &&
                result.anchorReactionL1Newtons > 0.0f &&
                result.maximumAnchorNodeReactionNewtons > 0.0f,
            "pectoralis fascia produced no fixed-node force transfer to bone");
    require(transaction.rollbackVerified && transaction.replayVerified,
            "pectoralis fascia did not close Human/Matter rollback and replay");
    const auto adapterDiagnostics = adapter.diagnostics();
    // The deliberate downstream-rejection probe aborts after this adapter's
    // pre-dynamics phase, so it owns rollback but is not a completed pass.
    const std::uint32_t expectedAdapterPasses =
        2u * driven.stepCount * driven.muscleMetalStepCount;
    require(adapterDiagnostics.initialized &&
                adapterDiagnostics.encodedPassCount == expectedAdapterPasses &&
                adapterDiagnostics.abortCount == 1u,
            "pectoralis fascia adapter transaction accounting is incomplete: " +
                adapterDiagnostics.message);
    id<MTLBuffer> matterStatusBuffer =
        (__bridge id<MTLBuffer>)runtime.statusBuffer();
    auto* statuses = static_cast<NMMatterStatusGPU*>(matterStatusBuffer.contents);
    require(statuses != nullptr && statuses[0].code == NM_STATUS_SUCCESS,
            "pectoralis fascia Matter transaction failed status=" +
                std::to_string(statuses == nullptr ? NM_STATUS_INVALID_DISPATCH : statuses[0].code) +
                " object=" + std::to_string(statuses == nullptr ? MR_INVALID_INDEX : statuses[0].objectIndex) +
                " index=" + std::to_string(statuses == nullptr ? MR_INVALID_INDEX : statuses[0].failingIndex));
    result.completedSteps = driven.persistentCompletedSteps;
    result.fgmresIterations = statuses[0].fgmresIterations;
    result.minimumDeterminant = statuses[0].diagnostics.x;
    result.deterministicReplayVerified = transaction.replayVerified;
    result.rollbackVerified = transaction.rollbackVerified;
    // The fascia solve is encoded inside the owning Human command-buffer
    // transaction.  This interval therefore covers that complete coupled
    // transaction; it is not an isolated Matter-kernel GPU measurement.
    result.coupledTransactionMilliseconds = driven.muscleMetalElapsedMilliseconds;
    for (std::size_t regionIndex = 0u;
         regionIndex < fascia.regions.size(); ++regionIndex) {
        const std::uint32_t bindingIndex = regionBindingIndices[regionIndex];
        require(bindingIndex < driven.finalTendonTransfers.size(),
                "pectoralis fascia published tendon transfer is unavailable");
        const auto& transfer = driven.finalTendonTransfers[bindingIndex];
        require(transfer.status == MR_NUMI_HUMAN_TENDON_TRANSFER_SUCCESS &&
                    transfer.bindingIndex == bindingIndex,
                "pectoralis fascia published tendon transfer failed");
        result.appliedForceNewtons += femLength({
            transfer.terminalWorldForce.x * fascia.header.muscleLoadFraction,
            transfer.terminalWorldForce.y * fascia.header.muscleLoadFraction,
            transfer.terminalWorldForce.z * fascia.header.muscleLoadFraction,
            0.0f,
        });
    }
    for (std::uint32_t index = 0u; index < accepted.femNodes.size(); ++index) {
        const auto& node = accepted.femNodes[index];
        result.nodes[index] = {
            node.positionAndMass.x, node.positionAndMass.y,
            node.positionAndMass.z, 1.0f,
        };
        if ((fascia.nodes[index].flags & 1u) == 0u) {
            result.maximumDisplacementMeters = std::max(
                result.maximumDisplacementMeters,
                femLength(femSubtract(result.nodes[index], result.restNodes[index]))
            );
        }
    }
    // Keep the efficient FEM envelope internal. Visible fascia uses the exact
    // anterior-facing subset of each provenance-pinned BodyParts3D pectoralis
    // surface, matching the compiler's declared generated-sheet boundary.
    // Rendering the entire closed muscle volume as fascia creates false
    // posterior and inferior lobes. A deterministic four-nearest map transfers
    // displacement only to vertices referenced by this anterior sheet.
    double squaredMappingDistanceSum = 0.0;
    std::uint64_t mappedVertexCount = 0u;
    for (std::uint32_t regionIndex = 0u;
         regionIndex < fascia.regions.size(); ++regionIndex) {
        const PectoralisFasciaRegion& region = fascia.regions[regionIndex];
        const auto tissueIterator = std::find_if(
            tissues.records.begin(), tissues.records.end(),
            [&region](const SoftTissueRecord& value) {
                return value.stableId == region.softTissueStableId;
            }
        );
        require(tissueIterator != tissues.records.end(),
                "pectoralis fascia anatomical presentation surface is unavailable");
        const SoftTissueRecord& tissue = *tissueIterator;
        const std::uint32_t layerWidth = region.nodeCount / 2u;
        require(layerWidth >= 4u && 2u * layerWidth == region.nodeCount,
                "pectoralis fascia anatomical presentation has an invalid mechanics layer");
        std::vector<mr_float4> sourceRestPoints(tissue.vertexCount);
        std::vector<mr_float4> sourceRestNormals(tissue.vertexCount);
        for (std::uint32_t offset = 0u; offset < tissue.vertexCount; ++offset) {
            const SoftTissueVertex& source =
                tissues.vertices[tissue.firstVertex + offset];
            sourceRestPoints[offset] =
                softTissueVertexBlendedWorld(
                    tissue, source, fasciaReferenceBodies);
            sourceRestNormals[offset] =
                softTissueVertexBlendedNormalWorld(
                    tissue, source, fasciaReferenceBodies);
        }
        std::vector<std::array<std::uint32_t, 3u>> anteriorTriangles;
        anteriorTriangles.reserve(
            fascia.presentationTriangles.size() / fascia.regions.size()
        );
        std::vector<bool> usedSourceVertex(tissue.vertexCount, false);
        for (const auto& sourceTriangle : fascia.presentationTriangles) {
            if (sourceTriangle.regionIndex != regionIndex) continue;
            const std::array<std::uint32_t, 3u> triangle{
                sourceTriangle.sourceVertex[0],
                sourceTriangle.sourceVertex[1],
                sourceTriangle.sourceVertex[2],
            };
            anteriorTriangles.push_back(triangle);
            for (const std::uint32_t vertex : triangle) {
                usedSourceVertex[vertex] = true;
            }
        }
        require(anteriorTriangles.size() >= 32u &&
                    std::count(usedSourceVertex.begin(), usedSourceVertex.end(), true) >= 32,
                "pectoralis fascia exact anterior presentation is incomplete");
        std::vector<std::uint32_t> localToPresentation(
            tissue.vertexCount, MR_INVALID_INDEX
        );
        for (std::uint32_t offset = 0u; offset < tissue.vertexCount; ++offset) {
            if (!usedSourceVertex[offset]) continue;
            const mr_float4 restPoint = sourceRestPoints[offset];
            std::array<std::pair<float, std::uint32_t>, 4u> nearest{};
            for (auto& entry : nearest) {
                entry = {std::numeric_limits<float>::infinity(), MR_INVALID_INDEX};
            }
            for (std::uint32_t local = 0u; local < layerWidth; ++local) {
                const std::uint32_t global = region.firstNode + local;
                const mr_float4 delta = femSubtract(
                    restPoint, result.restNodes[global]
                );
                const float squaredDistance = femDot(delta, delta);
                if (squaredDistance >= nearest.back().first) continue;
                nearest.back() = {squaredDistance, global};
                std::sort(nearest.begin(), nearest.end(),
                          [](const auto& left, const auto& right) {
                              return left.first < right.first;
                          });
            }
            require(nearest.front().second != MR_INVALID_INDEX &&
                        std::isfinite(nearest.front().first),
                    "pectoralis fascia anatomical presentation mapping failed");
            mr_float4 displacement{0.0f, 0.0f, 0.0f, 0.0f};
            if (nearest.front().first <= 1.0e-12f) {
                displacement = femSubtract(
                    result.nodes[nearest.front().second],
                    result.restNodes[nearest.front().second]
                );
            } else {
                float weightSum = 0.0f;
                for (const auto& [squaredDistance, global] : nearest) {
                    require(global != MR_INVALID_INDEX && std::isfinite(squaredDistance),
                            "pectoralis fascia anatomical presentation has incomplete support");
                    const float weight = 1.0f / std::max(squaredDistance, 1.0e-10f);
                    const mr_float4 nodeDisplacement = femSubtract(
                        result.nodes[global], result.restNodes[global]
                    );
                    displacement.x += weight * nodeDisplacement.x;
                    displacement.y += weight * nodeDisplacement.y;
                    displacement.z += weight * nodeDisplacement.z;
                    weightSum += weight;
                }
                require(weightSum > 0.0f && std::isfinite(weightSum),
                        "pectoralis fascia anatomical presentation weights are invalid");
                displacement.x /= weightSum;
                displacement.y /= weightSum;
                displacement.z /= weightSum;
            }
            const float mappingDistance = std::sqrt(nearest.front().first);
            constexpr float kFullDisplacementSupportMeters = 0.030f;
            constexpr float kMaximumDisplacementSupportMeters = 0.060f;
            float displacementSupport = 1.0f;
            if (mappingDistance >= kMaximumDisplacementSupportMeters) {
                displacementSupport = 0.0f;
            } else if (mappingDistance > kFullDisplacementSupportMeters) {
                const float t = (mappingDistance - kFullDisplacementSupportMeters) /
                    (kMaximumDisplacementSupportMeters - kFullDisplacementSupportMeters);
                displacementSupport = 1.0f - t * t * (3.0f - 2.0f * t);
            }
            displacement.x *= displacementSupport;
            displacement.y *= displacementSupport;
            displacement.z *= displacementSupport;
            if (displacementSupport > 0.0f) {
                ++result.anatomicallyDeformedVertexCount;
                result.maximumAppliedAnatomicalMappingDistanceMeters = std::max(
                    result.maximumAppliedAnatomicalMappingDistanceMeters,
                    mappingDistance
                );
            }
            mr_float4 deformed = femAdd(restPoint, displacement);
            deformed.w = 1.0f;
            localToPresentation[offset] = static_cast<std::uint32_t>(
                result.anatomicalSurfaceNodes.size()
            );
            result.anatomicalSurfaceNodes.push_back(deformed);
            mr_float4 sourceNormal = sourceRestNormals[offset];
            sourceNormal.w = 0.0f;
            result.anatomicalSurfaceRestNormals.push_back(sourceNormal);
            result.maximumAnatomicalMappingDistanceMeters = std::max(
                result.maximumAnatomicalMappingDistanceMeters, mappingDistance
            );
            squaredMappingDistanceSum += nearest.front().first;
            ++mappedVertexCount;
        }
        for (auto triangle : anteriorTriangles) {
            for (std::uint32_t& vertex : triangle) {
                require(vertex < localToPresentation.size() &&
                            localToPresentation[vertex] != MR_INVALID_INDEX,
                        "pectoralis fascia anterior triangle has no presentation vertex");
                vertex = localToPresentation[vertex];
            }
            result.anatomicalSurfaceTriangles.push_back(triangle);
        }
    }
    require(mappedVertexCount == result.anatomicalSurfaceNodes.size() &&
                result.anatomicalSurfaceRestNormals.size() ==
                    result.anatomicalSurfaceNodes.size() &&
                !result.anatomicalSurfaceTriangles.empty(),
            "pectoralis fascia anatomical presentation is empty");
    result.rmsAnatomicalMappingDistanceMeters = static_cast<float>(std::sqrt(
        squaredMappingDistanceSum / static_cast<double>(mappedVertexCount)
    ));
    require(result.minimumDeterminant > 0.35f &&
                result.maximumDisplacementMeters > 0.0f &&
                std::isfinite(result.appliedForceNewtons) &&
                std::isfinite(result.rmsAnatomicalMappingDistanceMeters) &&
                result.maximumAnatomicalMappingDistanceMeters < 0.25f &&
                result.anatomicallyDeformedVertexCount > 0u &&
                result.maximumAppliedAnatomicalMappingDistanceMeters <= 0.060001f,
            "pectoralis fascia deformation certificate is invalid");
    return result;
}

struct PlantarSurfacePatch {
    std::array<mr_float4, 4u> points{};
    std::array<std::uint32_t, 4u> sourceVertexIndices{};
    mr_float4 centroid{};
    float radiusMeters = 0.0f;
};

struct PlantarFasciaBandAudit {
    std::uint32_t ray = 0u;
    std::uint32_t calcaneusStableId = 0u;
    std::uint32_t metatarsalStableId = 0u;
    std::uint32_t proximalPhalanxStableId = 0u;
    std::uint32_t firstNode = 0u;
    std::uint32_t wrapFirstNode = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    double publishedRestLengthMeters = 0.0;
    double bodypartsRestLengthMeters = 0.0;
    double targetStiffnessNewtonsPerMillimeter = 0.0;
    double calibratedCrossSectionSquareMillimeters = 0.0;
    double calcanealPatchRadiusMeters = 0.0;
    double metatarsalPatchRadiusMeters = 0.0;
    double metatarsalPulleyRadiusMeters = 0.0;
    double phalanxPatchRadiusMeters = 0.0;
    double acceptedExtensionMeters = 0.0;
    double tensionNewtons = 0.0;
    double forceClosureResidualNewtons = 0.0;
    double momentClosureResidualNewtonMeters = 0.0;
    double calcanealReactionResultantNewtons = 0.0;
    double metatarsalReactionResultantNewtons = 0.0;
    double proximalFootBodyReactionResultantNewtons = 0.0;
    double phalanxReactionResultantNewtons = 0.0;
};

struct PlantarFasciaSideAudit {
    bool available = false;
    std::string side;
    std::string deviceName;
    std::uint32_t calcaneusBodyIndex = MR_INVALID_INDEX;
    std::uint32_t toesBodyIndex = MR_INVALID_INDEX;
    std::uint32_t mtpQIndex = MR_INVALID_INDEX;
    std::uint32_t completedSteps = 0u;
    std::uint32_t nodeCount = 0u;
    std::uint32_t tetrahedronCount = 0u;
    double qualificationMTPRadians = 0.0;
    double sourceLengthScale = 0.0;
    double maximumRestLengthPatternRelativeResidual = 0.0;
    double maximumFreeNodeDisplacementMeters = 0.0;
    double minimumDeterminant = std::numeric_limits<double>::infinity();
    double maximumDeterminant = 0.0;
    double maximumAnchorTargetResidualMeters = 0.0;
    double maximumConfigurationDeltaFromSourceJT = 0.0;
    double maximumVelocityDeltaFromSourceJT = 0.0;
    double totalTensionNewtons = 0.0;
    double maximumForceClosureResidualNewtons = 0.0;
    double maximumMomentClosureResidualNewtonMeters = 0.0;
    bool rollbackVerified = false;
    bool replayVerified = false;
    std::array<PlantarFasciaBandAudit, 5u> bands{};
};

const BoneRecord& plantarBone(
    const LoadedBones& bones,
    const std::uint32_t stableId,
    const std::uint32_t bodyIndex
) {
    const auto found = std::find_if(
        bones.records.begin(), bones.records.end(),
        [stableId](const BoneRecord& bone) { return bone.stableId == stableId; }
    );
    require(found != bones.records.end() && found->bodyIndex == bodyIndex,
            "plantar fascia named BodyParts3D bone/body binding is unavailable");
    return *found;
}

mr_float4 plantarBoneCentroid(
    const LoadedBones& bones,
    const BoneRecord& bone,
    const std::span<const MRBodyStateGPU> bodies
) {
    require(bone.bodyIndex < bodies.size(),
            "plantar fascia bone body pose is unavailable");
    mr_float4 centroid{};
    for (std::uint32_t offset = 0u; offset < bone.vertexCount; ++offset) {
        centroid = femAdd(
            centroid,
            boneVertexWorld(
                bone, bones.vertices[bone.firstVertex + offset],
                bodies[bone.bodyIndex]
            )
        );
    }
    return femScale(centroid, 1.0f / static_cast<float>(bone.vertexCount));
}

PlantarSurfacePatch plantarPatchFromTriangle(
    const LoadedBones& bones,
    const BoneRecord& bone,
    const std::span<const MRBodyStateGPU> bodies,
    const std::uint32_t triangleOffset,
    const mr_float4 sortWidthAxis,
    const mr_float4 sortThicknessAxis
) {
    require(triangleOffset + 2u < bone.indexCount && triangleOffset % 3u == 0u,
            "plantar fascia surface triangle is malformed");
    std::array<std::uint32_t, 3u> triangle{
        bones.indices[bone.firstIndex + triangleOffset],
        bones.indices[bone.firstIndex + triangleOffset + 1u],
        bones.indices[bone.firstIndex + triangleOffset + 2u],
    };
    mr_float4 triangleCentroid{};
    for (const std::uint32_t vertex : triangle) {
        triangleCentroid = femAdd(
            triangleCentroid,
            boneVertexWorld(bone, bones.vertices[vertex], bodies[bone.bodyIndex])
        );
    }
    triangleCentroid = femScale(triangleCentroid, 1.0f / 3.0f);
    std::uint32_t fourth = MR_INVALID_INDEX;
    float fourthDistanceSquared = std::numeric_limits<float>::infinity();
    for (std::uint32_t offset = 0u; offset < bone.indexCount; offset += 3u) {
        if (offset == triangleOffset) continue;
        const std::array<std::uint32_t, 3u> candidate{
            bones.indices[bone.firstIndex + offset],
            bones.indices[bone.firstIndex + offset + 1u],
            bones.indices[bone.firstIndex + offset + 2u],
        };
        std::uint32_t shared = 0u;
        std::uint32_t unique = MR_INVALID_INDEX;
        for (const std::uint32_t vertex : candidate) {
            if (std::find(triangle.begin(), triangle.end(), vertex) !=
                triangle.end()) {
                ++shared;
            } else {
                unique = vertex;
            }
        }
        if (shared != 2u || unique == MR_INVALID_INDEX) continue;
        const mr_float4 point = boneVertexWorld(
            bone, bones.vertices[unique], bodies[bone.bodyIndex]
        );
        const float distanceSquared = femDot(
            femSubtract(point, triangleCentroid),
            femSubtract(point, triangleCentroid)
        );
        if (distanceSquared < fourthDistanceSquared) {
            fourthDistanceSquared = distanceSquared;
            fourth = unique;
        }
    }
    require(fourth != MR_INVALID_INDEX,
            "plantar fascia attachment triangle has no connected fourth vertex");
    PlantarSurfacePatch patch;
    patch.sourceVertexIndices = {
        triangle[0u], triangle[1u], triangle[2u], fourth,
    };
    for (std::size_t index = 0u; index < patch.points.size(); ++index) {
        patch.points[index] = boneVertexWorld(
            bone, bones.vertices[patch.sourceVertexIndices[index]],
            bodies[bone.bodyIndex]
        );
        patch.centroid = femAdd(patch.centroid, patch.points[index]);
    }
    patch.centroid = femScale(patch.centroid, 0.25f);
    std::array<std::pair<float, std::size_t>, 4u> order{};
    for (std::size_t index = 0u; index < patch.points.size(); ++index) {
        const mr_float4 delta = femSubtract(patch.points[index], patch.centroid);
        order[index] = {
            std::atan2(
                femDot(delta, sortThicknessAxis),
                femDot(delta, sortWidthAxis)
            ),
            index,
        };
        patch.radiusMeters = std::max(
            patch.radiusMeters, femLength(delta)
        );
    }
    std::sort(order.begin(), order.end());
    const auto unsortedPoints = patch.points;
    const auto unsortedIndices = patch.sourceVertexIndices;
    for (std::size_t index = 0u; index < order.size(); ++index) {
        patch.points[index] = unsortedPoints[order[index].second];
        patch.sourceVertexIndices[index] =
            unsortedIndices[order[index].second];
    }
    require(std::isfinite(patch.radiusMeters) &&
                patch.radiusMeters > 1.0e-5f &&
                patch.radiusMeters < 0.02f,
            "plantar fascia exact surface patch radius is invalid");
    return patch;
}

PlantarSurfacePatch selectCalcanealPlantarPatch(
    const LoadedBones& bones,
    const BoneRecord& bone,
    const std::span<const MRBodyStateGPU> bodies,
    const mr_float4 forward,
    const mr_float4 vertical,
    const mr_float4 medial
) {
    std::array<float, 3u> minimum{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
    };
    std::array<float, 3u> maximum{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    for (std::uint32_t offset = 0u; offset < bone.vertexCount; ++offset) {
        const mr_float4 point = boneVertexWorld(
            bone, bones.vertices[bone.firstVertex + offset],
            bodies[bone.bodyIndex]
        );
        const std::array<float, 3u> projection{
            femDot(point, forward), femDot(point, vertical),
            femDot(point, medial),
        };
        for (std::size_t axis = 0u; axis < projection.size(); ++axis) {
            minimum[axis] = std::min(minimum[axis], projection[axis]);
            maximum[axis] = std::max(maximum[axis], projection[axis]);
        }
    }
    float bestScore = -std::numeric_limits<float>::infinity();
    std::uint32_t bestOffset = MR_INVALID_INDEX;
    for (std::uint32_t offset = 0u; offset < bone.indexCount; offset += 3u) {
        mr_float4 centroid{};
        for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
            const std::uint32_t vertex =
                bones.indices[bone.firstIndex + offset + corner];
            centroid = femAdd(
                centroid,
                boneVertexWorld(
                    bone, bones.vertices[vertex], bodies[bone.bodyIndex]
                )
            );
        }
        centroid = femScale(centroid, 1.0f / 3.0f);
        const float posterior = (maximum[0u] - femDot(centroid, forward)) /
            std::max(1.0e-6f, maximum[0u] - minimum[0u]);
        const float plantar = (maximum[1u] - femDot(centroid, vertical)) /
            std::max(1.0e-6f, maximum[1u] - minimum[1u]);
        const float medialFraction = (femDot(centroid, medial) - minimum[2u]) /
            std::max(1.0e-6f, maximum[2u] - minimum[2u]);
        const float score =
            0.45f * posterior + 0.40f * plantar + 0.15f * medialFraction;
        if (score > bestScore) {
            bestScore = score;
            bestOffset = offset;
        }
    }
    require(bestOffset != MR_INVALID_INDEX && bestScore > 0.65f,
            "plantar fascia medial calcaneal tubercle patch is unavailable");
    return plantarPatchFromTriangle(
        bones, bone, bodies, bestOffset, medial, vertical
    );
}

PlantarSurfacePatch selectProximalPhalanxPlantarPatch(
    const LoadedBones& bones,
    const BoneRecord& bone,
    const std::span<const MRBodyStateGPU> bodies,
    const mr_float4 forward,
    const mr_float4 vertical,
    const mr_float4 medial
) {
    float minimumForward = std::numeric_limits<float>::infinity();
    float maximumForward = -std::numeric_limits<float>::infinity();
    float minimumVertical = std::numeric_limits<float>::infinity();
    float maximumVertical = -std::numeric_limits<float>::infinity();
    for (std::uint32_t offset = 0u; offset < bone.vertexCount; ++offset) {
        const mr_float4 point = boneVertexWorld(
            bone, bones.vertices[bone.firstVertex + offset],
            bodies[bone.bodyIndex]
        );
        minimumForward = std::min(minimumForward, femDot(point, forward));
        maximumForward = std::max(maximumForward, femDot(point, forward));
        minimumVertical = std::min(minimumVertical, femDot(point, vertical));
        maximumVertical = std::max(maximumVertical, femDot(point, vertical));
    }
    float bestScore = -std::numeric_limits<float>::infinity();
    std::uint32_t bestOffset = MR_INVALID_INDEX;
    for (std::uint32_t offset = 0u; offset < bone.indexCount; offset += 3u) {
        mr_float4 centroid{};
        for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
            const std::uint32_t vertex =
                bones.indices[bone.firstIndex + offset + corner];
            centroid = femAdd(
                centroid,
                boneVertexWorld(
                    bone, bones.vertices[vertex], bodies[bone.bodyIndex]
                )
            );
        }
        centroid = femScale(centroid, 1.0f / 3.0f);
        const float proximal = (maximumForward - femDot(centroid, forward)) /
            std::max(1.0e-6f, maximumForward - minimumForward);
        const float plantar = (maximumVertical - femDot(centroid, vertical)) /
            std::max(1.0e-6f, maximumVertical - minimumVertical);
        const float score = 0.70f * proximal + 0.30f * plantar;
        if (score > bestScore) {
            bestScore = score;
            bestOffset = offset;
        }
    }
    require(bestOffset != MR_INVALID_INDEX && bestScore > 0.65f,
            "plantar fascia proximal-phalanx plantar patch is unavailable");
    return plantarPatchFromTriangle(
        bones, bone, bodies, bestOffset, medial, vertical
    );
}

PlantarSurfacePatch selectMetatarsalHeadPlantarPatch(
    const LoadedBones& bones,
    const BoneRecord& bone,
    const std::span<const MRBodyStateGPU> bodies,
    const mr_float4 forward,
    const mr_float4 vertical,
    const mr_float4 medial
) {
    float minimumForward = std::numeric_limits<float>::infinity();
    float maximumForward = -std::numeric_limits<float>::infinity();
    float minimumVertical = std::numeric_limits<float>::infinity();
    float maximumVertical = -std::numeric_limits<float>::infinity();
    for (std::uint32_t offset = 0u; offset < bone.vertexCount; ++offset) {
        const mr_float4 point = boneVertexWorld(
            bone, bones.vertices[bone.firstVertex + offset],
            bodies[bone.bodyIndex]
        );
        minimumForward = std::min(minimumForward, femDot(point, forward));
        maximumForward = std::max(maximumForward, femDot(point, forward));
        minimumVertical = std::min(minimumVertical, femDot(point, vertical));
        maximumVertical = std::max(maximumVertical, femDot(point, vertical));
    }
    float bestScore = -std::numeric_limits<float>::infinity();
    std::uint32_t bestOffset = MR_INVALID_INDEX;
    for (std::uint32_t offset = 0u; offset < bone.indexCount; offset += 3u) {
        mr_float4 centroid{};
        for (std::uint32_t corner = 0u; corner < 3u; ++corner) {
            const std::uint32_t vertex =
                bones.indices[bone.firstIndex + offset + corner];
            centroid = femAdd(
                centroid,
                boneVertexWorld(
                    bone, bones.vertices[vertex], bodies[bone.bodyIndex]
                )
            );
        }
        centroid = femScale(centroid, 1.0f / 3.0f);
        const float distal = (femDot(centroid, forward) - minimumForward) /
            std::max(1.0e-6f, maximumForward - minimumForward);
        const float plantar = (maximumVertical - femDot(centroid, vertical)) /
            std::max(1.0e-6f, maximumVertical - minimumVertical);
        const float score = 0.65f * distal + 0.35f * plantar;
        if (score > bestScore) {
            bestScore = score;
            bestOffset = offset;
        }
    }
    require(bestOffset != MR_INVALID_INDEX && bestScore > 0.65f,
            "plantar fascia metatarsal-head pulley patch is unavailable");
    return plantarPatchFromTriangle(
        bones, bone, bodies, bestOffset, medial, vertical
    );
}

mr_float4 plantarAnchorLocalPoint(
    const mr_float4 worldPoint,
    const MRBodyStateGPU& body
) {
    const mr_float4 inverse{
        -body.orientation.x, -body.orientation.y, -body.orientation.z,
        body.orientation.w,
    };
    return rotatePoint(inverse, femSubtract(worldPoint, body.position));
}

[[maybe_unused]] PlantarFasciaSideAudit runPlantarFasciaSide(
    const LoadedBones& bones,
    const LoadedMuscles& muscles,
    MuscleDrivenVisualState& driven,
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts& supportContacts,
    const LoadedJointEqualities& jointEqualities,
    const double timestepSeconds,
    const std::uint32_t stepCount,
    const std::size_t sideIndex,
    const std::filesystem::path& matterMetallib
) {
    require(sideIndex < 2u && stepCount >= 1u &&
                stepCount <= MR_NUMI_HUMAN_STAND_MAX_STEPS,
            "plantar fascia side qualification request is invalid");
    constexpr std::array<std::uint32_t, 2u> kCalcaneusStableIds{6u, 7u};
    constexpr std::array<std::uint32_t, 2u> kCalcaneusBodyIndices{138u, 152u};
    constexpr std::array<std::uint32_t, 2u> kToesBodyIndices{139u, 153u};
    constexpr std::array<std::uint32_t, 2u> kMTPQIndices{111u, 125u};
    constexpr std::array<std::array<std::uint32_t, 5u>, 2u>
        kProximalPhalanxStableIds{{
            {{131u, 133u, 136u, 139u, 142u}},
            {{150u, 152u, 155u, 158u, 161u}},
        }};
    constexpr std::array<std::array<std::uint32_t, 5u>, 2u>
        kMetatarsalStableIds{{
            {{126u, 127u, 128u, 129u, 130u}},
            {{145u, 146u, 147u, 148u, 149u}},
        }};
    constexpr std::array<double, 5u> kPublishedRestLengthsMeters{
        0.151, 0.149, 0.148, 0.140, 0.131,
    };
    constexpr std::array<double, 5u> kStiffnessNewtonsPerMillimeter{
        60.0, 50.0, 50.0, 20.0, 20.0,
    };
    constexpr double kYoungModulusPascals = 350.0e6;
    constexpr float kThicknessMeters = 0.002f;
    constexpr std::uint32_t kSegmentCount = 10u;
    // The Rajagopal/OpenSim MTP coordinate is negative in dorsiflexion.
    constexpr double kQualificationMTPRadians = -0.10;

    PlantarFasciaSideAudit result;
    result.side = sideIndex == 0u ? "right" : "left";
    result.calcaneusBodyIndex = kCalcaneusBodyIndices[sideIndex];
    result.toesBodyIndex = kToesBodyIndices[sideIndex];
    result.mtpQIndex = kMTPQIndices[sideIndex];
    result.qualificationMTPRadians = kQualificationMTPRadians;
    GroundAlignedSupport aligned = makeGroundAlignedSupport(model, supportContacts);
    require(result.mtpQIndex < aligned.q.size(),
            "plantar fascia MTP coordinate is unavailable");
    aligned.q[result.mtpQIndex] = 0.0;
    double maximumProjection = 0.0;
    const auto projection = metalrobo::projectNumiHumanJointEqualities(
        jointEqualities.payload.records, aligned.q, &maximumProjection
    );
    require(projection.succeeded() && std::isfinite(maximumProjection),
            "plantar fascia neutral equality projection failed");
    const std::vector<float> neutralQ = packMetalConfiguration(aligned.q);
    metalrobo::MetalArticulatedOperatorResult neutralPose;
    metalrobo::MetalArticulatedOperatorConfig poseConfig;
    poseConfig.pointJacobiansOnly = true;
    const auto neutralDiagnostics = metalrobo::runMetalArticulatedOperator(
        model, {
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = 0u,
            .q = neutralQ,
            .points = {},
        }, neutralPose, poseConfig
    );
    require(neutralDiagnostics.succeeded() && neutralDiagnostics.published &&
                neutralDiagnostics.successfulEnvironmentCount == 1u,
            "plantar fascia neutral Human pose failed: " +
                neutralDiagnostics.message);
    const std::vector<MRBodyStateGPU> neutralBodies =
        visualBodyStates(model, neutralPose.bodyPoses);
    require(result.calcaneusBodyIndex < neutralBodies.size() &&
                result.toesBodyIndex < neutralBodies.size(),
            "plantar fascia Human body poses are incomplete");
    const BoneRecord& calcaneus = plantarBone(
        bones, kCalcaneusStableIds[sideIndex], result.calcaneusBodyIndex
    );
    std::array<const BoneRecord*, 5u> phalanges{};
    std::array<const BoneRecord*, 5u> metatarsals{};
    mr_float4 phalanxCentroid{};
    for (std::size_t ray = 0u; ray < phalanges.size(); ++ray) {
        phalanges[ray] = &plantarBone(
            bones, kProximalPhalanxStableIds[sideIndex][ray],
            result.toesBodyIndex
        );
        metatarsals[ray] = &plantarBone(
            bones, kMetatarsalStableIds[sideIndex][ray],
            result.calcaneusBodyIndex
        );
        phalanxCentroid = femAdd(
            phalanxCentroid,
            plantarBoneCentroid(bones, *phalanges[ray], neutralBodies)
        );
    }
    phalanxCentroid = femScale(phalanxCentroid, 0.2f);
    const mr_float4 calcaneusCentroid =
        plantarBoneCentroid(bones, calcaneus, neutralBodies);
    const mr_float4 forward = femNormalized(
        femSubtract(phalanxCentroid, calcaneusCentroid),
        "plantar fascia foot longitudinal axis"
    );
    mr_float4 vertical{
        supportContacts.header.groundNormalX,
        supportContacts.header.groundNormalY,
        supportContacts.header.groundNormalZ, 0.0f,
    };
    vertical = femNormalized(vertical, "plantar fascia ground normal");
    mr_float4 medial = femSubtract(
        plantarBoneCentroid(bones, *phalanges[0u], neutralBodies),
        plantarBoneCentroid(bones, *phalanges[4u], neutralBodies)
    );
    medial = femSubtract(
        medial,
        femAdd(
            femScale(forward, femDot(medial, forward)),
            femScale(vertical, femDot(medial, vertical))
        )
    );
    medial = femNormalized(medial, "plantar fascia medial-lateral axis");
    const PlantarSurfacePatch calcanealPatch =
        selectCalcanealPlantarPatch(
            bones, calcaneus, neutralBodies, forward, vertical, medial
        );

    numi::matter::WorldSource worldSource;
    worldSource.environmentCount = 1u;
    worldSource.frameTimestep = timestepSeconds;
    worldSource.gravity = {0.0, 0.0, 0.0};
    worldSource.mixedSolver.newtonIterations = 10u;
    worldSource.mixedSolver.fgmresIterations = 28u;
    auto material = numi::matter::parseMatterFile(
        NUMI_HUMAN_PLANTAR_FASCIA_MATERIAL
    );
    require(material.succeeded(),
            "human plantar fascia Matter material did not parse");
    worldSource.materials.push_back(std::move(material.material));
    numi::matter::ObjectSource object;
    object.name = "bodyparts3d_reduced_plantar_aponeurosis_" + result.side;
    object.materialIndex = 0u;
    object.representation = numi::matter::Representation::fem;
    object.mixedFEM = false;
    object.deformableSelfContact = false;
    object.characteristicLength = 0.006;
    std::vector<NMNumiHumanTendonFEMNodeLoadGPU> nodeLoads;
    std::vector<NMNumiHumanTendonFEMNodeAnchorGPU> nodeAnchors;
    std::vector<std::array<std::uint32_t, 4u>> tetrahedra;
    std::vector<std::array<double, 3u>> restPoints;
    double scaleNumerator = 0.0;
    double scaleDenominator = 0.0;
    for (std::size_t ray = 0u; ray < phalanges.size(); ++ray) {
        PlantarFasciaBandAudit& audit = result.bands[ray];
        audit.ray = static_cast<std::uint32_t>(ray + 1u);
        audit.calcaneusStableId = kCalcaneusStableIds[sideIndex];
        audit.metatarsalStableId = kMetatarsalStableIds[sideIndex][ray];
        audit.proximalPhalanxStableId =
            kProximalPhalanxStableIds[sideIndex][ray];
        audit.publishedRestLengthMeters = kPublishedRestLengthsMeters[ray];
        audit.targetStiffnessNewtonsPerMillimeter =
            kStiffnessNewtonsPerMillimeter[ray];
        audit.calcanealPatchRadiusMeters = calcanealPatch.radiusMeters;
        const PlantarSurfacePatch distalPatch =
            selectProximalPhalanxPlantarPatch(
                bones, *phalanges[ray], neutralBodies,
                forward, vertical, medial
            );
        const PlantarSurfacePatch wrapPatch =
            selectMetatarsalHeadPlantarPatch(
                bones, *metatarsals[ray], neutralBodies,
                forward, vertical, medial
            );
        audit.metatarsalPatchRadiusMeters = wrapPatch.radiusMeters;
        audit.phalanxPatchRadiusMeters = distalPatch.radiusMeters;
        const double calcaneusToWrapMeters = femLength(
            femSubtract(wrapPatch.centroid, calcanealPatch.centroid));
        const double wrapToPhalanxMeters = femLength(
            femSubtract(distalPatch.centroid, wrapPatch.centroid));
        audit.bodypartsRestLengthMeters =
            calcaneusToWrapMeters + wrapToPhalanxMeters;
        require(audit.bodypartsRestLengthMeters > 0.09 &&
                    audit.bodypartsRestLengthMeters < 0.24 &&
                    calcaneusToWrapMeters > 0.07 &&
                    wrapToPhalanxMeters > 0.005 &&
                    wrapToPhalanxMeters < 0.06,
                "plantar fascia BodyParts3D band length is anatomically implausible");
        scaleNumerator += audit.bodypartsRestLengthMeters *
            audit.publishedRestLengthMeters;
        scaleDenominator += audit.publishedRestLengthMeters *
            audit.publishedRestLengthMeters;
        const double stiffnessNewtonsPerMeter =
            1000.0 * audit.targetStiffnessNewtonsPerMillimeter;
        const double areaSquareMeters = stiffnessNewtonsPerMeter *
            audit.bodypartsRestLengthMeters / kYoungModulusPascals;
        audit.calibratedCrossSectionSquareMillimeters =
            areaSquareMeters * 1.0e6;
        const float widthMeters = static_cast<float>(
            areaSquareMeters / static_cast<double>(kThicknessMeters)
        );
        require(widthMeters >= 0.003f && widthMeters <= 0.016f,
                "plantar fascia calibrated band width is implausible");
        const std::uint32_t wrapStation = std::clamp(
            static_cast<std::uint32_t>(std::lround(
                static_cast<double>(kSegmentCount) *
                calcaneusToWrapMeters / audit.bodypartsRestLengthMeters)),
            6u, kSegmentCount - 1u
        );
        audit.firstNode = static_cast<std::uint32_t>(object.femNodes.size());
        audit.wrapFirstNode = audit.firstNode + 4u * wrapStation;
        for (std::uint32_t station = 0u; station <= kSegmentCount; ++station) {
            std::array<mr_float4, 4u> stationPoints{};
            if (station == 0u) {
                stationPoints = calcanealPatch.points;
            } else if (station == wrapStation) {
                stationPoints = wrapPatch.points;
            } else if (station == kSegmentCount) {
                stationPoints = distalPatch.points;
            } else {
                const bool proximal = station < wrapStation;
                const float fraction = proximal
                    ? static_cast<float>(station) /
                        static_cast<float>(wrapStation)
                    : static_cast<float>(station - wrapStation) /
                        static_cast<float>(kSegmentCount - wrapStation);
                const mr_float4 firstCenter = proximal
                    ? calcanealPatch.centroid : wrapPatch.centroid;
                const mr_float4 secondCenter = proximal
                    ? wrapPatch.centroid : distalPatch.centroid;
                const mr_float4 center = femLerp(
                    firstCenter, secondCenter, fraction);
                const mr_float4 localAxis = femNormalized(
                    femSubtract(secondCenter, firstCenter),
                    "plantar fascia local pulley route axis");
                mr_float4 thicknessAxis = vertical;
                thicknessAxis = femSubtract(
                    thicknessAxis,
                    femScale(localAxis, femDot(thicknessAxis, localAxis))
                );
                thicknessAxis = femNormalized(
                    thicknessAxis, "plantar fascia local thickness axis");
                mr_float4 widthAxis = femNormalized(
                    femCross(thicknessAxis, localAxis),
                    "plantar fascia local width axis");
                if (femDot(widthAxis, medial) < 0.0f)
                    widthAxis = femScale(widthAxis, -1.0f);
                const float halfWidth = 0.5f * widthMeters;
                const float halfThickness = 0.5f * kThicknessMeters;
                stationPoints = {
                    femAdd(center, femAdd(
                        femScale(widthAxis, -halfWidth),
                        femScale(thicknessAxis, -halfThickness))),
                    femAdd(center, femAdd(
                        femScale(widthAxis, halfWidth),
                        femScale(thicknessAxis, -halfThickness))),
                    femAdd(center, femAdd(
                        femScale(widthAxis, halfWidth),
                        femScale(thicknessAxis, halfThickness))),
                    femAdd(center, femAdd(
                        femScale(widthAxis, -halfWidth),
                        femScale(thicknessAxis, halfThickness))),
                };
            }
            for (const mr_float4 point : stationPoints) {
                object.femNodes.push_back({point.x, point.y, point.z});
                restPoints.push_back({point.x, point.y, point.z});
                NMNumiHumanTendonFEMNodeLoadGPU load{};
                std::fill_n(load.endpointIndex, 4u, NM_INVALID_INDEX);
                nodeLoads.push_back(load);
                NMNumiHumanTendonFEMNodeAnchorGPU anchor{};
                anchor.bodyIndex = NM_INVALID_INDEX;
                nodeAnchors.push_back(anchor);
            }
        }
        audit.nodeCount = 4u * (kSegmentCount + 1u);
        const std::uint32_t first = audit.firstNode;
        for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
            const std::uint32_t originNode = first + corner;
            const std::uint32_t wrapNode = audit.wrapFirstNode + corner;
            const std::uint32_t distalNode =
                first + 4u * kSegmentCount + corner;
            object.femFixedNodes.push_back(originNode);
            object.femFixedNodes.push_back(wrapNode);
            object.femFixedNodes.push_back(distalNode);
            const mr_float4 originLocal = plantarAnchorLocalPoint(
                calcanealPatch.points[corner],
                neutralBodies[result.calcaneusBodyIndex]
            );
            nodeAnchors[originNode].bodyIndex = result.calcaneusBodyIndex;
            nodeAnchors[originNode].flags =
                NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
            nodeAnchors[originNode].localPoint = {
                originLocal.x, originLocal.y, originLocal.z, 0.0f};
            const mr_float4 wrapLocal = plantarAnchorLocalPoint(
                wrapPatch.points[corner],
                neutralBodies[result.calcaneusBodyIndex]
            );
            nodeAnchors[wrapNode].bodyIndex = result.calcaneusBodyIndex;
            nodeAnchors[wrapNode].flags =
                NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
            nodeAnchors[wrapNode].localPoint = {
                wrapLocal.x, wrapLocal.y, wrapLocal.z, 0.0f};
            const mr_float4 distalLocal = plantarAnchorLocalPoint(
                distalPatch.points[corner],
                neutralBodies[result.toesBodyIndex]
            );
            nodeAnchors[distalNode].bodyIndex = result.toesBodyIndex;
            nodeAnchors[distalNode].flags =
                NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE;
            nodeAnchors[distalNode].localPoint = {
                distalLocal.x, distalLocal.y, distalLocal.z, 0.0f};
        }
        const std::uint32_t firstTetrahedron =
            static_cast<std::uint32_t>(tetrahedra.size());
        for (std::uint32_t segment = 0u; segment < kSegmentCount; ++segment) {
            const std::uint32_t a = first + 4u * segment;
            const std::uint32_t b = a + 4u;
            std::array<std::array<std::uint32_t, 4u>, 5u> cells{{
                {{a + 0u, a + 1u, a + 3u, b + 0u}},
                {{a + 1u, a + 2u, a + 3u, b + 2u}},
                {{a + 1u, a + 3u, b + 0u, b + 2u}},
                {{a + 1u, b + 0u, b + 1u, b + 2u}},
                {{a + 3u, b + 0u, b + 2u, b + 3u}},
            }};
            for (auto cell : cells) {
                const double volume = femSignedTetrahedronVolume(
                    restPoints, cell
                );
                require(std::isfinite(volume) && std::abs(volume) > 1.0e-15,
                        "plantar fascia tetrahedron is degenerate");
                if (volume < 0.0) std::swap(cell[0u], cell[1u]);
                tetrahedra.push_back(cell);
                object.tetrahedra.push_back({cell});
            }
        }
        audit.tetrahedronCount = static_cast<std::uint32_t>(
            tetrahedra.size() - firstTetrahedron
        );
    }
    result.sourceLengthScale = scaleNumerator / scaleDenominator;
    require(std::isfinite(result.sourceLengthScale) &&
                result.sourceLengthScale >= 0.70 &&
                result.sourceLengthScale <= 1.30,
            "plantar fascia BodyParts3D/source resting-length scale is implausible");
    for (const PlantarFasciaBandAudit& audit : result.bands) {
        const double fitted = result.sourceLengthScale *
            audit.publishedRestLengthMeters;
        result.maximumRestLengthPatternRelativeResidual = std::max(
            result.maximumRestLengthPatternRelativeResidual,
            std::abs(audit.bodypartsRestLengthMeters - fitted) /
                std::max(1.0e-6, fitted)
        );
    }
    require(result.maximumRestLengthPatternRelativeResidual <= 0.20,
            "plantar fascia five-ray resting-length pattern disagrees with source");
    require(object.femFixedNodes.size() == 60u,
            "plantar fascia exact bone anchors are incomplete");
    object.femContactNodes = {object.femFixedNodes.front()};
    result.nodeCount = static_cast<std::uint32_t>(object.femNodes.size());
    result.tetrahedronCount = static_cast<std::uint32_t>(object.tetrahedra.size());
    worldSource.objects.push_back(std::move(object));
    numi::matter::CompileOptions compileOptions;
    compileOptions.maximumRateExponent = 0u;
    auto compiled = numi::matter::compileWorld(worldSource, compileOptions);
    std::string compileMessage;
    for (const numi::matter::Diagnostic& diagnostic : compiled.diagnostics)
        compileMessage += diagnostic.message + "; ";
    require(compiled.succeeded(),
            "plantar fascia world did not compile: " + compileMessage);
    numi::matter::Runtime runtime;
    const auto initialized = runtime.initialize(compiled.world, {
        .metallib = matterMetallib,
        .environmentCount = 1u,
        .captureEvents = true,
        .captureDiagnostics = true,
        .automaticIdentification = false,
        .adaptiveTransfer = false,
    });
    require(initialized.encoded && runtime.valid(),
            "could not initialize plantar fascia Matter runtime: " +
                initialized.message);
    result.deviceName = initialized.device;
    const auto initial = runtime.snapshot();
    require(initial.available && initial.femNodes.size() == result.nodeCount,
            "plantar fascia initial snapshot is unavailable");
    numi::matter::NumiHumanTendonFEMLoadAdapter adapter;
    require(adapter.initialize(runtime, {
                .nodeLoads = nodeLoads,
                .nodeAnchors = nodeAnchors,
                .endpointReplacements = {},
                .endpointCount = static_cast<std::uint32_t>(
                    muscles.tendonPayload.bindings.size()
                ),
                .environmentCount = 1u,
                .productionForceOwnerFraction = 0.0f,
            }, {.metallib = matterMetallib}),
            "could not initialize passive plantar fascia Human adapter");
    HumanTendonContinuumTransaction transaction{
        .program = adapter.program(),
        .runtime = &runtime,
        .initial = initial,
    };
    driven = integratePersistentMetalHumanState(
        model, muscles, supportContacts, jointEqualities,
        timestepSeconds, stepCount, 1.0, {}, false,
        true, false, true, &transaction,
        std::pair<std::uint32_t, double>{
            result.mtpQIndex, kQualificationMTPRadians},
        false
    );
    require(transaction.accepted.available &&
                transaction.accepted.femNodes.size() == result.nodeCount,
            "plantar fascia accepted snapshot is unavailable");
    const auto adapterDiagnostics = adapter.diagnostics();
    require(adapterDiagnostics.initialized &&
                adapterDiagnostics.abortCount == 1u &&
                adapterDiagnostics.encodedPassCount == 2u * stepCount,
            "plantar fascia adapter transaction accounting is incomplete: " +
                adapterDiagnostics.message);
    id<MTLBuffer> reactionBuffer = (__bridge id<MTLBuffer>)
        runtime.femConstraintReactionBuffer();
    require(reactionBuffer != nil,
            "plantar fascia fixed-node reaction buffer is unavailable");
    const NSUInteger reactionBytes = static_cast<NSUInteger>(
        result.nodeCount * sizeof(nm_float4));
    id<MTLBuffer> reactionReadback = [reactionBuffer.device
        newBufferWithLength:reactionBytes options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> reactionQueue = [reactionBuffer.device newCommandQueue];
    id<MTLCommandBuffer> reactionCommand = [reactionQueue commandBuffer];
    id<MTLBlitCommandEncoder> reactionBlit =
        [reactionCommand blitCommandEncoder];
    require(reactionReadback != nil && reactionQueue != nil &&
                reactionCommand != nil && reactionBlit != nil,
            "plantar fascia reaction readback could not be allocated");
    [reactionBlit copyFromBuffer:reactionBuffer sourceOffset:0u
                        toBuffer:reactionReadback destinationOffset:0u
                            size:reactionBytes];
    [reactionBlit endEncoding];
    [reactionCommand commit];
    [reactionCommand waitUntilCompleted];
    require(reactionCommand.status == MTLCommandBufferStatusCompleted,
            "plantar fascia reaction readback failed");
    const auto* reactions =
        static_cast<const nm_float4*>(reactionReadback.contents);
    std::vector<std::array<double, 3u>> acceptedPoints;
    acceptedPoints.reserve(result.nodeCount);
    for (std::uint32_t node = 0u; node < result.nodeCount; ++node) {
        const auto& value = transaction.accepted.femNodes[node].positionAndMass;
        acceptedPoints.push_back({value.x, value.y, value.z});
        if ((nodeAnchors[node].flags &
                NM_NUMI_HUMAN_TENDON_FEM_NODE_ANCHOR_ACTIVE) == 0u) {
            result.maximumFreeNodeDisplacementMeters = std::max(
                result.maximumFreeNodeDisplacementMeters,
                static_cast<double>(femLength(femSubtract(
                    {value.x, value.y, value.z, 1.0f},
                    {static_cast<float>(restPoints[node][0u]),
                     static_cast<float>(restPoints[node][1u]),
                     static_cast<float>(restPoints[node][2u]), 1.0f}
                )))
            );
        }
    }
    for (const auto& tetrahedron : tetrahedra) {
        const double restVolume = femSignedTetrahedronVolume(
            restPoints, tetrahedron
        );
        const double currentVolume = femSignedTetrahedronVolume(
            acceptedPoints, tetrahedron
        );
        const double determinant = currentVolume / restVolume;
        require(std::isfinite(determinant),
                "plantar fascia accepted determinant is nonfinite");
        result.minimumDeterminant = std::min(
            result.minimumDeterminant, determinant
        );
        result.maximumDeterminant = std::max(
            result.maximumDeterminant, determinant
        );
    }
    metalrobo::MetalArticulatedOperatorResult acceptedPose;
    const auto acceptedPoseDiagnostics = metalrobo::runMetalArticulatedOperator(
        model, {
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = 0u,
            .q = driven.q,
            .points = {},
        }, acceptedPose, poseConfig
    );
    require(acceptedPoseDiagnostics.succeeded() &&
                acceptedPoseDiagnostics.published,
            "plantar fascia accepted Human pose failed");
    const std::vector<MRBodyStateGPU> acceptedBodies =
        visualBodyStates(model, acceptedPose.bodyPoses);
    for (PlantarFasciaBandAudit& audit : result.bands) {
        mr_float4 originReaction{};
        mr_float4 wrapReaction{};
        mr_float4 distalReaction{};
        mr_float4 acceptedOriginCentroid{};
        mr_float4 acceptedWrapCentroid{};
        mr_float4 acceptedDistalCentroid{};
        const std::uint32_t distalFirst =
            audit.firstNode + audit.nodeCount - 4u;
        for (std::uint32_t corner = 0u; corner < 4u; ++corner) {
            const std::uint32_t originNode = audit.firstNode + corner;
            const std::uint32_t wrapNode = audit.wrapFirstNode + corner;
            const std::uint32_t distalNode = distalFirst + corner;
            originReaction = femAdd(originReaction, {
                reactions[originNode].x, reactions[originNode].y,
                reactions[originNode].z, 0.0f});
            wrapReaction = femAdd(wrapReaction, {
                reactions[wrapNode].x, reactions[wrapNode].y,
                reactions[wrapNode].z, 0.0f});
            distalReaction = femAdd(distalReaction, {
                reactions[distalNode].x, reactions[distalNode].y,
                reactions[distalNode].z, 0.0f});
            const auto target = [&](const std::uint32_t node) {
                const auto& anchor = nodeAnchors[node];
                const MRBodyStateGPU& body = acceptedBodies[anchor.bodyIndex];
                return femAdd(
                    body.position,
                    rotatePoint(body.orientation, {
                        anchor.localPoint.x, anchor.localPoint.y,
                        anchor.localPoint.z, 0.0f})
                );
            };
            const mr_float4 originTarget = target(originNode);
            const mr_float4 wrapTarget = target(wrapNode);
            const mr_float4 distalTarget = target(distalNode);
            acceptedOriginCentroid = femAdd(
                acceptedOriginCentroid, originTarget
            );
            acceptedWrapCentroid = femAdd(
                acceptedWrapCentroid, wrapTarget
            );
            acceptedDistalCentroid = femAdd(
                acceptedDistalCentroid, distalTarget
            );
            for (const auto [node, targetPoint] : {
                     std::pair<std::uint32_t, mr_float4>{originNode, originTarget},
                     std::pair<std::uint32_t, mr_float4>{wrapNode, wrapTarget},
                     std::pair<std::uint32_t, mr_float4>{distalNode, distalTarget},
                 }) {
                const auto& accepted =
                    transaction.accepted.femNodes[node].positionAndMass;
                result.maximumAnchorTargetResidualMeters = std::max(
                    result.maximumAnchorTargetResidualMeters,
                    static_cast<double>(femLength(femSubtract(
                        {accepted.x, accepted.y, accepted.z, 1.0f},
                        targetPoint
                    )))
                );
            }
        }
        acceptedOriginCentroid = femScale(acceptedOriginCentroid, 0.25f);
        acceptedWrapCentroid = femScale(acceptedWrapCentroid, 0.25f);
        acceptedDistalCentroid = femScale(acceptedDistalCentroid, 0.25f);
        audit.acceptedExtensionMeters =
            femLength(femSubtract(
                acceptedWrapCentroid, acceptedOriginCentroid)) +
            femLength(femSubtract(
                acceptedDistalCentroid, acceptedWrapCentroid)) -
            audit.bodypartsRestLengthMeters;
        audit.calcanealReactionResultantNewtons = femLength(originReaction);
        audit.metatarsalReactionResultantNewtons = femLength(wrapReaction);
        audit.proximalFootBodyReactionResultantNewtons = femLength(
            femAdd(originReaction, wrapReaction));
        audit.phalanxReactionResultantNewtons = femLength(distalReaction);
    }
    result.completedSteps = driven.persistentCompletedSteps;
    result.maximumConfigurationDeltaFromSourceJT =
        driven.tendonContinuumMaximumQDelta;
    result.maximumVelocityDeltaFromSourceJT =
        driven.tendonContinuumMaximumVDelta;
    result.rollbackVerified = transaction.rollbackVerified;
    result.replayVerified = transaction.replayVerified;
    const bool allBandsTransferred = std::all_of(
        result.bands.begin(), result.bands.end(),
        [](const PlantarFasciaBandAudit& audit) {
            return audit.acceptedExtensionMeters > 1.0e-6 &&
                audit.calcanealReactionResultantNewtons > 1.0e-5 &&
                audit.metatarsalReactionResultantNewtons > 1.0e-5 &&
                audit.proximalFootBodyReactionResultantNewtons > 1.0e-5 &&
                audit.phalanxReactionResultantNewtons > 1.0e-5;
        });
    result.available = allBandsTransferred &&
        result.nodeCount == 220u && result.tetrahedronCount == 250u &&
        result.minimumDeterminant >= 0.35 &&
        result.maximumDeterminant <= 2.5 &&
        result.maximumFreeNodeDisplacementMeters > 0.0 &&
        result.maximumAnchorTargetResidualMeters <= 2.0e-5 &&
        result.maximumVelocityDeltaFromSourceJT > 1.0e-9 &&
        result.rollbackVerified && result.replayVerified;
    require(
        result.available,
        "plantar fascia live Human/Matter qualification did not close: " +
            result.side + " min_J=" + std::to_string(result.minimumDeterminant) +
            " max_J=" + std::to_string(result.maximumDeterminant) +
            " free_displacement_m=" +
            std::to_string(result.maximumFreeNodeDisplacementMeters) +
            " anchor_residual_m=" +
            std::to_string(result.maximumAnchorTargetResidualMeters) +
            " q_delta=" +
            std::to_string(result.maximumConfigurationDeltaFromSourceJT) +
            " v_delta=" +
            std::to_string(result.maximumVelocityDeltaFromSourceJT) +
            " ray1_extension_m=" +
            std::to_string(result.bands[0u].acceptedExtensionMeters) +
            " ray1_calcaneal_reaction_n=" +
            std::to_string(
                result.bands[0u].calcanealReactionResultantNewtons) +
            " ray1_metatarsal_reaction_n=" +
            std::to_string(
                result.bands[0u].metatarsalReactionResultantNewtons) +
            " ray1_phalanx_reaction_n=" +
            std::to_string(
                result.bands[0u].phalanxReactionResultantNewtons)
    );
    driven.tendonContinuumPassiveReactionOnly = true;
    return result;
}

PlantarFasciaSideAudit runPlantarFasciaTensileSide(
    const LoadedBones& bones,
    const metalrobo::EngineModel& model,
    const LoadedSupportContacts& supportContacts,
    const LoadedJointEqualities& jointEqualities,
    const double timestepSeconds,
    const std::size_t sideIndex
) {
    require(sideIndex < 2u && std::isfinite(timestepSeconds) &&
                timestepSeconds >= 1.0e-6 && timestepSeconds <= 1.0e-3,
            "plantar fascia tensile-law request is invalid");
    constexpr std::array<std::uint32_t, 2u> kCalcaneusStableIds{6u, 7u};
    constexpr std::array<std::uint32_t, 2u> kCalcaneusBodyIndices{138u, 152u};
    constexpr std::array<std::uint32_t, 2u> kToesBodyIndices{139u, 153u};
    constexpr std::array<std::uint32_t, 2u> kMTPQIndices{111u, 125u};
    constexpr std::array<std::array<std::uint32_t, 5u>, 2u>
        kMetatarsalStableIds{{
            {{126u, 127u, 128u, 129u, 130u}},
            {{145u, 146u, 147u, 148u, 149u}},
        }};
    constexpr std::array<std::array<std::uint32_t, 5u>, 2u>
        kProximalPhalanxStableIds{{
            {{131u, 133u, 136u, 139u, 142u}},
            {{150u, 152u, 155u, 158u, 161u}},
        }};
    constexpr std::array<double, 5u> kPublishedRestLengthsMeters{
        0.151, 0.149, 0.148, 0.140, 0.131,
    };
    constexpr std::array<double, 5u> kStiffnessNewtonsPerMillimeter{
        60.0, 50.0, 50.0, 20.0, 20.0,
    };
    constexpr double kYoungModulusPascals = 350.0e6;
    constexpr double kQualificationMTPRadians = -0.10;

    PlantarFasciaSideAudit result;
    result.side = sideIndex == 0u ? "right" : "left";
    result.calcaneusBodyIndex = kCalcaneusBodyIndices[sideIndex];
    result.toesBodyIndex = kToesBodyIndices[sideIndex];
    result.mtpQIndex = kMTPQIndices[sideIndex];
    result.qualificationMTPRadians = kQualificationMTPRadians;

    GroundAlignedSupport neutral = makeGroundAlignedSupport(
        model, supportContacts);
    require(result.mtpQIndex < neutral.q.size(),
            "plantar fascia MTP coordinate is unavailable");
    neutral.q[result.mtpQIndex] = 0.0;
    double neutralProjection = 0.0;
    require(metalrobo::projectNumiHumanJointEqualities(
                jointEqualities.payload.records, neutral.q,
                &neutralProjection).succeeded(),
            "plantar fascia neutral equality projection failed");
    std::vector<double> qualifiedQ = neutral.q;
    qualifiedQ[result.mtpQIndex] = kQualificationMTPRadians;
    double qualifiedProjection = 0.0;
    require(metalrobo::projectNumiHumanJointEqualities(
                jointEqualities.payload.records, qualifiedQ,
                &qualifiedProjection).succeeded(),
            "plantar fascia dorsiflexion equality projection failed");

    const auto metalPose = [&](const std::span<const double> q,
                               std::string& deviceName) {
        metalrobo::MetalArticulatedOperatorResult pose;
        metalrobo::MetalArticulatedOperatorConfig config;
        config.pointJacobiansOnly = true;
        const std::vector<float> packed = packMetalConfiguration(q);
        const auto diagnostics = metalrobo::runMetalArticulatedOperator(
            model, {
                .articulationIndex = 0u,
                .environmentCount = 1u,
                .pointCount = 0u,
                .q = packed,
                .points = {},
            }, pose, config);
        require(diagnostics.succeeded() && diagnostics.dispatched &&
                    diagnostics.published &&
                    diagnostics.successfulEnvironmentCount == 1u,
                "plantar fascia Metal pose failed: " + diagnostics.message);
        deviceName = diagnostics.deviceName;
        return visualBodyStates(model, pose.bodyPoses);
    };
    std::string neutralDevice;
    const std::vector<MRBodyStateGPU> neutralBodies =
        metalPose(neutral.q, neutralDevice);
    const std::vector<MRBodyStateGPU> qualifiedBodies =
        metalPose(qualifiedQ, result.deviceName);
    require(result.deviceName == neutralDevice &&
                result.calcaneusBodyIndex < qualifiedBodies.size() &&
                result.toesBodyIndex < qualifiedBodies.size(),
            "plantar fascia Metal poses are incompatible");

    const BoneRecord& calcaneus = plantarBone(
        bones, kCalcaneusStableIds[sideIndex], result.calcaneusBodyIndex);
    std::array<const BoneRecord*, 5u> metatarsals{};
    std::array<const BoneRecord*, 5u> phalanges{};
    mr_float4 phalanxCentroid{};
    for (std::size_t ray = 0u; ray < 5u; ++ray) {
        metatarsals[ray] = &plantarBone(
            bones, kMetatarsalStableIds[sideIndex][ray],
            result.calcaneusBodyIndex);
        phalanges[ray] = &plantarBone(
            bones, kProximalPhalanxStableIds[sideIndex][ray],
            result.toesBodyIndex);
        phalanxCentroid = femAdd(
            phalanxCentroid,
            plantarBoneCentroid(bones, *phalanges[ray], neutralBodies));
    }
    phalanxCentroid = femScale(phalanxCentroid, 0.2f);
    const mr_float4 calcaneusCentroid =
        plantarBoneCentroid(bones, calcaneus, neutralBodies);
    const mr_float4 forward = femNormalized(
        femSubtract(phalanxCentroid, calcaneusCentroid),
        "plantar fascia longitudinal axis");
    mr_float4 vertical{
        supportContacts.header.groundNormalX,
        supportContacts.header.groundNormalY,
        supportContacts.header.groundNormalZ, 0.0f};
    vertical = femNormalized(vertical, "plantar fascia ground normal");
    mr_float4 medial = femSubtract(
        plantarBoneCentroid(bones, *phalanges[0u], neutralBodies),
        plantarBoneCentroid(bones, *phalanges[4u], neutralBodies));
    medial = femSubtract(
        medial,
        femAdd(
            femScale(forward, femDot(medial, forward)),
            femScale(vertical, femDot(medial, vertical))));
    medial = femNormalized(medial, "plantar fascia mediolateral axis");
    const PlantarSurfacePatch calcanealPatch =
        selectCalcanealPlantarPatch(
            bones, calcaneus, neutralBodies, forward, vertical, medial);
    const auto mtpJoint = std::find_if(
        model.joints.begin(), model.joints.end(),
        [&result](const MRJointDescriptorGPU& joint) {
            return joint.childBody == result.toesBodyIndex;
        });
    require(mtpJoint != model.joints.end() &&
                mtpJoint->parentBody == result.calcaneusBodyIndex,
            "plantar fascia source MTP joint ownership is unavailable");
    const MRBodyStateGPU& mtpParentBody =
        neutralBodies[mtpJoint->parentBody];
    const mr_float4 mtpJointAxis = femNormalized(
        rotatePoint(
            mtpParentBody.orientation,
            rotatePoint(mtpJoint->parentRotation, mtpJoint->axis0)),
        "plantar fascia source MTP joint axis");

    metalrobo::ArticulatedDynamicsConfig dynamicsConfig;
    dynamicsConfig.gravity = {0.0, 0.0, 0.0};
    dynamicsConfig.timestep = timestepSeconds;
    dynamicsConfig.implicitPassiveDofDamping = true;
    const std::vector<double> zeroVelocity(model.world.nv, 0.0);
    std::vector<double> passiveGeneralizedForce(model.world.nv, 0.0);
    double scaleNumerator = 0.0;
    double scaleDenominator = 0.0;

    const auto qualifiedWorldPoint = [&](const mr_float4 local,
                                         const std::uint32_t bodyIndex) {
        const MRBodyStateGPU& body = qualifiedBodies[bodyIndex];
        return femAdd(
            body.position, rotatePoint(body.orientation, local));
    };
    for (std::size_t ray = 0u; ray < 5u; ++ray) {
        PlantarFasciaBandAudit& audit = result.bands[ray];
        audit.ray = static_cast<std::uint32_t>(ray + 1u);
        audit.calcaneusStableId = kCalcaneusStableIds[sideIndex];
        audit.metatarsalStableId = kMetatarsalStableIds[sideIndex][ray];
        audit.proximalPhalanxStableId =
            kProximalPhalanxStableIds[sideIndex][ray];
        audit.publishedRestLengthMeters = kPublishedRestLengthsMeters[ray];
        audit.targetStiffnessNewtonsPerMillimeter =
            kStiffnessNewtonsPerMillimeter[ray];
        const PlantarSurfacePatch wrapPatch =
            selectMetatarsalHeadPlantarPatch(
                bones, *metatarsals[ray], neutralBodies,
                forward, vertical, medial);
        const PlantarSurfacePatch distalPatch =
            selectProximalPhalanxPlantarPatch(
                bones, *phalanges[ray], neutralBodies,
                forward, vertical, medial);
        audit.calcanealPatchRadiusMeters = calcanealPatch.radiusMeters;
        audit.metatarsalPatchRadiusMeters = wrapPatch.radiusMeters;
        float minimumMetatarsalForward =
            std::numeric_limits<float>::infinity();
        float maximumMetatarsalForward =
            -std::numeric_limits<float>::infinity();
        for (std::uint32_t offset = 0u;
             offset < metatarsals[ray]->vertexCount; ++offset) {
            const mr_float4 point = boneVertexWorld(
                *metatarsals[ray],
                bones.vertices[metatarsals[ray]->firstVertex + offset],
                neutralBodies[result.calcaneusBodyIndex]);
            const float projection = femDot(point, forward);
            minimumMetatarsalForward = std::min(
                minimumMetatarsalForward, projection);
            maximumMetatarsalForward = std::max(
                maximumMetatarsalForward, projection);
        }
        const float distalHeadThreshold =
            minimumMetatarsalForward + 0.80f *
                (maximumMetatarsalForward - minimumMetatarsalForward);
        mr_float4 distalHeadCentroid{};
        std::uint32_t distalHeadVertexCount = 0u;
        for (std::uint32_t offset = 0u;
             offset < metatarsals[ray]->vertexCount; ++offset) {
            const mr_float4 point = boneVertexWorld(
                *metatarsals[ray],
                bones.vertices[metatarsals[ray]->firstVertex + offset],
                neutralBodies[result.calcaneusBodyIndex]);
            if (femDot(point, forward) < distalHeadThreshold) continue;
            distalHeadCentroid = femAdd(distalHeadCentroid, point);
            ++distalHeadVertexCount;
        }
        require(distalHeadVertexCount >= 8u,
                "plantar fascia metatarsal head envelope is undersampled");
        distalHeadCentroid = femScale(
            distalHeadCentroid,
            1.0f / static_cast<float>(distalHeadVertexCount));
        for (std::uint32_t offset = 0u;
             offset < metatarsals[ray]->vertexCount; ++offset) {
            const mr_float4 point = boneVertexWorld(
                *metatarsals[ray],
                bones.vertices[metatarsals[ray]->firstVertex + offset],
                neutralBodies[result.calcaneusBodyIndex]);
            if (femDot(point, forward) < distalHeadThreshold) continue;
            const mr_float4 pulleyOffset = femSubtract(
                point, distalHeadCentroid);
            const double radius = femLength(femSubtract(
                pulleyOffset,
                femScale(
                    mtpJointAxis, femDot(pulleyOffset, mtpJointAxis))));
            audit.metatarsalPulleyRadiusMeters = std::max(
                audit.metatarsalPulleyRadiusMeters, radius);
        }
        require(audit.metatarsalPulleyRadiusMeters > 0.004 &&
                    audit.metatarsalPulleyRadiusMeters < 0.035,
                "plantar fascia metatarsal pulley radius is implausible: " +
                    result.side + " ray=" + std::to_string(audit.ray) +
                    " radius_m=" +
                    std::to_string(audit.metatarsalPulleyRadiusMeters));
        audit.phalanxPatchRadiusMeters = distalPatch.radiusMeters;
        audit.bodypartsRestLengthMeters =
            femLength(femSubtract(
                wrapPatch.centroid, calcanealPatch.centroid)) +
            femLength(femSubtract(
                distalPatch.centroid, wrapPatch.centroid));
        require(audit.bodypartsRestLengthMeters > 0.09 &&
                    audit.bodypartsRestLengthMeters < 0.24,
                "plantar fascia routed rest length is implausible");
        scaleNumerator += audit.bodypartsRestLengthMeters *
            audit.publishedRestLengthMeters;
        scaleDenominator += audit.publishedRestLengthMeters *
            audit.publishedRestLengthMeters;
        const double stiffnessNewtonsPerMeter = 1000.0 *
            audit.targetStiffnessNewtonsPerMillimeter;
        audit.calibratedCrossSectionSquareMillimeters =
            stiffnessNewtonsPerMeter * audit.bodypartsRestLengthMeters /
            kYoungModulusPascals * 1.0e6;

        const mr_float4 originLocal = plantarAnchorLocalPoint(
            calcanealPatch.centroid,
            neutralBodies[result.calcaneusBodyIndex]);
        const mr_float4 wrapLocal = plantarAnchorLocalPoint(
            wrapPatch.centroid,
            neutralBodies[result.calcaneusBodyIndex]);
        const mr_float4 distalLocal = plantarAnchorLocalPoint(
            distalPatch.centroid,
            neutralBodies[result.toesBodyIndex]);
        const mr_float4 origin = qualifiedWorldPoint(
            originLocal, result.calcaneusBodyIndex);
        const mr_float4 wrap = qualifiedWorldPoint(
            wrapLocal, result.calcaneusBodyIndex);
        const mr_float4 distal = qualifiedWorldPoint(
            distalLocal, result.toesBodyIndex);
        const double qualifiedLength =
            femLength(femSubtract(wrap, origin)) +
            femLength(femSubtract(distal, wrap)) +
            audit.metatarsalPulleyRadiusMeters *
                std::abs(kQualificationMTPRadians);
        audit.acceptedExtensionMeters =
            qualifiedLength - audit.bodypartsRestLengthMeters;
        require(audit.acceptedExtensionMeters > 1.0e-6,
                "plantar fascia dorsiflexion did not engage the windlass: " +
                    result.side + " ray=" + std::to_string(audit.ray) +
                    " extension_m=" +
                    std::to_string(audit.acceptedExtensionMeters));
        audit.tensionNewtons = stiffnessNewtonsPerMeter *
            audit.acceptedExtensionMeters;
        result.totalTensionNewtons += audit.tensionNewtons;
        const mr_float4 proximalDirection = femNormalized(
            femSubtract(wrap, origin),
            "plantar fascia proximal tensile direction");
        const mr_float4 distalDirection = femNormalized(
            femSubtract(distal, wrap),
            "plantar fascia distal tensile direction");
        const mr_float4 originForce = femScale(
            proximalDirection, static_cast<float>(audit.tensionNewtons));
        const mr_float4 wrapForce = femScale(
            femSubtract(distalDirection, proximalDirection),
            static_cast<float>(audit.tensionNewtons));
        const mr_float4 distalForce = femScale(
            distalDirection, static_cast<float>(-audit.tensionNewtons));
        const mr_float4 forceResidual = femAdd(
            femAdd(originForce, wrapForce), distalForce);
        const mr_float4 momentResidual = femAdd(
            femAdd(femCross(origin, originForce),
                   femCross(wrap, wrapForce)),
            femCross(distal, distalForce));
        audit.forceClosureResidualNewtons = femLength(forceResidual);
        audit.momentClosureResidualNewtonMeters = femLength(momentResidual);
        result.maximumForceClosureResidualNewtons = std::max(
            result.maximumForceClosureResidualNewtons,
            audit.forceClosureResidualNewtons);
        result.maximumMomentClosureResidualNewtonMeters = std::max(
            result.maximumMomentClosureResidualNewtonMeters,
            audit.momentClosureResidualNewtonMeters);
        audit.calcanealReactionResultantNewtons = femLength(originForce);
        audit.metatarsalReactionResultantNewtons = femLength(wrapForce);
        audit.proximalFootBodyReactionResultantNewtons = femLength(
            femAdd(originForce, wrapForce));
        audit.phalanxReactionResultantNewtons = femLength(distalForce);

        const std::array<metalrobo::ArticulatedPointQuery, 3u> queries{{
            {result.calcaneusBodyIndex,
             {originLocal.x, originLocal.y, originLocal.z}},
            {result.calcaneusBodyIndex,
             {wrapLocal.x, wrapLocal.y, wrapLocal.z}},
            {result.toesBodyIndex,
             {distalLocal.x, distalLocal.y, distalLocal.z}},
        }};
        std::array<metalrobo::ArticulatedPointKinematics, 3u> kinematics{};
        std::vector<double> jacobians(
            queries.size() * 3u * model.world.nv, 0.0);
        const auto jacobianDiagnostics =
            metalrobo::computeArticulatedPointJacobians(
                model, 0u, qualifiedQ, zeroVelocity, queries,
                kinematics, jacobians, dynamicsConfig);
        require(jacobianDiagnostics.succeeded(),
                "plantar fascia exact point Jacobians failed");
        const std::array<mr_float4, 3u> forces{
            originForce, wrapForce, distalForce};
        for (std::size_t point = 0u; point < forces.size(); ++point) {
            for (std::size_t axis = 0u; axis < 3u; ++axis) {
                const double component = axis == 0u ? forces[point].x
                    : (axis == 1u ? forces[point].y : forces[point].z);
                for (std::size_t dof = 0u; dof < model.world.nv; ++dof) {
                    passiveGeneralizedForce[dof] += component * jacobians[
                        (point * 3u + axis) * model.world.nv + dof];
                }
            }
        }
    }
    result.sourceLengthScale = scaleNumerator / scaleDenominator;
    for (const PlantarFasciaBandAudit& audit : result.bands) {
        const double fitted = result.sourceLengthScale *
            audit.publishedRestLengthMeters;
        result.maximumRestLengthPatternRelativeResidual = std::max(
            result.maximumRestLengthPatternRelativeResidual,
            std::abs(audit.bodypartsRestLengthMeters - fitted) /
                std::max(1.0e-6, fitted));
    }
    require(result.sourceLengthScale >= 0.70 &&
                result.sourceLengthScale <= 1.30 &&
                result.maximumRestLengthPatternRelativeResidual <= 0.20,
            "plantar fascia BodyParts3D five-ray pattern disagrees with source");
    std::vector<double> passiveQ = qualifiedQ;
    std::vector<double> loadedQ = qualifiedQ;
    std::vector<double> passiveV(model.world.nv, 0.0);
    std::vector<double> loadedV(model.world.nv, 0.0);
    const std::vector<double> zeroForce(model.world.nv, 0.0);
    require(metalrobo::integrateArticulatedState(
                model, 0u, passiveQ, passiveV, zeroForce, {},
                dynamicsConfig).succeeded() &&
                metalrobo::integrateArticulatedState(
                    model, 0u, loadedQ, loadedV,
                    passiveGeneralizedForce, {}, dynamicsConfig).succeeded(),
            "plantar fascia articulated response integration failed");
    std::vector<double> replayQ = qualifiedQ;
    std::vector<double> replayV(model.world.nv, 0.0);
    require(metalrobo::integrateArticulatedState(
                model, 0u, replayQ, replayV,
                passiveGeneralizedForce, {}, dynamicsConfig).succeeded(),
            "plantar fascia deterministic replay failed");
    result.replayVerified =
        std::memcmp(replayQ.data(), loadedQ.data(),
                    loadedQ.size() * sizeof(double)) == 0 &&
        std::memcmp(replayV.data(), loadedV.data(),
                    loadedV.size() * sizeof(double)) == 0;
    for (std::size_t index = 0u; index < loadedQ.size(); ++index) {
        result.maximumConfigurationDeltaFromSourceJT = std::max(
            result.maximumConfigurationDeltaFromSourceJT,
            std::abs(loadedQ[index] - passiveQ[index]));
    }
    for (std::size_t index = 0u; index < loadedV.size(); ++index) {
        result.maximumVelocityDeltaFromSourceJT = std::max(
            result.maximumVelocityDeltaFromSourceJT,
            std::abs(loadedV[index] - passiveV[index]));
    }
    result.completedSteps = 1u;
    result.available =
        result.totalTensionNewtons > 0.0 &&
        result.maximumForceClosureResidualNewtons <= 1.0e-4 &&
        result.maximumMomentClosureResidualNewtonMeters <= 1.0e-5 &&
        result.maximumConfigurationDeltaFromSourceJT > 1.0e-12 &&
        result.maximumVelocityDeltaFromSourceJT > 1.0e-9 &&
        result.replayVerified;
    require(result.available,
            "plantar fascia reduced tensile-law qualification did not close");
    return result;
}

GeometryRange appendPectoralisFasciaGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const PectoralisFasciaVisual& fascia
) {
    require(!fascia.anatomicalSurfaceNodes.empty() &&
                !fascia.anatomicalSurfaceTriangles.empty(),
            "pectoralis fascia has no exact anatomical presentation surface");
    const auto& nodes = fascia.anatomicalSurfaceNodes;
    const auto& triangles = fascia.anatomicalSurfaceTriangles;
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f};
    result.maximum = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f};
    std::vector<mr_float4> normals(nodes.size(), {0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<mr_float4> firstFaceNormals(nodes.size(), {0.0f, 0.0f, 0.0f, 0.0f});
    for (const auto& triangle : triangles) {
        const mr_float4 first = femSubtract(nodes[triangle[1]], nodes[triangle[0]]);
        const mr_float4 second = femSubtract(nodes[triangle[2]], nodes[triangle[0]]);
        const mr_float4 normal = femCross(first, second);
        for (const std::uint32_t node : triangle) {
            normals[node].x += normal.x;
            normals[node].y += normal.y;
            normals[node].z += normal.z;
            if (femDot(firstFaceNormals[node], firstFaceNormals[node]) <= 1.0e-12f &&
                femDot(normal, normal) > 1.0e-12f) {
                firstFaceNormals[node] = normal;
            }
        }
    }
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t index = 0u; index < nodes.size(); ++index) {
        const mr_float4 position = nodes[index];
        mr_float4 normal = normals[index];
        float normalSquared = femDot(normal, normal);
        if (!std::isfinite(normalSquared) || normalSquared <= 1.0e-12f) {
            normal = firstFaceNormals[index];
            normalSquared = femDot(normal, normal);
        }
        if (!std::isfinite(normalSquared) || normalSquared <= 1.0e-12f) {
            normal = fascia.anatomicalSurfaceRestNormals[index];
            normalSquared = femDot(normal, normal);
        }
        if (!std::isfinite(normalSquared) || normalSquared <= 1.0e-12f) {
            // Unreferenced source vertices are retained for provenance but do
            // not contribute an index. Their normal is never rasterized.
            normal = {0.0f, 0.0f, 1.0f, 0.0f};
        }
        normal = femNormalized(normal, "pectoralis fascia surface normal");
        normal.w = 1.0f;
        pack.vertices.push_back({position, normal, normalTangent(normal),
            {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}});
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    for (const auto& triangle : triangles) {
        pack.indices.insert(pack.indices.end(), {
            vertexBase + triangle[0], vertexBase + triangle[1], vertexBase + triangle[2],
        });
    }
    result.indexCount = static_cast<std::uint32_t>(pack.indices.size()) - result.firstIndex;
    return result;
}

mr_float4 passiveFEMMappedPoint(
    const PassiveFEMTissueVisual& tissue, const mr_float4& point
) {
    const mr_float4 restFirst = femRingCenter(tissue.restNodes, 0u);
    const mr_float4 restLast = femRingCenter(tissue.restNodes, 3u);
    const mr_float4 restAxisVector = femSubtract(restLast, restFirst);
    const float restLengthSquared = femDot(restAxisVector, restAxisVector);
    require(restLengthSquared > 1.0e-8f, "passive FEM tissue rest axis is degenerate");
    const float fraction = std::clamp(
        femDot(femSubtract(point, restFirst), restAxisVector) / restLengthSquared,
        0.0f, 1.0f
    );
    const float ringCoordinate = fraction * 3.0f;
    const std::uint32_t firstRing = std::min(
        static_cast<std::uint32_t>(std::floor(ringCoordinate)), 2u
    );
    const float localFraction = ringCoordinate - static_cast<float>(firstRing);
    const mr_float4 restCenter = femLerp(
        femRingCenter(tissue.restNodes, firstRing),
        femRingCenter(tissue.restNodes, firstRing + 1u), localFraction
    );
    const mr_float4 currentCenter = femLerp(
        femRingCenter(tissue.nodes, firstRing),
        femRingCenter(tissue.nodes, firstRing + 1u), localFraction
    );
    const float restRadius = femRingRadius(tissue.restNodes, firstRing) * (1.0f - localFraction) +
        femRingRadius(tissue.restNodes, firstRing + 1u) * localFraction;
    const float currentRadius = femRingRadius(tissue.nodes, firstRing) * (1.0f - localFraction) +
        femRingRadius(tissue.nodes, firstRing + 1u) * localFraction;
    return femAdd(currentCenter, femScale(
        femSubtract(point, restCenter), currentRadius / restRadius
    ));
}

GeometryRange appendPassiveFEMMappedSoftTissueGeometry(
    metalrobo::VisualAssetPackV2& pack,
    const LoadedSoftTissues& tissues,
    const SoftTissueRecord& source,
    const std::span<const MRBodyStateGPU> restBodies,
    const PassiveFEMTissueVisual& femTissue
) {
    GeometryRange result;
    result.firstIndex = static_cast<std::uint32_t>(pack.indices.size());
    result.minimum = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 1.0f};
    result.maximum = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 1.0f};
    const std::uint32_t vertexBase = static_cast<std::uint32_t>(pack.vertices.size());
    for (std::uint32_t offset = 0u; offset < source.vertexCount; ++offset) {
        const SoftTissueVertex& vertex = tissues.vertices[source.firstVertex + offset];
        const mr_float4 position = passiveFEMMappedPoint(
            femTissue, softTissueVertexBlendedWorld(source, vertex, restBodies)
        );
        mr_float4 normal = softTissueVertexBlendedNormalWorld(source, vertex, restBodies);
        normal.w = 1.0f;
        pack.vertices.push_back({position, normal, normalTangent(normal),
            {0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}});
        result.minimum.x = std::min(result.minimum.x, position.x);
        result.minimum.y = std::min(result.minimum.y, position.y);
        result.minimum.z = std::min(result.minimum.z, position.z);
        result.maximum.x = std::max(result.maximum.x, position.x);
        result.maximum.y = std::max(result.maximum.y, position.y);
        result.maximum.z = std::max(result.maximum.z, position.z);
    }
    for (std::uint32_t offset = 0u; offset < source.indexCount; ++offset) {
        pack.indices.push_back(vertexBase + tissues.indices[source.firstIndex + offset] - source.firstVertex);
    }
    result.indexCount = source.indexCount;
    return result;
}

metalrobo::VisualAssetPackV2 makeMarkerPack(
    const metalrobo::EngineModel& model,
    const LoadedMuscles& musclePayload,
    const LoadedBones* bonePayload,
    const metalrobo::NumiHumanKneePayload* openKneePayload,
    const LoadedOpenKneeLigamentFEM* openKneeLigamentFEM,
    const LoadedSoftTissues* softTissuePayload,
    const LoadedSkin* skinPayload,
    const LoadedTorsoAnatomy* torsoAnatomyPayload,
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const MRBodyStateGPU> restBodies,
    const PassiveFEMTissueVisual* passiveFEMTissue,
    const PectoralisFasciaVisual* pectoralisFascia,
    const bool muscleDriven,
    const std::span<const std::uint32_t> requestedBoneBodyIndices,
    const std::span<const std::uint32_t> requestedBoneStableIds,
    const std::span<const std::uint32_t> requestedSoftTissueStableIds,
    const bool zAnatomyCalfVisualSupplement,
    const bool tendonAttachmentCollarDiagnostic,
    const SourceRouteCentrelines* sourceRouteCentrelines,
    std::uint32_t& renderedBodies,
    std::uint32_t& renderedSoftTissues,
    std::uint32_t& renderedSkinShells,
    std::uint32_t& renderedTorsoAnatomySurfaces,
    std::uint32_t& renderedTendonAttachmentCollars,
    std::uint32_t& renderedTendonAttachmentEnvelopes,
    std::uint32_t& renderedRouteSegments,
    std::uint32_t& renderedPassiveFEMTissues,
    std::uint32_t& renderedPectoralisFascia,
    std::uint32_t& renderedOpenKneeRegions
) {
    metalrobo::VisualAssetPackV2 pack;
    pack.id = bonePayload != nullptr
        ? "myosim_fullbody_articulated_bodyparts_bones_view"
        : "myosim_fullbody_articulated_marker_view";
    pack.sourceUri = bonePayload != nullptr
        ? (skinPayload != nullptr
            ? "numi://bodyparts3d/NHBONES1+NHSKIN1+NHRIGID2+NHMYO1/articulated-shell-view"
            : (softTissuePayload != nullptr
                ? "numi://bodyparts3d/NHBONES1+NHTISS2-or-NHTISS3-or-NHTISS4+NHRIGID2+NHMYO1/articulated-anatomy-view"
                : (torsoAnatomyPayload != nullptr
                    ? "numi://bodyparts3d/NHBONES1+NHANAT1+NHRIGID2+NHMYO1/articulated-torso-anatomy-view"
                    : "numi://bodyparts3d/NHBONES1+NHRIGID2+NHMYO1/articulated-bone-view")))
        : "numi://myosim/NHRIGID2+NHMYO1/articulated-marker-view";
    pack.sourceContentHash = bonePayload != nullptr
        ? (skinPayload != nullptr
            ? "bodyparts3d-major-bones+skinned-shell+runtime-body-and-site-records"
            : (softTissuePayload != nullptr
                ? "bodyparts3d-major-bones+right-posterior-chain+runtime-body-and-site-records"
                : (torsoAnatomyPayload != nullptr
                    ? "bodyparts3d-major-bones+selected-torso-anatomy+runtime-body-and-site-records"
                    : "bodyparts3d-major-bones+runtime-body-and-site-records")))
        : "runtime-body-and-site-records";
    pack.license = bonePayload != nullptr ? "CC-BY-4.0 AND Apache-2.0" : "Apache-2.0";
    pack.preprocessingProvenance =
        bonePayload != nullptr
            ? (muscleDriven
                ? "bodyparts3d_source_import/provisional_rest_registration/cpu_fp64_mujoco_muscle_projection_and_articulated_free_body_step/metal_articulated_operator_pose_snapshot/native_visual_bone_pack.v2"
                : "bodyparts3d_source_import/provisional_rest_registration/metal_articulated_operator_pose_snapshot/native_visual_bone_pack.v2")
            : (muscleDriven
                ? "cpu_fp64_mujoco_muscle_projection_and_articulated_free_body_step/metal_articulated_operator_pose_snapshot/native_visual_marker_pack.v1"
                : "metal_articulated_operator_pose_snapshot/native_visual_marker_pack.v1");
    if (softTissuePayload != nullptr) {
        pack.preprocessingProvenance +=
            "/exact_bodyparts3d_surfaces_with_named_body_weighted_kinematic_binding";
    }
    if (torsoAnatomyPayload != nullptr) {
        pack.preprocessingProvenance +=
            "/exact_bodyparts3d_selected_torso_organ_vessel_and_neural_surfaces_with_single_link_kinematic_binding";
    }
    if (passiveFEMTissue != nullptr) {
        pack.preprocessingProvenance +=
            "/source_surface_derived_passive_matter_fem_cage_with_myosim_driven_endpoint_anchors";
    }
    if (pectoralisFascia != nullptr) {
        pack.preprocessingProvenance +=
            "/NHFASC2_source_derived_thin_solid_human_GOH_mean_fit_and_same_command_buffer_NHTENDON2_load_driven_Matter_FEM_with_compiler_authored_exact_anterior_BodyParts3D_presentation";
    }
    if (openKneePayload != nullptr) {
        const bool left = openKneePayload->side == metalrobo::NumiHumanKneeSide::left;
        pack.id = left
            ? "myosim_open_knee_oks003_left_articulated_anatomy"
            : "myosim_open_knee_oks003_right_mirrored_articulated_anatomy";
        pack.sourceUri =
            left
                ? "numi://open-knees/oks003/NHKNEE1+NHRIGID2/articulated-left-knee-view"
                : "numi://open-knees/oks003/NHKNEE1+NHRIGID2/articulated-right-mirrored-knee-view";
        pack.sourceContentHash =
            "open-knees-oks003-exact-regions-surfaces-and-live-myosim-left-knee-registration";
        pack.license = "CC-BY-4.0 AND Apache-2.0";
        pack.preprocessingProvenance +=
            bonePayload != nullptr
                ? "/BodyParts3D_full_bone_shaft_presentation_with_NHKNEE1_exact_oks003_articular_bone_ends_and_soft_tissue_attachment_contact_topology_in_articulated_femur_tibia_patella_frames"
                : "/NHKNEE1_exact_oks003_region_surfaces_with_proper_uniform_anatomical_registration_and_articulated_femur_tibia_patella_frames";
        if (openKneeLigamentFEM != nullptr) {
            pack.preprocessingProvenance += openKneeLigamentFEM->liveHumanCoupling
                ? "/accepted_exact_QAT_ACL_PCL_MCL_LCL_and_patellar_tendon_nodes_from_same_command_buffer_active_quadriceps_and_passive_Matter_FEM_reactions_on_live_full_Human_bodies"
                : "/accepted_NHKFEM1_or_NHKFEM2_exact_ligament_and_patellar_tendon_node_snapshot_from_multi_body_Matter_FEM_reaction_transaction";
        }
    }
    if (zAnatomyCalfVisualSupplement) {
        pack.id = "myosim_zanatomy_calf_articulated_visual_supplement";
        pack.sourceUri =
            "numi://bodyparts3d+zanatomy-right-calf/NHBONES1+NHTISS3+NHRIGID2+NHMYO1/articulated-anatomy-view";
        pack.sourceContentHash =
            "bodyparts3d-major-bones+zanatomy-right-calf-supplement+runtime-body-and-site-records";
        pack.license = "CC-BY-SA-4.0 AND CC-BY-4.0 AND Apache-2.0";
        pack.preprocessingProvenance +=
            "/zanatomy_cc_by_sa_right_calf_visual_supplement_with_transferred_named_bodyparts3d_myosim_nhtiss3_body_weights";
    }
    if (skinPayload != nullptr) {
        pack.preprocessingProvenance +=
            skinPayload->usesWorldRestNormals
                ? "/exact_bodyparts3d_outer_skin_sheet_with_four_registered_source_bone_surface_local_linear_blend_and_world_rest_normals"
                : skinPayload->usesSourceSurfaceLocalWeights
                ? "/exact_bodyparts3d_skin_shell_with_four_registered_source_bone_surface_local_linear_blend_kinematic_binding"
                : skinPayload->usesBoundaryLocalWeights
                ? "/exact_bodyparts3d_skin_shell_with_four_registered_bone_envelope_boundary_local_linear_blend_kinematic_binding"
                : "/exact_bodyparts3d_skin_shell_with_four_registered_bone_envelope_linear_blend_kinematic_binding";
    }
    if (sourceRouteCentrelines != nullptr) {
        pack.preprocessingProvenance +=
            "/cpu_fp64_mujoco_tangent_and_wrapped_arc_centreline_at_the_rendered_pose";
        if (sourceRouteCentrelines->surfaceProjectedAttachmentCount > 0u) {
            pack.preprocessingProvenance +=
                "/visual_only_nearest_bodyparts3d_triangle_attachment_projection";
        }
        if (musclePayload.tendonEnvelopeBindings > 0u) {
            pack.preprocessingProvenance +=
                musclePayload.tendonMigratedEnvelopeBindings > 0u
                    ? "/NHTENDON3_route_private_registered_bone_surface_force_moment_envelope_with_reference_scaled_architecture"
                    : "/NHTENDON2_source_point_preserving_connected_bone_surface_force_moment_envelope";
        }
    }
    pack.materials.push_back(makeMaterial(
        {0.82f, 0.86f, 0.88f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.52f
    ));
    pack.materials.push_back(makeMaterial(
        // Source muscle mesh detail is conveyed by real surface normals and
        // the anatomy light rig.  The former red emission lifted all shading
        // toward a flat, plastic appearance, so do not use self-illumination
        // as a substitute for an anatomical material.
        {0.56f, 0.018f, 0.014f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.58f, 0.035f
    ));
    pack.materials.push_back(makeMaterial(
        // Exact force-route diagnostics must remain legible over the red
        // BodyParts3D muscle layer.  Cyan is intentionally reserved for this
        // opt-in source-route / attachment visual, never for a tendon mesh.
        {0.035f, 0.82f, 0.98f, 1.0f}, {0.0f, 0.20f, 0.34f, 0.55f}, 0.28f, 0.16f
    ));
    pack.materials.push_back(makeMaterial(
        // Keep osseous anatomy a cool, matte ivory.  It is intentionally
        // distinct from collagen so the calcaneal insertion can be judged at
        // a glance without a diagnostic outline or false geometry.
        {0.57f, 0.63f, 0.64f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.62f
    ));
    pack.materials.push_back(makeMaterial(
        // Do not add red emission: it flattens the source muscle relief and
        // makes the layer look painted onto the skeleton rather than like an
        // anatomical surface.
        {0.50f, 0.022f, 0.014f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.50f, 0.055f
    ));
    pack.materials.push_back(makeMaterial(
        // Tendon is warm, non-metallic collagen—not a glowing route or a gold
        // overlay.  Its deliberately separate value from bone makes the
        // source-continuous calcaneal insertion inspectable in a single frame.
        {0.91f, 0.75f, 0.53f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.66f, 0.01f
    ));
    pack.materials.push_back(makeMaterial(
        // The shell remains a neutral source-anatomy presentation material.
        // Its relief comes from the exact imported normals and the same light
        // rig as the exposed anatomy—not an emission pass or painted detail.
        {0.74f, 0.37f, 0.25f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.56f, 0.025f
    ));
    pack.materials.push_back(makeMaterial(
        {0.48f, 0.08f, 0.06f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.54f, 0.025f
    ));
    pack.materials.push_back(makeMaterial(
        {0.55f, 0.012f, 0.020f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.48f, 0.035f
    ));
    pack.materials.push_back(makeMaterial(
        {0.76f, 0.54f, 0.12f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.52f, 0.02f
    ));
    pack.materials.push_back(makeMaterial(
        // Fascia is a pale collagen sheet. It remains opaque here because the
        // reference renderer has no order-independent transparency path.
        {0.78f, 0.70f, 0.56f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.72f, 0.015f
    ));
    pack.materials.push_back(makeMaterial(
        {0.34f, 0.64f, 0.72f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.38f, 0.02f
    ));
    pack.materials.push_back(makeMaterial(
        {0.20f, 0.42f, 0.44f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.56f, 0.015f
    ));
    pack.materials.push_back(makeMaterial(
        {0.86f, 0.78f, 0.61f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.72f, 0.01f
    ));
    pack.materials.push_back(makeMaterial(
        {0.94f, 0.72f, 0.48f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, 0.66f, 0.01f
    ));

    const auto appendInstance = [&pack](
        const GeometryRange& geometry,
        const std::uint32_t material,
        const std::uint32_t semantic,
        const std::uint32_t bindingKind,
        const std::uint32_t bodyIndex,
        const mr_float4 translation,
        const mr_float4 orientation,
        const std::uint32_t stableId
    ) {
        const std::uint32_t instanceIndex = static_cast<std::uint32_t>(pack.instances.size());
        MRVisualInstanceGPUV2 instance{};
        instance.translationAndScale = translation;
        instance.orientation = orientation;
        instance.binding = {
            0u, bodyIndex, bindingKind,
            MR_VISUAL_INSTANCE_CASTS_SHADOW |
                MR_VISUAL_INSTANCE_RECEIVES_SHADOW |
                MR_VISUAL_INSTANCE_VISIBLE_TO_SENSOR,
        };
        instance.identity = {semantic, stableId, bodyIndex, stableId};
        instance.geometry = {
            static_cast<std::uint32_t>(pack.primitives.size()), 1u, material, 0u,
        };
        pack.instances.push_back(instance);
        MRVisualPrimitiveGPUV2 primitive{};
        primitive.geometry = {geometry.firstIndex, geometry.indexCount, material, instanceIndex};
        primitive.identity = {semantic, stableId, bodyIndex, stableId};
        primitive.boundsMinimum = geometry.minimum;
        primitive.boundsMaximum = geometry.maximum;
        pack.primitives.push_back(primitive);
    };

    std::vector<std::uint32_t> supplementalBoneBodyIndices;
    if (zAnatomyCalfVisualSupplement && softTissuePayload != nullptr) {
        for (const SoftTissueRecord& tissue : softTissuePayload->records) {
            if (tissue.layer != kSoftTissueLayerSupplementalBone) continue;
            require(tissue.bindingCount == 1u &&
                        tissue.bodyIndex[0] != MR_INVALID_INDEX,
                    "Z-Anatomy supplemental bone must have one articulated body binding");
            supplementalBoneBodyIndices.push_back(tissue.bodyIndex[0]);
        }
        std::sort(supplementalBoneBodyIndices.begin(), supplementalBoneBodyIndices.end());
        require(std::adjacent_find(
                    supplementalBoneBodyIndices.begin(), supplementalBoneBodyIndices.end()
                ) == supplementalBoneBodyIndices.end(),
                "Z-Anatomy supplement repeats a bone overlay body");
    }

    renderedBodies = 0u;
    // An exterior source shell is opaque presentation geometry.  Retaining
    // the registered bone pack is still required to verify the shell's rest
    // frame, but drawing it behind source skin openings creates misleading
    // blue/ivory peeks that read as broken anatomy rather than an exterior.
    // Exposed bones remain available through the separate anatomy command.
    if (bonePayload != nullptr && skinPayload == nullptr) {
        const auto kneeBodies = openKneePayload != nullptr
            ? openKneeBodyIndices(*openKneePayload)
            : std::array<std::uint32_t, 3u>{
                MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX};
        for (const BoneRecord& bone : bonePayload->records) {
            if (openKneePayload != nullptr &&
                std::find(kneeBodies.begin(), kneeBodies.end(), bone.bodyIndex) ==
                    kneeBodies.end()) {
                // BodyParts3D remains the Human's full-bone geometry owner.
                // In a focused Open Knee view, draw only the three relevant
                // full articulated body groups rather than the entire body.
                continue;
            }
            if (std::binary_search(
                    supplementalBoneBodyIndices.begin(), supplementalBoneBodyIndices.end(), bone.bodyIndex
                )) {
                continue;
            }
            if (!requestedBoneBodyIndices.empty() &&
                !std::binary_search(
                    requestedBoneBodyIndices.begin(), requestedBoneBodyIndices.end(), bone.bodyIndex
                )) {
                continue;
            }
            if (!requestedBoneStableIds.empty() &&
                !std::binary_search(
                    requestedBoneStableIds.begin(), requestedBoneStableIds.end(), bone.stableId
                )) {
                continue;
            }
            const GeometryRange geometry = appendBoneGeometry(pack, *bonePayload, bone);
            appendInstance(
                geometry, 3u, kBoneSemantic, MR_VISUAL_BINDING_ARTICULATED_LINK,
                bone.bodyIndex,
                {bone.translationX, bone.translationY, bone.translationZ, bone.uniformScale},
                {bone.quaternionX, bone.quaternionY, bone.quaternionZ, bone.quaternionW},
                bone.stableId
            );
            ++renderedBodies;
        }
    } else if (bonePayload == nullptr) {
        for (std::size_t bodyIndex = 0u; bodyIndex < model.bodies.size(); ++bodyIndex) {
            const MRBodyPropertiesGPU& body = model.bodies[bodyIndex];
            if (!(body.massAndInverseMass.x > 1.0e-5f)) {
                continue;
            }
            const GeometryRange geometry = appendEllipsoid(pack, inertiaEllipsoid(body));
            appendInstance(
                geometry, 0u, kBodySemantic, MR_VISUAL_BINDING_ARTICULATED_LINK,
                static_cast<std::uint32_t>(bodyIndex),
                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                static_cast<std::uint32_t>(bodyIndex + 1u)
            );
            ++renderedBodies;
        }
    }
    renderedOpenKneeRegions = 0u;
    if (openKneePayload != nullptr) {
        for (std::uint32_t regionIndex = 0u;
             regionIndex < openKneePayload->regions.size(); ++regionIndex) {
            const auto& region = openKneePayload->regions[regionIndex];
            const bool usesAcceptedFEM = openKneeLigamentFEM != nullptr &&
                region.firstNode < openKneeLigamentFEM->deformedNodes.size() &&
                openKneeLigamentFEM->deformedNodes[region.firstNode];
            const auto geometry = appendOpenKneeGeometry(
                pack, *openKneePayload, region, openKneeLigamentFEM);
            std::uint32_t material = 3u;
            std::uint32_t semantic = kBoneSemantic;
            switch (region.kind) {
                case metalrobo::NumiHumanKneeRegionKind::bone:
                    break;
                case metalrobo::NumiHumanKneeRegionKind::cartilage:
                    material = 11u; semantic = kKneeCartilageSemantic; break;
                case metalrobo::NumiHumanKneeRegionKind::meniscus:
                    material = 12u; semantic = kKneeMeniscusSemantic; break;
                case metalrobo::NumiHumanKneeRegionKind::ligament:
                    material = 13u; semantic = kKneeLigamentSemantic; break;
                case metalrobo::NumiHumanKneeRegionKind::tendon:
                    material = 14u; semantic = kKneeTendonSemantic; break;
            }
            appendInstance(
                geometry, material, semantic,
                usesAcceptedFEM ? MR_VISUAL_BINDING_WORLD
                                : MR_VISUAL_BINDING_ARTICULATED_LINK,
                usesAcceptedFEM ? MR_INVALID_INDEX : region.visualBodyIndex,
                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                1000u + regionIndex
            );
            ++renderedOpenKneeRegions;
            ++renderedBodies;
        }
    }
    renderedSoftTissues = 0u;
    renderedPassiveFEMTissues = 0u;
    if (softTissuePayload != nullptr) {
        for (const SoftTissueRecord& tissue : softTissuePayload->records) {
            if (!requestedSoftTissueStableIds.empty() &&
                !std::binary_search(
                    requestedSoftTissueStableIds.begin(),
                    requestedSoftTissueStableIds.end(), tissue.stableId
                )) {
                continue;
            }
            if (pectoralisFascia != nullptr &&
                std::find(pectoralisFascia->sourceStableIds.begin(),
                          pectoralisFascia->sourceStableIds.end(),
                          tissue.stableId) != pectoralisFascia->sourceStableIds.end()) {
                // The fascia instance below already contains these exact
                // vertices and triangles with the solved FEM displacement
                // transferred onto them. Rendering the undeformed red muscle
                // copy as well would be duplicate coincident geometry.
                continue;
            }
            const bool usesPassiveFEM = passiveFEMTissue != nullptr &&
                passiveFEMTissue->stableId == tissue.stableId;
            const GeometryRange geometry = usesPassiveFEM
                ? appendPassiveFEMMappedSoftTissueGeometry(
                    pack, *softTissuePayload, tissue, restBodies, *passiveFEMTissue
                )
                : appendSoftTissueGeometry(pack, *softTissuePayload, tissue, bodies);
            const bool isMuscle = tissue.layer == kSoftTissueLayerMuscle;
            const bool isSupplementalBone = tissue.layer == kSoftTissueLayerSupplementalBone;
            appendInstance(
                geometry, isSupplementalBone ? 3u : (isMuscle ? 4u : 5u),
                usesPassiveFEM ? kPassiveFEMTissueSemantic :
                    (isSupplementalBone ? kBoneSemantic :
                        (isMuscle ? kMuscleSurfaceSemantic : kTendonSurfaceSemantic)),
                MR_VISUAL_BINDING_WORLD,
                isSupplementalBone ? tissue.bodyIndex[0] : MR_INVALID_INDEX,
                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                tissue.stableId
            );
            ++renderedSoftTissues;
            renderedBodies += isSupplementalBone ? 1u : 0u;
            renderedPassiveFEMTissues += usesPassiveFEM ? 1u : 0u;
        }
    }
    renderedPectoralisFascia = 0u;
    if (pectoralisFascia != nullptr) {
        appendInstance(
            appendPectoralisFasciaGeometry(pack, *pectoralisFascia),
            10u, kPectoralisFasciaSemantic, MR_VISUAL_BINDING_WORLD,
            MR_INVALID_INDEX,
            {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}, 1u
        );
        renderedPectoralisFascia = 1u;
    }
    renderedTendonAttachmentCollars = 0u;
    // Source tendons use their own triangle mesh and per-vertex named-bone
    // lock.  The optional collar is intentionally a diagnostic because its
    // generated quads can hide or protrude beyond an otherwise continuous
    // source insertion, especially in a close Z-Anatomy calf inspection.
    if (tendonAttachmentCollarDiagnostic &&
        bonePayload != nullptr && softTissuePayload != nullptr) {
        for (const SoftTissueRecord& tissue : softTissuePayload->records) {
            if (tissue.layer != kSoftTissueLayerTendon ||
                (!requestedSoftTissueStableIds.empty() &&
                 !std::binary_search(
                     requestedSoftTissueStableIds.begin(),
                     requestedSoftTissueStableIds.end(), tissue.stableId
                 )) ||
                (!requestedBoneBodyIndices.empty() &&
                 !std::binary_search(
                     requestedBoneBodyIndices.begin(),
                     requestedBoneBodyIndices.end(), tissue.bodyIndex[softTissueLastBinding(tissue)]
                 ))) {
                continue;
            }
            const GeometryRange collar = appendTendonAttachmentCollarGeometry(
                pack, *softTissuePayload, *bonePayload, tissue, bodies
            );
            if (collar.indexCount == 0u) continue;
            appendInstance(
                collar, 5u, kTendonAttachmentCollarSemantic,
                MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                1000u + tissue.stableId
            );
            ++renderedTendonAttachmentCollars;
        }
    }
    renderedSkinShells = 0u;
    if (skinPayload != nullptr) {
        appendInstance(
            appendSkinGeometry(pack, *skinPayload, bodies, restBodies),
            6u, kSkinShellSemantic, MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
            {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}, 1u
        );
        renderedSkinShells = 1u;
    }
    renderedTorsoAnatomySurfaces = 0u;
    if (torsoAnatomyPayload != nullptr) {
        for (const TorsoAnatomyRecord& surface : torsoAnatomyPayload->records) {
            const bool isOrgan = surface.layer == kTorsoAnatomyLayerOrgan;
            const bool isVessel = surface.layer == kTorsoAnatomyLayerVessel;
            appendInstance(
                appendTorsoAnatomyGeometry(pack, *torsoAnatomyPayload, surface),
                isOrgan ? 7u : (isVessel ? 8u : 9u),
                isOrgan ? kOrganSurfaceSemantic :
                    (isVessel ? kVesselSurfaceSemantic : kNerveSurfaceSemantic),
                MR_VISUAL_BINDING_ARTICULATED_LINK, surface.bodyIndex,
                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                surface.stableId
            );
            ++renderedTorsoAnatomySurfaces;
        }
    }
    renderedRouteSegments = 0u;
    renderedTendonAttachmentEnvelopes = 0u;
    std::uint32_t stableRouteId = 1u;
    if (sourceRouteCentrelines != nullptr) {
        for (const SourceRouteCentreline& route : sourceRouteCentrelines->muscles) {
            require(route.muscleIndex < musclePayload.muscles.size() && route.points.size() >= 2u,
                    "MyoSim source-route visual record is malformed");
            for (std::size_t index = 1u; index < route.points.size(); ++index) {
                const mr_float4 previous = route.points[index - 1u].world;
                const mr_float4 current = route.points[index].world;
                const float dx = current.x - previous.x;
                const float dy = current.y - previous.y;
                const float dz = current.z - previous.z;
                if (dx * dx + dy * dy + dz * dz > 1.0e-10f) {
                    appendInstance(
                        appendWorldTube(
                            pack, previous, current,
                            bonePayload != nullptr ? 0.0016f : 0.0024f
                        ), 2u,
                        kRouteSemantic, MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                        {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                        stableRouteId++
                    );
                    ++renderedRouteSegments;
                }
            }
            // These partially intersect the registered bone surface, making
            // the resolved route's source origin/insertion visibly terminate
            // at it.  They are intentionally only endpoint cues, not a
            // fabricated tendon surface or altered force path.
            for (const std::size_t pointIndex : {std::size_t{0u}, route.points.size() - 1u}) {
                const SourceRouteCentreline::Point& point = route.points[pointIndex];
                if (!point.surfaceProjected) continue;
                constexpr float kAttachmentCapRadius = 0.0048f;
                constexpr float kAttachmentCapSurfaceOverlap = 0.0015f;
                appendInstance(
                    appendEllipsoid(
                        pack,
                        {kAttachmentCapRadius, kAttachmentCapRadius, kAttachmentCapRadius}
                    ),
                    2u, kSiteSemantic, MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                    {
                        point.world.x + kAttachmentCapSurfaceOverlap * point.surfaceNormalWorld.x,
                        point.world.y + kAttachmentCapSurfaceOverlap * point.surfaceNormalWorld.y,
                        point.world.z + kAttachmentCapSurfaceOverlap * point.surfaceNormalWorld.z,
                        1.0f,
                    },
                    {0.0f, 0.0f, 0.0f, 1.0f}, stableRouteId++
                );
            }
            if (musclePayload.tendonPayload.payloadAbi == 2u ||
                musclePayload.tendonPayload.payloadAbi == 3u) {
                require(
                    bonePayload != nullptr &&
                        musclePayload.tendonPayload.boneCount == bonePayload->records.size() &&
                        musclePayload.tendonPayload.registrationFingerprint ==
                            bonePayload->header.reserved0,
                    "NHTENDON2/3 visual envelope does not match the loaded NHBONES1 registration"
                );
                for (std::uint32_t endpoint = 0u; endpoint < 2u; ++endpoint) {
                    const auto& binding = musclePayload.tendonPayload.bindings[
                        2u * route.muscleIndex + endpoint
                    ];
                    if (binding.mode != metalrobo::NumiHumanTendonAttachmentMode::registeredBoneDistributedEnvelope &&
                        binding.mode != metalrobo::NumiHumanTendonAttachmentMode::registeredBoneMigratedDistributedEnvelope) {
                        continue;
                    }
                    require(binding.bodyIndex < bodies.size() &&
                                binding.triangleIndex < musclePayload.tendonPayload.envelopes.size(),
                            "NHTENDON2/3 visual envelope binding is out of bounds");
                    const auto& envelope = musclePayload.tendonPayload.envelopes[binding.triangleIndex];
                    const MRBodyStateGPU& body = bodies[binding.bodyIndex];
                    std::array<mr_float4, 4u> nodes{};
                    for (std::size_t node = 0u; node < nodes.size(); ++node) {
                        const mr_float4 local{
                            static_cast<float>(envelope.localNodes[node][0]),
                            static_cast<float>(envelope.localNodes[node][1]),
                            static_cast<float>(envelope.localNodes[node][2]), 0.0f,
                        };
                        const mr_float4 rotated = rotatePoint(body.orientation, local);
                        nodes[node] = {
                            body.position.x + rotated.x,
                            body.position.y + rotated.y,
                            body.position.z + rotated.z, 1.0f,
                        };
                    }
                    const mr_float4 terminal = endpoint == 0u
                        ? route.points.front().world : route.points.back().world;
                    for (std::size_t node = 0u; node < nodes.size(); ++node) {
                        const float dx = nodes[node].x - terminal.x;
                        const float dy = nodes[node].y - terminal.y;
                        const float dz = nodes[node].z - terminal.z;
                        if (dx * dx + dy * dy + dz * dz > 1.0e-10f) {
                            appendInstance(
                                // A one-millimetre collagen bundle remains a
                                // diagnostic representation of the exact
                                // four-node transfer map, but survives the
                                // 2K presentation filter from every qualified
                                // camera. It changes no mechanical coordinate
                                // or force-distribution weight.
                                appendWorldTube(pack, terminal, nodes[node], 0.00115f),
                                5u, kTendonAttachmentEnvelopeSemantic,
                                MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                                stableRouteId++
                            );
                        }
                        const mr_float4 next = nodes[(node + 1u) % nodes.size()];
                        const float ex = next.x - nodes[node].x;
                        const float ey = next.y - nodes[node].y;
                        const float ez = next.z - nodes[node].z;
                        if (ex * ex + ey * ey + ez * ez > 1.0e-10f) {
                            appendInstance(
                                appendWorldTube(pack, nodes[node], next, 0.00082f),
                                5u, kTendonAttachmentEnvelopeSemantic,
                                MR_VISUAL_BINDING_WORLD, MR_INVALID_INDEX,
                                {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f},
                                stableRouteId++
                            );
                        }
                    }
                    ++renderedTendonAttachmentEnvelopes;
                }
            }
        }
    }
    pack.contentHash = metalrobo::computeVisualAssetPackContentHash(pack);
    std::string reason;
    require(pack.valid(&reason), "native Human marker pack is invalid: " + reason);
    return pack;
}

struct CameraFraming {
    mr_float4 center{};
    float distance = 0.0f;
    float sourceExtentMeters = 0.0f;
    bool usesSourceGeometryBounds = false;
    bool usesJointAnchor = false;
    float jointAnchorResidualMeters = 0.0f;
};

CameraFraming makeCameraFraming(
    const metalrobo::VisualAssetPackV2& pack,
    const std::span<const MRBodyStateGPU> bodies,
    const std::span<const MRJointDescriptorGPU> joints,
    const std::optional<std::uint32_t> focusBodyIndex,
    const std::optional<std::uint32_t> focusJointChildBodyIndex,
    const std::optional<float> focusDistanceMeters,
    const bool tendonAttachmentEnvelopeInspection,
    const metalrobo::NumiHumanKneePayload* openKneeInspection
) {
    require(!(focusBodyIndex.has_value() && focusJointChildBodyIndex.has_value()),
            "MyoSim visual body and joint focus are mutually exclusive");
    if (focusBodyIndex.has_value()) {
        require(*focusBodyIndex < bodies.size(), "MyoSim visual focus body index is out of bounds");
    }
    if (focusJointChildBodyIndex.has_value()) {
        require(*focusJointChildBodyIndex < bodies.size(),
                "MyoSim visual focus joint child body index is out of bounds");
        const MRJointDescriptorGPU* selected = nullptr;
        for (const MRJointDescriptorGPU& joint : joints) {
            if (joint.childBody != *focusJointChildBodyIndex) continue;
            require(selected == nullptr,
                    "MyoSim visual focus child body has multiple owning joints");
            selected = &joint;
        }
        require(selected != nullptr, "MyoSim visual focus child body has no owning joint");
        require(selected->parentBody < bodies.size() && selected->childBody < bodies.size(),
                "MyoSim visual focus joint body ownership is out of bounds");
        const mr_float4 parentAnchorWorld = addPoint(
            bodies[selected->parentBody].position,
            rotatePoint(bodies[selected->parentBody].orientation, selected->parentAnchor)
        );
        const mr_float4 childAnchorWorld = addPoint(
            bodies[selected->childBody].position,
            rotatePoint(bodies[selected->childBody].orientation, selected->childAnchor)
        );
        const mr_float4 residual = subtractPoint(parentAnchorWorld, childAnchorWorld);
        const float residualMeters = std::sqrt(dotPoint(residual, residual));
        require(std::isfinite(residualMeters) && residualMeters <= 2.0e-4f,
                "MyoSim visual focus joint anchors do not coincide in the posed state");
        return {
            .center = scalePoint(addPoint(parentAnchorWorld, childAnchorWorld), 0.5f),
            .distance = focusDistanceMeters.value_or(0.22f),
            .usesJointAnchor = true,
            .jointAnchorResidualMeters = residualMeters,
        };
    }

    mr_float4 minimum{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 0.0f,
    };
    mr_float4 maximum{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), 0.0f,
    };
    std::size_t boundedVertexCount = 0u;
    std::size_t includedPrimitiveCount = 0u;
    const auto include = [&minimum, &maximum](const mr_float4 point) {
        require(std::isfinite(point.x) && std::isfinite(point.y) &&
                    std::isfinite(point.z),
                "MyoSim visual source geometry has a non-finite world bound");
        minimum.x = std::min(minimum.x, point.x);
        minimum.y = std::min(minimum.y, point.y);
        minimum.z = std::min(minimum.z, point.z);
        maximum.x = std::max(maximum.x, point.x);
        maximum.y = std::max(maximum.y, point.y);
        maximum.z = std::max(maximum.z, point.z);
    };
    for (std::size_t instanceIndex = 0u; instanceIndex < pack.instances.size(); ++instanceIndex) {
        const MRVisualInstanceGPUV2& instance = pack.instances[instanceIndex];
        require(instance.geometry.x <= pack.primitives.size() &&
                    instance.geometry.y <= pack.primitives.size() - instance.geometry.x &&
                    std::isfinite(instance.translationAndScale.w) &&
                    instance.translationAndScale.w > 0.0f,
                "MyoSim visual instance has an invalid framing range");
        const bool articulated =
            instance.binding.z == MR_VISUAL_BINDING_ARTICULATED_LINK;
        require(articulated || instance.binding.z == MR_VISUAL_BINDING_WORLD,
                "MyoSim visual framing only supports world or articulated-link bindings");
        // Target focused inspections from the selected rendered anatomy, not
        // from a mechanics COM frame that can sit outside a small wrist or
        // digit mesh. World-bound route diagnostics remain rendered but must
        // not drag the anatomical camera target away from the selected bone.
        const auto kneeBodies = openKneeInspection != nullptr
            ? openKneeBodyIndices(*openKneeInspection)
            : std::array<std::uint32_t, 3u>{MR_INVALID_INDEX, MR_INVALID_INDEX, MR_INVALID_INDEX};
        const bool focusedKneeGroup = openKneeInspection != nullptr &&
            focusBodyIndex.has_value() &&
            std::find(kneeBodies.begin(), kneeBodies.end(), *focusBodyIndex) != kneeBodies.end();
        const bool includedKneeBody = focusedKneeGroup && articulated &&
            std::find(kneeBodies.begin(), kneeBodies.end(), instance.binding.y) != kneeBodies.end();
        if (focusBodyIndex.has_value() &&
            !includedKneeBody && (!articulated || instance.binding.y != *focusBodyIndex)) {
            continue;
        }
        const MRBodyStateGPU* body = nullptr;
        if (articulated) {
            require(instance.binding.y < bodies.size(),
                    "MyoSim visual framing articulated binding is out of bounds");
            body = &bodies[instance.binding.y];
        }
        const auto worldPoint = [&instance, body](const mr_float4 local) {
            const mr_float4 instancePoint = addPoint(
                instance.translationAndScale,
                rotatePoint(
                    instance.orientation,
                    scalePoint(local, instance.translationAndScale.w)
                )
            );
            return body == nullptr
                ? instancePoint
                : addPoint(body->position, rotatePoint(body->orientation, instancePoint));
        };
        for (std::uint32_t primitiveOffset = 0u;
             primitiveOffset < instance.geometry.y; ++primitiveOffset) {
            const MRVisualPrimitiveGPUV2& primitive =
                pack.primitives[instance.geometry.x + primitiveOffset];
            require(primitive.geometry.w == instanceIndex,
                    "MyoSim visual framing primitive/instance identity drifted");
            require(primitive.geometry.x <= pack.indices.size() &&
                        primitive.geometry.y <= pack.indices.size() - primitive.geometry.x,
                    "MyoSim visual framing primitive index range is invalid");
            // Use the actual rendered vertices for both bounds and target.
            // An AABB midpoint can land in empty space for an asymmetric
            // oblique anatomy view, which is why the prior frame showed most
            // of the body in one corner despite nominally correct bounds.
            for (std::uint32_t indexOffset = 0u;
                 indexOffset < primitive.geometry.y; ++indexOffset) {
                const std::uint32_t vertexIndex =
                    pack.indices[primitive.geometry.x + indexOffset];
                require(vertexIndex < pack.vertices.size(),
                        "MyoSim visual framing primitive references an invalid vertex");
                const mr_float4 point = worldPoint(pack.vertices[vertexIndex].position);
                include(point);
                ++boundedVertexCount;
            }
            ++includedPrimitiveCount;
        }
    }
    if (focusBodyIndex.has_value() &&
        (includedPrimitiveCount == 0u || boundedVertexCount == 0u)) {
        // Mechanics-only bodies can intentionally lack presentation geometry.
        // Preserve their deterministic legacy framing and expose the fallback
        // through the zero source extent in the receipt.
        const MRBodyStateGPU& focus = bodies[*focusBodyIndex];
        return {
            .center = {focus.position.x, focus.position.y, focus.position.z, 0.0f},
            .distance = tendonAttachmentEnvelopeInspection ? 0.45f : 0.70f,
        };
    }
    require(includedPrimitiveCount > 0u,
            "MyoSim visual source geometry has no primitives for framing");
    require(boundedVertexCount > 0u,
            "MyoSim visual source geometry has no vertices for framing");
    const float extent = std::max({
        maximum.x - minimum.x, maximum.y - minimum.y, maximum.z - minimum.z,
    });
    require(std::isfinite(extent) && extent > 1.0e-4f,
            "MyoSim visual source geometry has a degenerate framing bound");
    if (focusBodyIndex.has_value()) {
        return {
            .center = {
                0.5f * (minimum.x + maximum.x),
                0.5f * (minimum.y + maximum.y),
                0.5f * (minimum.z + maximum.z),
                0.0f,
            },
            // Preserve the close enthesis/envelope inspections. Ordinary
            // source bones scale from their own posed surface, with a 45 cm
            // floor that retains adjacent joint context and keeps lateral
            // cameras outside overlapping torso/pelvis anatomy.
            .distance = openKneeInspection != nullptr
                ? std::max(1.30f * extent, 0.20f)
                : *focusBodyIndex == 138u
                ? 0.25f
                : (tendonAttachmentEnvelopeInspection
                    ? 0.45f
                    : (pack.id == "myosim_zanatomy_calf_articulated_visual_supplement"
                        ? std::max(1.65f * extent, 0.58f)
                        : std::max(1.65f * extent, 0.45f))),
            .sourceExtentMeters = extent,
            .usesSourceGeometryBounds = true,
        };
    }
    return {
        // The vertex centroid is biased toward triangle-dense torso anatomy
        // and cropped the feet. The posed source AABB midpoint guarantees
        // whole-body coverage from every fixed validation angle.
        .center = {
            0.5f * (minimum.x + maximum.x),
            0.5f * (minimum.y + maximum.y),
            0.5f * (minimum.z + maximum.z),
            0.0f,
        },
        // The old global 1.85 m lower bound was appropriate for a 1.7 m
        // whole-body specimen but reduced a selected ankle or wrist surface
        // to a thumbnail.  Retain the full-body stand-off through its actual
        // extent, while allowing a filtered anatomical insertion inspection
        // to fill the frame with a conservative 0.25 m lower bound.
        .distance = std::max(0.92f * extent, 0.25f),
        .sourceExtentMeters = extent,
        .usesSourceGeometryBounds = true,
    };
}

std::array<mr_float4, 4u> cameraPositions(
    const CameraFraming& framing
) {
    const mr_float4 center = framing.center;
    const float distance = framing.distance;
    return {{
        {center.x, center.y - distance, center.z + 0.10f * distance, 0.0f},
        {
            center.x + 0.72f * distance, center.y - 0.72f * distance,
            center.z + 0.16f * distance, 0.0f,
        },
        {center.x + distance, center.y, center.z + 0.16f * distance, 0.0f},
        {center.x, center.y + distance, center.z + 0.10f * distance, 0.0f},
    }};
}

metalrobo::WorldTemplate makeWorld(
    const metalrobo::EngineModel& model,
    const CameraFraming& framing,
    const std::uint32_t dimension,
    std::array<std::string, 4u>& cameraNames
) {
    const mr_float4 center = framing.center;
    const std::array<mr_float4, 4u> positions = cameraPositions(framing);
    cameraNames = {"front", "oblique", "side", "rear"};
    metalrobo::EpisodeTwin episode;
    episode.id = "myosim_fullbody_articulated_marker_visualization";
    metalrobo::WorldAsset human;
    human.id = "myosim_human";
    human.semanticClass = "human_articulated_marker_view";
    human.role = MR_WORLD_ASSET_ROBOT;
    human.render = MR_WORLD_RENDER_MESH_PBR;
    human.collision = MR_WORLD_COLLISION_NONE;
    human.dynamics = MR_WORLD_DYNAMICS_ARTICULATED;
    human.articulationIndex = 0u;
    human.bodyIndices.resize(model.bodies.size());
    for (std::size_t index = 0u; index < human.bodyIndices.size(); ++index) {
        human.bodyIndices[index] = static_cast<std::uint32_t>(index);
    }
    episode.assets.push_back(std::move(human));
    episode.sensors = {
        makeCamera(cameraNames[0], positions[0], center, dimension),
        makeCamera(cameraNames[1], positions[1], center, dimension),
        makeCamera(cameraNames[2], positions[2], center, dimension),
        makeCamera(cameraNames[3], positions[3], center, dimension),
    };
    episode.task.id = "pose_snapshot_visualization";
    episode.task.robotAssetId = "myosim_human";
    episode.task.controlPeriodSeconds = 1.0 / 120.0;
    episode.task.horizonSeconds = 1.0;
    metalrobo::WorldTemplate world;
    const auto compiled = metalrobo::compileEpisodeTwin(episode, model, world);
    require(compiled.succeeded(), "native Human visual world compile failed: " + compiled.message);
    return world;
}

metalrobo::VisualMotionSampleBatchV1 makeMotion(
    const std::span<const MRBodyStateGPU> bodies
) {
    metalrobo::VisualMotionSampleBatchV1 motion;
    motion.environmentCount = 1u;
    motion.bodyCount = static_cast<std::uint32_t>(bodies.size());
    motion.sampleCount = 2u;
    motion.exposureOpenSeconds = 0.0;
    motion.exposureCloseSeconds = 1.0 / 120.0;
    motion.timestampsSeconds = {motion.exposureOpenSeconds, motion.exposureCloseSeconds};
    motion.bodyStates.insert(motion.bodyStates.end(), bodies.begin(), bodies.end());
    motion.bodyStates.insert(motion.bodyStates.end(), bodies.begin(), bodies.end());
    motion.scenarioIdentity = 0x4d594f53494dull;
    motion.source = MR_VISUAL_SOURCE_SIMULATION;
    return motion;
}

float linearToSrgb(const float value) {
    const float mapped = std::max(value, 0.0f) / (1.0f + std::max(value, 0.0f));
    return mapped <= 0.0031308f
        ? 12.92f * mapped
        : 1.055f * std::pow(mapped, 1.0f / 2.4f) - 0.055f;
}

bool writePng(
    const std::filesystem::path& path,
    const metalrobo::HybridObservationBatch& observations
) {
    const std::size_t pixels = static_cast<std::size_t>(observations.width) * observations.height;
    if (observations.environmentCount != 1u || observations.rgb.size() != pixels) {
        return false;
    }
    std::vector<std::uint8_t> rgba(pixels * 4u);
    for (std::size_t pixel = 0u; pixel < pixels; ++pixel) {
        const mr_float4 value = observations.rgb[pixel];
        rgba[pixel * 4u + 0u] = static_cast<std::uint8_t>(std::lround(
            255.0f * std::clamp(linearToSrgb(value.x), 0.0f, 1.0f)
        ));
        rgba[pixel * 4u + 1u] = static_cast<std::uint8_t>(std::lround(
            255.0f * std::clamp(linearToSrgb(value.y), 0.0f, 1.0f)
        ));
        rgba[pixel * 4u + 2u] = static_cast<std::uint8_t>(std::lround(
            255.0f * std::clamp(linearToSrgb(value.z), 0.0f, 1.0f)
        ));
        rgba[pixel * 4u + 3u] = 255u;
    }
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.string().size()), false
    );
    if (url == nullptr) return false;
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, rgba.data(), rgba.size(), nullptr);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGImageRef image = provider != nullptr && colorSpace != nullptr
        ? CGImageCreate(observations.width, observations.height, 8u, 32u,
              static_cast<std::size_t>(observations.width) * 4u, colorSpace,
              static_cast<CGBitmapInfo>(
                  static_cast<std::uint32_t>(kCGBitmapByteOrderDefault) |
                  static_cast<std::uint32_t>(kCGImageAlphaLast)
              ), provider, nullptr, false, kCGRenderingIntentDefault)
        : nullptr;
    CGImageDestinationRef destination = image != nullptr
        ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1u, nullptr)
        : nullptr;
    bool succeeded = false;
    if (destination != nullptr) {
        CGImageDestinationAddImage(destination, image, nullptr);
        succeeded = CGImageDestinationFinalize(destination);
    }
    if (destination != nullptr) CFRelease(destination);
    if (image != nullptr) CGImageRelease(image);
    if (colorSpace != nullptr) CGColorSpaceRelease(colorSpace);
    if (provider != nullptr) CGDataProviderRelease(provider);
    CFRelease(url);
    return succeeded;
}

std::size_t coverage(
    const metalrobo::HybridObservationBatch& observations,
    const std::uint32_t semantic
) {
    return static_cast<std::size_t>(std::count(
        observations.segmentation.begin(), observations.segmentation.end(), semantic
    ));
}

double parseMuscleStepSeconds(const std::string& value) {
    std::size_t parsed = 0u;
    double result = 0.0;
    try {
        result = std::stod(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("--muscle-step-seconds must be a finite decimal number");
    }
    require(parsed == value.size() && std::isfinite(result),
            "--muscle-step-seconds must be a finite decimal number");
    return result;
}

double parseMuscleActivation(const std::string& value) {
    std::size_t parsed = 0u;
    double result = 0.0;
    try {
        result = std::stod(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("--muscle-activation must be a finite decimal from 0 through 1");
    }
    require(parsed == value.size() && std::isfinite(result) && result >= 0.0 && result <= 1.0,
            "--muscle-activation must be a finite decimal from 0 through 1");
    return result;
}

double parsePoseCoordinate(const std::string& value) {
    std::size_t parsed = 0u;
    double result = 0.0;
    try {
        result = std::stod(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("--pose-q must use a finite decimal coordinate value");
    }
    require(parsed == value.size() && std::isfinite(result) && std::abs(result) <= 10.0,
            "--pose-q coordinate value must be finite and within [-10, 10]");
    return result;
}

float parseFocusDistanceMeters(const std::string& value) {
    std::size_t parsed = 0u;
    float result = 0.0f;
    try {
        result = std::stof(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "--focus-distance-m must be a finite decimal from 0.08 through 0.80"
        );
    }
    require(parsed == value.size() && std::isfinite(result) &&
                result >= 0.08f && result <= 0.80f,
            "--focus-distance-m must be a finite decimal from 0.08 through 0.80");
    return result;
}

std::uint32_t parseMuscleStepCount(const std::string& value) {
    std::size_t parsed = 0u;
    unsigned long result = 0ul;
    try {
        result = std::stoul(value, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("--muscle-step-count must be an integer from 1 through 64");
    }
    require(parsed == value.size() && result >= 1ul && result <= 64ul,
            "--muscle-step-count must be an integer from 1 through 64");
    return static_cast<std::uint32_t>(result);
}

std::uint32_t parseWholeBodyActivationSweeps(const std::string& value) {
    std::size_t parsed = 0u;
    unsigned long result = 0ul;
    try {
        result = std::stoul(value, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "--whole-body-activation-sweeps must be an integer from 1 "
            "through 8192");
    }
    require(parsed == value.size() && result >= 1ul && result <= 8192ul,
            "--whole-body-activation-sweeps must be an integer from 1 "
            "through 8192");
    return static_cast<std::uint32_t>(result);
}

std::uint32_t parseSourceRouteIndex(const std::string& value) {
    std::size_t parsed = 0u;
    unsigned long result = 0ul;
    try {
        result = std::stoul(value, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("--source-route-index must be a non-negative integer");
    }
    require(parsed == value.size() && result <= std::numeric_limits<std::uint32_t>::max(),
            "--source-route-index must be a 32-bit non-negative integer");
    return static_cast<std::uint32_t>(result);
}

std::uint32_t parseFrameDimension(const std::string& value) {
    std::size_t parsed = 0u;
    unsigned long result = 0ul;
    try {
        result = std::stoul(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("--dimension must be an integer multiple of 64 from 512 through 2048");
    }
    require(parsed == value.size() && result >= 512ul && result <= 2048ul && result % 64ul == 0ul,
            "--dimension must be an integer multiple of 64 from 512 through 2048");
    return static_cast<std::uint32_t>(result);
}

std::uint32_t parseCameraIndex(const std::string& value) {
    std::size_t parsed = 0u;
    unsigned long result = 0ul;
    try {
        result = std::stoul(value, &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error("--camera-index must be an integer from 0 through 3");
    }
    require(parsed == value.size() && result <= 3ul,
            "--camera-index must be an integer from 0 through 3");
    return static_cast<std::uint32_t>(result);
}

} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            std::optional<double> muscleStepSeconds;
            std::optional<double> muscleActivation;
            std::optional<std::uint32_t> muscleStepCount;
            bool persistentMetalStand = false;
            bool selectedTendonControl = false;
            bool standRootAssistance = false;
            bool standRemoveAssistance = false;
            bool standDeterministicReplay = false;
            bool bilateralAchillesCertificate = false;
            bool bilateralPlantarFasciaCertificate = false;
            bool wholeBodySupportCertificate = false;
            bool sourcePassiveJointTissue = false;
            bool wholeBodyAllResiduals = false;
            bool fifthMcpLowerStopCounterfactualRequested = false;
            bool sourceRouteCentrelines = false;
            bool surfaceProjectSourceSites = false;
            std::vector<std::uint32_t> requestedSourceRouteMuscles;
            std::vector<std::uint32_t> selectedSourceMuscleActivations;
            std::vector<std::uint32_t> requestedBoneBodyIndices;
            std::vector<std::uint32_t> requestedBoneStableIds;
            std::vector<std::uint32_t> requestedSoftTissueStableIds;
            bool zAnatomyCalfVisualSupplement = false;
            bool tendonAttachmentCollarDiagnostic = false;
            std::optional<std::uint32_t> focusBodyIndex;
            std::optional<std::uint32_t> focusJointChildBodyIndex;
            std::optional<float> focusDistanceMeters;
            std::optional<std::filesystem::path> softTissuePayloadPath;
            std::optional<std::filesystem::path> openKneePayloadPath;
            std::optional<std::filesystem::path> openKneeLigamentFEMPath;
            bool openKneeLiveTissueFEM = false;
            std::optional<std::filesystem::path> skinPayloadPath;
            std::optional<std::filesystem::path> torsoAnatomyPayloadPath;
            std::optional<std::filesystem::path> supportContactPayloadPath;
            std::optional<std::filesystem::path> tendonPayloadPath;
            std::optional<std::filesystem::path> jointEqualityPayloadPath;
            std::optional<std::uint32_t> passiveFEMTissueStableId;
            std::optional<std::uint32_t> passiveFEMStepCount;
            std::optional<std::filesystem::path> passiveFEMMetallibPath;
            std::optional<std::filesystem::path> pectoralisFasciaPayloadPath;
            std::optional<std::uint32_t> pectoralisFasciaStepCount;
            std::optional<std::uint32_t> requestedCameraIndex;
            std::optional<std::uint32_t> wholeBodyActivationSweeps;
            std::vector<std::pair<std::uint32_t, double>> requestedPoseCoordinates;
            std::uint32_t frameDimension = kDefaultFrameDimension;
            std::vector<std::string> positional;
            for (int index = 1; index < argc; ++index) {
                const std::string argument{argv[index]};
                if (argument == "--muscle-step-seconds") {
                    require(index + 1 < argc && !muscleStepSeconds.has_value(),
                            "--muscle-step-seconds requires one value and may be given only once");
                    muscleStepSeconds.emplace(parseMuscleStepSeconds(argv[++index]));
                } else if (argument == "--muscle-step-count") {
                    require(index + 1 < argc && !muscleStepCount.has_value(),
                            "--muscle-step-count requires one value and may be given only once");
                    muscleStepCount.emplace(parseMuscleStepCount(argv[++index]));
                } else if (argument == "--muscle-activation") {
                    require(index + 1 < argc && !muscleActivation.has_value(),
                            "--muscle-activation requires one value and may be given only once");
                    muscleActivation.emplace(parseMuscleActivation(argv[++index]));
                } else if (argument == "--persistent-metal-stand") {
                    require(!persistentMetalStand,
                            "--persistent-metal-stand may be given only once");
                    persistentMetalStand = true;
                } else if (argument == "--selected-tendon-control") {
                    require(!selectedTendonControl,
                            "--selected-tendon-control may be given only once");
                    selectedTendonControl = true;
                } else if (argument == "--stand-root-assistance") {
                    require(!standRootAssistance,
                            "--stand-root-assistance may be given only once");
                    standRootAssistance = true;
                } else if (argument == "--stand-remove-assistance") {
                    require(!standRemoveAssistance,
                            "--stand-remove-assistance may be given only once");
                    standRemoveAssistance = true;
                } else if (argument == "--stand-deterministic-replay") {
                    require(!standDeterministicReplay,
                            "--stand-deterministic-replay may be given only once");
                    standDeterministicReplay = true;
                } else if (argument == "--bilateral-achilles-certificate") {
                    require(!bilateralAchillesCertificate,
                            "--bilateral-achilles-certificate may be given only once");
                    bilateralAchillesCertificate = true;
                } else if (argument == "--bilateral-plantar-fascia-certificate") {
                    require(!bilateralPlantarFasciaCertificate,
                            "--bilateral-plantar-fascia-certificate may be given only once");
                    bilateralPlantarFasciaCertificate = true;
                } else if (argument == "--whole-body-support-certificate") {
                    require(!wholeBodySupportCertificate,
                            "--whole-body-support-certificate may be given only once");
                    wholeBodySupportCertificate = true;
                } else if (argument == "--whole-body-activation-sweeps") {
                    require(index + 1 < argc &&
                                !wholeBodyActivationSweeps.has_value(),
                            "--whole-body-activation-sweeps requires one "
                            "value and may be given only once");
                    wholeBodyActivationSweeps.emplace(
                        parseWholeBodyActivationSweeps(argv[++index]));
                } else if (argument == "--source-passive-joint-tissue") {
                    require(!sourcePassiveJointTissue,
                            "--source-passive-joint-tissue may be given only once");
                    sourcePassiveJointTissue = true;
                } else if (argument == "--whole-body-all-residuals") {
                    require(!wholeBodyAllResiduals,
                            "--whole-body-all-residuals may be given only once");
                    wholeBodyAllResiduals = true;
                } else if (argument == "--fifth-mcp-lower-stop-counterfactual") {
                    require(!fifthMcpLowerStopCounterfactualRequested,
                            "--fifth-mcp-lower-stop-counterfactual may be given only once");
                    fifthMcpLowerStopCounterfactualRequested = true;
                } else if (argument == "--activated-source-muscle-index") {
                    require(index + 1 < argc,
                            "--activated-source-muscle-index requires one muscle index");
                    selectedSourceMuscleActivations.push_back(
                        parseSourceRouteIndex(argv[++index])
                    );
                } else if (argument == "--source-route-centrelines") {
                    require(!sourceRouteCentrelines,
                            "--source-route-centrelines may be given only once");
                    sourceRouteCentrelines = true;
                } else if (argument == "--source-route-index") {
                    require(index + 1 < argc,
                            "--source-route-index requires one muscle index");
                    sourceRouteCentrelines = true;
                    requestedSourceRouteMuscles.push_back(parseSourceRouteIndex(argv[++index]));
                } else if (argument == "--surface-project-source-sites") {
                    require(!surfaceProjectSourceSites,
                            "--surface-project-source-sites may be given only once");
                    surfaceProjectSourceSites = true;
                } else if (argument == "--soft-tissue-stable-id") {
                    require(index + 1 < argc,
                            "--soft-tissue-stable-id requires one source stable ID");
                    requestedSoftTissueStableIds.push_back(
                        parseSourceRouteIndex(argv[++index])
                    );
                } else if (argument == "--zanatomy-calf-visual-supplement") {
                    require(!zAnatomyCalfVisualSupplement,
                            "--zanatomy-calf-visual-supplement may be given only once");
                    zAnatomyCalfVisualSupplement = true;
                } else if (argument == "--tendon-attachment-collar-diagnostic") {
                    require(!tendonAttachmentCollarDiagnostic,
                            "--tendon-attachment-collar-diagnostic may be given only once");
                    tendonAttachmentCollarDiagnostic = true;
                } else if (argument == "--visible-bone-body-index") {
                    require(index + 1 < argc,
                            "--visible-bone-body-index requires one articulated body index");
                    requestedBoneBodyIndices.push_back(
                        parseSourceRouteIndex(argv[++index])
                    );
                } else if (argument == "--visible-bone-stable-id") {
                    require(index + 1 < argc,
                            "--visible-bone-stable-id requires one NHBONES1 stable ID");
                    requestedBoneStableIds.push_back(
                        parseSourceRouteIndex(argv[++index])
                    );
                } else if (argument == "--focus-body-index") {
                    require(index + 1 < argc && !focusBodyIndex.has_value(),
                            "--focus-body-index requires one body index and may be given only once");
                    focusBodyIndex.emplace(parseSourceRouteIndex(argv[++index]));
                } else if (argument == "--focus-joint-child-body-index") {
                    require(index + 1 < argc && !focusJointChildBodyIndex.has_value(),
                            "--focus-joint-child-body-index requires one body index and may be given only once");
                    focusJointChildBodyIndex.emplace(parseSourceRouteIndex(argv[++index]));
                } else if (argument == "--focus-distance-m") {
                    require(index + 1 < argc && !focusDistanceMeters.has_value(),
                            "--focus-distance-m requires one value and may be given only once");
                    focusDistanceMeters.emplace(parseFocusDistanceMeters(argv[++index]));
                } else if (argument == "--soft-tissue-payload") {
                    require(index + 1 < argc && !softTissuePayloadPath.has_value(),
                            "--soft-tissue-payload requires one path and may be given only once");
                    softTissuePayloadPath.emplace(argv[++index]);
                } else if (argument == "--open-knee-payload") {
                    require(index + 1 < argc && !openKneePayloadPath.has_value(),
                            "--open-knee-payload requires one path and may be given only once");
                    openKneePayloadPath.emplace(argv[++index]);
                } else if (argument == "--open-knee-live-tissue-fem") {
                    require(!openKneeLiveTissueFEM,
                            "--open-knee-live-tissue-fem may be given only once");
                    openKneeLiveTissueFEM = true;
                } else if (argument == "--open-knee-tissue-fem-snapshot" ||
                           argument == "--open-knee-ligament-fem-snapshot") {
                    require(index + 1 < argc && !openKneeLigamentFEMPath.has_value(),
                            "--open-knee-tissue-fem-snapshot requires one NHKFEM1/2 path and may be given only once");
                    openKneeLigamentFEMPath.emplace(argv[++index]);
                } else if (argument == "--skin-payload") {
                    require(index + 1 < argc && !skinPayloadPath.has_value(),
                            "--skin-payload requires one path and may be given only once");
                    skinPayloadPath.emplace(argv[++index]);
                } else if (argument == "--torso-anatomy-payload") {
                    require(index + 1 < argc && !torsoAnatomyPayloadPath.has_value(),
                            "--torso-anatomy-payload requires one path and may be given only once");
                    torsoAnatomyPayloadPath.emplace(argv[++index]);
                } else if (argument == "--support-contact-payload") {
                    require(index + 1 < argc && !supportContactPayloadPath.has_value(),
                            "--support-contact-payload requires one path and may be given only once");
                    supportContactPayloadPath.emplace(argv[++index]);
                } else if (argument == "--tendon-payload") {
                    require(index + 1 < argc && !tendonPayloadPath.has_value(),
                            "--tendon-payload requires one path and may be given only once");
                    tendonPayloadPath.emplace(argv[++index]);
                } else if (argument == "--joint-equality-payload") {
                    require(index + 1 < argc &&
                                !jointEqualityPayloadPath.has_value(),
                            "--joint-equality-payload requires one path and may be given only once");
                    jointEqualityPayloadPath.emplace(argv[++index]);
                } else if (argument == "--passive-fem-tissue-stable-id") {
                    require(index + 1 < argc && !passiveFEMTissueStableId.has_value(),
                            "--passive-fem-tissue-stable-id requires one source stable ID and may be given only once");
                    passiveFEMTissueStableId.emplace(parseSourceRouteIndex(argv[++index]));
                } else if (argument == "--passive-fem-step-count") {
                    require(index + 1 < argc && !passiveFEMStepCount.has_value(),
                            "--passive-fem-step-count requires one count and may be given only once");
                    passiveFEMStepCount.emplace(parseMuscleStepCount(argv[++index]));
                } else if (argument == "--passive-fem-metallib") {
                    require(index + 1 < argc && !passiveFEMMetallibPath.has_value(),
                            "--passive-fem-metallib requires one path and may be given only once");
                    passiveFEMMetallibPath.emplace(argv[++index]);
                } else if (argument == "--pectoralis-fascia-payload") {
                    require(index + 1 < argc && !pectoralisFasciaPayloadPath.has_value(),
                            "--pectoralis-fascia-payload requires one path and may be given only once");
                    pectoralisFasciaPayloadPath.emplace(argv[++index]);
                } else if (argument == "--pectoralis-fascia-step-count") {
                    require(index + 1 < argc && !pectoralisFasciaStepCount.has_value(),
                            "--pectoralis-fascia-step-count requires one count and may be given only once");
                    pectoralisFasciaStepCount.emplace(parseMuscleStepCount(argv[++index]));
                } else if (argument == "--dimension") {
                    require(index + 1 < argc && frameDimension == kDefaultFrameDimension,
                            "--dimension requires one value and may be given only once");
                    frameDimension = parseFrameDimension(argv[++index]);
                } else if (argument == "--camera-index") {
                    require(index + 1 < argc && !requestedCameraIndex.has_value(),
                            "--camera-index requires one value and may be given only once");
                    requestedCameraIndex.emplace(parseCameraIndex(argv[++index]));
                } else if (argument == "--pose-q") {
                    require(index + 2 < argc,
                            "--pose-q requires one q index and one coordinate value");
                    const std::uint32_t qIndex = parseSourceRouteIndex(argv[++index]);
                    const double qValue = parsePoseCoordinate(argv[++index]);
                    requestedPoseCoordinates.emplace_back(qIndex, qValue);
                } else if (!argument.starts_with("--")) {
                    positional.push_back(argument);
                } else {
                    throw std::runtime_error("unknown visual option " + argument);
                }
            }
            if (positional.size() != 3u && positional.size() != 4u) {
                std::cerr << "usage: " << argv[0]
                          << " <myosim-fullbody-core-reference.nhrigid>"
                          << " <myosim-fullbody-muscle-reference.nhmyo>"
                          << " [bodyparts3d-myosim-major-bones.nhbones] <output-directory>"
                          << " [--muscle-step-seconds <1e-6..1e-3>]"
                          << " [--muscle-step-count <1..64>]"
                          << " [--muscle-activation <0..1>]"
                          << " [--persistent-metal-stand] [--selected-tendon-control] [--stand-root-assistance] [--stand-remove-assistance] [--stand-deterministic-replay]"
                          << " [--bilateral-achilles-certificate]"
                          << " [--bilateral-plantar-fascia-certificate]"
                          << " [--whole-body-support-certificate]"
                          << " [--whole-body-activation-sweeps <1..8192>]"
                          << " [--whole-body-all-residuals]"
                          << " [--fifth-mcp-lower-stop-counterfactual]"
                          << " [--source-passive-joint-tissue]"
                          << " [--activated-source-muscle-index <0..415>]..."
                          << " [--source-route-centrelines] [--source-route-index <0..415>]..."
                          << " [--surface-project-source-sites]"
                          << " [--soft-tissue-payload <NHTISS2-or-NHTISS3-or-NHTISS4>]"
                          << " [--open-knee-payload <NHKNEE1>]"
                          << " [--open-knee-live-tissue-fem]"
                          << " [--open-knee-tissue-fem-snapshot <NHKFEM1-or-NHKFEM2>]"
                          << " [--skin-payload <NHSKIN1>]"
                          << " [--torso-anatomy-payload <NHANAT1>]"
                          << " [--passive-fem-tissue-stable-id <1..N>]"
                          << " [--passive-fem-step-count <1..64>]"
                          << " [--passive-fem-metallib <NumiMatter.metallib>]"
                          << " [--pectoralis-fascia-payload <NHFASC2>]"
                          << " [--pectoralis-fascia-step-count <1..64>]"
                          << " [--visible-bone-body-index <0..156>]..."
                          << " [--visible-bone-stable-id <1..NHBONES1 bone count>]..."
                          << " [--soft-tissue-stable-id <1..N>]..."
                          << " [--zanatomy-calf-visual-supplement]"
                          << " [--tendon-attachment-collar-diagnostic]"
                          << " [--support-contact-payload <NHCNT1>]"
                          << " [--tendon-payload <NHTENDON1-or-NHTENDON2-or-NHTENDON3>]"
                          << " [--joint-equality-payload <NHEQ1>]"
                          << " [--focus-body-index <0..156>]"
                          << " [--focus-joint-child-body-index <1..156>] [--focus-distance-m <0.08..0.80>]"
                          << " [--camera-index <0..3>]"
                          << " [--pose-q <q-index> <coordinate-value>]..."
                          << " [--dimension <512..2048; multiple-of-64>]\n";
                return 2;
            }
            const bool bodypartsBoneVisual = positional.size() == 4u;
            const LoadedRigid rigid = loadRigid(positional[0]);
            LoadedMuscles musclePayload = loadMuscles(positional[1], rigid.header);
            const bool compliantMusclePayload =
                musclePayload.header.payloadAbi == kMusclePayloadAbi;
            if (tendonPayloadPath.has_value()) {
                applyNumiHumanTendonPayload(*tendonPayloadPath, rigid, musclePayload);
                std::cout << "tendon_payload=NHTENDON" << musclePayload.tendonPayload.payloadAbi
                          << " tendon_endpoints="
                          << musclePayload.tendonPointBindings + musclePayload.tendonTriangleBindings +
                                 musclePayload.tendonEnvelopeBindings
                          << " tendon_point_bindings=" << musclePayload.tendonPointBindings
                          << " tendon_triangle_bindings=" << musclePayload.tendonTriangleBindings
                          << " tendon_envelope_bindings=" << musclePayload.tendonEnvelopeBindings
                          << " tendon_migrated_envelope_bindings="
                          << musclePayload.tendonMigratedEnvelopeBindings
                          << " tendon_max_reference_path_delta_m="
                          << musclePayload.maximumTendonReferencePathDelta
                          << " tendon_max_architecture_scale_change="
                          << musclePayload.maximumTendonArchitectureScaleChange
                          << "\n";
            }
            require(!focusBodyIndex.has_value() || *focusBodyIndex < rigid.header.engineBodyCount,
                    "--focus-body-index exceeds the source body count");
            require(!(focusBodyIndex.has_value() && focusJointChildBodyIndex.has_value()),
                    "--focus-body-index and --focus-joint-child-body-index are mutually exclusive");
            require(!focusJointChildBodyIndex.has_value() ||
                        (*focusJointChildBodyIndex > 0u &&
                         *focusJointChildBodyIndex < rigid.header.engineBodyCount),
                    "--focus-joint-child-body-index exceeds the source body count");
            require(!focusDistanceMeters.has_value() || focusJointChildBodyIndex.has_value(),
                    "--focus-distance-m requires --focus-joint-child-body-index");
            std::sort(requestedSourceRouteMuscles.begin(), requestedSourceRouteMuscles.end());
            const auto duplicate = std::adjacent_find(
                requestedSourceRouteMuscles.begin(), requestedSourceRouteMuscles.end()
            );
            require(duplicate == requestedSourceRouteMuscles.end(),
                    "--source-route-index values must be unique");
            require(std::all_of(
                        requestedSourceRouteMuscles.begin(), requestedSourceRouteMuscles.end(),
                        [&musclePayload](const std::uint32_t index) {
                            return index < musclePayload.referenceMuscles.size();
                        }
                    ),
                    "--source-route-index exceeds the source muscle count");
            std::sort(
                selectedSourceMuscleActivations.begin(), selectedSourceMuscleActivations.end()
            );
            const auto duplicateActivation = std::adjacent_find(
                selectedSourceMuscleActivations.begin(), selectedSourceMuscleActivations.end()
            );
            require(duplicateActivation == selectedSourceMuscleActivations.end(),
                    "--activated-source-muscle-index values must be unique");
            require(std::all_of(
                        selectedSourceMuscleActivations.begin(),
                        selectedSourceMuscleActivations.end(),
                        [&musclePayload](const std::uint32_t index) {
                            return index < musclePayload.referenceMuscles.size();
                        }
                    ),
                    "--activated-source-muscle-index exceeds the source muscle count");
            if (bilateralAchillesCertificate) {
                constexpr std::array<std::uint32_t, 6u>
                    kBilateralAchillesMuscles{
                        348u, 349u, 369u, 388u, 389u, 409u,
                    };
                require(
                    !persistentMetalStand && selectedTendonControl &&
                        muscleStepSeconds.has_value() &&
                        tendonPayloadPath.has_value() &&
                        jointEqualityPayloadPath.has_value() &&
                        supportContactPayloadPath.has_value() &&
                        selectedSourceMuscleActivations.size() ==
                            kBilateralAchillesMuscles.size() &&
                        std::equal(
                            selectedSourceMuscleActivations.begin(),
                            selectedSourceMuscleActivations.end(),
                            kBilateralAchillesMuscles.begin()
                        ),
                    "--bilateral-achilles-certificate requires the persistent "
                    "selected NHTENDON3 transaction and exactly source muscles "
                    "348,349,369,388,389,409"
                );
                require(
                    !bilateralPlantarFasciaCertificate &&
                        !wholeBodySupportCertificate &&
                        !sourceRouteCentrelines &&
                        requestedBoneBodyIndices.empty() &&
                        requestedBoneStableIds.empty() &&
                        requestedSoftTissueStableIds.empty() &&
                        !passiveFEMTissueStableId.has_value() &&
                        !pectoralisFasciaPayloadPath.has_value() &&
                        !openKneePayloadPath.has_value() &&
                        !openKneeLigamentFEMPath.has_value(),
                    "--bilateral-achilles-certificate is a nonvisual mechanics "
                    "qualification and cannot be combined with presentation scopes"
                );
            }
            if (bilateralPlantarFasciaCertificate) {
                require(
                        !bilateralAchillesCertificate &&
                        !wholeBodySupportCertificate && bodypartsBoneVisual &&
                        !persistentMetalStand && !selectedTendonControl &&
                        muscleStepSeconds.has_value() &&
                        !muscleStepCount.has_value() &&
                        !muscleActivation.has_value() &&
                        selectedSourceMuscleActivations.empty() &&
                        jointEqualityPayloadPath.has_value() &&
                        supportContactPayloadPath.has_value(),
                    "--bilateral-plantar-fascia-certificate requires the "
                    "BodyParts3D bone payload, NHEQ1, NHCNT1, a response "
                    "timestep, and no muscle activation or horizon override"
                );
                require(
                    !sourceRouteCentrelines &&
                        requestedBoneBodyIndices.empty() &&
                        requestedBoneStableIds.empty() &&
                        requestedSoftTissueStableIds.empty() &&
                        !softTissuePayloadPath.has_value() &&
                        !skinPayloadPath.has_value() &&
                        !torsoAnatomyPayloadPath.has_value() &&
                        !passiveFEMTissueStableId.has_value() &&
                        !passiveFEMMetallibPath.has_value() &&
                        !pectoralisFasciaPayloadPath.has_value() &&
                        !openKneePayloadPath.has_value() &&
                        !openKneeLigamentFEMPath.has_value() &&
                        !openKneeLiveTissueFEM &&
                        requestedPoseCoordinates.empty(),
                    "--bilateral-plantar-fascia-certificate is a nonvisual "
                    "mechanics qualification and cannot be combined with "
                    "presentation or another continuum scope"
                );
            }
            if (wholeBodySupportCertificate) {
                require(
                    !bilateralAchillesCertificate &&
                        !bilateralPlantarFasciaCertificate &&
                        !persistentMetalStand && !selectedTendonControl &&
                        muscleStepSeconds.has_value() &&
                        !muscleStepCount.has_value() &&
                        !muscleActivation.has_value() &&
                        selectedSourceMuscleActivations.empty() &&
                        supportContactPayloadPath.has_value() &&
                        jointEqualityPayloadPath.has_value() &&
                        !sourceRouteCentrelines &&
                        requestedBoneBodyIndices.empty() &&
                        requestedBoneStableIds.empty() &&
                        requestedSoftTissueStableIds.empty() &&
                        !softTissuePayloadPath.has_value() &&
                        !skinPayloadPath.has_value() &&
                        !torsoAnatomyPayloadPath.has_value() &&
                        !passiveFEMTissueStableId.has_value() &&
                        !pectoralisFasciaPayloadPath.has_value() &&
                        !openKneePayloadPath.has_value() &&
                        !openKneeLigamentFEMPath.has_value() &&
                        requestedPoseCoordinates.empty(),
                    "--whole-body-support-certificate requires NHCNT1, "
                    "NHEQ1, a response timestep, and no presentation, "
                    "activation, or continuum scope"
                );
            }
            require(!sourcePassiveJointTissue || wholeBodySupportCertificate,
                    "--source-passive-joint-tissue requires "
                    "--whole-body-support-certificate");
            require(!wholeBodyActivationSweeps.has_value() ||
                        wholeBodySupportCertificate,
                    "--whole-body-activation-sweeps requires "
                    "--whole-body-support-certificate");
            require(!wholeBodyAllResiduals || wholeBodySupportCertificate,
                    "--whole-body-all-residuals requires "
                    "--whole-body-support-certificate");
            require(!fifthMcpLowerStopCounterfactualRequested ||
                        (wholeBodySupportCertificate && sourcePassiveJointTissue),
                    "--fifth-mcp-lower-stop-counterfactual requires "
                    "--whole-body-support-certificate and "
                    "--source-passive-joint-tissue");
            std::sort(requestedBoneBodyIndices.begin(), requestedBoneBodyIndices.end());
            const auto duplicateBoneBody = std::adjacent_find(
                requestedBoneBodyIndices.begin(), requestedBoneBodyIndices.end()
            );
            require(duplicateBoneBody == requestedBoneBodyIndices.end(),
                    "--visible-bone-body-index values must be unique");
            require(std::all_of(
                        requestedBoneBodyIndices.begin(), requestedBoneBodyIndices.end(),
                        [&rigid](const std::uint32_t index) {
                            return index < rigid.header.engineBodyCount;
                        }
                    ),
                    "--visible-bone-body-index exceeds the source body count");
            std::sort(requestedBoneStableIds.begin(), requestedBoneStableIds.end());
            const auto duplicateBoneStableId = std::adjacent_find(
                requestedBoneStableIds.begin(), requestedBoneStableIds.end()
            );
            require(duplicateBoneStableId == requestedBoneStableIds.end(),
                    "--visible-bone-stable-id values must be unique");
            std::sort(
                requestedSoftTissueStableIds.begin(),
                requestedSoftTissueStableIds.end()
            );
            const auto duplicateSoftTissue = std::adjacent_find(
                requestedSoftTissueStableIds.begin(), requestedSoftTissueStableIds.end()
            );
            require(duplicateSoftTissue == requestedSoftTissueStableIds.end(),
                    "--soft-tissue-stable-id values must be unique");
            std::optional<LoadedBones> bonePayload;
            if (bodypartsBoneVisual) {
                bonePayload.emplace(loadBones(positional[2], rigid.header));
            }
            std::optional<metalrobo::NumiHumanKneePayload> openKneePayload;
            if (openKneePayloadPath.has_value()) {
                require(bodypartsBoneVisual,
                        "--open-knee-payload requires the BodyParts3D bone positional payload");
                openKneePayload.emplace(loadOpenKnee(*openKneePayloadPath, rigid.header));
            }
            std::optional<LoadedOpenKneeLigamentFEM> openKneeLigamentFEM;
            if (openKneeLigamentFEMPath.has_value()) {
                require(openKneePayload.has_value(),
                        "--open-knee-ligament-fem-snapshot requires --open-knee-payload");
                openKneeLigamentFEM.emplace(loadOpenKneeLigamentFEM(
                    *openKneeLigamentFEMPath, *openKneePayload));
                std::cout << "open_knee_tissue_fem_snapshot=NHKFEM"
                          << openKneeLigamentFEM->header.abi
                          << " regions=" << openKneeLigamentFEM->header.regionCount
                          << " nodes=" << openKneeLigamentFEM->header.nodeCount
                          << " tetrahedra="
                          << openKneeLigamentFEM->header.tetrahedronCount
                          << " maximum_displacement_m="
                          << openKneeLigamentFEM->maximumDisplacementMeters
                          << " pose_boundary=accepted_submicron_preflight_only\n";
            }
            require(
                (requestedBoneBodyIndices.empty() && requestedBoneStableIds.empty()) ||
                    bonePayload.has_value(),
                "bone visibility selection requires a BodyParts3D bone payload"
            );
            if (!requestedBoneBodyIndices.empty()) {
                for (const std::uint32_t bodyIndex : requestedBoneBodyIndices) {
                    const bool present = std::any_of(
                        bonePayload->records.begin(), bonePayload->records.end(),
                        [bodyIndex](const BoneRecord& bone) { return bone.bodyIndex == bodyIndex; }
                    );
                    require(present,
                            "--visible-bone-body-index has no source mesh in the supplied payload");
                }
            }
            if (!requestedBoneStableIds.empty()) {
                for (const std::uint32_t stableId : requestedBoneStableIds) {
                    const bool present = std::any_of(
                        bonePayload->records.begin(), bonePayload->records.end(),
                        [stableId](const BoneRecord& bone) {
                            return bone.stableId == stableId;
                        }
                    );
                    require(present,
                            "--visible-bone-stable-id has no source mesh in the supplied payload");
                }
            }
            std::optional<LoadedSoftTissues> softTissuePayload;
            if (softTissuePayloadPath.has_value()) {
                require(bodypartsBoneVisual,
                        "--soft-tissue-payload requires a BodyParts3D bone payload");
                softTissuePayload.emplace(loadSoftTissues(*softTissuePayloadPath, rigid.header));
                require(
                    bonePayload->header.reserved0 == softTissuePayload->header.reserved0,
                    "BodyParts3D bone and soft-tissue payloads have different visual registrations"
                );
            }
            std::optional<LoadedPectoralisFascia> pectoralisFasciaPayload;
            if (pectoralisFasciaPayloadPath.has_value()) {
                require(softTissuePayload.has_value(),
                        "--pectoralis-fascia-payload requires --soft-tissue-payload");
                pectoralisFasciaPayload.emplace(loadPectoralisFascia(
                    *pectoralisFasciaPayloadPath, *softTissuePayload, musclePayload
                ));
            }
            std::optional<LoadedSkin> skinPayload;
            if (skinPayloadPath.has_value()) {
                require(bodypartsBoneVisual,
                        "--skin-payload requires a BodyParts3D bone payload");
                skinPayload.emplace(loadSkin(
                    *skinPayloadPath, rigid.header, bonePayload->header.reserved0
                ));
            }
            std::optional<LoadedTorsoAnatomy> torsoAnatomyPayload;
            if (torsoAnatomyPayloadPath.has_value()) {
                require(bodypartsBoneVisual,
                        "--torso-anatomy-payload requires a BodyParts3D bone payload");
                torsoAnatomyPayload.emplace(loadTorsoAnatomy(
                    *torsoAnatomyPayloadPath, rigid.header, bonePayload->header.reserved0
                ));
            }
            require(requestedSoftTissueStableIds.empty() || softTissuePayload.has_value(),
                    "--soft-tissue-stable-id requires --soft-tissue-payload");
            require(!zAnatomyCalfVisualSupplement || softTissuePayload.has_value(),
                    "--zanatomy-calf-visual-supplement requires --soft-tissue-payload");
            if (!requestedSoftTissueStableIds.empty()) {
                for (const std::uint32_t stableId : requestedSoftTissueStableIds) {
                    const bool present = std::any_of(
                        softTissuePayload->records.begin(), softTissuePayload->records.end(),
                        [stableId](const SoftTissueRecord& tissue) {
                            return tissue.stableId == stableId;
                        }
                    );
                    require(present,
                            "--soft-tissue-stable-id is not present in the supplied payload");
                }
            }
            if (zAnatomyCalfVisualSupplement) {
                constexpr std::array<std::uint32_t, 5u> kExpectedCalfSurfaceIds{1u, 2u, 3u, 4u, 5u};
                require(softTissuePayload->records.size() == kExpectedCalfSurfaceIds.size(),
                        "Z-Anatomy calf supplement must contain four scoped right-calf surfaces and one calcaneus overlay");
                require(requestedSoftTissueStableIds.size() == kExpectedCalfSurfaceIds.size() &&
                            std::equal(
                                requestedSoftTissueStableIds.begin(), requestedSoftTissueStableIds.end(),
                                kExpectedCalfSurfaceIds.begin()
                            ),
                        "Z-Anatomy calf supplement must render all four scoped right-calf surfaces and its calcaneus overlay");
                for (const std::uint32_t stableId : kExpectedCalfSurfaceIds) {
                    const auto selected = std::find_if(
                        softTissuePayload->records.begin(), softTissuePayload->records.end(),
                        [stableId](const SoftTissueRecord& tissue) { return tissue.stableId == stableId; }
                    );
                    const std::uint32_t expectedLayer = stableId == 4u
                        ? kSoftTissueLayerTendon
                        : (stableId == 5u
                            ? kSoftTissueLayerSupplementalBone
                            : kSoftTissueLayerMuscle);
                    require(selected != softTissuePayload->records.end() &&
                                selected->layer == expectedLayer,
                            "Z-Anatomy calf supplement source layers do not match its fixed calf scope");
                }
            }
            std::optional<LoadedSupportContacts> supportContactPayload;
            if (supportContactPayloadPath.has_value()) {
                require(muscleStepSeconds.has_value(),
                        "--support-contact-payload requires --muscle-step-seconds");
                supportContactPayload.emplace(loadSupportContacts(
                    *supportContactPayloadPath, rigid.header
                ));
            }
            std::optional<LoadedJointEqualities> jointEqualityPayload;
            if (jointEqualityPayloadPath.has_value()) {
                jointEqualityPayload.emplace(loadJointEqualities(
                    *jointEqualityPayloadPath, rigid.header
                ));
                std::cout << "joint_equality_payload=NHEQ1 joint_equalities="
                          << jointEqualityPayload->payload.records.size()
                          << "\n";
            }
            require(!surfaceProjectSourceSites ||
                        (bodypartsBoneVisual && sourceRouteCentrelines),
                    "--surface-project-source-sites requires BodyParts3D bones and a source-route inspection");
            require(!muscleActivation.has_value() || muscleStepSeconds.has_value(),
                    "--muscle-activation requires --muscle-step-seconds");
            require(!muscleStepCount.has_value() || muscleStepSeconds.has_value(),
                    "--muscle-step-count requires --muscle-step-seconds");
            require(selectedSourceMuscleActivations.empty() || muscleStepSeconds.has_value(),
                    "--activated-source-muscle-index requires --muscle-step-seconds");
            require(requestedPoseCoordinates.empty() ||
                        (!muscleStepSeconds.has_value() && jointEqualityPayload.has_value()),
                    "--pose-q requires NHEQ1 joint equalities and cannot be combined with muscle stepping");
            require(!openKneeLigamentFEM.has_value() ||
                        (requestedPoseCoordinates.empty() &&
                         !muscleStepSeconds.has_value()),
                    "NHKFEM1/2 is an accepted neutral preflight snapshot and cannot be rendered as an arbitrary pose");
            require(!openKneeLiveTissueFEM ||
                        (openKneePayload.has_value() &&
                         (persistentMetalStand || selectedTendonControl) &&
                         muscleStepSeconds.has_value() &&
                         supportContactPayload.has_value() &&
                         jointEqualityPayload.has_value() &&
                         (musclePayload.tendonPayload.payloadAbi == 2u ||
                          musclePayload.tendonPayload.payloadAbi == 3u)),
                    "--open-knee-live-tissue-fem requires NHKNEE1 and a persistent NHTENDON2/3 Human transaction");
            require(!openKneeLiveTissueFEM ||
                        (!openKneeLigamentFEM.has_value() &&
                         !pectoralisFasciaPayload.has_value()),
                    "--open-knee-live-tissue-fem cannot share the single continuum slot with NHKFEM1/2 or pectoralis fascia");
            require(!persistentMetalStand ||
                        (muscleStepSeconds.has_value() &&
                         supportContactPayload.has_value() &&
                         jointEqualityPayload.has_value()),
                    "--persistent-metal-stand requires muscle timestep, support contacts, and NHEQ1 joint equalities");
            require(!selectedTendonControl ||
                        (muscleStepSeconds.has_value() &&
                         supportContactPayload.has_value() &&
                         jointEqualityPayload.has_value() &&
                         !selectedSourceMuscleActivations.empty()),
                    "--selected-tendon-control requires muscle timestep, support contacts, NHEQ1 joint equalities, and selected source muscles");
            require(!persistentMetalStand || !selectedTendonControl,
                    "--persistent-metal-stand and --selected-tendon-control are mutually exclusive");
            require(!standRootAssistance || persistentMetalStand,
                    "--stand-root-assistance requires --persistent-metal-stand");
            require(!standRemoveAssistance || standRootAssistance,
                    "--stand-remove-assistance requires --stand-root-assistance");
            require(!standDeterministicReplay || persistentMetalStand,
                    "--stand-deterministic-replay requires --persistent-metal-stand");
            require(!passiveFEMTissueStableId.has_value() ||
                        (softTissuePayload.has_value() && muscleStepSeconds.has_value()),
                    "--passive-fem-tissue-stable-id requires --soft-tissue-payload and --muscle-step-seconds");
            require(!pectoralisFasciaPayload.has_value() ||
                        ((persistentMetalStand || selectedTendonControl) &&
                         muscleStepSeconds.has_value() &&
                         (musclePayload.tendonPayload.payloadAbi == 2u ||
                          musclePayload.tendonPayload.payloadAbi == 3u)),
                    "--pectoralis-fascia-payload requires a persistent NHTENDON2/3 muscle transaction");
            require(!pectoralisFasciaStepCount.has_value() ||
                        pectoralisFasciaPayload.has_value(),
                    "--pectoralis-fascia-step-count requires --pectoralis-fascia-payload");
            require(!pectoralisFasciaStepCount.has_value() ||
                        *pectoralisFasciaStepCount ==
                            muscleStepCount.value_or(1u),
                    "--pectoralis-fascia-step-count must equal the owning Human muscle horizon");
            require(!passiveFEMStepCount.has_value() || passiveFEMTissueStableId.has_value(),
                    "--passive-fem-step-count requires --passive-fem-tissue-stable-id");
            require(!passiveFEMMetallibPath.has_value() || passiveFEMTissueStableId.has_value(),
                    "--passive-fem-metallib requires --passive-fem-tissue-stable-id");
            if (passiveFEMMetallibPath.has_value()) {
                require(std::filesystem::is_regular_file(*passiveFEMMetallibPath),
                        "--passive-fem-metallib is not a regular file");
            }
            if (passiveFEMTissueStableId.has_value()) {
                const auto selected = std::find_if(
                    softTissuePayload->records.begin(), softTissuePayload->records.end(),
                    [stableId = *passiveFEMTissueStableId](const SoftTissueRecord& tissue) {
                        return tissue.stableId == stableId;
                    }
                );
                require(selected != softTissuePayload->records.end() &&
                            selected->layer == kSoftTissueLayerMuscle,
                        "--passive-fem-tissue-stable-id must name a source muscle surface");
            }
            if (wholeBodySupportCertificate) {
                require(supportContactPayload.has_value() &&
                            jointEqualityPayload.has_value(),
                        "whole-body support payloads were not loaded");
                const GroundAlignedSupport aligned = makeGroundAlignedSupport(
                    rigid.model, *supportContactPayload);
                const auto passiveCouplings = sourcePassiveJointTissue
                    ? wholeBodyUpperPassiveCoordinateCouplings()
                    : std::vector<
                        metalrobo::NumiHumanPassiveCoordinateCoupling>{};
                const CompiledStandActivation support =
                    compileStaticStandActivation(
                        rigid.model, musclePayload, *jointEqualityPayload,
                        aligned.q, 1.0, {}, &*supportContactPayload,
                        passiveCouplings,
                        wholeBodyActivationSweeps.value_or(240u));
                const CompiledStandActivation replaySupport =
                    compileStaticStandActivation(
                        rigid.model, musclePayload, *jointEqualityPayload,
                        aligned.q, 1.0, {}, &*supportContactPayload,
                        passiveCouplings,
                        wholeBodyActivationSweeps.value_or(240u));
                const auto bitwiseEqual = [](const auto& first,
                                             const auto& second) {
                    using Value = typename std::decay_t<decltype(first)>::value_type;
                    return first.size() == second.size() &&
                        (first.empty() || std::memcmp(
                            first.data(), second.data(),
                            first.size() * sizeof(Value)) == 0);
                };
                require(
                    bitwiseEqual(support.q, replaySupport.q) &&
                        bitwiseEqual(support.activation,
                                     replaySupport.activation) &&
                        bitwiseEqual(support.generalizedMuscleForce,
                                     replaySupport.generalizedMuscleForce) &&
                        bitwiseEqual(support.muscleTendonForce,
                                     replaySupport.muscleTendonForce) &&
                        bitwiseEqual(support.passiveMuscleTendonForce,
                                     replaySupport.passiveMuscleTendonForce) &&
                        bitwiseEqual(
                            support.generalizedPassiveCoordinateForce,
                            replaySupport.generalizedPassiveCoordinateForce) &&
                        bitwiseEqual(support.supportNormalForce,
                                     replaySupport.supportNormalForce),
                    "whole-body support compile did not replay bitwise"
                );
                std::optional<CompiledStandActivation>
                    fifthMcpLowerStopCounterfactual;
                if (fifthMcpLowerStopCounterfactualRequested) {
                    std::vector<double> counterfactualQ = aligned.q;
                    for (const std::uint32_t dofIndex : {59u, 97u}) {
                        const MRDofPropertiesGPU& dof =
                            rigid.model.dofs.at(dofIndex);
                        require(
                            dof.qIndex != MR_INVALID_INDEX &&
                                (dof.flags & MR_DOF_FLAG_POSITION_LIMIT) != 0u &&
                                dof.limits.x < 0.0f,
                            "fifth-MCP counterfactual requires bounded source coordinates"
                        );
                        counterfactualQ.at(dof.qIndex) = dof.limits.x;
                    }
                    double equalityProjection = 0.0;
                    const auto projectionDiagnostics =
                        metalrobo::projectNumiHumanJointEqualities(
                            jointEqualityPayload->payload.records,
                            counterfactualQ, &equalityProjection
                        );
                    require(
                        projectionDiagnostics.succeeded() &&
                            std::abs(
                                counterfactualQ.at(
                                    rigid.model.dofs.at(59u).qIndex) -
                                rigid.model.dofs.at(59u).limits.x) <= 1.0e-12 &&
                            std::abs(
                                counterfactualQ.at(
                                    rigid.model.dofs.at(97u).qIndex) -
                                rigid.model.dofs.at(97u).limits.x) <= 1.0e-12,
                        "fifth-MCP lower-stop counterfactual changed a target coordinate"
                    );
                    fifthMcpLowerStopCounterfactual.emplace(
                        compileStaticStandActivation(
                            rigid.model, musclePayload,
                            *jointEqualityPayload, counterfactualQ, 1.0, {},
                            &*supportContactPayload, passiveCouplings,
                            wholeBodyActivationSweeps.value_or(240u)
                        )
                    );
                    const CompiledStandActivation counterfactualReplay =
                        compileStaticStandActivation(
                            rigid.model, musclePayload,
                            *jointEqualityPayload, counterfactualQ, 1.0, {},
                            &*supportContactPayload, passiveCouplings,
                            wholeBodyActivationSweeps.value_or(240u)
                        );
                    require(
                        bitwiseEqual(
                            fifthMcpLowerStopCounterfactual->q,
                            counterfactualReplay.q) &&
                            bitwiseEqual(
                                fifthMcpLowerStopCounterfactual->activation,
                                counterfactualReplay.activation) &&
                            bitwiseEqual(
                                fifthMcpLowerStopCounterfactual->
                                    generalizedForceResidual,
                                counterfactualReplay.
                                    generalizedForceResidual) &&
                            bitwiseEqual(
                                fifthMcpLowerStopCounterfactual->
                                    generalizedPositionLimitForce,
                                counterfactualReplay.
                                    generalizedPositionLimitForce),
                        "fifth-MCP lower-stop counterfactual did not replay bitwise"
                    );
                }
                double bodyMassKilograms = 0.0;
                for (const MRBodyPropertiesGPU& body : rigid.model.bodies) {
                    bodyMassKilograms += body.massAndInverseMass.x;
                }
                const double gravityMagnitude = std::sqrt(
                    rigid.model.world.gravityAndTimestep.x *
                        rigid.model.world.gravityAndTimestep.x +
                    rigid.model.world.gravityAndTimestep.y *
                        rigid.model.world.gravityAndTimestep.y +
                    rigid.model.world.gravityAndTimestep.z *
                        rigid.model.world.gravityAndTimestep.z);
                const double expectedWeightNewtons =
                    bodyMassKilograms * gravityMagnitude;
                const double relativeWeightError = std::abs(
                    support.totalSupportForceNewtons - expectedWeightNewtons) /
                    expectedWeightNewtons;
                if (fifthMcpLowerStopCounterfactual.has_value()) {
                    const auto& counterfactual =
                        *fifthMcpLowerStopCounterfactual;
                    std::cout
                        << " fifth_mcp_lower_stop_counterfactual=true"
                        << " fifth_mcp_lower_stop_right_residual_nm="
                        << counterfactual.generalizedForceResidual[59]
                        << " fifth_mcp_lower_stop_left_residual_nm="
                        << counterfactual.generalizedForceResidual[97]
                        << " fifth_mcp_lower_stop_right_acceleration_rad_s2="
                        << counterfactual.generalizedAccelerationResidual[59]
                        << " fifth_mcp_lower_stop_left_acceleration_rad_s2="
                        << counterfactual.generalizedAccelerationResidual[97]
                        << " fifth_mcp_lower_stop_normalized_residual_rms="
                        << counterfactual.normalizedResidualRms
                        << " fifth_mcp_lower_stop_right_limit_reaction_nm="
                        << counterfactual.generalizedPositionLimitForce[59]
                        << " fifth_mcp_lower_stop_left_limit_reaction_nm="
                        << counterfactual.generalizedPositionLimitForce[97]
                        << " fifth_mcp_lower_stop_replay=bitwise ";
                }
                require(
                    support.supportNormalForce.size() ==
                        supportContactPayload->records.size() &&
                        support.generalizedMuscleForce.size() ==
                            rigid.model.world.nv &&
                        support.generalizedPositionLimitForce.size() ==
                            rigid.model.world.nv &&
                        support.generalizedSupportForce.size() ==
                            rigid.model.world.nv &&
                        support.generalizedPassiveCoordinateForce.size() ==
                            rigid.model.world.nv &&
                        support.gravityTarget.size() == rigid.model.world.nv &&
                        support.generalizedForceResidual.size() ==
                            rigid.model.world.nv &&
                        support.generalizedAccelerationResidual.size() ==
                            rigid.model.world.nv &&
                        support.muscleTendonForce.size() ==
                            musclePayload.referenceMuscles.size() &&
                        support.passiveMuscleTendonForce.size() ==
                            musclePayload.referenceMuscles.size() &&
                        support.supportContactCount ==
                            supportContactPayload->records.size() &&
                        support.activeSupportContactCount > 0u &&
                        std::all_of(
                            support.supportNormalForce.begin(),
                            support.supportNormalForce.end(),
                            [](const double force) {
                                return std::isfinite(force) && force >= 0.0;
                            }) &&
                        std::isfinite(expectedWeightNewtons) &&
                        expectedWeightNewtons > 0.0 &&
                        relativeWeightError <= 1.0e-5 &&
                        support.maximumRootForceResidual <= 1.0e-3,
                    "whole-body unilateral support wrench did not close"
                );
                std::vector<std::uint32_t> residualOrder(
                    rigid.model.world.nv);
                std::iota(residualOrder.begin(), residualOrder.end(), 0u);
                std::stable_sort(
                    residualOrder.begin(), residualOrder.end(),
                    [&support](const std::uint32_t first,
                               const std::uint32_t second) {
                        return std::abs(
                            support.generalizedAccelerationResidual[first]) >
                            std::abs(
                                support.generalizedAccelerationResidual[second]);
                    });
                metalrobo::ArticulatedDynamicsConfig diagnosticDynamics;
                diagnosticDynamics.gravity = {
                    rigid.model.world.gravityAndTimestep.x,
                    rigid.model.world.gravityAndTimestep.y,
                    rigid.model.world.gravityAndTimestep.z,
                };
                diagnosticDynamics.timestep = *muscleStepSeconds;
                const std::vector<double> zeroVelocity(
                    rigid.model.world.nv, 0.0);
                std::vector<metalrobo::MujocoMuscleResult> musclePaths(
                    musclePayload.referenceMuscles.size());
                for (std::size_t muscle = 0u;
                     muscle < musclePaths.size(); ++muscle) {
                    const auto pathDiagnostics =
                        metalrobo::evaluateMujocoMuscle(
                            rigid.model, 0u, support.q, zeroVelocity,
                            musclePayload.referenceSites,
                            musclePayload.referenceWraps,
                            musclePayload.referenceMuscles[muscle], {},
                            musclePaths[muscle], diagnosticDynamics);
                    require(pathDiagnostics.succeeded() &&
                                musclePaths[muscle].path.lengthJacobian.size() ==
                                    rigid.model.world.nv,
                            "whole-body residual muscle decomposition failed");
                }
                std::cout << std::setprecision(12)
                          << "numi_human_whole_body_support_wrench=ok"
                          << " source_model=pinned_MyoSim_full_body"
                          << " support_payload=NHCNT1"
                          << " joint_manifold=NHEQ1"
                          << " passive_joint_tissue="
                          << (sourcePassiveJointTissue
                              ? "linearized_experimental_upper_v1" : "none")
                          << " passive_coordinate_couplings="
                          << passiveCouplings.size()
                          << " body_mass_kg=" << bodyMassKilograms
                          << " expected_weight_n=" << expectedWeightNewtons
                          << " total_support_force_n="
                          << support.totalSupportForceNewtons
                          << " relative_weight_error=" << relativeWeightError
                          << " support_contacts=" << support.supportContactCount
                          << " active_support_contacts="
                          << support.activeSupportContactCount
                          << " max_root_force_residual="
                          << support.maximumRootForceResidual
                          << " max_root_acceleration_residual="
                          << support.maximumRootAccelerationResidual
                          << " internal_normalized_residual_rms="
                          << support.normalizedResidualRms
                          << " activation_sweeps="
                          << support.activationSweeps
                          << " global_activation_polish_iterations="
                          << support.globalActivationPolishIterations
                          << " accepted_global_activation_polish_steps="
                          << support.acceptedGlobalActivationPolishSteps
                          << " accepted_pose_steps="
                          << support.acceptedPoseSteps
                          << " active_position_limits="
                          << support.activePositionLimitCount
                          << " internal_balanced="
                          << (support.balanced ? "true" : "false")
                          << " replay=bitwise";
                for (std::size_t index = 0u;
                     index < support.supportNormalForce.size(); ++index) {
                    std::cout << " contact_" << index << "_body="
                              << supportContactPayload->records[index].bodyIndex
                              << " contact_" << index << "_normal_force_n="
                              << support.supportNormalForce[index];
                }
                constexpr std::size_t kDefaultReportedResidualCount = 12u;
                const std::size_t reportedResidualCount =
                    wholeBodyAllResiduals
                    ? residualOrder.size() : kDefaultReportedResidualCount;
                const MRArticulationGPU& articulation =
                    rigid.model.articulations.front();
                for (std::size_t rank = 0u;
                     rank < std::min(
                         reportedResidualCount, residualOrder.size());
                     ++rank) {
                    const std::uint32_t dof = residualOrder[rank];
                    const std::size_t globalDof = articulation.vOffset + dof;
                    const std::string name =
                        globalDof < rigid.model.dofNames.size()
                        ? rigid.model.dofNames[globalDof]
                        : "unnamed";
                    const MRDofPropertiesGPU& properties =
                        rigid.model.dofs[globalDof];
                    const std::uint32_t childBody =
                        properties.jointIndex < rigid.model.joints.size()
                        ? rigid.model.joints[properties.jointIndex].childBody
                        : MR_INVALID_INDEX;
                    const bool hasPosition =
                        properties.qIndex != MR_INVALID_INDEX &&
                        properties.qIndex >= articulation.qOffset &&
                        properties.qIndex <
                            articulation.qOffset + articulation.nq;
                    const std::size_t localQ = hasPosition
                        ? properties.qIndex - articulation.qOffset : 0u;
                    std::cout
                        << " residual_rank_" << rank << "_dof=" << dof
                        << " residual_rank_" << rank << "_joint="
                        << properties.jointIndex
                        << " residual_rank_" << rank << "_child_body="
                        << childBody
                        << " residual_rank_" << rank << "_name=\"" << name
                        << "\" residual_rank_" << rank << "_force="
                        << support.generalizedForceResidual[dof]
                        << " residual_rank_" << rank << "_acceleration="
                        << support.generalizedAccelerationResidual[dof]
                        << " residual_rank_" << rank << "_muscle_force="
                        << support.generalizedMuscleForce[dof]
                        << " residual_rank_" << rank << "_support_force="
                        << support.generalizedSupportForce[dof]
                        << " residual_rank_" << rank << "_passive_force="
                        << support.generalizedPassiveCoordinateForce[dof]
                        << " residual_rank_" << rank << "_gravity_target="
                        << support.gravityTarget[dof]
                        << " residual_rank_" << rank << "_position="
                        << (hasPosition ? support.q[localQ] : 0.0)
                        << " residual_rank_" << rank << "_lower_limit="
                        << properties.limits.x
                        << " residual_rank_" << rank << "_upper_limit="
                        << properties.limits.y
                        << " residual_rank_" << rank << "_limit_force="
                        << support.generalizedPositionLimitForce[dof];
                    std::vector<std::uint32_t> muscleOrder(
                        musclePaths.size());
                    std::iota(muscleOrder.begin(), muscleOrder.end(), 0u);
                    std::stable_sort(
                        muscleOrder.begin(), muscleOrder.end(),
                        [&support, &musclePaths, dof](
                            const std::uint32_t first,
                            const std::uint32_t second) {
                            return std::abs(
                                support.muscleTendonForce[first] *
                                musclePaths[first].path.lengthJacobian[dof]) >
                                std::abs(
                                    support.muscleTendonForce[second] *
                                    musclePaths[second].path.lengthJacobian[dof]);
                        });
                    // Eight covers every source muscle spanning the most
                    // densely actuated hand coordinate. Keeping the complete
                    // local antagonist set visible prevents a saturated top
                    // three from hiding a missing or unusable route.
                    constexpr std::size_t kContributorCount = 8u;
                    for (std::size_t contributor = 0u;
                         contributor < std::min(
                             kContributorCount, muscleOrder.size());
                         ++contributor) {
                        const std::uint32_t muscle =
                            muscleOrder[contributor];
                        const double momentArm =
                            musclePaths[muscle].path.lengthJacobian[dof];
                        std::cout
                            << " residual_rank_" << rank << "_muscle_"
                            << contributor << "_index=" << muscle
                            << " residual_rank_" << rank << "_muscle_"
                            << contributor << "_activation="
                            << support.activation[muscle]
                            << " residual_rank_" << rank << "_muscle_"
                            << contributor << "_force_n="
                            << support.muscleTendonForce[muscle]
                            << " residual_rank_" << rank << "_muscle_"
                            << contributor << "_passive_force_n="
                            << support.passiveMuscleTendonForce[muscle]
                            << " residual_rank_" << rank << "_muscle_"
                            << contributor << "_moment_arm_m=" << momentArm
                            << " residual_rank_" << rank << "_muscle_"
                            << contributor << "_generalized_force="
                            << support.muscleTendonForce[muscle] * momentArm;
                    }
                }
                std::cout
                    << " boundary=static_unilateral_floating_base_wrench_only_not_internal_muscle_balance_dynamic_contact_or_sustained_standing\n";
                return 0;
            }
            std::optional<MuscleDrivenVisualState> muscleDrivenState;
            std::vector<float> overriddenPoseQ;
            std::span<const float> poseQ = rigid.model.defaultQ;
            if (!requestedPoseCoordinates.empty()) {
                std::sort(requestedPoseCoordinates.begin(), requestedPoseCoordinates.end());
                require(std::adjacent_find(
                            requestedPoseCoordinates.begin(), requestedPoseCoordinates.end(),
                            [](const auto& first, const auto& second) {
                                return first.first == second.first;
                            }
                        ) == requestedPoseCoordinates.end(),
                        "--pose-q may assign each q index only once");
                std::vector<double> projected(
                    rigid.model.defaultQ.begin(), rigid.model.defaultQ.end()
                );
                for (const auto& [index, value] : requestedPoseCoordinates) {
                    require(index < projected.size(), "--pose-q index exceeds the source nq");
                    const auto dof = std::find_if(
                        rigid.model.dofs.begin(), rigid.model.dofs.end(),
                        [index](const MRDofPropertiesGPU& candidate) {
                            return candidate.qIndex == index;
                        }
                    );
                    require(dof != rigid.model.dofs.end(),
                            "--pose-q may override only a scalar source DoF coordinate");
                    require((dof->flags & MR_DOF_FLAG_POSITION_LIMIT) == 0u ||
                                (value >= static_cast<double>(dof->limits.x) - 1.0e-9 &&
                                 value <= static_cast<double>(dof->limits.y) + 1.0e-9),
                            "--pose-q coordinate value exceeds its source position range");
                    projected[index] = value;
                }
                double maximumProjection = 0.0;
                const auto projectionDiagnostics =
                    metalrobo::projectNumiHumanJointEqualities(
                        jointEqualityPayload->payload.records, projected,
                        &maximumProjection
                    );
                require(projectionDiagnostics.succeeded(),
                        std::string("--pose-q equality projection failed: ") +
                            metalrobo::numiHumanJointEqualityStatusName(
                                projectionDiagnostics.status
                            ));
                overriddenPoseQ.reserve(projected.size());
                std::transform(
                    projected.begin(), projected.end(),
                    std::back_inserter(overriddenPoseQ),
                    [](const double value) { return static_cast<float>(value); }
                );
                poseQ = overriddenPoseQ;
                std::cout << "pose_q_override_count=" << requestedPoseCoordinates.size()
                          << " pose_q_equality_maximum_correction=" << maximumProjection
                          << "\n";
            }
            std::optional<PectoralisFasciaVisual> pectoralisFascia;
            std::vector<MRBodyStateGPU> precomputedRestBodies;
            if (pectoralisFasciaPayload.has_value()) {
                const metalrobo::MetalArticulatedOperatorInput restInput{
                    .articulationIndex = 0u,
                    .environmentCount = 1u,
                    .pointCount = 0u,
                    .q = rigid.model.defaultQ,
                    .points = {},
                };
                metalrobo::MetalArticulatedOperatorConfig restConfig;
                restConfig.pointJacobiansOnly = true;
                metalrobo::MetalArticulatedOperatorResult restResult;
                const auto restDiagnostics =
                    metalrobo::runMetalArticulatedOperator(
                        rigid.model, restInput, restResult, restConfig
                    );
                require(restDiagnostics.succeeded() &&
                            restDiagnostics.dispatched &&
                            restDiagnostics.published &&
                            restDiagnostics.successfulEnvironmentCount == 1u,
                        "native Human Metal rest pose for coupled fascia failed: " +
                            restDiagnostics.message);
                precomputedRestBodies = visualBodyStates(
                    rigid.model, restResult.bodyPoses
                );
            }
            if (muscleStepSeconds.has_value()) {
                if (bilateralPlantarFasciaCertificate) {
                    std::array<PlantarFasciaSideAudit, 2u> audits;
                    for (std::size_t side = 0u; side < audits.size(); ++side) {
                        audits[side] = runPlantarFasciaTensileSide(
                            *bonePayload, rigid.model,
                            *supportContactPayload, *jointEqualityPayload,
                            *muscleStepSeconds, side);
                        require(audits[side].available,
                                "bilateral plantar fascia tensile proof is incomplete");
                    }
                    std::cout << std::setprecision(12)
                              << "numi_human_bilateral_plantar_fascia=ok"
                              << " geometry=BodyParts3D_4_0_exact_named_surface_patches"
                              << " model=five_ray_metatarsal_head_pulley_reduced_tensile_law"
                              << " execution=Metal_kinematics_plus_FP64_exact_point_JT_and_articulated_response"
                              << " aggregate_target_stiffness_n_per_mm=200"
                              << " published_ray_stiffness_n_per_mm=60,50,50,20,20"
                              << " published_ray_rest_lengths_m=0.151,0.149,0.148,0.140,0.131"
                              << " force_authority=single_passive_three_point_force_system_mapped_once_by_exact_point_JT"
                              << " plantar_actuation=none"
                              << " toe_articulation=one_source_MTP_coordinate_per_foot_shared_by_all_five_rays";
                    for (const PlantarFasciaSideAudit& audit : audits) {
                        const std::string prefix = " " + audit.side + "_";
                        std::cout
                            << prefix << "device=\"" << audit.deviceName << "\""
                            << prefix << "calcaneus_body=" << audit.calcaneusBodyIndex
                            << prefix << "toes_body=" << audit.toesBodyIndex
                            << prefix << "mtp_q_index=" << audit.mtpQIndex
                            << prefix << "qualification_mtp_rad="
                            << audit.qualificationMTPRadians
                            << prefix << "response_steps=" << audit.completedSteps
                            << prefix << "source_length_scale=" << audit.sourceLengthScale
                            << prefix << "max_rest_pattern_relative_residual="
                            << audit.maximumRestLengthPatternRelativeResidual
                            << prefix << "total_tension_n="
                            << audit.totalTensionNewtons
                            << prefix << "max_force_closure_residual_n="
                            << audit.maximumForceClosureResidualNewtons
                            << prefix << "max_moment_closure_residual_nm="
                            << audit.maximumMomentClosureResidualNewtonMeters
                            << prefix << "max_q_delta_from_no_plantar_force="
                            << audit.maximumConfigurationDeltaFromSourceJT
                            << prefix << "max_v_delta_from_no_plantar_force="
                            << audit.maximumVelocityDeltaFromSourceJT
                            << prefix << "replay=bitwise";
                        for (const PlantarFasciaBandAudit& band : audit.bands) {
                            const std::string bandPrefix = prefix + "ray" +
                                std::to_string(band.ray) + "_";
                            std::cout
                                << bandPrefix << "calcaneus_stable_id="
                                << band.calcaneusStableId
                                << bandPrefix << "metatarsal_stable_id="
                                << band.metatarsalStableId
                                << bandPrefix << "phalanx_stable_id="
                                << band.proximalPhalanxStableId
                                << bandPrefix << "bodyparts_rest_length_m="
                                << band.bodypartsRestLengthMeters
                                << bandPrefix << "cross_section_mm2="
                                << band.calibratedCrossSectionSquareMillimeters
                                << bandPrefix << "extension_m="
                                << band.acceptedExtensionMeters
                                << bandPrefix << "tension_n="
                                << band.tensionNewtons
                                << bandPrefix << "force_closure_residual_n="
                                << band.forceClosureResidualNewtons
                                << bandPrefix << "moment_closure_residual_nm="
                                << band.momentClosureResidualNewtonMeters
                                << bandPrefix << "calcaneal_reaction_n="
                                << band.calcanealReactionResultantNewtons
                                << bandPrefix << "metatarsal_reaction_n="
                                << band.metatarsalReactionResultantNewtons
                                << bandPrefix << "proximal_foot_body_reaction_n="
                                << band.proximalFootBodyReactionResultantNewtons
                                << bandPrefix << "phalanx_reaction_n="
                                << band.phalanxReactionResultantNewtons
                                << bandPrefix << "calcaneal_patch_radius_m="
                                << band.calcanealPatchRadiusMeters
                                << bandPrefix << "metatarsal_patch_radius_m="
                                << band.metatarsalPatchRadiusMeters
                                << bandPrefix << "metatarsal_pulley_radius_m="
                                << band.metatarsalPulleyRadiusMeters
                                << bandPrefix << "phalanx_patch_radius_m="
                                << band.phalanxPatchRadiusMeters;
                        }
                    }
                    std::cout
                        << " boundary=bounded_bilateral_windlass_force_transfer_law_not_deformable_FEM_subject_specific_nonlinear_toe_region_rate_failure_loaded_gait_or_clinical_validation\n";
                    return 0;
                } else if (openKneeLiveTissueFEM) {
                    MuscleDrivenVisualState coupledDriven;
                    openKneeLigamentFEM.emplace(runLiveOpenKneeTissueFEM(
                        *openKneePayload, musclePayload, coupledDriven,
                        rigid.model, *supportContactPayload,
                        *jointEqualityPayload, *muscleStepSeconds,
                        muscleStepCount.value_or(1u),
                        muscleActivation.value_or(0.5),
                        selectedSourceMuscleActivations,
                        selectedTendonControl, standRootAssistance,
                        standRemoveAssistance,
                        passiveFEMMetallibPath.value_or(NUMI_MATTER_METALLIB)
                    ));
                    muscleDrivenState.emplace(std::move(coupledDriven));
                    std::cout
                        << "open_knee_live_tissue_fem=accepted"
                        << " side="
                        << (openKneePayload->side ==
                                metalrobo::NumiHumanKneeSide::left
                                ? "left" : "right_mirrored")
                        << " device=\"" << openKneeLigamentFEM->deviceName
                        << "\" regions="
                        << openKneeLigamentFEM->header.regionCount
                        << " nodes=" << openKneeLigamentFEM->header.nodeCount
                        << " tetrahedra="
                        << openKneeLigamentFEM->header.tetrahedronCount
                        << " maximum_displacement_m="
                        << openKneeLigamentFEM->maximumDisplacementMeters
                        << " qualification_flexion_rad="
                        << openKneeLigamentFEM->qualificationFlexionRadians
                        << " projected_rest_visual_correction_max_m="
                        << openKneeLigamentFEM->maximumProjectedRestVisualCorrectionMeters
                        << " projected_rest_reconstruction_max_residual_m="
                        << openKneeLigamentFEM->maximumProjectedRestReconstructionResidualMeters
                        << " min_J="
                        << openKneeLigamentFEM->minimumDeterminant
                        << " max_J="
                        << openKneeLigamentFEM->maximumDeterminant
                        << " femur_reaction_l1_n="
                        << openKneeLigamentFEM->bodyReactionL1Newtons[0u]
                        << " tibia_reaction_l1_n="
                        << openKneeLigamentFEM->bodyReactionL1Newtons[1u]
                        << " patella_reaction_l1_n="
                        << openKneeLigamentFEM->bodyReactionL1Newtons[2u]
                        << " femur_anchor_target_max_residual_m="
                        << openKneeLigamentFEM->maximumAnchorTargetResidualMeters[0u]
                        << " tibia_anchor_target_max_residual_m="
                        << openKneeLigamentFEM->maximumAnchorTargetResidualMeters[1u]
                        << " patella_anchor_target_max_residual_m="
                        << openKneeLigamentFEM->maximumAnchorTargetResidualMeters[2u]
                        << " replay=bitwise rollback=verified"
                        << " force_owner=full_source_row_replaced_by_QAT_enthesis_and_bipolar_PTL_enthesis_transfer_plus_passive_bone_reactions"
                        << " quadriceps_endpoints="
                        << openKneeLigamentFEM->quadricepsEndpointCount
                        << " quadriceps_owner_fraction="
                        << openKneeLigamentFEM->quadricepsForceOwnerFraction
                        << " quadriceps_force_l1_n="
                        << openKneeLigamentFEM->quadricepsAppliedForceL1Newtons
                        << " quadriceps_force_resultant_n="
                        << openKneeLigamentFEM->quadricepsAppliedForceResultantNewtons
                        << " quadriceps_enthesis_reaction_resultant_n="
                        << openKneeLigamentFEM->quadricepsEnthesisReactionResultantNewtons
                        << " ptl_force_l1_n="
                        << openKneeLigamentFEM->patellarTendonForceL1Newtons
                        << " ptl_force_resultant_n="
                        << openKneeLigamentFEM->patellarTendonForceResultantNewtons
                        << " ptl_patella_reaction_resultant_n="
                        << openKneeLigamentFEM->patellarTendonPatellaReactionResultantNewtons
                        << " ptl_tibia_reaction_resultant_n="
                        << openKneeLigamentFEM->patellarTendonTibiaReactionResultantNewtons
                        << " assembled_external_force_l1_n="
                        << openKneeLigamentFEM->assembledExternalForceL1Newtons
                        << " assembled_external_force_resultant_n="
                        << openKneeLigamentFEM->assembledExternalForceResultantNewtons
                        << " articular_pairs="
                        << openKneeLigamentFEM->articularPairCount
                        << " articular_samples="
                        << openKneeLigamentFEM->articularContactSampleCount
                        << " articular_mechanical_samples="
                        << openKneeLigamentFEM->articularMechanicalSampleCount
                        << " articular_internal_same_body_samples="
                        << openKneeLigamentFEM->articularInternalSameBodySampleCount
                        << " articular_closed_samples="
                        << openKneeLigamentFEM->articularClosedSampleCount
                        << " articular_contact_area_m2="
                        << openKneeLigamentFEM->articularContactAreaSquareMeters
                        << " articular_normal_force_n="
                        << openKneeLigamentFEM->articularNormalForceNewtons
                        << " articular_max_pressure_pa="
                        << openKneeLigamentFEM->articularMaximumPressurePascals
                        << " articular_body_force_l1_n="
                        << openKneeLigamentFEM->articularBodyForceL1Newtons
                        << " articular_force_residual_n="
                        << openKneeLigamentFEM->articularForceResidualNewtons
                        << " articular_moment_residual_nm="
                        << openKneeLigamentFEM->articularMomentResidualNewtonMeters
                        << " articular_stored_energy_j="
                        << openKneeLigamentFEM->articularStoredEnergyJoules
                        << " articular_max_normal_strain="
                        << openKneeLigamentFEM->articularMaximumNormalStrain
                        << " articular_max_closure_m="
                        << openKneeLigamentFEM->articularMaximumClosureMeters
                        << " articular_audited_steps="
                        << openKneeLigamentFEM->articularAuditedStepCount
                        << " articular_trajectory_min_closed_samples="
                        << openKneeLigamentFEM->articularTrajectoryMinimumClosedSampleCount
                        << " articular_trajectory_max_closed_samples="
                        << openKneeLigamentFEM->articularTrajectoryMaximumClosedSampleCount
                        << " articular_trajectory_min_normal_force_n="
                        << openKneeLigamentFEM->articularTrajectoryMinimumNormalForceNewtons
                        << " articular_trajectory_max_normal_force_n="
                        << openKneeLigamentFEM->articularTrajectoryMaximumNormalForceNewtons
                        << " articular_trajectory_max_pressure_pa="
                        << openKneeLigamentFEM->articularTrajectoryMaximumPressurePascals
                        << " articular_trajectory_max_stored_energy_j="
                        << openKneeLigamentFEM->articularTrajectoryMaximumStoredEnergyJoules
                        << " articular_trajectory_max_normal_strain="
                        << openKneeLigamentFEM->articularTrajectoryMaximumNormalStrain
                        << " articular_trajectory_max_closure_m="
                        << openKneeLigamentFEM->articularTrajectoryMaximumClosureMeters
                        << " articular_trajectory_max_force_residual_n="
                        << openKneeLigamentFEM->articularTrajectoryMaximumForceResidualNewtons
                        << " articular_trajectory_max_moment_residual_nm="
                        << openKneeLigamentFEM->articularTrajectoryMaximumMomentResidualNewtonMeters
                        << " qat_load_nodes="
                        << openKneeLigamentFEM->quadricepsLoadNodeCount
                        << " qat_load_patch_area_m2="
                        << openKneeLigamentFEM->quadricepsLoadPatchAreaSquareMeters
                        << " ptl_patella_load_nodes="
                        << openKneeLigamentFEM->patellarTendonPatellaLoadNodeCount
                        << " ptl_tibia_load_nodes="
                        << openKneeLigamentFEM->patellarTendonTibiaLoadNodeCount
                        << " ptl_patella_patch_area_m2="
                        << openKneeLigamentFEM->patellarTendonPatellaPatchAreaSquareMeters
                        << " ptl_tibia_patch_area_m2="
                        << openKneeLigamentFEM->patellarTendonTibiaPatchAreaSquareMeters
                        << " extensor_chain=nonlinear_source_tendon_force_QAT_to_patella_to_PTL_to_tibia_exact_entheses\n";
                } else if (pectoralisFasciaPayload.has_value()) {
                    MuscleDrivenVisualState coupledDriven;
                    pectoralisFascia.emplace(runPectoralisFascia(
                        *pectoralisFasciaPayload, *softTissuePayload,
                        musclePayload, coupledDriven, rigid.model,
                        *supportContactPayload, *jointEqualityPayload,
                        precomputedRestBodies, *muscleStepSeconds,
                        muscleStepCount.value_or(1u),
                        muscleActivation.value_or(0.5),
                        selectedSourceMuscleActivations,
                        selectedTendonControl, standRootAssistance,
                        standRemoveAssistance,
                        passiveFEMMetallibPath.value_or(NUMI_MATTER_METALLIB)
                    ));
                    muscleDrivenState.emplace(std::move(coupledDriven));
                } else if (persistentMetalStand || selectedTendonControl) {
                    muscleDrivenState.emplace(
                        integratePersistentMetalHumanState(
                            rigid.model,
                            musclePayload,
                            *supportContactPayload,
                            *jointEqualityPayload,
                            *muscleStepSeconds,
                            muscleStepCount.value_or(1u),
                            muscleActivation.value_or(0.5),
                            selectedSourceMuscleActivations,
                            selectedTendonControl,
                            standRootAssistance,
                            standRemoveAssistance,
                            standDeterministicReplay || selectedTendonControl
                        )
                    );
                } else {
                    muscleDrivenState.emplace(integrateMuscleDrivenVisualState(
                        rigid.model, musclePayload, *muscleStepSeconds,
                        muscleStepCount.value_or(1u),
                        muscleActivation.value_or(0.5),
                        selectedSourceMuscleActivations,
                        supportContactPayload.has_value() ? &*supportContactPayload : nullptr
                    ));
                }
                poseQ = muscleDrivenState->q;
            }
            if (bilateralAchillesCertificate) {
                require(
                    muscleDrivenState.has_value() &&
                        muscleDrivenState->achilles[0u].available &&
                        muscleDrivenState->achilles[1u].available &&
                        muscleDrivenState->tendonBorrowedConsumerVerified &&
                        muscleDrivenState->tendonRollbackVerified &&
                        muscleDrivenState->tendonRigidStateIdentityVerified &&
                        muscleDrivenState->deterministicReplayVerified,
                    "bilateral Achilles qualification lacks transactional proof"
                );
                std::cout << std::setprecision(12)
                          << "numi_human_bilateral_achilles_force_transfer=ok"
                          << " device=\""
                          << muscleDrivenState->muscleMetalDeviceName << "\""
                          << " source_model=pinned_MyoSim_full_body"
                          << " geometry=BodyParts3D_4_0_named_calcaneus_envelopes"
                          << " tendon_law=NHMYO2_nonlinear_compliant_fiber_tendon_equilibrium"
                          << " transfer=NHTENDON3_four_node_wrench_equivalent_enthesis"
                          << " selected_muscles=348,349,369,388,389,409"
                          << " selected_muscle_names=gaslat_r,gasmed_r,soleus_r,gaslat_l,gasmed_l,soleus_l"
                          << " accepted_steps="
                          << muscleDrivenState->persistentCompletedSteps
                          << " replay=bitwise"
                          << " rollback=consumer_rejection_preserved_result"
                          << " borrowed_consumer=same_command_buffer_exact_snapshot"
                          << " force_authority=single_source_route_JT_with_distributed_enthesis_correction"
                          << " compiled_support_contacts="
                          << muscleDrivenState->compiledSupportContactCount
                          << " compiled_active_support_contacts="
                          << muscleDrivenState->compiledActiveSupportContactCount
                          << " compiled_total_support_force_n="
                          << muscleDrivenState->compiledTotalSupportForceNewtons
                          << " compiled_max_root_force_residual="
                          << muscleDrivenState->compiledMaximumRootForceResidual
                          << " compiled_max_root_acceleration_residual="
                          << muscleDrivenState->compiledMaximumRootAccelerationResidual
                          << " compiled_normalized_residual_rms="
                          << muscleDrivenState->compiledActivationResidualRms
                          << " compiled_initial_normalized_residual_rms="
                          << muscleDrivenState->compiledInitialActivationResidualRms
                          << " compiled_balanced="
                          << (muscleDrivenState->compiledBalanced ? "true" : "false");
                constexpr std::array<const char*, 2u> kSideNames{
                    "right", "left",
                };
                for (std::size_t side = 0u;
                     side < muscleDrivenState->achilles.size(); ++side) {
                    const auto& audit = muscleDrivenState->achilles[side];
                    const std::string prefix = std::string(" ") +
                        kSideNames[side] + "_";
                    std::cout
                        << prefix << "calcaneus_body="
                        << audit.calcaneusBodyIndex
                        << prefix << "calcaneus_bone_stable_id="
                        << audit.calcaneusBoneStableId
                        << prefix << "ankle_q_index=" << audit.ankleQIndex
                        << prefix << "ankle_dof_index=" << audit.ankleDofIndex
                        << prefix << "muscles=" << audit.muscleCount
                        << prefix << "distributed_endpoints="
                        << audit.distributedEndpointCount
                        << prefix << "represented_force_l1_n="
                        << audit.representedForceL1Newtons
                        << prefix << "represented_force_increment_l1_n="
                        << audit.representedForceIncrementL1Newtons
                        << prefix << "terminal_force_resultant_n="
                        << audit.terminalForceResultantNewtons
                        << prefix << "terminal_force_increment_resultant_n="
                        << audit.terminalForceIncrementResultantNewtons
                        << prefix << "nodal_force_resultant_n="
                        << audit.nodalForceResultantNewtons
                        << prefix << "aggregate_force_residual_n="
                        << audit.aggregateForceResidualNewtons
                        << prefix << "max_endpoint_force_residual_n="
                        << audit.maximumEndpointForceResidualNewtons
                        << prefix << "max_endpoint_moment_residual_nm="
                        << audit.maximumEndpointMomentResidualNewtonMeters
                        << prefix << "source_ankle_torque_nm="
                        << audit.sourceAnkleTorqueNewtonMeters
                        << prefix << "source_ankle_torque_increment_nm="
                        << audit.sourceAnkleTorqueIncrementNewtonMeters
                        << prefix << "distributed_ankle_torque_correction_nm="
                        << audit.distributedAnkleTorqueCorrectionNewtonMeters
                        << prefix << "min_normalized_tendon_tension="
                        << audit.minimumNormalizedTendonTension
                        << prefix << "max_normalized_tendon_tension="
                        << audit.maximumNormalizedTendonTension
                        << prefix << "max_equilibrium_residual="
                        << audit.maximumDampedEquilibriumResidual
                        << prefix << "min_patch_radius_m="
                        << audit.minimumPatchRadiusMeters
                        << prefix << "max_patch_radius_m="
                        << audit.maximumPatchRadiusMeters
                        << prefix << "ankle_q_increment_rad="
                        << audit.configurationIncrementRadians
                        << prefix << "ankle_v_increment_rad_s="
                        << audit.velocityIncrementRadiansPerSecond;
                }
                std::cout
                    << " boundary=bounded_bilateral_Achilles_active_force_transfer_certificate_not_deformable_volumetric_tendon_contact_sustained_gait_or_clinical_validation\n";
                return 0;
            }
            const metalrobo::MetalArticulatedOperatorInput input{
                .articulationIndex = 0u,
                .environmentCount = 1u,
                .pointCount = 0u,
                .q = poseQ,
                .points = {},
            };
            metalrobo::MetalArticulatedOperatorConfig operatorConfig;
            operatorConfig.pointJacobiansOnly = true;
            metalrobo::MetalArticulatedOperatorResult poseResult;
            const auto poseDiagnostics = metalrobo::runMetalArticulatedOperator(
                rigid.model, input, poseResult, operatorConfig
            );
            require(poseDiagnostics.succeeded() && poseDiagnostics.dispatched &&
                        poseDiagnostics.published && poseDiagnostics.successfulEnvironmentCount == 1u,
                    "native Human Metal pose pass failed: " + poseDiagnostics.message);
            const std::vector<MRBodyStateGPU> bodies = visualBodyStates(
                rigid.model, poseResult.bodyPoses
            );
            std::vector<MRBodyStateGPU> restBodies =
                precomputedRestBodies.empty() ? bodies : precomputedRestBodies;
            if (precomputedRestBodies.empty() &&
                (passiveFEMTissueStableId.has_value() ||
                 (skinPayload.has_value() && skinPayload->usesWorldRestNormals))) {
                const metalrobo::MetalArticulatedOperatorInput restInput{
                    .articulationIndex = 0u,
                    .environmentCount = 1u,
                    .pointCount = 0u,
                    .q = rigid.model.defaultQ,
                    .points = {},
                };
                metalrobo::MetalArticulatedOperatorResult restPoseResult;
                const auto restPoseDiagnostics = metalrobo::runMetalArticulatedOperator(
                    rigid.model, restInput, restPoseResult, operatorConfig
                );
                require(restPoseDiagnostics.succeeded() && restPoseDiagnostics.dispatched &&
                            restPoseDiagnostics.published &&
                            restPoseDiagnostics.successfulEnvironmentCount == 1u,
                        "native Human Metal rest pose for passive FEM failed: " +
                            restPoseDiagnostics.message);
                restBodies = visualBodyStates(rigid.model, restPoseResult.bodyPoses);
            }
            std::optional<SourceRouteCentrelines> resolvedRouteCentrelines;
            if (sourceRouteCentrelines) {
                resolvedRouteCentrelines.emplace(resolveSourceRouteCentrelines(
                    rigid.model, musclePayload, poseQ, requestedSourceRouteMuscles
                ));
                if (surfaceProjectSourceSites) {
                    projectSourceSiteEndpointsToBoneSurfaces(
                        *resolvedRouteCentrelines, *bonePayload, bodies
                    );
                }
            }
            std::optional<PassiveFEMTissueVisual> passiveFEMTissue;
            if (passiveFEMTissueStableId.has_value()) {
                passiveFEMTissue.emplace(runPassiveFEMTissue(
                    *softTissuePayload, restBodies, bodies, *passiveFEMTissueStableId,
                    *muscleStepSeconds, passiveFEMStepCount.value_or(8u),
                    passiveFEMMetallibPath.value_or(NUMI_MATTER_METALLIB)
                ));
            }
            std::uint32_t renderedBodies = 0u;
            std::uint32_t renderedSoftTissues = 0u;
            std::uint32_t renderedSkinShells = 0u;
            std::uint32_t renderedTorsoAnatomySurfaces = 0u;
            std::uint32_t renderedTendonAttachmentCollars = 0u;
            std::uint32_t renderedTendonAttachmentEnvelopes = 0u;
            std::uint32_t renderedRouteSegments = 0u;
            std::uint32_t renderedPassiveFEMTissues = 0u;
            std::uint32_t renderedPectoralisFascia = 0u;
            std::uint32_t renderedOpenKneeRegions = 0u;
            bool anyRequestedRouteVisible = false;
            const metalrobo::VisualAssetPackV2 pack = makeMarkerPack(
                rigid.model, musclePayload,
                bonePayload.has_value() ? &*bonePayload : nullptr,
                openKneePayload.has_value() ? &*openKneePayload : nullptr,
                openKneeLigamentFEM.has_value() ? &*openKneeLigamentFEM : nullptr,
                softTissuePayload.has_value() ? &*softTissuePayload : nullptr,
                skinPayload.has_value() ? &*skinPayload : nullptr,
                torsoAnatomyPayload.has_value() ? &*torsoAnatomyPayload : nullptr,
                bodies, restBodies,
                passiveFEMTissue.has_value() ? &*passiveFEMTissue : nullptr,
                pectoralisFascia.has_value() ? &*pectoralisFascia : nullptr,
                muscleDrivenState.has_value(),
                requestedBoneBodyIndices,
                requestedBoneStableIds,
                requestedSoftTissueStableIds,
                zAnatomyCalfVisualSupplement,
                tendonAttachmentCollarDiagnostic,
                resolvedRouteCentrelines.has_value() ? &*resolvedRouteCentrelines : nullptr,
                renderedBodies, renderedSoftTissues, renderedSkinShells, renderedTorsoAnatomySurfaces,
                renderedTendonAttachmentCollars, renderedTendonAttachmentEnvelopes,
                renderedRouteSegments,
                renderedPassiveFEMTissues,
                renderedPectoralisFascia,
                renderedOpenKneeRegions
            );
            require(requestedSoftTissueStableIds.empty() ||
                        renderedSoftTissues + (pectoralisFascia.has_value()
                            ? pectoralisFascia->sourceStableIds.size() : 0u) ==
                            requestedSoftTissueStableIds.size(),
                    "native Human visual soft-tissue selection did not render every requested source surface");
            require(
                (requestedBoneBodyIndices.empty() && requestedBoneStableIds.empty()) ||
                    renderedBodies > 0u,
                    "native Human visual bone selection rendered no source mesh");
            require(!passiveFEMTissue.has_value() || renderedPassiveFEMTissues == 1u,
                    "native Human visual passive FEM tissue selection did not render its source surface");
            require(!pectoralisFascia.has_value() || renderedPectoralisFascia == 1u,
                    "native Human visual pectoralis fascia did not render");
            const std::uint32_t expectedRenderedOpenKneeRegions =
                !openKneePayload.has_value() ? 0u :
                    static_cast<std::uint32_t>(openKneePayload->regions.size());
            require(!openKneePayload.has_value() ||
                        renderedOpenKneeRegions == expectedRenderedOpenKneeRegions,
                    "native Human visual Open Knee(s) presentation region count drifted");
            require(!torsoAnatomyPayload.has_value() ||
                        renderedTorsoAnatomySurfaces == torsoAnatomyPayload->records.size(),
                    "native Human visual torso anatomy did not render every source surface");
            const CameraFraming cameraFraming = makeCameraFraming(
                pack, bodies, rigid.model.joints, focusBodyIndex,
                focusJointChildBodyIndex, focusDistanceMeters,
                renderedTendonAttachmentEnvelopes > 0u,
                openKneePayload.has_value() ? &*openKneePayload : nullptr
            );
            const std::array<mr_float4, 4u> positions = cameraPositions(cameraFraming);
            std::array<std::string, 4u> cameraNames;
            const metalrobo::WorldTemplate world = makeWorld(
                rigid.model, cameraFraming, frameDimension, cameraNames
            );
            metalrobo::WorldProgram program;
            program.id = bodypartsBoneVisual
                ? "myosim_fullbody_articulated_bodyparts_bone_visual_program"
                : "myosim_fullbody_articulated_marker_visual_program";
            metalrobo::WorldFamily family;
            const auto familyCompile = metalrobo::compileWorldFamily(world, program, family);
            require(familyCompile.succeeded(), "native Human visual family compile failed: " + familyCompile.message);
            const std::filesystem::path outputDirectory{positional.back()};
            std::filesystem::create_directories(outputDirectory);
            const std::string stem = std::string(bodypartsBoneVisual
                ? "myosim-fullbody-articulated-bodyparts-bones"
                : "myosim-fullbody-articulated-markers") +
                (softTissuePayload.has_value() ? "-source-soft-tissues" : "") +
                (zAnatomyCalfVisualSupplement ? "-zanatomy-calf-supplement" : "") +
                (skinPayload.has_value() ? "-source-skinned-shell" : "") +
                (torsoAnatomyPayload.has_value() ? "-source-torso-anatomy" : "") +
                (openKneePayload.has_value()
                    ? (openKneePayload->side == metalrobo::NumiHumanKneeSide::left
                        ? "-open-knee-oks003-left"
                        : "-open-knee-oks003-right-mirrored")
                    : "") +
                (openKneeLigamentFEM.has_value()
                    ? "-accepted-tissue-fem"
                    : "") +
                (passiveFEMTissue.has_value() ? "-passive-fem-tissue" : "") +
                (muscleDrivenState.has_value() ? "-muscle-driven" : "") +
                (!selectedSourceMuscleActivations.empty() ? "-selected-actuators" : "") +
                (supportContactPayload.has_value() ? "-source-support-contact" : "") +
                (sourceRouteCentrelines ? "-source-route-centrelines" : "") +
                (renderedTendonAttachmentEnvelopes > 0u ? "-tendon-attachment-envelopes" : "") +
                (!requestedPoseCoordinates.empty() ? "-posed" : "") +
                (surfaceProjectSourceSites ? "-surface-projected-sites" : "") +
                (focusBodyIndex.has_value()
                    ? "-focus-body-" + std::to_string(*focusBodyIndex) : "") +
                (focusJointChildBodyIndex.has_value()
                    ? "-focus-joint-child-body-" +
                        std::to_string(*focusJointChildBodyIndex) : "");
            const std::filesystem::path packPath = outputDirectory / (stem + ".mrvpack");
            std::string reason;
            require(metalrobo::writeVisualAssetPack(pack, packPath, &reason),
                    "could not write native Human visual pack: " + reason);
            const std::array references{
                metalrobo::VisualAssetReferenceV3{
                    packPath, pack.contentHash, 0u,
                    skinPayload.has_value()
                        ? kSkinShellSemantic
                        : (bodypartsBoneVisual ? kBoneSemantic : kBodySemantic),
                    1u,
                },
            };
            metalrobo::VisualSceneManifestV3 manifest;
            require(metalrobo::compileVisualSceneManifestV3(
                        world, references, metalrobo::makeNeutralStudioEnvironmentV2(),
                        makeHumanAnatomyLightRig(
                            cameraFraming.center, positions.front(), cameraFraming.distance
                        ), manifest, &reason
                    ),
                    "native Human visual scene compile failed: " + reason);
            require(metalrobo::writeVisualSceneManifestV3(
                        manifest, outputDirectory / (stem + ".visual.v3.json"), &reason
                    ),
                    "could not write native Human visual manifest: " + reason);

            metalrobo::VisualMotionSampleBatchV1 motion = makeMotion(bodies);
            bool completeVisualCoverage = true;
            bool anyTendonAttachmentEnvelopeVisible = false;
            bool capturedRenderer = false;
            std::string rendererDeviceName;
            double rendererCompileMilliseconds = 0.0;
            // Source/anatomy captures are presentation evidence, rather than
            // training observations. The regular sensor-reference profile is
            // deliberately modest (8 temporal / 8 area-light samples) so it
            // is inexpensive for routine diagnostics, but that leaves grain
            // on the close, light-coloured calcaneus. The scoped Z-Anatomy
            // inspection is the one visual intended for human review, so
            // raise its native ray and softbox integration without changing
            // geometry, material parameters, pose, or physics state.
            metalrobo::VisualRendererProfileV1 rendererProfile =
                metalrobo::VisualRendererProfileV1::sensorReference();
            if (zAnatomyCalfVisualSupplement || openKneePayload.has_value()) {
                rendererProfile.id = openKneePayload.has_value()
                    ? "open_knee_anatomy_showcase_reference"
                    : "human_anatomy_showcase_reference";
                rendererProfile.temporalSamples = 32u;
                rendererProfile.shadowMapResolution = 1024u;
                rendererProfile.areaLightSamples = 32u;
                rendererProfile.fingerprint =
                    metalrobo::computeVisualRendererProfileFingerprint(rendererProfile);
            }
            std::string rendererProfileReason;
            require(rendererProfile.valid(&rendererProfileReason),
                    "native Human renderer profile is invalid: " + rendererProfileReason);
            for (std::size_t camera = 0u; camera < cameraNames.size(); ++camera) {
                if (requestedCameraIndex.has_value() && camera != *requestedCameraIndex) {
                    continue;
                }
                @autoreleasepool {
                // Reference ray workspaces can retain a large drawable and
                // acceleration structure.  Build one isolated renderer per
                // fixed angle so 2048 px anatomy review cannot reuse a prior
                // camera's in-flight workspace.
                metalrobo::VisualSceneManifestV3 cameraManifest;
                require(metalrobo::compileVisualSceneManifestV3(
                            world, references, metalrobo::makeNeutralStudioEnvironmentV2(),
                            makeHumanAnatomyLightRig(
                                cameraFraming.center, positions[camera], cameraFraming.distance
                            ), cameraManifest, &reason
                        ),
                        "native Human per-camera visual scene compile failed: " + reason);
                metalrobo::MetalHybridRendererConfig rendererConfig;
                rendererConfig.width = frameDimension;
                rendererConfig.height = frameDimension;
                rendererConfig.maximumReferenceFramesInFlight = 1u;
                // A neutral charcoal keeps the medical/source-validation
                // framing free of fabricated scenery while preserving enough
                // shadow separation for an exterior silhouette to read.
                rendererConfig.clearColorAndDepth = {0.012f, 0.019f, 0.030f, 1.0e30f};
                metalrobo::MetalHybridRenderer renderer(rendererConfig);
                const auto rendererCompile = renderer.compile(
                    std::move(cameraManifest.renderScene),
                    rendererProfile, 1u
                );
                require(rendererCompile.succeeded(), "native Human renderer compile failed: " + rendererCompile.message);
                if (!capturedRenderer) {
                    rendererDeviceName = rendererCompile.deviceName;
                    rendererCompileMilliseconds = rendererCompile.elapsedMilliseconds;
                    capturedRenderer = true;
                } else {
                    require(rendererCompile.deviceName == rendererDeviceName,
                            "native Human visual cameras selected different renderer devices");
                }
                motion.sensorIdentity = camera + 1u;
                motion.sensorSequence = static_cast<std::uint32_t>(camera + 1u);
                motion.frameIndex = camera + 1u;
                // MetalWorldFamilyContext owns sampled sensor-side state.  It
                // must be fresh with the matching reference renderer so no
                // camera can inherit a prior angle's drawable resources.
                metalrobo::MetalWorldFamilyContext worlds;
                const auto worldsCompile = worlds.compile(family, 1u);
                require(worldsCompile.succeeded(),
                        "native Human per-camera device world compile failed: " + worldsCompile.message);
                const auto worldsSample = worlds.sample(1u, 0x4d594f53494dull);
                require(worldsSample.succeeded(),
                        "native Human per-camera world sample failed: " + worldsSample.message);
                const auto render = renderer.renderFrame(worlds, motion, static_cast<std::uint32_t>(camera));
                require(render.succeeded(), "native Human render failed: " + render.message);
                metalrobo::HybridObservationBatch observation;
                const auto readback = renderer.readback(observation);
                require(readback.succeeded(), "native Human render readback failed: " + readback.message);
                const std::filesystem::path frame = outputDirectory /
                    (stem + "-" + cameraNames[camera] + ".png");
                require(writePng(frame, observation), "could not write native Human PNG " + frame.string());
                const std::size_t bodyPixels = coverage(observation, kBodySemantic);
                const std::size_t bonePixels = coverage(observation, kBoneSemantic);
                const std::size_t sitePixels = coverage(observation, kSiteSemantic);
                const std::size_t routePixels = coverage(observation, kRouteSemantic);
                const std::size_t muscleSurfacePixels = coverage(observation, kMuscleSurfaceSemantic);
                const std::size_t tendonSurfacePixels = coverage(observation, kTendonSurfaceSemantic);
                const std::size_t tendonAttachmentCollarPixels = coverage(
                    observation, kTendonAttachmentCollarSemantic
                );
                const std::size_t tendonAttachmentEnvelopePixels = coverage(
                    observation, kTendonAttachmentEnvelopeSemantic
                );
                const std::size_t passiveFEMTissuePixels = coverage(
                    observation, kPassiveFEMTissueSemantic
                );
                const std::size_t pectoralisFasciaPixels = coverage(
                    observation, kPectoralisFasciaSemantic
                );
                const std::size_t skinShellPixels = coverage(observation, kSkinShellSemantic);
                const std::size_t organSurfacePixels = coverage(observation, kOrganSurfaceSemantic);
                const std::size_t vesselSurfacePixels = coverage(observation, kVesselSurfaceSemantic);
                const std::size_t nerveSurfacePixels = coverage(observation, kNerveSurfaceSemantic);
                const std::size_t kneeCartilagePixels = coverage(observation, kKneeCartilageSemantic);
                const std::size_t kneeMeniscusPixels = coverage(observation, kKneeMeniscusSemantic);
                const std::size_t kneeLigamentPixels = coverage(observation, kKneeLigamentSemantic);
                const std::size_t kneeTendonPixels = coverage(observation, kKneeTendonSemantic);
                completeVisualCoverage = completeVisualCoverage &&
                    (skinPayload.has_value()
                        ? skinShellPixels > 0u
                        : (bodypartsBoneVisual ? bonePixels > 0u : bodyPixels > 0u));
                completeVisualCoverage = completeVisualCoverage &&
                    (!passiveFEMTissue.has_value() || passiveFEMTissuePixels > 0u);
                completeVisualCoverage = completeVisualCoverage &&
                    (!pectoralisFascia.has_value() || pectoralisFasciaPixels > 0u);
                completeVisualCoverage = completeVisualCoverage &&
                    (!torsoAnatomyPayload.has_value() ||
                     (organSurfacePixels > 0u && vesselSurfacePixels > 0u && nerveSurfacePixels > 0u));
                anyRequestedRouteVisible = anyRequestedRouteVisible || routePixels > 0u;
                anyTendonAttachmentEnvelopeVisible = anyTendonAttachmentEnvelopeVisible ||
                    tendonAttachmentEnvelopePixels > 0u;
                std::cout << "view=" << cameraNames[camera]
                          << " body_pixels=" << bodyPixels
                          << " bone_pixels=" << bonePixels
                          << " muscle_site_pixels=" << sitePixels
                          << " muscle_route_pixels=" << routePixels
                          << " muscle_surface_pixels=" << muscleSurfacePixels
                          << " tendon_surface_pixels=" << tendonSurfacePixels
                          << " tendon_attachment_collar_pixels=" << tendonAttachmentCollarPixels
                          << " tendon_attachment_envelope_pixels=" << tendonAttachmentEnvelopePixels
                          << " passive_fem_tissue_pixels=" << passiveFEMTissuePixels
                          << " pectoralis_fascia_pixels=" << pectoralisFasciaPixels
                          << " skin_shell_pixels=" << skinShellPixels
                          << " organ_surface_pixels=" << organSurfacePixels
                          << " vessel_surface_pixels=" << vesselSurfacePixels
                          << " nerve_surface_pixels=" << nerveSurfacePixels
                          << " knee_cartilage_pixels=" << kneeCartilagePixels
                          << " knee_meniscus_pixels=" << kneeMeniscusPixels
                          << " knee_ligament_pixels=" << kneeLigamentPixels
                          << " knee_tendon_pixels=" << kneeTendonPixels
                          << " frame=" << frame.string() << '\n';
                }
            }
            require(completeVisualCoverage,
                    "one or more native Human frames have no linked-body coverage");
            require(!sourceRouteCentrelines || anyRequestedRouteVisible,
                    "requested source route is completely occluded from all native Human cameras");
            require(renderedTendonAttachmentEnvelopes == 0u || anyTendonAttachmentEnvelopeVisible,
                    "requested NHTENDON2/3 attachment envelope is completely occluded from all cameras");
            const std::string tendonProgramName = "NHTENDON" +
                std::to_string(musclePayload.tendonPayload.payloadAbi);
            const bool sourceSupportContact = muscleDrivenState.has_value() &&
                muscleDrivenState->supportContactApplied;
            const std::string poseSource = !muscleDrivenState.has_value()
                ? (!requestedPoseCoordinates.empty()
                    ? "explicit_source_q_override_projected_through_NHEQ1_to_metal_kinematic_pose"
                    : "source_default_q_to_metal_kinematic_pose")
                : muscleDrivenState->selectedTendonControl
                    ? "persistent_metal_compiled_posture_baseline_plus_selected_source_activation_increment_all_416_mujoco_routes_" + tendonProgramName + "_transaction_gravity_joint_equalities_and_source_foot_support"
                : muscleDrivenState->persistentMetalHorizon
                    ? "persistent_metal_current_pose_all_416_mujoco_routes_activation_gravity_large_state_dynamics_and_source_foot_support"
                : sourceSupportContact
                    ? "metal_all_416_mujoco_force_projection_and_activation_state_then_cpu_fp64_free_dynamics_and_dynamic_source_foot_witness_plane_contact_then_metal_kinematic_pose"
                    : "metal_all_416_mujoco_force_projection_and_activation_state_then_cpu_fp64_free_dynamics_then_metal_kinematic_pose";
            std::string evidenceBoundary =
                muscleDrivenState.has_value() &&
                    muscleDrivenState->selectedTendonControl
                ? "bounded_selected_source_muscle_increment_over_compiled_posture_baseline_on_persistent_apple_metal_with_all_416_routes_" + tendonProgramName + "_force_transfer_joint_equalities_gravity_and_source_foot_support_not_closed_loop_control_deformable_tendon_or_clinical_validation"
                : muscleDrivenState.has_value() &&
                    muscleDrivenState->persistentMetalHorizon
                ? (compliantMusclePayload
                    ? "bounded_persistent_apple_metal_all_416_source_routes_with_inferred_positive_fiber_tendon_architecture_damped_backward_euler_equilibrium_explicit_" + tendonProgramName + "_force_transfer_large_state_mass_gravity_low_velocity_bias_and_source_foot_support_not_anatomically_calibrated_pennation_exact_jdot_rnea_joint_limit_general_collision_or_closed_loop_standing_qualification"
                    : "bounded_persistent_apple_metal_all_416_mujoco_activation_dependent_route_force_large_state_mass_gravity_low_velocity_bias_and_source_foot_support_horizon_imported_passive_bias_excluded_until_registered_equilibrium_calibration_not_exact_jdot_rnea_joint_limit_general_collision_or_closed_loop_standing_qualification")
                : !muscleDrivenState.has_value()
                ? (bodypartsBoneVisual
                    ? (softTissuePayload.has_value()
                        ? "metal_pose_snapshot_to_native_renderer_with_provisional_bodyparts_bone_registration_and_named_body_weighted_source_soft_tissue_visuals_not_collision_or_live_rollout"
                        : "metal_pose_snapshot_to_native_renderer_with_provisional_bodyparts_bone_registration_not_collision_or_live_rollout")
                    : "metal_pose_snapshot_to_native_renderer_not_bodyparts_registration_or_live_rollout")
                : sourceSupportContact
                    ? (bodypartsBoneVisual
                        ? (softTissuePayload.has_value()
                            ? "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_with_dynamic_source_foot_witness_plane_contact_and_metal_pose_with_provisional_bodyparts_bones_and_named_body_weighted_soft_tissue_visuals_metal_fullbody_contact_not_admitted_not_general_collision_stable_posture_or_live_rollout"
                            : "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_with_dynamic_source_foot_witness_plane_contact_and_metal_pose_with_provisional_bodyparts_bones_metal_fullbody_contact_not_admitted_not_general_collision_stable_posture_or_live_rollout")
                        : "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_with_dynamic_source_foot_witness_plane_contact_and_metal_pose_metal_fullbody_contact_not_admitted_not_general_collision_stable_posture_or_live_rollout")
                    : (bodypartsBoneVisual
                        ? (softTissuePayload.has_value()
                            ? "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_to_metal_pose_snapshot_with_provisional_bodyparts_bone_registration_and_named_body_weighted_soft_tissue_visuals_not_contact_or_live_rollout"
                            : "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_to_metal_pose_snapshot_with_provisional_bodyparts_bone_registration_not_contact_or_live_rollout")
                        : "bounded_multistep_all_416_mujoco_metal_force_projection_and_activation_state_then_cpu_fp64_dynamics_to_metal_pose_snapshot_not_contact_or_live_rollout");
            if (renderedTendonAttachmentCollars > 0u) {
                evidenceBoundary +=
                    "_with_source_proximity_derived_tendon_to_named_bone_visual_collars_not_a_tendon_weld_or_force_transfer";
            }
            if (renderedTendonAttachmentEnvelopes > 0u) {
                evidenceBoundary += musclePayload.tendonMigratedEnvelopeBindings > 0u
                    ? "_with_NHTENDON3_per_step_route_private_exact_registered_bone_surface_endpoints_reference_scaled_fiber_tendon_architecture_and_force_moment_conserving_terminal_load_transaction_same_command_buffer_consumer_boundary_not_clinical_validation_or_deformable_tendon_continuum"
                    : "_with_NHTENDON2_per_step_source_point_preserving_force_and_moment_conserving_terminal_load_transaction_and_same_command_buffer_consumer_boundary_not_a_clinical_enthesis_certificate_or_deformable_tendon_continuum";
            }
            if (!selectedSourceMuscleActivations.empty()) {
                evidenceBoundary +=
                    "_with_explicit_source_actuator_subset_excitation_all_416_source_paths_still_evaluated";
            }
            if (!requestedPoseCoordinates.empty()) {
                evidenceBoundary +=
                    "_with_explicit_kinematic_source_coordinate_override_and_exact_dependent_polynomial_projection_not_dynamics_or_loaded_contact_validation";
            }
            if (muscleDrivenState.has_value() &&
                muscleDrivenState->assistanceRemovalEvaluated) {
                evidenceBoundary +=
                    "_with_sequential_assisted_then_zero_root_wrench_device_horizons";
            }
            if (muscleDrivenState.has_value() &&
                muscleDrivenState->persistentMetalHorizon &&
                !muscleDrivenState->compiledBalanced) {
                evidenceBoundary +=
                    "_with_reduced_bounded_acceleration_recruitment_not_a_static_equilibrium_certificate";
            }
            if (skinPayload.has_value()) {
                evidenceBoundary +=
                    skinPayload->usesSourceSurfaceLocalWeights
                        ? "_with_four_bone_source_surface_local_linear_blend_bodyparts3d_skin_shell_visual_not_deformable_skin_collision_or_tissue_physics"
                        : skinPayload->usesBoundaryLocalWeights
                        ? "_with_four_bone_boundary_local_linear_blend_bodyparts3d_skin_shell_visual_not_deformable_skin_collision_or_tissue_physics"
                        : "_with_four_bone_linear_blend_bodyparts3d_skin_shell_visual_not_deformable_skin_collision_or_tissue_physics";
            }
            if (torsoAnatomyPayload.has_value()) {
                evidenceBoundary +=
                    "_with_selected_exact_bodyparts3d_organ_vessel_and_spinal_cord_surfaces_single_link_kinematic_visual_bindings_not_organ_or_vessel_mechanics";
            }
            if (passiveFEMTissue.has_value()) {
                evidenceBoundary +=
                    "_with_source_surface_derived_passive_matter_fem_cage_and_fixed_end_rings_prescribed_from_the_bounded_muscle_driven_myoSim_pose_not_a_calibrated_volumetric_muscle_or_full_body_soft_tissue_coupling";
            }
            if (zAnatomyCalfVisualSupplement) {
                evidenceBoundary +=
                    "_with_cc_by_sa_zanatomy_right_calf_visual_supplement_and_transferred_named_bodyparts3d_myosim_articulated_body_weights_not_a_new_force_path_tendon_law_or_continuum";
            }
            if (openKneePayload.has_value()) {
                evidenceBoundary +=
                    openKneePayload->side == metalrobo::NumiHumanKneeSide::left
                        ? "_with_exact_open_knees_oks003_neutral_left_knee_regions_surfaces_ties_and_contact_pairs_registered_by_one_proper_rotation_one_uniform_scale_and_translation_to_live_myoSim_frames_not_subject_matched_loaded_contact_or_deformable_ligament_qualification"
                        : "_with_exact_open_knees_oks003_left_specimen_topology_neutral_sagittal_mirror_into_measured_live_right_knee_frames_not_an_independently_segmented_right_subject_loaded_contact_or_deformable_ligament_qualification";
                if (bonePayload.has_value()) {
                    evidenceBoundary +=
                        "_with_full_BodyParts3D_femur_tibia_fibula_and_patella_presentation_bones_extending_the_exact_NHKNEE1_joint_bone_attachment_and_contact_surfaces_overlap_requires_visual_inspection";
                }
            }
            if (openKneeLigamentFEM.has_value()) {
                evidenceBoundary += openKneeLigamentFEM->liveHumanCoupling
                    ? "_with_all_four_source_quadriceps_nonlinear_Hill_tendon_resultants_replacing_their_complete_patella_terminal_rows_through_the_exact_QAT_patellar_enthesis_and_a_zero_resultant_PTL_patella_to_tibial_enthesis_force_couple_plus_exact_QAT_ACL_PCL_MCL_LCL_and_PTL_passive_Matter_FEM_anchor_reactions_in_the_same_Human_command_buffer_with_bitwise_replay_and_rollback_not_active_volumetric_tendon_transverse_isotropy_contact_sustained_tracking_or_clinical_validation"
                    : openKneeLigamentFEM->header.abi == 2u
                    ? "_with_four_exact_ligament_and_exact_patellar_tendon_surfaces_owned_by_an_accepted_NHKFEM2_three_body_attachment_reaction_snapshot_under_submicron_tibia_translation_not_loaded_flexion_quadriceps_tendon_source_transverse_isotropy_or_clinical_validation"
                    : "_with_four_exact_ligament_surfaces_owned_by_an_accepted_NHKFEM1_two_body_attachment_reaction_snapshot_under_submicron_tibia_translation_not_loaded_flexion_source_transverse_isotropy_or_clinical_validation";
            }
            std::cout << std::setprecision(12)
                      << (bodypartsBoneVisual
                              ? "myosim_articulated_bodyparts_bone_visual=ok"
                              : "myosim_articulated_marker_visual=ok")
                      << " metal_pose_device=\"" << poseDiagnostics.deviceName << "\""
                      << " renderer_device=\"" << rendererDeviceName << "\""
                      << " frame_dimension=" << frameDimension
                      << " renderer_profile=" << rendererProfile.id
                      << " renderer_temporal_samples=" << rendererProfile.temporalSamples
                      << " renderer_area_light_samples=" << rendererProfile.areaLightSamples
                      << " core_bodies=" << rigid.header.engineBodyCount
                      << " rendered_link_visuals=" << renderedBodies
                      << " bodyparts_bones=" << (bonePayload.has_value() ? bonePayload->records.size() : 0u)
                      << " requested_bone_bodies=" << requestedBoneBodyIndices.size()
                      << " requested_bone_stable_ids=" << requestedBoneStableIds.size()
                      << " bodyparts_soft_tissues=" << renderedSoftTissues
                      << " requested_soft_tissue_surfaces=" << requestedSoftTissueStableIds.size()
                      << " soft_tissue_binding=" << (softTissuePayload.has_value()
                              ? (softTissuePayload->usesRouteBodySparseWeights
                                  ? "exact_route_body_sparse_four_influence_world_surface_snapshot"
                                  : "named_body_weighted_world_surface_snapshot")
                              : "none")
                      << " bodyparts_tendon_attachment_collars="
                      << renderedTendonAttachmentCollars
                      << " bodyparts_tendon_attachment_envelopes="
                      << renderedTendonAttachmentEnvelopes
                      << " bodyparts_skin_shells=" << renderedSkinShells
                      << " bodyparts_torso_anatomy_surfaces=" << renderedTorsoAnatomySurfaces
                      << " torso_anatomy_binding=" << (torsoAnatomyPayload.has_value()
                              ? "registered_single_link_kinematic_source_surfaces" : "none")
                      << " passive_fem_tissues=" << renderedPassiveFEMTissues
                      << " pectoralis_fascia_bodies=" << renderedPectoralisFascia
                      << " open_knee_regions=" << renderedOpenKneeRegions
                      << " open_knee_bone_visual_owner="
                      << (openKneePayload.has_value()
                              ? (bonePayload.has_value()
                                  ? "bodyparts3d_full_shafts_plus_nhknee1_articular_bone_ends"
                                  : "nhknee1_joint_only_bones")
                              : "none")
                      << " open_knee_accepted_tissue_fem_regions="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->header.regionCount : 0u)
                      << " open_knee_accepted_tissue_fem_nodes="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->header.nodeCount : 0u)
                      << " open_knee_accepted_tissue_fem_max_displacement_m="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->maximumDisplacementMeters : 0.0f)
                      << " open_knee_live_tissue_fem="
                      << (openKneeLigamentFEM.has_value() &&
                              openKneeLigamentFEM->liveHumanCoupling
                              ? (openKneeLigamentFEM->activeQuadricepsTendonCoupling
                                  ? "same_command_buffer_active_QAT_PTL_extensor_chain_and_passive_bone_reaction"
                                  : "same_command_buffer_passive_bone_reaction")
                              : "none")
                      << " open_knee_quadriceps_endpoint_count="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->quadricepsEndpointCount : 0u)
                      << " open_knee_quadriceps_owner_fraction="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->quadricepsForceOwnerFraction : 0.0)
                      << " open_knee_qat_load_nodes="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->quadricepsLoadNodeCount : 0u)
                      << " open_knee_qat_load_patch_area_m2="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->quadricepsLoadPatchAreaSquareMeters
                              : 0.0)
                      << " open_knee_quadriceps_force_l1_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->quadricepsAppliedForceL1Newtons
                              : 0.0)
                      << " open_knee_quadriceps_force_resultant_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->quadricepsAppliedForceResultantNewtons
                              : 0.0)
                      << " open_knee_quadriceps_enthesis_reaction_resultant_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->quadricepsEnthesisReactionResultantNewtons
                              : 0.0)
                      << " open_knee_ptl_force_l1_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->patellarTendonForceL1Newtons
                              : 0.0)
                      << " open_knee_ptl_force_resultant_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->patellarTendonForceResultantNewtons
                              : 0.0)
                      << " open_knee_ptl_patella_reaction_resultant_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->patellarTendonPatellaReactionResultantNewtons
                              : 0.0)
                      << " open_knee_ptl_tibia_reaction_resultant_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->patellarTendonTibiaReactionResultantNewtons
                              : 0.0)
                      << " open_knee_ptl_patella_load_nodes="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->patellarTendonPatellaLoadNodeCount
                              : 0u)
                      << " open_knee_ptl_tibia_load_nodes="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->patellarTendonTibiaLoadNodeCount
                              : 0u)
                      << " open_knee_assembled_external_force_l1_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->assembledExternalForceL1Newtons
                              : 0.0)
                      << " open_knee_assembled_external_force_resultant_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->assembledExternalForceResultantNewtons
                              : 0.0)
                      << " open_knee_articular_pairs="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularPairCount : 0u)
                      << " open_knee_articular_samples="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularContactSampleCount : 0u)
                      << " open_knee_articular_mechanical_samples="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularMechanicalSampleCount : 0u)
                      << " open_knee_articular_internal_same_body_samples="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularInternalSameBodySampleCount : 0u)
                      << " open_knee_articular_closed_samples="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularClosedSampleCount : 0u)
                      << " open_knee_articular_contact_area_m2="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularContactAreaSquareMeters : 0.0)
                      << " open_knee_articular_normal_force_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularNormalForceNewtons : 0.0)
                      << " open_knee_articular_max_pressure_pa="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularMaximumPressurePascals : 0.0)
                      << " open_knee_articular_body_force_l1_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularBodyForceL1Newtons : 0.0)
                      << " open_knee_articular_force_residual_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularForceResidualNewtons : 0.0)
                      << " open_knee_articular_moment_residual_nm="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularMomentResidualNewtonMeters : 0.0)
                      << " open_knee_articular_stored_energy_j="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularStoredEnergyJoules : 0.0)
                      << " open_knee_articular_max_normal_strain="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularMaximumNormalStrain : 0.0)
                      << " open_knee_articular_max_closure_m="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularMaximumClosureMeters : 0.0)
                      << " open_knee_articular_audited_steps="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularAuditedStepCount : 0u)
                      << " open_knee_articular_trajectory_min_closed_samples="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMinimumClosedSampleCount : 0u)
                      << " open_knee_articular_trajectory_max_closed_samples="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMaximumClosedSampleCount : 0u)
                      << " open_knee_articular_trajectory_min_normal_force_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMinimumNormalForceNewtons : 0.0)
                      << " open_knee_articular_trajectory_max_normal_force_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMaximumNormalForceNewtons : 0.0)
                      << " open_knee_articular_trajectory_max_pressure_pa="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMaximumPressurePascals : 0.0)
                      << " open_knee_articular_trajectory_max_stored_energy_j="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMaximumStoredEnergyJoules : 0.0)
                      << " open_knee_articular_trajectory_max_normal_strain="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMaximumNormalStrain : 0.0)
                      << " open_knee_articular_trajectory_max_closure_m="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMaximumClosureMeters : 0.0)
                      << " open_knee_articular_trajectory_max_force_residual_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMaximumForceResidualNewtons : 0.0)
                      << " open_knee_articular_trajectory_max_moment_residual_nm="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->articularTrajectoryMaximumMomentResidualNewtonMeters : 0.0)
                      << " open_knee_tissue_fem_qualification_flexion_rad="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->qualificationFlexionRadians : 0.0)
                      << " open_knee_projected_rest_visual_correction_max_m="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->maximumProjectedRestVisualCorrectionMeters
                              : 0.0)
                      << " open_knee_projected_rest_reconstruction_max_residual_m="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->maximumProjectedRestReconstructionResidualMeters
                              : 0.0)
                      << " open_knee_tissue_fem_min_J="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->minimumDeterminant : 1.0f)
                      << " open_knee_tissue_fem_max_J="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->maximumDeterminant : 1.0f)
                      << " open_knee_tissue_fem_femur_reaction_l1_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->bodyReactionL1Newtons[0u] : 0.0)
                      << " open_knee_tissue_fem_tibia_reaction_l1_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->bodyReactionL1Newtons[1u] : 0.0)
                      << " open_knee_tissue_fem_patella_reaction_l1_n="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->bodyReactionL1Newtons[2u] : 0.0)
                      << " open_knee_tissue_fem_tibia_anchor_target_max_residual_m="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->maximumAnchorTargetResidualMeters[1u]
                              : 0.0)
                      << " open_knee_tissue_fem_patella_anchor_target_max_residual_m="
                      << (openKneeLigamentFEM.has_value()
                              ? openKneeLigamentFEM->maximumAnchorTargetResidualMeters[2u]
                              : 0.0)
                      << " visual_supplement="
                      << (zAnatomyCalfVisualSupplement ? "zanatomy_right_calf_cc_by_sa" : "none")
                      << " skin_shell_binding=" << (skinPayload.has_value()
                              ? (skinPayload->usesWorldRestNormals
                                  ? "four_body_registered_source_bone_surface_local_linear_blend_world_rest_normals"
                                  : skinPayload->usesSourceSurfaceLocalWeights
                                  ? "four_body_registered_source_bone_surface_local_linear_blend_world_surface_snapshot"
                                  : skinPayload->usesBoundaryLocalWeights
                                  ? "four_body_registered_bone_envelope_boundary_local_linear_blend_world_surface_snapshot"
                                  : "four_body_registered_bone_envelope_linear_blend_world_surface_snapshot")
                              : "none")
                      << " muscle_sites=" << musclePayload.sites.size()
                      << " route_centerline_segments=" << renderedRouteSegments
                      << " source_route_centrelines=" << (sourceRouteCentrelines ? "true" : "false")
                      << " source_route_muscles=" << (resolvedRouteCentrelines.has_value()
                              ? resolvedRouteCentrelines->muscles.size() : 0u)
                      << " source_route_applied_wraps=" << (resolvedRouteCentrelines.has_value()
                              ? resolvedRouteCentrelines->appliedWrapCount : 0u)
                      << " source_route_surface_projected_sites=" << (resolvedRouteCentrelines.has_value()
                              ? resolvedRouteCentrelines->surfaceProjectedAttachmentCount : 0u)
                      << " focus_body_index=" << (focusBodyIndex.has_value()
                              ? std::to_string(*focusBodyIndex) : "none")
                      << " focus_joint_child_body_index="
                      << (focusJointChildBodyIndex.has_value()
                              ? std::to_string(*focusJointChildBodyIndex) : "none")
                      << " camera_framing=" << (cameraFraming.usesJointAnchor
                              ? "posed_mechanics_joint_anchor"
                              : (!focusBodyIndex.has_value()
                                  ? "exact_rendered_source_geometry_bounds"
                                  : (cameraFraming.usesSourceGeometryBounds
                                  ? "focused_body_source_geometry_bounds"
                                  : "focused_body_inspection_fallback")))
                      << " camera_joint_anchor_residual_m="
                      << cameraFraming.jointAnchorResidualMeters
                      << " camera_source_extent_m=" << cameraFraming.sourceExtentMeters
                      << " camera_distance_m=" << cameraFraming.distance
                      << " pose_stage_elapsed_ms=" << poseDiagnostics.elapsedMilliseconds
                      << " renderer_compile_ms_first_camera=" << rendererCompileMilliseconds
                      << " pose_source=" << poseSource
                      << " pose_q_override_count=" << requestedPoseCoordinates.size()
                      << " muscle_step_seconds=" << (muscleStepSeconds.has_value()
                              ? *muscleStepSeconds : 0.0)
                      << " muscle_step_count=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->stepCount : 0u)
                      << " muscle_activation=" << (muscleStepSeconds.has_value()
                              ? muscleActivation.value_or(0.5) : 0.0)
                      << " muscle_activation_scope=" << (muscleDrivenState.has_value()
                              ? (muscleDrivenState->selectedTendonControl
                                  ? "selected_increment_over_compiled_posture"
                                  : muscleDrivenState->selectedSourceMuscleActivationCount == 0u
                                  ? "all_source_muscles"
                                  : "selected_source_muscles")
                              : "none")
                      << " muscle_control_mode=" << (muscleDrivenState.has_value()
                              ? (muscleDrivenState->selectedTendonControl
                                  ? "compiled_posture_plus_selected_increment_" + tendonProgramName + "_transaction"
                                  : muscleDrivenState->persistentMetalHorizon
                                  ? "compiled_persistent_stand"
                                  : "bounded_visual_difference")
                              : "none")
                      << " muscle_selected_source_activation_count=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->selectedSourceMuscleActivationCount : 0u)
                      << " selected_control_baseline=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->selectedControlBaselineEvaluated
                                  ? "evaluated" : "not_requested")
                      << " selected_control_baseline_max_q_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->selectedControlBaselineMaximumQDelta : 0.0)
                      << " selected_control_baseline_max_q_delta_index=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->selectedControlBaselineMaximumQDeltaIndex
                              : MR_INVALID_INDEX)
                      << " selected_control_baseline_max_v_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->selectedControlBaselineMaximumVDelta : 0.0)
                      << " selected_control_baseline_max_v_delta_index=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->selectedControlBaselineMaximumVDeltaIndex
                              : MR_INVALID_INDEX)
                      << " selected_control_baseline_elapsed_ms=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->selectedControlBaselineElapsedMilliseconds : 0.0)
                      << " muscle_passive_baseline=" << (muscleDrivenState.has_value()
                              ? (compliantMusclePayload
                                  ? "nhmyo2_damped_fiber_tendon_equilibrium"
                                  : muscleDrivenState->persistentMetalHorizon
                                  ? "source_passive_bias_excluded_not_registered_equilibrium_preload"
                                  : "source_default_activation_zero_subtracted")
                              : "none")
                      << " persistent_metal_horizon=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->persistentMetalHorizon ? "true" : "false")
                      << " persistent_completed_steps=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->persistentCompletedSteps : 0u)
                      << " tendon_step_transaction=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->tendonStepTransactionEnabled
                                  ? tendonProgramName : "none")
                      << " tendon_borrowed_consumer=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->tendonBorrowedConsumerVerified
                                  ? "same_command_buffer_exact_snapshot" : "not_verified")
                      << " tendon_rollback=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->tendonRollbackVerified
                                  ? "consumer_rejection_preserved_result" : "not_verified")
                      << " tendon_rigid_state_effect=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->tendonContinuumReactionVerified
                                  ? (muscleDrivenState->tendonContinuumPassiveReactionOnly
                                      ? "source_JT_retained_plus_same_command_passive_continuum_anchor_reaction"
                                      : "source_JT_share_replaced_by_same_command_continuum_anchor_reaction")
                                  : (muscleDrivenState.has_value() &&
                                     muscleDrivenState->tendonRigidStateIdentityVerified
                                      ? "transfer_only_bitwise_identical_no_direct_joint_torque"
                                      : "not_verified"))
                      << " tendon_continuum_max_q_delta_from_source_JT="
                      << (muscleDrivenState.has_value()
                              ? muscleDrivenState->tendonContinuumMaximumQDelta : 0.0)
                      << " tendon_continuum_max_v_delta_from_source_JT="
                      << (muscleDrivenState.has_value()
                              ? muscleDrivenState->tendonContinuumMaximumVDelta : 0.0)
                      << " tendon_step_transfers=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->tendonTransferCount : 0u)
                      << " tendon_step_envelope_transfers=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->tendonEnvelopeTransferCount : 0u)
                      << " tendon_step_point_fallbacks=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->tendonPointTransferCount : 0u)
                      << " tendon_step_max_force_residual_n=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->tendonMaximumForceResidual : 0.0)
                      << " tendon_step_max_moment_residual_nm=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->tendonMaximumMomentResidual : 0.0)
                      << " tendon_step_max_generalized_correction=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->tendonMaximumGeneralizedCorrection : 0.0)
                      << " persistent_root_assistance=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->rootAssistanceEnabled ? "world_root_wrench" : "none")
                      << " persistent_assistance_removal=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->assistanceRemovalEvaluated ? "evaluated" : "not_evaluated")
                      << " persistent_max_acceleration=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->persistentMaximumAcceleration : 0.0)
                      << " persistent_max_penetration_m=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->persistentMaximumPenetrationMeters : 0.0)
                      << " persistent_normal_impulse=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->persistentNormalImpulse : 0.0)
                      << " persistent_max_root_assistance_force_n=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->persistentRootAssistanceForce : 0.0)
                      << " persistent_max_root_assistance_torque_nm=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->persistentRootAssistanceTorque : 0.0)
                      << " compiled_stand_active_muscles=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledActiveMuscleCount : 0u)
                      << " compiled_stand_recruited_muscles=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledRecruitedMuscleCount : 0u)
                      << " compiled_stand_active_limits=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledActivePositionLimitCount : 0u)
                      << " compiled_stand_pose_steps=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledAcceptedPoseSteps : 0u)
                      << " compiled_stand_normalized_residual_rms=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledActivationResidualRms : 0.0)
                      << " compiled_stand_initial_normalized_residual_rms=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledInitialActivationResidualRms : 0.0)
                      << " compiled_stand_max_acceleration_residual=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledMaximumAccelerationResidual : 0.0)
                      << " compiled_stand_support_contacts=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledSupportContactCount : 0u)
                      << " compiled_stand_active_support_contacts=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledActiveSupportContactCount : 0u)
                      << " compiled_stand_total_support_force_n=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledTotalSupportForceNewtons : 0.0)
                      << " compiled_stand_max_root_force_residual=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledMaximumRootForceResidual : 0.0)
                      << " compiled_stand_max_root_acceleration_residual=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledMaximumRootAccelerationResidual : 0.0)
                      << " compiled_stand_max_velocity_increment=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledMaximumVelocityIncrement : 0.0)
                      << " compiled_stand_balanced=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->compiledBalanced ? "true" : "false")
                      << " compiled_stand_max_activation=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledMaximumActivation : 0.0)
                      << " compiled_stand_max_equality_reaction=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledMaximumEqualityReaction : 0.0)
                      << " compiled_stand_max_limit_reaction=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledMaximumLimitReaction : 0.0)
                      << " stand_joint_equalities=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->jointEqualityCount : 0u)
                      << " stand_max_equality_position_error=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumJointEqualityPositionError : 0.0)
                      << " stand_max_equality_velocity_error=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumJointEqualityVelocityError : 0.0)
                      << " stand_max_equality_impulse=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumJointEqualityImpulse : 0.0)
                      << " stand_total_equality_impulse=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->totalJointEqualityImpulse : 0.0)
                      << " assisted_configuration_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->assistedConfigurationDelta : 0.0)
                      << " removal_configuration_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->removalConfigurationDelta : 0.0)
                      << " stand_one_step_max_q_error=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->oneStepParityMaximumQError : 0.0)
                      << " stand_one_step_max_v_error=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->oneStepParityMaximumVError : 0.0)
                      << " stand_deterministic_replay=" << (muscleDrivenState.has_value() &&
                              muscleDrivenState->deterministicReplayVerified
                                  ? "bitwise" : "not_requested")
                      << " stand_replay_elapsed_ms=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->deterministicReplayElapsedMilliseconds : 0.0)
                      << " muscle_step_applied_wraps=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->appliedWrapCount : 0u)
                      << " muscle_force_metal_device=\"" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->muscleMetalDeviceName : "none") << "\""
                      << " muscle_force_metal_transactions=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->muscleMetalStepCount : 0u)
                      << " muscle_force_metal_active_records=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->muscleMetalForceRecordCount : 0u)
                      << " muscle_force_metal_elapsed_ms=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->muscleMetalElapsedMilliseconds : 0.0)
                      << " passive_fem_tissue_stable_id=" << (passiveFEMTissue.has_value()
                              ? std::to_string(passiveFEMTissue->stableId) : "none")
                      << " passive_fem_tetrahedra=" << (passiveFEMTissue.has_value()
                              ? passiveFEMTissue->tetrahedronCount : 0u)
                      << " passive_fem_steps=" << (passiveFEMTissue.has_value()
                              ? passiveFEMTissue->completedSteps : 0u)
                      << " passive_fem_fgmres_iterations=" << (passiveFEMTissue.has_value()
                              ? passiveFEMTissue->fgmresIterations : 0u)
                      << " passive_fem_anchor_displacement_m=" << (passiveFEMTissue.has_value()
                              ? passiveFEMTissue->maximumAnchorDisplacementMeters : 0.0f)
                      << " passive_fem_free_displacement_m=" << (passiveFEMTissue.has_value()
                              ? passiveFEMTissue->maximumFreeDisplacementMeters : 0.0f)
                      << " passive_fem_minimum_J=" << (passiveFEMTissue.has_value()
                              ? passiveFEMTissue->minimumDeterminant : 0.0f)
                      << " passive_fem_gpu_elapsed_ms=" << (passiveFEMTissue.has_value()
                              ? passiveFEMTissue->gpuMilliseconds : 0.0)
                      << " passive_fem_device=\"" << (passiveFEMTissue.has_value()
                              ? passiveFEMTissue->deviceName : "none") << "\""
                      << " pectoralis_fascia_nodes=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->nodes.size() : 0u)
                      << " pectoralis_fascia_tetrahedra=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->tetrahedronCount : 0u)
                      << " pectoralis_fascia_anatomical_vertices=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->anatomicalSurfaceNodes.size() : 0u)
                      << " pectoralis_fascia_anatomical_triangles=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->anatomicalSurfaceTriangles.size() : 0u)
                      << " pectoralis_fascia_mapping_max_distance_m=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->maximumAnatomicalMappingDistanceMeters : 0.0f)
                      << " pectoralis_fascia_mapping_rms_distance_m=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->rmsAnatomicalMappingDistanceMeters : 0.0f)
                      << " pectoralis_fascia_deformed_anatomical_vertices=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->anatomicallyDeformedVertexCount : 0u)
                      << " pectoralis_fascia_mapping_max_applied_distance_m=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->maximumAppliedAnatomicalMappingDistanceMeters : 0.0f)
                      << " pectoralis_fascia_fixed_nodes=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->fixedNodeCount : 0u)
                      << " pectoralis_fascia_load_nodes=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->loadNodeCount : 0u)
                      << " pectoralis_fascia_load_fraction=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->loadFraction : 0.0f)
                      << " pectoralis_fascia_applied_force_n=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->appliedForceNewtons : 0.0f)
                      << " pectoralis_fascia_anchor_reaction_resultant_n=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->anchorReactionResultantNewtons : 0.0f)
                      << " pectoralis_fascia_anchor_reaction_l1_n=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->anchorReactionL1Newtons : 0.0f)
                      << " pectoralis_fascia_anchor_reaction_max_node_n=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->maximumAnchorNodeReactionNewtons : 0.0f)
                      << " pectoralis_fascia_steps=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->completedSteps : 0u)
                      << " pectoralis_fascia_fgmres_iterations=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->fgmresIterations : 0u)
                      << " pectoralis_fascia_max_displacement_m=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->maximumDisplacementMeters : 0.0f)
                      << " pectoralis_fascia_minimum_J=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->minimumDeterminant : 0.0f)
                      << " pectoralis_fascia_replay=" << (pectoralisFascia.has_value() &&
                              pectoralisFascia->deterministicReplayVerified ? "bitwise" : "none")
                      << " pectoralis_fascia_rollback=" << (pectoralisFascia.has_value() &&
                              pectoralisFascia->rollbackVerified ? "verified" : "none")
                      << " pectoralis_fascia_coupled_transaction_elapsed_ms=" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->coupledTransactionMilliseconds : 0.0)
                      << " pectoralis_fascia_device=\"" << (pectoralisFascia.has_value()
                              ? pectoralisFascia->deviceName : "none") << "\""
                      << " muscle_step_max_velocity_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumVelocityDelta : 0.0)
                      << " muscle_step_max_velocity_delta_dof=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumVelocityDeltaDof : MR_INVALID_INDEX)
                      << " muscle_step_max_configuration_delta=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumConfigurationDelta : 0.0)
                      << " muscle_step_max_configuration_delta_q=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->maximumConfigurationDeltaQ : MR_INVALID_INDEX)
                      << " compiled_ankle_r=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledFootCoordinates[0] : 0.0)
                      << " compiled_subtalar_r=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledFootCoordinates[1] : 0.0)
                      << " compiled_mtp_r=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledFootCoordinates[2] : 0.0)
                      << " compiled_ankle_l=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledFootCoordinates[3] : 0.0)
                      << " compiled_subtalar_l=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledFootCoordinates[4] : 0.0)
                      << " compiled_mtp_l=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->compiledFootCoordinates[5] : 0.0)
                      << " final_ankle_r=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->finalFootCoordinates[0] : 0.0)
                      << " final_subtalar_r=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->finalFootCoordinates[1] : 0.0)
                      << " final_mtp_r=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->finalFootCoordinates[2] : 0.0)
                      << " final_ankle_l=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->finalFootCoordinates[3] : 0.0)
                      << " final_subtalar_l=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->finalFootCoordinates[4] : 0.0)
                      << " final_mtp_l=" << (muscleDrivenState.has_value()
                              ? muscleDrivenState->finalFootCoordinates[5] : 0.0)
                      << " source_support_contact=" << (sourceSupportContact ? "true" : "false")
                      << " source_support_witnesses=" << (sourceSupportContact
                              ? muscleDrivenState->supportWitnessCount : 0u)
                      << " source_support_active_contacts=" << (sourceSupportContact
                              ? muscleDrivenState->activeSupportContactCount : 0u)
                      << " source_support_max_active_contacts=" << (sourceSupportContact
                              ? muscleDrivenState->maximumActiveSupportContactCount : 0u)
                      << " source_support_min_plane_gap_m=" << (sourceSupportContact
                              ? muscleDrivenState->minimumSupportPlaneGapMeters : 0.0)
                      << " source_support_seed_translation_m=" << (sourceSupportContact
                              ? muscleDrivenState->supportSeedTranslationMeters : 0.0)
                      << " source_support_metal_device=\"" << (sourceSupportContact
                              ? muscleDrivenState->supportDeviceName : "none") << "\""
                      << " source_support_metal_status=" << (sourceSupportContact
                              ? muscleDrivenState->supportMetalStatus : "not_requested")
                      << " source_support_metal_elapsed_ms=" << (sourceSupportContact
                              ? muscleDrivenState->supportGpuElapsedMilliseconds : 0.0)
                      << " source_support_max_gpu_cpu_velocity_error=" << (sourceSupportContact
                              ? muscleDrivenState->supportMaximumGpuCpuVelocityError : 0.0)
                      << " boundary=" << evidenceBoundary
                      << " route_geometry=" << (sourceRouteCentrelines
                              ? (surfaceProjectSourceSites
                                  ? "cpu_fp64_mujoco_tangent_and_wrapped_arc_centreline_at_same_q_with_visual_only_nearest_bodyparts3d_triangle_source_site_projection_not_a_force_path_or_tendon_surface_certificate"
                                  : "cpu_fp64_mujoco_tangent_and_wrapped_arc_centreline_at_same_q_not_an_anatomical_tendon_surface_or_bodyparts3d_surface_attachment_certificate")
                              : "hidden_until_a_source_route_centreline_inspection_is_requested")
                      << '\n';
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "myosim_articulated_visual=failed error=\""
                      << error.what() << "\"\n";
            return 1;
        }
    }
}
