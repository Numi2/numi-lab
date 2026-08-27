#include "metalrobo/ArticulatedDynamics.hpp"
#include "metalrobo/MetalArticulatedOperator.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr std::array<char, 8u> kRigidMagic{'N', 'H', 'R', 'I', 'G', 'I', 'D', '2'};
constexpr std::array<char, 8u> kMuscleMagic{'N', 'H', 'M', 'Y', 'O', '1', '\0', '\0'};
constexpr std::uint32_t kRigidABI = 1u;
constexpr std::uint32_t kMuscleABI = 1u;
constexpr std::uint32_t kBridgeABI = 1u;
constexpr std::uint64_t kFNVOffset = 14695981039346656037ull;
constexpr std::uint64_t kFNVPrime = 1099511628211ull;

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
    float values[7]{};
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
#pragma pack(pop)

static_assert(sizeof(RigidHeader) == 80u);
static_assert(sizeof(SourcePoseRecord) == 28u);
static_assert(sizeof(MuscleHeader) == 76u);
static_assert(sizeof(SiteRecord) == 16u);
static_assert(sizeof(WrapRecord) == 64u);
static_assert(sizeof(RouteRecord) == 16u);
static_assert(sizeof(MuscleRecord) == 164u);

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
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
    if (count > 0u) {
        input.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(count * sizeof(T))
        );
        require(input.good(), std::string("truncated ") + description);
    }
    return result;
}

struct LoadedRigid {
    metalrobo::EngineModel model;
    RigidHeader header{};
};

LoadedRigid loadRigid(const char* path) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), std::string("cannot open rigid payload ") + path);
    LoadedRigid loaded;
    readObject(input, loaded.header, "MyoSim rigid header");
    require(
        loaded.header.magic == kRigidMagic &&
            loaded.header.payloadABI == kRigidABI &&
            loaded.header.engineABI == MR_ENGINE_ABI_VERSION &&
            loaded.header.rootBodyIndex == 0u &&
            loaded.header.reserved0 == 0u &&
            loaded.header.sourceBodyCount > 0u &&
            loaded.header.engineBodyCount >= loaded.header.sourceBodyCount &&
            loaded.header.jointCount + 1u == loaded.header.engineBodyCount &&
            loaded.header.nq == loaded.header.nv + 1u,
        "invalid MyoSim rigid header"
    );
    loaded.model.name = "numibrain_myosim_bridge";
    readObject(input, loaded.model.world, "MyoSim world");
    MRArticulationGPU articulation{};
    readObject(input, articulation, "MyoSim articulation");
    loaded.model.articulations.push_back(articulation);
    loaded.model.bodies = readVector<MRBodyPropertiesGPU>(
        input, loaded.header.engineBodyCount, "MyoSim bodies"
    );
    loaded.model.joints = readVector<MRJointDescriptorGPU>(
        input, loaded.header.jointCount, "MyoSim joints"
    );
    loaded.model.dofs = readVector<MRDofPropertiesGPU>(
        input, loaded.header.nv, "MyoSim DoFs"
    );
    loaded.model.defaultQ = readVector<float>(
        input, loaded.header.nq, "MyoSim default q"
    );
    loaded.model.defaultV = readVector<float>(
        input, loaded.header.nv, "MyoSim default v"
    );
    (void)readVector<std::uint32_t>(
        input, loaded.header.sourceBodyCount, "MyoSim body map"
    );
    (void)readVector<SourcePoseRecord>(
        input, loaded.header.sourceBodyCount, "MyoSim source poses"
    );
    require(
        input.peek() == std::char_traits<char>::eof(),
        "MyoSim rigid payload has trailing bytes"
    );
    require(
        articulation.rootType == MR_ROOT_FLOATING &&
            articulation.rootBody == 0u &&
            articulation.bodyCount == loaded.header.engineBodyCount &&
            articulation.nq == loaded.header.nq &&
            articulation.nv == loaded.header.nv,
        "MyoSim articulation does not match its header"
    );
    std::string reason;
    require(loaded.model.valid(&reason), "invalid MyoSim model: " + reason);
    return loaded;
}

struct LoadedMuscles {
    std::vector<MRMujocoMuscleSiteGPU> sites;
    std::vector<MRMujocoMuscleWrapGPU> wraps;
    std::vector<MRMujocoMuscleRouteNodeGPU> routes;
    std::vector<MRMujocoMuscleGPU> muscles;
    std::vector<std::uint32_t> sourceTendonIdentifiers;
};

bool validRouteType(const std::uint32_t type) {
    return type == MR_MUJOCO_MUSCLE_ROUTE_SITE ||
        type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE ||
        type == MR_MUJOCO_MUSCLE_ROUTE_CYLINDER;
}

LoadedMuscles loadMuscles(
    const char* path,
    const RigidHeader& rigid,
    const std::uint32_t requestedMuscles
) {
    std::ifstream input(path, std::ios::binary);
    require(input.is_open(), std::string("cannot open muscle payload ") + path);
    MuscleHeader header{};
    readObject(input, header, "MyoSim muscle header");
    const std::uint32_t selectedMuscleCount = requestedMuscles == 0u
        ? header.muscleCount : requestedMuscles;
    require(
        header.magic == kMuscleMagic &&
            header.payloadABI == kMuscleABI &&
            header.reserved0 == 0u && header.reserved1 == 0u &&
            header.engineBodyCount == rigid.engineBodyCount &&
            header.sourceSHA256 == rigid.sourceSHA256 &&
            header.muscleCount > 0u &&
            selectedMuscleCount <= header.muscleCount,
        "invalid MyoSim muscle header or requested muscle count"
    );
    const auto sourceSites = readVector<SiteRecord>(
        input, header.siteCount, "MyoSim sites"
    );
    const auto sourceWraps = readVector<WrapRecord>(
        input, header.wrapCount, "MyoSim wraps"
    );
    const auto sourceRoutes = readVector<RouteRecord>(
        input, header.routeNodeCount, "MyoSim routes"
    );
    const auto sourceMuscles = readVector<MuscleRecord>(
        input, header.muscleCount, "MyoSim muscles"
    );
    require(
        input.peek() == std::char_traits<char>::eof(),
        "MyoSim muscle payload has trailing bytes"
    );

    LoadedMuscles loaded;
    loaded.sites.reserve(sourceSites.size());
    for (const SiteRecord& source : sourceSites) {
        require(source.bodyIndex < rigid.engineBodyCount, "MyoSim site body is invalid");
        MRMujocoMuscleSiteGPU site{};
        site.bodyIndex = source.bodyIndex;
        site.localPoint = {source.x, source.y, source.z, 0.0f};
        loaded.sites.push_back(site);
    }
    loaded.wraps.reserve(sourceWraps.size());
    for (const WrapRecord& source : sourceWraps) {
        require(
            source.bodyIndex < rigid.engineBodyCount &&
                (source.type == MR_MUJOCO_MUSCLE_ROUTE_SPHERE ||
                 source.type == MR_MUJOCO_MUSCLE_ROUTE_CYLINDER) &&
                source.reserved0 == 0.0f,
            "MyoSim wrap is invalid"
        );
        MRMujocoMuscleWrapGPU wrap{};
        wrap.bodyIndex = source.bodyIndex;
        wrap.type = source.type;
        wrap.localCenter = {source.centerX, source.centerY, source.centerZ, 0.0f};
        wrap.rotationRow0 = {
            source.rotation[0], source.rotation[1], source.rotation[2], 0.0f,
        };
        wrap.rotationRow1 = {
            source.rotation[3], source.rotation[4], source.rotation[5], 0.0f,
        };
        wrap.rotationRow2 = {
            source.rotation[6], source.rotation[7], source.rotation[8], 0.0f,
        };
        wrap.radius = {source.radius, 0.0f, 0.0f, 0.0f};
        loaded.wraps.push_back(wrap);
    }
    loaded.routes.reserve(sourceRoutes.size());
    for (const RouteRecord& source : sourceRoutes) {
        require(
            validRouteType(source.type) && source.reserved0 == 0u,
            "MyoSim route is invalid"
        );
        MRMujocoMuscleRouteNodeGPU route{};
        route.type = source.type;
        route.targetIndex = source.targetIndex;
        route.sideSiteIndex = source.sideSiteIndex;
        loaded.routes.push_back(route);
    }
    loaded.muscles.reserve(selectedMuscleCount);
    loaded.sourceTendonIdentifiers.reserve(selectedMuscleCount);
    for (std::uint32_t muscleIndex = 0u;
         muscleIndex < selectedMuscleCount;
         ++muscleIndex) {
        const MuscleRecord& source = sourceMuscles[muscleIndex];
        require(
            source.sourceTendonIndex < header.sourceTendonCount &&
                std::ranges::find(
                    loaded.sourceTendonIdentifiers,
                    source.sourceTendonIndex
                ) == loaded.sourceTendonIdentifiers.end() &&
                source.reserved0 == 0u &&
                source.routeOffset <= sourceRoutes.size() &&
                source.routeCount <= sourceRoutes.size() - source.routeOffset,
            "MyoSim muscle identifier or route range is invalid"
        );
        MRMujocoMuscleGPU muscle{};
        muscle.route = {source.routeOffset, source.routeCount, 0u, 0u};
        muscle.lengthRangeAndAcceleration = {
            source.values[0], source.values[1], source.values[2], 0.0f,
        };
        muscle.controlRange = {
            source.values[3], source.values[4], 0.0f, 0.0f,
        };
        for (std::size_t index = 0u; index < 10u; ++index) {
            (&muscle.gainParameters[index / 4u].x)[index % 4u] =
                source.values[5u + index];
            (&muscle.biasParameters[index / 4u].x)[index % 4u] =
                source.values[15u + index];
            (&muscle.dynamicParameters[index / 4u].x)[index % 4u] =
                source.values[25u + index];
        }
        loaded.muscles.push_back(muscle);
        loaded.sourceTendonIdentifiers.push_back(source.sourceTendonIndex);
    }
    return loaded;
}

template <typename T>
void mixValue(std::uint64_t& hash, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto bytes = std::as_bytes(std::span{&value, 1u});
    for (const std::byte byte : bytes) {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= kFNVPrime;
    }
}

std::uint64_t stateFingerprint(
    const std::vector<double>& q,
    const std::vector<double>& v,
    const std::vector<MRMujocoMuscleStateGPU>& states,
    const std::uint64_t generation
) {
    std::uint64_t hash = kFNVOffset;
    mixValue(hash, kBridgeABI);
    mixValue(hash, generation);
    for (const double value : q) mixValue(hash, value);
    for (const double value : v) mixValue(hash, value);
    for (const auto& state : states) mixValue(hash, state);
    return hash;
}

void writeError(char* output, const std::size_t capacity, const std::string& message) {
    if (output == nullptr || capacity == 0u) return;
    const std::size_t count = std::min(capacity - 1u, message.size());
    std::memcpy(output, message.data(), count);
    output[count] = '\0';
}

struct CandidateState {
    std::vector<double> q;
    std::vector<double> v;
    std::vector<MRMujocoMuscleStateGPU> states;
    std::uint64_t fingerprint = 0u;
    std::uint64_t borrowedGPUAddress = 0u;
    std::uint64_t transactionFingerprint = 0u;
    std::uint64_t substepFingerprint = 0u;
    float maximumExcitation = 0.0f;
    float maximumAbsoluteGeneralizedForce = 0.0f;
    float maximumAbsoluteMuscleForce = 0.0f;
    std::uint32_t maximumForceMuscleIdentifier =
        std::numeric_limits<std::uint32_t>::max();
    double maximumAbsoluteVelocityDelta = 0.0;
    double maximumAbsoluteConfigurationDelta = 0.0;
};

struct Bridge {
    metalrobo::EngineModel model;
    LoadedMuscles program;
    std::vector<MRArticulatedPointImpulseGPU> points;
    std::uint32_t bodyJacobianPointOffset = 0u;
    std::vector<double> committedQ;
    std::vector<double> committedV;
    std::vector<MRMujocoMuscleStateGPU> committedStates;
    std::uint64_t committedGeneration = 0u;
    std::vector<double> rootQ;
    std::vector<double> rootV;
    std::vector<MRMujocoMuscleStateGPU> rootStates;
    std::uint64_t rootGeneration = 0u;
    std::unique_ptr<metalrobo::MetalArticulatedOperatorContext> context;
    std::unique_ptr<CandidateState> pending;
    std::string lastError;
    bool rootOpen = false;
};

} // namespace

extern "C" void* mr_numibrain_myosim_bridge_create(
    const char* rigidPath,
    const char* musclePath,
    const std::uint32_t muscleCount,
    char* error,
    const std::size_t errorCapacity
) {
    try {
        require(rigidPath != nullptr && musclePath != nullptr, "payload path is null");
        LoadedRigid rigid = loadRigid(rigidPath);
        auto bridge = std::make_unique<Bridge>();
        bridge->model = std::move(rigid.model);
        bridge->program = loadMuscles(musclePath, rigid.header, muscleCount);
        const MRArticulationGPU& articulation = bridge->model.articulations.at(0u);
        bridge->bodyJacobianPointOffset = 0u;
        bridge->points.reserve(4u * articulation.bodyCount);
        for (std::uint32_t localBody = 0u;
             localBody < articulation.bodyCount;
             ++localBody) {
            const std::uint32_t body = articulation.firstBody + localBody;
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
                bridge->points.push_back(point);
            }
        }
        bridge->committedQ.assign(
            bridge->model.defaultQ.begin(), bridge->model.defaultQ.end()
        );
        bridge->committedV.assign(
            bridge->model.defaultV.begin(), bridge->model.defaultV.end()
        );
        bridge->committedStates.resize(bridge->program.muscles.size());
        for (auto& state : bridge->committedStates) {
            state.excitationAndActivation = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        metalrobo::MetalArticulatedOperatorConfig config;
        config.pointJacobiansOnly = true;
        config.mujocoActivationTimestepSeconds = 1.0e-6f;
        bridge->context = std::make_unique<metalrobo::MetalArticulatedOperatorContext>(
            std::move(config)
        );
        writeError(error, errorCapacity, {});
        return bridge.release();
    } catch (const std::exception& exception) {
        writeError(error, errorCapacity, exception.what());
        return nullptr;
    }
}

extern "C" void mr_numibrain_myosim_bridge_destroy(void* handle) {
    delete static_cast<Bridge*>(handle);
}

extern "C" const char* mr_numibrain_myosim_bridge_last_error(void* handle) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge == nullptr ? "bridge handle is null" : bridge->lastError.c_str();
}

extern "C" std::uint32_t mr_numibrain_myosim_bridge_muscle_count(void* handle) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge == nullptr
        ? 0u : static_cast<std::uint32_t>(bridge->program.muscles.size());
}

extern "C" std::uint32_t mr_numibrain_myosim_bridge_muscle_identifier(
    void* handle,
    const std::uint32_t muscleIndex
) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge == nullptr ||
            muscleIndex >= bridge->program.sourceTendonIdentifiers.size()
        ? std::numeric_limits<std::uint32_t>::max()
        : bridge->program.sourceTendonIdentifiers[muscleIndex];
}

extern "C" std::uint32_t mr_numibrain_myosim_bridge_begin_root(void* handle) {
    try {
        auto& bridge = *static_cast<Bridge*>(handle);
        require(!bridge.rootOpen && bridge.pending == nullptr, "root is already open");
        bridge.rootQ = bridge.committedQ;
        bridge.rootV = bridge.committedV;
        bridge.rootStates = bridge.committedStates;
        bridge.rootGeneration = bridge.committedGeneration;
        bridge.rootOpen = true;
        return 0u;
    } catch (...) {
        return 1u;
    }
}

extern "C" std::uint32_t mr_numibrain_myosim_bridge_run_candidate(
    void* handle,
    const void* excitationMetalBuffer,
    const std::uint64_t excitationGPUAddress,
    const std::uint32_t muscleCount,
    const std::uint32_t excitationByteCount,
    const std::uint32_t durationMicroseconds,
    const std::uint64_t transactionFingerprint,
    const std::uint64_t substepFingerprint
) {
    try {
        auto& bridge = *static_cast<Bridge*>(handle);
        bridge.lastError.clear();
        require(
            bridge.rootOpen && bridge.pending == nullptr &&
                excitationMetalBuffer != nullptr &&
                muscleCount == bridge.program.muscles.size() &&
                excitationByteCount == muscleCount * sizeof(float) &&
                durationMicroseconds == 1u &&
                transactionFingerprint != 0u && substepFingerprint != 0u,
            "invalid NumiBrain MyoSim candidate"
        );
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)excitationMetalBuffer;
        require(
            buffer != nil && buffer.gpuAddress == excitationGPUAddress,
            "NumiBrain Metal buffer identity drift"
        );
        std::vector<float> q(bridge.rootQ.begin(), bridge.rootQ.end());
        const metalrobo::MetalArticulatedOperatorInput input{
            .articulationIndex = 0u,
            .environmentCount = 1u,
            .pointCount = bridge.points.size(),
            .q = q,
            .points = bridge.points,
            .mujoco = {
                .muscles = bridge.program.muscles,
                .states = bridge.rootStates,
                .sites = bridge.program.sites,
                .wraps = bridge.program.wraps,
                .routeNodes = bridge.program.routes,
                .bodyJacobianPointOffset = bridge.bodyJacobianPointOffset,
            },
        };
        const metalrobo::MetalBorrowedMujocoExcitationBuffer borrowed{
            .nativeMetalBuffer = excitationMetalBuffer,
            .offsetBytes = 0u,
            .byteCount = excitationByteCount,
        };
        metalrobo::MetalArticulatedOperatorResult result;
        const auto diagnostics = bridge.context->run(
            bridge.model, input, borrowed, result
        );
        if (!diagnostics.succeeded() || !diagnostics.dispatched ||
            !diagnostics.published ||
            result.mujocoActivationStates.size() != muscleCount ||
            result.mujocoResults.size() != muscleCount ||
            result.mujocoGeneralizedForces.size() != bridge.rootV.size()) {
            throw std::runtime_error(
                "NumanX MyoSim candidate failed: host_status=" +
                std::to_string(static_cast<std::uint32_t>(diagnostics.status)) +
                " gpu_status=" + std::to_string(diagnostics.firstGPUStatusCode) +
                " dispatched=" + std::to_string(diagnostics.dispatched) +
                " published=" + std::to_string(diagnostics.published) +
                " states=" +
                std::to_string(result.mujocoActivationStates.size()) +
                " results=" + std::to_string(result.mujocoResults.size()) +
                " forces=" +
                std::to_string(result.mujocoGeneralizedForces.size())
            );
        }
        auto candidate = std::make_unique<CandidateState>();
        candidate->q = bridge.rootQ;
        candidate->v = bridge.rootV;
        candidate->states = result.mujocoActivationStates;
        candidate->borrowedGPUAddress = excitationGPUAddress;
        candidate->transactionFingerprint = transactionFingerprint;
        candidate->substepFingerprint = substepFingerprint;
        for (const auto& state : candidate->states) {
            candidate->maximumExcitation = std::max(
                candidate->maximumExcitation,
                state.excitationAndActivation.x
            );
        }
        for (std::size_t index = 0u; index < result.mujocoResults.size(); ++index) {
            const float absoluteForce = std::abs(
                result.mujocoResults[index].pathForceAndActivationDerivative.z
            );
            if (candidate->maximumForceMuscleIdentifier ==
                    std::numeric_limits<std::uint32_t>::max() ||
                absoluteForce > candidate->maximumAbsoluteMuscleForce) {
                candidate->maximumAbsoluteMuscleForce = absoluteForce;
                candidate->maximumForceMuscleIdentifier =
                    bridge.program.sourceTendonIdentifiers[index];
            }
        }
        require(
            candidate->maximumForceMuscleIdentifier !=
                std::numeric_limits<std::uint32_t>::max(),
            "NumanX MyoSim candidate has no muscle-force receptor identity"
        );
        std::vector<double> generalizedForce(result.mujocoGeneralizedForces.begin(),
                                             result.mujocoGeneralizedForces.end());
        for (const double force : generalizedForce) {
            candidate->maximumAbsoluteGeneralizedForce = std::max(
                candidate->maximumAbsoluteGeneralizedForce,
                static_cast<float>(std::abs(force))
            );
        }
        metalrobo::ArticulatedDynamicsConfig physicalConfig;
        physicalConfig.timestep = 1.0e-6;
        const auto physical = metalrobo::integrateArticulatedState(
            bridge.model,
            0u,
            candidate->q,
            candidate->v,
            generalizedForce,
            {},
            physicalConfig
        );
        require(physical.succeeded(), "NumanX articulated candidate integration failed");
        for (std::size_t index = 0u; index < candidate->v.size(); ++index) {
            candidate->maximumAbsoluteVelocityDelta = std::max(
                candidate->maximumAbsoluteVelocityDelta,
                std::abs(candidate->v[index] - bridge.rootV[index])
            );
        }
        for (std::size_t index = 0u; index < candidate->q.size(); ++index) {
            candidate->maximumAbsoluteConfigurationDelta = std::max(
                candidate->maximumAbsoluteConfigurationDelta,
                std::abs(candidate->q[index] - bridge.rootQ[index])
            );
        }
        candidate->fingerprint = stateFingerprint(
            candidate->q,
            candidate->v,
            candidate->states,
            bridge.rootGeneration + 1u
        );
        require(candidate->fingerprint != 0u, "physical candidate has no identity");
        bridge.pending = std::move(candidate);
        return 0u;
    } catch (const std::exception& exception) {
        if (handle != nullptr) {
            static_cast<Bridge*>(handle)->lastError = exception.what();
        }
        return 1u;
    } catch (...) {
        if (handle != nullptr) {
            static_cast<Bridge*>(handle)->lastError = "unknown physical candidate failure";
        }
        return 1u;
    }
}

extern "C" std::uint32_t mr_numibrain_myosim_bridge_accept_candidate(void* handle) {
    try {
        auto& bridge = *static_cast<Bridge*>(handle);
        require(bridge.rootOpen && bridge.pending != nullptr, "no physical candidate");
        bridge.rootQ = std::move(bridge.pending->q);
        bridge.rootV = std::move(bridge.pending->v);
        bridge.rootStates = std::move(bridge.pending->states);
        ++bridge.rootGeneration;
        bridge.pending.reset();
        return 0u;
    } catch (...) {
        return 1u;
    }
}

extern "C" void mr_numibrain_myosim_bridge_reject_candidate(void* handle) {
    if (handle != nullptr) static_cast<Bridge*>(handle)->pending.reset();
}

extern "C" std::uint32_t mr_numibrain_myosim_bridge_commit_root(void* handle) {
    try {
        auto& bridge = *static_cast<Bridge*>(handle);
        require(bridge.rootOpen && bridge.pending == nullptr, "root cannot commit");
        bridge.committedQ = std::move(bridge.rootQ);
        bridge.committedV = std::move(bridge.rootV);
        bridge.committedStates = std::move(bridge.rootStates);
        bridge.committedGeneration = bridge.rootGeneration;
        bridge.rootOpen = false;
        return 0u;
    } catch (...) {
        return 1u;
    }
}

extern "C" void mr_numibrain_myosim_bridge_abort_root(void* handle) {
    if (handle == nullptr) return;
    auto& bridge = *static_cast<Bridge*>(handle);
    bridge.pending.reset();
    bridge.rootQ.clear();
    bridge.rootV.clear();
    bridge.rootStates.clear();
    bridge.rootGeneration = bridge.committedGeneration;
    bridge.rootOpen = false;
}

extern "C" std::uint64_t mr_numibrain_myosim_bridge_pending_fingerprint(void* handle) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge != nullptr && bridge->pending != nullptr
        ? bridge->pending->fingerprint : 0u;
}

extern "C" std::uint64_t mr_numibrain_myosim_bridge_pending_borrowed_gpu_address(void* handle) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge != nullptr && bridge->pending != nullptr
        ? bridge->pending->borrowedGPUAddress : 0u;
}

extern "C" float mr_numibrain_myosim_bridge_pending_maximum_excitation(void* handle) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge != nullptr && bridge->pending != nullptr
        ? bridge->pending->maximumExcitation : 0.0f;
}

extern "C" float mr_numibrain_myosim_bridge_pending_maximum_force(void* handle) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge != nullptr && bridge->pending != nullptr
        ? bridge->pending->maximumAbsoluteGeneralizedForce : 0.0f;
}

extern "C" float mr_numibrain_myosim_bridge_pending_maximum_muscle_force(
    void* handle
) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge != nullptr && bridge->pending != nullptr
        ? bridge->pending->maximumAbsoluteMuscleForce : 0.0f;
}

extern "C" std::uint32_t mr_numibrain_myosim_bridge_pending_maximum_force_muscle_identifier(
    void* handle
) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge != nullptr && bridge->pending != nullptr
        ? bridge->pending->maximumForceMuscleIdentifier
        : std::numeric_limits<std::uint32_t>::max();
}

extern "C" double mr_numibrain_myosim_bridge_pending_maximum_velocity_delta(
    void* handle
) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge != nullptr && bridge->pending != nullptr
        ? bridge->pending->maximumAbsoluteVelocityDelta : 0.0;
}

extern "C" double mr_numibrain_myosim_bridge_pending_maximum_configuration_delta(
    void* handle
) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge != nullptr && bridge->pending != nullptr
        ? bridge->pending->maximumAbsoluteConfigurationDelta : 0.0;
}

extern "C" std::uint64_t mr_numibrain_myosim_bridge_committed_fingerprint(void* handle) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge == nullptr ? 0u : stateFingerprint(
        bridge->committedQ,
        bridge->committedV,
        bridge->committedStates,
        bridge->committedGeneration
    );
}

extern "C" std::uint64_t mr_numibrain_myosim_bridge_committed_generation(void* handle) {
    const auto* bridge = static_cast<const Bridge*>(handle);
    return bridge == nullptr ? 0u : bridge->committedGeneration;
}
