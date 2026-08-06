#pragma once

#include "numi/matter/shared.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace numi::matter {

struct Dimension {
    std::int8_t length = 0;
    std::int8_t mass = 0;
    std::int8_t time = 0;
    std::int8_t temperature = 0;
    friend constexpr bool operator==(const Dimension&, const Dimension&) = default;
};

inline constexpr Dimension dimensionless{};
inline constexpr Dimension lengthDimension{1, 0, 0, 0};
inline constexpr Dimension massDimension{0, 1, 0, 0};
inline constexpr Dimension timeDimension{0, 0, 1, 0};
inline constexpr Dimension temperatureDimension{0, 0, 0, 1};
inline constexpr Dimension velocityDimension{1, 0, -1, 0};
inline constexpr Dimension densityDimension{-3, 1, 0, 0};
inline constexpr Dimension pressureDimension{-1, 1, -2, 0};
inline constexpr Dimension viscosityDimension{-1, 1, -1, 0};

[[nodiscard]] constexpr Dimension operator+(
    const Dimension left,
    const Dimension right
) noexcept {
    return {
        static_cast<std::int8_t>(left.length + right.length),
        static_cast<std::int8_t>(left.mass + right.mass),
        static_cast<std::int8_t>(left.time + right.time),
        static_cast<std::int8_t>(left.temperature + right.temperature),
    };
}

[[nodiscard]] constexpr Dimension operator-(
    const Dimension left,
    const Dimension right
) noexcept {
    return {
        static_cast<std::int8_t>(left.length - right.length),
        static_cast<std::int8_t>(left.mass - right.mass),
        static_cast<std::int8_t>(left.time - right.time),
        static_cast<std::int8_t>(left.temperature - right.temperature),
    };
}

[[nodiscard]] constexpr Dimension operator*(
    const Dimension value,
    const int exponent
) noexcept {
    return {
        static_cast<std::int8_t>(value.length * exponent),
        static_cast<std::int8_t>(value.mass * exponent),
        static_cast<std::int8_t>(value.time * exponent),
        static_cast<std::int8_t>(value.temperature * exponent),
    };
}

struct Quantity {
    double value = 0.0;
    Dimension dimension{};
};

enum class ExprKind : std::uint8_t {
    constant,
    parameter,
    state,
    deformation,
    deformationDirection,
    rate,
    rateDirection,
    add,
    subtract,
    multiply,
    divide,
    negate,
    logarithm,
    exponential,
    squareRoot,
    absolute,
    minimum,
    maximum,
    integerPower,
    clamp,
};

struct Expr {
    ExprKind kind = ExprKind::constant;
    Dimension dimension{};
    double constant = 0.0;
    std::uint32_t index = 0u;
    int integer = 0;
    std::array<std::uint32_t, 3> arguments{
        NM_INVALID_INDEX,
        NM_INVALID_INDEX,
        NM_INVALID_INDEX,
    };
};

struct ExpressionGraph {
    std::vector<Expr> nodes;

    [[nodiscard]] std::uint32_t append(Expr expression);
    [[nodiscard]] std::uint32_t constant(
        double value,
        Dimension dimension = {}
    );
    [[nodiscard]] std::uint32_t cloneFrom(
        const ExpressionGraph& source,
        std::uint32_t root,
        std::unordered_map<std::uint32_t, std::uint32_t>& memo
    );
};

struct Parameter {
    std::string name;
    Dimension dimension{};
    double defaultValue = 0.0;
    double lower = 0.0;
    double upper = 0.0;
    double proposalSigma = 0.1;
    bool logarithmic = false;
    bool identifiable = false;
};

struct InternalState {
    std::string name;
    Dimension dimension{};
    double initialValue = 0.0;
};

enum class Representation : std::uint8_t {
    rigid,
    mpm,
    fem,
    rod,
    surface,
};

enum class ConstitutiveHint : std::uint8_t {
    generic,
    neoHookean,
    corotated,
    hencky,
    druckerPrager,
    vonMises,
    newtonian,
    viscoHyperelastic,
};

struct MaterialProgram {
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<InternalState> internalState;
    ExpressionGraph expressions;
    std::uint32_t energyRoot = NM_INVALID_INDEX;
    std::uint32_t dissipationRoot = NM_INVALID_INDEX;
    std::uint32_t validityRoot = NM_INVALID_INDEX;
    std::vector<Representation> supportedRepresentations;
    ConstitutiveHint hint = ConstitutiveHint::generic;
    double staticFriction = 0.6;
    double dynamicFriction = 0.5;
    double restitution = 0.0;
    double adhesion = 0.0;
    double minimumDeterminant = 0.05;
    double maximumDeterminant = 20.0;
    double maximumStress = 1.0e12;
    double maximumEnergyDensity = 1.0e12;
    std::uint64_t fingerprint = 0u;
};

struct ScalarBytecode {
    std::vector<NMExpressionInstructionGPU> instructions;
    std::uint32_t maximumStack = 0u;
};

struct ConstitutiveProgram {
    MaterialProgram material;
    std::array<ScalarBytecode, 9> stress;
    std::array<ScalarBytecode, 9> tangentVector;
    std::array<ScalarBytecode, 9> dissipativeStress;
    std::array<ScalarBytecode, 9> dissipativeTangentVector;
    std::optional<ScalarBytecode> validity;
    std::vector<NMParameterRangeGPU> parameters;
    NMMaterialGPU gpu{};
    bool hasDissipation = false;
    std::uint64_t fingerprint = 0u;
};

struct ParticleSource {
    std::array<double, 3> position{};
    std::array<double, 3> velocity{};
    double mass = 0.0;
    double referenceVolume = 0.0;
};

struct TetrahedronSource {
    std::array<std::uint32_t, 4> nodes{};
};

struct RigidProxySource {
    NMRigidShapeKind shape = NM_RIGID_SHAPE_PLANE;
    std::uint32_t bodyIndex = NM_INVALID_INDEX;
    std::uint32_t materialIndex = 0u;
    std::array<double, 3> center{};
    std::array<double, 3> extent{};
    std::array<double, 4> orientation{0.0, 0.0, 0.0, 1.0};
    double radiusOrOffset = 0.0;
    double contactOffset = 0.0;
    bool articulated = false;
    bool dynamic = false;
    bool kinematic = false;
};

struct ObjectSource {
    std::string name;
    std::uint32_t materialIndex = 0u;
    Representation representation = Representation::mpm;
    bool automaticRepresentation = false;
    bool twoWayCoupling = true;
    bool adaptive = false;
    bool identifiable = false;
    std::uint32_t rigidBinding = NM_INVALID_INDEX;
    double characteristicLength = 0.01;
    double deformationTolerance = 1.0e-4;
    double promoteStrain = 0.01;
    double demoteStrain = 0.002;
    std::vector<ParticleSource> particles;
    std::vector<std::array<double, 3>> femNodes;
    std::vector<TetrahedronSource> tetrahedra;
};

struct WorldSource {
    double frameTimestep = 1.0 / 60.0;
    std::array<double, 3> gravity{0.0, 0.0, -9.81};
    std::vector<MaterialProgram> materials;
    std::vector<ObjectSource> objects;
    std::vector<RigidProxySource> rigidProxies;
    std::uint32_t environmentCount = 1u;
    std::uint32_t femPCGIterations = 32u;
    std::uint32_t identificationCandidates = 0u;
    std::uint32_t eventCapacityPerEnvironment = 256u;
    bool deterministic = true;
};

struct CookedMPM {
    std::vector<NMParticleStateGPU> particles;
    std::vector<NMGridNodeStateGPU> nodes;
    std::vector<NMMPMStencilGPU> stencils;
    std::vector<std::uint32_t> nodeIncidence;
    std::vector<NMIncidenceRangeGPU> nodeRanges;
};

struct CookedFEM {
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMTetrahedronGPU> tetrahedra;
    std::vector<std::uint32_t> nodeIncidence;
    std::vector<NMIncidenceRangeGPU> nodeRanges;
};

struct CookedContact {
    std::vector<NMRigidProxyGPU> rigidProxies;
    std::vector<NMContactPairGPU> pairs;
    std::vector<std::uint32_t> continuumIncidence;
    std::vector<NMIncidenceRangeGPU> continuumRanges;
    std::vector<std::uint32_t> rigidIncidence;
    std::vector<NMIncidenceRangeGPU> rigidRanges;
};

enum class ExecutionStageKind : std::uint32_t {
    initializeStatus,
    identifyGenerate,
    mpmGather,
    mpmIntegrate,
    femForce,
    femPCG,
    rigidRefresh,
    contactGenerate,
    contactContinuumGather,
    contactRigidGather,
    mpmCommit,
    femCommit,
    adaptiveMeasure,
    schedulerUpdate,
    adaptiveTransfer,
    rigidBridge,
    identifyUpdate,
};

struct ExecutionStage {
    ExecutionStageKind kind = ExecutionStageKind::initializeStatus;
    std::uint32_t repeat = 1u;
    std::uint32_t workItems = 0u;
    std::uint32_t flags = 0u;
};

struct CompiledMatterWorld {
    NMMatterDispatchGPU dispatch{};
    std::vector<ConstitutiveProgram> constitutive;
    std::vector<NMMaterialGPU> materials;
    std::vector<NMParameterRangeGPU> parameters;
    std::vector<NMExpressionInstructionGPU> instructions;
    std::vector<NMScalarProgramGPU> scalarPrograms;
    std::vector<NMContinuumObjectGPU> objects;
    CookedMPM mpm;
    CookedFEM fem;
    CookedContact contact;
    std::vector<NMAdaptiveStateGPU> adaptive;
    std::vector<NMSchedulerStateGPU> schedulers;
    std::vector<NMIdentificationDistributionGPU> identification;
    std::vector<ExecutionStage> executionPlan;
    std::uint64_t fingerprint = 0u;
};

enum class SectionKind : std::uint32_t {
    dispatch = 1u,
    materials,
    parameters,
    instructions,
    scalarPrograms,
    objects,
    particles,
    gridNodes,
    mpmStencils,
    mpmNodeIncidence,
    mpmNodeRanges,
    femNodes,
    tetrahedra,
    femNodeIncidence,
    femNodeRanges,
    rigidProxies,
    contactPairs,
    contactContinuumIncidence,
    contactContinuumRanges,
    contactRigidIncidence,
    contactRigidRanges,
    adaptive,
    schedulers,
    identification,
    executionPlan,
    generatedMetal,
};

struct PackageSection {
    SectionKind kind = SectionKind::dispatch;
    std::uint64_t offset = 0u;
    std::uint64_t bytes = 0u;
    std::uint64_t count = 0u;
    std::uint64_t fingerprint = 0u;
};

struct MatterPackageHeader {
    std::array<char, 8> magic{'N', 'U', 'M', 'I', 'M', 'A', 'T', 'R'};
    std::uint32_t formatVersion = 1u;
    std::uint32_t abiVersion = NM_MATTER_ABI_VERSION;
    std::uint64_t fingerprint = 0u;
    std::uint64_t sectionCount = 0u;
    std::uint64_t sectionTableOffset = sizeof(MatterPackageHeader);
    std::uint64_t payloadOffset = 0u;
    std::uint64_t payloadBytes = 0u;
};

struct Diagnostic {
    enum class Severity : std::uint8_t { note, warning, error };
    Severity severity = Severity::error;
    std::size_t line = 0u;
    std::size_t column = 0u;
    std::string message;
};

} // namespace numi::matter
