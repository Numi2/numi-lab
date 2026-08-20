#pragma once

#include "numi/matter/shared.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
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

inline constexpr Dimension kDimensionless{};
inline constexpr Dimension kLength{1, 0, 0, 0};
inline constexpr Dimension kMass{0, 1, 0, 0};
inline constexpr Dimension kTime{0, 0, 1, 0};
inline constexpr Dimension kTemperature{0, 0, 0, 1};
inline constexpr Dimension kVelocity{1, 0, -1, 0};
inline constexpr Dimension kDensity{-3, 1, 0, 0};
inline constexpr Dimension kPressure{-1, 1, -2, 0};
inline constexpr Dimension kViscosity{-1, 1, -1, 0};
inline constexpr Dimension kRate{0, 0, -1, 0};
inline constexpr Dimension kPowerDensity{-1, 1, -3, 0};

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

struct Diagnostic {
    enum class Severity : std::uint8_t { note, warning, error };
    Severity severity = Severity::error;
    std::size_t line = 0u;
    std::size_t column = 0u;
    std::string message;
};

enum class ExprKind : std::uint8_t {
    constant,
    parameter,
    internalState,
    candidateState,
    deformation,
    deformationDirection,
    deformationRate,
    timeStep,
    temperature,
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

    [[nodiscard]] std::uint32_t append(Expr value);
    [[nodiscard]] std::uint32_t constant(
        double value,
        Dimension dimension = {}
    );
};

struct Parameter {
    std::string name;
    Dimension dimension{};
    double defaultValue = 0.0;
    double lower = 0.0;
    double upper = 0.0;
    bool logarithmic = false;
    bool identifiable = false;
};

struct InternalState {
    enum class Transfer : std::uint8_t { average, maximum, sum };
    std::string name;
    Dimension dimension{};
    double initialValue = 0.0;
    Transfer transfer = Transfer::average;
};

enum class Representation : std::uint8_t {
    rigid,
    mpm,
    fem,
};

enum class ConstitutiveHint : std::uint8_t {
    generic,
    neoHookean,
    corotated,
    druckerPrager,
    vonMises,
    viscoHyperelastic,
    polyconvexICNN,
};

struct MixedMaterialSource {
    // Zero requests a compiler-derived hydrostatic tangent at identity.
    double bulkModulus = 0.0;
    double thermalExpansion = 0.0;
    double biotCoefficient = 0.0;
    double referenceTemperature = 293.15;
    double heatCapacity = 0.0;
    double thermalConductivity = 0.0;
    double heatSource = 0.0;
    double jouleHeatFraction = 1.0;
    double poreStorage = 0.0;
    double poreMobility = 0.0;
    double poreSource = 0.0;
    double electricalConductivity = 0.0;
    double activationDiffusivity = 0.0;
    double activationOnRate = 0.0;
    double activationOffRate = 0.0;
    std::array<double, 3> fibreDirection{1.0, 0.0, 0.0};
    double maximumActiveTension = 0.0;
    double activationThreshold = 0.0;
    double activationSlope = 1.0;
    double cohesiveStrength = 0.0;
    double fractureEnergy = 0.0;
};

struct LearnedLayerSource {
    std::uint32_t inputWidth = 0u;
    std::uint32_t outputWidth = 0u;
    // Row-major input and recurrent ICNN weights followed by one bias per
    // output. Recurrent weights are constrained nonnegative.
    std::vector<float> inputWeights;
    std::vector<float> recurrentWeights;
    std::vector<float> biases;
};

struct LearnedMaterialSource {
    std::uint32_t invariantCount = 4u;
    float softplusBeta = 4.0f;
    float determinantFloor = 0.05f;
    float growthCoefficient = 1.0e-6f;
    std::vector<LearnedLayerSource> layers;
    std::uint64_t fingerprint = 0u;
};

struct MaterialProgram {
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<InternalState> internalState;
    ExpressionGraph expressions;
    std::uint32_t energyRoot = NM_INVALID_INDEX;
    std::uint32_t dissipationRoot = NM_INVALID_INDEX;
    std::uint32_t validityRoot = NM_INVALID_INDEX;
    // One next-state expression per internal state. Missing entries are
    // compiled as identity updates, so all state transitions remain explicit
    // and transactionally reproducible on the GPU.
    std::vector<std::uint32_t> stateUpdateRoots;
    // A residual R(next(state), state, F, Fdot, dt, T) = 0 for each
    // implicitly integrated state. A state may own an update or a residual,
    // never both.
    std::vector<std::uint32_t> stateImplicitRoots;
    std::vector<Representation> supportedRepresentations;
    ConstitutiveHint hint = ConstitutiveHint::generic;
    MixedMaterialSource mixed;
    std::optional<LearnedMaterialSource> learned;
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
    std::array<ScalarBytecode, 9> viscousStress;
    std::array<ScalarBytecode, 9> viscousTangentVector;
    std::vector<ScalarBytecode> stateUpdates;
    std::vector<ScalarBytecode> implicitResiduals;
    std::vector<ScalarBytecode> implicitJacobians;
    std::vector<ScalarBytecode> implicitDeformationDirections;
    std::vector<ScalarBytecode> stressStateDerivatives;
    std::optional<ScalarBytecode> dissipation;
    std::optional<ScalarBytecode> validity;
    NMMaterialGPU gpu{};
    std::vector<NMParameterRangeGPU> parameters;
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

struct MixedSolverSource {
    std::uint32_t newtonIterations = NM_MIXED_NEWTON_ITERATIONS;
    std::uint32_t fgmresRestart = NM_MIXED_FGMRES_DEFAULT_RESTART;
    std::uint32_t fgmresIterations = NM_MIXED_FGMRES_ITERATIONS;
    std::uint32_t lineSearchSteps = NM_MIXED_LINE_SEARCH_STEPS;
    // This is a fixed-pass right-preconditioner component. FGMRES remains the
    // sole linear convergence and publication authority.
    std::uint32_t fieldSmootherPasses =
        NM_MIXED_FIELD_SMOOTHER_MAX_PASSES;
    std::uint32_t mutationRestarts = NM_MIXED_MUTATION_RESTARTS;
    double relativeResidual = 1.0e-4;
    double volumeTolerance = 1.0e-4;
    double pressureTolerance = 1.0e-4;
    double transportTolerance = 1.0e-4;
    // Dimensionless floor relative to WorldSource::contactSlop. A separate
    // coordinate-scale FP32 floor is always retained by the Metal kernel.
    double minimumContactSeparationRatio = 1.0e-4;
    double diagonalFloor = 1.0e-8;
    double initialLMShift = 1.0e-6;
    double maximumLMShift = 1.0e4;
    double curvatureTolerance = 1.0e-7;
    double armijo = 1.0e-4;
    double minimumTemperature = 1.0;
    double activationEpsilon = 1.0e-6;
    double pressureStabilization = 0.1;
};

struct FEMCapacitySource {
    // Zero selects the exact authored count. Nonzero values must not be lower
    // than the authored topology and are fingerprinted without blanket growth.
    std::uint32_t nodes = 0u;
    std::uint32_t tetrahedra = 0u;
    std::uint32_t cohesiveFaces = 0u;
    std::uint32_t punctureChannels = 0u;
    std::uint32_t mutationCommands = 0u;
    // Maximum simultaneously active continuum/rigid contact rows. Zero keeps
    // the compatibility behavior of reserving every cooked eligible pair.
    std::uint32_t activeContacts = 0u;
    // Maximum swept continuum primitive pairs per environment. Zero derives
    // a linear manifold budget from FEM faces or MPM material points.
    std::uint32_t deformableContacts = 0u;
};

struct MultiphysicsSource {
    bool enabled = false;
    double initialTemperature = 293.15;
    double initialMechanicalPressure = 0.0;
    double initialPorePressure = 0.0;
    double initialElectricPotential = 0.0;
    double initialActivation = 0.0;
};

struct FieldBoundarySource {
    std::uint32_t node = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t stableIdentifier = 0u;
    std::array<double, 4> value{};
    std::array<double, 3> flux{};
};

struct MutationPolicySource {
    bool enabled = false;
    bool cohesiveFracture = false;
    // Minimum accepted normal impulse required before device contact may
    // trigger puncture. Zero keeps automatic puncture disabled.
    double punctureImpulseThreshold = 0.0;
};

struct MutationCommandSource {
    NMMutationKind kind = NM_MUTATION_DEACTIVATE_TETRAHEDRON;
    std::uint32_t stableIdentifier = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t target = NM_INVALID_INDEX;
    std::uint32_t priority = 0u;
    std::array<double, 4> geometry0{};
    std::array<double, 4> geometry1{};
};

struct RigidProxySource {
    NMRigidShapeKind shape = NM_RIGID_PLANE;
    // Global EngineModel body index used by the global body-state and wrench
    // arenas. NM_INVALID_INDEX creates a world-fixed proxy.
    std::uint32_t bodyIndex = NM_INVALID_INDEX;
    // Environment-local scene-body index. Required only for a non-articulated
    // dynamic proxy that may receive adaptive rigid-state publication.
    std::uint32_t sceneBodyIndex = NM_INVALID_INDEX;
    std::uint32_t materialIndex = 0u;
    // For a puncture-tip capsule, localCenter is the sharp endpoint and
    // localExtent is the base of the tapered tip segment. Capsule contact is
    // geometrically symmetric, but this ordering gives topology mutation an
    // unambiguous physical direction and tract extent.
    std::array<double, 3> localCenter{};
    std::array<double, 3> localExtent{};
    std::array<double, 4> localOrientation{0.0, 0.0, 0.0, 1.0};
    double radiusOrOffset = 0.0;
    bool articulated = false;
    bool dynamic = false;
    bool punctureTip = false;
    bool punctureDilator = false;
    // Live MetalWorld DER capsule. When enabled, body/scene bindings and local
    // capsule endpoints are ignored; strandNodeA/B address the global
    // environment-local rod-node arena and radiusOrOffset remains physical.
    // The segment follows compatible puncture channels and receives the
    // equal-and-opposite accepted Matter impulse on-device.
    bool sutureStrand = false;
    std::uint32_t strandNodeA = NM_INVALID_INDEX;
    std::uint32_t strandNodeB = NM_INVALID_INDEX;
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
    // MPM uses a fixed-capacity Eulerian background grid. These bounds are
    // particle-centre limits in world coordinates; the compiler expands them
    // by the quadratic B-spline support radius and rejects a particle that
    // subsequently leaves the cooked domain rather than retaining stale
    // particle-to-grid bindings.
    std::array<double, 3> mpmGridMinimum{};
    std::array<double, 3> mpmGridMaximum{};
    double rigidTolerance = 1.0e-4;
    double promotionStrain = 0.01;
    double demotionStrain = 0.002;
    std::vector<ParticleSource> particles;
    // FEM topology is authored in the rest frame; this uniform velocity is
    // copied to every unconstrained node at initialization.
    std::array<double, 3> femInitialVelocity{};
    std::vector<std::array<double, 3>> femNodes;
    // Local FEM node indices whose position is prescribed at the authored
    // rest position. Fixed nodes retain their assembled mass for accounting,
    // but have zero velocity and inverse mass in the executable state.
    std::vector<std::uint32_t> femFixedNodes;
    // Optional authored collision surface. Empty preserves source-world
    // compatibility by cooking every FEM node; otherwise only these local
    // nodes receive continuum/rigid contact rows. Topology mutation may use
    // reserved capacity for newly exposed nodes in a later active-list pass.
    std::vector<std::uint32_t> femContactNodes;
    std::vector<TetrahedronSource> tetrahedra;
    bool mixedFEM = true;
    FEMCapacitySource femCapacity;
    MultiphysicsSource multiphysics;
    std::vector<FieldBoundarySource> fieldBoundaries;
    MutationPolicySource mutationPolicy;
    std::vector<MutationCommandSource> mutationCommands;
};

struct WorldSource {
    double frameTimestep = 1.0 / 60.0;
    std::array<double, 3> gravity{0.0, 0.0, -9.81};
    double contactSlop = 1.0e-5;
    double maximumDepenetrationSpeed = 5.0;
    std::uint32_t environmentCount = 1u;
    std::uint32_t identificationCandidates = 0u;
    bool deterministic = true;
    MixedSolverSource mixedSolver;
    std::vector<MaterialProgram> materials;
    std::vector<ObjectSource> objects;
    std::vector<RigidProxySource> rigidProxies;
};

struct CookedMPM {
    std::vector<NMParticleStateGPU> particles;
    std::vector<NMGridNodeStateGPU> nodes;
    std::vector<NMMPMGridGPU> grids;
    std::vector<NMMPMBlockGPU> blocks;
    std::vector<std::uint32_t> blockLookup;
    // Retained for package compatibility; live transfers use sparse block
    // classification and recompute quadratic support from current positions.
    std::vector<NMMPMStencilGPU> stencils;
    std::vector<std::uint32_t> nodeIncidence;
    std::vector<NMIncidenceRangeGPU> nodeRanges;
};

struct CookedFEM {
    std::vector<NMFEMNodeStateGPU> nodes;
    std::vector<NMTetrahedronGPU> tetrahedra;
    std::vector<NMFEMSurfaceFaceGPU> surfaceFaces;
    std::vector<std::uint32_t> nodeIncidence;
    std::vector<NMIncidenceRangeGPU> nodeRanges;
    std::vector<NMFEMCapacityGPU> capacities;
    std::vector<NMFEMFieldStateGPU> fields;
    std::vector<NMFieldBoundaryGPU> fieldBoundaries;
    std::vector<NMFEMTopologyNodeGPU> topologyNodes;
    std::vector<NMCohesiveFaceGPU> cohesiveFaces;
    std::vector<NMMutationCommandGPU> mutationCommands;
    std::vector<NMPunctureChannelGPU> punctureChannels;
};

struct CookedContact {
    std::vector<NMRigidProxyGPU> rigidProxies;
    std::vector<NMContactPairGPU> pairs;
    std::vector<std::uint32_t> nodeIncidence;
    std::vector<NMIncidenceRangeGPU> nodeRanges;
    std::vector<std::uint32_t> rigidIncidence;
    std::vector<NMIncidenceRangeGPU> rigidRanges;
};

struct CompiledWorld {
    NMMatterDispatchGPU dispatch{};
    NMMixedSolverGPU mixedSolver{};
    std::vector<ConstitutiveProgram> constitutive;
    std::vector<NMMaterialGPU> materials;
    std::vector<NMMixedMaterialGPU> mixedMaterials;
    std::vector<NMParameterRangeGPU> parameters;
    std::vector<float> stateInitials;
    std::vector<NMExpressionInstructionGPU> instructions;
    std::vector<NMScalarProgramGPU> scalarPrograms;
    std::vector<NMContinuumObjectGPU> objects;
    CookedMPM mpm;
    CookedFEM fem;
    CookedContact contact;
    std::vector<NMAdaptiveStateGPU> adaptive;
    std::vector<NMSchedulerStateGPU> schedulers;
    std::vector<NMIdentificationDistributionGPU> identification;
    std::vector<NMLearnedMaterialGPU> learnedMaterials;
    std::vector<NMLearnedLayerGPU> learnedLayers;
    std::vector<float> learnedWeights;
    // Canonical authored-physics identity with allocation-only FEM capacities
    // removed. This remains stable across compatible topology growth recooks.
    std::uint64_t physicsFingerprint = 0u;
    // Exact cooked-package identity, including all arena capacities.
    std::uint64_t fingerprint = 0u;
};

struct ParseResult {
    MaterialProgram material;
    std::vector<Diagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept;
};

struct CompileOptions {
    double cfl = 0.35;
    std::uint32_t maximumRateExponent = NM_MAX_RATE_EXPONENT;
    std::uint32_t maximumExpressionStack = NM_EXPRESSION_STACK_CAPACITY;
    bool emitSpecializedMetal = true;
};

struct CompileResult {
    CompiledWorld world;
    std::string generatedMetal;
    std::vector<Diagnostic> diagnostics;

    [[nodiscard]] bool succeeded() const noexcept;
};

[[nodiscard]] ParseResult parseMatter(std::string_view source);
[[nodiscard]] ParseResult parseMatterFile(const std::filesystem::path& path);
[[nodiscard]] std::uint64_t compiledWorldFingerprint(
    const CompiledWorld& world
) noexcept;
// Validates every pointer-free range, material program, sparse MPM block,
// FEM incidence, contact binding, adaptive authority and canonical
// fingerprint before a cooked world is serialized or allocated on Metal.
[[nodiscard]] bool validateCompiledWorldLayout(
    const CompiledWorld& world,
    std::string* error = nullptr
);
[[nodiscard]] CompileResult compileWorld(
    const WorldSource& source,
    const CompileOptions& options = {}
);
[[nodiscard]] std::string emitSpecializedMetal(
    std::span<const ConstitutiveProgram> programs
);
[[nodiscard]] std::string dimensionName(Dimension dimension);

[[nodiscard]] bool writePackage(
    const CompileResult& compiled,
    const std::filesystem::path& path,
    std::string* error = nullptr
);
[[nodiscard]] bool readPackage(
    const std::filesystem::path& path,
    CompiledWorld& world,
    std::string* generatedMetal = nullptr,
    std::string* error = nullptr
);
[[nodiscard]] bool writeLearnedMaterial(
    const LearnedMaterialSource& material,
    const std::filesystem::path& path,
    std::string* error = nullptr
);
[[nodiscard]] bool readLearnedMaterial(
    const std::filesystem::path& path,
    LearnedMaterialSource& material,
    std::string* error = nullptr
);

struct RuntimeConfiguration {
    std::filesystem::path metallib;
    std::uint32_t environmentCount = 0u;
    bool captureEvents = true;
    bool captureDiagnostics = false;
    // Normal rollout uses the current identified mean. Candidate sampling and
    // distribution updates are opt-in because they deliberately perturb the
    // material parameters of designated environments.
    bool automaticIdentification = false;
    bool adaptiveTransfer = true;
};

struct BorrowedRigidWorldBuffers {
    void* q = nullptr;                   // id<MTLBuffer>, float
    void* v = nullptr;                   // id<MTLBuffer>, float
    void* currentBodies = nullptr;       // id<MTLBuffer>, MRBodyStateGPU
    // Environment-major global body wrench arena. Articulated ABA consumes
    // its owned body range; MetalWorld scene prediction consumes free-body
    // entries from the same arena.
    void* bodyWrenches = nullptr;        // id<MTLBuffer>, MRABABodyWrenchGPU
    void* sceneBodies = nullptr;         // id<MTLBuffer>, MRBodyStateGPU
    void* rodNodes = nullptr;             // id<MTLBuffer>, MRRodNodeStateGPU
    void* rodInverseMasses = nullptr;     // id<MTLBuffer>, float
    std::uint32_t currentBodyCount = 0u;
    std::uint32_t currentBodyStride = 0u;
    std::uint32_t bodyWrenchCount = 0u;
    std::uint32_t sceneBodyCount = 0u;
    std::uint32_t bodyWrenchStride = 0u;
    std::uint32_t sceneStride = 0u;
    std::uint32_t rodNodeCount = 0u;
    std::uint32_t rodNodeStride = 0u;
    std::uint32_t qStride = 0u;
    std::uint32_t vStride = 0u;
};

enum class EncodePhase : std::uint32_t {
    preDynamics = 0u,
    postCommit = 1u,
};

enum class CoupledCandidateOperation : std::uint32_t {
    candidateKinematics = 0u,
    massAction = 1u,
    inverseMassPreconditioner = 2u,
    publishCandidate = 3u,
};

struct CoupledCandidateQuery {
    void* input = nullptr;
    void* output = nullptr;
    void* candidateQ = nullptr;
    void* candidateBodies = nullptr;
    void* statuses = nullptr;
    void* pointQueries = nullptr;
    void* pointJacobians = nullptr;
    CoupledCandidateOperation operation =
        CoupledCandidateOperation::candidateKinematics;
    std::uint32_t generalizedVectorStride = 0u;
    std::uint32_t candidateQStride = 0u;
    std::uint32_t candidateBodyStride = 0u;
    std::uint32_t statusStride = 0u;
    std::uint32_t pointCount = 0u;
    std::uint32_t pointStride = 0u;
    std::uint32_t pointJacobianStride = 0u;
};

using EncodeCoupledCandidate = bool (*)(
    void* context,
    const CoupledCandidateQuery& query
);

struct EncodeRequest {
    void* commandBuffer = nullptr; // borrowed id<MTLCommandBuffer>
    BorrowedRigidWorldBuffers rigid{};
    EncodePhase phase = EncodePhase::preDynamics;
    // Optional borrowed [control step][environment] reset stream. A zero
    // stride disables reset consumption. Selected environments restore their
    // continuum/adaptive state and parameter overlay; the learned posterior
    // distribution remains persistent and republishes its current mean.
    void* resetMasks = nullptr; // id<MTLBuffer>, uint32_t
    // Borrowed [environment] MRMetalWorldStatusGPU stream. Post-commit
    // reconciliation uses it to roll continuum state back whenever the
    // enclosing rigid transaction rejected that environment.
    void* environmentStatuses = nullptr;
    // Optional borrowed final MetalWorld contact solve, laid out
    // [environment][rigidContactConstraintStride].  Matter consumes it only
    // during a final post-commit adaptive transfer to re-promote the exact
    // adaptive fallback body that contacted the rigid world.
    void* rigidContactConstraints = nullptr; // id<MTLBuffer>, MRContactConstraintGPU
    void* rigidContactStatuses = nullptr; // id<MTLBuffer>, MRMetalWorldContactStatusGPU
    void* coupledCandidateContext = nullptr;
    EncodeCoupledCandidate encodeCoupledCandidate = nullptr;
    // Optional fixed-stride device command stream. The runtime validates and
    // stages it into candidate topology; callers retain ownership.
    void* mutationCommands = nullptr; // NMMutationCommandGPU
    // Optional complete learned-weight candidate, validated on-device before
    // it becomes part of the control-step transaction.
    void* learnedWeightUpdate = nullptr; // float
    std::uint32_t mutationCommandCount = 0u;
    std::uint32_t mutationCommandStride = 0u;
    std::uint32_t learnedWeightCount = 0u;
    std::uint32_t learnedWeightRevision = 0u;
    std::uint64_t expectedMutationFingerprint = 0u;
    std::uint64_t expectedLearnedFingerprint = 0u;
    std::uint32_t resetMaskStepStride = 0u;
    std::uint32_t rigidContactConstraintStride = 0u;
    std::uint32_t articulationRootBody = 0u;
    std::uint32_t controlStep = 0u;
    std::uint32_t physicsSubstep = 0u;
    std::uint32_t physicsSubsteps = 1u;
    // Optional enclosing MetalWorld substep. Grouped Matter cadence keeps
    // physicsSubstep local to Matter while failure publication must identify
    // the exact rigid/DER transaction that was rejected.
    std::uint32_t rigidWorldPhysicsSubstep = NM_INVALID_INDEX;
    std::uint64_t seed = 0u;
    // Per-call frame duration. Zero selects the cooked package duration.
    float timestepSeconds = 0.0f;
    bool runIdentification = false;
    bool runAdaptiveTransfer = false;
};

struct RuntimeDiagnostics {
    bool encoded = false;
    std::size_t residentBytes = 0u;
    // Host-side command-graph accounting. These count encoded work requests,
    // not necessarily active shader lanes, and require no GPU readback.
    std::uint64_t threadDispatchCount = 0u;
    std::uint64_t simdgroupDispatchCount = 0u;
    std::uint64_t indirectDispatchCount = 0u;
    std::uint64_t requestedThreadCount = 0u;
    std::uint64_t requestedThreadgroupCount = 0u;
    std::string device;
    std::string message;
};

struct TopologyGrowthRequest {
    bool required = false;
    std::uint32_t allocationGeneration = 0u;
    std::uint32_t firstObject = NM_INVALID_INDEX;
    std::uint32_t reason = NM_STATUS_SUCCESS;
    std::uint32_t nodes = 0u;
    std::uint32_t tetrahedra = 0u;
    std::uint32_t cohesiveFaces = 0u;
    std::uint32_t punctureChannels = 0u;
    std::uint32_t incidence = 0u;
    std::uint32_t mutationCommands = 0u;
    std::uint32_t rigidContacts = 0u;
    std::uint32_t deformableContacts = 0u;
};

// Explicit completion-time diagnostic readback.  This is intentionally a
// snapshot rather than a live mapped view: authoritative simulation buffers
// remain private to Metal and callers cannot observe partially encoded work.
struct RuntimeStateSnapshot {
    bool available = false;
    std::string message;
    // Stable authored-physics identity, independent of allocation growth.
    std::uint64_t sourcePhysicsFingerprint = 0u;
    // Exact cooked layout, runtime policy, ABI, and metallib identity. A live
    // restore requires this identity; source equivalence alone is not enough
    // to establish byte-compatible private arenas.
    std::uint64_t deviceProgramFingerprint = 0u;
    // Completed transaction cursor and automatic-identification ownership.
    // These are restored with the private state so a replay does not silently
    // advance a scheduled update twice.
    std::uint32_t controlStep = 0u;
    std::uint32_t physicsSubstep = 0u;
    std::uint32_t identificationGeneration = 0u;
    std::uint32_t identificationCheckpoint = 0u;
    bool identificationAdvanced = false;
    // Live DER edge bound to each compiled strand-proxy slot, in slot order.
    // The revision advances only after a completed phase-boundary GPU update.
    std::vector<std::uint32_t> sutureProxyEdges;
    std::uint64_t sutureProxyBindingRevision = 0u;
    // Invocation cadence relative to the cooked base DER/Matter timestep.
    // The active timestep is base * multiplier / divisor. Exactly one side
    // differs from one, so the ratio has a canonical replay representation.
    std::uint32_t coupledTimestepMultiplier = 1u;
    std::uint32_t coupledTimestepDivisor = 1u;
    // Zero selects the iteration budget fingerprinted into the cooked world.
    // A nonzero value is a completion-boundary runtime override and is part of
    // deterministic continuation authority even though it does not resize the
    // fixed restarted-FGMRES basis.
    std::uint32_t fgmresIterationBudgetOverride = 0u;
    std::vector<NMParticleStateGPU> particles;
    std::vector<NMFEMNodeStateGPU> femNodes;
    std::vector<NMFEMFieldStateGPU> femFields;
    std::vector<NMFEMTopologyNodeGPU> femTopologyNodes;
    std::vector<NMTetrahedronGPU> femTopologyTetrahedra;
    std::vector<NMCohesiveFaceGPU> cohesiveFaces;
    std::vector<NMPunctureChannelGPU> punctureChannels;
    std::vector<NMFEMTopologyStateGPU> topologyStates;
    // Completion-boundary device status, retained even when the enclosing
    // MetalWorld transaction rolled the physical state back.
    std::vector<NMMatterStatusGPU> statuses;
    // Maximum accepted topology/allocation generation across environments.
    // Exact replay records this beside completed TopologyGrowthRequest values.
    std::uint32_t allocationGeneration = 0u;
    std::vector<NMSolverCertificateGPU> solverCertificates;
    // Deterministic active MPM map from the last completed microstep.
    std::vector<std::uint32_t> mpmActiveNodeIndices;
    std::vector<std::uint32_t> mpmNodeToActive;
    std::vector<std::uint32_t> mpmActiveNodeCounts;
    // Last accepted/published generalized rigid increment block.
    std::vector<float> rigidGeneralizedCandidate;
    std::vector<float> learnedWeights;
    std::uint32_t learnedWeightRevision = 0u;
    std::vector<NMAdaptiveStateGPU> adaptive;
    std::vector<NMSchedulerStateGPU> schedulers;
    std::vector<NMRigidReactionGPU> reactions;
    // Completion-boundary projected proxy geometry/kinematics. This is
    // diagnostic readback, not an independently writable rigid state.
    std::vector<NMRigidStateGPU> rigidStates;
    // Completion-boundary primal-contact diagnostics, populated only when
    // RuntimeConfiguration::captureDiagnostics is enabled.
    std::vector<NMContactSampleGPU> contactSamples;
    std::vector<nm_float4> contactHistories;
    std::vector<NMDeformableContactHistoryGPU> deformableContactHistories;
    std::uint32_t materialStateStride = 0u;
    std::vector<float> particleMaterialState;
    std::vector<float> femMaterialState;
    // Diagnostic posterior state and environment-local parameter overlay.
    // These are copied only at the explicit snapshot boundary.
    std::vector<NMIdentificationDistributionGPU> identification;
    std::vector<float> environmentParameters;
};

class Runtime {
public:
    Runtime();
    ~Runtime();
    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    [[nodiscard]] RuntimeDiagnostics initialize(
        const CompiledWorld& world,
        const RuntimeConfiguration& configuration = {}
    );
    [[nodiscard]] RuntimeDiagnostics encode(const EncodeRequest& request);
    [[nodiscard]] TopologyGrowthRequest pendingTopologyGrowth() const noexcept;
    // Encode accepted-state migration from an older runtime into this already
    // initialized, geometrically larger runtime. The caller owns submission;
    // neither runtime commits or waits for the borrowed command buffer.
    [[nodiscard]] RuntimeDiagnostics encodeTopologyGrowth(
        void* commandBuffer,
        const Runtime& source
    );
    // Allocation-owning growth entry point. This destination must be empty;
    // the expanded world is validated and allocated before migration is
    // encoded. The caller still owns submission of the borrowed buffer.
    [[nodiscard]] RuntimeDiagnostics encodeTopologyGrowth(
        void* commandBuffer,
        const Runtime& source,
        const CompiledWorld& expandedWorld,
        const RuntimeConfiguration& configuration = {}
    );
    // Releases a pre/post transaction when the enclosing MetalWorld command
    // buffer is abandoned before commit. Safe to call for an unrelated or
    // already-completed command buffer.
    void cancel(void* commandBuffer) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;
    [[nodiscard]] std::uint64_t sourcePhysicsFingerprint() const noexcept;
    // Fingerprint of world semantics, runtime execution policy, ABI and the
    // exact loaded Matter metallib, used by MetalWorld run identity.
    [[nodiscard]] std::uint64_t deviceProgramFingerprint() const noexcept;
    [[nodiscard]] bool automaticIdentificationEnabled() const noexcept;
    [[nodiscard]] bool adaptiveTransferEnabled() const noexcept;
    [[nodiscard]] bool requiresBodyWrenches() const noexcept;
    [[nodiscard]] bool requiresRodNodes() const noexcept;
    [[nodiscard]] bool requiresCoupledCandidate() const noexcept;
    [[nodiscard]] std::uint32_t coupledCandidatePointCapacity() const noexcept;
    [[nodiscard]] bool requiresRigidContactEvidence() const noexcept;
    // Move a compact live-DER contact set between completed command buffers.
    // firstRodEdges is ordered by compiled strand-proxy slot and must contain
    // unique in-range edges. A transition may replace at most one slot,
    // retaining every other slot as stable physical ownership; the retired
    // slot's friction history is cleared on GPU. Sparse sets allow one fixed-
    // cost graph to follow material through multiple puncture tracts. This
    // method submits and waits for one bounded maintenance blit/dispatch.
    [[nodiscard]] RuntimeDiagnostics setSutureProxyEdges(
        std::span<const std::uint32_t> firstRodEdges,
        std::uint32_t rodNodeCount
    );
    // Selects a power-of-two coarsening of an externally coupled DER/Matter
    // transaction without moving accepted state. The cooked world must own a
    // live strand, use no internal Matter microticks, and be between command
    // buffers. The MetalWorld control step must contain the same number of
    // base DER substeps. Invocation configuration records the active cadence;
    // the immutable device-program/layout fingerprint does not change.
    [[nodiscard]] bool setCoupledTimestepMultiplier(
        std::uint32_t multiplier
    ) noexcept;
    [[nodiscard]] std::uint32_t coupledTimestepMultiplier() const noexcept;
    // Selects a power-of-two refinement of the externally coupled
    // DER/Matter transaction. Accepted state is retained, no hidden Matter
    // microticks are introduced, and the owning MetalWorld step must use the
    // same refined timestep. Selecting a divisor resets the multiplier to one.
    [[nodiscard]] bool setCoupledTimestepDivisor(
        std::uint32_t divisor
    ) noexcept;
    [[nodiscard]] std::uint32_t coupledTimestepDivisor() const noexcept;
    [[nodiscard]] float timestepSeconds() const noexcept;
    // Select a bounded total restarted-FGMRES iteration budget between
    // completed command buffers. The restart width, allocated basis, solver
    // tolerances, and accepted physical state remain unchanged. Passing the
    // cooked budget removes the runtime override.
    [[nodiscard]] bool setFGMRESIterationBudget(
        std::uint32_t iterations
    ) noexcept;
    [[nodiscard]] std::uint32_t fgmresIterationBudget() const noexcept;
    // Synchronously restores a completion-boundary snapshot into an already
    // initialized runtime with the exact same device-program fingerprint.
    // All transactional accepted/candidate/checkpoint mirrors are updated as
    // one bounded Metal command. Validation failures occur before submission;
    // any device-command failure is reported to the caller.
    [[nodiscard]] RuntimeDiagnostics restore(
        const RuntimeStateSnapshot& snapshot
    );
    [[nodiscard]] RuntimeStateSnapshot snapshot() const;
    [[nodiscard]] void* eventBuffer() const noexcept;
    [[nodiscard]] void* statusBuffer() const noexcept;
    [[nodiscard]] void* parameterBuffer() const noexcept;
    [[nodiscard]] void* identificationLossBuffer() const noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace numi::matter
