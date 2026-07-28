#pragma once

#include "metalrobo/engine_types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace metalrobo {

// ConstraintIR is the CPU reference for the next pointer-free constraint ABI.
// Records deliberately use fixed-width scalar fields and 16-byte alignment so
// a later Metal header can adopt the same layouts without inheriting STL
// containers. The vectors below own packed streams; records contain offsets,
// never pointers.
inline constexpr std::uint32_t kConstraintIRAbiVersion = 2u;
inline constexpr std::uint32_t kConstraintIRInvalidIndex =
    MR_INVALID_INDEX;
inline constexpr float kConstraintIRUnbounded = 3.402823466e+38F;

enum ConstraintIRBlockFlags : std::uint32_t {
    constraintIRBlockNewImpact = MR_CONSTRAINT_FLAG_NEW_IMPACT,
    constraintIRBlockWarmStarted = MR_CONSTRAINT_FLAG_WARM_STARTED,
    constraintIRBlockDisabled = MR_CONSTRAINT_FLAG_DISABLED,
};

enum ConstraintIRRowFlags : std::uint32_t {
    constraintIRRowPositionStabilized = 1u << 0u,
    constraintIRRowUnilateral = 1u << 1u,
    constraintIRRowContactNormal = 1u << 2u,
    constraintIRRowContactTangent = 1u << 3u,
    constraintIRRowContactTorsion = 1u << 4u,
};

enum ConstraintIREndpointRole : std::uint32_t {
    constraintIREndpointA = 0u,
    constraintIREndpointB = 1u,
    constraintIREndpointWorld = 2u,
};

enum ConstraintIRJacobianKind : std::uint32_t {
    // Anchor is already a world point. This is used by the v1 compatibility
    // adapter because MRContactConstraintGPU does not retain local anchors.
    constraintIRJacobianWorldPoint = 0u,
    constraintIRJacobianBodyLocalPoint = 1u,
    constraintIRJacobianGeneralized = 2u,
    constraintIRJacobianAngular = 3u,
};

struct alignas(16) ConstraintIRStableKey {
    std::uint32_t words[4]{};
};

struct alignas(16) ConstraintIRBlock {
    ConstraintIRStableKey key{};

    std::uint32_t type = MR_CONSTRAINT_CONTACT;
    std::uint32_t dimension = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t islandIndex = 0u;

    std::uint32_t endpointOffset = 0u;
    std::uint32_t endpointCount = 0u;
    std::uint32_t rowOffset = 0u;
    std::uint32_t impulseOffset = 0u;

    std::uint32_t coneIndex = kConstraintIRInvalidIndex;
    std::uint32_t eventSlot = kConstraintIRInvalidIndex;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
};

struct alignas(16) ConstraintIREndpoint {
    std::uint32_t objectIndex = kConstraintIRInvalidIndex;
    std::uint32_t articulationIndex = kConstraintIRInvalidIndex;
    std::uint32_t linkIndex = kConstraintIRInvalidIndex;
    std::uint32_t role = constraintIREndpointWorld;

    std::uint32_t jacobianKind = constraintIRJacobianWorldPoint;
    std::uint32_t flags = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;

    mr_float4 anchor{};
    mr_float4 axis{};
};

// A row stores continuous semantics. Timestep-dependent targets and
// regularization are derived by evaluateConstraintIR; solvers must not
// reinterpret these fields independently.
struct alignas(16) ConstraintIRRow {
    // World direction for spatial rows. Abstract generalized rows may use 0.
    mr_float4 direction{};

    float positionError = 0.0F;
    float targetVelocity = 0.0F;
    float compliance = 0.0F;
    float dissipation = 0.0F;

    float timeConstant = 0.01F;
    float dampingRatio = 1.0F;
    // Contact rows use one canonical redundant encoding so row-driven and
    // cone-driven consumers cannot disagree: the normal is [0, cone cap] (or
    // [0, kConstraintIRUnbounded] when uncapped), while tangent and torsion
    // rows are [-kConstraintIRUnbounded, kConstraintIRUnbounded]. The coupled
    // cone remains the executable friction limit.
    float impulseLower = -kConstraintIRUnbounded;
    float impulseUpper = kConstraintIRUnbounded;

    std::uint32_t flags = 0u;
    std::uint32_t reserved0 = 0u;
    std::uint32_t reserved1 = 0u;
    std::uint32_t reserved2 = 0u;
};

// Elliptic Coulomb data is solver-neutral. The first implementation supports
// exact normal + two-axis friction and optional torsion. Rolling and adhesion
// fields are reserved in the executable semantics and rejected explicitly
// until corresponding rows/projections exist.
struct alignas(16) ConstraintIRCone {
    float staticFrictionU = 0.0F;
    float staticFrictionV = 0.0F;
    float dynamicFrictionU = 0.0F;
    float dynamicFrictionV = 0.0F;

    float rollingLength = 0.0F;
    float torsionalLength = 0.0F;
    float restitution = 0.0F;
    float restitutionThreshold = 0.0F;

    float adhesionImpulse = 0.0F;
    // Zero means unbounded, matching MRContactConstraintGPU::response.w.
    float maximumNormalImpulse = 0.0F;
    float stictionTransitionVelocity = 1.0e-3F;
    float reserved = 0.0F;
};

static_assert(sizeof(ConstraintIRStableKey) == 16u);
static_assert(sizeof(ConstraintIRBlock) == 64u);
static_assert(sizeof(ConstraintIREndpoint) == 64u);
static_assert(sizeof(ConstraintIRRow) == 64u);
static_assert(sizeof(ConstraintIRCone) == 48u);
static_assert(std::is_standard_layout_v<ConstraintIRStableKey>);
static_assert(std::is_standard_layout_v<ConstraintIRBlock>);
static_assert(std::is_standard_layout_v<ConstraintIREndpoint>);
static_assert(std::is_standard_layout_v<ConstraintIRRow>);
static_assert(std::is_standard_layout_v<ConstraintIRCone>);
static_assert(std::is_trivially_copyable_v<ConstraintIRStableKey>);
static_assert(std::is_trivially_copyable_v<ConstraintIRBlock>);
static_assert(std::is_trivially_copyable_v<ConstraintIREndpoint>);
static_assert(std::is_trivially_copyable_v<ConstraintIRRow>);
static_assert(std::is_trivially_copyable_v<ConstraintIRCone>);

struct ConstraintIR {
    std::uint32_t abiVersion = kConstraintIRAbiVersion;
    std::vector<ConstraintIRBlock> blocks;
    std::vector<ConstraintIREndpoint> endpoints;
    std::vector<ConstraintIRRow> rows;
    std::vector<ConstraintIRCone> cones;
    // Packed by each block's impulseOffset and dimension.
    std::vector<float> warmImpulses;

    [[nodiscard]] bool empty() const noexcept {
        return blocks.empty() && endpoints.empty() && rows.empty() &&
            cones.empty() && warmImpulses.empty();
    }
};

enum class ConstraintIRStatus : std::uint32_t {
    success = 0u,
    invalidAbiVersion,
    invalidCount,
    nonCanonicalOrder,
    invalidRange,
    invalidBlock,
    invalidEndpoint,
    nonfiniteData,
    invalidRow,
    invalidCone,
    infeasibleWarmStart,
    unsupportedSemantics,
    invalidEvaluationConfig,
    invalidEvaluationInput,
    invalidResidualInput,
};

struct ConstraintIRDiagnostics {
    ConstraintIRStatus status = ConstraintIRStatus::success;
    std::uint32_t blockIndex = kConstraintIRInvalidIndex;
    std::uint32_t rowIndex = kConstraintIRInvalidIndex;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ConstraintIRStatus::success;
    }
};

[[nodiscard]] bool constraintIRKeyLess(
    const ConstraintIRStableKey& left,
    const ConstraintIRStableKey& right
) noexcept;

[[nodiscard]] bool constraintIRKeyEqual(
    const ConstraintIRStableKey& left,
    const ConstraintIRStableKey& right
) noexcept;

// Validation is read-only. Canonical streams are strictly key-sorted, densely
// packed with no gaps or overlap, and contain no NaN/Inf. Unsupported
// semantics are distinguished from malformed data.
[[nodiscard]] ConstraintIRDiagnostics validateConstraintIR(
    const ConstraintIR& ir
);

struct ConstraintIREvaluationConfig {
    double timestep = 1.0 / 240.0;
    double penetrationSlop = 1.0e-4;
    double maximumDepenetrationVelocity = 2.0;
    // The evaluator clamps tau to at least this multiple of the local step.
    double minimumTimeConstantRatio = 2.0;
    double stictionTransitionVelocity = 1.0e-3;
};

struct ConstraintIREvaluationInput {
    // J*v at the evaluation configuration, packed in IR row order.
    std::span<const float> relativeVelocities{};
    // J*v immediately before impact detection. Empty means reuse
    // relativeVelocities. Restitution only reads a new contact's normal row.
    std::span<const float> preSolveVelocities{};
};

struct alignas(16) EvaluatedConstraintIRRow {
    mr_float4 direction{};

    float targetVelocity = 0.0F;
    // R = compliance / h^2 + dissipation / h.
    float regularization = 0.0F;
    float impulseLower = -kConstraintIRUnbounded;
    float impulseUpper = kConstraintIRUnbounded;

    float sourcePositionError = 0.0F;
    float stabilizationVelocity = 0.0F;
    float sourceTargetVelocity = 0.0F;
    float relativeVelocity = 0.0F;

    float preSolveVelocity = 0.0F;
    float reserved0 = 0.0F;
    float reserved1 = 0.0F;
    float reserved2 = 0.0F;
};

struct alignas(16) EvaluatedConstraintIRCone {
    float effectiveFrictionU = 0.0F;
    float effectiveFrictionV = 0.0F;
    float staticFrictionU = 0.0F;
    float staticFrictionV = 0.0F;

    float dynamicFrictionU = 0.0F;
    float dynamicFrictionV = 0.0F;
    float rollingLength = 0.0F;
    float torsionalLength = 0.0F;

    // Applied bounce speed, zero when restitution did not activate.
    float restitutionVelocity = 0.0F;
    float restitutionThreshold = 0.0F;
    float adhesionImpulse = 0.0F;
    // Zero retains the ABI convention "unbounded".
    float maximumNormalImpulse = 0.0F;
};

static_assert(sizeof(EvaluatedConstraintIRRow) == 64u);
static_assert(sizeof(EvaluatedConstraintIRCone) == 48u);
static_assert(std::is_standard_layout_v<EvaluatedConstraintIRRow>);
static_assert(std::is_standard_layout_v<EvaluatedConstraintIRCone>);
static_assert(std::is_trivially_copyable_v<EvaluatedConstraintIRRow>);
static_assert(std::is_trivially_copyable_v<EvaluatedConstraintIRCone>);

struct EvaluatedConstraintIR {
    std::vector<ConstraintIRBlock> blocks;
    std::vector<ConstraintIREndpoint> endpoints;
    std::vector<EvaluatedConstraintIRRow> rows;
    std::vector<EvaluatedConstraintIRCone> cones;
    // Projected into each contact's selected effective cone. Consumers must
    // use this cache, not the unevaluated IR cache.
    std::vector<float> warmImpulses;
    std::uint64_t semanticFingerprint = 0u;

    [[nodiscard]] bool empty() const noexcept {
        return blocks.empty() && endpoints.empty() && rows.empty() &&
            cones.empty() && warmImpulses.empty();
    }
};

struct ConstraintIREvaluationResult {
    ConstraintIRDiagnostics diagnostics{};
    EvaluatedConstraintIR evaluated;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.succeeded();
    }
};

// This is the only timestep-dependent semantic evaluator. Neither quality nor
// throughput mode receives the unevaluated rows.
[[nodiscard]] ConstraintIREvaluationResult evaluateConstraintIR(
    const ConstraintIR& ir,
    const ConstraintIREvaluationInput& input,
    const ConstraintIREvaluationConfig& config = {}
);

enum class ConstraintIRConsumer : std::uint32_t {
    quality = 0u,
    throughput = 1u,
};

// Consumer views intentionally reference the exact same evaluated buffers.
// The consumer tag selects a numerical algorithm elsewhere; it cannot alter
// constraint semantics.
struct ConstraintIREvaluationView {
    ConstraintIRConsumer consumer = ConstraintIRConsumer::quality;
    std::span<const ConstraintIRBlock> blocks;
    std::span<const ConstraintIREndpoint> endpoints;
    std::span<const EvaluatedConstraintIRRow> rows;
    std::span<const EvaluatedConstraintIRCone> cones;
    std::span<const float> warmImpulses;
    std::uint64_t semanticFingerprint = 0u;
};

[[nodiscard]] ConstraintIREvaluationView makeConstraintIREvaluationView(
    const EvaluatedConstraintIR& evaluated,
    ConstraintIRConsumer consumer
) noexcept;

[[nodiscard]] std::uint64_t fingerprintConstraintSemantics(
    const EvaluatedConstraintIR& evaluated
) noexcept;

struct ConstraintIRResidualConfig {
    // Natural-map step in the scaled constraint metric. Residuals are divided
    // by this value, and only the numerically safe [1e-6, 1e6] range is
    // accepted, so convergence tolerance is independent of this choice.
    double projectionStep = 1.0;
    double impulseTolerance = 1.0e-7;
    double residualTolerance = 1.0e-6;
};

struct ConstraintIRResidualReport {
    ConstraintIRStatus status = ConstraintIRStatus::success;
    std::uint32_t activeBlocks = 0u;
    std::uint32_t scalarRows = 0u;
    std::uint32_t cappedContacts = 0u;
    std::uint32_t coupledTorsionContacts = 0u;
    double maximumNaturalResidual = 0.0;
    double maximumPrimalViolation = 0.0;
    double maximumDualViolation = 0.0;
    double maximumComplementarityResidual = 0.0;
    double maximumScalarKktResidual = 0.0;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status == ConstraintIRStatus::success;
    }

    [[nodiscard]] bool withinTolerance(
        const ConstraintIRResidualConfig& config
    ) const noexcept;
};

// Evaluates a solver-independent natural/KKT residual from post-solve J*v and
// impulses. Contact rows use an exact projection in the scaled metric,
// including the coupled normal/tangent/torsion feasible set; scalar rows use
// bound KKT conditions. Dual-cone/complementarity fields apply to ordinary
// three-row Coulomb cones, while the natural residual is authoritative for a
// four-row torsional patch. A successful report means the diagnostic was
// evaluated, not that the candidate converged.
[[nodiscard]] ConstraintIRResidualReport evaluateConstraintIRResidual(
    const ConstraintIREvaluationView& semantics,
    std::span<const float> relativeVelocities,
    std::span<const float> impulses,
    const ConstraintIRResidualConfig& config = {}
);

struct ConstraintIRV1AdapterConfig {
    float timeConstant = 0.01F;
    float dampingRatio = 1.0F;
    float dissipation = 0.0F;
    float stictionTransitionVelocity = 1.0e-3F;
};

struct ConstraintIRV1AdapterResult {
    ConstraintIRDiagnostics diagnostics{};
    ConstraintIR ir;
    // Packed in row order. Tangential entries equal their authored surface
    // targets because v1 does not retain pre-solve tangential body velocity.
    std::vector<float> preSolveVelocities;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.succeeded();
    }
};

// Converts a v1 contact stream through a private working value. On any
// failure, both output payloads are empty; a partially converted prefix is
// never published.
[[nodiscard]] ConstraintIRV1AdapterResult adaptV1ContactsToConstraintIR(
    std::span<const MRContactConstraintGPU> contacts,
    const ConstraintIRV1AdapterConfig& config = {}
);

} // namespace metalrobo
