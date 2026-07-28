#pragma once

#include "metalrobo/engine_types.h"
#include "metalrobo/constraint_ir_shared.h"

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
inline constexpr std::uint32_t kConstraintIRAbiVersion =
    MR_CONSTRAINT_IR_ABI_VERSION;
inline constexpr std::uint32_t kConstraintIRInvalidIndex =
    MR_CONSTRAINT_IR_INVALID_INDEX;
inline constexpr float kConstraintIRUnbounded =
    MR_CONSTRAINT_IR_UNBOUNDED;

enum ConstraintIRBlockFlags : std::uint32_t {
    constraintIRBlockNewImpact = MR_CONSTRAINT_IR_BLOCK_NEW_IMPACT,
    constraintIRBlockWarmStarted =
        MR_CONSTRAINT_IR_BLOCK_WARM_STARTED,
    constraintIRBlockDisabled = MR_CONSTRAINT_IR_BLOCK_DISABLED,
};

enum ConstraintIRRowFlags : std::uint32_t {
    constraintIRRowPositionStabilized =
        MR_CONSTRAINT_IR_ROW_POSITION_STABILIZED,
    constraintIRRowUnilateral = MR_CONSTRAINT_IR_ROW_UNILATERAL,
    constraintIRRowContactNormal =
        MR_CONSTRAINT_IR_ROW_CONTACT_NORMAL,
    constraintIRRowContactTangent =
        MR_CONSTRAINT_IR_ROW_CONTACT_TANGENT,
    constraintIRRowContactTorsion =
        MR_CONSTRAINT_IR_ROW_CONTACT_TORSION,
};

enum ConstraintIREndpointRole : std::uint32_t {
    constraintIREndpointA = MR_CONSTRAINT_IR_ENDPOINT_A,
    constraintIREndpointB = MR_CONSTRAINT_IR_ENDPOINT_B,
    constraintIREndpointWorld = MR_CONSTRAINT_IR_ENDPOINT_WORLD,
};

enum ConstraintIRJacobianKind : std::uint32_t {
    // Anchor is already a world point. This is used by the v1 compatibility
    // adapter because MRContactConstraintGPU does not retain local anchors.
    constraintIRJacobianWorldPoint =
        MR_CONSTRAINT_IR_JACOBIAN_WORLD_POINT,
    constraintIRJacobianBodyLocalPoint =
        MR_CONSTRAINT_IR_JACOBIAN_BODY_LOCAL_POINT,
    constraintIRJacobianGeneralized =
        MR_CONSTRAINT_IR_JACOBIAN_GENERALIZED,
    constraintIRJacobianAngular =
        MR_CONSTRAINT_IR_JACOBIAN_ANGULAR,
};

using ConstraintIRStableKey = MRConstraintIRStableKeyGPU;
using ConstraintIRBlock = MRConstraintIRBlockGPU;
using ConstraintIREndpoint = MRConstraintIREndpointGPU;
using ConstraintIRRow = MRConstraintIRRowGPU;
using ConstraintIRCone = MRConstraintIRConeGPU;

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
    // Consumer-independent diagonal floor applied after compliance/h^2 and
    // dissipation/h are evaluated. Set this once at the semantic boundary so
    // quality and throughput solvers cannot invent different regularization.
    double minimumRegularization = 0.0;
};

struct ConstraintIREvaluationInput {
    // J*v at the evaluation configuration, packed in IR row order.
    std::span<const float> relativeVelocities{};
    // J*v immediately before impact detection. Empty means reuse
    // relativeVelocities. Restitution only reads a new contact's normal row.
    std::span<const float> preSolveVelocities{};
};

using EvaluatedConstraintIRRow = MREvaluatedConstraintIRRowGPU;
using EvaluatedConstraintIRCone = MREvaluatedConstraintIRConeGPU;

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

// Validates the complete evaluated stream, including canonical packing,
// contact frames, warm-start feasibility, and the semantic fingerprint.
// Solvers and adapters should call this at trust boundaries rather than
// independently reinterpreting individual records.
[[nodiscard]] ConstraintIRDiagnostics
validateConstraintIREvaluationView(
    const ConstraintIREvaluationView& view
);

[[nodiscard]] std::uint64_t fingerprintConstraintSemantics(
    const EvaluatedConstraintIR& evaluated
) noexcept;

struct ConstraintIRVelocityResult {
    ConstraintIRDiagnostics diagnostics{};
    // J*v in IR row order.
    std::vector<float> relativeVelocities;

    [[nodiscard]] bool succeeded() const noexcept {
        return diagnostics.succeeded();
    }
};

// Computes contact J*v directly from canonical MRBodyStateGPU records for IR
// whose endpoints are world-point Jacobians. Linear rows use
// dot(direction, v_B(point_B) - v_A(point_A)); a fourth contact row uses the
// relative angular velocity for torsion. Other constraint/Jacobian kinds are
// rejected explicitly. The result is transactional.
[[nodiscard]] ConstraintIRVelocityResult
computeConstraintIRWorldPointVelocities(
    const ConstraintIR& ir,
    std::span<const MRBodyStateGPU> bodyStates
);

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
    // Maps canonical block order back to the input v1 contact span.
    std::vector<std::uint32_t> sourceConstraintIndices;
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
