#pragma once

#include "metalrobo/ConstraintIR.hpp"
#include "metalrobo/EngineModel.hpp"
#include "metalrobo/ParallelABASchedule.hpp"
#include "metalrobo/generalized_constraint_shared.h"
#include "metalrobo/parallel_aba_shared.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metalrobo {

namespace detail {
struct MetalMultiArticulatedConstraintContextState;
struct MetalMultiArticulatedConstraintSubmissionState;
} // namespace detail

struct MetalMultiArticulatedConstraintInput;
struct MetalMultiArticulatedConstraintConfig;
struct MetalMultiArticulatedConstraintResult;
struct MetalMultiArticulatedConstraintDiagnostics;

// Immutable host-owned execution plan shared by standalone Metal and the
// future MLX encoder adapter. Model validation, tree scheduling, generalized
// Jacobian compilation, and RHS chunking occur once at cook time. Its spans
// are uploaded into persistent device arenas by the runtime adapter.
class CompiledMetalMultiArticulatedProgram {
public:
    CompiledMetalMultiArticulatedProgram() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] const EngineModel& model() const noexcept;
    [[nodiscard]] const ParallelABASchedule& abaSchedule()
        const noexcept;
    [[nodiscard]] std::span<const float> generalizedJacobian()
        const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> rowChunkOffsets()
        const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> rowChunkCounts()
        const noexcept;
    [[nodiscard]] std::uint32_t rowCount() const noexcept;
    [[nodiscard]] std::uint64_t fingerprint() const noexcept;

private:
    friend MetalMultiArticulatedConstraintDiagnostics
    compileMetalMultiArticulatedProgram(
        const EngineModel&,
        CompiledMetalMultiArticulatedProgram&
    );
    friend MetalMultiArticulatedConstraintDiagnostics
    solveMetalMultiArticulatedConstraints(
        const CompiledMetalMultiArticulatedProgram&,
        const MetalMultiArticulatedConstraintInput&,
        MetalMultiArticulatedConstraintResult&,
        const MetalMultiArticulatedConstraintConfig&
    );

    EngineModel model_;
    ParallelABASchedule abaSchedule_;
    std::vector<float> generalizedJacobian_;
    std::vector<std::uint32_t> rowChunkOffsets_;
    std::vector<std::uint32_t> rowChunkCounts_;
    std::uint64_t fingerprint_ = 0u;
};

struct MetalMultiArticulatedConstraintInput {
    std::size_t environmentCount = 0u;
    // Packed [environment][global q] and [environment][global v].
    std::span<const float> q{};
    std::span<const float> freeVelocity{};
};

enum class MetalGeneralizedConstraintSolverMode : std::uint32_t {
    throughputPGS = 0u,
    qualitySemismoothNewton = 1u,
};

struct MetalMultiArticulatedConstraintConfig {
    ConstraintIREvaluationConfig evaluation{};
    MetalGeneralizedConstraintSolverMode solverMode =
        MetalGeneralizedConstraintSolverMode::throughputPGS;
    std::uint32_t solverIterations = 128u;
    std::uint32_t qualityCGIterations = 96u;
    std::uint32_t qualityLineSearchIterations = 16u;
    float convergenceTolerance = 2.0e-5f;
    float diagonalFloor = 1.0e-12f;
    float qualityNormalEquationRegularization = 1.0e-8f;
    std::string metallibPath;
};

enum class MetalMultiArticulatedConstraintStatus : std::uint32_t {
    success = 0u,
    invalidConfiguration,
    invalidModel,
    unsupportedTopology,
    unsupportedConstraint,
    invalidDimensions,
    arithmeticOverflow,
    nonfiniteInput,
    metallibUnavailable,
    metalDeviceUnavailable,
    metalDeviceUnsupported,
    metalLibraryFailure,
    metalPipelineFailure,
    metalBufferFailure,
    metalCommandFailure,
    gpuEnvironmentFailure,
    nonfiniteResult,
    internalFailure,
    contextBusy,
};

struct MetalMultiArticulatedConstraintLayout {
    MRGeneralizedConstraintDispatchGPU dispatch{};
    std::vector<MRMultiInverseMassDispatchGPU>
        inverseMassDispatches;
    std::size_t qElements = 0u;
    std::size_t velocityElements = 0u;
    std::size_t jacobianElements = 0u;
    std::size_t responseElements = 0u;
    std::size_t delassusElements = 0u;
    std::size_t impulseElements = 0u;
    std::size_t inverseStatusElements = 0u;
    std::size_t totalAllocatedBytes = 0u;
};

struct MetalMultiArticulatedConstraintResult {
    MetalMultiArticulatedConstraintLayout layout;
    std::vector<float> nextVelocity;
    std::vector<float> impulses;
    std::vector<MRGeneralizedConstraintStatusGPU> statuses;
    std::vector<MRInverseMassStatusGPU> inverseMassStatuses;
};

struct MetalMultiArticulatedConstraintDiagnostics {
    MetalMultiArticulatedConstraintStatus status =
        MetalMultiArticulatedConstraintStatus::success;
    MetalMultiArticulatedConstraintLayout layout;
    bool dispatched = false;
    bool published = false;
    std::uint32_t firstFailingEnvironment = MR_INVALID_INDEX;
    std::uint32_t firstGPUStatusCode =
        MR_GENERALIZED_CONSTRAINT_SUCCESS;
    std::uint32_t firstFailingRow = MR_INVALID_INDEX;
    std::uint32_t firstFailingInverseWork = MR_INVALID_INDEX;
    double elapsedMilliseconds = 0.0;
    std::string deviceName;
    std::string message;

    [[nodiscard]] bool succeeded() const noexcept {
        return status ==
            MetalMultiArticulatedConstraintStatus::success;
    }
};

// Transactionally cooks the immutable multi-articulation program. Failed
// compilation leaves output unchanged.
[[nodiscard]] MetalMultiArticulatedConstraintDiagnostics
compileMetalMultiArticulatedProgram(
    const EngineModel& model,
    CompiledMetalMultiArticulatedProgram& output
);

struct MetalMultiArticulatedConstraintContextStats {
    std::uint64_t pipelineCreationCount = 0u;
    std::uint64_t immutableUploadCount = 0u;
    std::uint64_t bufferAllocationCount = 0u;
    std::uint64_t bufferGrowthCount = 0u;
    std::uint64_t submissionCount = 0u;
    std::uint64_t completedSubmissionCount = 0u;
    std::size_t retainedBufferBytes = 0u;
    std::uint64_t programFingerprint = 0u;
    std::uint32_t parallelABAFrontierCount = 0u;
    std::uint32_t maximumABAFrontierWidth = 0u;
    bool hasInFlightSubmission = false;
};

class MetalMultiArticulatedConstraintContext;

class MetalMultiArticulatedConstraintSubmission {
public:
    MetalMultiArticulatedConstraintSubmission() noexcept;
    ~MetalMultiArticulatedConstraintSubmission();

    MetalMultiArticulatedConstraintSubmission(
        MetalMultiArticulatedConstraintSubmission&& other
    ) noexcept;
    MetalMultiArticulatedConstraintSubmission& operator=(
        MetalMultiArticulatedConstraintSubmission&& other
    ) noexcept;

    MetalMultiArticulatedConstraintSubmission(
        const MetalMultiArticulatedConstraintSubmission&
    ) = delete;
    MetalMultiArticulatedConstraintSubmission& operator=(
        const MetalMultiArticulatedConstraintSubmission&
    ) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] MetalMultiArticulatedConstraintDiagnostics wait(
        MetalMultiArticulatedConstraintResult& output
    );

private:
    friend class MetalMultiArticulatedConstraintContext;
    std::unique_ptr<
        detail::MetalMultiArticulatedConstraintSubmissionState
    > state_;
};

// Persistent one-program executor. Pipeline creation and immutable program
// upload occur once; only environment-major input/output capacity can grow.
// One submission may be in flight because the arena is intentionally reused.
class MetalMultiArticulatedConstraintContext {
public:
    MetalMultiArticulatedConstraintContext(
        const CompiledMetalMultiArticulatedProgram& program,
        MetalMultiArticulatedConstraintConfig config = {}
    );
    ~MetalMultiArticulatedConstraintContext();

    MetalMultiArticulatedConstraintContext(
        MetalMultiArticulatedConstraintContext&& other
    ) noexcept;
    MetalMultiArticulatedConstraintContext& operator=(
        MetalMultiArticulatedConstraintContext&& other
    ) noexcept;

    MetalMultiArticulatedConstraintContext(
        const MetalMultiArticulatedConstraintContext&
    ) = delete;
    MetalMultiArticulatedConstraintContext& operator=(
        const MetalMultiArticulatedConstraintContext&
    ) = delete;

    [[nodiscard]] MetalMultiArticulatedConstraintDiagnostics submit(
        const MetalMultiArticulatedConstraintInput& input,
        MetalMultiArticulatedConstraintSubmission& submission
    );
    [[nodiscard]] MetalMultiArticulatedConstraintDiagnostics run(
        const MetalMultiArticulatedConstraintInput& input,
        MetalMultiArticulatedConstraintResult& output
    );
    [[nodiscard]] MetalMultiArticulatedConstraintContextStats stats()
        const noexcept;

private:
    std::shared_ptr<
        detail::MetalMultiArticulatedConstraintContextState
    > state_;
};

// Solves the immutable model-owned non-contact generalized ConstraintIR
// program over all articulations. Sparse J' columns are factor-applied by the
// multi-articulation ABA operator, J M^-1 J' is formed on Metal, and the
// resulting bounded scalar blocks update one global velocity tensor.
// Publication is transactional.
[[nodiscard]] MetalMultiArticulatedConstraintDiagnostics
solveMetalMultiArticulatedConstraints(
    const EngineModel& model,
    const MetalMultiArticulatedConstraintInput& input,
    MetalMultiArticulatedConstraintResult& output,
    const MetalMultiArticulatedConstraintConfig& config = {}
);

// Fast path for repeated execution. It reuses the validated model snapshot,
// cooked ABA topology, generalized Jacobian, and row packetization.
[[nodiscard]] MetalMultiArticulatedConstraintDiagnostics
solveMetalMultiArticulatedConstraints(
    const CompiledMetalMultiArticulatedProgram& program,
    const MetalMultiArticulatedConstraintInput& input,
    MetalMultiArticulatedConstraintResult& output,
    const MetalMultiArticulatedConstraintConfig& config = {}
);

[[nodiscard]] const char*
metalMultiArticulatedConstraintStatusName(
    MetalMultiArticulatedConstraintStatus status
) noexcept;

} // namespace metalrobo
