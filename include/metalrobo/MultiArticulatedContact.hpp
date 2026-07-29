#pragma once

#include "metalrobo/MultiArticulatedWorld.hpp"
#include "metalrobo/QualityContactSolver.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace metalrobo {

enum class MultiArticulatedContactStatus : std::uint32_t {
    success = 0u,
    invalidModel,
    invalidDimensions,
    invalidContact,
    nonfiniteInput,
    kinematicsFailure,
    factorizationFailure,
    unsupportedConstraint,
    constraintEvaluationFailure,
    constraintFactorizationFailure,
    solverFailure,
    nonfiniteResult,
};

struct MultiContactSceneBodyVelocity {
    std::array<double, 3> linear{};
    std::array<double, 3> angular{};
};

// Exact contact-space operator spanning every articulation in an EngineModel.
// Jacobian and responseColumns are row-major [3 * contactCount][nv].
// responseColumns[row] is M^-1 J_row' and is assembled through the retained
// block-local articulation factors; no dense global mass matrix or inverse is
// constructed.
struct MultiArticulatedContactProblem {
    std::uint32_t nv = 0u;
    // Prefix owned by EngineModel articulations. Dynamic scene bodies append
    // six coordinates each in [linear xyz, angular xyz] order.
    std::uint32_t articulatedNv = 0u;
    std::uint32_t contactCount = 0u;
    MultiArticulationFactorCache factors;
    // One entry per supplied scene body. Dynamic bodies store their packed
    // velocity offset; static and kinematic bodies store MR_INVALID_INDEX.
    std::vector<std::uint32_t> sceneBodyVelocityOffsets;
    std::vector<MultiContactSceneBodyVelocity>
        sceneBodyFreeVelocities;
    std::vector<double> freeVelocity;
    std::vector<double> contactJacobian;
    std::vector<double> responseColumns;
    // Relative point velocity contributed by static/kinematic endpoints that
    // own no generalized coordinates.
    std::vector<double> prescribedContactVelocity;
    // Optional exact reduction through model-owned unbounded generalized
    // equality rows. Contact response is projected into their null space;
    // these records retain the eliminated impulses and evidence.
    std::uint32_t generalizedConstraintRowCount = 0u;
    std::vector<double> generalizedConstraintJacobian;
    std::vector<double> generalizedConstraintResponseColumns;
    std::vector<double> generalizedConstraintFreeImpulses;
    // Row-major [generalized row][contact row]. Final equality impulse is
    // freeImpulse - contactCoupling * contactImpulse.
    std::vector<double> generalizedConstraintContactCoupling;
    std::vector<double> generalizedConstraintTargets;
    std::vector<double> generalizedConstraintRegularization;
    ContactSpaceConicProblem conic;
    std::vector<ArticulatedPointKinematics> pointA;
    std::vector<ArticulatedPointKinematics> pointB;
};

enum class MultiContactEndpointKind : std::uint32_t {
    articulatedBody = 0u,
    sceneBody = 1u,
    staticWorld = 2u,
};

struct MultiContactEndpoint {
    MultiContactEndpointKind kind =
        MultiContactEndpointKind::articulatedBody;
    // Global EngineModel body index for articulatedBody; index into the
    // sceneBodies span for sceneBody; ignored for staticWorld.
    std::uint32_t body = 0u;
    // COM-relative body-axis point except for staticWorld, where this is the
    // witness position in world coordinates.
    std::array<double, 3> localPoint{};
};

struct MultiArticulatedIslandContact {
    MultiContactEndpoint endpointA{};
    MultiContactEndpoint endpointB{};
    std::array<double, 3> normal{0.0, 0.0, 1.0};
    std::array<double, 3> tangentU{1.0, 0.0, 0.0};
    std::array<double, 3> tangentV{0.0, 1.0, 0.0};
    std::array<double, 3> targetVelocity{};
    std::array<double, 3> regularization{
        1.0e-10,
        1.0e-10,
        1.0e-10,
    };
    std::array<double, 3> warmImpulse{};
    double friction = 0.7;
};

struct MultiArticulatedContactSolution {
    std::vector<double> generalizedVelocity;
    std::vector<double> articulatedVelocity;
    // Static and kinematic velocities copy through unchanged.
    std::vector<MultiContactSceneBodyVelocity>
        sceneBodyVelocities;
    std::vector<double> impulses;
    std::vector<double> generalizedConstraintImpulses;
    QualityContactSolution quality;
};

struct MultiArticulatedContactDiagnostics {
    MultiArticulatedContactStatus status =
        MultiArticulatedContactStatus::success;
    ArticulatedDynamicsStatus dynamicsStatus =
        ArticulatedDynamicsStatus::success;
    std::uint32_t articulationCount = 0u;
    std::uint32_t contactCount = 0u;
    std::uint32_t rowCount = 0u;
    std::uint32_t firstFailingContact = MR_INVALID_INDEX;
    std::uint32_t firstFailingArticulation = MR_INVALID_INDEX;
    double maximumFactorResidual = 0.0;
    double maximumDelassusAsymmetry = 0.0;
    double maximumContactVelocityResidual = 0.0;
    double maximumGeneralizedConstraintResidual = 0.0;
    double minimumDelassusDiagonal = 0.0;

    [[nodiscard]] bool succeeded() const noexcept {
        return status ==
            MultiArticulatedContactStatus::success;
    }
};

// Builds a transactional FP64 exact-cone contact problem for contacts whose
// dynamic endpoints are articulation-owned bodies. bodyB may be
// kArticulatedStaticWorld. Self-contact and contacts between distinct
// articulations share the same path.
[[nodiscard]] MultiArticulatedContactDiagnostics
buildMultiArticulatedContactProblem(
    const EngineModel& model,
    std::span<const double> q,
    std::span<const double> freeVelocity,
    std::span<const ArticulatedContact> contacts,
    MultiArticulatedContactProblem& output,
    const ArticulatedDynamicsConfig& config = {}
);

// General heterogeneous island builder. Dynamic scene bodies append 6D
// maximal-coordinate response blocks to the articulated prefix. Kinematic and
// static scene bodies contribute prescribed point velocity but no inverse
// mass. At least one endpoint of every contact must be dynamic.
[[nodiscard]] MultiArticulatedContactDiagnostics
buildMultiArticulatedIslandContactProblem(
    const EngineModel& model,
    std::span<const double> q,
    std::span<const double> freeArticulationVelocity,
    std::span<const MRBodyStateGPU> sceneBodies,
    std::span<const MultiArticulatedIslandContact> contacts,
    MultiArticulatedContactProblem& output,
    const ArticulatedDynamicsConfig& config = {}
);

// Exactly eliminates model-owned unbounded generalized equality/gear rows
// from an already built contact problem. It forms the small constraint
// Schur complement from articulation-local inverse-mass applications,
// projects free velocity and every contact response column, and never
// materializes a dense generalized inverse. Publication is transactional.
[[nodiscard]] MultiArticulatedContactDiagnostics
projectMultiArticulatedContactThroughGeneralizedEqualities(
    const EngineModel& model,
    MultiArticulatedContactProblem& problem,
    const ConstraintIREvaluationConfig& config = {}
);

// Solves the precomputed contact-space problem and transactionally publishes
// generalized velocity and impulses. The post-contact generalized velocity is
// reconstructed from the retained response columns.
[[nodiscard]] MultiArticulatedContactDiagnostics
solveMultiArticulatedContactProblem(
    const MultiArticulatedContactProblem& problem,
    MultiArticulatedContactSolution& output,
    const QualityContactSolverConfig& config = {}
);

[[nodiscard]] const char* multiArticulatedContactStatusName(
    MultiArticulatedContactStatus status
) noexcept;

} // namespace metalrobo
