#include "metalrobo/G1.hpp"
#include "metalrobo/MillardMuscleReference.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

metalrobo::MillardMuscleDefinition makeDefinition(
    const std::uint32_t firstBody,
    const std::uint32_t secondBody
) {
    metalrobo::MillardMuscleDefinition definition;
    definition.maxIsometricForce = 1200.0;
    definition.optimalFiberLength = 0.11;
    definition.tendonSlackLength = 0.24;
    definition.pennationAngleAtOptimal = 0.12;
    definition.fiberDamping = 0.1;
    definition.minimumActivation = 0.01;
    definition.pathPoints = {
        {firstBody, {0.018, -0.014, 0.011}},
        {secondBody, {-0.021, 0.017, -0.009}},
    };
    return definition;
}

} // namespace

int main() {
    try {
        const metalrobo::EngineModel model =
            metalrobo::makeUnitreeG1EngineModel();
        std::string modelReason;
        require(model.valid(&modelReason), "invalid G1 model: " + modelReason);
        const MRArticulationGPU& articulation = model.articulations.at(0u);
        std::vector<double> q(
            model.defaultQ.begin() + articulation.qOffset,
            model.defaultQ.begin() + articulation.qOffset + articulation.nq
        );
        std::vector<double> v(articulation.nv, 0.0);
        const auto definition = makeDefinition(6u, 12u);

        metalrobo::MillardActivationState activationState{
            .activation = definition.minimumActivation,
            .fiberLengthWarmStart = definition.optimalFiberLength,
        };
        const auto activationDiagnostics = metalrobo::advanceMillardActivation(
            activationState, 0.8, definition.minimumActivation, 0.015, 0.050, 0.010
        );
        require(
            activationDiagnostics.succeeded() && activationState.activation > definition.minimumActivation &&
                activationState.activation < 0.8,
            "persistent Millard activation update failed"
        );

        metalrobo::MillardMuscleState state;
        state.activation = 0.5;
        state.normalizedFiberVelocity = 0.0;
        state.fiberLength = definition.optimalFiberLength;
        state.curves.activeForceLength = 1.0;
        state.curves.forceVelocity = 1.0;
        state.curves.passiveForceLength = 0.0;
        state.curves.tendonForceLength =
            state.activation * std::cos(definition.pennationAngleAtOptimal);
        metalrobo::MillardMuscleForce muscleForce;
        const auto forceDiagnostics = metalrobo::evaluateMillardMuscleForce(
            definition, state, muscleForce
        );
        require(
            forceDiagnostics.succeeded() &&
                std::abs(muscleForce.equilibriumResidual) < 1.0e-12,
            "Millard equilibrium-force identity failed"
        );

        const metalrobo::MillardSourceCurveDefinition sourceCurves{
            .minNormActiveFiberLength = 0.25,
            .transitionNormFiberLength = 0.77,
            .maxNormActiveFiberLength = 1.9,
            .shallowAscendingSlope = 0.75,
            .activeMinimumValue = 0.0,
            .concentricSlopeAtVmax = 0.0,
            .concentricSlopeNearVmax = 0.25,
            .isometricSlope = 5.0,
            .eccentricSlopeAtVmax = 0.0,
            .eccentricSlopeNearVmax = 0.15,
            .maxEccentricVelocityForceMultiplier = 1.4,
            .concentricCurviness = 0.6,
            .eccentricCurviness = 0.9,
            .fiberStrainAtZeroForce = -0.18,
            .fiberStrainAtOneNormForce = 0.50,
            .fiberStiffnessAtLowForce = 0.2,
            .fiberStiffnessAtOneNormForce = 2.9,
            .fiberCurviness = 0.75,
            .tendonStrainAtOneNormForce = 0.049,
            .tendonStiffnessAtOneNormForce = 1.375 / 0.049,
            .tendonNormForceAtToeEnd = 2.0 / 3.0,
            .tendonCurviness = 0.5,
        };
        metalrobo::MillardCurveValues evaluatedCurves;
        const auto curveDiagnostics = metalrobo::evaluateMillardSourceCurves(
            sourceCurves, 1.0, 0.0, 1.049, evaluatedCurves
        );
        require(
            curveDiagnostics.succeeded() &&
                std::abs(evaluatedCurves.activeForceLength - 1.0) < 1.0e-10 &&
                std::abs(evaluatedCurves.forceVelocity - 1.0) < 1.0e-10 &&
                evaluatedCurves.passiveForceLength >= 0.0 &&
                std::abs(evaluatedCurves.tendonForceLength - 1.0) < 1.0e-10,
            "source Millard curve construction/evaluation failed"
        );

        metalrobo::MillardMusclePathResult path;
        const auto pathDiagnostics = metalrobo::evaluateMillardMusclePath(
            model, 0u, q, v, definition, path
        );
        require(
            pathDiagnostics.succeeded() && path.length > 1.0e-4 &&
                path.appliedCylinderWrapCount == 0u,
            "unwrapped GeometryPath evaluation failed"
        );
        std::vector<double> generalizedForce(articulation.nv, 0.0);
        const double tendonForce = muscleForce.tendonForce;
        const auto projectionDiagnostics =
            metalrobo::projectMillardMuscleTension(
                model, 0u, q, v, definition, tendonForce,
                generalizedForce, nullptr
            );
        require(
            projectionDiagnostics.succeeded(),
            "analytic Millard generalized-force projection failed"
        );
        double maximumProjection = 0.0;
        double maximumFiniteDifferenceError = 0.0;
        std::uint32_t finiteDifferenceDofCount = 0u;
        constexpr double perturbation = 1.0e-6;
        for (std::size_t dof = 0u; dof < generalizedForce.size(); ++dof) {
            maximumProjection = std::max(
                maximumProjection, std::abs(generalizedForce[dof])
            );
            const MRDofPropertiesGPU& dofMetadata =
                model.dofs[articulation.vOffset + dof];
            if (dofMetadata.qIndex == MR_INVALID_INDEX) {
                continue;
            }
            require(
                dofMetadata.qIndex >= articulation.qOffset &&
                    dofMetadata.qIndex <
                        articulation.qOffset + articulation.nq,
                "G1 DoF q index is outside articulation"
            );
            const std::size_t qIndex =
                dofMetadata.qIndex - articulation.qOffset;
            std::vector<double> plus = q;
            std::vector<double> minus = q;
            plus[qIndex] += perturbation;
            minus[qIndex] -= perturbation;
            metalrobo::MillardMusclePathResult plusPath;
            metalrobo::MillardMusclePathResult minusPath;
            require(
                metalrobo::evaluateMillardMusclePath(
                    model, 0u, plus, v, definition, plusPath
                ).succeeded() &&
                metalrobo::evaluateMillardMusclePath(
                    model, 0u, minus, v, definition, minusPath
                ).succeeded(),
                "finite-difference GeometryPath evaluation failed"
            );
            const double expected = -tendonForce *
                (plusPath.length - minusPath.length) / (2.0 * perturbation);
            maximumFiniteDifferenceError = std::max(
                maximumFiniteDifferenceError,
                std::abs(generalizedForce[dof] - expected)
            );
            ++finiteDifferenceDofCount;
        }
        require(
            maximumProjection > 1.0e-5 &&
                finiteDifferenceDofCount > 0u &&
                maximumFiniteDifferenceError < 2.0e-5,
            "Millard generalized-force projection disagrees with path derivative"
        );

        auto wrappedDefinition = makeDefinition(6u, 6u);
        wrappedDefinition.pathPoints = {
            {6u, {-0.10, 0.0, 0.0}},
            {6u, {0.10, 0.0, 0.0}},
        };
        wrappedDefinition.cylinderWraps = {{
            .bodyIndex = 6u,
            .center = {0.0, 0.0, 0.0},
            .xyzBodyRotation = {0.0, 0.0, 0.0},
            .radius = 0.03,
            .length = 0.5,
        }};
        metalrobo::MillardMusclePathResult wrappedPath;
        const auto wrapDiagnostics = metalrobo::evaluateMillardMusclePath(
            model, 0u, q, v, wrappedDefinition, wrappedPath
        );
        require(
            wrapDiagnostics.succeeded() &&
                wrappedPath.appliedCylinderWrapCount == 1u &&
                wrappedPath.length > 0.2,
            "finite-cylinder GeometryPath wrapping failed"
        );

        std::cout << std::scientific << std::setprecision(6)
                  << "millard_muscle_reference=ok"
                  << " tendon_force=" << tendonForce
                  << " source_curve_tendon=" << evaluatedCurves.tendonForceLength
                  << " path_length=" << path.length
                  << " maximum_projection=" << maximumProjection
                  << " finite_difference_error=" << maximumFiniteDifferenceError
                  << " wrapped_path_length=" << wrappedPath.length
                  << " wraps=" << wrappedPath.appliedCylinderWrapCount
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "millard_muscle_reference=failed " << error.what() << '\n';
        return 1;
    }
}
