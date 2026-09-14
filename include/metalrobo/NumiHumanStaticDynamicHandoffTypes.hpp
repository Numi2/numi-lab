#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace metalrobo {

inline constexpr std::size_t kNumiHumanHandoffMuscleCount = 416u;
inline constexpr std::size_t kNumiHumanHandoffDofCount = 128u;
inline constexpr std::string_view kNumiHumanPassiveBiasPolicy =
    "legacy_zero_activation_bias_excluded_compliant_tendon_force_retained";
inline constexpr std::string_view kNumiHumanGravityConvention =
    "force_residual=muscle+equality+limit+support+passive-gravity_target";
inline constexpr std::string_view kNumiHumanAcceptedFiberStateSource =
    "compiled_equilibrium_reference_fiber_length";

struct NumiHumanHandoffThresholds {
    double activationAbsolute = 1.0e-7;
    double fiberAbsoluteMeters = 5.0e-7;
    double fiberRelative = 5.0e-6;
    double forceAbsoluteNewtons = 5.0e-2;
    double forceRelative = 5.0e-5;
    double decompositionAbsoluteNewtons = 1.0e-6;
    double residualAbsolute = 1.0e-3;
    double assemblyAbsolute = 1.0e-4;
    double maximumDampedEquilibriumResidual = 1.0e-5;
};

struct NumiHumanHandoffComparison {
    std::size_t count = 0u;
    std::size_t worstIndex = 0u;
    double maximumAbsoluteDelta = 0.0;
    double rmsAbsoluteDelta = 0.0;
    double maximumNormalizedError = 0.0;
    double worstReference = 0.0;
    double worstCandidate = 0.0;
    double worstAbsoluteDelta = 0.0;
    bool passed = false;
};

struct NumiHumanDynamicHandoffSnapshot {
    std::span<const double> activation;
    std::span<const double> fiberLengthMeters;
    std::span<const double> sourceTotalActuatorForceNewtons;
    std::span<const double> excludedPassiveBiasForceNewtons;
    std::span<const double> drivenActuatorForceNewtons;
    std::span<const double> dampedEquilibriumResidual;
    std::span<const double> generalizedMuscleForce;
    std::span<const double> generalizedJointEqualityForce;
    std::span<const double> generalizedPositionLimitForce;
    std::span<const double> generalizedSupportForce;
    std::span<const double> generalizedPassiveForce;
    std::span<const double> gravityTarget;
    std::span<const double> generalizedForceResidual;
    std::string_view fiberStateSource;
    std::string_view stateOwner;
    std::string_view forceOwner;
};

struct NumiHumanStaticDynamicHandoffInput {
    std::span<const double> staticActivation;
    std::span<const double> staticFiberLengthMeters;
    std::span<const double> staticSourceTotalActuatorForceNewtons;
    std::span<const double> staticExcludedPassiveBiasForceNewtons;
    std::span<const double> staticDrivenActuatorForceNewtons;
    std::span<const double> staticGeneralizedMuscleForce;
    std::span<const double> staticGeneralizedJointEqualityForce;
    std::span<const double> staticGeneralizedPositionLimitForce;
    std::span<const double> staticGeneralizedSupportForce;
    std::span<const double> staticGeneralizedPassiveForce;
    std::span<const double> staticGravityTarget;
    std::span<const double> staticGeneralizedForceResidual;
    NumiHumanDynamicHandoffSnapshot dynamic;
};

struct NumiHumanStaticDynamicHandoffEvidence {
    bool inputValid = false;
    NumiHumanHandoffComparison activation;
    NumiHumanHandoffComparison fiberLength;
    NumiHumanHandoffComparison sourceTotalActuatorForce;
    NumiHumanHandoffComparison excludedPassiveBiasForce;
    NumiHumanHandoffComparison drivenActuatorForce;
    NumiHumanHandoffComparison sourceForceDecomposition;
    NumiHumanHandoffComparison generalizedMuscleForce;
    NumiHumanHandoffComparison generalizedJointEqualityForce;
    NumiHumanHandoffComparison generalizedPositionLimitForce;
    NumiHumanHandoffComparison generalizedSupportForce;
    NumiHumanHandoffComparison generalizedPassiveForce;
    NumiHumanHandoffComparison gravityTarget;
    NumiHumanHandoffComparison generalizedForceResidual;
    NumiHumanHandoffComparison staticForceAssembly;
    NumiHumanHandoffComparison dynamicForceAssembly;
    std::size_t worstDampedEquilibriumMuscle = 0u;
    double maximumDampedEquilibriumResidual = 0.0;
    bool activationAndFiberStateParity = false;
    bool perMuscleForceParity = false;
    bool sourceForceDecompositionClosed = false;
    bool fullForceOwnerParity = false;
    bool staticForceAssemblyClosed = false;
    bool dynamicForceAssemblyClosed = false;
    bool generalizedForceParity = false;
    bool fiberTendonEquilibriumClosed = false;
    bool complete = false;
    std::string error;
};

} // namespace metalrobo
