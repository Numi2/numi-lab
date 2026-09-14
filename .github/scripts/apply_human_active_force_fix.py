#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "src/core/NumiHumanMuscleEquilibrium.cpp"
HEADER = ROOT / "include/metalrobo/NumiHumanMuscleEquilibrium.hpp"
APP = ROOT / "apps/numilab_human_myosim_visual_probe.mm"


def replace_exact(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    observed = text.count(old)
    if observed != count:
        raise RuntimeError(
            f"{path}: expected {count} exact occurrences, observed {observed}: {old[:120]!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def replace_regex(path: Path, pattern: str, replacement: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    updated, observed = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if observed != count:
        raise RuntimeError(
            f"{path}: expected {count} regex occurrences, observed {observed}: {pattern[:120]!r}"
        )
    path.write_text(updated, encoding="utf-8")


replace_exact(
    HEADER,
    """    std::vector<double> fiberLength;
    std::vector<double> muscleTendonForce;
    std::vector<double> passiveMuscleTendonForce;
    std::vector<double> generalizedMuscleForce;
""",
    """    std::vector<double> fiberLength;
    // Exact source actuator force before the Human passive-bias policy.
    std::vector<double> muscleTendonForce;
    // Exact zero-activation source force at the accepted path length.
    std::vector<double> passiveMuscleTendonForce;
    // Force actually registered in Human standing dynamics. Legacy MuJoCo
    // muscles exclude their zero-activation bias; compliant muscles retain
    // their full tendon force, matching mr_mujoco_muscle_active_force_rows.
    std::vector<double> drivenMuscleTendonForce;
    std::vector<double> generalizedMuscleForce;
""",
)

replace_exact(
    CPP,
    """constexpr double kMinimum = 1.0e-12;

struct PoseState {
""",
    """constexpr double kMinimum = 1.0e-12;

[[nodiscard]] double humanDrivenMuscleForce(
    const double sourceForce,
    const double zeroActivationForce,
    const MujocoCompliantMuscleArchitecture& architecture
) noexcept {
    const bool compliant = architecture.optimalFiberLength > 0.0 &&
        architecture.tendonSlackLength > 0.0;
    return compliant ? sourceForce : sourceForce - zeroActivationForce;
}

struct PoseState {
""",
)
replace_exact(
    CPP,
    """    std::vector<double> fiberLength;
    std::vector<double> muscleTendonForce;
    std::vector<double> passiveMuscleTendonForce;
    std::vector<double> muscleForce;
""",
    """    std::vector<double> fiberLength;
    std::vector<double> muscleTendonForce;
    std::vector<double> passiveMuscleTendonForce;
    std::vector<double> drivenMuscleTendonForce;
    std::vector<double> muscleForce;
""",
)

replace_exact(
    CPP,
    """    state.fiberLength.assign(muscles.size(), 0.0);
    state.muscleTendonForce.assign(muscles.size(), 0.0);
    state.passiveMuscleTendonForce.assign(muscles.size(), 0.0);
    state.muscleForce.assign(nv, 0.0);
""",
    """    state.fiberLength.assign(muscles.size(), 0.0);
    state.muscleTendonForce.assign(muscles.size(), 0.0);
    state.passiveMuscleTendonForce.assign(muscles.size(), 0.0);
    state.drivenMuscleTendonForce.assign(muscles.size(), 0.0);
    state.muscleForce.assign(nv, 0.0);
""",
)

replace_exact(
    CPP,
    """        const double passive = forceSamples[muscle * sampleCount];
        state.passiveMuscleTendonForce[muscle] = passive;
        double initialForce = passive;
        if (initializeFromAcceptedState) {
            diagnostics = evaluateStaticForce(
                resolved[muscle].pathLength, state.activation[muscle],
                config.timestep, muscles[muscle], architectures[muscle],
                initialForce, state.fiberLength[muscle],
                static_cast<std::uint32_t>(muscle)
            );
            if (!diagnostics.succeeded()) return diagnostics;
        }
        optimizerForce[muscle] = initialForce;
        state.muscleTendonForce[muscle] = initialForce;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.muscleForce[dof] +=
                initialForce * resolved[muscle].jacobian[dof];
            objectiveMuscleAcceleration[dof] +=
                initialForce * objectiveJacobians[muscle][dof];
        }
""",
    """        const double passive = forceSamples[muscle * sampleCount];
        state.passiveMuscleTendonForce[muscle] = passive;
        for (std::uint32_t sample = 0u; sample < sampleCount; ++sample) {
            const std::size_t index = muscle * sampleCount + sample;
            forceSamples[index] = humanDrivenMuscleForce(
                forceSamples[index], passive, architectures[muscle]
            );
        }
        double sourceForce = passive;
        if (initializeFromAcceptedState) {
            diagnostics = evaluateStaticForce(
                resolved[muscle].pathLength, state.activation[muscle],
                config.timestep, muscles[muscle], architectures[muscle],
                sourceForce, state.fiberLength[muscle],
                static_cast<std::uint32_t>(muscle)
            );
            if (!diagnostics.succeeded()) return diagnostics;
        }
        const double drivenForce = humanDrivenMuscleForce(
            sourceForce, passive, architectures[muscle]
        );
        optimizerForce[muscle] = drivenForce;
        state.muscleTendonForce[muscle] = sourceForce;
        state.drivenMuscleTendonForce[muscle] = drivenForce;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.muscleForce[dof] +=
                drivenForce * resolved[muscle].jacobian[dof];
            objectiveMuscleAcceleration[dof] +=
                drivenForce * objectiveJacobians[muscle][dof];
        }
""",
)

replace_exact(
    CPP,
    """        candidate.target = state.target;
        candidate.passiveCoordinateForce = state.passiveCoordinateForce;
        candidate.weights = state.weights;
""",
    """        candidate.target = state.target;
        candidate.passiveCoordinateForce = state.passiveCoordinateForce;
        candidate.passiveMuscleTendonForce =
            state.passiveMuscleTendonForce;
        candidate.weights = state.weights;
""",
)
replace_exact(
    CPP,
    """        candidate.fiberLength.assign(muscles.size(), 0.0);
        candidate.muscleForce.assign(nv, 0.0);
        std::vector<double> exactForce(muscles.size(), 0.0);
        for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
            auto exactDiagnostics = evaluateStaticForce(
                resolved[muscle].pathLength, candidate.activation[muscle],
                config.timestep, muscles[muscle], architectures[muscle],
                exactForce[muscle], candidate.fiberLength[muscle],
                static_cast<std::uint32_t>(muscle));
            if (!exactDiagnostics.succeeded()) return exactDiagnostics;
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                candidate.muscleForce[dof] +=
                    exactForce[muscle] * resolved[muscle].jacobian[dof];
            }
        }
""",
    """        candidate.fiberLength.assign(muscles.size(), 0.0);
        candidate.muscleTendonForce.assign(muscles.size(), 0.0);
        candidate.drivenMuscleTendonForce.assign(muscles.size(), 0.0);
        candidate.muscleForce.assign(nv, 0.0);
        std::vector<double> exactDrivenForce(muscles.size(), 0.0);
        for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
            double sourceForce = 0.0;
            auto exactDiagnostics = evaluateStaticForce(
                resolved[muscle].pathLength, candidate.activation[muscle],
                config.timestep, muscles[muscle], architectures[muscle],
                sourceForce, candidate.fiberLength[muscle],
                static_cast<std::uint32_t>(muscle));
            if (!exactDiagnostics.succeeded()) return exactDiagnostics;
            const double drivenForce = humanDrivenMuscleForce(
                sourceForce, candidate.passiveMuscleTendonForce[muscle],
                architectures[muscle]
            );
            candidate.muscleTendonForce[muscle] = sourceForce;
            candidate.drivenMuscleTendonForce[muscle] = drivenForce;
            exactDrivenForce[muscle] = drivenForce;
            for (std::size_t dof = 0u; dof < nv; ++dof) {
                candidate.muscleForce[dof] +=
                    drivenForce * resolved[muscle].jacobian[dof];
            }
        }
""",
)
replace_exact(
    CPP,
    """        optimizerForce = std::move(exactForce);
""",
    """        optimizerForce = std::move(exactDrivenForce);
        state.muscleTendonForce = candidate.muscleTendonForce;
        state.drivenMuscleTendonForce =
            candidate.drivenMuscleTendonForce;
""",
)

replace_exact(
    CPP,
    """            candidate.fiberLength.assign(muscles.size(), 0.0);
            candidate.muscleTendonForce.assign(muscles.size(), 0.0);
            candidate.muscleForce.assign(nv, 0.0);
            for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
                double force = 0.0;
                diagnostics = evaluateStaticForce(
                    resolved[muscle].pathLength,
                    candidate.activation[muscle], config.timestep,
                    muscles[muscle], architectures[muscle], force,
                    candidate.fiberLength[muscle],
                    static_cast<std::uint32_t>(muscle));
                if (!diagnostics.succeeded()) return diagnostics;
                candidate.muscleTendonForce[muscle] = force;
                for (std::size_t dof = 0u; dof < nv; ++dof) {
                    candidate.muscleForce[dof] +=
                        force * resolved[muscle].jacobian[dof];
                }
            }
""",
    """            candidate.fiberLength.assign(muscles.size(), 0.0);
            candidate.muscleTendonForce.assign(muscles.size(), 0.0);
            candidate.drivenMuscleTendonForce.assign(muscles.size(), 0.0);
            candidate.muscleForce.assign(nv, 0.0);
            for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
                double sourceForce = 0.0;
                diagnostics = evaluateStaticForce(
                    resolved[muscle].pathLength,
                    candidate.activation[muscle], config.timestep,
                    muscles[muscle], architectures[muscle], sourceForce,
                    candidate.fiberLength[muscle],
                    static_cast<std::uint32_t>(muscle));
                if (!diagnostics.succeeded()) return diagnostics;
                const double drivenForce = humanDrivenMuscleForce(
                    sourceForce,
                    candidate.passiveMuscleTendonForce[muscle],
                    architectures[muscle]
                );
                candidate.muscleTendonForce[muscle] = sourceForce;
                candidate.drivenMuscleTendonForce[muscle] = drivenForce;
                for (std::size_t dof = 0u; dof < nv; ++dof) {
                    candidate.muscleForce[dof] +=
                        drivenForce * resolved[muscle].jacobian[dof];
                }
            }
""",
)

replace_exact(
    CPP,
    """    state.fiberLength.assign(muscles.size(), 0.0);
    state.muscleTendonForce.assign(muscles.size(), 0.0);
    state.passiveMuscleTendonForce.assign(muscles.size(), 0.0);
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        double force = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, state.activation[muscle],
            config.timestep, muscles[muscle], architectures[muscle], force,
            state.fiberLength[muscle], static_cast<std::uint32_t>(muscle)
        );
        if (!diagnostics.succeeded()) return diagnostics;
        state.muscleTendonForce[muscle] = force;
        double passiveForce = 0.0;
        double passiveFiberLength = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, 0.0, config.timestep,
            muscles[muscle], architectures[muscle], passiveForce,
            passiveFiberLength, static_cast<std::uint32_t>(muscle));
        if (!diagnostics.succeeded()) return diagnostics;
        state.passiveMuscleTendonForce[muscle] = passiveForce;
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            state.muscleForce[dof] +=
                force * resolved[muscle].jacobian[dof];
        }
    }
""",
    """    state.fiberLength.assign(muscles.size(), 0.0);
    state.muscleTendonForce.assign(muscles.size(), 0.0);
    state.passiveMuscleTendonForce.assign(muscles.size(), 0.0);
    state.drivenMuscleTendonForce.assign(muscles.size(), 0.0);
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        double sourceForce = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, state.activation[muscle],
            config.timestep, muscles[muscle], architectures[muscle],
            sourceForce, state.fiberLength[muscle],
            static_cast<std::uint32_t>(muscle)
        );
        if (!diagnostics.succeeded()) return diagnostics;
        double passiveForce = 0.0;
        double passiveFiberLength = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, 0.0, config.timestep,
            muscles[muscle], architectures[muscle], passiveForce,
            passiveFiberLength, static_cast<std::uint32_t>(muscle));
        if (!diagnostics.succeeded()) return diagnostics;
        const double drivenForce = humanDrivenMuscleForce(
            sourceForce, passiveForce, architectures[muscle]
        );
        state.muscleTendonForce[muscle] = sourceForce;
        state.passiveMuscleTendonForce[muscle] = passiveForce;
        state.drivenMuscleTendonForce[muscle] = drivenForce;
        for (std::size_t dof = 0u; dof < articulation.nv; ++dof) {
            state.muscleForce[dof] +=
                drivenForce * resolved[muscle].jacobian[dof];
        }
    }
""",
)

replace_exact(
    CPP,
    """    candidate.muscleTendonForce = current.muscleTendonForce;
    candidate.passiveMuscleTendonForce = current.passiveMuscleTendonForce;
    candidate.generalizedMuscleForce = current.muscleForce;
""",
    """    candidate.muscleTendonForce = current.muscleTendonForce;
    candidate.passiveMuscleTendonForce = current.passiveMuscleTendonForce;
    candidate.drivenMuscleTendonForce =
        current.drivenMuscleTendonForce;
    candidate.generalizedMuscleForce = current.muscleForce;
""",
)

replace_exact(
    APP,
    """                writeReactionVector(\"actuator_force_n\", support.muscleTendonForce);
                writeReactionVector(\"passive_actuator_force_n\", support.passiveMuscleTendonForce);
""",
    """                writeReactionVector(\"actuator_force_n\", support.muscleTendonForce);
                writeReactionVector(\"passive_actuator_force_n\", support.passiveMuscleTendonForce);
                writeReactionVector(\"driven_actuator_force_n\", support.drivenMuscleTendonForce);
""",
    count=1,
)
replace_exact(
    APP,
    """    field(\"actuator_force_n\",s.muscleTendonForce);std::cout<<\"}\\
\";
""",
    """    field(\"actuator_force_n\",s.muscleTendonForce);field(\"passive_actuator_force_n\",s.passiveMuscleTendonForce);
    field(\"driven_actuator_force_n\",s.drivenMuscleTendonForce);std::cout<<\"}\\
\";
""",
    count=1,
)

for path in (HEADER, CPP, APP):
    text = path.read_text(encoding="utf-8")
    if "drivenMuscleTendonForce" not in text:
        raise RuntimeError(f"{path}: driven Human muscle force was not installed")

print("applied Human static/dynamic active-force parity correction")
