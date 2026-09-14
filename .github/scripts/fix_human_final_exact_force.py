#!/usr/bin/env python3
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "src/core/NumiHumanMuscleEquilibrium.cpp"
text = path.read_text(encoding="utf-8")
old = """    std::fill(state.muscleForce.begin(), state.muscleForce.end(), 0.0);
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        double force = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, state.activation[muscle],
            config.timestep, muscles[muscle], architectures[muscle], force,
            state.fiberLength[muscle], static_cast<std::uint32_t>(muscle)
        );
        if (!diagnostics.succeeded()) return diagnostics;
        state.muscleTendonForce[muscle] = force;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.muscleForce[dof] +=
                force * resolved[muscle].jacobian[dof];
        }
    }
"""
new = """    std::fill(state.muscleForce.begin(), state.muscleForce.end(), 0.0);
    for (std::size_t muscle = 0u; muscle < muscles.size(); ++muscle) {
        double sourceForce = 0.0;
        diagnostics = evaluateStaticForce(
            resolved[muscle].pathLength, state.activation[muscle],
            config.timestep, muscles[muscle], architectures[muscle],
            sourceForce, state.fiberLength[muscle],
            static_cast<std::uint32_t>(muscle)
        );
        if (!diagnostics.succeeded()) return diagnostics;
        const double drivenForce = humanDrivenMuscleForce(
            sourceForce, state.passiveMuscleTendonForce[muscle],
            architectures[muscle]
        );
        state.muscleTendonForce[muscle] = sourceForce;
        state.drivenMuscleTendonForce[muscle] = drivenForce;
        for (std::size_t dof = 0u; dof < nv; ++dof) {
            state.muscleForce[dof] +=
                drivenForce * resolved[muscle].jacobian[dof];
        }
    }
"""
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected one final exact source-force block, observed {count}")
path.write_text(text.replace(old, new), encoding="utf-8")
print("corrected final exact Human generalized force publication")
