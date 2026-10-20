# 042d synthetic-probe configuration

042c passes all133 contact geometry rows, exact-VT-JVP controls, floor controls and8/256 accepted production integration tests. Pair-local coordinate/gap and accumulation error are zero; original energy-delta relative error improves from0.32851269 to0.11432487, within the unchanged0.25 gate.

The full FEM-reference check then stops with: vascular principal block requires1 through64 vascular unknowns. It inherited the Human's opt-in vascular block environment, but this synthetic reference model has zero vascular unknowns. Explicitly set that environment selector to0 only for the nonvascular FEM reference, kinematic-release and mutable-snapshot probes in the entrypoint. Keep the Human integrated scene and vascular probe at their declared selector1. No production source or Human physical configuration changes. Retain042c as a failed preparation attempt and rerun preparation in042d; the same continuation stopping rule remains.
