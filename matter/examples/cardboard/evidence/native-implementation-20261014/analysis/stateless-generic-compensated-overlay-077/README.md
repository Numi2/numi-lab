# Opt-in compensated stateless generic stored-energy overlay (077)

This immutable overlay is based on frozen source 076:
 /Users/home/cardboard-implementation-20261014/snapshots/source-stable-free-graph-and-general-arithmetic-076

It adds one narrowly admitted compiler case for paperCompensatedArithmetic on a generic, stateless stored-energy material. The material must have no internal state, explicit or implicit state updates, dissipation, learned law, Hill policy, or rate/time/temperature/history dependency in its energy or validity graph. Cooked-layout validation independently checks descriptor shape and rejects rate, time, temperature, and state opcodes in stress/validity programs; only tangent programs may contain the deformation-direction opcode. The compiler’s existing fiber-specific exponential bytecode rejection remains active.

The numerical test uses the retained nagasawa2013_medium_orthotropic.nmatter asset. Its CPU reference independently evaluates the authored St. Venant–Kirchhoff orthotropic stress and directional tangent. At a near-rest deformation with low-word components, ordinary FP32 stress relative error is 0.925984; the existing ordered two-word evaluator gives stress error 1.18691e-7 and tangent error 8.41643e-15. The compiled FP64 bytecode stress/tangent are checked against the independent source equations, and the independent tangent is checked by finite differences. This is a CPU/compiler check only; no Metal execution or physical qualification is claimed.

## Reproduction

Apply patch.diff with patch -p1 to a copy of frozen source 076. The patch changes only:
- matter/src/constitutive_impl.cpp
- matter/src/validation_impl.cpp
- matter/tools/cardboard_generic_compensated_projection_check.cpp

run_cpu_overlay.sh reproduces the compile/link/run check using the frozen-076 compiler archive bound in manifest.json. It also runs the enhanced test against the unmodified frozen archive as a negative control; the expected result is rejection at the original stateless-generic policy gate. evidence/patch-check.log records successful dry-run/application and exact output hashes. No CMake, solver/runtime, Metal kernel, or physical material equation was changed.

## Prior stable-prefix manifest status

The first-ready stable-prefix manifest was cited with SHA-256 prefix 2e873611…, but its original bytes are not present in the retained overlay directory or source-076 snapshot. The only retained file is the later syntax-check revision, SHA-256 f33e175531afb233de918e71e9ee359a922172b9a1852a325b3f1c52b4046eaa, which source 076 itself binds. I did not represent that later file as the original. See historical-first-ready-manifest-recovery-status.json.
