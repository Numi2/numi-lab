# Strict-math varied-state control

The fast-math varied-state screen failed the 5e-5 gate in both the existing and two-segment models. Re-run the exact same 96 inputs, independent CPU oracle, deterministic-repeat requirement and error threshold with the existing isolated strict-math metallib, separately for both bodies. This is a paired compiler-mode diagnostic; do not promote either capacity or math mode based on a pass in isolation.
