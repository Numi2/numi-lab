# GPU vascular principal block 033

Problem: native030 passes the repaired FEM limiter, accepts23 steps and then
rejects root23 on the vascular line search with the existing64-column budget.
The prior trace identified an exhausted coupled Krylov direction. Raising
iteration limits or relaxing tissue/vascular tolerances is not this trial.

Opt-in NUMI_MATTER_VASCULAR_BLOCK_PRECONDITIONER=1 factors the exact normalized
vascular principal Jacobian, with FEM geometry frozen only for its vascular
unit columns. It reuses GPU LU once per Newton candidate/valve working set
through all Krylov restarts. The full coupled operator and all physical
residuals, wall forces, line searches and acceptance certificates are retained.
The default rowwise preconditioner remains unchanged; singular/nonfinite
factors or solves retain that existing output. The policy changes execution
identity, not anatomy, material parameters, physical state or mass ownership.

Before full-scene enablement: compare the production factor/apply against an
independent FP64 solve, with constrained valve rows, two right-hand sides,
a nonzero vascular vector base and fallback controls. Run the existing
preparation, geometry, force/interface, FEM limiter and rejected-step checks.
Recheck the default-policy first two accepted CSV rows against native030.

Integrated trial: same native030 assets, pose, initialization and125us step,
32 requested steps, FGMRES64, no parallel GPU job. Success requires exit0 and
all32 accepted steps with existing certificates and no root assistance.
A rejected trial retains its exact failure witness and accepted prefix.
Even success is4ms of numerical integration, not sustained support, full
ligament coverage, physiological qualification or clinical validation.
No visual verification is requested or performed.
