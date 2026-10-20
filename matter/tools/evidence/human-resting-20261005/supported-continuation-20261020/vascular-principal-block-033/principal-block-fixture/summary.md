# Principal block algebra control

Production kernels passed the46-row/2-environment reference-topology fixture
with one explicitly synthetic shell-cavity overlay. Blood species, tissue and
exchange transport are omitted from this algebra fixture, not from the owning
physiology implementation. FP64 LU uses columns extracted through the existing
production JVP with zero FEM direction; this independently checks factor/apply
algebra, while the existing full cavity fixture checks the physical JVP.

Maximum relative solution error3.6285023883549796e-7; maximum relative principal
residual1.0072837791342655e-6. Nonzero vascularBase135, one closed-valve active row,
two RHS per factor, retained nonvascular prefix/suffix, factor reuse, unchanged
input/state and invalid-factor diagonal-output preservation passed.
The invalid-factor control sets valid=0; it does not prove every possible
singular/nonfinite input. This is numerical interface evidence, with no
biological calibration or whole-body completion claim.
