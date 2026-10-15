# Interpretation note

The vectors in this package are Cauchy directions for the *post-Newton linear remainder* `r = b - A*delta_v`. They are residual-refinement increments relative to the captured Newton correction, not fallback directions from the current Newton center. For a fallback evaluated from the current center, the residual is `b` and the gradient is `A^T b`; that separate base-state experiment is in `../captured-base-state-cauchy-419-001/`.

The original report and vectors remain unchanged as a distinct calculation. This note corrects no numerical output and does not claim native nonlinear descent.
