# Native497 nonlinear correction replay and captured linear action (CPU only)

This package binds one exact root20/Newton63 state on macmini. It reconstructs the CGLS v3 candidate action from source-defined scratch segments and compares the linearized residual with strict compensated-FP64 material projections and assembled force changes over a small-alpha grid. It also collects separate CPU nonlinear curves for warm CGLS, exact LU, and the captured original FGMRES correction. No vectors are added across these candidates.

The native center residual/stress and CGLS captures are exact inputs. Trial materials, stresses, and residuals are CPU evaluations, not native trial execution. They do not qualify an accepted native trajectory or full coupon. Strict material gates and branch outcomes are recorded per point.
