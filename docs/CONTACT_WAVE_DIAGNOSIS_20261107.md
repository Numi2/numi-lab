# Contact-free acoustic wave diagnostic

This test asks whether the volume mesh alone disperses waves at spatial
frequencies relevant to impact. It does not measure impact contact force and
cannot qualify the fixed-law force or impulse gates, which remain 0.25%.

## Prediction registered before measurements

For the unchanged 20 mm, 1100 kg/m³ Neo-Hookean block with mu=2000 Pa and
lambda=5000 Pa, constrain lateral nodal positions, cancel gravity, and leave
both normal end faces traction-free. The plane is inactive. The independent
linear continuum solution is u=A cos(m pi z/L) cos(c m pi t/L), with
c=sqrt((lambda+2 mu)/rho), A=10 nm. For mode 8, compare n=8 and n=16 with
mode 2 on the same mesh: the high mode should show greater normalized
space-time error; increasing spatial resolution should reduce it. This
prediction attributes measurable wave dispersion to the bulk discretization;
it does not predict what fraction of the impact-force error it causes.

Run n=8,16, modes 2 and 8, RK4 h=2 and 1 microseconds, 10 ms duration,
sampling at 0.1 ms. Compare the two temporal traces and require their maximum
difference in modal projection <=0.25% of initial amplitude before interpreting
the fine trace spatially. Run the n=16 mode-8 case also at A=5 nm at h=1 µs;
the maximum difference between amplitude-normalized modal projections must
be <=0.25% to control the nonlinear/reference approximation. Preserve any
failed checks and do not infer qualification from them. A per-run 180-second
timeout is the predeclared budget. Record full traces and inputs.

The continuum formula is exact for the *linearized* constitutive law. For
F=diag(1,1,1+s), Pzz=mu(1+s)+(lambda log(1+s)-mu)/(1+s) and Pzx=Pzy=0.
The endpoint derivative vanishes, so linearized endpoint traction is zero;
lateral displacement constraints supply lateral reactions. Initial strain
amplitude A*m*pi/L is at most 1.257e-5. The source reports the maximum actual
sampled tetrahedral strain and the pointwise departure of exact nonlinear
Pzz from its linearized value, normalized by the continuum stress scale.
Together with the amplitude-halving run, these bound/check the linearization
for this experiment, but the halving check is empirical, not a rigorous bound
on the continuum solution error. Contact-free status is checked at every RK4
stage, and material and mass assembly use the independent FEM source. This is
a constrained wave diagnostic, not the original free-sided impact geometry.

## Retained result

All nine runs completed on the CPU, with no timeout or GPU work. See
[evidence](evidence/contact-wave-20261107/manifest.json) for the exact
prediction commit, arguments, elapsed time, and source and binary hashes;
CSV traces and the analysis script are in the same directory.

| Mesh | Half-wave mode | Peak normalized displacement L2 error | First modal zero delay |
|---|---:|---:|---:|
| 8 | 2 | 0.23127 | 2.69% |
| 16 | 2 | 0.05610 | 0.65% |
| 8 | 8 | 1.94367 | 57.14% |
| 16 | 8 | 1.96738 | 11.06% |

Peak L2 is divided by the initial mass-weighted modal RMS amplitude, and
is the maximum across all 0.1 ms samples over 10 ms. The registered high-mode
versus low-mode prediction passed, but **the registered peak-error reduction
from n=8 to n=16 failed**. Its near-two maximum can saturate when numerical
and reference waves become nearly opposite in phase; this is an interpretation,
not proof of a specific impact error. First-zero delays are additional,
post hoc phase measurements. Their linear interpolation on the 0.1 ms sample
grid was also applied to the analytic trace: mode 8's interpolated analytic
zero is 0.437135 ms versus its exact 0.437004 ms, a 0.030% timing bias.

The four maximum 2 µs versus 1 µs modal-projection differences range from
3.87e-10 to 6.97e-10 of initial amplitude; the n=16 mode-8 amplitude-halving
difference is 6.68e-10. All meet the registered 0.25% diagnostic thresholds.
These tests concern the sampled **modal projection**, not a certified bound on
every node or on the nonlinear continuum solution. The largest sampled
tetrahedral strain is 1.293e-5; the largest sampled local nonlinear stress
departure from the linear stress, normalized by the initial continuum stress
scale, is 1.404e-5. Plane activation was checked at every RK4 stage; the
smallest sampled bottom gap was 199.982 µm, well above the 100 µm activation
distance. The 5 nm run provides an independent amplitude-scaling check but
does not constitute a mathematical error bound for the nonlinear continuum.

The constrained block demonstrates that high-frequency bulk wave dispersion
can be substantial even when contact quadrature is absent. Its boundaries,
loading and tracked displacement differ from the impact fixture, so this
cannot quantify the contribution of bulk waves to the retained 12.43% impact
force-bin discrepancy or establish a force-qualified control. The original
0.25% force-bin and impulse gates remain unmet.
