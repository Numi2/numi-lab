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
