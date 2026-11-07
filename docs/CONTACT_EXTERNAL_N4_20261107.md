# External n=4 P1 comparison

The n=2 scikit-fem and DOP853 result independently matched the C++ RK4
trajectory but is not spatially accurate enough. The next bounded test
constructs n=4 with the same six-tetrahedron cell pattern and the same
physical and nodal-contact parameters, using the separately assembled
external model. It is an independent assembly and time integration of the
same spatial family, not an independent high-order spatial reference.

## Prediction registered before execution

Run 10 ms at the same DOP853 coarse and fine settings as n=2: max step
8 and 4 µs, rtol 1e-8 and 1e-10, and position atol 1e-12 and 1e-13 m.
Compare 1 ms force bins and integrated impulse against each other and the
retained n=4 RK4 h=0.5 µs outputs. All differences must meet the unchanged
0.25% gates before interpreting spatial effects. Require positive sampled
clearance and momentum balance <=0.25% of impulse. Then compare the fine
n=2 and n=4 external force bins and impulse using the n=4 peak bin and
impulse, retaining the 0.25% spatial gates. Based on retained C++ mesh
results, expect that spatial check to fail; a different outcome warrants
investigation, not gate relaxation.

Nominal cost extrapolated from n=2 (13.8/19.5 s) and 8× elements is roughly
two to four minutes per run, with uncertainty from vectorization, adaptive
steps and memory. Set a 300-second timeout per process, execute sequentially,
and retain a timeout or numerical failure without retrying. Stop on execution
failure. Single-thread BLAS; no GPU. If either prerequisite fails, do not
interpret the spatial comparison. Even a passing pair would not prove
convergence of n=4 or a force-qualified production fixture.

## Result

The coarse and fine n=4 runs completed in 23.1 s and 46.6 s. Their maximum
force-bin and total-impulse differences were 3.36e-7 and 1.14e-7; fine
external versus retained n=4 RK4 differences were 4.04e-8 and 1.47e-8.
All meet the unchanged 0.25% gates. Minimum sampled gap was 90.024 µm;
all evaluated stages also reject nonpositive gaps. The maximum momentum
imbalance was 2.96e-15 of total impulse.

The fine external n=2 versus n=4 comparison **failed** the spatial gates:
92.90% for binned force and 20.01% for integrated impulse. The outcome
matches the declared prediction, but does not establish the n=16→32
discrepancy's cause or a converged spatial reference. The noncontact and
uniform-plane analytic n=4 controls, inputs and traces are in the
[evidence directory](evidence/contact-external-n4-20261107/).
No GPU or material validation was involved.
