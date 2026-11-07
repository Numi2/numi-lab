# External n=8 P1 comparison

The external n=4 P1/DOP853 trajectory passed its temporal and retained C++
RK4 comparisons; n=2→4 failed the 0.25% force and impulse spatial gates.
This next test builds n=8 with exactly the same geometry, six-tetrahedron
cell pattern, gravity, initial state, material and nodal plane law. It uses
the same external assembly and time integrator; same-family refinement does
not itself establish a converged independent spatial reference.

## Prediction registered before execution

Run the 10 ms n=8 trajectory at the n=4 coarse and fine DOP853 settings:
maximum steps 8 and 4 µs, rtol 1e-8 and 1e-10, position atol 1e-12 and
1e-13 m, velocity atol 1000 times greater. Compare the resulting 1 ms
force bins and total impulses against each other, and fine against retained
n=8 C++ RK4 h=0.5 µs. Require all differences <=0.25% using the fine n=8
peak bin and total impulse; sampled gap must stay positive and momentum
imbalance <=0.25% of total impulse. Then compare external fine n=4 and n=8
with the unchanged same normalizers and 0.25% spatial gates. The prior C++
result predicts spatial failure: 70.43% force and 3.374% impulse differences.
Do not interpret the spatial comparison unless the numerical prerequisite
gates pass.

n=8 has 3,072 tetrahedra and 729 nodes, eight times n=4 elements. Prior n=4
coarse/fine costs were 23.1/46.6 s; a conservative linear extrapolation is
about 185/373 s, but vectorized memory and adaptive behavior make the estimate
uncertain. Set sequential timeouts of 300 and 600 s, retaining any failure
and stopping; no retry without another plan. Both runs use one BLAS thread and
no GPU. A pass would compare only n=4 to n=8 and cannot qualify n=8 as a
converged reference or the production contact fixture.

## Result

The coarse and fine runs completed in 148.3 s and 281.5 s. Coarse versus
fine force-bin and total-impulse differences were 2.46e-7 and 5.62e-8;
fine external versus retained n=8 RK4 differences were 4.46e-8 and
1.42e-8. All pass the unchanged 0.25% thresholds. Minimum sampled gap was
91.859 µm, evaluated stages reject nonpositive gaps, and maximum momentum
imbalance was 3.11e-15 of total impulse.

External n=4→8 force and impulse differences were **70.43% and 3.374%**
of the fine n=8 peak force bin and total impulse. Both spatial gates fail,
as predicted. This verifies agreement of two independently assembled and
integrated P1 implementations at n=8; it does not establish spatial
convergence, physical accuracy or production qualification. Fine
contact work was -1.43434e-7 J and is a diagnostic, not an energy verdict.

The [evidence](evidence/contact-external-n8-20261107/) includes inputs and
code hashes, analytic controls, raw traces, timeouts and exact analysis.
No GPU was used.
