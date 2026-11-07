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
