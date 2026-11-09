# Force and impulse landmarks in the bound spatial references

The independent CPU references at 64 and 128 divisions each pass their own
timestep refinement. The 64-division fine run is the hash-bound repeat; the
128-division input is checked against its retained analysis. Reproduce the
machine-readable receipt with
`python3 tests/contact/audit_bound_spatial_landmarks.py`. Neither result is a
native Matter rollout or a physical material calibration.

The integration-step-sampled peak is 0.1226473 N at 64 divisions and
0.1016194 N at 128 divisions, a 20.69% difference when normalized by the
128-division peak. The separate 2 microsecond trace reaches its largest sample
at 9.858 ms and 9.928 ms, respectively; those times are not claimed to be the
times of the integration-step maxima. The largest difference between
corresponding 2 microsecond force samples is 66.51% of the 128-division peak.
That pointwise measure includes timing and oscillation differences and is not
an error estimate for either mesh.

The largest 1 ms-bin difference occurs at 4–5 ms: 0.0238617 N at 64
versus 0.0254208 N at 128. Normalized by the largest 128-division bin over the
whole trace, the maximum bin difference is 1.727%. The largest cumulative
impulse difference occurs at 5.078 ms, with 3.14695e-5 N s at 64 and
2.94009e-5 N s at 128; normalized by final 128-division impulse it is
0.5037%. The final impulse differs by 0.1618%. These distinct quantities do
not establish spatial convergence or an integrator speedup.
