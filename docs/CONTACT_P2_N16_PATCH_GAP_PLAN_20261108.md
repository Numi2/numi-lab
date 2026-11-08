# Fixed-state patch gap diagnosis

Register before execution. At the retained 4.8 ms n=16 onset state, evaluate
the actual gap and the n16-to-n8-to-n16 transferred gap on the same n16
quadrature coordinates. The fixed regularized law is unchanged. Integrate
force on a 4×4 equal-area surface grid; report each patch's signed force
difference and mean, RMS and maximum absolute gap difference. This identifies
where the sampled force difference is located and separates approximation of
the gap field from numerical surface integration at the same points.

Capture SHA-256 b2dd05ceee940b154d7038bf8684bde176d64a8b375958275c72a9d75d0aadc7;
the retained transfer and input hashes are also recorded by the runner.
Use both 1e-12 and 1e-13 m three-coordinate insets. At Duffy-48/64 require
positive sampled gaps, all 16 patch areas and their sum to agree with their
analytical values within 1e-10 relative, and constant-gap force to agree
with analytical area times pressure within 1e-10 relative. Actual and
transferred total Duffy-64 forces must reproduce the retained n16 and
n16-to-n8 forces within 1e-7 of the n8 full peak bin. Require maximum
per-patch 48/64 force differences <=0.05% of that peak and inset-force
differences <=1e-7 of it. Preserve all signed and absolute values and do
not interpret the patch map if any control fails.

Run single-thread CPU only, 600 s process-group timeout, 4 GiB sampled-RSS
stop, 1 Hz monitoring, with source and input identities and complete logs.
The previous transfer used 2,180,640 KiB sampled RSS and 5.08 s; estimate
under 30 s and about 2.5 GiB, with uncertain extra quadrature allocations.
Coordinate with the parent before execution. This sampled-state diagnostic
cannot establish integrated-force causation, continuous clearance, a 10 ms
spatial reference, or physical validity; 0.25% temporal and mesh gates stay
unchanged.

## Result

The registered process exited 0 in 2.04 s, with peak sampled RSS
1,647,952 KiB and no resource stop. All controls passed: maximum patch
area error 9.33e-14 relative, actual total-force reproduction 1.27e-15
and transferred total reproduction 7.39e-8 of the n=8 peak bin, maximum
per-patch Duffy difference 7.08e-13 and inset difference 3.85e-9 of that
peak. All evaluated gaps were positive.

At this snapshot the transferred gap was larger on average in every one
of the sixteen patches, by 0.438–0.701 µm, and all sixteen transferred
forces decreased. The actual total was 0.10649716 N and transferred total
0.07525209 N; the largest patch loss was 0.00233692 N. Signed patch forces,
gap means, RMS and maxima are retained in the analysis. These common-point
results identify a distributed fixed-state interpolation effect with
quadrature differences far below the control thresholds; they do not
establish an integrated-force cause, trajectory convergence or physical
qualification.
