# Native497 warm-CGLS action audit addendum

This addendum corrects the interpretation of the unchanged native497-analytic-warm-cgls-action-003 payload. Source185 confirms captured Ax is the final fresh action on the total candidate X, not the last Arnoldi/CG action workspace. It also binds x0 to correction_delta_v_m_s (the current Newton femSolution), not cumulative delta_velocity_m_s.

The strict material/tet/frame/nodal action results are retained verbatim from 003. Against the same-generation captured FP32-cache CSR, strict mass is exact and the cached tangent assembly agrees with stiffness at 1.19e-12 relative. The strict FP64 analytic tangent differs from that cached tangent by 2.916e-5 globally at stiffness and the total assembled action differs by 2.985e-5 globally (max nodal row relative 4.033e-4). Center stress vs captured native readback differs 2.39e-8 globally. These are CPU reconstruction results, not a native action replay.

The sidecar identity A*x = A*x0 + B - R misses by 3.77e-9 Ns absolute (1.999e-4 relative to captured native A*x). Same-generation CSR checks show A*(X-x0) differs from B-R by 3.625e-9 Ns, 1.743e-2 relative to that small residual. Because the capture did not retain the native A*x0 vector, this cannot isolate native FP32 action error from CGLS recurrence/update/reduction drift. It remains an unresolved solver-side discrepancy.

Run python3 seal_addendum.py with the source185 task venv and existing source/capture inputs available. The script verifies prior report identity and records the bound source lines and derived action checks; it does not alter the prior artifact.
