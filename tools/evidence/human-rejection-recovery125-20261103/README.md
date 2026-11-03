# Native Human rejection recovery

The full scene remains unqualified: this increment fixes contact-energy precision, not full-body convergence.

The historical serial-contact full-body trace identifies positive-gap contact Armijo bounds shrinking from 1/512 to 1.455e-11. Exact per-alpha replay of contact3390947 shows collision-certified trials rejected for a one-Float32-ULP energy increase while the predicted derivative is negative. Preserve this failed trial.

The reference-contact limiter now retains both words of angular mollifier and IPC energy through its Armijo comparison. The condition, authored clearance, residual tolerances, materials, mass, contact topology, and accepted-state transaction remain intact. Non-reference contacts retain their established comparison.

The new Metal checker tests22 actual current/trial states from Newton8..12, checks the Float64 equation and resolved difference signs, and protects4 output canaries. Maximum observed relative energy error3.47611e-13. Full GPU kernel-contract suite:80880 checks passed; this is not full Human motion qualification.

Current native full-body confirmation is still pending at this commit. Its exact plan and hashes are retained on SSH Numi under rejection-recovery-125/paired-energy.

## Native confirmation and angular-metric experiment

Paired-energy full Human confirmation still rejected:14 active Newton updates,28 columns; norm0.709674001, contact2726083. The spurious1e-11 alpha collapse is removed but physical acceptance is not achieved.

The optional NM_FEM_RESOLVED_MOLLIFIER_CURVATURE=1 metric retains the existing positive gradient-outer-product coefficient B/m instead of suppressing it below m=1e-3. It changes the quasi-Newton metric only; energy, contact forces, topology, material/mass and acceptance rules remain unchanged. Admission requires one FEM-only contact environment. Exactly-zero-gradient behavior and m>=1e-3 cases retain their previous arithmetic.

Native metric tests:100 parameter cases against Float64, finite PSD checks, unchanged cases,4 angular corrections that decrease the unchanged nonlinear potential,4 output canaries. Complete GPU suite:81055 checks passed.

Full Human result with this option: initial alpha doubles from1/8192 to1/4096, but14 Newton updates/28 columns still reject contact2726083; minimum norm0.709653914. This does not establish a meaningful whole-body convergence gain.

The existing EW tightening exponent4 avoids that local contact failure and completes all16 Newton updates (92 columns), but final vascular row0 certification rejects residual1.2229686e-5 against1e-5; mechanical norm remains0.70825386 and is not converged. No iteration budget or physical tolerance was raised. All three trials accepted zero physical steps. Their failed receipts and exact plans are retained.

The option remains experimental and defaults off. No real-time factor or full motion claim is supported.
