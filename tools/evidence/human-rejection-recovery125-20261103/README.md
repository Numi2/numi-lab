# Native Human rejection recovery

The full scene remains unqualified: this increment fixes contact-energy precision, not full-body convergence.

The historical serial-contact full-body trace identifies positive-gap contact Armijo bounds shrinking from 1/512 to 1.455e-11. Exact per-alpha replay of contact3390947 shows collision-certified trials rejected for a one-Float32-ULP energy increase while the predicted derivative is negative. Preserve this failed trial.

The reference-contact limiter now retains both words of angular mollifier and IPC energy through its Armijo comparison. The condition, authored clearance, residual tolerances, materials, mass, contact topology, and accepted-state transaction remain intact. Non-reference contacts retain their established comparison.

The new Metal checker tests22 actual current/trial states from Newton8..12, checks the Float64 equation and resolved difference signs, and protects4 output canaries. Maximum observed relative energy error3.47611e-13. Full GPU kernel-contract suite:80880 checks passed; this is not full Human motion qualification.

Current native full-body confirmation is still pending at this commit. Its exact plan and hashes are retained on SSH Numi under rejection-recovery-125/paired-energy.
