# First rejected paired-contact tangent row

**The dominant rejected-state tangent residual belongs to a rod–sphere contact whose impulse is inside its friction limit while its captured tangential relative velocity is nonzero.** This identifies where the solver certificate fails; it does not establish the cause of the stall or validate the contact law.

The preregistered diagnostic ran on Apple M6 over ssh numi using the retained isolated paired-normal candidate. The unchanged and instrumented shader each produced 12 accepted and eight rejected steps. Their entire exposed traces match each other and the previously retained candidate after excluding the overwritten scratch fields; capture-free versus captured exposed state and repeated traces also match. The first rejection is step 12, status 11. These are observations of the rejected candidate state, not published physical evolution; hidden caches and full owner restoration remain unqualified.

| Row | Contact | Normal impulse | Friction utilization | Tangential velocity | Projected tangent residual |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | sphere–plane | 0.00951754 N s | 0.000616 | 0.000000343 m/s | 0.000000343 m/s |
| 1 | rod–sphere | 0.00000809785 N s | 0.870874 | -0.0160683 m/s | 0.00260291 m/s |
| 2 | rod–sphere | 0.00000105082 N s | 1.000000 | +0.0127832 m/s | approximately 0 m/s |

Utilization is the tangent impulse magnitude divided by the local friction limit, here with equal friction coefficients of 0.5. The second rod row is at its friction boundary and its projected tangent residual is near zero. The first is inside the boundary and has the largest residual. Both contacts use the captured instantaneous solver variables; these values do not show whether the iteration oscillates, suffers poor conditioning, or is affected by simultaneous normal updates.

An independent CPU evaluation of the projected residual from the captured tangent impulse, preconditioner, right-hand side and friction limits gives 0.002602912781 m/s, agreeing with the reported maximum 0.002602912718 m/s within 6.4e-11 m/s. This independently checks the projection arithmetic using captured inputs; it does not independently reconstruct the rod velocity or certify the underlying operator. The retained research stopping criterion has known small-impulse limitations.

The next discriminating test should capture the evolution of both rod rows over the final sweeps and independently evaluate the coupled normal–tangent update. Predeclare its controls and preserve this failure. Do not promote or retune the candidate, conflate rejected candidate values with accepted contact forces, or claim physical force accuracy from these captures. Thin-impact CCD completion, the FEM–plane force reference, full rollback, and other programme milestones remain open. Experimental directional CCD and augmented-Lagrangian contact remain disabled.

Reproduce with build.py and the commands in commands.txt. The trace shader, AIR and metallib are retained on ssh numi under /Users/numi/Work/mixed-pair-tangent-diagnostic-artifacts-20261108; input-sha256.txt records their pre-relocation identities. The build depends on the earlier research candidate on that host. Raw JSON, source, analysis and build output are retained here; this is not a portable release artifact or a performance measurement.
