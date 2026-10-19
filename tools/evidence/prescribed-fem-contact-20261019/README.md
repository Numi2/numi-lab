# Prescribed FEM contact rows

Contact residual accumulation had added impulses to prescribed static FEM rows after the mechanical owner zeroed them. Newton could not satisfy the resulting equations. Rigid, sparse-rigid and deformable FEM contact residual kernels now consult the actual candidate and omit static/massless prescribed rows. Human attachment tag 2 remains dynamic; its J-transpose reaction path is retained. The existing separate contact-support reaction publication is unchanged. The trial Armijo residual uses its own candidate. No residual tolerance is changed.

Native Apple M6 on SSH Numi, no visual verification:
- 023: isolation source rejected for missing payload identity.
- 024: two tetrahedra and isolated closed linear CV compartment still fail residual check before the fix (0.000626488). This confirms circulation is not required to trigger it.
- 025: same isolation after correction passes late rejection, bitwise restoration, retry and clean parity.
- 026: two tetrahedra with full retained 21-compartment/24-edge CV graph pass the same regression. Nonzero deformable contact history, material-history change and accepted one-microtick advancement are required.
- 027: fixed-plane contact with full graph also passes.

Each attempt uses one 0.25 ms microtick, synthetic history-bearing silicone, production residual tolerance 1e-4, 32 Newton iterations and 16 line-search trials. Node-state maximum includes velocities. These are bounded numerical regressions, not whole-body contact, biological material validation, prolonged circulation or sustained blood-liquid qualification. The exact source, binary and library snapshots remain in each remote named run directory; compact commands, source identities, hashes, device and outputs are retained here. Older failing evidence remains unchanged.

Follow-up 028 also passes with explicit checks that all seven prescribed node positions remain bitwise unchanged and the fixed target receives nonzero contact support force (L1 1.73319 N). The fixture blits the existing private reaction buffer once for this regression; this is not a production per-step readback.
