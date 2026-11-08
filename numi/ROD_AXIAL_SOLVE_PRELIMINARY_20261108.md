# Preliminary constrained axial rod results

The isolated research candidate implements the [declared exactly axial domain](ROD_AXIAL_SOLVE_PLAN_20261108.md) with a direct tridiagonal FP32 implicit solve and retains the prior stretch guard for other paths. It is not a general rod solver and has not been promoted.

On the Apple M6, it completed all three previously failing free-middle cases with the predeclared force and state gates. The 2-node witness passed with 3.27e-8 relative force error. The 41-step prescribed-endpoint series passed with maximum relative force error 1.021e-6 and work versus energy-change discrepancy 1.466e-6. The earlier fully pinned three-node case uses three attachments, outside this path, and still rejects under the stretch guard.

A separate dense-pivoted FP64 reference tested 5-, 7- and 17-node chains with varying node masses and edge stiffnesses, moving endpoints and axial gravity. All passed the state and force gates and repeated identically from the same initial state. The largest position error was 9.41e-9 m, velocity error 1.07e-5 m/s and support-force error 8.69e-7 N. Reversed target ordering explicitly rejected without publishing, and the input host state remained unchanged. This is not evidence of full resident-device transaction restoration or continuation from a saved checkpoint.

The existing nine-node bent/contact probe passed its own gates and matches the stretch-guard output for the listed physical diagnostics. It is not a force-qualified general rod reference. The candidate's precise-domain admission, large-coordinate and longer-chain conditioning, residual certification, continuous multi-step restoration and any deployment integration remain open. This run cannot qualify general sutures, bending, friction, contact, biological response or force accuracy outside the tested axial cases.

The [raw outputs, series, compiled identities and failure statuses](measurements/evidence/rod-axial-20261108) are retained. Candidate source is commit 98e00f7a7; the separate FP64 probe is at c28cab841. The original uncorrected and stretch-only failures remain in the parent evidence. Timing in the ordinary probe was not controlled.
