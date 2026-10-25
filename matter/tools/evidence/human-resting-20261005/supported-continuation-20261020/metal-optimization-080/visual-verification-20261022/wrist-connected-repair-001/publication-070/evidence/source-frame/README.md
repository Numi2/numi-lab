# Current 294 source-frame audit

This source-only audit pins the 294 NHSKIN1 payload, its native emitted current-pose CSV, the D36 direct/barycentric ancestry, the AP basis and rigid placement, and the saved 039-to-294 diagnostic displacement. It does not edit or emit an NHSKIN payload and does not rerun the harmonic solve.

The native renderer computes a point as a weighted sum across all skin bindings. The whole-body mechanical region records that same world point, then also derives a dominant-donor frame for its per-node mechanical coordinates. A change to raw NHSKIN vertex.position should therefore be obtained from the full per-vertex weighted linear-skinning Jacobian, not by treating the dominant-donor inverse rotation as the source-position transform.

Run on SSH Numi:

    /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v test_reference_frame_mapping
    /Users/n/numi-human-prep-venv-20261005/bin/python reference_frame_mapping.py --output /Users/n/Work/294-source-frame-audit-001/run-001

The report checks the pinned body-pose/binding map against the existing native current-pose CSV before it writes a diagnostic source-position-delta NPZ. The result is not a mesh, collision, material, mass, FEM, native-runtime, or physical acceptance gate.

## Bounded P1 offset-transport experiment

The pinned D36 owner helper was inspected without changing its production source. makeCurrentCoarseBaseline rotates the coarse-to-detail reference offset with the donor's rigid pose. applyReference then adds the barycentrically interpolated coarse world displacement to the detailed point. It does not carry the offset through a local P1 deformation gradient.

An isolated copy of the owner header adds applyReferenceWithP1OffsetTransport, which evaluates
x_detail_new = x_coarse_accepted + F_P1 * (x_detail_current - x_coarse_baseline_current).
The existing additive method is unchanged. A small C++ fixture checks affine transport, identity parity with the legacy method, shape mismatch and nonfinite rejection.

The bounded run uses the frozen AP baseline and connected-volume continuation 039 iteration 005. It evaluates 4,777 outer faces, each paired with its unique material-1 boundary tetrahedron, and 2,411 direct/barycentric anchors. Face P1 gradients all had positive determinant (minimum 0.3564); nodal gradients are area-weighted incident-face averages and also had positive determinants (minimum 0.4227). The frame-converted D36 deltas reproduce the frozen candidate at direct anchors within 3.88e-18 m and barycentric anchors within 2.61e-18 m. Relative to additive transfer, the transported anchor correction has p95 0.0479 mm and maximum 5.424 mm; this is a measured alternative displacement, not evidence that the correspondence residual is repaired.

This remains an anchor-only experiment. It does not extend a P1 map to all 61,950 detailed vertices, rerun harmonic prolongation, change NHSKIN, or pass geometry, collision, mass, FEM, native-runtime, or physical acceptance. The nodal gradient average is a reconstruction because a P1 gradient is discontinuous at vertices.

Reproduce the pure mapping tests and helper fixture from this directory:

    /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v test_reference_frame_mapping
    clang++ -std=c++20 -O2 -Wall -Wextra -Werror -I. test_owner_p1_offset_transport.cpp -o /tmp/test_owner_p1_offset_transport
    /tmp/test_owner_p1_offset_transport

The measured run is preserved at p1-transport-039/run-004/. Its report pins the exact data inputs and output hashes. Earlier wrong-frame runs 001 and 002 remain preserved as failed diagnostics; do not use them as results.
