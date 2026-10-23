# Human FEM barycentric surface attachments

This isolated increment adds a finite, stateless barycentric spring from one
FEM surface triangle to a point fixed in an articulated body's frame. The
triangle nodes receive the transpose-scattered force; the body receives the
equal-and-opposite force and moment. The energy and generalized-load
projection share the existing Human FEM body-contact transaction and rollback
audit.

The sample stores k in Pa/m, tributary slave area in m^2, and therefore kA in
N/m. It is not a hard node pin, a global closest-surface search, or a
broad-phase bone-contact system. A caller must already own the correct
triangle/site correspondence. Each record is validated on the host; accepted
body pose, node positions, triangle area, force, pressure, strain and energy
are validated again on Metal before force use.

The existing 64-byte legacy body-contact record and 96-byte dispatch record
remain the same size. The optional span is appended to
NumiHumanTendonFEMLoadSource, and its new typed record is 80 bytes. This
changes the C++ source/owner layout: rebuild the complete Matter library and
its caller together. Do not link a previous compiled owner library to the
changed source declaration.

Run the isolated checks on the configured Numi host:

    python3 /tmp/numi-human-barycentric-attachment-001/overlay/tests/numi_human_fem_surface_attachment/run.py

The runner compiles the actual owner translation unit with -Werror, checks
the production host input validator for empty, surface-only and mixed source
spans plus malformed-record rejection, checks B^T work and the complete
off-diagonal tangent by finite differences, then builds and dispatches the
actual owner Metal kernels on Apple hardware. The GPU coupon covers
surface-only, legacy-plus-surface, legacy-only empty-surface binding,
generalized-load projection, accepted audit commit, and invalid state records.
run-receipt.json records hashes and outputs for the last run.

The source-span cases validate the added-record input gate and the GPU coupon
exercises its kernel transaction; the full adapter still follows its existing
node-anchor and endpoint requirements. This is an owner/kernel coupon only. It
does not qualify any Human scene,
attachment-site selection, whole-body contact, production simulation, or
clinical/biological behavior. The overlay is intentionally isolated; its
patch must be reviewed and applied to the matching source revision before a
full library rebuild or app integration.
