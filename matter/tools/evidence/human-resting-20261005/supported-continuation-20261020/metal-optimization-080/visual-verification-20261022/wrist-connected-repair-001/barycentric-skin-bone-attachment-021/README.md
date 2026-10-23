# Barycentric skin/bone attachment owner - Numi 021

The production Human FEM load owner now accepts a finite spring between one
articulated-body material point and a barycentric point on three FEM nodes.
It preserves one source attachment, tributary area and kA spring law under
surface refinement. Node forces are B-transpose scattered, and the existing
body wrench, generalized projection, accepted audit and rollback owner carry
the reaction. No CPU simulation or hidden positional constraint is added.

On SSH Numi / Apple M6, the release runner passes production owner -Werror,
host validation, transpose work, all nine node-pair tangent blocks by finite
differences, actual Metal force/wrench/energy/projection/accepted-audit tests,
and invalid-state rejection. CPU work error is 9.04e-10, tangent FD maximum
error 1.91e-8. GPU surface-only gives 200 N / 10 J, mixed 230 N / 11.5 J,
legacy-only 30 N / 1.5 J; projected reactions are -200/-230/-30.

A fresh full Matter library and native Human target build completed. Actual
adapter initialization tests empty, legacy, surface and mixed optional spans,
ten malformed-input cases, snapshot source authority and unchanged physical
nodes. Existing node-anchor and endpoint admission remains intact. No physical
time advances in this new initialization test.

The unchanged existing tendon/FEM coupled-owner regression passes actual
physical updates, accepted histories, rejection exclusion, bitwise replay and
verified rollback, including the legacy elastic attachment law. This checks
regression of that path; it does not qualify new surface attachments throughout
a full Human motion run.

The optional source span and diagnostics field change C++ layouts. All library
and app callers were rebuilt together; no retained 039 library was overwritten.
The 64-byte legacy record and 96-byte dispatch retain their sizes. One reserved
dispatch word carries surface count. Snapshot format remains unchanged; source
fingerprints include new records. Site correspondence and implicit spring
operators are not provided by this increment.

This does not admit the 011/014 skin/fat candidate, remap the six changed hand
sites, solve global bone contact, fix the failed elbow surface, qualify natural
joint range, or establish full-body performance. No images or viewer inspection
were performed. The full build uses the preserved dirty checkout; exact inputs,
commands and binary hashes are retained separately from the narrowly published
owner change. Failed anatomy trials remain untouched.
