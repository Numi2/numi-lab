# Prepared FEM-node to moving-bone triangle contact candidate

This isolated candidate is at /tmp/numi-human-bone-triangle-contact-001, based on repository HEAD a1a4b5340181fa48caf3006b5a1b8c565d2cc471. No runtime or application headers were changed in the shared checkout. candidate.patch is the reviewable diff against that HEAD; test-receipt.json records source hashes and bounded evidence.

## Implementation

The candidate adds an opt-in triangle-contact sample to the existing HumanFEMBodyContact path. The sample remains 64 bytes. Its 12 floats encode a body-local triangle (three vertices), explicit slave FEM tributary area, contact stiffness, and strain-per-pressure response. The host validates the sample mode, reserved bits, finite payload, positive area/stiffness/strain, triangle area, and finite area-times-stiffness. The current pose quaternion is validated in the Metal state path, not by the sample-only host validator.

Metal computes the closest point on the prepared triangle, including face, edge, and vertex features. Contact uses zero physical gap with a 1e-7 m numerical deadband, outward authored winding, feature-aware force tangent, explicit slave area, and the existing force, wrench, and audit owner kernels. Legacy plane and elastic-attachment branches remain available and are parity-tested. The CPU force/tangent oracle is isolated under tests/numi_human_fem_body_triangle_contact/support; it is not installed in the production include tree.

## Reproduce on Numi

~~~sh
SNAPSHOT=/tmp/numi-human-bone-triangle-contact-001
REPO=/Users/n/numi-human-aortic-contact-opt-20261019
T="$SNAPSHOT/tests/numi_human_fem_body_triangle_contact"

clang++ -std=c++23 -Wall -Wextra -Wpedantic -Werror \
  -I "$SNAPSHOT/matter/include" -I "$REPO/matter/include" -I "$REPO/include" \
  "$T/fem_body_triangle_contact_math_test.cpp" -o "$T/fem_body_triangle_contact_math_test"
"$T/fem_body_triangle_contact_math_test"

xcrun -sdk macosx metal -std=metal4.0 -O0 -fno-fast-math \
  -I "$SNAPSHOT/matter/include" -I "$REPO/matter/include" \
  -I "$REPO/include" -I "$SNAPSHOT/matter/src" \
  -c "$T/fem_body_triangle_contact_coupon.metal" -o "$T/fem_body_triangle_contact_coupon.air"
xcrun -sdk macosx metallib "$T/fem_body_triangle_contact_coupon.air" \
  -o "$T/fem_body_triangle_contact_coupon.metallib"

clang++ -std=c++23 -O0 -Wall -Wextra -Wpedantic -Werror -fobjc-arc \
  -I "$SNAPSHOT/matter/include" -I "$REPO/matter/include" -I "$REPO/include" \
  "$T/fem_body_triangle_contact_coupon.mm" -framework Foundation -framework Metal \
  -o "$T/fem_body_triangle_contact_coupon"
"$T/fem_body_triangle_contact_coupon" "$T/fem_body_triangle_contact_coupon.metallib"

clang++ -DNUMI_MATTER_DEFAULT_METALLIB='"$SNAPSHOT/NumiMatter.metallib"' \
  -I "$SNAPSHOT/matter/include" -I "$REPO/matter/include" -I "$REPO/include" \
  -x objective-c++ -O3 -DNDEBUG -std=c++23 -arch arm64 -fPIC -fobjc-arc \
  -Wall -Wextra -Wpedantic -Werror -fsyntax-only "$SNAPSHOT/matter/src/numi_human.mm"
~~~

The repository CMake target uses metal4.0, so the coupon does too. Its Metal build emitted 15 warnings from unchanged shared runtime includes; host coupon and owner syntax compilation passed with -Werror.

## Evidence

The CPU test passed with relative tangent errors of 2.01e-11 (face), 1.35e-9 (edge), and 3.4e-9 (vertex), zero action/reaction force residual, and 1.1e-30 moment residual. The Metal coupon passed on Apple M6 for five triangle states and two legacy states. It dispatched the existing owner load, body-wrench, and accepted-audit kernels and checked their outputs against the local triangle state.

## Limits

A prepared triangle query is not global nearest-surface contact or broadphase. Signed distance is only meaningful for the owning triangle and its valid narrowphase neighborhood; it must not be extrapolated behind a distant or unrelated triangle. One-contact-per-slave admission remains necessary to prevent duplicate tributary area. This candidate has no full-scene contact integration, coverage, convergence/stability, multi-triangle collision, or complete FEM qualification claim. Scene attachment remains explicit and opt-in.
