# Parent reconstruction guard

Native import-time validation reconstructs registered O/I/D points from the exact current parent tetrahedron, four barycentric weights, and declared residual. It retains the existing 2 cm registration bound, rejects missing O registration or wrong tissue material, and requires reconstruction within 20 nm. This is preparation code; it evaluates no per-step physics.

Strict C++23 compilation and synthetic correct/wrong-cell/missing-map/wrong-material regressions pass. A wrong-cell row that otherwise has valid finite barycentric weights is rejected, and caller output remains unchanged. The retained attempt-007 sidecar parses but is correctly rejected for missing O parent registration. A separate diagnostic of its original I/D stencils rejects at a 27.4 nm mismatch against capture-016 current Float32 parent points; rebuilding residuals against that actual parent packet is required. Tolerances were not relaxed.

No changed child volume, application scene hook, simulation step, physiological result, or motion qualification is established by this check. The lower-level parsing entrypoint remains usable for inspecting incomplete sidecars; the new `loadPhysicalSkinRegisteredSurfaceCSVSetForParent` entrypoint requires complete physical parent registration and publishes all outputs atomically.

On SSH Numi, compile `check.cpp` with `c++ -std=c++23 -O1 -Wall -Wextra -Wpedantic -Werror -Iapps -Iinclude -Imatter/include`, then run it with the exact sidecar directory and parent mass packet paths retained in `receipt.json`. Pass `--expect-pass` only for a complete matched registration set; its success is still an import check, not physical admission.
