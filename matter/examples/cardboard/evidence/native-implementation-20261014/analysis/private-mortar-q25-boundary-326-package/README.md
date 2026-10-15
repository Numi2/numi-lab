# Exact paired-boundary admissibility overlay

This isolated overlay is based on source snapshot `source-exact-tiling-inactive-fields-100`. It addresses the 4 nm off-edge false suppression in the retained native-325 case. The old paired-boundary test used an `8e-6` barycentric proximity tolerance; a 4 nm offset on a 4 mm edge is about `1e-6`, so the old predicate could treat a real initial gap as a shared seam.

The overlay changes interior correspondence to exact Q25 collinearity and exact rational segment-parameter equality, permits only exactly corresponding endpoint witnesses outside Q25, and requires both paired physical endpoints to coincide exactly at the accepted start. It also routes that physical-segment test through the shared host/Metal helper. The `8e-6` constant is named and used only as a minimum edge-length degeneracy check; it no longer establishes paired-feature identity.

The CPU guard test passes 19 checks, including exact shared-edge acceptance, a 4 nm parameter offset rejection, 4 nm physical offsets at either endpoint, and one-ULP/near-edge negatives. The complete Metal kernel source compiles with the repository Metal 4.0 flags. This was not a GPU run. The checker `.mm` translation unit also passes Objective-C++ syntax checking after suppressing two unrelated pre-existing unused-symbol warnings in this frozen fixture. Repository CMake configure could not complete against the archived source bundle because `bindings/swift/MetalRoboTaskRollout.swift` is absent; CPU and Metal checks were therefore compiled directly.

The retained native-325 case is pre-fix evidence. Root should apply the patch to the bound source and rerun the full private-self-sweep matrix. Three-/four-patch positive cases that remain rejected are still unresolved; this overlay does not claim contact qualification.

Apply from the repository root:

```sh
git apply /absolute/path/to/q25-paired-boundary.patch
```

Check `manifest.json` for exact base and changed-file hashes.
