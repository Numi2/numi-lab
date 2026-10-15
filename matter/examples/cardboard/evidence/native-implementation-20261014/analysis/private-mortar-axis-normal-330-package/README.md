# Axis-aligned normal-relative whole-parent certificate

This is a separate candidate after the 327 common-translation guard and 329 bounded-TwoSum supplement. It adds a second whole-parent early-return proof for exact, fully covered cohesive parent pairs whose triangles stay in parallel XY planes. Both parents must keep their shape by exact common linear translation; their projected parent vertices must match at both endpoints under one permutation; and each patch ID must have the same projected vertex sweep on both sides. This admits only relative Z-axis closure or opening. It never discards an individual CCD witness.

Every uncertain case falls back to the existing CCD path: lateral offset (including a representable 3.725 nm control), projected shear, rotation/nonplanarity, mid-sweep collapse, incomplete coverage, cross-wired patch IDs, or a projected cell whose orientation is numerically unresolved. Same-ID patch tilings are checked as complete lists, not by a single matching point.

The shared CPU/Metal helper suite passes 47 checks, including 2-, 3-, and 4-patch tilings. Full `matter_kernels.metal` compiles and links with Metal 4.0, `-O3`, and `-fno-fast-math`; two unrelated pre-existing FGMRES maybe-uninitialized warnings remain. The patch applies cleanly to the exact 329 source. No GPU/native run was made, so this is not contact qualification.

Apply after the 329 `two-sum-range.patch`:

```sh
git apply /absolute/path/to/axis-normal-relative.patch
```

The proof is intentionally axis-aligned and conservative; non-dyadic mapped trajectories may decline the optimization and remain on ordinary CCD. See `manifest.json` for source and patch bindings.
