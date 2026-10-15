# Whole-parent nondegeneracy supplement

This is a separate supplement to the sealed 321 whole-parent identity candidate. It closes one specific proof gap: 321 checked that each parent triangle was nondegenerate at both sweep endpoints, but the triangle could still collapse at an intermediate time while two vertices exchange positions.

The supplement admits the whole-parent skip only if all three vertices of each triangle have exactly equal linear displacement vectors across the complete start-to-finish interval. It compares FP32 endpoint differences with an error-free `TwoSum` high/low expansion rather than rounded FP32 subtraction. Under the linear endpoint trajectory, this common translation keeps each triangle shape unchanged for every time in the interval. Any deforming or numerically unresolved motion falls back to the existing CCD route.

The CPU suite passes 29 checks. The new adversarial control has nondegenerate start and finish triangles but degenerates at mid-sweep; the candidate rejects it. An exact common translation still passes. The full Metal kernel compiles and links with the repository flags; only the two previously observed unrelated FGMRES warnings remain. The patch applies cleanly after 321. No GPU/native run was made.

This is intentionally conservative. It does not establish safety for relative-normal motion, shear, opening, or any shape deformation. The supplement does not cure positive mortar sweep cases that remain rejected and is not contact qualification.

Apply after the 321 patch:

```sh
git apply /absolute/path/to/common-translation-nondegenerate.patch
```

See `manifest.json` for exact 321 base hashes and changed-file hashes.
