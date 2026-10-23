# LAW112 implicit elastic predictor repair

This bundle extends the five-file adhesive repair published in `62afeeb2`
with one narrowly scoped LAW112 branch in `engine/source/elements/solid/solide8z/mmats.F`.
It does not supply a consistent plastic/damage tangent or qualify a cardboard box.

The paired input is the source-derived liner MD tensile coupon in
`cases-material-transfer-06-direct`. The prior binary took about 2,100 small
steps and stopped near 0.067 s. Native correction output placed the initial
Poisson contraction in Z, although this LAW112 parameter set has no X–Z elastic
coupling. Inspection located the first defect in MMATS: LAW112 was absent from
its orthotropic dispatch and received the isotropic iteration matrix.

The new branch feeds the material's existing elastic coefficients into
MSTIFORTH: C11/C22/C33 from UPARAM 6/9/3, C12 from the symmetric average of
UPARAM 7/8, zero C13/C23, and shear moduli from UPARAM 10/11/12. The constitutive
force/history routine is unchanged. The elastic matrix is checked for positive
definiteness and the directional source moduli; it remains an initial iteration
matrix when the material yields.

The unchanged diagnostic coupon now completes 2.2478386 s in 121 cycles
(0.24 s reported Engine time, about 56 MB reported process memory). Its first
correction has the expected Y contraction, -0.00018 mm for 0.001 mm X motion.
The four liner/medium MD/CD source-transfer coupons also finish with direct
MUMPS. Their native measured stress/strain comparisons are separate evidence;
normal termination alone does not establish constitutive accuracy.

The PCG route still fails on these inputs and is not admitted. Retained prior
attempts include the invalid tensor-output keyword, PCG failures, direct
MUMPS timestep-floor failure, and an unsuccessful quasi-Newton attempt. They
are not replaced by the successful run.

Candidate 04 Engine SHA-256:
`c931e420bb8219d8ee9cdfb06cb1192e8efc8b422bafed7926d78ac2b11127d8`.
Combined patch SHA-256:
`034281f6880cb6e2da196767066cff9f74c1b96b5a9db48c3ab7677c1872e699`.
The packaged Starter and shared libraries are unchanged. The adapter requires
the distinct `arm64-mumps-paper` profile and verifies its source/build identity.
The older `arm64-mumps-type43` profile is preserved.

The accepted three-stage adhesive-birth protocol was replayed on Candidate 04.
All three native history files match Candidate 03 byte for byte after masking
only the validated 24-byte wall-clock timestamp record. See
`adhesive-regression.json`; the raw histories are retained.

## Reproduction

The task root on macmini is `/Users/n/cardboard-opencourant-20261023`.
The pinned upstream source is mounted inside the `cardboard-mumps` Lima VM.
The narrow build command is:

```sh
python3 tools/build_cardboard_opencourant_type43.py \
  --task-root /Users/n/cardboard-opencourant-20261023 \
  --patch-directory /Users/n/cardboard-opencourant-20261023/adhesive-fix/paper-predictor \
  --output /Users/n/cardboard-opencourant-20261023/adhesive-fix/candidate04 \
  --profile arm64-mumps-paper
```

Run each saved `case.json` with `cardboard_opencourant.py run`, the Candidate 04
runtime, `--runtime-profile arm64-mumps-paper`, `--threads 1`, a new output
directory and an explicit timeout. Run receipts retain exact arguments and
source, binary, library, parameter and checkpoint hashes.
