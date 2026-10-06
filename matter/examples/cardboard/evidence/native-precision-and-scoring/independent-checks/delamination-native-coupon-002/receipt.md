# Native bleached-paperboard Mode-I coupon receipt

The native constitutive evaluator check passed on Apple M4. CTest reported
`1/1` passed in 6.95 seconds with 11,894 assertions. The test invoked the
production constitutive evaluator on a one-tetra instrument fixture and ran
zero physical FEM steps. This is a material-point evaluator result, not a
corrugated-board, starch-adhesive, crease, or folding qualification.

Run command:

```sh
ctest --test-dir build-cardboard -R matter.metal.cardboard_delamination_coupon --output-on-failure
```

`LastTest.log` preserves CTest's full output and `ctest.log` its concise result.
`build-artifacts/` preserves the checker executable and specialized metallib.
`source-final/` preserves the material, metadata, CPU oracle, native checker and
kernel, plotting tool, and Robertsson assessment. `SHA256SUMS` binds every
file in this bundle except itself. `source-revision-ledger.md` records retained
failures and the limits of historical source binding.

Measured checks:

- The material fingerprint was `5570626462714653447`. At the 0.1 mm opening,
  native traction was `0.1701978437 MPa`. Native work was `26.10949601 J/m²`
  versus `26.10950156 J/m²` from the independent source oracle. Work error
  fell from `1.471085757e-3 J/m²` at 32 increments to `5.547617313e-6 J/m²`
  at 512 increments.
- Maximum native loading-tangent relative error against the source envelope
  was `2.083090985e-6`; maximum centered native stress finite-difference error
  was `1.123424749%`, under the `4%` test limit.
- At the exact FP32 history junction, `epsilon=1e-8` gives residual
  `1.414213562e-8` against local projection tolerance `2.022309478e-5`. The
  exact max-history seed is retained; the production tolerance admits that
  residual. The generated tangent selects the midpoint of the distinct source
  loading and unloading tangents, which is a numerical generalized derivative,
  not a unique source derivative.
- One thousand repeated native holds at the representable junction changed
  damage by exactly zero. The unload check preserved damage and measured
  free energy `0.156083926 J/m²` and dissipation `23.80819347 J/m²`.
- The positive seed stretch is FP32-quantized: native seed traction was
  `392978.5276 Pa`, `51.47236163 Pa` below the exact source seed traction of
  `393030 Pa`. The difference is reported rather than fitted away.
- Native admission rejected compression, zero opening, shear, and in-plane
  strain outside the declared Mode-I tolerance, while retaining the previously
  accepted state.

The source-oracle CSV and SVG show analytical load, unload, reload,
new-maximum, damage, work, free-energy, and dissipation trajectories. They are
not additional native measurements. The source material is single-ply bleached
clay-coated paperboard. This test does not implement a zero-thickness cohesive
face, finite separation/debonding topology, or a calibrated corrugated-board
starch law.
