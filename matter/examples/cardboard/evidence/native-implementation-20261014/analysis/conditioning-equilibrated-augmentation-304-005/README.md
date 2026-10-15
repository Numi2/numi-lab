# Same-Newton coarse basis equilibration, capture 304

This CPU-only offline study tests the online-feasible restart-16 correction as an extra coarse mode for the exact captured Newton-63 operator in run 304. It changes only the representation and scale of the coarse basis. The captured physical operator `A`, right-hand side, outer tolerances, native certificates, and history are unchanged. No native solve or physical qualification is claimed.

## Result

The original 26-mode coarse block has median absolute diagonal `0.0094878378151` and condition number `2047.8036`. The first-restart correction, after Euclidean projection against `P` and unit normalization, has raw `qᵀAq = +2.41407518e-4`; its Schur complement after eliminating `P` is `−6.35566215e-9`. The small negative number is the Schur pivot, not the raw `qᵀAq` of that Euclidean residual.

The useful basis change is to form an A-left-orthogonal representative within the same augmented span,

`qA = q − P (PᵀAP)⁻¹ PᵀAq`,

then normalize it and scale its column by `sqrt(median(abs(diag(PᵀAP))) / abs(qAᵀAqA))`. This gives a scale of `1244.6805`, makes the added Schur pivot about `−0.00948784`, and preserves the 27-dimensional span of `[P,q]`. For the FP32-constructed A-orthogonal mode and FP32 RAP product, the RAP is rank 27/27 at the existing `1e-7` singular-value cutoff, condition `2047.8094`, minimum absolute LU pivot `3.5967e-4`, and its Schur pivot is `−0.0094509163` versus the FP64 same-A reference `−0.0094878391`. The FP64 and FP32 signs agree. The coarse symmetric part remains indefinite; this is an LU conditioning observation, not an SPD claim.

The plain Euclidean `q` scaled directly to match its **raw** `qᵀAq` reaches rank 27 but has condition about `6.98e5`. Scaling the Euclidean mode by its tiny Schur pivot without first A-orthogonalizing instead leaves rank 26 at the unchanged cutoff and worsens condition to roughly `2.14e9`. The A-orthogonal basis change is therefore essential to the favorable factorization metrics.

In an approximate offline restart replay, the best FP32 A-orthogonal/scaled candidate reaches true residual norms `6.5051e-5 Ns` at 32 iterations, `1.0005e-5 Ns` at 64, and `4.0015e-8 Ns` at 128. The corresponding relative residual at 128 is `1.4348e-4`. The requested `1e-6` relative gate is an absolute `2.7888e-10 Ns`, so this candidate remains about 143 times above that gate. This is not enough to resolve the native linear solve.

The unscaled Schur pivot agrees in sign across the captured FP64 matrix and the offline FP32 cast/product: `−6.12423e-9` versus `−6.19730e-9` (about 1.2% magnitude difference). That resolves the sign for these offline calculations, but does not establish native Metal accumulation or factorization parity.

## Reproduction and binding

Run:

```sh
python3 /Users/home/cardboard-implementation-20261014/analysis/conditioning-equilibrated-augmentation-304-005/run-v2.py
```

The machine-readable results are in `report-v2.json`; `report-v1.json` is retained as the earlier Euclidean-only comparison. The v2 script SHA-256 is `2ebdec2b591d35a76318220960c13db27db2c9503ef9195d94df968be4c6d40f`; report SHA-256 is `2716a91017dcb152847079f3085384c4ab31e155bbb9862d6c75fcf42b1f7036`.

The report binds the exact input run and files: prediction SHA `6fd0e39606d2b05eb172a5ba76c2d589103f196604de1eb00999c74204ec00a1`, operator capture SHA `609ebcc9751f7ba5a5d42154f1969d065bd779ea44e901c80bb3c925152089ab`, operator report SHA `9d55e082c17fa246a23a98e96ca17832fff8814d017f0bf69eaf1d2196408f6d`, regional manifest SHA `75fd9d878fdd22c01929f7bb231046480d0daec9cefd77318d4703aff09ce8b00`, and failed-run receipt SHA `abd525c6de4a9932da3869a080fa00ef3a3307d76f0b70a834f9bc8a41910a3a`.

## Limits

- `A` is the exact captured offline operator matrix (2538 free DOFs, 71,046 stored nonzeros). The FP32 candidate casts that same matrix and the basis to FP32 and forms AP/RAP with NumPy/SciPy CPU products; this is not native Metal arithmetic parity.
- The restart-16 mode is generated from an offline reconstruction of the same-Newton preconditioner using captured `A`, not from a native live restart readback. The captured-native and direct-solution modes in the report are retrospective controls.
- Balanced-Schwarz factors are reconstructed from `A` and solved in FP32 offline. Arnoldi and its restarted replay are offline calculations.
- The approximate restarted replay does not pass the `1e-6` true-residual gate. No production change is recommended from this result alone. A native same-Newton restart capture and dense/sparse factor parity check are still required before integration.
- No rank threshold was lowered, no physical operator was regularized, and no tolerance was changed.
