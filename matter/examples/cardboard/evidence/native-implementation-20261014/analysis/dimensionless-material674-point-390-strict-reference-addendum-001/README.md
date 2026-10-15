# Point 390 strict FP64 root addendum

The pre-existing FP64 reference helper stopped under its own weighted residual criterion, which does not imply the production dynamic scaled-correction gate. This addendum recomputes the dynamic correction using `max(abs(accepted), abs(candidate), parameter[26])` and the current complex-step Jacobian. It preserves the earlier return, records its failed dynamic correction check, then performs exactly one extra FP64 Newton update and rechecks the unchanged gates.

Run with Python 3.10 and NumPy:

```sh
python3.10 strict_reference.py
```

No production thresholds or equations are modified. The native comparison remains one point and one direction; it is not an FD/noise-floor or physical qualification. `fp64-reference-vectors-674.json` is the frozen point reference used by the opt-in verifier package.
