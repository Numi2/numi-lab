# Full cached tangent against independent FP64

This CPU audit compares all nine native cached material-tangent columns at the exact source125 run419, environment0, control1/Newton48 state with an independent FP64 constitutive model. It widens the separately captured FP32 high and low deformation words, holds the captured accepted history fixed, independently reprojects all 1,728 paper states, and differentiates the full implicit update. It also checks the192 finite elastic starch tets analytically. Every one of the17,280 columns is nonzero and below the requested1e-3 relative gate; the maximum is6.1358993e-5 for a medium column. The CSV and NPZ preserve all absolute errors, reference norms and matrices.

This resolves the selected analytic Metal material derivative comparison. It does not resolve the separate FP32 endpoint finite-difference noise question or establish correctness at other states, damage, interfaces or nonzero rates. The prior independent branch-preserving FP64 finite-difference intervals apply to one captured direction at each paper tet, not every basis column. No measured absolute floor is supplied for near-zero comparisons; zero-reference columns would be inconclusive. No coupon, contact, calibration or physical qualification follows.

The first consumer attempt incorrectly treated parameter26 as a rate exponent; the actual named parameter is `fb_epsilon`. That failed assertion, its exact script and logs remain in `attempt-001`. The corrected audit binds the existing capture's independently checked rate-independent program contract. The source capture and production implementation were unchanged. Reference-projection warnings are retained in the report; final state, residual, correction and derivative validity are checked separately.

To reproduce into a fresh output directory, copy `audit.py` there and run it using the qualified Python3.10.12 interpreter. Its exact source/capture/helper paths and SHA256 bindings are in `prediction.json` and `manifest.json`. It refuses to overwrite a report. For example:

```sh
mkdir /tmp/cardboard-full-tangent-replay
cp audit.py /tmp/cardboard-full-tangent-replay/audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/home/.pyenv/versions/3.10.12/bin/python3 /tmp/cardboard-full-tangent-replay/audit.py
```
