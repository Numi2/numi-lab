# Source125 native derivative and input-separation note

This CPU-only package audits the existing source125 root 1 / Newton 48 rich capture and records a concrete next diagnostic. It does not rerun the solver, modify production files, or promote the 419 run to convergence, calibration, or physical qualification.

Run the reproducible audit into a fresh path with:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 \
  /Users/home/cardboard-implementation-20261014/analysis/source125-derivative-noise-plan-001/audit.py \
  --output /tmp/source125-derivative-noise-recomputed.json
```

The script verifies frozen source, run, capture, adapter, and sweep hashes; recomputes exact FP32-high-plus-low endpoint differences from the rich capture; and recomputes the independent elementwise signed-work decomposition from the 419 adapter's exact tet/node maps and force action. The output is `report.json`.

## What the capture already resolves

The selected capture binds environment 0, root 1 / Newton 48, 882 nodes, 1,920 tets, and 1,728 paper tets at `dt=9.99999975e-5 s`. The JSON writer round-trips FP32 words with `max_digits10`; the analyzer explicitly rounds each captured high/low word back to FP32 and widens the two words separately before addition. The side records use the same accepted material history as the center. The independent replay in `operator-fd-e2-adapter-001/coupon419-003` already reprojects those exact side inputs at FP64 and compares stress/state, with absolute Pa errors retained alongside relative errors.

At `epsilon=1/64`, the full production residual pair is admissible under the existing stage/status/finite-output checks, but the independent material branch audit still sees 2 liner and 31 medium plus/minus multiplier-activity transitions (liner: 0 minus, 2 plus; medium: 7 minus, 24 plus). At `epsilon=1/32`, global residual-stage status is not clear, although the per-tet local projection-valid fields are set for all captured paper rows. Do not interpret the side record's `flags_status_determinant[2]` as a side-local projection code: the kernel copies the environment-level diagnostic status there. The largest local FP32-vs-FP64 stress differences are 12–62 Pa for liner and 0.03–0.06 Pa for medium, while global relative stress differences are about `2e-8`–`3e-8`; these are observed same-input discrepancies, not universal arithmetic floors. The paired F high+low words differ from ideal requested endpoints by up to `4.60e-8` at 1/64 and `1.18e-7` at 1/32; the realized direction error is separately reported. None of these facts alone establishes the centered material derivative or the assembled operator derivative.

## Smallest next native diagnostic

Use the existing observational FEM operator-FD path, with a fresh output path and no production-source change:

```text
NM_FEM_OPERATOR_FD_SELECTOR=1:48
NM_FEM_OPERATOR_FD_EPSILON_SCALE=0.25
NM_FEM_OPERATOR_FD_CAPTURE_PATH=<fresh-capture-path>
```

In source125 `runtime.mm:10768–10807`, the scale is accepted only as a finite power of two in `[1/1024,16]`; the two base perturbations are `1/32` and `1/64`. Scale `1/4` therefore adds actual native pairs at `1/128` and `1/256`. Bind the run to the exact source125 package/provenance and require the new center candidate nodes, accepted history, F high/low words, parameters, frame/rest data, and dt to match run419 at the selected generation before combining results. If the candidate differs, analyze it as a separate linearization.

The existing rich capture already records exact node endpoint states and residuals, and per-tet center/plus/minus `F_local` high/low, `Fdot`, P, projection diagnostics, history, and tangent inputs (`fem.metalinc:3761–3865`, with writes later in the same kernel). `audit_capture.py:626–676` shows the correct independent replay pattern: round exact captured high/low words to FP32, widen, hold the accepted history fixed, independently project each endpoint, and compare native P/state in absolute and relative terms. The selected finite-difference comparison must then keep four quantities separate:

1. **Input representation:** requested `F0 ± epsilon*dt*H` versus the exact captured high+low endpoint, and the realized `(F+−F−)/(2*epsilon*dt)` direction.
2. **Projection/stress arithmetic:** native endpoint P/state versus FP64 projection/stress at those same exact endpoints and the same accepted history.
3. **Material derivative:** center tangent applied to the realized direction versus the FP64 and native stress secants. Compare nominal direction separately. Keep center and both endpoints' return-map branch signatures; a crossing is a branch result, not a smooth centered derivative.
4. **Assembled derivative:** production residual secant against the captured tangent action and an independent FP64 assembled action. Keep mass, material-element, and any contact/external residual terms separate. Use absolute L2 and max-component errors in Pa, N/s, and N·s; only report relative values where the corresponding reference norm is nonzero. A zero or below-resolution direction is inconclusive, not a pass.

`audit_capture.py` currently checks only the captured plastic-multiplier activity predicate in its endpoint summary; the richer native side fields and the independent accepted/projected states permit adding yield-sign and Fischer–Burmeister-path labels in the CPU report. Any new branch labels must be derived from the exact captured endpoint and source law, and must not be represented as a native active-set guarantee.

Do not enable `NM_FEM_REGIONAL_RAP_CAPTURE_ONLY=1` for this residual-FD run; that mode suppresses finite-difference readback. Also avoid combining a full regional RAP readback: at run419 the 22.89 MiB RAP payload plus 17.82 MiB FD upper bound exceeded the fixed 32 MiB aggregate cap. Keep each run diagnostic-only and preserve failed/invalid endpoints instead of filtering them out. A sequence of finite differences can reveal a trend, but it does not define or subtract a universal noise floor.

## Negative quadratic-form question

The sweep defines `K` as the assembled derivative of internal force for the captured velocity direction. `sweep.py:131–140` applies `dP` through the affine reference-gradient map with the standard internal-force minus sign; `sweep.py:614–646` forms `A dv = M dv - dt^2 K dv`. Production uses the same force convention in `fem.metalinc:260–276` and the same `M dv - dt^2 derivativeForce` gather in `fgmres.metalinc:5100–5122`.

The independent signed-work breakdown is `v·Mv=+0.01497290446 J`, `v·(-dt²Kv)=−0.01581523578 J`, and total `v·Av=−0.00084233132 J`. Reassembling `v·K v` from the independent element forces by material reproduces the negative stiffness work within `8.4e-12 J`: liner contributes `−0.02327130131 J` after the `−dt²` factor, medium `+0.00644636342 J`, glue `+0.00100970210 J`. Thus the sign is in the material-tangent internal-force derivative after the expected `−dt²` factor and is dominated by the liner contribution. This is not a separate geometric-stiffness matrix and is not explained by a residual-sign bookkeeping mismatch. The reference-rest gradient and fixed material-frame transforms are in the assembly map; the code does not add a separate geometric Hessian.

A negative direction of this Newton operator is not itself a physical energy violation or a bug. It means the assembled algorithmic tangent has positive internal-force directional work along this captured direction, while the solver's signed stiffness contribution is negative. To determine whether that is a valid constitutive tangent feature or a material-law issue, the next small audit is elementwise `dF_e : C_e : dF_e` (equivalently `v_e·K_e v_e`) for the captured 1,728 paper tets, grouped by material and branch, followed by local tangent symmetry/eigenvalue diagnostics. Keep that separate from the residual sign and from stored-energy evaluation.
