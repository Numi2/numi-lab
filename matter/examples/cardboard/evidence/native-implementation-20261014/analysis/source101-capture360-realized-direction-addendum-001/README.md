# Source101 realized-direction diagnostic addendum

This addendum uses the stored v3 FP64 center and endpoint projections from the sealed uniform package006. It does not reproject endpoints, change material parameters, or modify package006. For all 1,728 environment-0 rows at root 2 / Newton 23, it applies the independent FP64 implicit tangent twice: first to the captured nominal rate direction `H`, then to the direction represented by the actual stored endpoint pair,

`H_realized = (F_plus - F_minus) / (2 * epsilon * dt)`.

The central stress difference uses the same scaling, `(P_plus - P_minus)/(2*epsilon*dt)`. The captured `dt` and accepted history are fixed; the configured rate exponent for this test is zero. The actual plus/minus `||F_endpoint-F_center||_F` values range from `9.9999999917e-8` to `1.0000000008e-7`; the largest pair-midpoint displacement is `8.27e-25`.

Against the nominal direction, the addendum reproduces the package006 result: 1,727/1,728 rows are within `1e-6`, with tet1186 at `1.214904e-6`. Against `H_realized`, all 1,728 are within `1e-6`; the maximum is `7.324850e-7` at tet1515, median `4.286796e-9`. Across rows, `H_realized` differs from `H` by relative L2 from `4.30e-11` to `1.61e-9` (median `7.82e-10`), while the resulting tangent action differs from the nominal action by as much as `9.60244e-7` at tet1186.

For tet1186, the realized-direction FD comparison is `2.56076e-7`, versus `1.214904e-6` for the nominal-direction comparison. The endpoint projections and recorded branch predicates are stable. This explains the common-step outlier as a direction-representation sensitivity for that row; it does **not** replace the nominal-direction gate, establish a universal noise floor, or qualify the entire 1,728-row tangent validation under a new criterion. The original package006 gate and report remain unchanged.

## Reproduction and verification

Run `python3 addendum.py` to produce the bounded analysis, then `python3 verify.py` to check bindings and row-level invariants. The run completed 1,728 rows in 0.093 CPU seconds with no warnings. It uses no GPU and no new projection solves. Per-row results are in `output-001/realized_direction_rows.jsonl`; the aggregate is `output-001/report.json`.
