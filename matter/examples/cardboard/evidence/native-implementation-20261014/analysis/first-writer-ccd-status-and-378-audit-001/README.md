# First-writer private CCD status and 378 comparison

This source-bound audit preserves the strict 362/369/371 result as `FAIL_UNEXPECTED_STATUS`. It traces the Newton-7 status tuple to the private deformable-contact narrowphase and separately classifies run 378, which uses a changed edge-edge CCD implementation.

The two Newton-7 error tuples are consistent with different uncertified CCD candidates winning a shared status race:

- 362 and 369: failing slot 283, candidate primitive IDs 92 and 127, point-point CCD not certified.
- 371: failing slot 124, candidate primitive IDs 24 and 127, point-triangle CCD not certified.

In source 106, `recordFailure` at `common.metalinc:294-318` sets the status code with a compare-exchange from success. The winning thread writes the rest of the status and later contenders return. `nm_contact_narrowphase_deformable` processes one candidate slot per invocation, and the point-point and point-triangle uncertified branches at `contact.metalinc:3789-3797` and `3821-3833` write those exact diagnostic forms. No deterministic minimum-key reduction appears in the audited path.

This establishes that first-arrival status selection is possible and explains the tuple change, but the private candidate/failure list was not captured. The evidence does not prove that both candidates failed in every individual run. No status normalization or physics pass is claimed.

Run 378 is a separate physical-source variant: source 107 changes the edge-edge closest-distance loop from 17 to 33 iterations (`contact.metalinc:3365`). Its Newton trajectory differs by iteration 5; its Newton-7 trial is accepted, and Newton 9 later fails ordinary residual Armijo with trial status clear. Its center capture differs in 22 of 145 arrays from 371, so the 371/378 status tuples are not a same-input parity comparison.

## Reproduce

```sh
python3 audit.py
```

The script validates source hashes and native receipts, extracts the selected Newton rows, compares full center-capture array payloads, and writes `report.json`. A strict run that exposes the unexpected status remains a failure by design.

The immutable inputs and exact hashes are listed in `manifest.json`.
