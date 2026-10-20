# Terminal replay: root 25, tick 3, slot 1990

This read-only replay reproduces the default039 terminal contact failure after 25 accepted steps (3.125 ms of the requested 4 ms). The replay exited 1 after 540.128 s. Its accepted CSV is byte-identical to `native-default32/resting-coupled.csv` (25 data rows); its common failure JSON is byte-identical too.

The selected event is Newton iteration 18 at `after_deformable_limit`: object 9, contact slot 1990, Matter code 6 (`contact_barrier_armijo_failed`). The identity kind is 1, which is EDGE_EDGE. The contact uses nodes [761, 757, 3042, 3135] and source [2040, 6976, 1990, 1].

All 12 tested alphas from 1 through 1/2048 passed full-fraction CCD certification, initial-touch admissibility, and finite-energy checks. Every trial failed only the feature barrier Armijo gate. The base values were E=2.65384648e-9, directional derivative=-4.49245641e-10, and separation=9.93649792e-6 m. At alpha=1, trial E=3.27947558e-9 and separation=9.93185040e-6 m. At alpha=1/2048, trial E=2.68289280e-9 and separation=9.93626691e-6 m, still above the base energy and Armijo right-hand side.

The trace reports `geometry_coordinate_frame=world`, zero reference origin, and zero low components. Its potential endpoints equal the projected CCD endpoints. The all-reference compensated expansion branch from 039 was not active for this event. Intermediate alphas show float32 endpoint plateaus: alpha 1/128 through 1/512 produces the same weighted realized displacement and trial energy, while smaller alpha changes to a different quantized endpoint. This is consistent with world-coordinate endpoint quantization at the roughly 0.5 m coordinate scale; it is an inference from the trace, not a separate precision oracle.

The correct diagnostic conclusion is that the represented trial states increase the contact potential at every tested alpha. Relaxing Armijo would accept a higher-energy state. A future correction must preserve sub-ULP trial motion in the state and potential representation that is actually committed, or retain the current safe failure.

## Bound evidence

- Run source base: `4642668b37f385ee7dde3d9fdde097fc0f5d9ab2`
- Executable SHA-256: `1520584bf3f534aafc17684817a91277f587587bca45af26cf780b4a6e8b0b16`
- Metallib SHA-256: `ccb3da2f4c0f806de944916e809e3f1043eebb17d2e842c425e7189fb79662fc`
- Run stderr SHA-256: `d34a6f8c821bde4e09294f4e781790b1d1be97d37118c3069e82d6b9544b47c5`
- Exact CSV and common-failure hashes/parity are in `summary.json`.
- `selected-traces.json` and `selected-traces.tsv` retain the selected alpha sweep; `summary.json` records parser/capture consistency.

This is a failed 3.125 ms continuation and a single-contact replay, not sustained-support or physiology qualification.
