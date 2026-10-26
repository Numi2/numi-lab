# P150: offline surface-clearance construction checkpoint

P150 records an offline geometry-construction candidate from the full-plane raw LP branch. Its exact owner-candidate check reports **2326 surface pairs, 0 contained pairs, 0 self pairs, 185 complete banks, and 237 typed anchor rows**. The candidate SHA-256 is `881e8e588a15bb74057ad1b587125b47aad7b483e424a0b75f29f9b82ecf94db`; the owner-check report SHA-256 is `851dfe665dd145985fa75b85b46be6d58f989e72040d89f74abb33a8f81ec092`. The owner-final report independently repeats the same exact counts. It reports maximum source displacement 11.995442 mm (below the recorded 12 mm cap), maximum step 0.499992 mm, and a passing fixed-reference area floor check (minimum ratio 0.100093569 against 0.1).

The persisted owner-final state is included verbatim and retains `candidate_selected: false`. P150 therefore records a fully checked construction candidate; it does not promote that state flag. The full exact check is the recorded owner-candidate-check using the 130 and 159 evidence; no geometry audit was launched by this publisher.

## Negative evidence retained

The bounded full-plane cut-generation trials 003, 004, and 005 produced intermediate candidates with 76, 10, and 7 exact self pairs respectively. Their emitted candidates and self-audit evidence are retained. Trial 005 then passed the owner candidate check at 2326 / 0 contained / 0 self.

The backtracked 2321 candidate (`2efbf111ab53f7a75e111e9a3556d071caf174a3a6784a139f7b28aab6c951bd`) reduced surface-pair count from 2409 to 2321 while leaving point pairs at zero. Its depth report recorded 225 introduced surface pairs and no construction-depth progress; the fresh edge-gradient gate failed. This branch remains a failed alternative and is not the published checkpoint.

The selected trial retains its earlier raw directions, projections, failed self trials, backtrack iterations, witness reports, and solver evidence. The large `fit-stage.log` stdout JSON mirror is omitted; its SHA/size and the retained structured `fit/report.json` are recorded in `omitted-payloads.json`.

## Inputs and source pins

P148's 2409 baseline candidate and P149's 2395 checkpoint remain available at the exact committed paths listed in `prior-publications.json`; P150 references those inputs rather than copying them again. P150 includes the bounded cut-generation helpers and receipts, raw LP capture helper with its source pins and tests, and the area-quality 548 helper/tests. Its baseline face 22027 arithmetic receipt gives area ratio 0.100803669 and minimum angle 4.995613 degrees; it is explicitly arithmetic-only, with flip replay NOT_RUN and flip eligibility NOT_ASSESSED. The whole-mesh fixed-reference minimum is 0.100093569 against a 0.1 floor. The area-quality unit tests passed 5/5; the current raw LP capture suite passed 18/18 with the pinned helper. Their receipts and exact sources are included. Launch receipts preserve exact source/input hashes; `source-pin-index.json` maps available pinned Python snapshots.

The verified P144/P145 staging recovery receipt is included at `verification/prior-publication-staging-recovery-20261026.json`.

## Qualification limits

This is offline construction evidence. It records no visual inspection, native runtime admission, physical skin/fat transfer, FEM/contact acceptance, or clinical/anatomical qualification. `native_admission` is false and `accepted_physical_steps` is zero throughout. It does not establish closed-skin containment or physical performance.
