# Ordered cohesive-page stage diagnostics, native run 287

Run 287 is a retained failed-prefix comparison of legacy versus tiled cohesive-page storage with the same authored solver/material settings. The native receipt reports exit 1, no timeout, unchanged inputs, and 234.68 s. The source snapshot is `source-captured-point-and-contact-scope-086`; the analysis script verifies all 772 retained source-file hashes against the copied source-build manifest and checks the executed checker and required Metal libraries against its artifact hashes.

The expected zero-face negative control runs first and fails closed before a transaction. The four arms then run in order: one-face legacy, one-face tiled, three-face legacy, three-face tiled. The earliest unexpected failed action is one-face legacy closure validation, requested control step 2, after activation step 1 was accepted. The matching one-face tiled closure also rejects: code 10, 16 FGMRES iterations, nonlinear residual `1.69753184e-4`, invalid certificate. The solver trace reaches Newton record 31 with only tiny residual reduction on that last pass. Before and after persistent accepted-state vector digests match across the rejected transaction, while serialized snapshots differ because the rejected status/certificate is retained.

The later three-face arms fail at their first action: activation is rejected with code 6, `failing_index=78`, `diagnostics=[16,99,0,1]`; the first trace record is already rejected at Newton 0, and the remaining records repeat that status. The raw before/after files and accepted-state hashes match exactly between legacy and tiled layouts. This report preserves those diagnostics without guessing their source-level meaning.

For both face counts, paired legacy/tiled raw transaction snapshots are byte-for-byte equal, paired status/certificate records match, and paired Armijo traces match exactly. The zero-face negative control fails closed before any transaction as expected. This establishes observational parity only for the executed prefixes. It is not a complete four-arm parity result: the one-face arm cannot pass closure, and the three-face arm cannot pass activation. No tiled-layout repair is indicated by this comparison because the legacy arm has the same failed states and bytes. The next useful step is a narrowly scoped test diagnostic mapping the three-face status index to its producing activation/contact command and stable face ID; solver/material or geometry changes are not justified by this parity evidence.

Reproduce the audit:

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 -W error \
  /Users/home/cardboard-implementation-20261014/analysis/tiled-cohesive-stage-diagnostics-287/analyze_stage_diagnostics.py
```

`report.json` binds the report to the run receipt, prediction, copied source/build manifest, executed binary and libraries, arm logs, and raw transaction snapshots. `manifest.json` adds hashes for this analyzer and README as well as the report and native inputs.
