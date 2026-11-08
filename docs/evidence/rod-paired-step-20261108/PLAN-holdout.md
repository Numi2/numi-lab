# Paired axial step: holdout

After the predefined 12 x-axis replication cases all passed, test the exact
unchanged candidate and binary on six-node x-axis rods at offsets 0.4, 1.25,
and 100 m, fixed, moving and internal anchors, twice each: 9 distinct cases,
18 executions. Use the same independent dense FP64 reference, 5/10 nm energy
gradient checks, and all gates defined in PLAN.md, without adjustment. Report
all rejections, failed gates and repeat differences. This holdout remains a
one-step synthetic straight axial no-contact test and does not qualify the
unsupported fallback, rollback, coupled or production state.

Instrument correction, before any valid holdout execution: all initial 18
launches returned usage status 2 before loading the model because probe.mm
only admits 5 or 8 nodes. Preserve those raw launches as an instrument failure.
Allow six nodes in a copy of the probe; keep candidate-host.mm and Metal binary
byte-identical, rebuild the probe and run the same holdout unchanged.
