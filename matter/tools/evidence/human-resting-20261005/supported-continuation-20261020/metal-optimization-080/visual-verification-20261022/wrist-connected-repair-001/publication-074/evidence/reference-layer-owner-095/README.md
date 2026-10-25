# Reference layer owner audit 095

Status: read-only source audit; not physical/native qualification.

D28 uses per-row construction fields over 2,411 rows: skin 0.5002-2.4998 mm and fat 0.5051-9.9338 mm. The generic layer default is 10 mm fat with 2-20 mm bounds, described as an inferred reference rather than a measurement.

For 75 selected head rows where D was classified inside, authored rest I-D spacing has a 9.9006 mm median; exact 039 current emitted positions give 10.2565 mm. O-I is about 2.4508 mm. The current statistic is posed spacing, not authored thickness, and does not establish an anatomy or registration error. Across head, arm-bone donor IDs 41/42/43/91/92/93, and hands, current I-D medians are 9.4910, 9.4743, and 3.7512 mm. The arm-bone grouping includes humerus, ulna, and radius; it is not a forearm-only sample.

Per-row thickness is supported in a new reference candidate, but requires regenerated reference coordinates/topology. Preserve compiled native parent-cell mass by same-material overlap accounting and typed donor moments; do not globally rescale.

The prior 073 payload hash 4421ee... is a legacy failed transfer and is not the final result. Use final P073 parent-local partition receipt partition-receipt-manifest-004.json (SHA-256 67e16865e40e6b30d6534a0f9a9c8e6f8f13ea543aa323c9d539d30c6c8785de), payload SHA-256 300e435da66fa372566e3c394446e7b7e409be064e943ebd4dd7bd795291933a, and report SHA-256 cc68137c4b6f2c854ebf8003aa90df60a0c30412fb18f8710c502129648f9447. This is offline accounting, not clearance or physical qualification.

Reproduce source field, geometry spacing, regional summaries, and compiled cell masses with:
python3 /Users/n/Work/numi-human-connected-surface-069/D/reference-layer-owner-095/reproduce_095.py

The script validates all five pinned inputs and writes reproduction-output.json. Complete pins and calculations are in analysis.json.
