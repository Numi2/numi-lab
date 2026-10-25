# Current 294 detailed-surface displacement prolongation

This is an executable, diagnostic-only mapping from the O-layer displacement in connected-volume-continuation-039 iteration 005 to the current detailed 294 skin surface. It uses the exact D36 direct source-vertex and barycentric face anchors, inverse-edge-length graph-harmonic interpolation, and a fixed collar. It does not rewrite or emit an NHSKIN payload.

The latest run exactly reproduces 2,332 direct displacement anchors and 79 barycentric anchors and has zero nonzero displacement outside its 20,804-face support. The D36-to-AP placement check is within 0.124 micrometres. The detailed-surface correspondence is poor: direct-position residual is at most 2.946 mm and barycentric residual is at most 11.598 mm. The prolongation maximum displacement is 23.134 mm; its minimum affected-face area ratio is 0.0578 and 531 affected faces have nonpositive old/new normal dot. Those normal-dot values are diagnostic warnings, not by themselves proof of inversion. Exact self/collision, bone/tissue, layer, volume, mass, native runtime, and physical gates were not run.

Run from this directory with:
 /Users/n/numi-human-prep-venv-20261005/bin/python -m unittest -v
then:
 /Users/n/numi-human-prep-venv-20261005/bin/python build_current294_prolongation.py

The as-run result is in run-002/. run-001/ is retained as the earlier identical numerical result before implementation source hashes were embedded in the report. Inputs and their absolute paths/hashes are pinned inside each run report.
