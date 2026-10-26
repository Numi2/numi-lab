# Whole-face projected objective 429

This is an opt-in row-builder replacement for the face part of frozen 408.
It captures and calls the original 408 builder, keeps its point rows, and
replaces barycentric face rows with positive per-corner projected deficits
toward the same 335 endpoint plane.

Use replacement_for(module.build_clearance_rows) only for a selected run.
These are optimization targets only. They do not certify an endpoint, a whole
triangle, or a complete surface outside any bone bank. Keep exact audits and
admission gates.

Test on numi:
    /Users/n/numi-human-prep-venv-20261005/bin/python test_whole_face_objective_429.py
