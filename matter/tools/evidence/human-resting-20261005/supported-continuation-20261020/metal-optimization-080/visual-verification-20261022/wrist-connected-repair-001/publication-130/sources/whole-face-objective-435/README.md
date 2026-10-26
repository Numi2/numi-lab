# Face/bone grouped exit-plane objective 435

This opt-in objective groups certified 335 target records by (skin face row, bank body, stable bone ID) and selects one existing certified endpoint plane per group. Selection minimizes the largest positive corner deficit, with the original 335 source index as the deterministic tie-break. It emits positive per-corner projected rows for only that selected plane.

The helper preserves the complete point-row matrix, right-hand sides, weights, and metadata emitted by frozen 408. Face weights retain the existing point/face split. Selected planes remain proposal objectives only: they certify neither the whole triangle nor all surface clearance. The 391 unique face/bone groups with all three vertices outside still have an interior crossing witness; they need a face-level or subdivision/remesh repair decision.

The 427 audit-159 and audit-130 evidence are distinct: audit-159 reports 3,239 skin/bone triangle crossing pairs; audit-130's zero PLC triangle-pair count is from the skin self-audit. Its saved point-by-bank rows show 391/1,747 unique face/bone groups have all three corners outside the implicated bank and 1,356 are mixed. These groups can have an interior barycentric crossing despite outside corners.

Do not treat this helper as an endpoint or admission result. Keep exact all-bank surface and point audits mandatory.

Run on numi with the configured prep environment:
 /Users/n/numi-human-prep-venv-20261005/bin/python test_face_bone_grouped_objective_435.py
