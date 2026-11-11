# Diagnostic stop: incorrect body index

The toe drive body and joint mapping passed a standalone check, and all 28,440 available q/v/action samples matched the baseline even after perch contacts. Inspection found that the candidate compared the scene-local course index directly to a full-world contact body index. The process was deliberately stopped once the inactive sensor was understood (exit -15); this is partial evidence, with no completed first-episode or release claim. The corrected world-index candidate is separate.
