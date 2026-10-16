# Audit attempt note

The first independent audit run is retained in `audit.stderr.log`. It verified all 1,350 distinct staged Git blobs against the staged path inventory before stopping at the qualification metadata check. That first script incorrectly expected `physical_validation` on `summary.json`; the field is correctly in `publication-binding-reference029.json`. No payload verification had run at that point. The corrected `audit.py` reads the binding field and completed the full audit; the initial failure remains visible rather than being overwritten.
