# Regression stopped before dispatch

The runner stopped at preflight without GPU dispatch because three relative script paths were resolved from the evidence directory rather than the repository root. The executable, shader and owner hashes passed; the script files could not be opened. No CCD or rollback result follows. The original manifest and failed check are retained; a separate absolute-path manifest is required.
