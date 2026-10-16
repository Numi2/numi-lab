# Independent publication029 staged-byte audit

This is a read-only audit of the staged publication at repository HEAD `a5ccffa52ffe368b49c877094a5081b7ee246ebf`. It does not depend on `publication029-payload-staging`: source inputs are read from their original external case paths, and published bytes are checked at their repository destinations and against the Git index through `git cat-file --batch`.

## Result

The final audit passes. The declared 2,841 staged path list is byte-identical to `git diff --cached --name-only -z`. All 1,350 distinct stage-0 Git blob objects, covering 176,838,556 staged bytes with duplicate path contents counted, match their worktree files byte-for-byte; no staged mode mismatch was found. All 802 source168 files match the frozen source manifest and working tree.

Publication029 appends exactly 44 records to the 1,480-record summary-reference028 prefix, producing 1,524 current records. The 5,566-entry reference028 NUL pathspec is an exact byte prefix of the current 8,376-entry pathspec. All 2,810 added pathspec entries are staged. For the 2,808 new payload entries, all 2,750 raw copies match their external source size and SHA-256; all 58 gzip copies match their compressed hashes and decompress to the original size and SHA-256. Each published payload path is staged.

The current summary and publication binding hashes match. The six verdicts remain open: solver correctness is partial component evidence with full nonlinear solve unqualified; numerical convergence is unqualified; component calibration is partial source-data only; analogue validation and physical assembly are unqualified; crush/impact is unexecuted. `physical_validation` remains false. The active native470 run is absent from the new-case list and staged artifact paths. The two untracked work areas remain unstaged.

The retained warm-start case is the actual v2 checker execution: its receipt binds v2 manifest `29043b640b6fde5fc66c864ff84d80400cb09771fb5e455368afdd7ae76c7f8d`, records GPU execution with return code 1 and zero physical transactions, and stderr reports `declined refinement cleared original certificate`. This accurately records a checker failure; it does not claim solver qualification.

## Documentation review

The current top-level summaries and source167 milestone state that native468/469 failed, derivative misses remain, no complete coupon or shipping qualification is established, and all six axes remain distinct. The newer milestone summary does not overstate physical validation. One editorial inconsistency remains: both long ledgers retain lower sections titled “Current milestone: source142/source139/…” beneath their newer source167/source168 opening summary. Those historical sections carry scoped failure/open-gate caveats, but the headings should eventually be renamed to historical milestones to avoid ambiguity about which revision is current.

## Preserved initial audit attempt

The first audit attempt is retained in `audit.stderr.log`; it stopped after the Git blob check because the draft incorrectly looked for `physical_validation` in `summary.json` instead of the publication binding. It had not yet checked payloads. `audit.py` was corrected to read the binding, and the final complete run is `audit-final.stdout.log` / `audit-final.stderr.log`. The initial failure is documented in `initial-attempt-note.md` rather than being hidden.

`audit.py` and `payload-audit.jsonl` contain the reproducible byte-level checks and per-payload hashes. `manifest.json` binds the audit outputs and source records. No repository file, staged entry, commit, build, or GPU job was changed or launched by this audit.
