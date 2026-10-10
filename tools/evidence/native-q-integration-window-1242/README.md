# Bounded accepted-state diagnostics

A requested Q audit window now splits submissions only inside its declared physical-step interval and returns to the ordinary observation grid afterward. Physiological and COM traces retain their normal cadence; per-step Q and support-slip diagnostics have separate files. The option is disabled by default. The first/last bounds must be provided together and lie within the requested horizon.

A window ending at the horizon exposed a terminal tendon verifier bug: it inferred the last submission length from the ordinary cap, although the diagnostic scheduler had submitted one step. The verifier now records the actual accepted submission length. Transfer/correction byte equality, counters, status, and failure checks remain enforced.

Validation on the SSH Mac mini used the original flat-bed ec5664/1cd0c3 scene, 64 contact iterations, release initialization, and rigid hands. No support or physiology forces changed. These are instrumentation checks, not a completed anatomy or physiology qualification.

- The scheduler and accepted-geometry cadence C++ tests passed with release flags; their source hashes are retained in the CPU test report.
- A 512-step control and middle window (173–221) passed on the main-base build.
- Before the tendon fix, a 512-step terminal window (464–512) reproduced the terminal failure.
- With the fix, the 512-step control and terminal window both completed. All normal Q/V/compensated-root fingerprints and non-submission physiological fields matched; support/surface rows and captures at steps 0 and 512 matched exactly. Only the explicitly reported submission-scope diagnostics differed. The full report retains every differing column.
- The earlier 155,000-step diagnostic replay reached 310 simulated seconds but failed the terminal verifier. It remains failed evidence: no final MRVPACK or successful completion footer. Its 19,375 accepted physical-state fingerprints and physiological observations matched the unchanged baseline, and all 620,000 support rows matched. The last surface row is absent. Separate failed-run accounting covers the retained 2,500 Q rows without promoting that run to successful publication.

The late diagnostic reconstructs all 305,000 scalar updates using fused Float32 arithmetic, and attributes root translation to recorded velocity integration. It does not explain why those velocities developed, prove force/energy closure, admit remaining skin crossings, or establish clinical validity. No successful long replay of the fixed binary is claimed here.

The evidence map binds compact reports to retained files and source hashes. Execution receipts contain exact launch argv, declared inputs, and post-run pin checks. Full traces, native movies, geometry packs, declarations, and build logs remain at those Mac-mini paths. Analysis scripts retain their original absolute input bindings. The main-base runtime differs from the frozen runtime used for the failed long diagnostic; the reports do not claim cross-runtime equivalence.
