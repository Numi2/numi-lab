# Native vascular row-23 026

Execution exited 1 after 378.073 s. The output CSV contains 20 accepted coupled rows through 2.5 ms (21 × 125 µs requested; the final step did not complete). This is a short mechanics-only replay, not a sustained-contact or physiological qualification.

The run stopped at segment/root 20 with matter code 10, failing index 23 and 3 microsteps. Its matter code, object sentinel, failing index, microstep count, and four diagnostics exactly match native-contact-precision-025. The full 20-row coupled CSV is byte-identical to that prior run. The exact 49-line raw vascular trace for root 20, microtick 3 is retained separately; the complete 1.36 MB stderr is also preserved.

All 13 file assets explicitly present in the invocation were rehashed and match the 025 manifest exactly; no input assets or large result pack were copied. The invocation and provenance bind source revision, source files, metallib and suite library. They both report executable SHA-256 83bf98bee369a57a9dd19e74a6aea343cc767486e189d0ff6a2aa6c8982e5b2e. The executable path was rebuilt after this run; its current bytes hash 37d11b2888dcd4413cc22052f284d26b3f36bce971989c4112c1a2f524e82867, so the run-time executable bytes themselves were not independently retained or rehashed after execution.
