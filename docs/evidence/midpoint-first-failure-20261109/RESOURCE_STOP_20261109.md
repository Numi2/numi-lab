# Free-space stops on the fine reference and control

On 9 November 2026 both long-running jobs reached their pre-existing free-space
guards. The n=128 fine FP64 reference exited by termination after 4313 seconds;
it had not emitted a completed trajectory. Its original 0.5 microsecond coarse
result remains valid, but the 0.25 microsecond result and the n=64 to n=128
spatial comparison are not qualified. See rk4-n128-retry1-h025-execution.json.

The 31.25 nanosecond backward Euler trajectory was stopped after 11033 seconds,
at approximately step 233997 of 320000. Its JSON is incomplete and its replay
was not run. The supervisor correctly declined to start coarser runs. The
partial trace and log have separate lossless gzip archives and SHA-256
manifests; their execution record reports the stop and failure to parse a
complete JSON object. The previously retained 5 ms partial diagnostic is not
a complete-run accuracy, replay, or cost result.

The observed abrupt free-space reduction coincided with building additional
local tests, but its allocation source has not been established. Existing
resource guards were not disabled. Preserve the partial data and do not label
a restarted attempt as a completion of this one. The two studies require new
attempts, enough free storage headroom and an uncontended machine for the
separate cost timing.
