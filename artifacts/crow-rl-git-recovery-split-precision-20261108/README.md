# Executed recovery: native split-precision experiment

All 82 original raw files were recovered from actual committed Git objects.
The restored Metal binary/library and compiled FP32 inputs executed all
fourteen native submissions. Both scientific reports match exactly, excluding
only result.json's administrative raw-byte total, which depends on the
replay workspace contents. All 26 native binary outputs match byte-for-byte.
All original raw bytes remain exact after execution.

The 163-file, 1,608,129-byte temporary workspace is separately inventoried
with source hash, size, mode and nanosecond mtime. Every unique content is
either referenced from the original actual-Git payload or stored in a
content-addressed recovery gzip. Cleanup requires remote HEAD parity,
actual Git readback, exact unchanged local files and zero open handles
after the executor has exited.

The replay repeats existing numerical evidence. It does not advance the
world or qualify physical crow behavior. Production sources, current native
runtime, protected deterministic feeding policy and the full goal remain
unchanged. See final-verification.json for closure after cleanup.
