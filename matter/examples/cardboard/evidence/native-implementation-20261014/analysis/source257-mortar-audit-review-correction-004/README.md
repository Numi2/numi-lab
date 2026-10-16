# Strict JSON correction for source257 review003

This addendum preserves the prior review package and its malformed report.json unchanged. The source file ends in the literal two-byte suffix backslash-n after the closing JSON brace; that causes strict JSON parsing to fail. This package removes only that terminal suffix, parses the unchanged JSON value, and serializes that same value as strict JSON with a final newline. It does not alter the audit findings, inputs, source bindings, or qualification status.

The old report SHA-256 is bound in manifest.json. replay_correction.py reproduces the correction and emits hashes. The original review003 remains the historical artifact; this corrected file is an additive machine-readable copy.
