# Retained native attempt 350

This is the unmodified output from `analysis/private-ccd-metal-native-350`. The runner receipt records exit code 0 and reports four diagnostic cases, but `native-result.json` is malformed JSON: parsing fails on line 1, column 5580. The per-case serializer closed the `iterations` array and `instrumented` object but omitted the enclosing case-object brace before the next case. This attempt is retained as a failed report-serialization attempt and is not accepted as native diagnostic evidence.

The next package version fixes the close sequence and validates the complete JSON document with `NSJSONSerialization` before it writes output or returns success. It also exercises the exact serializer on four synthetic cases in a CPU-only mode.
