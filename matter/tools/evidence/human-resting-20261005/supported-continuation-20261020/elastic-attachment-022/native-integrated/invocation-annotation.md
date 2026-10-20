The historical launch of native-elastic-attachment-022 was recorded before the
reproduction helper began refreshing invocation.source_sha256. That one legacy
top-level field was inherited from the control template. The actual tested app
source SHA is in provenance.json.source_sha256 and the binary/metallib/library
hashes are refreshed there. Do not treat the inherited field as the trial source.
The helper now refreshes it for all subsequent runs. The original invocation is
retained unchanged, with this annotation, to avoid rewriting execution history.
