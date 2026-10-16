# Source160-002 read-only implementation review

Reviewed source package:

- `selector5-restart-consumer-capture-source160-002`
- manifest SHA-256 `0363e4cd3ea4fbc014127eb4e51c3ab16177d091687c7a1cfa2bdec727ff7a36`
- runtime `ab916ed67b669769bf8bbb8e10974782f62c1245f23434d69ae9f9d45aef6748`
- header `f97bdbd1e6729861d1f232f6262ef08c405bbe0410a7c5172add180044c662e9`

The absent-variable path creates only a disabled request and no capture
encoder, snapshot, or copy. The active path validates paired selector/path
settings, exact unsigned selector syntax, selector5 eligibility, and the
zero-rate single-microtick scope in `source/matter/src/runtime.mm:11559-11627`.
At the selected event, the one-shot claim gates all capture work; a memory
barrier follows graph apply before readback (`runtime.mm:21622-21654`). Each
array has checked shape/byte sizing, source-range bounds, and aggregate-cap
admission (`runtime.mm:21662-21723`). The payload copies only the selected
environment and retains the graph original matrices and LU, current vectors,
material/cache state, augmented coarse state, and generation tags
(`runtime.mm:21725-21968`). The bounded shared snapshot is copied in the same
command buffer and files are written only after successful command-buffer
completion (`runtime.mm:21970-22100`). Final files are created with `O_EXCL`
and partial files are removed on writer error (`runtime.mm:1305-1339`).

The header's checked aggregate limit is 64 MiB and the selector parser and
array arithmetic use checked uint32/uint64 bounds
(`regional_consumer_capture.hpp:13-14, 25-35, 113-139`). The captured
source464 dimensions imply a 39,825,484-byte one-environment payload, including
37,770,696 bytes for graph original matrices plus LU. It fits the 67,108,864
byte payload cap by 27,283,380 bytes; it would not fit the former 32 MiB limit.
The underlying source464 metadata SHA is recorded in
`preflight/source464-size-preflight.json`. This is a size-only estimate; it
does not establish device allocation success, full-process peak memory, or
native behavior.

No default solver path or physical update is modified. The only active capture
effects are the scoped buffer snapshot/blit and completion-handler file
write. This review found no source-level blocker to the advertised bounded
readback, with native execution remaining unverified.
