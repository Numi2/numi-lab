# Coupled private trial capture overlay 003

This diagnostic-only overlay is based on frozen source 094. It preserves the 32 MiB payload cap and 192-array cap; it adds bounded per-spec extent details to the existing skip record. The failure path still latches capture failure and returns before allocating the snapshot or issuing a blit. It does not change solver behavior, acceptance, histories, or production buffers.

## Diagnosed source 310 failure

Native capture-on run 310 and capture-off run 311 used the same frozen source 094 and were both rejected. Their non-log production files and 32 Armijo records were byte-identical. The 310 skip record was generic and reported `candidate_before_private_contact_rebuild`, `bytes_already_planned=0`, with no buffer detail. That failure is preserved; no corrected native capture has run yet.

The bounded detail added here reports each invalid present spec (at most 192 entries / 128 KiB): name, declared dtype, source byte length, element byte size, divisibility, source and target device IDs, phase bytes before and after, already planned payload bytes, and every failed extent predicate. It also reports total invalid count and metadata truncation.

An audit against the retained 291 full capture and source 094 allocation declarations found two trial-capture schema mistakes:

- `fem_topology_nodes_candidate` is allocated from `NMFEMTopologyNodeGPU` (16 bytes). The trial context mislabeled it `NMFEMNodeStateGPU` (80 bytes). In capture 291 the topology array is 36 × 16 = 576 bytes, which is not divisible by 80 and explains the first pre-capture rejection.
- `coupled_inverse_statuses` is allocated as `MRInverseMassStatusGPU` (48 bytes). The trial context mislabeled it `NMMatterStatusGPU` (also 48 bytes), which would not cause an extent failure but would misdescribe the captured records. This overlay corrects it. The older complete-state capture labels those 96 bytes as 24 `uint32` values; that legacy metadata label is recorded in the crosswalk and is not treated as proof of the C++ record type.

The other 26 literal trial-context names that overlap capture 291 agree with the retained dtype/element-size map. `source291_capture_crosswalk.json` binds the exact retained manifest and package hashes and records these comparisons. Capture 291 is supporting schema evidence, not a replacement for rerunning the failing 310 selection.

## Verification and limits

`check_trial_capture_overlay.py` passes 98 source/crosswalk assertions. Objective-C++ syntax-only compilation of the overlay `runtime.mm` passes against the source-094 compile command. These checks verify the diagnostic edits and schema crosswalk only. A fresh native capture is still required to confirm that no additional bad spec remains; neither 310 nor 311 is a successful solver step or physical qualification.
