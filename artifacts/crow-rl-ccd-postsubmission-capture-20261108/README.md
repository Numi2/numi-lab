# Post-submission CCD capture rejected by replay admission

The optional bounded capture in source `f9c5108b` writes at most1MiB after a
failed native submission, with exclusive creation and no physics changes.
Full Release integration build and the generic native contact regression pass;
the successful control writes no capture.

One diagnostic execution uses the original unbiased policy, seeds and fixed
LR.0003/gamma.999/lambda1 recipe. It reproduces the previous native failure
exactly, all4canonical training artifacts, and all41model/Adam arrays and
metadata. No held-out evaluation or adaptive-rate trial runs. Raw evidence is
250,259,928B before its final summary; retained rollouts deduplicate against the
prior committed archive. This run is diagnosis of a known failure, not a retry
to obtain a successful learning outcome.

The capture is13,968B. An initial header assertion assumed the global failing
substep14 also appeared in contact status. The raw contact status still carries
13 because CCD may fail before narrowphase stamps the current pass; the world
transaction latch records current pass14. The original plan/script/failure are
preserved, with an explicit offline analysis revision and no native rerun.

More substantially, the standalone probe rejects the snapshot before encoding
GPU work: `Reported failed candidate absent`. Its active candidate prefix has
two records for pair108; the failed pair100 is absent. The event cursors have
already advanced to the final encoded substep. Later statically encoded passes
reuse transient buffers after the first error. Consequently this post-submission
scratch **is not admitted as the failed sweep**. The prospective replay
prediction is contradicted. No query accuracy, CCD repair or causal failure
mechanism is inferred from these stale buffers.

`manifest.json` preserves all raw evidence and helper sources with compressed
Git content deduplication. `runtime-recovery.json` binds exact native runtimes,
the original coupled checkpoint10 and the rejected capture. The restored
standalone probe must repeat the same pre-GPU rejection, and all restored
inputs/checkpoint components must hash-match, before temporary and raw-copy
cleanup. This is scoped negative-measurement reproduction and byte recovery;
it does not claim an independently executed full learner recovery for this
diagnostic archive.

Next capture the first unsafe unresolved pair **inside the GPU CCD resolver**,
including both source shapes, current/future paired projections and the event
cursor, into storage protected from later passes. Require exact isolated
candidate replay before an independent advancement/event regression. Preserve
normal physics, actor inputs, rewards and learning rate. Broader learned food
manipulation, ecology and social learning remain unfinished.
