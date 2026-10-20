# Rejected global stop-residual policy and first reuse control050

The first strict global stop-load residual guard failed the existing CPU
regression: it prevented the redundant two-muscle fixture from unloading its
analytic joint stop. The failure is retained in checks.log; this first guard
is not the accepted final policy. The follow-up051 introduces an explicit
caller-selected bound while preserving the existing default unloading policy.
The first test batch also could not run the monolithic field check because
its target was not yet built; building it and rerunning passed. Material-frame
Metal serial/parallel/cache/direct checks passed.

One .01-activation-bound 125microsecond native control completed with the
first geometry-cache fast path and zero-world-material field guard. Every
first coupled CSV field matches048 exactly. Physical segment wall time was
15.103505375s (RTF8.2762248123e-6); full launch wall26.793395917s. This is one
root, not a steady performance estimate or an improvement claim. In particular
the zero-world-material guard does not skip nontransport objects when another
object in the same world has transport materials. Follow-up051 addresses
that measured nodal-gather overhead and stateless repeated stress work.

No cap1/cadence qualification was run with this rejected policy. Source patch
and exact invocation/binary/library/metallib identities bind this short test.
No visual verification or full resting-body qualification.
