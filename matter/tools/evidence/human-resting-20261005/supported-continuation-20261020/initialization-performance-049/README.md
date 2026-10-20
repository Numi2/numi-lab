# Frozen initialization and GPU profile 049

All four runs execute the precompiled 084f0c9a baseline binary, SHA-256
707fd05b6aa3d030cdac912bb362a89179962ee123cc3e4224f7922b2921df04,
on Apple M6 over SSH Numi. They use the fresh 048 supported Float32 pose.
Source edits made while these measurements ran are not in that binary;
the exact executable/metallib/library identities in each invocation are
authoritative. No visual inspection was performed.

The first selected-kernel profile overflowed the instrument (290 dropped
counter samples), retained as a negative instrumentation attempt. Repeating
at stride 64 avoided overflow and preserved the exact first accepted coupled
CSV row from ordinary 048. The stride-64 timing means are samples, not complete
kernel totals. In particular a single tangent-build sample must not be
extrapolated as a reliable total.

A follow-up measured all 96 tangent-cache builds and all 100 contact sorts
with no dropped samples, again exact first-row parity. Tangent-cache builds
sum to 4.747243003 seconds; 59 exceed 1 ms and 37 are short converged tails.
Contact sorting sums to 0.728123996 seconds. The sampled transport node
operator averaged 109845.7 ns over 385 of 24676 dispatches. This localizes
work; the profiled run is not a production performance qualification.

Changing only the recruitment activation upper bound from .01 to 1.0 stopped
at initialization before any physical step. The optimized initial RMS was
about 0.000022, then optional stop-load refinement worsened it to .013707
(maximum acceleration .059287, root .005405, support 811.443483 N).
The unchanged app no-worse-than-initial gate rejected it. This exposes a
secondary-objective acceptance defect; it does not justify loosening the gate.
The .01 baseline RMS was 148.302898026 with maximum acceleration
556.614687329. Actual recruitment, not a prescribed uniform 1.0 activation,
was compared. No static-rest or whole-body qualification is claimed.

Child invocations retain exact commands, inputs, configuration, timing, logs
and compact output. Large packs/assets are hashed or referenced using the
existing evidence packaging; they are not copied into Git.
