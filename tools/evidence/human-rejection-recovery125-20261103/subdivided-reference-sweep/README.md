# Bounded compensated CCD refinement

A new GPU fallback certifies an unresolved moving feature by splitting its affine sweep
into at most 16 dyadic intervals (31 visits, depth four). Each accepted interval uses a
fixed axis and outward-rounded projection bounds of the ORIGINAL compensated endpoints.
Interpolated points only choose candidate axes; they are never proof inputs. Every leaf
must prove strict separation beyond the complete, unchanged activation thickness. An
unresolved leaf, within-band contact, true crossing, nonfinite input or capacity exhaustion
remains rejected. Legacy-certified results and all physical residual/contact/rollback gates
are unchanged. Work is local and bounded on the GPU, with no new host readback or state.

The focused test first failed with the original shader (case 34), then passed all 54 cases
and canaries after the change. Controls cover VT/EE/point contacts, translated compensated
coordinates, 1000x scale changes, reversed sweeps, penetration and nonfinite input.
The full native solver check passed 811,095 checks on SSH Numi, Apple M6 / 16 GiB.

The regression includes rounded endpoint arguments retained from 017, explicitly NOT its
missing compensated low words. Float64 distance sampling is diagnostic, not a sweep proof.
Only the production outward interval predicates certify the full represented trajectory.

Integrated trial 018, identical cavity batch and physical parameters to 017:
**5 accepted steps (78.125 microseconds), then failure**. It passes the prior root-3 VTCCD
rejection. It stops at root 5 after all 16 Newton corrections with vascular residual
1.08423055e-5 above the unchanged 1e-5 limit (row 0). Skin/fat minimum J at rejection:
0.829614401. This is not sustained motion, supported rest or >=0.1x qualification.
The older unconditioned topology retained 8 accepted steps in the matched earlier setup.
The cavity batch remains unpromoted as the accepted scene.

Per-step accepted GPU times and compact initial-row parity are in comparison.json. There
is no repeated-pair performance qualification; cold startup and instrumentation matter.
The next controlled trial raises only the bounded Newton budget from 16 to 24 and selects
root-5 traces to distinguish incomplete convergence from geometric constraint saturation.
Higher budget is additional work, not an optimization claim.

Plans, exact source/binary pins, source archive, logs, compact physiology, failure and
resource monitor are retained. The sources/binary were not changed during the trial.
No image/visual verification was performed.
