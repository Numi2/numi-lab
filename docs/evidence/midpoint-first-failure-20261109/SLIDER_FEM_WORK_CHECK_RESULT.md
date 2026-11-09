# FEM-side interface-work instrument check

The separately fingerprinted, default-disabled native diagnostic accepted one
step at each preregistered initial gap. At 200 µm the normal impulse and FEM
contact work were zero. At 95 µm the FEM normal impulse was
3.8390845987e-6 N s. The FEM work reconstructed from the accepted contact
samples was 3.4481306022e-9 J; the independent FP64 pressure-law reconstruction
was 3.4481305734e-9 J. The largest per-node difference was 0.005812 of its
existing impulse-derived work bound. Both instrumentation checks passed.

The plan, exact build and input hashes, commands, and complete per-step outputs
are retained in `SLIDER_FEM_WORK_CHECK_PLAN.md`,
`slider-fem-work-check-001-execution.json`, and
`slider-fem-work-{far,near}-001.json`. The test also rejects a deliberately
corrupted FEM work gate. These are two one-step checks, not a complete
reacting-body rollout, proof of spatial convergence, an energy-accuracy
qualification, or a performance result. Historical traces without this new
per-node field are not silently promoted by the new analyzer.


The initial instrument check recorded a maximum ratio but the complete-rollout
analyzer did not actually apply it. The follow-up records all per-node impulse
and velocity terms and independently reconstructs both work sums and the
gate in the analyzer. The same two one-step native cases passed again; the
maximum near-contact ratio remained 0.005812. Missing or corrupted per-node
terms are rejected by the analyzer test. See the separately retained
`slider-fem-work-check-002-execution.json` and its corresponding outputs.
This repair does not upgrade either one-step test to a complete rollout.

The updated diagnostic also passed forced late rejection and exact retry both
before contact and after a nonzero contact. The failure category remained 11
and both the rigid state and Matter physical state were restored. See
`slider-fem-work-rollback-002-execution.json`. This remains a short retry
check, not complete-trajectory qualification.


A separate retrospective impulse check of these two accepted steps found that
on the 95 µm contact step, the measured body-wrench impulse differed from the
opposite FEM impulse by 0.00616 of the existing bound, and the independently
inferred rigid momentum impulse differed by 0.223 of that bound. On the 200 µm
pre-contact step, the measured contact impulses were zero but the independently
inferred momentum impulse was 4.692e-12 N s, exceeding the existing 1e-12 N s
bound by a factor of 4.692. Thus that accepted step does not pass the full
independent impulse gate. This is consistent with the retained FP32
representability diagnosis; it does not establish an exact cause for every
failure. See `slider-work-impulse-retrospective-001.json`.

The full-rollout analyzer now separately requires both the measured wrench
impulse and the momentum-derived impulse to oppose the FEM impulse under the
unchanged bound. Previously the measured-wrench comparison was indirect and
could be ineffective when the interface velocity was zero. A synthetic
perturbation of the wrench impulse is rejected by the analyzer. Neither
one-step result is a complete reacting-body qualification.
