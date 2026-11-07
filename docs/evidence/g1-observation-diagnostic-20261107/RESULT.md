# Native delayed-action observation fault isolation

The preregistered historical-owner probe completed on Apple M6 without a
reported physical failure or task termination, and returned exit 1 because
528 of 528 sampled delay-one actor observations differed from the known
nonzero actions. All 528 observed values were zero. The probe checked twelve
leg actions across eight environments at steps 3–4, then seven unaffected
environments at steps 5–8 while environment 1 received a requested reset at
step 5. The reset environment after that request was intentionally excluded
from the value oracle, so this does not qualify its reset-history semantics.

The native task matched all four retained policy-contract fingerprints; the
probe used externally supplied actions, not the learned policy. The complete
per-observation table, requested input formula, source, command, exit and
executable, dynamic-library and metallib hashes are retained here. The result
isolates the owner-path symptom and agrees with the retained policy data and
the unsigned-index CPU oracle. It does not establish that this one expression
is the only cause, nor that the policy's falls resulted from it.

The seven other environments continued to show the same mismatch after the
mixed reset; this does not explain the separate finite-difference anomaly in
the longer policy trace. A correction must test normal history and reset
history, and must address compatibility of previously compiled policy
fingerprints whose observation semantics would change.
