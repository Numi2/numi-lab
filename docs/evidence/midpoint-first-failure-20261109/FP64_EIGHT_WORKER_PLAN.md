# Ordered eight-worker reference parity

The ongoing four-worker 128-division trial remains unchanged. To make a later
independent reference less costly, evaluate the same ordered per-element and
per-node calculations using eight workers. Worker count changes only which
independent elements or nodes a worker owns; each node still sums its ordered
incidence list in the same sequence. This is a prediction to verify, not an
assumption. Require byte-identical complete outputs against the retained
one-worker references at n=2 and n=8 with 0.25 microsecond steps, and n=32
with 0.5 microsecond steps, plus the energy-gradient self-test. Any failure
rules out using the eight-worker executable for the follow-up. Capture the
binary and source hashes. Timing while the four-worker reference is running
is observational and cannot support a native GPU performance claim. A
successful parity check alone does not establish spatial convergence.
