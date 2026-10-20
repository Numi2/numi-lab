# Supported equilibrium and immutable operator work reuse 051

Follow-up to negative 050 CPU stop-unloading regression. Add an explicit
caller residual-increase budget, preserving the legacy stop-unloading policy
for callers that do not request the strict bound. The supported registered
pose requests 1e-12, leaving its no-worse-than-initial admission gate unchanged.

Keep all source assets/Float32 pose, support ownership, forces, tolerances,
budgets and accepted-state transaction unchanged. GPU optimizations: valid
ordinary FEM tangent hits defer unused base-state geometry; stateless generic
cache columns validate common stress once in column zero and require all
nine validity bits; nontransport nodes skip gathering exact-zero field
actions but preserve boundary action. Stateful/mixed/direct fallback unchanged.

Run the existing CPU support regression (strict preservation and legacy
redundant unloading), Metal material-frame serial/parallel cache exactness,
and nonzero monolithic field test. One .01 activation-bound root must match
the frozen048 first physical CSV row exactly. Compare its physical wall
time with050's 15.103505375-second control root, label single-root timing
as bounded evidence only. Then cap1.0 8-root native candidate plus2-root
late-reject/retry, preserving exact accepted-row replay and reporting actual
activation distribution and all numerical gates. Source-compiled balance
is not a proof of full FEM equilibrium or sustained resting physiology.

After passing, at most one1ms root tests larger cadence, stop on failure.
No relaxed gates/budgets. Optional profiling uses existing counters only.
No visual verification; one native/build/large CPU job at a time on SSH Numi.
