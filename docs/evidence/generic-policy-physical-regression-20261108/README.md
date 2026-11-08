# Generic policy selection: physical-failure regression

A synthetic WorldPack and URDF candidate had a higher reward and one contact
termination among sixteen environments versus none for the incumbent. Before
the change the selector published the candidate with a positive score while
also recording "termination rate increased" as a regression. The retained
before.json and red.txt show this contradiction. The new rule requires the
reported physical-failure rate not to rise before reward or tracking can
advance an imported-task candidate. The selector still keeps the candidate
artifact. Rejected physics steps remain an independent veto.

Reason 7 is the native food-consumed success reason, not a physical failure.
The prior counterexample and failing test are retained; the revised
classification excludes that reason as well as ordinary timeout reason 4.
This assumes the native termination-reason ABI and does not prove the
classification of arbitrary custom-task semantics or absent/malformed data.

The focused tests exercise both imported sources, an ordinary timeout, known
success, and the actual selector command with a synthetic evaluator and policy
files, including which bytes are copied to deployment. The full suite ran 46
tests and retained one pre-existing guided G1 reset-only assertion failure.
All 45 remaining tests passed; that exclusion is explicit in the output.
These are selector/CLI checks, not native physical-policy qualification.
