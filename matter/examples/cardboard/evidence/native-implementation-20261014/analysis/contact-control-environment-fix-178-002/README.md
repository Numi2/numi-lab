# Bind contact incidence sorting in native run contracts

The native runtime already reads NM_CONTACT_NODE_INCIDENCE_SORT, but the source-bound launcher previously tracked and scrubbed only NM_FEM controls and timing. This patch adds the existing contact selector to the explicit absent/set contract, scrubs all inherited NM_CONTACT keys, validates the exact supported value1, and expands source-registry discovery. Unknown contact overrides are rejected. No operator, capacity or tolerance changes.

Contracts missing the newly required key are rejected by the updated reader; old records remain intact and must be interpreted with their archived source-bound reader. An omitted historical key is never invented as absent. Existing source173 runs are not relabeled.
