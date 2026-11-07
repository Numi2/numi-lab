# Guided auto-load binding

The guided auto-load decision now uses the same selected-policy and evidence-hash binding and held-out balance checks as the catalog. Missing bindings, altered policy or evidence, and a failed screen do not auto-load; the message distinguishes failed from unavailable evidence. Manual selection of research policies remains available. CPU fixtures exercise selected incumbent and candidate, missing metadata, changed policy, changed evidence, failed screen, and a passing bound screen.

As an additional read-only check, the retained integrated installed smoke's bound but failed incumbent and the 200-update trial's older unbound candidate both remain unloaded. Their output and policy hashes are recorded here. This uses the shared decision function directly and does not establish a visible UI or native GPU rollout.
