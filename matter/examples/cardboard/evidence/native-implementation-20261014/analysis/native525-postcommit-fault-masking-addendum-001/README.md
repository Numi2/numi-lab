# Native 525 postCommit negative-control addendum

This read-only addendum qualifies the cause of the third transaction's failure without changing the prior byte-audit package.

The fixture's third step requests a postCommit motion-type fault. The source first builds the original endpoint for preDynamics, then supplies the modified endpoint at postCommit. The negative-control check only asserts that the final status is unsuccessful and that rollback/certificate clearing occurs.

The rejected trace shows an earlier failure: Newton 18 ends with solver status 10 at failing index 18 and trial_outcome no_trial_accepted_status_clear. Newton 19 onward is blocked by that status. The accepted cursor and accepted-state vector hashes roll back correctly, but that result is attributable to the earlier solver failure; the intended postCommit-specific failure is not isolated by this run.

Exact source anchors and input hashes are in report.json. The previous native525 byte audit remains unchanged; this addendum narrows only the interpretation of its negative-control coverage.
