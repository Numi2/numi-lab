# Rejected registration attempt

The first call to Numi Lab's v2 `register` command rejected this draft because its arm command referenced the dedicated build's loaded metallib paths without binding those absolute paths as artifacts. No trial ran and no observation was collected. The final plan adds and hashes the loaded library paths alongside the frozen copies, then registered with verified integrity. The rejected draft is retained unchanged for provenance.
