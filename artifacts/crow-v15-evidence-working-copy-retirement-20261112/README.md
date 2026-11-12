# Crow research working-copy retirement

The listed 14 files were independently downloaded from GitHub at the recorded commit, verified as gzip streams, and matched byte-for-byte by SHA-256 against the macmini working copies and Git index before local working copies were removed. Their Git history and remote copies remain intact. Only the listed files were touched. Each tracked path was marked skip-worktree before removal.

To restore one in the owning checkout, replace PATH with its exact manifest path, then run `git update-index --no-skip-worktree -- PATH` followed by `git checkout-index -- PATH`. Alternatively, stream directly without restoring the working copy using `git show 4ab800743a06cc955774b1812849c720ae19d24f:PATH | gzip -dc`. Ensure adequate free space first.
