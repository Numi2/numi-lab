# Crow evidence recovery and space retirement

The manifests list exact local evidence candidates and independently verified remote bytes at the immutable Git commit recorded in each line. A `verified` record identifies a match; consult subsequent retirement receipts before assuming that a local copy was actually removed. Skipped records were not eligible for retirement. This is data preservation, not a research qualification.

Restore into a chosen empty destination, without overwriting existing files:

```
python3 restore.py --manifest batch1-verified.jsonl --destination /path/to/restore --path artifacts/path/from/manifest
```

Omit `--path` to restore all verified entries. Supply `GITHUB_TOKEN` (or an authenticated `gh` CLI) for the GitHub archive. The restorer checks the exact SHA-256 and byte count, preserves recorded file mode and modification time, and refuses overwrite. Files were selected only after tracked/untracked, path, link and open-handle checks. Recorded `git_blob` is the Git object identifier. For tracked files retired under a skip-worktree flag, after restoring the exact file in the original checkout, `git update-index --no-skip-worktree -- <path>` makes the index monitor it normally; do not clear unrelated flags. The archive retains earlier failures as well as successful results.
