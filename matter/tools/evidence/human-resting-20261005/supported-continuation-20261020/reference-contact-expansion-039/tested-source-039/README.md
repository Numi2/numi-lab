# Captured 039 source delta

The patch captures tracked source/tool changes relative to 4642668b37f385ee7dde3d9fdde097fc0f5d9ab2, excluding evidence documents. It includes the frozen 039f source set and the tracked Python replay tool as a separately identified extra; it excludes modified evidence prose. No untracked file is included.

The patch passed git apply --check and apply in a temporary detached worktree at the recorded base. Reconstructed source SHA-256 values match the 039f preparation provenance for all 33 frozen source files and the 039f native runtime source set (24 entries, including unchanged files not listed in preparation); the additional tracked Python tool was also matched to the captured working file. See source-hashes.json and reconstruction-check.json.
