# Lab integration, tooling, and historical engine work

Numi Lab owns this record of application, tooling, release, robot integration, and historical engine work. `numisolver` is the default engine branch and `main` is a live engine and integration branch. Both continue as branches. The old `coupled` name resolves to a compatibility tag at its final integration revision.

This handoff indexes 25 former Numi Lab branch heads and imports 148 byte-preserved source and evidence files. The snapshots are historical research inputs stored under `docs` or `Docs`. Reproducing a result requires the original source revision, hardware, inputs, and acceptance criteria recorded in its history. The owning repository’s live build keeps its existing source and targets.

## Research repositories

- [Brain and Human bridge research](https://github.com/Numi2/numi-brain/tree/main/docs/NumiLabMigration) covers 2 former heads.
- [Human mechanics, joint research, and evidence](https://github.com/Numi2/numilab-human/tree/main/Docs/NumiLabMigration) covers 36 former heads.
- [Matter, contact, and suture research](https://github.com/Numi2/coupled-physics-solver/tree/main/docs/NumiLabMigration) covers 52 former heads.
- [Rod and mechanical solver research](https://github.com/Numi2/numi-solver/tree/main/docs/NumiLabMigration) covers 8 former heads.
- [BirdFlowMetal Crow and Dove handoff](https://github.com/Numi2/numi-gpu-physics-pigeon-sim/tree/main/Integrations/NumiLab) covers the six branches from the earlier migration.

The branch ledgers in these repositories record the original Numi Lab refs and recovery tags.

## Branch and file records

- [Branch ledger](branches.tsv) records the original branch name, tip, root tree, archive tag, comparison base, and selected snapshot paths.
- [Changed paths](changed-paths.tsv.gz) lists every path changed against the ledger’s comparison base, including deletions and original Git object IDs. Branches that share their tip with their comparison base have an empty delta, and branches without a shared base are compared with an empty tree.
- [Source manifest](source-files.tsv) and [SHA-256 checksums](source-files.sha256) record the original revision, source path, Git mode, blob ID, and content of each imported file. From this directory, run `shasum -a 256 -c source-files.sha256` to verify the imports.

## Restore a branch

The complete Git history and bulk artifacts remain available from the annotated archive tags in [Numi Lab](https://github.com/Numi2/numi-lab/tags). Copy the original branch name and archive tag from the ledger. This example fetches the complete object history for that tag into a fresh local repository and recreates the original branch.

```sh
git init --bare restored-numi-lab.git
branch='<source_branch from branches.tsv>'
tag='<archive_tag from branches.tsv>'
git -C restored-numi-lab.git fetch --no-tags https://github.com/Numi2/numi-lab.git \
  "refs/tags/$tag:refs/tags/$tag"
git -C restored-numi-lab.git branch "$branch" "refs/tags/$tag^{commit}"
git -C restored-numi-lab.git rev-parse "refs/heads/$branch" "refs/heads/$branch^{tree}"
```

The final command prints the source tip and root tree from the ledger.
