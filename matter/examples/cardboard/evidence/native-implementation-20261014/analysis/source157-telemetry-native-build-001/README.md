# Source157 telemetry build and compile-only admission

Source157 was created from immutable source154 with `cp -c -p -R` on the APFS volume, then the sealed source157 one-file patch was applied. The explicit parent is source154. The frozen source map contains 787 entries: only `matter/tools/cardboard_probe.mm` differs from source154. All 1,034 configure-support hashes match source154. Recovery metadata and the source overlay manifest are in the source157 snapshot root.

A fresh provenance build used the same CMake topology as source154: `-S <snapshot>/matter`, Release, exported compile commands, target `numi-matter-cardboard-probe`. It used two build jobs (source154 used four). The source-bound build record reports 787 source inputs and `changed_sources: []`. The full record, configure/build command logs, CMake cache, and compile commands are copied under `build-metadata/`.

The build produced exactly three provenance-bound artifacts:

- `numi-matter-cardboard-probe`: `19492c5cff7dd94148335f7ed53a358bbcebcaa163241cfd8012abbac9f765e0`
- `NumiMatter.metallib`: `89f9d756a0298fad44786b5333d67172636a8255e8c4f71b2696e4e5d6271114`
- `NumiMatterPhysicalStateDigest.metallib`: `0401b29d3ac2165153f36bea19786be653a10a242eed140e1c9008595ec1f729`

Both metallib files are byte-identical to source154's build. The executable is source157-specific.

The source157 executable admitted the source154 compile-only protocol at Newton budgets 64 and 128. Both emitted `status=compile_only`, `runtime_execution=false`, `accepted_steps=0`, and `physical_validation=false`; both packages contain 882 nodes and 1,920 tetrahedra. These runs used source154 material-file paths, whose bytes and SHA-256 values are identical to the corresponding source157 material assets. They are compile/admission checks, not physical trajectories.

The build log retains two Metal shared-variable initialization warnings from `fgmres.metalinc` and one existing unused-function warning in `cardboard_probe.mm`; the build completed successfully. No native physical simulation or GPU run was performed. Disk remained above the requested 512 MiB floor: approximately 1.6 GiB before snapshot creation, 1.4 GiB before build, and 1.0 GiB after compile-only checks and retained receipts. The accompanying resource record includes the observed free-memory pages and exact final `df` output.
