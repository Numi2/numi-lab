# Human holistic preconditioner work, 2026-11-04

Source: `0526640f89f4bedf257a1533e98770e254739007`, Numi2/numi-lab/main, author and committer numi2. Builds and measurements ran on SSH Numi, Apple M6 / Mac18,5 / macOS27.0 (26A425). Unrelated dirty execution work was preserved.

## Measured result

Packed combined native216 completed32 accepted full-coordinate steps. All11 CSV traces and3 MRV captures are byte-identical to control193. This is **0.500000024ms of a release transient**, not supported rest, sustained breathing/heartbeats, native presentation qualification or clinical validation.

| Configuration | Native-loop wall seconds | Summed GPU seconds | Prefix real-time factor |
| --- | ---: | ---: | ---: |
| Unchanged control193 | 233.448333 | 230.970979 | 0.00000214180x |
| Combined, strided scratch213 | 233.952550 | 231.359746 | 0.00000213719x |
| Combined, packed scratch216 | 230.894866 | 228.289066 | 0.00000216549x |

Complete-loop throughput improves **1.11% in one pair**. Roots1–31 GPU throughput improves1.83%; the slower initial root remains in complete-loop timing. Preparation takes about3minutes separately. Process manifests include preparation. Root0 uses stage instrumentation; later roots have no selected-kernel timing. Whole-command timing remains.

**0.1x remains unmet by approximately46,179x.** Three matched measurement pairs, sustained performance and15wall-fps presentation remain unfulfilled. No Human timestep increase was attempted.

## Changes

- Fix late GPU uploads: the immutable/state UploadPlan was finalized before3 existing contact-cache flags and2 coarse tables were appended. Those copies had no live blit encoder. A second bounded metadata batch now finishes after its last producer and before use. Appending after finalization and finishing twice fail closed.
- Compute each four-node lumped patch once, then preserve the original ordered CSR fold. Reuse existing element-operator scratch, packing float4 results contiguously instead of one result every64bytes. The production operator subsequently overwrites it.
- Preserve the legacy compensated coarse sums and SIMD ordering. Reuse diagonal sums within a current linear solve, rebuilding at column0 of each Newton/restart. Apply corrections in256-node tiles. The coarse read reduction still uses one SIMD32 group per object.
- Share vector/mass/row loads across12 regional projection modes, preserving each mode's256-lane reduction tree.
- Add direct/cached-ICB paths, admission/capacity checks, shape signatures and an optional in-context legacy/new coarse comparison using dead operator scratch.

The Human configuration adds about43KiB of bounded coarse metadata/scratch. Patch and diagnostic paths add no bulk physical state. Geometry, material equations, mass/provenance, contacts, timestep15.625us, controller clocks, certification and accepted-state ownership remain unchanged.

Paths remain **off by default**. Measured configuration:

    NM_FGMRES_PARALLEL_BASE_PRECONDITIONER=1
    NM_FGMRES_PARALLEL_BASE_PRECONDITIONER_MASK=7
    NM_FGMRES_COARSE_PARITY_DIAGNOSTIC=0

Internal bits:1 patch,2 coarse,4 projection. Admission requires existing immutable full-coordinate selector0/6 FEM, one environment and empty MPM. No restrictions were bypassed.

## Validation and retained failures

Final GPU contract suite: **2,665,138 checks PASS**. Focused suite: **1,732,190 checks PASS**, covering broad float ranges, cancellation, signed zero, invalid/inactive/duplicate rows, empty/completed work, capacity canaries and isolated multi-environment cases. Runtime initialization, +/- gravity on each axis, analytic momentum and bitwise replay also pass. Native213/216 match all11 traces and3 captures. No visual inspection was performed.

- 195 failed force/strain comparisons;197/205 failed exact endpoint/traces. Isolated199/201/203 matched the reference. Exact source variants are retained.
- In-context208/210 detected coarse differences before the upload fix. Rebuilding diagonals205 and preserving all SIMD lanes210 did not fix the missing uploads. Those hypotheses are not established root causes.
- 212 passed its coarse comparison and solver certificate before a disk stop interrupted the endpoint dump. All788,882 captured force/strain records match190, but this is an incomplete endpoint.
- Packed215 hit the disk guard before completing a step.216 retried the same binary/source after cleanup. An unnecessary empty producer override in216 was rejected before GPU launch; the failed prelaunch and corrected resume are preserved.
- Mask6/native214 completed27 profile rows, then hit the12GiB disk guard during root27. It has no completed full-prefix comparison and is not promoted. This resource stop is not labeled a numerical rejection.

Active solves, fine skin/fat conditioning, repeated assembly/globalization and final contact certification remain major costs. Prior distributed reductions/contact work already exists; this increment does not claim to reimplement it. The late214 working-set/disk peak still needs attribution; reported OS swap alone does not explain it.

## Storage, reconstruction and limits

Closed historical negative captures were losslessly compressed, SHA256-restored and deduplicated with independent APFS clones:0.89GiB net recovery. A transparent-compression pilot had no benefit and was discarded without changing inputs. Git maintenance preserved **all89,030 object IDs**, including unreachable objects, and passed connectivity checks:1.12GiB recovered. No branch/source/object pruning occurred. Receipts retain exact paths, content hashes, measured space and restoration instructions; logical APFS sizes are not claimed as recovered space.

Native guards stayed12GiB available, at most3GiB additional swap and900s per trial. Sources and current references remain. Historical MRV gzip archives must be decompressed before replay.

`execution-source-reconstruction.json` resolves executed pins to the published revision, canonical170 archive or deduplicated `source-deltas.tar.gz`. Compact traces and full capture identities are retained without copying hundreds of megabytes of repeated diagnostics. Full detailed outputs remain on Numi; this package does not claim that all large local outputs are archived remotely.

Each plan/build manifest/runner retains exact binary/device/input identities, environment, launch argv and timing. Archived executed runners depend on the pinned existing inputs and machine-local helper paths; they are not a new portable scene format. No supported-rest, sustained physiology, motion or0.1x qualification is inferred from these prefixes.
