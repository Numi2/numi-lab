# Connected skin/fat reconstruction and transfer

This increment extends the existing skin/fat authoring and accepted-transfer owners. It does not install a candidate mesh, change material laws, or qualify a native simulation. All execution was on SSH Numi (Apple M6); no visual inspection was performed.

## Cell-quality correction and connected reconstruction

The earlier 084 report used a mean-ratio quality metric. That metric is not the Matter author's signed determinant / longest-edge cubed gate. Replaying the actual gate found five failing cells despite the earlier positive result. This correction is explicit; the original report and failed output are retained.

The connected reconstruction preserves all 7,233 original vertices and their source identities. Local internal reconnection and elimination of an unnecessary generated interior point produce 7,360 vertices and 32,796 cells, including 125 generated boundary vertices with source-face ancestry and two generated interior vertices. The fixed reference/current baseline has zero invalid or subthreshold cells under the actual gate (minimum quality 3.55771315e-5). Stable boundary chains and per-material volume checks pass. This establishes a usable authoring topology, not bone clearance.

Generated boundary points now participate in the same skin/fat thickness constraints as the original columns. The helper covers all 125 generated boundary vertices, with 362 source-parent relationships and 724 hard inequalities; it deliberately excludes the two interior vertices. It supports non-prefix node numbering and rejects incomplete or invalid ancestry.

## Detailed skin and conservative transfer

The refined detail-to-mechanics map uses the existing additive displacement owner, with 61,950 detailed vertices, 54,551 active rows and 7,399 unchanged inactive rows. Parent-face identity is preserved: refined barycentrics search only children of the existing source parent, without a nearest-face fallback. Reference pullback and partition-of-unity correction remove the prior 221 mapping failures. Detailed geometry is bitwise unchanged at the construction baseline; the real-map transpose-work check has relative error 1.8e-16.

The frozen transfer report contains a reversed-sign formula string. A separate correction binds the intended and actually tested positive additive formula; the original bytes are retained. One of 53 added outer nodes has zero detailed-map support and is explicitly reported. This is not silently replaced with an arbitrary donor.

Mass transfer is evaluated geometrically against all 28,632 native cooked source cells. The original cooked nodal mass remains authoritative, with source contribution identities preserved. The full overlap scan saved its CSR before admission checks. Exact rational intersection classification shows the 22 apparent cross-material volumes are rank-2 shared-face contacts caused by the tolerance clip; no full-dimensional cross-material overlap was found. The unchanged coverage gate nevertheless found six child cells and three parent cells beyond 1e-7 relative error. These failures are independent of those 22 contacts. The failed CSR, witnesses and exact classifications are retained. No global rescaling or cross-material mass reassignment is permitted.

A bounded local precision correction re-evaluates only 1,025 pairs incident to the failed rows in pair-local coordinates. It changes 54 overlap values; the maximum change is 2.45e-14 cubic metres. Both unchanged coverage gates then pass (maximum child error 8.918e-8, parent 9.618e-8). Exact-rank checks confirm the two old positive pairs removed by this correction are shared-face contacts.

The old overlap/native-volume mass rule still fails the stricter source-ledger closure gate and is retained as a failed trial. The explicit conservative variant first checks raw coverage and material identity, then distributes each native cooked parent mass using overlap divided by that parent's covered overlap. It leaves the raw geometric CSR bitwise unchanged. This local partition correction is reported separately from geometry; it is not a global body-mass rescale. Total and per-material cooked mass close to about 1.3e-16 relative, with zero density violations. The original 074 source-ledger mass/first/second-moment gates pass (maximum errors 1.47e-15 / 1.67e-15 / 1.73e-15). Those moments use retained donor-local source positions. A separate geometric reference-space check measures a maximum centre-of-mass shift of 7.85e-9 mm; lumped second-moment changes are 1.08e-5 for skin and 4.02e-5 for fat. These latter values are measurements, not a new qualification gate.

## Connected clearance progress

All comparisons here use the same 7,360-node / 32,796-cell topology, the same 185 bone banks, actual emitted Float32 positions, and the same detailed-surface mapping. These figures must not be compared as if they were the former 9,320-node mesh.

| Stage | Edge/bone crossing groups | Nodes inside bones | Invalid/subthreshold cells | Detailed self-intersections |
| --- | ---: | ---: | ---: | ---: |
| Connected baseline | 2,498 | 140 | 0 | 0 |
| 087 hard constraints, two iterations | 2,262 | 130 | 0 | 0 |
| 088 third iteration | 2,173 | 130 | 0 | 0 |
| 089, true coarse area/rotation constraints | 2,121 | 129 | 0 | 0 |

The exact detailed audit checks all 108,987 faces and only allows intersections confined to the actual shared indexed feature. It does not exempt adjacency generally. These are static authoring results; the bone edge witnesses are not pressure/contact-force measurements or certified penetration depths.

The soft least-squares trial stalled after one group of improvement. Direct hard-constraint LPs made more progress. The subsequent exact Float32 line search identifies surface normal distortion as a limiting gate. Each failed step records its first failing gate, cell witnesses, determinant bounds, hard-row slack and contact deficit. We do not loosen these gates or accept the crossing candidate.

## Remaining head-layer defect

A separate pinned audit of frozen 088 iteration three identifies 75 original head-hit columns with D (deep fat) inside a head bank; one also has I inside, and none has O (outer skin) inside. One generated point is also inside. Median source O-I / I-D spacing in these columns is 2.451 / 10.256 mm. Candidate spacing is 2.252 / 10.277 mm. The ordinary deep-layer inside count increased by three over the matched 088 baseline, even while the global edge crossing count improved. We retain this regional regression explicitly.

This concentration points to inner-layer placement/thickness or registration as an unresolved anatomical issue. Infinite closest-face-plane intersections are diagnostic only and do not establish a complete finite-solid clearance certificate or a feasible final thickness. The data do not justify simply inflating the outer head skin, removing mass, or weakening material/quality gates.

## Native transfer owner preparation

The current-main accepted-transfer owner gains an opt-in prepared refined-domain import. It verifies pinned source/candidate identities, exact original prefix, stable parent/child ancestry, donor identities, convex weights, generated-node prolongation and an explicit compact-outer-to-full-mechanics index map. Failed installation leaves the existing owner unchanged. Direct-source rows are distinguished from nearly one-hot projected rows, so projection cannot switch to a different incident parent. CPU reference checks verify zero-baseline transfer, generated-node motion and transpose virtual-work equality. C++17/C++20 warnings-as-errors and sanitizer checks pass.

This is a validated receiver, not a wired native scene: the existing app still builds rows from its old region inputs. A trusted refined input regeneration/loader and callsite update remain necessary after anatomical admission. No full Human compilation or native shader parity is implied by the standalone test.

## Qualification and ownership

The Metal runtime remains the sole per-step physics owner. No native physical step, controller history, anatomy replacement, or physiology improvement is claimed. Remaining work includes resolving bone clearance, integrating the passing conservative cooked mass map, wiring the regenerated refined input into the existing native owner, and testing motion and accepted-state rollback.

The physical PLC's side closures do not establish closure of the detailed head-height skin loops. Unrelated back geometry, source identities and the dirty native checkout remain untouched. Mixed-source anatomy remains a reference assembly. Only about 2.9 GiB was free at verification; the declared 12 GiB native-qualification floor is unmet.

All 73 Python checks and 11 subtests pass. The standalone prepared-domain C++ check also passes on SSH Numi.

## Reproduction

The execution context binds the device, checkout and published parent. Source manifests verify original and stored bytes, including failed trials. Ten major reused inputs were independently read from their published Git blobs, decompressed and matched to current source hashes; their immutable commit paths are retained. Canonical manifests bind tested owner changes.

```sh
python3 verify_retained_artifacts.py --canonical-root /path/to/numi-lab --git-root /path/to/numi-lab
cd /path/to/numi-lab/matter/tools
python3 -m pytest -q test_human_skin_fat*.py
clang++ -std=c++20 -O2 -Wall -Wextra -Werror -I /path/to/numi-lab/apps /path/to/numi-lab/apps/NumiHumanAcceptedFEMSkinTransferPreparedDomainCheck.cpp -o /tmp/prepared-domain-check
/tmp/prepared-domain-check
```

Static NumPy/SciPy authoring does not provide CPU per-step physics. Static geometric validity, mass conservation, anatomy clearance, runtime ingestion and native physical qualification remain separate gates.
