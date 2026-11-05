# Integrated Human solver and memory work, 2026-11-04

## Result and scope

The complete current Human scene still falls far short of 0.1x.
The verified default change removes 426,089,952 bytes (406.35 MiB) of
persistent duplicate immutable tetrahedron storage. It does not accelerate
the measured body run materially. The 32 accepted roots cover only
0.000500000023749 simulated seconds of a release transient, not supported
rest, a complete breath, a heartbeat, or sustained qualification.

Trial 238: body 228.965443333 s, 0.000002183735748x. The comparable 234
body time is 229.075407625 s; this is noise-level change, not a claimed
speedup. Initialization-inclusive 238 elapsed time is approximately 416 s.
Presentation qualification is pending. The scene is balanced=false; this
is not evidence of an equilibrated supported checkpoint.

## Implemented changes

* Proven-empty FEM transport assembly: replace four whole-mesh inactive
  field passes with one explicit publication of unchanged transport
  solution and zero residual scratch. Admission uses the existing proof
  that no FEM transport objects or field boundaries are active. Mechanical
  pressure and vascular/physiological equations remain active.
* Owner-incidence certification, opt-in
  NM_MIXED_CERTIFICATE_OWNER_INCIDENCE=1: reuse freshly generated
  deterministic node/contact incidence rather than scan every contact
  independently for every continuum owner. Validate encoded slots/corners,
  select the first supporting corner per owner, and retain failure
  witnesses. This path assumes the existing generator's support/owner
  invariants and complete fresh incidence; it is not a new collision
  detector or a reusable stale contact cache. Default remains off pending
  broader layout qualification.
* Bounded column telemetry, opt-in NM_FGMRES_COLUMN_TRACE=1 with the
  existing selected Newton trace root: capture active column residual
  history on the same GPU timeline, then read once at completion. No
  per-column host synchronization or encoder-splitting timing. Distinguish
  scaled Arnoldi estimates from fresh physical residuals.
* Direct sparse block assembly, experimental
  NM_FEM_ASSEMBLED_XYZ_DIRECT_EDGE=1: accumulate the existing deterministic
  contribution order directly into block CSR, eliminating the 144-float
  per-element intermediate. Preserve tangent, product/sum validity and
  physical certificates. Incompatible with graph factor reuse requiring
  that intermediate. Default remains off.
* Immutable tetrahedron sharing, default on: accepted, candidate and
  checkpoint views alias immutable reference/connectivity/material
  records when no mutation, puncture, cohesive or mutable-topology path
  is admitted. Dynamic positions, pressure and material history remain
  independently owned. Restore still checks exact immutable layout;
  borrowed mutation commands reject before encoding.
  NM_FEM_SHARE_IMMUTABLE_TETRAHEDRA=0 retains the comparison path.

## Native results

| Trial | Result | Actual work | Interpretation |
|---|---|---|---|
| 232 | 32 accepted | 144 solves / 2,304 columns / 62 budget exhaustions | Empty-field + owner-incidence paths; no whole-run gain |
| 233 | 0 accepted | capacity rejection | 2M candidate capacity was insufficient; 4M retained |
| 234 | 32 accepted | 144 / 2,304 / 62 | Late-root column/alpha diagnostic |
| 235 | 25 accepted, root 25 rejects | 85 / 1,360 / 32 | Existing materialized sparse assembly |
| 236 | Same root 25 rejection | 85 / 1,360 / 32 | Direct CSR exactly matches 235 compact outputs/failure; removes 852,179,904 bytes |
| 237 | 31 accepted, step 32 contact rejection | 172 / 2,752 / 79 | Contact proof-composition experiment failed; production change removed |
| 238 | 32 accepted | 144 / 2,304 / 62 | Immutable element sharing; all ten compact CSV traces byte-identical to 234 |
| 239 | 0 accepted, vascular line-search rejection | Corrupted constants; counts invalid as convergence evidence | Superseded by the ownership diagnosis below |
| 266 | 1 accepted | All 4,096 memory snapshots intact | Instrumented correctness trial; excluded from performance |
| 267 | 32 accepted | 93 solves / 1,488 columns / 7 budget exhaustions | Repaired composite, diagnostics removed |
| 268 | 32 accepted | 144 / 2,304 / 62 | Same clean binary, regional reference; all ten compact traces byte-identical to 238 |

All ten compact traces of 232 and 234 also match exactly. The parallel
owner certificate changes 12 floating reduction outputs by at most
1.119e-7 relative; it retains the same accepted trajectory. Direct CSR
has only been qualified against its equally rejecting reference through
root 25, not promoted as the Human production setting.

After Krylov allocation, Metal reports 5,935,087,616 bytes in 234 and
5,082,890,240 in 238. This phase difference includes upload/transient
lifetimes; only 426,089,952 bytes are identified as the persistent sharing
saving. Do not add these two figures as separate gains.

## Diagnosis and next work

Root 28 uses sixteen active Newton solves, each with sixteen columns.
Its later scaled Arnoldi residual falls only to approximately 28.8% of
the initial norm; several accepted line-search fractions are about
0.000346. Inner solve cost, conditioning and globalization remain coupled.
No early-stopping omission is claimed; completed-work suppression exists.

The opt-in thin-cell/regional composite combines the existing regional
basis with 8,290 fine-cell patches covering 30,768 nodes (11,335 selected
cells). It removes no physical rows. Its earlier apparent seven-column
completion was invalid diagnostic evidence: initialization subsequently
replaced its live regional basis/RHS/coefficient buffers with a 16-byte
dummy intended only for the standalone thin-cell path. This affected the opt-in composite prototype introduced in this session;
the regional reference did not use that cleanup branch. The sparse-mode
reduction wrote beyond that placeholder and corrupted adjacent solver
constants. The resulting zero correction and failed physical certificate
were consequences of this ownership bug, not useful convergence.

Trial 265 traced every dispatch and then the individual indirect
commands: the first changed constant snapshot followed
nm_partitioned_reduction_reduce_sparse_mode_partials. The corrected
initialization retains the live regional buffers for the composite and
checks their ownership and extents before installing the runtime.
Trial 266 accepted one complete Human step, with all 4,096 GPU constant
snapshots and the final host check intact. Replaying that captured current
operator input, with its original unmodified headers, passed 311 checks
and ordinary/ICB bitwise parity. This is a correctness result, not a
performance claim. Earlier corrupted composite trials remain retained
and are not used to judge solver convergence.

Indirect-command barriers are now set before encoding their dispatches,
as required by [Apple's setBarrier documentation](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand/setbarrier/).
The same correction covers Newton-tail, operator, owner-mass, vascular,
CGS2 and aggregate-prepare commands. Certification and the command-buffer
timeline remain mandatory. Temporary per-dispatch snapshots, source-header
capture and shader-validation experiments were archived outside production
runtime routing before the matched longer comparison.

The clean build passed the complete 2,665,964-check FGMRES contract suite.
Trials 267 and 268 completed the same bounded 32-root release horizon in
the same clean binary. The composite used 93 solves / 1,488 columns / 7
budget exhaustions, versus 144 / 2,304 / 62 for the regional reference.
Body times were 194.554049 s and 228.821494 s respectively: 15.0% less
body time, or 17.6% higher observed throughput in this one pair.
The composite body rate is 0.000002569980x; 0.1x remains about 38,911x
higher. These are initialization-excluded body timings, not sustained
rest or qualified native presentation throughput.

The clean reference reproduced all ten compact traces of 238 byte for
byte; correcting barrier encoding did not change that reference trajectory.
The composite has different accepted approximations and remains opt-in.
Full-field comparison 269 versus 270 failed: skin/fat regional
volume-weighted strain error was 124.29%, and its free internal-force L1
difference was 2.81938 N against a 2.60512 N reference magnitude (108.22%).
The aggregate anchor history differed by only 0.739%; this did not reveal
the local tissue failure. Trial 269 combined the composite preconditioner
and cooperative tangent mode 2, so this result does not isolate the
preconditioner. Direct mode-1 composite trial 273 isolates the preconditioner and fails with the same 124.29% skin/fat strain and 108.22% bulk force discrepancy. It remains disabled; the 15% one-pair timing gain is not qualified.
Ten compact traces and the endpoint render payload of 267/269 match,
but those outputs do not establish full FEM force/strain equivalence.

Compact tangent owner tasks are separately implemented behind
NM_FEM_TANGENT_OWNER_TASKS=1 (default off). Compiler-certified whole-map
materials dispatch one owner per tetrahedron; every fallback retains its
nine columns. Actual tasks fall from 13,315,311 to 1,505,999: 1,476,164
whole-map tetrahedra and 3,315 fallbacks. Five eligible material families
cover 99.78% of tetrahedra. State, cache layout, precision policy, and all
physical/certification rows are unchanged. The extra immutable worklist
uses 5,917,916 bytes. Admission rejects mutable topology/material ownership,
unsupported bundles and incompatible worklists.

Trials 271 (cached Newton ICB) and 272 (ordinary dispatch) each accept all
32 roots with the reference work counts 144/2,304/62. All ten CSV traces
and all 318,974 node plus 1,479,479 tetrahedron endpoint records match
reference 270 byte for byte. The endpoint-record SHA-256 is
cc6eb0edd0b5ed60a24fc911f11e6965bde10a237151f59bb48e6e872f17d23b.
Their body times are 233.425754 s and 233.315623 s, versus 234.752598 s
for 270. Summed root GPU durations improve by about 0.7% in one pair:
this is not an established speedup or a solution to the active-solve cost.
The full build-45 FGMRES suite passes 2,666,153 checks, including complete
worklist coverage, empty/all-whole/all-fallback layouts, overflow, and
ordinary/ICB paths. These are the same 0.5 ms release horizon, not
three consecutive horizons or sustained physiological qualification.

Remaining: supported equilibrium initialization, efficient valid local
and interface solves, conditioning of retained thin cells, compact
material operator storage, fresh contact certification attribution,
inactive Krylov backing layout and sustained coupled qualification.
Do not retry the retired larger-timestep path or the failed aggressive
66x coordinate reduction as accepted solutions. No 0.1x result, full
anatomy motion qualification, or clinical validation is claimed.

## Validation and reproducibility

Native device: SSH Numi, Apple M6, Mac18,5, 16 GB, macOS 27.0 (26A425).
Builds use the working Metal.xctoolchain bin directory on PATH, without
requiring a full Xcode install. Source execution worktree:
    /Users/numi/Work/skin-fat-native-integration-209
baseline f75955f08d6edf0f85027eac7614017ac9fbc996, with exact dirty
execution patches retained per trial. Publication intentionally preserves
unrelated changes already present on main.

Passing relevant native checks:
* deterministic contact incidence and owner enumeration: 7,911 checks;
* FGMRES true residual, scaling, empty fields, telemetry, assembled
  products and ICB paths: 2,665,964 checks in clean build 43;
* captured current Human CGS2 input: 311 checks, original headers retained,
  ordinary/ICB bitwise parity;
* direct sparse assembly component: 5,123 checks;
* immutable one/two-environment storage, restore/tamper and borrowed
  mutation admission: 362 checks;
* complete material/frame/accepted-state suite, including late rejection,
  rollback, replay, reset and immutable restore: PASS.

Each run directory retains the launch argv/environment, source and
binary/library/input hashes, exact patch, initialization/body timing,
compact traces, resource guard and execution result. Replay command:
    python3 run_native_plan.py install-motion
Review the frozen paths and hashes and use a fresh output directory.
The driver pipeline archive is a mutable cache; its initial hash belongs
to that exact run, not later reuse. Large immutable mass preparation
dumps are hard-linked, not regenerated for each trial. Current
source-carrier mass inputs are protected after verified restoration from
the archived cold evidence; see the recovery receipts. Failed trials
remain evidence, not defaults.

## Additional verified storage recovery

Byte-identical closed native outputs were replaced with verified APFS
clones while preserving paths, contents, modes, owners and nanosecond
modification times. The additional pass recovered 5,783,883,776 physical
bytes (5.39 GiB): available bytes increased from 15,795,802,112 to
21,579,685,888. This is a filesystem measurement, not the sum of logical
file sizes. Current sources, mass inputs and failure evidence remain.

The following closed-output pass recovered another 2,249,322,496 physical
bytes (2.09 GiB), from 19,428,036,608 to 21,677,359,104 available bytes.
The archive at
[the verified cold-evidence release](https://github.com/Numi2/macmini-evidence-archive/releases/tag/numi-20261104-cold-human-evidence)
also retains the bounded solver input and exact diagnostic source for
the regional-buffer failure and correction. The independent download restored and verified all 54 archived members,
including their hashes, modes and nanosecond modification times. Original
current evidence is retained.

The failed 269 endpoint and 269-272 compact/source/configuration evidence
are retained in human-tissue-qualification-269-272.tar.zst on the same
GitHub release. An independent download restored and verified all 103
members, hashes, modes and exact modification times before the older
failed 269 plain stdout was retired. Source inputs remain local. The older 270/271/273 endpoint stdout is now in the independently restored 147-member archive; the newest exact endpoint is 276. See the manifest and verification
receipt; no failed evidence was discarded.

## Lossless immutable contact storage (build 46)

The static body-contact tables expand one source surface over many rigid
body banks. Their logical order and complete collision queries remain
unchanged. Private storage now factors repeated source templates and
body/range seeds, reconstructing the original 64-byte sample and 32-byte
range exactly on the GPU. CPU initialization round-trips all rows before
installing the immutable buffers. No sample is quantized, culled or
reassigned; no geometric or force state changes.

The opt-in setting NM_HUMAN_COMPACT_IMMUTABLE_CONTACTS=1 requires aliased
primary/solver/grouped layouts, multi-master triangle contacts, no implicit
spring rows, checked offsets and a real storage reduction. Unsupported
layouts fail explicit admission. Public assets and replay ABIs remain
unchanged. Logical counts survive release of redundant host upload spans.

Actual scene 276 contains 15,453,677 rows, 180,197 unique source templates
and 595 seeds. Sample/range storage decreases from 1,483,552,992 to
135,219,184 bytes: 1,348,333,808 bytes (1.256 GiB) identified saving.
The accepted-step Metal allocation is 8,367,013,888 bytes, versus
9,715,335,168 for reference 272. This allocation delta is not an additional
saving to add to the identified table reduction.

All 32 roots accept with unchanged 144 active solves, 2,304 columns and 62
budget exhaustions. All ten compact traces and all 1,798,453 endpoint node
and tetrahedron records are byte-identical to 272, with endpoint SHA-256
cc6eb0edd0b5ed60a24fc911f11e6965bde10a237151f59bb48e6e872f17d23b.
The complete 2,670,289-check FGMRES suite passes, including CPU/GPU exact
decoding, signed zero, invalid indexes, corrupt records and empty/capacity
cases. The complete material/frame/rollback check also passes.

Body time is 235.947012 s, versus 233.315623 s for 272: this single
comparison is approximately 1.1% slower. Inclusive execution is 426.876769 s,
but startup differs between trials and this is not a demonstrated
end-to-end speedup. Body rate is 0.00000211912x. The measured result is a
lossless memory reduction, not 0.1x performance or sustained physiology.

Trial 274 stopped during configuration because direct-edge assembly
requires the assembled-operator option. Corrected trial 275 was stopped
by the unchanged 12 GiB disk guard during initialization; it has no valid
performance or acceptance result. The pretrial target was raised to 16 GiB to accommodate initial output and swap growth. After trial 276 proved a 1.35 GB GPU storage reduction, the packed-contact interior trial uses a 14 GiB pretrial threshold while retaining the unchanged 12 GiB disk and 3 GiB swap-growth stop guards. Both trials remain retained. The 147-member recovery archive was independently
downloaded, fully restored and hash/metadata checked before local retirement.

## Registered outer fine-coordinate preparation

The modest interior-reduction experiment first stopped before physics:
the outer-fine selector assumed the legacy contiguous three-layer shell.
The current explicit unpaired volume includes genuine interior nodes,
which correctly carry INVALID O/I/D layer labels. It can also contain
all-new nodes with verified parent mappings and zero retained-original
node count. Neither case is an invalid anatomical layer.

Build 48 selects the exact existing O inventory and checks all three
registered role maps for complete counts, matching labels, bounds and
unique membership. Original and parent-mapped interiors remain distinct
from boundaries; no layer or parent is invented. Legacy invalid labels,
missing new-node parents and corrupted role maps remain failures. The
existing full geometry, source-domain, mass and ancestry verification
still runs before this policy. No physical material, mass, attachment,
geometry or runtime equation changes.

The complete registered skin/fat reduction-label suite passes, including
13 malformed-policy cases and both retained-original and all-new mapped
interiors. Preparation trials 277 and 278 remain retained as failed
evidence. The source correction passes initialization on the actual registered
volume. Trials 279 and 280 stop on inherited full-coordinate-only options
(the inexact linear certificate and isolated dense45 vascular solver).
Trial 281 initializes 248,202 free coordinates versus 956,112 (3.85x),
including 246,246 fine-complement coordinates and 1,956 q coordinates,
then refuses an incompatible legacy diagnostic certification option
before encoding a physical step. The hybrid solver requires its own
mandatory full-operator certificate, independent of that option.
These are retained preparation/configuration failures, not accepted
physics or performance results. The unsupported options are removed in
trial 282. Force, regional strain, contact and final physical acceptance
remain unqualified.

The exact older 272 endpoint and initial 275 viewer payloads are now
retained in human-contact-storage-276.tar.zst on the cold-evidence
release. Independent download and restoration verified all 40 members,
content hashes, modes and nanosecond modification times before local
retirement. Current full endpoint 276 remains local.

## Cooperative tangent dispatch and owner-certificate composition (builds 49–50)

Mode-2 cooperative cache builders now use GPU-written per-subset dispatch
arguments for the completed-Newton latch. Each immutable material subset
keeps its exact eight-tetrahedron group count. Iteration zero, forced
rebuilds, unhealthy environments and other environment domains retain
their required work. Command offsets and barriers are explicit; the
mandatory physical certificate remains separately scheduled. This is not
a new approximation or a default switch to mode 2.

The complete FGMRES suite passes 2,671,255 checks in build 49 and
2,671,801 in build 50. The added argument tests cover empty, large and
overflow-edge counts, initial/late iterations, healthy/unhealthy and
completed/noncompleted state, forced rebuilds, and padded canaries.

Trial 284 uses mode 2 with all twenty material pipeline variants. Its
318,974 node and 1,479,479 tetrahedron records reproduce endpoint SHA-256
cc6eb0edd0b5ed60a24fc911f11e6965bde10a237151f59bb48e6e872f17d23b.
All nine numerical CSV traces match 276 byte for byte. The surface audit
omits one duplicate final row because extra MRV captures are disabled;
all unique audit rows match. It accepts all 32 roots with unchanged
144 solves, 2,304 columns and 62 exhaustions.

Body time is 247.742629 s versus 235.947012 s for mode-1 reference 276,
about 5.0% slower in this single pair. Inclusive time is 532.507208 s.
No speedup is claimed, and the faster mode-1 configuration is retained.
This is still only 0.5 ms of release motion, not supported rest or a
sustained coupled performance result.

The owner-incidence contact certificate now has a composition proof:
actual legacy, bitonic and radix GPU incidence output feeds the production
owner enumerator, compared against an independent active-list oracle.
The fixtures include duplicate corners/active indices, invalid and
inactive slots, empty input, multiple environments and the hierarchical
radix threshold. The contact suite passes 8,102 checks. The complete
certificate fixture also compares owner and legacy paths, including
519 contacts, deterministic failure witnesses, pressure/transport and
nonfinite/geometry failures. Production owner routing remains explicit;
no physical contact is removed.

## Bounded observation and moderate-reduction status

The routine trial launcher streams the full native endpoint trace into
lossless gzip, checks its CRC/UTF-8 stream and compressed/uncompressed
hashes, propagates native failures, and retains compact physiological
traces and the initial native-viewer pack. Its self-check preserves a
deliberate exit code 7 and exact stream content. Optional extra MRVs are
disabled for these performance trials; physical geometry checks remain.
The 12 GiB disk, 3 GiB swap-growth and 900 s guards are unchanged.

Five byte-identical closed MRVs were APFS-cloned with exact hash and
metadata checks, recovering 1,606,656,000 physical bytes. Lossless
compression of the retained 276 endpoint recovered another 694,226,944
physical bytes after full independent decompression/hash verification.
Current source, mass inputs and endpoint records remain available.

Trial 282 initializes the moderate 3.85x coordinate reduction and accepts
one physical step, then stops on a root-specific observer admission
mismatch. Trial 283 removes that optional witness but is stopped by the
12 GiB disk guard before a useful motion result. Both remain failed
qualification evidence. The compact-output retry is 285. It retains the
exact outer O boundary, 246,246 fine-complement coordinates, 1,956 reduced
coordinates, full material/contact evaluation and mandatory full physical
certification. Force, regional strain and sustained-motion gates have not
yet passed. The old failed 66x basis is not promoted.

## Compact radix workspace and rejected contact-block experiment (builds 52–55)

The radix incidence path no longer allocates or emits bitonic ulong keys.
Its payload stride is the complete logical corner capacity rather than a
power-of-two padding. The separate bitonic reference keeps its original
keys and padding. Histogram/scan scratch remains independently sized for
the full admitted capacity. Active records, deterministic order, duplicate
corners, history identity and capacity/overflow checks are unchanged.

For this Human, 4,207,296 contact slots require 25,243,776 logical corner
records versus 33,554,432 padded records. Identified workspace decreases
from 402,653,184 to 151,659,968 bytes: 250,993,216 bytes (239.37 MiB).
The native contact suite passes 8,116 checks, including small odd/even
radix passes, multiple environments, empty/invalid/duplicate corners and
hierarchical scans with exactly sized buffers and end canaries.
The complete FGMRES suite passes 2,671,801 checks in build 55.

Trial 290 accepts all 32 roots. All 1,798,453 endpoint records and nine
numerical traces match reference 276 byte for byte. The optional surface
audit omits a duplicate export row; its unique rows match exactly.
The strict all-file byte comparison remains false, while the numerical
state comparison passes. Body time is 230.856208 s versus 235.947012 s:
approximately 2.2% higher throughput in this single pair, not a repeated
speedup estimate. Inclusive time is 432.311483 s versus 426.876769 s;
startup differs and no end-to-end improvement is established.
Metal allocation is 7,863,582,720 bytes. Its delta includes driver-dependent
effects and must not be added to the identified workspace saving.
All 144 active solves, 2,304 columns and 62 exhaustions remain unchanged.
This is 0.5 ms of release motion at 0.00000216585x, not supported rest,
sustained physiology, presentation qualification or the 0.1x goal.

The separate nodal contact-block prototype reused the full PSD spatial
3x3 metric, with component tests for rotated rigid and deformable contact,
timescale factors and independent CPU double-precision inverses. Despite
passing those tests, correctly scoped trial 289 rejects root 21 after 21
accepted roots. Its final nonlinear norm is 0.000486566 and the latch is
false after all 16 solves. Geometry remains positive; this is a physical
nonlinear convergence failure, not an assertion that completed work was
unsuppressed. The prototype is removed from production in build 55; its
exact patch, failure and configuration are retained. Trial 288 was stopped
before GPU measurement on discovery of an unscoped option.

The moderate 3.85x coordinate trial 285 stops on the unchanged disk guard.
After headroom recovery, trial 287 accepts three roots and rejects the
fourth physical step with residual 1.34069e-5 against 1e-5. It is not
promoted. Full material/contact certification and rollback remain intact.
These trials establish failures requiring residual-directed refinement,
not proof that a modest reduction cannot work.

Two closed-output cleanup passes preserve every path, hash and metadata
using verified APFS clones. The first also retires a regenerable validation
pipeline cache and recovers 5,305,237,504 physical bytes (4.94 GiB); the
second recovers 642,596,864 bytes (0.60 GiB). Sources, mass mapping, current
assets and failed evidence are retained. These are historical filesystem
deltas, not claims of current available space. Receipts and exact run pins
are in compact-radix-54.
