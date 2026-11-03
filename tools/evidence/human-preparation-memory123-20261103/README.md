# Human preparation memory and native contact work: checkpoint 123

**Partial result. The complete current skin/fat scene has zero accepted physical steps. 0.1x, supported rest, native motion, and presentation throughput are not qualified.**

All execution occurred on SSH Numi: Apple M6, Mac18,5, 16 GiB, macOS27.0(26A425). The execution checkout is `skin-fat-native-integration-209`, HEAD `f75955f08d6edf0f85027eac7614017ac9fbc996` with retained dirty work. Its source pins, binary/library/shader hashes, configurations, exact commands, timing and failed traces accompany each run. This is not a claim that a clean published main passed qualification.

Implemented and tested

- Preparation: avoid a complete authoring-world clone; move NPZ payload allocations; map the compressed archive; release import, support, material-default and authoring scratch at last use. A bounded exact-coordinate/face digest cache reuses only successful repeated exact intersection sweeps; finite, ancestry, topology, orientation, closure and volume checks still run. Failed admissions never cache.
- Metal uploads: private upload staging is bounded to 32 MiB, completed batches release scratch, and initialization remains transactional. Existing active/inactive attachment/contact coupons pass at both 32 MiB and 256-byte staging, including bitwise rejected-state/cursor/history checks. Optional allocation-phase telemetry separates physical/operator/contact/Krylov storage.
- Contact: optional `NUMI_HUMAN_CONTACT_ROW_CACHE=1` combines per-row validation with a current-pose geometry query. Source-order grouped force assembly reuses its force. Audits and wrench reductions skip only finite rows certified inactive by that same encode; active/invalid rows retain the original query and rejection. Scratch is overwritten every encode, with existing ordered encoder dependencies. Filtered/reordered/unsupported layouts retain the original path. Default remains off pending broader accepted-motion qualification.
- Native launcher exposes Matter's existing bounded FGMRES budgets up to 1024 in multiples of16; defaults, restart 16, residual certification and physical acceptance are unchanged. Increasing the budget did not cure this scene's convergence.

Measured evidence and limits

| Measurement | Result | Scope |
|---|---:|---|
| Import scratch release, trial 004 | footprint5.271 ->2.940GB |2.331GB released at last use inside one run; not a matched whole-program memory speedup |
| NPZ reader, three alternating pairs | peak RSS89.19-89.21 ->50.58-50.59MB |about43.3% lower isolated reader RSS, matching payload count/checksum; not whole-scene throughput |
| GPU allocation, full candidate arena |9.93GB |device recommended working set12.71GB; guard retained |
| GPU allocation with contact cache |10.17GB |247,258,832bytes additional derived scratch |
| Matched failed attempt 005 ->006 |51.835 ->31.742seconds |single instrumented pair,1.633x; identical64-column residual trace/rejection; **not accepted-step performance** |
|256-column solve, trial 007 |true residual6.03974695e-5 |from0.709982038; code10 rejection; no accepted physical time |

The production-kernel cache check covers two environments; zero/tile-boundary/large row counts; elastic and exact triangle-bank contact; active/inactive/malformed samples, attachments and bank ranges; source-order forces; fresh cache after node motion. Fused validation flags match the original bitwise. Maximum force error is1.49011612e-8N (relative1.22301988e-7), within the existing1% comparison gate. Original serial versus tiled audit accumulation reaches relative0.000420067377. Source-order force addition is preserved; the first proposed bitwise-force criterion failed because scratch changes rounding, and that failed trial is retained.

The new CMake test target is `numi-human-contact-row-cache-check`; run it with the production `NumiMatter.metallib`. The execution build has `BUILD_TESTING=OFF`, so its attempted CTest invocation found no tests; it is **not** counted as a CTest pass. The built executable was run directly and passed. No paid CI was added.

Trials and blockers

1.001/002/003 stopped on resource guards before physical execution. Restart cleared the recorded53.4 GiB idle decoder allocation; it is separate from measured runtime allocations.
2.004 reached GPU execution and rejected3,631,871broadphase candidate pairs against469,440capacity. Mandatory rollback retained; no physiology/controller advancement is claimed.
3.005 used a4,000,000skin/fat candidate allowance, retained the device guard, and reached real linear-solve exhaustion.
4.006 isolated the first contact cache against005.007 fused validation/query and used256FGMRES/4Newton budgets; the first solve still exhausted its budget. Later encoded tails are not active Newton solves.
5.008 rejected the default regional preconditioner before GPU execution because36-face chunks exceeded170regions.
6.009 used supported256-face chunks and one source-tet 0refinement patch. Region admission passed, then the compiler rejected dense coefficient rows exceeding its checked 384 MiB budget. The current sparse export occurs **after** dense construction. Do not remove this guard or allocate the dense basis on this 16 GiB host. The required implementation is sparse regional construction through combination, mass whitening and GPU export, with dense-oracle Bq/B^Tf/projected-mass/tangent parity before another full run.

These are short initialization/release diagnostics at 15.625 us, not a settled supported checkpoint. No realtime factor is assigned to rejected steps. The0.1x target remains unmet; longer baseline/intervention/video qualification must wait for accepted mechanics and an accuracy-qualified timestep. Physical geometry, material laws, mass accounting, force ownership, source ancestry and acceptance tolerances were not relaxed.

Storage and recovery

The combined verified cleanup and user restart recovered 24.699686912 GB at the post-restart checkpoint. Subsequent builds/trials and OS paging change current free space; `cleanup-final.json.gz` retains fresh byte counts rather than adding logical archive totals. Current sources, accepted anatomy evidence, role inputs and both required all-child mass proofs remain on Numi. 19 historical batches were independently downloaded, archive/member-hash checked and restore tested before retiring exact old paths. Byte-identical transparent compression also retained paths and hashes.

Private recovery release: https://github.com/Numi2/macmini-evidence-archive/releases/tag/numi-20261103-historical-physics-evidence-memory123

`memory123-native009-source.tar.gz` plus its existing per-file manifest and verification receipt preserves 556 native source/configuration dependencies, with no source retirement. Its SHA256 is `e9d6d3adef4309125d0bc3fd00aa7c001527dd4b63dcd4bf01d3d564efdcbafd`. Apple toolchain source is excluded. Background media analysis was restored after measurements; no task-owned native GPU benchmark remains running.

All included `.gz` files are compressed copies of existing configuration, receipt, trace, patch or log formats. Failed and inconclusive trials remain alongside passing controls. This checkpoint is numerical/software evidence, not anatomical motion, physiological, or clinical validation.

### Background decoder root cause resolved

A fresh `sudo launchctl procinfo` mapped the recurrent VTDecoderXPCService
allocation to macOS WallpaperAerialsExtension (responsible PID 463). Selecting
a static black desktop through the public NSWorkspace desktop-image API, then
ending the old aerial client, released its retained surfaces without a reboot.
The post-change snapshot reports kernel_task 237 MiB (previously 8,426 MiB),
6,927 MiB unused physical memory, swap used 486.19 MiB, and 32,866,764 KiB
available disk. No simulation was running. These are host-memory observations,
not solver performance. The old wallpaper configuration is retained in the
remote diagnostics; no login/security setting or system service was disabled.
The idle-screen configuration is unchanged. Native trials should use a bounded
task-owned `caffeinate -di` while running. See `wallpaper-memory-resolution.json.gz`
and `decoder-procinfo.txt.gz`.

### Certified closed-solid contact exclusion

`NUMI_HUMAN_CONTACT_AABB_EXCLUSION=1` is an additional opt-in requiring the
admitted `NUMI_HUMAN_CONTACT_ROW_CACHE=1` layout. It returns exact zero unilateral
closure only for points separated from a certified closed/oriented rigid bank's
AABB by the existing Float32 guard. Points inside/near the box and uncertified
banks retain complete nearest-feature queries. The existing cache and direct
paths remain available as controls. Material laws, mass, timestep, contact row
identity, physiological state and physical acceptance thresholds are unchanged.

The production GPU test passes closed interiors, exterior queries inside the
50mm band, uncertain near-boundary points, uncertified banks, malformed ranges
and root metadata, source-order loads, reactions and changed-node cache refresh
across two environments. Maximum force error is 1.49011612e-8 N absolute and
1.22301988e-7 relative; the existing 1% comparison passes. The first new fixture
failed because its attachment triangle was collinear; that failed log is retained.

Same-build, same-input trials011(control) and012(exclusion) took32.100873917s
and27.023549791s per rejected root: one instrumented pair, **1.187885x**, not a
sustained-throughput qualification. All64 true-residual cycle records match
exactly, including exhaustion after256 columns and final valid residual
6.03974695e-5. Both accept **zero** physical steps. No accepted-time real-time
factor is reported. Initial trial010 agrees; the final source additionally
checks malformed root metadata before tighter exclusion, validated by the
retained GPU test. Native013 separately tests this final source at an existing
1024-column diagnostic budget, without promoting that budget to a default.

The runtime currently still requires dense basis coefficients. Compiler sparse
fields and export helpers alone do not make the full Human basis usable. Sparse
construction **and runtime consumption** remain required; neither the384MiB
coefficient guard nor the170-region bound was removed.

### Inexact Newton linear-certificate experiment

The strict1024-column diagnostic013 still rejected the step:64active restart
cycles in one Newton solve, final valid true linear residual1.36105709e-5
against1e-5, and59.553241708s for the authoritative failed segment. It accepted
zero physical time. Raising iteration limits does not resolve this blocker.

`NM_FGMRES_INEXACT_COUPLED_LINEAR_CERTIFICATE=1` is a separate default-off
experiment. Admission requires existing EW forcing, relative inner stopping,
distributed reductions, the independent nonlinear domain latch, and complete
coupled true-residual certification. Internal layout bit24 changes the linear
`||b-A*dx||` target to adaptive relative forcing. It does not change material,
force, mass, contact, timestep, final mechanical/pressure/vascular/transport
residual thresholds, or accepted-state rollback. Every restart completion still
uses freshly recomputed full-operator residuals. Nonlinear acceptance remains
independent; a permitted inexact linear correction is not an accepted step.

The production GPU focused test passes1084checks, including the unchanged
strict default, true-versus-Arnoldi residual disagreement, malformed/nonfinite
rejection, separate mechanical/vascular nonlinear gates, mixed pressure,
support admissibility, original RHS normalization and EW history handling.
Run `numi-matter-fgmres-true-residual-native-check --inexact-coupled-certificate-only`.
The first broader test exposed a function-constant pipeline dependency; its
failed log is retained. The final implementation uses existing internal layout
metadata and introduces no new function constant or public artifact/API.
Full Human trial014 remains a separate qualification; coupon success is not
physical convergence or throughput evidence. The0.1x target remains unmet.

### Complete solver regression and further archival cleanup

The wider production GPU checker passes77,514checks. Its first run exposed
stale telemetry fixtures: aggregate ICB range, exhaustion, domain totals and
cycle-finalize fixtures omitted the current per-microtick buffer/indices.
These bindings are repaired and per-tick/total parity is checked. No production
physics or acceptance gate was changed to make this suite pass. Both failed
suite logs and the isolated failing ICB log remain alongside the passing run.

Historical cleanup now has24 independently downloaded, all-member-hash checked
and restore-tested batches through026. The final versioned summary is uploaded
to the existing private recovery release; a duplicate-name summary upload
failed after the verified retirements, so the prior release summary is preserved
and `summary-postbatch026.json` is the current consolidated receipt. The initial
retirement script also met a read-only directory at022; that entire downloaded
archive was reverified before completing its exact protected-path-checked
retirement and publishing the recovery receipt.

The post-cleanup snapshot has36,074,360,832bytes available, exceeding the
12,105,162,752byte starting snapshot by23,969,198,080bytes. This net disk change
includes OS swap changes; it is not all historical-file deletion. Retired old
paths account for14,175,891,456allocated bytes across529distinct inodes;
APFS extents may be shared. Earlier byte-identical transparent compression is
recorded separately. Current source/asset hashes were excluded from retirement.
152of153roster paths exist; the historical native-import-120/preparation.py
pathname is absent and does not occur in any retirement manifest. Current
execution inputs are independently pinned and validated by the native loader.
The aerial decoder remains absent. Trial014 is the next bounded full Human
convergence diagnostic; no accepted motion or0.1x performance is claimed yet.

### Parent-input recovery and first inexact integrated trials

Correction to the earlier protection-list statement: trial014 failed before GPU
execution because four active parent-mesh files nested under the anatomy
receipt were omitted from the protection roster and archived in batch025.
They were recovered from the independently downloaded archive at their exact
original paths and hashes. All eight parent source hashes now pass; both
canonical and aliased paths and content hashes are protected, and the runner
checks these transitive input pins before launch. The failed014 log, original
roster, recovery receipt and corrected roster are retained. No missing source
was substituted or synthesized. The original cleanup receipts remain historical
records rather than a claim that this omission did not occur.

Trials015/016 passed loading and reached the coupled GPU solve. Each used two
actual Krylov columns in one active Newton solve and accepted zero physical
steps. The complete true linear residual was0.0966456309 from RHS0.709982038;
that meets the guarded relative linear certificate, but is not the final
nonlinear physical tolerance. The first shared line-search fraction becomes
zero and the physical step rejects code10. Failed authoritative segments took
24.135009208s and23.944809791s, respectively.016 enabled the existing fresh
common-alpha reset; its unchanged failure does not support that hypothesis.
A bounded existing alpha/vascular trace is the next diagnostic, not another
iteration-budget increase.

The new linear policy now contributes an explicit INEXCER1 execution-fingerprint
marker. The final full production GPU measurement checker passes77,514checks;
the existing fresh-direction reset focused checker passes28checks. These are
numerical and scheduling fixtures, not full-Human convergence or throughput.

Trial015 inherited four output environment paths from013. Its actual failure
and reference-pose diagnostics were copied into015/inherited-diagnostic-outputs
with an ownership notice.013 stdout/stderr/source/config were separately
preserved before015 and remain valid; the overwritten diagnostic output paths
must not be treated as immutable013 evidence.016 rebases all four output paths.
The runner's legacy source_import_pass Boolean requires an absent log token;
false there is not a loader failure when the same trial reached GPU execution.
The0.1x, motion and physiological qualification requirements remain unmet.

### Reference-frame collision-floor fix

Trial017's bounded alpha snapshots locate the first zero at the deformable
contact limiter, before rigid, support or vascular line search.018 identifies
all32 lane minima as initial-hit fractions on the bonded skin/fat owner15.
One retained VT witness (contact1714, active slot180) has a1.9333072um gap
but receives a3.20477875um arithmetic floor derived from the world position
at y=-1.68022704m. Its CCD geometry is already evaluated in reference-local
coordinates. This mixes coordinate frames and creates a zero-time numerical
hit before the coupled Newton correction can progress. It is distinct from
an actual source intersection and from the authored clearance requirement.

The limiter now derives its roundoff scale from the same reference-local
projected start/end coordinates used by CCD when every support is in
compensated FEM reference mode. Nonreference and MPM contacts retain their
exact world-coordinate calculation. Authored minimum separation remains an
independent lower bound; material/contact forces, actual geometry, masses,
nonlinear residuals, true intersection tests and rollback are unchanged.

A production-limiter GPU regression fails on the old shader with
"reference contact uses world-coordinate roundoff floor" and passes384checks
with the fix. It covers rigid world translation of authoritative compensated
coordinates, crossing motion, a larger authored clearance, a closing initial
contact, nonfinite direction rejection, untouched physical state and preserved
legacy behavior. The complete GPU checker now passes77,895checks. Full current
Human trial019 separately tests the fix; it is not yet accepted-motion or
performance evidence. No0.1x result is claimed.

Completed task-owned diagnostic runs001-018 also had32byte-identical large
output duplicates replaced atomically by hard links after per-file SHA checks.
Sources, assets, mass proofs and active-run files were excluded. Frozen shared
outputs are read-only, and the next runner rejects output environment paths
outside its newly created run root. The retained manifest records each alias
and hash. Available disk after this operation was39,378,882,560bytes, a net
27,273,719,808bytes above the original starting snapshot; OS paging still
changes current availability during execution.

### Compensated contact-angle consistency

Full-scene019/020 gets past the earlier false initial contacts but still accepts
zero physical time. It rejects at skin/fat VT contact1589443, active slot265372,
with a finite, certified collision-free trial. A sparse020 trace shows trial
energy remains about1.5e-5 relatively above base even as alpha tends to zero.
The high/low position expansions converge correctly: this is not an actual
finite jump of the skin. Rounding nearly parallel feature edges before their
angular mollifier changes the projected edge representation at a halfway case.

A captured production-helper GPU regression fails on the old angular path.
The new overload keeps compensated edges through the cross product, squared
norms and analytic angle derivative. Assembly and trial energy share this
calculation for reference-mode contact; the existing Float32 nonreference
path and the dimensionless C1 mollifier equation are retained. No clearance,
physical acceptance threshold, force row or anatomical geometry is removed.

The focused checker passes140checks: captured base/trial continuity, angular
values and gradients against Float64 analytic equations, three length scales,
parallel/antiparallel neighborhoods, zero edges and nonfinite rejection. The
complete production kernel checker passes78,032checks. Earlier pre-assertion
fixture logs are retained as diagnostics; the v2 logs are the full equation
checks. This qualifies the numerical helper, not whole-Human dynamics.

The failed019segment took25.486149791s;020reproduces the same failure. Both
accepted zero physical steps and therefore provide no achieved real-time factor.
Trial021 is running with the new angular path and otherwise matched physical
settings. The0.1x, accepted-motion and physiology requirements remain open.
