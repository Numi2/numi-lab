# Numi Human constraint-impulse trace receipt

## Scope and identity

This receipt adds read-only owner diagnostics to the persistent Numi Human
stand trace. It is a bounded numerical debugging receipt, not a
passive-standing, anatomical, physiological, walking, safety, or clinical
qualification.

The source implementation is commit
`826b3029a9f8372b564f0729917cbd20e680b419` on public branch
`human-persistent-constraint-impulse-trace-20260915`. It was made from the
previous published Human runtime package commit
`489dd2c7ffb8889a771090bce35fe2d005d6af7b` without changing the solver's
candidate velocity, iteration count, constraint ordering, tolerances, or
acceptance policy.

The ABI advances from v5 to v6 and retains, for every accepted stand step:

- maximum normal and tangential contact impulses with support-contact owners;
- maximum and total absolute source-limit impulse with a source velocity-DOF
  owner; and
- the equality record owning the largest absolute bilateral impulse.

The host rejects non-finite diagnostics and out-of-range owner indices. The
segmented trace schema is now
`numi.human.persistent-stand-trace.v3`. The focused
`metalrobo_numi_human_stand_coupling_probe` asserts the new contact, source
limit, and equality owner fields while preserving its existing FP64 gate.

The full-body input hashes are unchanged from the source package:

| Payload | SHA-256 |
| --- | --- |
| `myosim-fullbody-core-reference.nhrigid` | `6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44` |
| `myosim-fullbody-muscle-reference.nhmyo` | `9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05` |
| `numi-human-tendon-attachments.nhtendon` | `a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555` |
| `myosim-fullbody-support-contact.nhcnt` | `4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907` |
| `myosim-fullbody-joint-equalities.nheq` | `b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a` |

## Local full-body trace

A local Apple M4 execution used the source-bound 6.4 ms configuration:
512 steps at 12.5 us, activation `0.8`, 64 coupled sweeps, source support
contact, NHTENDON3 tendon transfer, NHEQ1 equalities, experimental passive
joint tissue, zero root assistance, trace capture, and deterministic replay.
It produced 513 samples and a bitwise-equivalent segmented endpoint, with
zero reported penetration. Each sample retains the full q/v trajectory,
constraint residuals, tendon residuals, impulses, owners, and continuous-work
terms. The uncompressed stdout SHA-256 is
`fe31eb64173d3479ee2b65e6f95c66199764d2f182680a9d3aaf2d884822a239`;
the retained compressed transcript is
[`local-m4-stdout.txt.gz`](media/numi-human-constraint-impulse-trace-20260915/local-m4-stdout.txt.gz)
(SHA-256 `db38f8912447369ee6d1d658ec61a507a96a0190a68cd4886eb4c2492115a0c9`).

The largest kernel acceleration and post-projection source-limit residual in
this trace occurred together at step 279 (3.4875 ms):

| Observation | Recorded value |
| --- | ---: |
| Kernel acceleration | `1.2233552932739258 m/s2 or rad/s2` |
| Acceleration owner | equality 43, dependent velocity DOF 113 |
| Maximum source-limit impulse | `9.796405720408075e-06 Ns or Nms` |
| Source-limit owner | DOF 113 |
| Largest equality impulse | `9.643103112466633e-06 Ns or Nms` |
| Equality-impulse owner | equality 43 |
| Maximum normal-contact impulse | `9.024464020512823e-07 Ns`, contact 5 |
| Maximum tangential-contact impulse | `3.339825696002663e-07 Ns`, contact 3 |
| Pre-projection equality target residual | `1.526547930552624e-05 m/s or rad/s` |
| Post-projection equality target residual | `2.3925024288107277e-13 m/s or rad/s` |
| Post-projection source-limit target residual | `1.5265481124515645e-05 m/s or rad/s` |

The largest normal impulse anywhere in the trace was
`2.743953018580214e-06 Ns` at step 319, support contact 5. The maximum
tendon-force residual was `6.823938019806519e-05 N` at step 475. Continuous
generalized virtual work summed to `6.362753090252625e-06 J` for muscle,
`-9.219922710208107e-07 J` for preload, and
`6.664269483200882e-06 J` for support. Those work totals explicitly exclude
impulsive contact and equality projection.

This narrows the measured production interaction to equality 43 / source
limit DOF 113 at the acceleration event and records the simultaneous contact
owners. It does not establish that those rows alone cause the temporal
refinement failure, nor does it supply a full FP64 contact/friction/RHS/time
integration reference.

### Same-trace FP64 two-row check

The existing read-only reference route was rerun at this exact trace point:
step 279, equality 43, and its dependent upper-limit DOF 113. It independently
obtained an equality derivative of `0.024761449566623395`; a centered finite
difference gave `0.024761449638269203` (absolute difference
`7.1645807342424206e-11`). The two-row Delassus correlation was
`-0.99997094609204384`, with positive eigenvalues
`0.0029024017843966021` and `199.79134150280635`, condition number
`68836.555495827808`, and a minimum articulated Cholesky pivot of
`0.010022659996858297`.

The raw result is retained as
[`step-279-fp64-rank-audit.txt`](media/numi-human-constraint-impulse-trace-20260915/step-279-fp64-rank-audit.txt)
(SHA-256 `65ab14651c592cdc8f36ff7846fc385a679c1773c760a584aa942de33b22189d`).
This confirms a strongly coupled but non-singular two-row operator at the
observed event. It still excludes contact/friction rows, the production
muscle/passive-force right-hand side, unilateral complementarity, and time
integration; `full_active_set_qualified` remains false.

## Physical M4 Pro execution control

The Mac mini data volume had only about 390 MiB free and its existing Human
worktree was dirty, so this receipt did not create another source worktree or
claim a remote source build. Instead, an execution-only bundle of the exact
local arm64 binary, `libmetalrobo.dylib`, `MetalRobo.metallib`, and the pinned
inputs was copied into an isolated temporary directory. Their SHA-256 values
were verified byte-for-byte before execution. Metal API validation was enabled.

| Bundle member | SHA-256 |
| --- | --- |
| full-body trace executable | `9dc70fe94a9531b9412b1bb08ebf57deb288a0e866c474186f1270359290ab9a` |
| focused coupling executable | `b5046e5787953bec0ed5b69dc28bfe9b138baadb394173b8237b2ad4cf618953` |
| `libmetalrobo.dylib` | `e6b618ca482ddf9d628db866b71a4f4773e362e20d9d31c820b95f40ea02cb85` |
| `MetalRobo.metallib` | `c33d5223ccb1eab5ea784221074b811ce602996100cc0452dd1ac37c6c7fc3bb` |

The physical M4 Pro ran the focused production-path coupling probe successfully.
Its stdout and validation stderr are retained as
[`m4-pro-execution-stdout.txt`](media/numi-human-constraint-impulse-trace-20260915/m4-pro-execution-stdout.txt)
(SHA-256 `95cc267b37559d4d685d20b2cd2d74b2646b71498f60a33af0b72fb7cf37332e`)
and
[`m4-pro-execution-stderr.txt`](media/numi-human-constraint-impulse-trace-20260915/m4-pro-execution-stderr.txt)
(SHA-256 `94a6e1bb4dfff9e2216c9d7e05d898e64b45330607d4ae3976ec287627ceb96c`).

The existing 12.5 us full-triad FP64 comparison remains a **FAIL**:
`1.0879757724720971e-05 m/s` exceeds its declared `1e-05 m/s` gate. The
execution also retained the established source-limit/equality projection
trade-off. This is intentional evidence preservation, not a re-baselined pass.

An initial full-body attempt was stopped when a separate
`human-coupled-velocity-closure-control-fixed-512` process acquired the
device. After that owner released it, the exact full 6.4 ms configuration ran
to completion on the M4 Pro with Metal API validation. Its raw stdout is
retained as
[`m4-pro-full-stdout.txt.gz`](media/numi-human-constraint-impulse-trace-20260915/m4-pro-full-stdout.txt.gz)
(uncompressed SHA-256 `ceece3bc8d2dd835e80cb133dec994010d30be987594034dfb44586c85036276`,
compressed SHA-256 `1b813581fdde17a1df4468dbe4778e9f44faf615df132115e461f962ce7f98ef`)
with its validation stderr in
[`m4-pro-full-stderr.txt`](media/numi-human-constraint-impulse-trace-20260915/m4-pro-full-stderr.txt)
(SHA-256 `079c67292998a3e5c60d6d987eb13e5114412c0e5c7403ca4533b3e0069e502b`).

The 4,240,792-byte serialized `persistent_stand_trace` payload from the
physical M4 Pro has SHA-256
`f1c5177007e7085aacf226b498c2fab81366d981cf05e5c39af87a630da733e8`
and is byte-identical to the local Apple M4 payload. It records the same
513 samples, bitwise segmented endpoint, zero penetration, step-279
equality-43/source-limit-113 event, contact owners, tendon residual, and
continuous work values listed above. Whole stdout is not claimed
cross-hardware-identical because it includes host-specific rendering and
timing fields.

This is a physical-M4-Pro full-body execution receipt, but not a remote
source-build qualification: the device had insufficient capacity for another
worktree and ran the verified exact local arm64 bundle listed above.

## What remains failed or unqualified

The prior common-duration refinement result remains failed; this receipt runs
only its 12.5 us member and does not reevaluate 100, 50, and 25 us. The
recorded local trace is deterministic and diagnostically richer, but it does
not demonstrate temporal convergence, passive anatomical calibration,
physiological standing, or a general coupled-solver correction. The next
reference should use the now-recorded full production owner identities and
retain the force, contact, equality, limit, tendon-residual, and work terms
at the same trace point.
