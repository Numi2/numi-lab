# Numi Human fibre-repair common-duration refinement receipt

## Scope

This is a bounded numerical-debugging receipt for the published native Human
runtime. It reruns the same 6.4 ms persistent production horizon at four
timesteps after the accepted-static-fibre handoff and stationary-fibre repair.
It is not a passive-standing, anatomical, physiological, walking, safety, or
clinical qualification.

The local Apple M4 executable was built from
`cced6011f97a4fdcf0c8b6b3f061e19d4d381a14` and has SHA-256
`9dc70fe94a9531b9412b1bb08ebf57deb288a0e866c474186f1270359290ab9a`.
The independent FP64 reference probe has SHA-256
`31f3fbb283668cb51f267d538ffd46b498bc7b2e5c78ca5253c4d6ab54fae24f`.
Both are source-bound local Apple-M4 executables; this is not a new physical
M4 Pro source-build claim.

Every member uses the same pinned NHRIGID2, NHMYO2, NHTEND3, NHCNT1, and
NHEQ1 inputs, activation `0.8`, zero root assistance, source support contact,
tendon transfer, NHEQ1 equalities, experimental passive joint tissue, 64
coupled sweeps, persistent trace capture, and Metal API validation. Only the
timestep/step-count pair changes:

| Timestep | Steps | Duration |
| ---: | ---: | ---: |
| 100 us | 64 | 6.4 ms |
| 50 us | 128 | 6.4 ms |
| 25 us | 256 | 6.4 ms |
| 12.5 us | 512 | 6.4 ms |

The 100 us receipt also requested `--stand-deterministic-replay`, which runs
a second unchanged horizon. That extra replay does not alter the captured
trace. All four members independently report a bitwise-equivalent segmented
endpoint against their uninterrupted horizon.

The input identities are unchanged from the native runtime package:

| Payload | SHA-256 |
| --- | --- |
| `myosim-fullbody-core-reference.nhrigid` | `6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44` |
| `myosim-fullbody-muscle-reference.nhmyo` | `9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05` |
| `numi-human-tendon-attachments.nhtendon` | `a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555` |
| `myosim-fullbody-support-contact.nhcnt` | `4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907` |
| `myosim-fullbody-joint-equalities.nheq` | `b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a` |

## Common-duration result

All four traces had zero reported penetration and a bitwise segmented endpoint.
The peak kernel acceleration, source-limit residual, tendon-force residual,
and active owner do not converge consistently as the timestep is reduced.

| Timestep | Samples | Peak kernel acceleration and owner | Maximum post-projection source-limit residual | Peak tendon-force residual |
| ---: | ---: | --- | ---: | ---: |
| 100 us | 65 | `0.11560563743114471`, step 57, position-limit velocity DOF 39 | `2.1077732981211739e-06` at step 59 | `3.49622787325643e-05 N` at step 62 |
| 50 us | 129 | `0.1143425852060318`, step 1, position-limit velocity DOF 80 | `3.9441106309823226e-06` at step 70, equality 50 / DOF 127 | `3.49622787325643e-05 N` at step 114 |
| 25 us | 257 | `0.30844077467918396`, step 123, equality 50 / DOF 127 | `7.6894739322597161e-06` at step 123, equality 50 / DOF 127 | `6.1154249124228954e-05 N` at step 184 |
| 12.5 us | 513 | `1.2233552932739258`, step 279, equality 43 / DOF 113 | `1.5265481124515645e-05` at step 279, equality 43 / DOF 113 | `6.8239380198065192e-05 N` at step 482 |

The accompanying impulse and continuous-work terms also change materially.
The work scope is continuous generalized source-muscle, preload, and support
virtual work; it explicitly excludes impulsive contact and equality projection.

| Timestep | Max normal impulse | Max tangential impulse | Muscle work | Preload work | Support work |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 us | `4.7864855332591105e-06 Ns`, contact 5 | `9.217138199346664e-07 Ns`, contact 5 | `2.645612469111926e-06 J` | `-1.3188758792249133e-06 J` | `4.8641093324128757e-07 J` |
| 50 us | `3.9602405195182655e-06 Ns`, contact 5 | `8.423256758760545e-07 Ns`, contact 5 | `2.1045252189185443e-06 J` | `-8.211257451577392e-07 J` | `6.463497322817973e-07 J` |
| 25 us | `3.0399664865399245e-06 Ns`, contact 5 | `7.118009648365842e-07 Ns`, contact 5 | `3.3136719209093567e-06 J` | `-9.047490767728079e-07 J` | `4.4109874752766395e-07 J` |
| 12.5 us | `2.743953018580214e-06 Ns`, contact 5 | `6.064677791073336e-07 Ns`, contact 5 | `6.362753090252625e-06 J` | `-9.219922710208107e-07 J` | `6.664269483200882e-06 J` |

At exact common physical times, the terminal configuration differences against
the 12.5 us trace are small but are not sufficient evidence of force
convergence. The maximum q/v differences over the common samples are also
non-monotone between 100 and 50 us.

| Coarser trace vs. 12.5 us | Largest q difference | Largest v difference | Endpoint q difference | Endpoint v difference |
| --- | ---: | ---: | ---: | ---: |
| 100 us | `3.1674795764047303e-07` at 6.4 ms | `9.500290525465971e-05` at 4.4 ms | `3.1674795764047303e-07` | `6.686326196359005e-05` |
| 50 us | `3.188994028846537e-07` at 6.4 ms | `9.590744957677089e-05` at 4.55 ms | `3.188994028846537e-07` | `4.47530019300757e-05` |
| 25 us | `2.420013061055215e-07` at 6.4 ms | `8.236428402597085e-05` at 3.975 ms | `2.420013061055215e-07` | `3.6018635910295416e-05` |

This is therefore a **FAIL** for common-duration temporal force consistency:
the same repaired source and inputs have a peak acceleration that rises from
`0.1143425852060318` at 50 us to `1.2233552932739258` at 12.5 us, while the
peak post-projection source-limit residual rises from
`3.9441106309823226e-06` to `1.5265481124515645e-05`. Zero penetration,
endpoint replay, and small terminal q/v deltas do not turn that failure into a
dynamic, standing, or physiological pass.

## Read-only FP64 constraint inspection at the 25 us event

The exact 25 us event was inspected at step 123 with equality 50 and its
dependent upper-limit DOF 127. Its dependent coordinate is q 128 and master
coordinate q 120; the equality derivative is `0.024761449406654566`. A
centered finite difference reports `0.024761449478327698`, an absolute and
relative difference of `7.1673132706617793e-11`. The two-row Delassus
correlation is `-0.99997094619206084`; its positive eigenvalues are
`0.0029023917927446519` and `199.7913414957514`, for condition number
`68836.792467228675` and minimum articulated Cholesky pivot
`0.010022659996858268`.

The broader read-only equality/near-boundary-limit snapshot has 51 equality
rows, 54 near-boundary limit rows, and 105 operator rows over 128 velocity
coordinates. Its declared normalized pivot cutoff is `1e-10`; the audit
reports numerical rank 81, retaining 41 equality and 40 near-boundary-limit
directions. It reports maximum normalized off-diagonal correlation
`0.99999290632891391`; its most-coupled pair is equality 20 / upper-limit
DOF 28 with condition `281940.46256380697`.

Those checks establish finite derivatives, high coupling, and rank structure
at the observed trace state. They deliberately do not include contact or
friction rows, the production muscle/passive-force RHS, unilateral
complementarity, or time integration. They neither validate a regularization
choice nor identify a coupled-solver replacement.

## Retained receipts

The three new local runs have Metal-validation stderr and complete compressed
stdout transcripts. The existing 12.5 us member remains the v5 local trace in
[`NUMI_HUMAN_CONSTRAINT_IMPULSE_TRACE_20260915.md`](NUMI_HUMAN_CONSTRAINT_IMPULSE_TRACE_20260915.md).

| Receipt | Uncompressed stdout SHA-256 | Retained artifact SHA-256 |
| --- | --- | --- |
| 100 us | `762a8c6dd511f488d166f2383497f34457ecf3b40c70d6b2e6de63f51fc0d851` | [`local-m4-100us-stdout.txt.gz`](media/numi-human-fibre-repair-common-duration-refinement-20260915/local-m4-100us-stdout.txt.gz) `02fd3b027df737509a73cac60a40f5c1f3ba77dd232002e3768973c126a1ee76` |
| 50 us | `816eb7a8e7a370dafe02a6cec97afcedef82675ebeaac645ff894f0b22a18a83` | [`local-m4-50us-stdout.txt.gz`](media/numi-human-fibre-repair-common-duration-refinement-20260915/local-m4-50us-stdout.txt.gz) `be54ff09d3cd30a987119a4afd1e1e6ab847c6b16f86b6c950b67464cc2f2901` |
| 25 us | `aefadd9def76d04fe96276e182891ebfc1c77f0ea977235307058d24a6aa4ca9` | [`local-m4-25us-stdout.txt.gz`](media/numi-human-fibre-repair-common-duration-refinement-20260915/local-m4-25us-stdout.txt.gz) `bac692ac8b068480a5c64a1b6251c44c3c4a63ae143a1eb7b0cc925d6cbf29d1` |
| 12.5 us | `fe31eb64173d3479ee2b65e6f95c66199764d2f182680a9d3aaf2d884822a239` | [`local-m4-stdout.txt.gz`](media/numi-human-constraint-impulse-trace-20260915/local-m4-stdout.txt.gz) `db38f8912447369ee6d1d658ec61a507a96a0190a68cd4886eb4c2492115a0c9` |

The validation stderr hashes are
[`100 us`](media/numi-human-fibre-repair-common-duration-refinement-20260915/local-m4-100us-stderr.txt)
`5ecbfa87bafd4aa616ada43beca27d8d4c12c73c96304629fc1424d1e98a56e6`,
[`50 us`](media/numi-human-fibre-repair-common-duration-refinement-20260915/local-m4-50us-stderr.txt)
`ddded132ffe632791ed48c2fd674158f740825becd6b8eab5762ba0906fd6f4c`,
and
[`25 us`](media/numi-human-fibre-repair-common-duration-refinement-20260915/local-m4-25us-stderr.txt)
`2e4bd7663775394b44d93f95e5d9fb74dda1e0796c7925fa71acad1c38c354a7`.
The two FP64 raw results are
[`two-row rank audit`](media/numi-human-fibre-repair-common-duration-refinement-20260915/local-m4-25us-step123-equality50-rank-audit.txt)
`6e89d5d976d34ce3aa66cfadd6becb57e658d64f170aa6520bc4b2b64a3e7d50`
and
[`full equality/limit active-set audit`](media/numi-human-fibre-repair-common-duration-refinement-20260915/local-m4-25us-step123-active-set-audit.txt)
`50958cd9e2c6e81f373904dc5f79632b36d72f6359fadfdb0def5c5c0a310319`.

## Next bounded discriminator

No solver, regularization, threshold, or passive-stiffness change was made.
The appropriate next experiment is a small production-path reference that
captures the actual force RHS, contact-surface target, friction, equality,
and source-limit state around both observed event families: equality 50 / DOF
127 at 25 us and equality 43 / DOF 113 at 12.5 us. It must compare against a
declared high-precision reference before a coupled formulation is changed.
