# Numi Human persistent-stand common-duration refinement

## Scope and identity

This is a bounded numerical receipt for the persistent Human stand diagnostic.
It is not a passive-standing, anatomy, physiology, walking, safety, or
clinical qualification.

The executable was built from a fresh sparse public checkout of annotated tag
`human-native-postprojection-constraint-diagnostics-20260915`, resolving to
source commit `8b1759e45099c0a770d87eb65257bd184eed90df`. The Apple M4 Pro
binary SHA-256 was
`510f2e9653a973e042bd46fc83f5714d604ee65d2f22680d55cd60821c8ec27b`.
Metal API validation was enabled for every receipt.

The retained input payload SHA-256 values were:

| Payload | SHA-256 |
| --- | --- |
| `myosim-fullbody-core-reference.nhrigid` | `6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44` |
| `myosim-fullbody-muscle-reference.nhmyo` | `9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05` |
| `numi-human-tendon-attachments.nhtendon` | `a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555` |
| `myosim-fullbody-support-contact.nhcnt` | `4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907` |
| `myosim-fullbody-joint-equalities.nheq` | `b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a` |

All four runs used a 6.4 ms horizon, activation `0.8`, 64 coupled sweeps,
source support contact, NHTENDON3 transfer, NHEQ1 equalities, the 40-entry
experimental upper-joint passive configuration, zero root assistance, and
`--persistent-stand-trace`. Each segmented trace was endpoint-bitwise
equivalent to its uninterrupted production horizon.

## Common-duration result

The q/v deltas below compare every common physical sample to the 12.5 us
trace, not only the endpoint. Acceleration is reported separately before and
after the final equality projection because those are different states.

| Timestep | Steps | Kernel peak acceleration (m/s2 or rad/s2) | Published-state peak | Max post-projection source-limit residual | Max tendon force residual (N) | Max common-time q delta vs 12.5 us | Max common-time v delta vs 12.5 us |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 us | 64 | `0.11255022883415222` | `0.11255022400291637` | `2.0995755676267436e-06` | `3.259278673795052e-05` | `3.152100873649033e-07` | `9.489658987149596e-05` |
| 50 us | 128 | `0.11489091813564301` | `0.11489091775729321` | `3.9445185393560678e-06` | `3.259278673795052e-05` | `3.1595077132351435e-07` | `9.547722675051773e-05` |
| 25 us | 256 | `0.30855658650398254` | `0.11592414921324234` | `7.692586223129183e-06` | `6.1154249124228954e-05` | `2.4171285417651234e-07` | `8.21469566290034e-05` |
| 12.5 us | 512 | `1.2236777544021606` | `0.20006611407552555` | `1.5269326468114741e-05` | `6.5185573475901037e-05` | `0` | `0` |

Penetration was zero in all four runs. The 12.5 us trace accumulated
`6.4857646478434881e-06 J` muscle virtual work,
`-9.3408827245791543e-07 J` preload virtual work, and
`6.6992220301644606e-06 J` support virtual work. These continuous-work
figures explicitly exclude impulsive contact and equality projection.

The refinement does not converge under this measured criterion: reducing the
timestep from 100 us to 12.5 us increased the kernel peak by more than an
order of magnitude and the post-projection limit residual by about 7.3x.
Therefore this receipt is a **FAIL** for temporal force/acceleration
convergence, even though initialization, zero penetration, and replay are
individually successful.

## Located constraint interaction

At 12.5 us, the maximum post-projection source-limit residual occurred at
step 281 (3.5125 ms). The pre-projection equality target residual was
`1.5269326468114741e-05`; final equality projection reduced it to
`4.7450411655436397e-13`, while the source-limit residual rose from
`1.5529947328118965e-09` to `1.5269326468114741e-05`.

The corresponding kernel acceleration owner was dependent equality velocity
DOF 113 (joint 141, child body 142). Its initial force-audit components were
`+1177.2114210878403 N` equality force and `-1178.2461639471467 N` limit
force, with a `-2.55709055234488e-05 N` row residual. Across all 128 audit
rows, the largest initial component residual was `0.032620927143028666 N`.

This locates a coupled equality/limit interaction. The next section records
the corresponding smallest production-muscle/passive/contact/equality
discriminator and its high-precision reference. It does not prove that this
row alone causes the whole refinement failure and does not justify adding
regularization or replacing the solver.

## Smallest production-path coupled-solve discriminator

`metalrobo_numi_human_stand_coupling_probe` is a deliberately small,
one-step production-path fixture: a floating root, two scalar joints,
source-style normal contact, passive preload, tendon transfer, one bilateral
equality, and the dependent coordinate's upper limit. It is neither a full
plane-contact oracle nor a long-horizon standing reference.

Two fresh Apple M4 Pro runs with Metal API validation produced byte-identical
stdout (SHA-256
`3ed9e994c2c55411158a252b5271691bd6ead6c0b50cea72335315b2c1ae558a`).
The independent FP64 route derivative was `1.1928223077229005`; its finite
difference was `1.1928223076773659` (absolute difference
`4.55346e-11`). The no-contact production path differed from its FP64
source/equality reference by `8.5276565897629553e-07 m/s2`, with zero
reported equality and tendon residuals.

For the frictionless normal-contact/equality/upper-limit triad, the FP64
Schur/KKT reference had a minimum pivot of `0.36363652396693968` and reported
a maximum regularized-KKT residual of `4.4664279309739511e-18`. This is
evidence that the
small reference instance is not rank-singular; it is not evidence that a
regularization change is appropriate in the production solve.

| Timestep | Metal-to-FP64 velocity difference (m/s) | Gate (tolerance `1e-5`) |
| ---: | ---: | :--- |
| 100 us | `2.5485201911652844e-06` | pass |
| 50 us | `3.285915848120853e-06` | pass |
| 25 us | `5.6662694291367402e-06` | pass |
| 12.5 us | `1.0879757724720971e-05` | **fail** |

At 12.5 us, no supported iteration count (1, 4, 16, 32, or the device ABI
maximum 64) met the `1e-5 m/s` gate; the respective differences were
`1.6560523651669524e-04`, `1.4560544802470855e-04`,
`3.9925146821316106e-05`, `1.5506986074539293e-05`, and
`1.0879757724720971e-05`. That is a bounded numerical failure, not a reason
to silently raise iteration limits or relax the acceptance criterion.

The final equality-coordinate assignment also reproduces the row conflict
directly in the full triad at 12.5 us: it changes the equality target
residual from `2.2428295665122278e-07` to zero, while changing the
source-limit target residual from zero to `2.2428295665122278e-07`.
The no-contact, inactive-limit, and no-equality controls do not show that
trade-off. This supports an ordering/formulation hypothesis for the coupled
equality/limit path; it does not establish a full-Human root cause or a
replacement formulation.

## Passive-coupling discriminator

The configuration reports 40 experimental passive coordinate couplings, but
their maximum generalized force at the accepted state was `0 N`. A paired
0.8 ms, 12.5 us, 64-sweep run with
`--persistent-runtime-without-passive-joint-tissue` differed only by omitting
the runtime passive preload. Its persistent trace and 128-row force audit
were byte-identical to the runtime-passive-on trace
(`a374ccf901f5fe8a78d3b63e479f76b13e12a58460fb4513305f171fc1ddfd0c`).

This demonstrates only that the current runtime preload is inactive for this
accepted short-horizon state. It does not validate the passive model,
establish physiological standing, or rule out a passive effect after a
different preparation or larger excursion.

## Receipt hashes and reviewed-source boundary

| Receipt | SHA-256 of raw stdout |
| --- | --- |
| 100 us / 6.4 ms | `27f9e8d434a7c38e66234fdd523b68caaf18aeaa1e8e04ad85740268fa151a6a` |
| 50 us / 6.4 ms | `1771f75658914993a6bd28dd2a475f260fa4d900aef46a8c76e89286d39a5010` |
| 25 us / 6.4 ms | `922b480768d34e3abf2960bcc3249409675cacd30ddf4fc72e470919e6cb7f11` |
| 12.5 us / 6.4 ms | `db8de20f02a10d5c686ab3133b961ba267bb39e38cb8d301293a2bd16c187644` |
| 12.5 us / 0.8 ms, runtime passive on | `e61a9755158cccaea00f44b06d8bac342852fb30d42929d96d255f685bd8ac5d` |
| 12.5 us / 0.8 ms, runtime passive off | `bdbf73e8cb7aa6948415f21b6f9751135d4ac07d821bd76b3fca9becd8651516` |

The 0.8 ms runtime-passive-on result reproduces a kernel peak of
`0.11579056829214096`, consistent with the retained approximate `0.116`
receipt. The available public and local source histories did not contain the
reviewed `e0d4579` revision. The reproducible 6.4 ms source-bound result here
is `1.2236777544021606`, not the separately reported approximate `4.827`.
Those values therefore cannot be treated as a same-source comparison until
the reviewed runtime revision and its dependencies are published or supplied.

## Reproduction

Build the public source tag, then run each timestep with the same input hashes
and only the timestep/step-count pair changed:

```sh
cmake --build <build-dir> --target metalrobo_numilab_human_myosim_visual_probe
MTL_DEBUG_LAYER=1 <build-dir>/bin/metalrobo_numilab_human_myosim_visual_probe \
  <rigid.nhrigid> <muscle.nhmyo> <output-dir> \
  --tendon-payload <attachments.nhtendon> \
  --support-contact-payload <support.nhcnt> \
  --joint-equality-payload <equalities.nheq> \
  --muscle-step-seconds <dt> --muscle-step-count <count> \
  --muscle-activation 0.8 --persistent-metal-stand \
  --persistent-source-passive-joint-tissue --persistent-stand-trace \
  --stand-contact-iterations 64
```

Use `(dt, count)` values `(0.0001, 64)`, `(0.00005, 128)`,
`(0.000025, 256)`, and `(0.0000125, 512)`. Keep the raw trace, not merely
its summary, when testing a focused coupled-solve change.
