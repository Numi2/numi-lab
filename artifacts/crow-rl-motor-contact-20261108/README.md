# Joint motor/contact primitive passes physical gates; crow integration remains open

The native slider diagnosis confirms a real one-way split inconsistency. All 96
registered conditions are accepted. In six unsaturated contact conditions, motor
force is 13.4415–62.4060% below the force law at accepted q/v. Free and continuously
saturated controls agree within 7.912e-10 N; momentum balances within 1.573e-12 Ns.
This identifies the split's inconsistency for the registered implicit step,
not the complete cause of the hybrid crow's tilt.

A motor impulse can be represented by the existing canonical scalar ConstraintIR:
impulse bounds are +-h*cap, target velocity is K*(target-q0)/(D+h*K), and source
compliance is h/(D+h*K). The owning native quality solver then solves motor and
contact together. The host authors this manufactured row and analyzes observations;
it never advances physical state.

## Controlled revision and limits

The first joint trial stops before motor rows: empty-contact quality assembly
reads an uninitialized physical mass factor, because its indirect point dispatch
is zero. MetalWorld now dispatches the factor unconditionally for quality mode.
TemporalCone keeps the original indirect behavior. The failed free-flight case
and the native contact owner pass after this fix. Full Release succeeds.

The continued comparison stops on a published fixed-force control whose momentum
residual is 2.16473e-9 Ns, above the frozen 2e-9 gate. Tightening invocation-only
optimality tolerance from 1e-7 to 1e-10 fixes that engineering replay. Feasibility
and all physical gates remain unchanged. No global solver defaults change.

A separately sealed comparison on fresh masses, step sizes and targets accepts
all 144 native conditions. Candidate maximum force-law error is 5.507e-10 N,
maximum momentum residual 1.680e-12 Ns, q-hv error 1.241e-14 m and contact
penetration 1.426e-10 m. The fixed-force quality control agrees with the original
split force within 1.193e-9 N: changing the contact solver alone does not restore
motor feedback. See `precision/analysis.json` and the frozen plans/predictors.

The complete registered benefit prediction is nevertheless **contradicted**:
one of six primary reference cases is already saturated and passes the absolute
force gate. Its error is numerical noise, so relative error reduction is 96.8784%,
below the frozen 99% requirement. All candidate absolute physical gates pass;
all six primary cases remain in `precision/endpoint-details.json`. This post-trial
explanation is descriptive and does not replace the failed registered endpoint.

A fresh default 640-transition crow replay matches all ten archived initial,
trajectory and final files byte for byte (including the six required state/trace
files). Protected deployment97 is unchanged. No policy, profile or experimental
motor mode is promoted.

These are finite single-DoF engineering conditions with zero generalized
friction work. The motor row is authored by this probe; native crow drive-row
generation, dry friction, coupled joints, hybrid impact/resting-contact timing,
reset/checkpoint/partition contracts, energy and throughput remain unfinished.
In hybrid CCD, pre-impact smooth motor force and an instantaneous impact impulse
must retain their distinct timing; this conventional implicit-step result does
not authorize feeding an endpoint impulse backward through pre-impact motion.
The next integration must solve persistent contact and motors together while
preserving that event timing and requalifying the actual crow fixture.

Every failed build/helper, partial trial, rejected prediction, exact runtime and
native outcome is preserved through `payload/manifest.json`. Actual Git readback,
executed native recovery and exact cleanup are documented in the
[recovery controls](../crow-rl-git-recovery-motor-contact-20261108/README.md).
