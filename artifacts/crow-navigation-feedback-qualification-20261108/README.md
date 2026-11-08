# Active-head navigation teacher and attribution qualification

Status: development only. The autonomous navigation course is not solved.
The policies in this record are diagnostic candidates and must not be promoted.

The active-head v15 task now has a task-fingerprinted navigation feedback flag.
It prevents generic journey feedback from overwriting route-specific teacher
feedback. Legacy v10 semantics remain unchanged. Explicit teacher control is
permitted for v15; it remains off by default. An old provisional v15 policy is
rejected by the changed task fingerprint. The rebound candidate is a diagnostic
transfer, not proof that the older policy's inputs have equivalent semantics.

On one 180-step training-course trajectory the prior and corrected autonomous
states were byte-identical. With student authority one, physical states also
matched, while wing and yaw teacher labels changed and the head/beak labels
matched exactly. This is a bounded regression check, not a proof over all states.
See result.json and qualify.py.

Strict v15 navigation teacher gating clears generic teacher weights before the
first waypoint and accepts only the prefix through a physically realized
configured waypoint threshold; it rejects failure on the threshold itself.
The default threshold remains five, the full course. A partial threshold is an
explicit curriculum and does not qualify the later course. On a retained native
512-frame, two-update no-waypoint rollout the old attribution produced 512
teacher weights; strict attribution produced zero and the actor remained byte
unchanged. See ../crow-navigation-gating-smoke-20261108/result.json and
tests/crow_navigation_teacher_gating_test.py.

For diagnosis only, a separate metallib replaced the residual navigation teacher
with full flight commands for the original 15 action lanes; head/beak remained
student-controlled. It is not the production metallib. On 16 development lanes
it reached the first waypoint in 5, reached no course completion, and every
first episode ended in physical contact. The diagnostic patch, trace and
results are in ../crow-navigation-direct-teacher-20261108 and
../crow-navigation-direct-versioned-20261108. Calibration and teacher
comparisons are under adjacent crow-navigation-* artifacts.

Two 102,400-sample fixed-learning-rate imitation trials used explicit direct
teacher control, the training course, strict gating, and threshold one. This
is a first-waypoint curriculum, not a full-route expert demonstration.
They used the same initial model and seeds, with global gradient norm limits 1
and 100 respectively; recorded data and commands are in the prefix-training
and prefix-unclipped directories. The saved final rollouts contain 1,331 and
464 qualified samples respectively, not counts for all updates. The first
trial's average reported actor/critic gradient norms over updates range from
0.056–0.176 and 2.18–3.57 respectively; these are minibatch summaries, not
parameter-update magnitudes. Actor maximum absolute parameter changes were
0.00708 and 0.00820; Adam and changing trajectories prevent inferring a
simple proportional effect of clipping. No adaptive learning rate was used.

With teacher OFF and the production shader, all 16 baseline and all 16 limit-1
candidate first episodes reached zero waypoints and zero course completions.
The limit-100 candidate reached waypoint one in 1/16, with zero completions;
all 16 first episodes ended in physical contact. Simulation failed-step count
was zero. The limit-1 candidate had five physical failures and eleven timeouts,
versus 16 physical failures for the baseline. The two layouts and seeds had
already been used in development and are not untouched qualification data.
See ../crow-navigation-prefix-evaluation-20261108 and
../crow-navigation-prefix-unclipped-20261108/result.json.

Validation: native task program check, standalone strict-gating regression,
seven MLX worker tests and eleven recurrent-policy tests passed. Native rollout
results establish neither a robust flying controller nor supported landing.
A reliable flight carrier and multi-waypoint expert/curriculum remain necessary;
then the autonomous candidate needs fresh-seed full-course qualification.
