# Numifly No Legs

Numifly No Legs is a separate robot identity: `numifly-no-legs`. It derives
from Numifly's 9%-scale G1 upper body and the same provenance-bound bilateral
Maeda wing pack, but removes both complete six-link leg subtrees from the
authored mechanics and presentation.

The retained morphology is the floating pelvis, three-joint waist, torso,
head, both seven-joint arms/hands, and the live bilateral wing surface. The
pelvis remains because it is the floating-root body that owns the upper-body
articulation. No hip, knee, ankle, sole, foot collision, leg joint, leg action,
or leg visual pack remains.

## Separate compiled contract

- robot: `numifly-no-legs`;
- task: `numifly.no_legs.flight.v1`;
- 18 articulated bodies and 17 joints;
- `nq = 24`, `nv = 23`;
- 17 upper-body position drives plus 20 measured-wing residual actions;
- 37 actions total;
- a separate RobotPack, run/task/sensor/reality identities, fingerprints, and
  Numi Window catalog scene;
- separate policy compatibility through the `numifly-no-legs` robot ID.

The compiler removes topology rather than disabling or hiding it. It rebuilds
body, joint, generalized-coordinate, collider, and collision-exclusion indices
and rejects the morphology unless exactly the canonical 12 G1 leg bodies and
12 leg joints are removed. The generated visual URDF independently requires
the same 12-link/12-joint removal before cooking presentation packs.

## Wings and evidence

The right wing remains the measured Maeda surface and the left remains its
documented sagittal reflection. Live deformation, aerodynamic loading, torso
wrench application, accepted/candidate/checkpoint transactionality, and the
borrowed-encoder visual presentation are identical in ownership to Numifly.
The new robot does not duplicate or fork a shader.

The deterministic 256-candidate, 18-step Metal force-balance probe measured:

- mass `0.0138285 kg` and weight `0.135658 N`;
- neutral mean upward force `0.262064 N`;
- best sampled upward force `1.12734 N`, or `8.31017x` robot weight;
- near-weight trim candidate upward force `0.134191 N`;
- bit-identical signed-load replay;
- zero failed environment steps.

This is simulator force-authority evidence. It is not a trained stable-hover
policy and is not hardware flight evidence.

A deliberately tiny executable-learning smoke used 8 environments, 4 steps,
and one PPO update (32 samples). It advanced policy revision 1 to 2 at `83.1`
environment steps per second with zero failed environment steps, `15,925,952`
retained native bytes, and `27,808,452` peak MLX bytes on Apple M4. Its KL was
`1347.29`, so this only proves that the separate robot can complete the native
rollout-to-MLX update path; it is not a viable promoted policy or flight-quality
evidence.

## Run and train

```sh
./build/bin/metalrobo_task_rollout \
  --robot-source numifly-no-legs \
  --task numifly-no-legs-flight --scene ground \
  --numifly-wing-manifest \
    assets/numifly/maeda-wing-pack-v1/manifest.json \
  --envs 1 --steps 64 --chunk 1 --zero-actions \
  --metallib build/shaders/MetalRobo.metallib
```

```sh
./build/bin/metalrobo_task_train \
  --task numifly-no-legs-flight --scene ground \
  --numifly-wing-manifest \
    assets/numifly/maeda-wing-pack-v1/manifest.json \
  --initialize-policy numifly_no_legs_flight_v1 \
  --policy-pack /tmp/numifly-no-legs-initial.policypack \
  --updated-policy-pack /tmp/numifly-no-legs-updated.policypack \
  --rollout-pack /tmp/numifly-no-legs.rolloutpack \
  --mlx-python python/.venv/bin/python --python-root python \
  --native-library build/lib/libmetalrobo.dylib \
  --metallib build/shaders/MetalRobo.metallib
```

The Numi Window scene is `numifly-no-legs-ground-flight`. Its generated visual
observation contains only the upper-body packs and one explicitly owned
`measured_surface` pack for both live wings.
