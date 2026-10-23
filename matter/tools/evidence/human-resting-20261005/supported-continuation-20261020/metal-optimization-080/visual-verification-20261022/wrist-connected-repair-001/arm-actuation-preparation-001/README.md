# Bilateral elbow-flexor core actuation check

This evidence qualifies a narrow software behavior: the selected-tendon core applies the authored bilateral elbow-flexor cohort in the expected direction over a matched 10 ms run. It does not qualify whole-body FEM/contact, grasping, full-range motion, Brain control, biological behavior, or production use. The rest of the whole-body hand/arm-contact path remains separately unqualified.

## Trial and result

The paired full-flexor run used 80 accepted steps at 125 microseconds (10 ms simulated), 0.5 activation versus the same-input zero-activation baseline, and no root assistance or elbow pose locks. It used the existing core selected-tendon path and the hash-pinned 060 skin payload. The eight source muscle actuator IDs were 230-233 and 293-296: bilateral biceps long/short heads, brachialis, and brachioradialis. The authored elbow coordinates start at their 0 rad lower limit and were left free.

`arm-flexor-assessment-006-final.json` reports `PASS_CORE_DIRECTIONAL_10MS_OWNER_LAW`, with no errors. Right elbow q increased by 4.66697e-5 rad versus baseline and v by 0.0286264 rad/s; left q increased by 3.71410e-5 rad and v by 0.0279756 rad/s. The matched selected-muscle generalized torque increments were +0.261164 Nm (right) and +0.260946 Nm (left). These are positive directional responses over a very short horizon, not evidence of useful-range flexion.

The accepted source-force and tendon-force products match the existing native telemetry exactly in Float32; normalized owner tension is within one Float32 ULP, and the largest accepted normalized equilibrium residual is 2.99e-8 (baseline 6.37e-12; the existing owner tolerance is 1e-4). The endpoint MyoSim source comparison uses each endpoint's own q and computes the torque difference as `J(q_active)F_active - J(q_baseline)F_baseline`. Maximum absolute torque difference is 0.000641 Nm, about 0.245% of the corresponding net native torque increment (about 0.0752% of native torque L1). This is a reported comparison, not a newly invented acceptance threshold. The source baseline state is reconstructed from the exact paired 006 q/v values for the emitted arm/hand coordinates; the standalone 004 baseline differed by at most 1.50e-7 rad q and 4.83e-5 rad/s v, so it is not represented as bitwise identical.

The 006 telemetry-only binary also reproduces the 004 BIC-only run bitwise for all 129 accepted q values, all 128 v values, all 44 hand/wrist telemetry rows, and the shared BIC muscle rows. The 006 full-flexor and BIC-only matched baselines agree bitwise for the 54 emitted arm/hand coordinates.

## Reproduction and pinned inputs

Run the CPU assessor against the retained native receipts (it does not launch native/GPU work):

```sh
E=/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-actuation-preparation-001
/Users/n/numi-human-prep-venv-20261005/bin/python "$E/assess_elbow_flexor_006.py" --evidence "$E" --output "$E/arm-flexor-assessment-new.json"
```

The assessed binary is `/tmp/numi-human-native-arm-telemetry-006`, SHA-256 `edf5b50aa4df13f3e3ca3e404107e4d35198dc9b6b615c9727bac3f244e091d2`; its scoped build manifest is `scoped-native-build-manifest-006.json`. The muscle payload SHA-256 is `e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b`, with embedded MyoSim source archive SHA-256 `280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975`. The assessor receipt pins the remaining scene, tendon, skin, source-manifest, binary, and build-manifest inputs, as well as the exact commands and native logs.

Raw native results remain under `results/`; the original 004/006 parity, baseline and full-flexor results are retained. No physics equations, solver gates, muscle forces, or model inputs were changed for this assessment.

## 100 ms follow-up and new-skin thumb response

The 006 full-flexor follow-up completed 800 accepted steps (100 ms) with bitwise replay in 74.3445 wall seconds. The accepted right/left elbows reached 0.0110459/0.0108233 rad; selected-muscle torque increments were +0.872537/+0.868660 Nm. All eight accepted muscle/tendon owner-law checks pass. The literal source-limit comparison fails for tiny baseline/other endpoint excursions: the elbow baselines are below their zero lower limits by 2.39e-13/1.62e-12 rad, and the maximum over emitted arm/hand targets is 3.67e-10 rad. These deviations remain reported; no clamping or tolerance was added. See results/bilateral-elbow-flexors006-100ms/owner-assessment-final-002.json. This is an actuation response, not full-body contact or useful-range qualification.

The separate bilateral opponens test used the exact 069h detailed skin, 80 steps and the same 006 core physics. It passed the existing matched-baseline torque/motion and replay checks; its result is in ../hand-actuation-qualification-001/results/scoped006-opponens069h-80-step10ms. Rendering the new skin in this core does not admit the unfinished skin/fat FEM map. A subsequent actual-pose surface audit found six left ring-proximal skin/bone crossings during wrist flexion, first introduced in 068; that failed evidence is retained separately.

The maintained 100 ms owner assessor accepts --output FRESH_PATH and refuses to overwrite retained results before loading inputs. Assessment-output-preservation-test.json verifies that rejection preserves bytes. The original executed assessor is retained separately.
