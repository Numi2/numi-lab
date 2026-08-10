# Physically Qualified Imagination II

## Stable-base composition of generated upper-body intent on Apple Silicon

Numan Thabit - Numi Lab, Tromso, Norway - 10 August 2026

### Abstract

Generated humanoid motion can express a command while failing as an executable physical trajectory. We study three deliberately simple commands - raise the right hand, raise the left hand, and raise both hands - because they isolate embodiment from locomotion discovery. ARDY supplies the motion geometry. Numi Lab either executes the raw whole-body retarget or composes only the requested arm trajectory onto a fixed standing reference, uniformly scaling each arm only when required by compiled G1 position or velocity limits. NumiSolver remains authoritative for gravity, contact, actuation, termination, and accepted state.

The preregistered confirmatory matrix contains 30 cells: two methods, three tasks, and five untouched robustness seeds. Each seed perturbs stiffness and damping independently within 0.95 to 1.05 of nominal. Qualified success requires every requested shoulder to cross -0.15 rad and realize at least 0.25 rad of negative-pitch excursion, zero terminations, zero failed environment steps, minimum root height above 0.64 m, and maximum tilt below 0.50 rad. Stable-base composition qualified 15 of 15 runs. Raw generated motion qualified 0 of 15. All 30 runs completed with zero failed solver steps. This is Apple Silicon simulator evidence, not hardware evidence, and the shoulder endpoint is a mechanism-level proxy rather than a measured hand-height outcome.

### 1. Question

Paper I proposed physically qualified imagination as a generator-realizer-solver-qualifier-student architecture. Its broad learning hypothesis was intentionally ambitious. The present study asks a narrower question first: can generated upper-body intent become executable without asking the generator to rediscover a solved stability substrate?

The distinction matters. "Raise your right hand" is not a standing task. Stability is a prerequisite. Treating the generated root, legs, waist, and arms as one indivisible teacher makes a motion generator responsible for dynamics it did not measure. Stable-base composition instead preserves only the semantic degrees of freedom needed by the command.

### 2. Methods

#### 2.1 Motion source and embodiment

ARDY is an autoregressive diffusion model for interactive motion generation with text and kinematic conditioning. We use its G1-coordinate proposals as kinematic intent, not physical truth. Each source clip is converted into the exact 29-DoF G1 joint order. Predicted bilateral foot contact remains a contact-mode hypothesis; no force, pressure, or center-of-pressure field is synthesized.

For unilateral tasks, the requested seven-joint arm trajectory is rebased at its first frame. Root, both legs, waist, and the contralateral arm are fixed to the standing reference. For the bilateral task, the independently selected left- and right-arm proposals are composed. A single positive scale is applied within each arm, preserving its within-arm coordination, and is reduced only enough to satisfy every compiled joint position and discrete velocity bound. The final clips contain 16 frames at 50 Hz.

#### 2.2 Physical execution

The native Metal task executes position targets through the authored G1 drives and the TemporalCone contact solver. The physics-gated reference clock advances only while realized support remains admissible and freezes after a physical fall. A prior implementation also required 0.30 s of literal quietness before advancing. That rule deadlocked balancing controllers at frame zero; Paper II removes stillness as an advancement prerequisite while retaining quiet-support time as evidence.

The safety envelope is unchanged: minimum root height 0.55 m, maximum tilt 0.50 rad, and no undesired contact. The study endpoint is stricter on root height at 0.64 m. Solver errors remain transactional failures.

#### 2.3 Design and preregistration

The final protocol was frozen at revision `e510cf4e40c8f1bf71092ca14fa123170d182b05`, SHA-256 `637c122666689b797f07e3a18ec135662effeeac1a4f0cae5e53ddeacaa50f49`. Five untouched seeds (2650443586 through 2650443590) independently sample 0.95-1.05 stiffness and damping multipliers. The three task horizons are 40, 40, and 36 control steps for right, left, and bilateral raising. The exact command, state trace, pack hash, revision, and per-run evidence are retained.

An earlier endpoint audit used an absolute shoulder angle alone. It exposed a false-positive mode: a raw clip could begin with an already raised shoulder and receive credit without producing a raise. Protocol v2 therefore requires both an absolute crossing and at least 0.25 rad of realized excursion, then moves to five untouched seeds. Version 1 is not used for paper claims.

### 3. Results

| Task | Raw generated | Stable-base composition | Mean max tilt, raw | Mean max tilt, composed | Mean minimum requested-arm excursion, composed |
|---|---:|---:|---:|---:|---:|
| Raise right hand | 0 / 5 | 5 / 5 | 0.515 rad | 0.432 rad | 0.748 rad |
| Raise left hand | 0 / 5 | 5 / 5 | 0.520 rad | 0.209 rad | 0.568 rad |
| Raise both hands | 0 / 5 | 5 / 5 | 0.509 rad | 0.411 rad | 0.713 rad |
| **All** | **0 / 15** | **15 / 15** | - | - | - |

Every raw run terminated at the 0.50 rad tilt boundary. Every composed run completed its horizon without termination. All 30 runs reported zero failed environment steps. The result is therefore not explained by accepting failed solver transactions. The comparison isolates representation: raw execution asks the entire generated body to become physical, whereas composition retains generated intent only in requested arms.

The deterministic task logic reproduced across bounded controller calibration error. This does not establish broad robustness. It establishes that the result survives the preregistered 10% stiffness/damping interval and the selected horizons.

### 4. Negative learning pilot

Before freezing the compositional study, we tested the broader Paper I learning hypothesis on the right-hand task. A 64-update, 1,048,576-transition PPO pilot completed with zero failed environment steps at 4,735 transitions/s. The candidate increased held-out physical failure from 0.9453 to 0.9688 and reduced tracking from 0.7980 to 0.7820, so the deployment gate retained the incumbent. This pilot was conducted during protocol development and is not confirmatory evidence. Its role is diagnostic: a generated teacher does not rescue a mismatched or insufficient stability actor merely by increasing distillation weight.

### 5. Interpretation

The experiment supports a compositional principle: when a capability is already understood, hold its coordinates invariant and ask imagination only for the unresolved semantic subspace. This is stronger than filtering a bad motion after execution. The representation itself prevents the generator from rewriting the stability substrate.

The bilateral result is especially informative. It is not a third generated clip selected after trial. It composes the unilateral arm proposals under one joint-limit projection. Its 5/5 result shows closure for this small composition, under this solver and calibration envelope.

### 6. Limits

The study is simulator-only on Apple M4. No real G1, actuator calibration, sim-to-real, energy, impact safety, or hardware reliability claim is made. The primary motion endpoint is accepted shoulder pitch and excursion. Negative shoulder pitch raises the hand forward in this G1 mechanism, but Paper II does not yet publish a solver-resident wrist-height outcome across all environments. The horizons are short, the sample contains five controller seeds rather than five independent generated motions, and the robustness interval covers only stiffness and damping. The task uses a fixed ground scene and bilateral predicted support.

The 15/15 result must therefore be read as a controlled embodiment result, not a general humanoid manipulation result and not evidence that stable-base composition replaces closed-loop whole-body learning.

### 7. Next research

The next paper should compile task-owned Cartesian hand outcomes and solver-resident wrist trajectories, then test sequential composition: right, left, both, asymmetric reach, and reach-with-load from one stable lower-body controller. The harder learning question should return only after a currently compatible stability actor is qualified under the same action contract. The confirmatory comparison should then disconnect ARDY and evaluate whether a student preserves both hand outcome and support recovery.

### References

1. N. Thabit, "Physically Qualified Imagination: Generative Motion Models as Training-Time Teachers for Robot Learning on Apple Silicon," Numi Lab Paper I, 2026.
2. K. Zhao, M. Petrovich, H. Zhang, T. Wang, S. Tang, and D. Rempe, "ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation," arXiv:2607.08741, 2026. https://arxiv.org/abs/2607.08741
3. NVIDIA Spatial Intelligence Lab, "ARDY: Interactive Human Motion Generation," 2026. https://research.nvidia.com/labs/sil/projects/ardy/
4. Numi Lab, "Numi Lab source and executable protocol," 2026. https://github.com/Numi2/numi-lab
5. Reproducibility artifacts: `research/pqi2/upper_body_protocol.json`, `research/pqi2/results/study-summary.json`; command: `numi research upper-body --output PATH`.
