# Respiratory drive pilot 003

This retained paired pilot tested whether halving the respiratory controller's
drive causes downstream gas changes in the existing GPU-resident reduced
cardiopulmonary model. It is a single deterministic pair, not a population
estimate, and it does not qualify the whole-body anatomy or clinical validity.

## Frozen run

- Device: Apple M4 Pro; native Metal runner, Dense45 enabled, Brain respiratory
  control enabled.
- Each arm accepted 30,000 steps at `dt=0.002 s` (60.00000285 s); intervention
  window was 15–35 s at drive scale 0.5. The control used the same schedule with
  scale 1.0. Both used world fingerprint `6020581806073115747` and experimental
  unit `resting-reference-initialization-v1-71709890b4404ad1a2b1`.
- Runner SHA-256:
  `462a089285cf2b05b022650937375ef91151a2ecba1a984220b71e34f36c814f`.
- Matter metallib SHA-256:
  `8c5f0c6078f7f385b3cce3b08b816df95721528cc797da27e134c14717e03a93`.
- Respiration metallib SHA-256:
  `f1fe31119fb00f310d3f64d42b6a6afa6c8e7183971e88e64b21c1b9a2eb72d3`.
- Network SHA-256:
  `eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818`.
- Parameter file SHA-256:
  `27100bf8941fcd623bba1ea87474f6fe6a4b1ab4f89b3e6ec0df5e4f8d9551f3`.
- Build source identity: Git revision
  `106243df31c3ce8fa8d0860fbcf7baec4e28e73a`, dirty snapshot hash
  `5148ffc20290ab9ecae50f96c03c6629115777d09405d71c571c1b0ef0e8dfea`.
- Exact commands, loaded/frozen input paths, process/timing receipts and file
  hashes are retained in each trial's `process.json`, `started.json`, and
  `receipt.json`; the preregistered plan and frozen build identity are in
  `registration/`.

## Paired response

The registered endpoint was the control-corrected change in mean PaCO2 during
the dose window relative to the pre-window. The matched pair produced
`+1.97891 mmHg` treatment-minus-control (analysis verdict `supported`; one pair,
no estimated sampling uncertainty). Under half drive, ventilation declined,
PaCO2 rose, and oxygen measures declined. In the treatment arm ventilation
returned near its pre-window value after normal drive resumed. The control's
ventilation also drifted during recovery, so recovery contrast is descriptive
and not a clean causal endpoint.

Both arms completed 70 filling/ejection cycles; LV stroke volume was about
70.03 ml. Maximum blood-volume error was 0.0168 ml. Maximum oxygen and CO2
ledger errors were 0.000116 and 0.000466 ml STPD. Real-time factors were 0.2743
and 0.2751. These are numerical and execution checks for this reduced model;
they do not establish measured-person fidelity or clinical validity.

## Retained raw evidence

`study/trials/` contains both native CSV traces, stdout/stderr, process records,
start records, and receipts. `study/analysis.json`,
`study/registration.json`, and the registration files preserve the parsed
paired result and registration identity. No trace from the earlier mismatched
world study 002 was used in this pair.
