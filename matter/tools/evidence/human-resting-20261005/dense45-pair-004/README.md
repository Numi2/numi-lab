# Dense45 paired runtime check

This is a 6.0000003 s numerical A/B run of the existing CVSim21 vascular owner using its general FGMRES solve and the opt-in fused Dense45 solve. Both runs used the same coupled respiration/Brain runner, source network, parameters, initial state, and Apple M4 Pro. It checks solve-path agreement and throughput; it is not whole-body, physiological, or clinical qualification.

Source checkout was based on `42fb4997a10661633db1ba8c2283bc4a3cc0450f`, with uncommitted integrated-scene changes present. The executable and exact relevant source/input hashes are recorded below so the run can be distinguished from later builds.

```text
device: Apple M4 Pro
timestep: 0.00200000009499 s
command: numi-human-resting matter/tools/fixtures/cvsim21.native.v3.json matter/examples/resting-reference-respiration.json OUTPUT.csv --steps 3000 --dt 0.002 [--vascular-dense45]
FGMRES: accepted 3000/3000; simulated 6.00000028498 s; wall 37.483681292 s; GPU 33.5687850853 s; RTF 0.160069664403
Dense45: accepted 3000/3000; simulated 6.00000028498 s; wall 21.261836 s; GPU 20.8238866692 s; RTF 0.282195774861
both final: 7 complete fill/ejection cycles, last LV stroke about 70.501 ml, breaths counter 1
max |blood volume ledger error|: 0.00325963 ml (FGMRES), 0.00279397 ml (Dense45)
max |O2 ledger error|: 0.000116416 ml STPD in each run
max |CO2 ledger error|: 0.000465662 ml STPD in each run
max Dense45 vs FGMRES trace difference: airflow 0.035856 ml/s; pleural pressure 0.004822 Pa; alveolar pressure 0.004776 Pa; PaO2 0.001908 mmHg; total blood 0.001397 ml
transaction check: PASS; forced rejected candidate left circulation, circulation clock, respiration, and Brain controller history bitwise unchanged; restored accepted replay was bitwise identical
```

The same binary ran both paired paths and the transaction check. Remote retained originals: `/Users/n/numi-human-resting-evidence-20261005/dense45-pair-004/`. The CSV traces and logs are copied next to this receipt.

```text
binary f1121f70f0f3bb1aa4ca93559922e42488e8bb6df49d1c987a70d0e9d620ee81  matter/numi-human-resting
NumiMatter.metallib 8c5f0c6078f7f385b3cce3b08b816df95721528cc797da27e134c14717e03a93
HumanRespiration.metallib 079714a5efe672ab5e5040814e434e996188c2eaeed0bffee8392cb558dc312b
CVSim21 JSON eeb6ebc5dad5cb413587038532ac5badc3d3e8aa419cca111604239e7f692818
respiration parameters 27100bf8941fcd623bba1ea87474f6fe6a4b1ab4f89b3e6ec0df5e4f8d9551f3
matter.hpp 36af4eacdd3ffc732aac3a8beffbad55584c77fa97cc5f74593eb0dcb3635065
runtime.mm 3c4e548e1d98d9918d9b0526fe74e2450fbce2fcd977df22779b9ea51bfe2f11
vascular.metalinc 3e125ad35485f11cd6c47bba1dfa7fa72b5f51cc80d089e617e1621608e6012b
human_resting.mm 4d179431c3ddeb889d1d6a071dc8169eb001697759037d9f7c3aee553f45717a
human_resting_runtime.hpp 27a549d77e1b1669b046fcc7bca3bf176811c45d2ac028e1b6c70b43a8090ca7
human_respiration.metal fbb84f29902cacc2ae6dd9373c09126cad66f58f7926071d975b5e6dacb30ae3
human_respiration.h 1e3708a0fa80f1315d092c75f73196af73f9c4088c9fa84e7300d992217c2fa6
human_respiration_parameters.hpp 25ed0482231702401bae49b6989efc5c0afdbd9fe2bc1beb98fbaae9266478a7
```
