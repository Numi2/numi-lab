# Numi Human bilateral Achilles force transfer

The live Human now has a nonvisual, fail-closed certificate for the reduced
Achilles mechanics already present in the source-route transaction. The check
selects bilateral gastrocnemius lateralis, gastrocnemius medialis, and soleus,
then follows each accepted insertion through the nonlinear NHMYO2
fibre-tendon equilibrium, the NHTENDON3 four-node attachment map, and the
source ankle generalized-force row.

On Apple M4 Pro, a two-step `0.1 ms` transaction transferred about `2.424 kN`
per side into named BodyParts3D calcaneal envelopes. The `0.2` selected
activation increment changed ankle torque by `-63.90 N m` right and
`-61.67 N m` left. Aggregate enthesis force residuals were `0.000436 N` and
`0.000634 N`; maximum per-endpoint moment residuals were `9.35e-7 N m` and
`3.58e-6 N m`. The borrowed consumer matched the producer in the same command
buffer, injected rejection rolled back, and replay was bitwise.

The accepted machine-readable receipt is
[`bilateral-m4-pro.json`](media/numi-human-achilles-force-transfer-v1/bilateral-m4-pro.json).

## Reproduction

```sh
metalrobo_numilab_human_myosim_visual_probe \
  myosim-fullbody-core-reference.nhrigid \
  myosim-fullbody-muscle-reference.nhmyo \
  /tmp/unused-visual-output \
  --muscle-step-seconds 0.0001 \
  --muscle-step-count 2 \
  --muscle-activation 0.2 \
  --selected-tendon-control \
  --activated-source-muscle-index 348 \
  --activated-source-muscle-index 349 \
  --activated-source-muscle-index 369 \
  --activated-source-muscle-index 388 \
  --activated-source-muscle-index 389 \
  --activated-source-muscle-index 409 \
  --support-contact-payload myosim-fullbody-support-contact.nhcnt \
  --tendon-payload numi-human-tendon-attachments.nhtendon \
  --joint-equality-payload myosim-fullbody-joint-equalities.nheq \
  --bilateral-achilles-certificate
```

The certificate exits before camera setup or rendering. It uses the source
route `J^T` as the sole rigid-force authority and audits only the distributed
calcaneal representation, so it cannot double-apply plantar-flexion torque.

## Evidence boundary

This is the intended reduced tendon force-transfer law, not a claim that the
Achilles is already a volumetric deformable body. The current mechanics source
is the pinned MyoSim full-body payload. OpenSim Rajagopal/Lai/Uhlrich parameter
equivalence is not established by this receipt and remains a separate model
integration gate. Ankle cartilage/contact, passive ligament/capsule mechanics,
sustained gait, material calibration, and clinical validation also remain.
