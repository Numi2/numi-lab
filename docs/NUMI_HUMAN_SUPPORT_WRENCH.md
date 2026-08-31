# Numi Human whole-body support-wrench preflight

The Human static compiler now separates external support from internal muscle
recruitment. Ten NHCNT1 foot witnesses contribute nonnegative normal reactions
that first close the six floating-base wrench equations; muscles are then
recruited against the remaining internal equations. The legacy overload, when
called without support witnesses, retains its previous internal-DoF behavior.

Run the fail-closed nonvisual certificate on Apple silicon:

```sh
metalrobo_numilab_human_myosim_visual_probe \
  myosim-fullbody-core-reference.nhrigid \
  myosim-fullbody-muscle-reference.nhmyo \
  /tmp/numi-human-support \
  --muscle-step-seconds 0.0001 \
  --support-contact-payload myosim-fullbody-support-contact.nhcnt \
  --joint-equality-payload myosim-fullbody-joint-equalities.nheq \
  --whole-body-support-certificate
```

The Apple M4 Pro receipt closes 952.864475 N of reaction against 952.864477 N
of body weight (relative error `2.30e-9`), keeps all ten reactions nonnegative,
closes floating-base force/moment residual to `2.20e-6`, and recompiles
bitwise-identically.

This is intentionally a preflight. The measured internal normalized residual
is `22.9850`, the coupled root-acceleration residual is `5.7539`, and
`internal_balanced=false`. Therefore this receipt is not dynamic contact or a
sustained-standing claim. The next blocker is the internal muscle/passive-force
equilibrium, not missing body-weight reaction.

Machine-readable evidence:
[`bilateral-m4-pro.json`](media/numi-human-support-wrench-v1/bilateral-m4-pro.json).
