# Row-scaled rank-two predictor experiment (58)

This is a default-off solver prototype, not an accepted Human performance change.
The unchanged production control 304b accepted 32 roots / 0.5 ms and exactly matched
reference 290: 1,798,453 node/tetrahedron records plus numeric traces. Its body took
231.374 s versus 230.856 s (ordinary single-pair variation); 144 active solves,
2,304 columns and 62 exhausted budgets were unchanged.

The prototype fixes stale-linearization operator ordering, completed-Newton-tail
state mutation, physical versus scaled norm ownership, and cancellation in the
two-direction projection. It uses current-operator admission in both raw and
row-scaled norms and preserves the final physical certificate. It recycles two
Newton corrections within a root; it is not accepted-step Krylov recycling.
The second orthogonal direction is measured directly with a distributed QR pass,
rather than subtracting near-equal Gram diagonal terms. No tolerances are relaxed.

Native synthetic checks pass 2,743,464 assertions, including 71,767 predictor checks
and 14 admission cases. The retained earlier 58d cancellation test failed; 58f fixes
it at the original tolerance.

## Native outcomes

- 299: existing zero-action shortcut admission guard stopped execution; no accepted steps.
- 300: 11 accepted, then a contact rejection.
- 301: deliberately stopped before physics after discovering stale cache ordering.
- 302: corrected current-operator ordering; 27 accepted, then a fresh physical
  certificate rejected residual 1.19238e-5 against 1e-5.
- 303: scaled QR and two-norm admission; 30 accepted, then contact status 6,
  object 15, index 765002. Inclusive elapsed 446.180 s.
- 304: control preparation captured a mutable pipeline archive while 303 was active;
  the pin check correctly refused the launch. Pins were regenerated after completion.
- 304b: production defaults, 32 accepted and exact reference state equality.

The failure origin for contact status 6 is not localized by the current typed
certificate witness. No claim is made that it is a clearance or normal floor.
Partial accepted prefixes do not establish full-horizon fidelity or a speedup.
The predictor remains disabled. The previous failed trials and original source
patches are retained alongside execution pins.

All builds and trials ran on SSH Numi's physical Apple M6 GPU. The benchmark is a
release transient with zero muscle activation, not supported rest or physiological
qualification. No sustained-throughput, native presentation or 0.1x claim follows.
