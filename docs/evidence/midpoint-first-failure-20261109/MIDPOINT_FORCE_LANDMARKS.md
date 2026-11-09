# Retained midpoint force and impulse landmarks

These are complete native fixed-plane rollouts compared against the independent, temporally checked reference for the **same mesh**. The accompanying machine-readable receipt binds each input and its separate integrity audit. Reproduce with `python3 tests/contact/audit_midpoint_force_landmarks.py`.

| Nodes | Native midpoint stage-sampled peak | Reference integration-step-sampled peak | Peak error | Maximum 1 ms-bin error | Maximum cumulative-impulse error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 27 | 0.7228014 N | 0.7246741 N | 0.2584% | 0.08504% | 0.05104% |
| 729 | 0.3611336 N | 0.3616562 N | 0.1445% | 0.07169% | 0.04027% |

Bin error is normalized by the largest reference 1 ms-bin force; cumulative error is normalized by the final reference impulse. For 27 nodes, the largest cumulative difference is 2.5853e-7 N s at 4.882 ms on the reference sample clock. For 729 nodes, it is 1.7592e-7 N s at 9.742 ms. The worst force bin is 9–10 ms in both cases.

The native peak is a midpoint-stage sample, at 4.80469 ms and 5.87109 ms respectively. The reference peak value is sampled each 0.25 µs integration step, but its time was not retained at that rate. Its separate 2 µs trace has maxima at 4.798 ms and 5.872 ms; the latter has a slightly smaller value than the integration-step maximum. Neither the sampled maximum nor the coarser trace proves a continuous-time peak.

These results do not establish spatial convergence, a runtime benefit, or material calibration. The discrepancy between the two mesh-specific peak forces also must not be interpreted as an integrator error: the spatial study evaluates that question separately.
