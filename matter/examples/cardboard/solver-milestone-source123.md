# Source123: two-stage local material merit

The [source125 continuation](solver-milestone-source125.md) captures the next
global failure, independently checks its material/assembled action, and retains
the unresolved native finite differences and subsequent diagnostic failures.

The accurate 15-state bilinear paper return map now uses raw-residual Armijo descent until its existing raw convergence gate is met, then the frozen row-scaled merit to resolve the correction gate. Both final stopping gates, material equations, parameters, local iteration budget and global production certificate remain unchanged. The scaled Newton solve remains active throughout. This repairs two complementary retained failures: a raw-only search stalls near the tet 674 root, while a scaled-only search takes tiny steps far from the tet 858 root.

Native 407 resolves the exact retained tet 858 candidate in 14 updates, with valid direct stress and tangent. Against an independently projected FP64 root, state, stress and the declared local Fxx tangent action differ by 2.47e-13, 5.20e-8 and 3.44e-8 relative respectively. The promoted native state also meets both unchanged gates under the independent equations. Native 408 retains the tet 674 repair in 5 updates; all five trace/direct/tangent output comparisons have zero bit mismatches. Native 411 traces tet 858's 14 updates with the same parity. These are material-point results. The retained snapshot still lacks the first failure's dispatch-generation identity.

Native 410 evaluates 3,072 samples across the two authored diagnostic policies, with no invalid projection, stress or tangent result. Its accurate sweep authors 16 local updates and its default sweep 32; those budgets remain separate from the exact failed-point tests, both compiled 32. All nine medium tangent columns at two retained centers meet the 1e-3 analytic comparison gate. An independent 90-digit endpoint audit retains three unique states, nine directions, three declared steps and both stress paths. At step 1e-4, all 27 nonzero-vector comparisons per path pass 1e-3, with worst error 1.03e-4. At 1e-5 and 1e-6 one direction fails per path; the smaller-step failures remain evidence. These selected realized-direction comparisons do not establish an absolute noise floor or replace the outstanding FP64 nominal-direction, element and assembled-action gates.

The full 27-control load, hold, unload, relaxation and release coupon 409 closed after 676.33 seconds with exit 2. The first paired control was accepted; the bent specimen rejected the second loading control after residual descent failed. The rejected geometry, material state, conditioning snapshot and complete attempted history are retained. This incomplete cycle does not qualify any mesh/time study.

## Exact contact-helper evidence

The opt-in private-trial observer records the actual production vertex/triangle CCD call, including all endpoint arguments and its returned certification. Native 405/406 preserve all 129 original capture arrays, 32 Armijo rows and transaction histories against 398/399; only the separately bound device-program fingerprint changes in serialized state. The new diagnostic adds one 336-byte array.

At trial 21 the selected helper exhausts its 16-step budget without certifying; at trial 22 it returns a near-threshold hit. The two separations lie only 0.0318 nm above and 0.0846 nm below the existing effective threshold. An independent binary64 trajectory calculation also finds swept contact. An uncertified query remains rejected. No thickness, feasibility or descent threshold is relaxed. The complete contact fixture still fails.

The native trace host's output contains a stale `source_snapshot` label naming 121. The source-bound launcher receipt and frozen 122/123 build identities determine the diagnostic source; the input coupon was produced by 118. A metadata-only follow-up distinguishes those identities. The rate direction already uses Hdot=H/dt.

| Verdict | Current boundary |
|---|---|
| Solver correctness | Both retained local material failures repaired and selected analytic comparisons pass. First-failure generation, complete coupon, near-zero derivative and nonlinear contact qualification remain open. |
| Numerical convergence | No complete spatial, timestep or slower-loading coupon series. Combined refinement still exceeds selector 5 admission. |
| Component calibration | No new corrugated damage, starch or hot-melt fracture calibration. Source assumptions and identifiability limits remain. |
| Analogue validation | No new complete-board analogue validation. |
| Physical assembly in simulation | No fully scored, folded, bonded, released and settled 300 × 200 × 200 mm shipping-box checkpoint. |
| Crush and impact prediction | No qualified capacity curve, frozenFpeak, capacity-derived mass series, gravity crush or drop impact. |

The [evidence index](evidence/native-implementation-20261014/summary.json) retains commands, exact source restoration packages, rejected runs and numerical comparisons. The kraft/NSSC/starch recipe and target dimensions remain unchanged.
