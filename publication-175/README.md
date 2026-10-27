# P175: refined connected-surface construction evidence

P175 extends P174 (commit ed5b9a9adc7f5fe516db184ad7390b0482744aef; manifest SHA-256 9b4bfdec4e37902db1ae6969fdc7950c7c78214d1cec651a38576b316fd8723e). The strict 764-pair P174 candidate is the inherited baseline and is referenced by immutable path and hash.

## Candidate progression

| Branch | Surface pairs | Contained pairs | Self pairs | Banks | Status |
|---|---:|---:|---:|---:|---|
| P174 baseline | 764 | 0 | 0 | 185/185 | inherited |
| 638 refinement intermediate | 770 | 2 | 0 | 185/185 | six midpoint nodes; not strict |
| 715 | 715 | 0 | 0 | 185/185 | strict 130/159 and arithmetic checks pass |
| 703 | 703 | 0 | 0 | 185/185 | strict checks; 30 identities removed and 18 introduced vs 715 |
| 701 | 701 | 0 | 0 | 185/185 | exact checks pass, but two bank counts regress |
| 697 | 697 | 0 | 0 | 185/185 | filtered strict result; no bank increases against the 764 baseline |

Mapping child faces through the pinned refinement parent map, 697 leaves 691 distinct parent-pair identities versus 764 at baseline: 202 are removed and 129 introduced. 703 to 697 removes 13 same-topology pair identities and introduces 7. The 701 intermediate is retained because its bank deltas include +2 at 43/16 and +2 at 65/99; the later 697 selector filters those increases.

The 703 and 697 candidates share the same 20,895-point, 41,745-face topology; only points_world_m changes. Their 237 typed anchor rows and source identifiers are bitwise unchanged. The 697 area gate passes the 0.1 floor, but the fixed 037 reference minimum is 0.1000096383. For 703 it is 0.1000096383 and the current oriented minimum is 0.1017480666. These are narrow construction-floor passes, not volume-element quality results.

## Fit evidence

The 638-to-715 fit wrote an unadmitted construction proposal; its 49 MB JSON report is losslessly archived with a fresh decompression, SHA-256 and JSON parse receipt. The later 703 and 697 fits both report LP_NO_PROPOSAL: component 9 timed out in the 703-from-715 run and component 7 timed out in the 697-from-703 run. Their sidecars preserve the failure status, the opt-in zero fallback and raw direction capture as unadmitted. Neither partial solve is global LP convergence or a global optimum. The exact geometry candidates were checked separately after component retention/filtering.

## Limits

All included states remain candidate_selected=false, native_admission=false, and accepted_physical_steps=0. The checks cover the emitted rest pose and 185 typed bank surfaces. They do not establish closed-solid or swept-motion clearance, tissue volume construction, cell positivity/conditioning, conservative mass or thickness transfer, contact dynamics, eight-pose behavior, or native 16/60-second performance. Forty-five known head boundary edges remain open.

The existing skin/fat owner is NumiHumanCoarseTissuePreparation.hpp: buildRegisteredSkinFatCandidateLayers and appendRegisteredSkinFatFEMFromLayers. Existing passing fixtures are synthetic. The prior 541 transfer experiment used a different topology and recorded prism orientation and conditioning failures; it is not reused to qualify these candidates. No physical volume mesh was built for P175.

P175 is an unpublished local staging bundle. Original run directories remain intact. The independent checker and exact audit/source pins are under evidence/independent-review and source-pin-index.json.
