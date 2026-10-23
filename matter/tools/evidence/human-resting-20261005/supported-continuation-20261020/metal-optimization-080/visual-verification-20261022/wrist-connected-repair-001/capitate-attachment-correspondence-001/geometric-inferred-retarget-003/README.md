# Geometric retarget proposals for six frozen deep-fat attachment sites

This folder records a separate, non-admitted construction proposal for the six historical sites examined in `source-slot-provenance-002`. It does not alter those sites, recover their source identities, or establish that a new capitate attachment is anatomically correct. The source-identity audit recovered 0/6 identities. A separate earlier strict inferred-attachment screen produced 1/6 new proposal. Both prior results and their files remain unchanged; neither count is the current geometric screen’s 5/6 candidates.

The detailed hand replacement reuses vertex IDs, so a historical raw `sourceVertex` integer cannot be treated as a persistent anatomical identity. The candidate generator instead binds a proposed material point to an exact single-parent face on the C014 deep boundary, records face kind/parent/ancestry and barycentrics, and gives each proposal a new stable identity linked by `migrates_from_stable_id` to the old record. The original donor body (49 or 99), donor name, spring `kA`, and tributary area are copied into the proposal without alteration. The new zero-strain target is the current barycentric deep-boundary point transformed into the original donor frame; it is explicitly a new inferred reference.

## Result

| Old site | Preserved donor | New D-face / parent | Same-side region witness | D-to-owner-bone distance | Capitate-bone distance | Proposal |
|---:|---|---:|---|---:|---:|---|
| 2 | capitate_r (49) | 9307 / 3188 | fifthmc_r; outward projection +0.130 mm | 0.143 mm | 50.733 mm | 0.020 mm target shift; appears at 2/5/10/20 mm rungs |
| 3 | capitate_l (99) | 6152 / 9871 | firstmc_l; outward projection +1.605 mm | 1.675 mm | 35.724 mm | 13.778 mm target shift; appears only at 20 mm |
| 9 | capitate_r (49) | 14021 / 105773 | fourth/third MC right; outward projection +5.679 mm | 17.744 mm to nearest owner MC | 14.747 mm | 0.258 mm target shift; appears at 2/5/10/20 mm rungs |
| 10 | capitate_l (99) | 5828 / 10333 | fifthmc_l; outward projection +6.827 mm | 8.744 mm | 23.198 mm | 5.993 mm target shift; appears at 10/20 mm |
| 16 | capitate_r (49) | — | nearest exact same-side face: 8844 / 2064, secondmc_r | — | — | no proposal within 20 mm; nearest exact face is 23.705 mm away |
| 17 | capitate_l (99) | 6459 / 9872 | second/first MC left; outward projection +3.148 mm | 6.321 mm to nearest owner MC | 47.776 mm | 6.173 mm target shift; appears at 10/20 mm |

These are geometric screening results, not attachment validation. Site 2 is especially cautionary: its proposed deep point is close to the fifth metacarpal surface and about 50.7 mm from the preserved capitate donor surface. Sites 3, 10, and 17 also lie closer to adjacent metacarpal geometry than to their preserved capitate donor. This does not prove that the historical spring should attach to those bones; it shows that preserving the old donor label does not itself establish the physical pathway. Site 9 is also ambiguous because the preserved capitate surface is closer than its nearest owner metacarpal surface.

## Exact-weld bone-solid follow-up

The original report’s bone containment labels were provisional because it counted duplicate seam coordinates as open boundaries. The follow-up at exact-weld-bone-solid-004/exact-weld-bone-witnesses.json welds only exactly equal registered Float32 coordinates (canonicalizing signed zero; no tolerance, averaging, motion, or caps), preserves raw edge counts, and stores the raw-to-weld maps in exact-weld-bone-solid-004/exact-weld-maps.npz. All 13 queried same-side capitate/metacarpal meshes are closed, consistently oriented manifolds after this exact weld, with zero boundary, nonmanifold, or inconsistent-orientation edges. Their pre-weld raw boundary counts remain in the report. All five candidate points and site 16’s recorded nearest-face diagnostic classify outside every queried same-side bone component by signed solid-angle winding on the closed components.

Use geometric-retarget-ladder-exact-weld-amended.json for the corrected solid classifications. It links to and hashes both source reports, preserves each former provisional label, and states the identity counts separately: 0/6 source identities recovered, 1/6 earlier strict inferred proposal, and 5/6 current geometric candidates. The original geometric-retarget-ladder.json remains unchanged.

## Method and limits

The generator reads the exact C014 deep-boundary surface and source ancestry, the 069i registered donor-frame CSV, the independent capitate-frame review, and the retained registered major-bone triangle payload. It screens exact, non-inferred single-parent faces whose three corner owners are in the same-side carpal/metacarpal family, and requires the face normal to agree with the authored outward normal by at least 0.90. It uses closest point-to-triangle witnesses, records adjacent same-side bone distances and outward projections. Its original open-mesh AABB/ray/winding labels are provisional and are superseded in the exact-weld amended report. The 2/5/10/20 mm values are search radii only; a candidate found at a radius is not accepted because it is within that radius.

Exact-weld containment is valid for the listed closed bone meshes and recorded points; it only establishes that those points are outside those bone components in this registered pose. It does not establish a ligament/fascia path, material-point identity, force transfer, mass/strain budget, global layer intersection, dynamic clearance, solver behavior, compiler record, or native admission. The candidate set must not be promoted or used as six usable attachment records without a separately justified anatomical migration and the relevant physical and compiler checks. No header or runtime file was changed.

## Reproduction

Run on `numi` with the retained environment:

```sh
/Users/n/numi-human-prep-venv-20261005/bin/python \
  /Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/capitate-attachment-correspondence-001/geometric-inferred-retarget-003/generate_geometric_retargets_003.py
```

The script uses fixed evidence paths embedded at its top and writes the proposal report in this folder. Verify the input and output SHA-256 values against `evidence-manifest.json` before comparing a rerun. The existing geometry search was not rerun for the exact-weld correction. Reproduce the correction without searching for new candidate points with the two scripts classify_existing_witnesses_exact_weld_004.py and amend_proposals_with_exact_weld_004.py in exact-weld-bone-solid-004. The base report, amended report, weld report, and weld-map hashes are in evidence-manifest.json.
