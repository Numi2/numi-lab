# Lumbar interspinous increment (34-row catalog)

This package records the four added single, midline L1-L2 through L4-L5 interspinous effective bands. The frozen catalog header is bound by source-identity.json; its SHA-256 is a9d4aa9687e18c2c3598fabe181710bf9eb26557d7783dc08ca4754e03b33590. The exact native-support-multipoint-048 invocation was used for source pose and surface geometry. Its NHEQ projection included pose overrides 36=0.10 and 74=0.10; a common root transform is omitted from the frame-local geometry checks.

The actual-pose catalog receipt admits 34/34 rows. Every row was projected onto its pinned source surface, converted to body-local float32 anchors and reconstructed in the exact pose. Each segment was checked against all 185 registered bone meshes; the receipt reports zero segment-triangle hits for this catalog. This is a surface-triangle clearance check only: the source meshes have open boundaries, so it does not establish nonpenetration through closed solids or other tissues.

The four lumbar candidate paths are midline and are not duplicated left/right springs. At q7=0, their source-reference lengths are 14.2384, 16.1344, 11.5159, and 17.8692 mm. Under the bounded q7=-0.05 rad flexion probe, each lengthens by 0.3694, 0.0522, 0.0298, and 0.0249 mm, respectively. The corresponding source-FK relative rotations of each cranial vertebral body against its caudal neighbor, resolved in the instantaneous entry-joint frame, are -0.7305, -0.6618, -0.5844, and -0.5300 degrees about local +Z. Here +X is anterior, +Y superior, and +Z lateral, so this negative component tilts the cranial superior direction anteriorly. These are kinematic geometry results at one small master-coordinate perturbation, not a simulation of physiological loading.

The parameter prior is Ellingson et al.'s lumbar FE model input: interspinous modulus 9.8-12.0 MPa and area 30 mm2 (Table 2; modelled using 17 two-node truss elements over an L3-sacrum FE model). This catalog uses the midpoint 10.9 MPa and A=30 mm2 with k_ref=E*A/L0. Those values are FE model inputs, not subject-specific measured properties, and the present single effective line per level does not reproduce the 17-element sheet.

| Midline interval | L0 (mm) | Length change at q7=-0.05 (mm) | Relative sagittal rotation change (degrees) | Derived k_ref (N/mm) |
| --- | ---: | ---: | ---: | ---: |
| L1-L2 | 14.2384 | +0.3694 | -0.7305 | 22.9660 |
| L2-L3 | 16.1344 | +0.0522 | -0.6618 | 20.2673 |
| L3-L4 | 11.5159 | +0.0298 | -0.5844 | 28.3955 |
| L4-L5 | 17.8692 | +0.0249 | -0.5300 | 18.2996 |

The relative rotation is calculated from NHEQ-projected source FK as the cranial vertebra relative to its caudal neighbor in the moving entry-joint frame; the table gives the change about that frame's local lateral +Z axis. k_ref is derived from the listed midpoint modulus, effective area, and row length, not measured as a row-specific experimental stiffness. Iwanaga et al.'s cadaver anatomy supports the interspinous location/level variability; its failure-force results were not treated as stiffness. Adams et al. supports the qualitative role of the posterior interspinous/supraspinous complex in resisting late flexion.

The endpoints are inferred posterior midline regional proxies projected to registered vertebral source surfaces, not measured entheses. The effective rows are massless and zero-prestrain at construction. The current source supports only proximal lumbar L1-L5 interspinous coverage; cervical and thoracic ligament groups remain unsupported by the segmented rigid-body scene. The warnings-as-errors C++ catalog fixture subsequently passed in the combined052 build. A single125us integrated control with this header also passed; it is only short numerical continuation, not sustained support or physiological qualification. See numerical-validation.json.

Primary sources:

- Ellingson AM, Shaw MN, Giambini H, An KN. [Comparative role of disc degeneration and ligament failure on functional mechanics of the lumbar spine](https://pmc.ncbi.nlm.nih.gov/articles/PMC4808500/). DOI: 10.1080/10255842.2015.1088524; PMID: 26404463.
- Iwanaga J et al. [Anatomical and Biomechanical Study of the Lumbar Interspinous Ligament](https://pmc.ncbi.nlm.nih.gov/articles/PMC6896651/). DOI: 10.4103/ajns.AJNS_87_19; PMID: 31903363.
- Adams MA, Hutton WC, Stott JR. [The resistance to flexion of the lumbar intervertebral joint](https://pubmed.ncbi.nlm.nih.gov/7394664/). DOI: 10.1097/00007632-198005000-00007; PMID: 7394664.

Reproduction commands and exact hashes are in commands.txt and source-identity.json. flexion-candidates-048-before-relative-rotation.json preserves the earlier geometric receipt; the canonical enriched flexion receipt is audit_lumbar_interspinous_flexion_candidates_048.json.
