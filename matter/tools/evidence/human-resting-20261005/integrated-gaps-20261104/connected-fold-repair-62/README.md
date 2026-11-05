# Connected source-fold repair candidate (62)

This is an offline proposal, not promoted native geometry or a repaired simulation.
The accepted production scene remains trial 307.

The repeated late VT witness uses global FEM nodes 24343/24409/24345/24341,
corresponding to current skin/fat rows 22677/22743/22679/22675. The reference
point projects inside the nonincident triangle only 2.164 nm from its plane;
it is a genuine source fold, not an exact duplicate or a removable contact row.

Two directed contractions of the existing 7.513 micrometre edge were tested
on the current conditioned volume. Both pass surface and tetrahedral link
conditions and retain positive cells. The preferred 22677 -> 22679 contraction:
- removes three cells from the 32-cell union star;
- modifies three retained cells and two retained outer faces;
- improves the modified-cell minimum q from 1.1791e-5 to 7.3977e-5;
- passes the unchanged 1e-18 m3 and q>=Float32(1e-5) gates;
- retains minimum modified-cell volume 1.77997e-17 m3;
- has minimum face-normal cosine 0.9994528;
- changes cavity volume by 7.22691e-18 m3, not zero.

The existing exact-sign registered-surface predicate checks every AABB-overlap
pair involving a changed face against the full outer surface, including adjacent
features. All 60 candidate pairs pass in the registered and unposed coordinate
views; those views are identical here, so these are not 120 distinct cases.
No graph-distance or source-face collision exclusion is added.

The reverse contraction creates eight genuine self-crossing witnesses and is
rejected despite positive cells. Its witnesses remain in report.json.
This is why positive determinants alone are insufficient for repair admission.

The retained NPZ is a small local patch using existing geometry/provenance fields.
It is deliberately not presented as a completely rebound source volume.
Before promotion, the owning pipeline must regenerate typed face-parent unions,
the active O row map, corresponding skin-volume/P1 ancestry and conservative
source mass routing; then check I/D, bones and every nearby structure.
Current-node 22677 becomes inactive, so stale source-row and support/receptor
mappings cannot be reused. Native acceptance, rollback, strain/force comparisons,
supported rest and motion qualification are still required.

Back412 is outside this patch; all input point coordinates are unchanged.
Source assets and their exact hashes are in report.json. No runtime force,
clearance, mass, physical state or source asset was modified by this experiment.

Reproduction on SSH Numi:
compile exact-pair.cpp with clang++ -std=c++20 -O2 -fno-fast-math -dynamiclib,
then run the two retained Python scripts with the qualified preparation Python.
The wrapper calls the existing NumiHumanRegisteredSurfaceIntersection.hpp
predicate; it is offline initialization validation, never per-step CPU physics.
