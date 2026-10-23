# Preserve physical boundary orientation independently of provenance winding

The explicit refined-layer validator assumed outer registered triangle normals agreed with positive tetrahedral boundary orientation. The actual canonical Human parent mesh uses the opposite source-face convention. This rejected the unchanged positive-volume owner and its valid elbow children.

The validator now derives the expected exposed facet from the same sorted-corner prism and positive-volume swap as the existing default owner. It compares the child facet against that physical orientation. Plane containment, area closure, shared-face winding, parent containment, volume, material, source identity, Float32 conditioning, density and mass gates remain unchanged. No coordinates, forces, material equations or solver settings change.

Strict C++ regression: new test fails against core004 and passes against core005. Reversed provenance winding is tested at the boundary helper, independently of the still-required whole-surface signed-volume audit; reversing an actual boundary facet remains rejected. The initially attempted globally reversed closed-source end-to-end test correctly failed the separate inward-source audit and is retained as a test-construction failure.

Actual 2332-row default versus explicit identity topology is exact for 6996 nodes / 27714 tetrahedra, connectivity, material, contact, donor mass and stable face ancestry. The frozen D15 elbow candidate also passes actual FromLayers under this owner, with 2411 rows / 4777 faces / 7233 nodes / 28632 children. D15 is still unadmitted; these are CPU preparation checks with no physical time advance, natural-motion or anatomy qualification.

All builds and tests ran on SSH Numi. Exact commands, source/binary hashes and failed checks are retained in the adjacent receipts. Candidate source packages remain separate and are not promoted by this owner correction.
