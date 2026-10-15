# Multi-mortar parent-pair interim guard and CPU union prototype

This package has two separate outputs:

1. `overlay.patch` is an interim source overlay for the frozen 086 tree. It
   rejects an unsupported exact-parent-pair ambiguity during compilation only
   for same-object FEM self-contact worlds. It does not reject duplicate
   cohesive geometry in a noncontact world and does not claim that admitted
   source patches overlap.
2. `mortar_union_partition.py` is a standard-library CPU geometry prototype
   for bounded covered/uncovered partitions on a parent triangle. It is not
   integrated into Metal contact, CCD, cohesive force/history evaluation, or a
   native qualification run.

## Why the interim guard exists

In the frozen contact source, `activeMixedCohesiveFaceForPrimitivePair` takes
one output face and scans the cohesive incidence range. It returns `false`
when it encounters a second active mixed-mode face for the same ordered
primitive pair. Two production paths use it: candidate contact generation and
private Armijo trial validation. The contact partition builder then creates a
single covered patch plus its complement. A three-way tiling of one parent pair
therefore cannot be represented by selecting one face, even if the three
patches are nonoverlapping. Choosing the first face would silently omit the
rest of the covered union.

The compiler overlay builds the canonical key from exact tetrahedron-face
primitive identities (`4*tet + oppositeCorner`), not just the pair of parent
tetrahedra. It tracks both mixed-mode paired-face sources and mortar-patch
sources, so one of each on the same pair cannot escape the runtime ambiguity.
When `deformableSelfContact` is false, the guard is not applied; the cohesive
faces remain authored and compiled independently. When it is true, a second
mixed-mode patch on an already-seen exact pair is rejected with a diagnostic
that explicitly calls this a conservative contact-capability rejection, not a
geometric-overlap verdict.

The source fixture retains the old three disjoint centroid tiles as an early
rejection control. Its page-parity control instead uses three distinct
parent-face pairs and must compile. Additional CPU compiler cases cover a
duplicate overlapping map, a mixed paired interface plus mortar patch on one
pair, and acceptance of those authored faces when self-contact is disabled.
The overlap case is rejected by the same capability guard; it does not test a
geometric overlap validator.

The overlay and checker were syntax-compiled only. No Metal run, native
contact parity, or box assembly claim is made by this package.

## CPU prototype contract

The prototype uses barycentric triangle coordinates independently on parent
sides A and B. It normalizes each active triangle orientation, rejects invalid
or active patch intersections whose area exceeds the configured geometric
tolerance, and subtracts each patch from the
currently uncovered convex polygons. The resulting triangle cells retain each
covered patch's stable ID; uncovered cells have no patch ID. Shared patch edges
are allowed. A point on a shared edge may be reported in both adjacent patch
IDs; covered/uncovered contact classification should depend on whether the
union membership is nonempty, while each cohesive law/history remains per
patch. Dormant patches do not enter the active union. The cell bound is
explicit and overflow fails closed.

Run the geometry tests with:

```sh
python3 -m unittest -v test_mortar_union_partition.py
```

The tests cover the full parent triangle, three-sector union, adjacent patches
sharing an edge, partial patch complement, distinct A/B maps, dormant patches,
positive-area overlap, invalid barycentric inputs, duplicate IDs, and bounded
workspace overflow. These tests establish CPU polygon arithmetic only.

## Requirements for native multi-patch support

Native support needs one shared bounded partition builder used identically by
both contact candidate generation and private Armijo validation. It must
gather every active mixed-mode face for the exact canonical primitive pair,
preserve the stable face identity and per-face constitutive/history row, and
build separate covered unions in side-A and side-B parent barycentric spaces.
It must process the complete covered/complement partition and skip ordinary
contact only when the corresponding feature is covered on both sides. Coverage
from two different patches must still count as union coverage, including the
declared shared-edge tie rule. Patch complement contact and swept CCD must use
the same partition; a point-only union predicate is not enough.

The source/compiler must impose explicit maxima for patch count, polygon
vertices/cells, and temporary storage; every overflow or unsupported topology
must reject before omitting a patch. Patch interiors should be checked for
overlap independently on each parent side, while edge/vertex adjacency remains
valid. The native force/history evaluator should continue evaluating each
cohesive patch independently rather than merging material state. Until that
shared implementation exists, the compiler guard prevents self-contact from
reaching the ambiguous lookup.

