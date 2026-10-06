# FEFCO 0201 corrugated blank authoring helper

`matter/tools/cardboard_box_blank.hpp` authors a flat, unscored FEFCO 0201 / regular-slotted-container blank from a `ConformingGlueMesh` section. It keeps the source paper/medium/glue tetrahedral section continuous through the body and flaps, clips the section at slot and score-to-score panel boundaries, and leaves score lines as metadata. It does not prescribe folded node coordinates, simulate creasing, or claim physical assembly.

## Source geometry and interpretation

The [FEFCO 12th-edition code](https://www.fefco.org/sites/default/files/files/FEFCO%20Code_WEB%287%29.pdf) identifies the 0201 drawing on printed p. 18 as a four-panel `L-W-L-W` blank with top and bottom flaps of one-half `W`. The [PMMI/Fibre Box Association RSC technical report](https://www.fibrebox.org/assets/2025/07/B155_TR2-1_RSC_Containers_2018_Edition.pdf) defines an RSC as one scored and slotted sheet with four panels and eight flaps (printed p. 6); its Figure 7 shows the flat blank (printed p. 7). The report defines panel dimensions score-center to score-center and flap depth score-center to edge (p. 7), and explains that score allowance depends on board and score pattern (p. 7). It defines slots as paired cuts removing a narrow strip (p. 7).

The report calls itself voluntary and explicitly says it is **not normative** (printed p. 2). Its tolerance scope is B- or C-flute single-wall board with panel dimensions from 4 to 25 inches (printed p. 4). The small geometry fixture below is outside that dimensional scope and is a helper test, not a standards-conforming package or manufacturing drawing. FEFCO specifies a box style, not this fixture's absolute dimensions, slot kerf, score allowance, or joint width.

The implementation uses a body-height-only manufacturer tab. The report describes the manufacturer joint as an overlap tab fastened by adhesive or staples and permits it to be inside or outside the container (printed p. 6); it does not supply a tab width. The tab width is therefore an explicit caller input. The helper places three narrow, through-thickness slot strips at the three internal panel boundaries at both ends. Its `slotKerf` is likewise an explicit idealized cut width: the sources describe paired cuts and removal of a narrow strip but do not provide a nominal kerf for this case.

## Executable geometry fixture

The CPU fixture in `matter/tools/cardboard_box_blank_check.cpp` builds on the existing source-labeled literature proxy cross section and specifies the following SI inputs:

| Input | Value | Status |
|---|---:|---|
| Flute pitch | 7.9 mm | Haj-Ali et al. (2009), Table 5, measured specimen value; the chosen board recipe is still a mixed-source proxy |
| Board caliper | 4.210 mm | Popil (2017), Table 5.4, commercial WC 4226C 42 board |
| Liner thickness, each face | 0.277 mm | Popil (2017), Table 4.1, separate 42# liner coupon; not identified as the WC 4226C or Haj-Ali test roll |
| Medium thickness | 0.191 mm | Popil (2017), Table 4.1, separate NSSC medium coupon; not identified as the WC 4226C or Haj-Ali test roll |
| Lower finite glue bridge | 0.800 mm wide × 0.100 mm gap | Haj-Ali et al. (2009), Table 8, Type II double-back zone |
| Upper finite glue bridge | 0.600 mm wide × 0.030 mm gap | Haj-Ali et al. (2009), Table 8, Type II single-face zone |
| Sinusoid centerline height | 3.335 mm peak-to-trough | Derived closure: `4.210 − 2×0.277 − 0.191 − 0.100 − 0.030`; not a reported flute-height measurement |
| Score-to-score panel length `L` | 31.6 mm = 4 pitches | Explicit example choice |
| Score-to-score panel width `W` | 23.7 mm = 3 pitches | Explicit example choice |
| Body height `H` | 31.6 mm = 4 pitches | Explicit example choice |
| Manufacturer-tab width `J` | 7.9 mm = 1 pitch | Explicit example choice; not source-specified |
| Slot strip kerf | 0.5 mm | Numerical geometry input; not source-specified |
| Extrusion subdivisions per blank band | 2 | Mesh resolution input |

This gives a perimeter span `J + 2(L + W) = 118.5 mm` (15 pitches), a blank height `H + W = 55.3 mm`, and flap depth `W/2 = 11.85 mm`. The corrugation section runs around the perimeter while the blank is extruded through its height; the flutes therefore run vertically in this top-opening blank, consistent with the report's usual top-opening RSC orientation (printed p. 6). The fixture's L/W/H/J choices are solely small, manageable authoring inputs. The paper thicknesses and finished-board caliper come from separate source samples, so the fixture is not an exact same-roll reconstruction.

The helper signature is:

```cpp
FEFCO0201Blank buildFEFCO0201Blank(
    const FEFCO0201BlankConfig& config,
    const ConformingGlueMesh& section);
```

`section.config.length` must match `J + 2(L + W)`. `blank.points` and `blank.tetrahedra` are directly adaptable to native FEM nodes, tetrahedra, material indices, and material-frame rotations. `blank.panels` identifies the four body spans; `blank.slots` records each end cut; `blank.scoreLines` records four body crease centerlines and eight flap-score centerlines with `physicallyScored=false`. The section clipping occurs at the actual caller-provided score and slot coordinates, including when those coordinates do not coincide with a flute pitch or source mesh vertex.

The fixture creates **5,238 nodes and 15,312 tetrahedra**, with three material roles (liners, medium, finite glue bridges), 12 score-centerline metadata records, six slot strips, and 15 ordered assembly-action targets. These counts are for the stated resolution and are not converged box results. Test output and checks are limited to generated geometry: positive tetrahedron orientation, finite and correctly bounded SI extents, per-material integrated volume, manifold tetrahedral faces, absence of direct liner-medium contact, connected laminate/blank topology, expected panel and flap layout, no cells in removed slot strips, and invalid-input rejection.

Score metadata uses the authored `z=0` sheet-face reference plane. It does not
define a neutral axis or a physical hinge. A later tooling/folding consumer
must explicitly resolve the score reference plane and through-thickness
deformation before applying any motion.

## Assembly targets are not a fold simulation

`blank.assembly` orders four wall/tab fold targets, manufacturer-joint bonding, then the inner-flap and outer-flap targets and end-seal operations. Each action has `executed=false`; the schedule status is `targets-authored-not-executed`. Its 90-degree fold values describe unattained target angles only. In layout JSON v2, each action's `feature` is a typed reference: wall actions name `blank.panels[index]` and separately give `hinge_score_line_index` into `blank.scoreLines`; flap actions name their score-line index directly; joint bonding names the singleton tab region; and end sealing uses 0 for bottom and 1 for top. New exports use layout schema v2; already-retained v1 run evidence remains unchanged. These references disambiguate metadata only and do not specify calibrated tool contact points, grip forces, stroke histories, joint adhesive volume, or a solved motion. No score crushing, paper fracture, springback, force-angle response, contact, or physical box closure is represented. The RSC report says scores should support a 180-degree fold without fracture or continuous checking (printed p. 8), but it does not provide this board's scoring tool, crease law, or load path; accordingly this helper does not invent one.

The fixture is registered as `matter.compiler.cardboard_box_blank`. A standalone CPU check is:

```sh
c++ -std=c++23 -O1 -Wall -Wextra -Wpedantic -Werror \
  -Imatter/tools matter/tools/cardboard_box_blank_check.cpp \
  -o /tmp/cardboard_box_blank_check
/tmp/cardboard_box_blank_check
```

For the literature-model caveats, including glue and paper constitutive assumptions, see [`manufacturing-sources.md`](manufacturing-sources.md) and [`manufacturing-source-manifest.json`](manufacturing-source-manifest.json).
