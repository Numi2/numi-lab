# Pin-adhesion source observables and calibration boundary

Run the source-bound extractor from the repository root:

```sh
python3 matter/tools/cardboard_pin_adhesion_source_observables.py \
  --output /tmp/pin-adhesion-source-observables.json
python3 matter/tools/cardboard_pin_adhesion_source_observables_test.py
```

The extractor validates the catalog, CSV, and engineering-profile hashes before
reporting any values. It treats the 2015 Johnson–Popil article as protocol
context and the 2016 author presentation as the direct source of the digitized
pixels. It fails if the article is silently added as a second pixel source.

The article reports a 50 × 160 mm C-flute specimen, six alternating pins, and
12.5 mm/min platen approach. It does not report pin diameter/cross-section,
pitch or flute registration, pin-end geometry, centering tolerance, specimen
MD/CD orientation, specimen-specific caliper, or moisture state. The checked-in
1 mm diameter, 8 mm pitch, centered array, 1 mm overhang, and 5 mm caliper are
engineering prototype choices. The report binds those choices by hash but does
not present them as published dimensions or nuisance bounds.

The low-adhesion slide-15 digitization supports a sampled visible peak of 78 N
(pixel interval 68–88 N), a first-to-peak visible secant of 32 N/mm (conservative
pixel interval about 22.89–43.28 N/mm), and a trapezoid area of 76.625 N·mm
(outer pointwise digitization enclosure about 50.855–105.075 N·mm). These are
visible-curve descriptors only. The source displacement origin is missing, so
the secant is not the initial fixture stiffness. The plotted trace is sparse,
and its pixel bands do not bound the unknown curve shape between digitized
points. The area is not the separately reported slide-16 pin-work quantity.

The high-response slide-15 trace reaches a visible sampled 364 N (352–376 N
pixel interval) before its later branch is obscured. Its 11.6–13.4 mm
first-to-last visible branch secant is 185.56 N/mm (about 167.71–205.95
N/mm from the stated point bands), but is not initial stiffness. Its full-curve
peak and work therefore remain unavailable. Slide 16 contains unpaired marker ranges:
strength centers 0.13–0.82 N/mm and stiffness centers 56–163 N/mm. The article
narrative separately reports 0.17–0.8 N/mm; the ranges remain distinct because
there is no point/specimen mapping between article Figure 5 and slide 16. The
slide's work axis remains unresolved because the printed N·m label conflicts
with integrating the slide-15 N-by-mm trace. These summary markers are not
joined to the slide-15 traces or article Figure 5, and their visible error-bar
meaning is unspecified.

No condition-level source failure-locus labels are available. Whole-fixture
pin response cannot identify local starch traction, GIc, GIIc, or mixed-mode
interaction by itself. The article study context is 205 g/m² unbleached kraft
with 122 g/m² regular NSSC; the retained `literature2009` recipe uses a separate
cross-source paper/adhesive composition. No published geometry or grade
transfer interval bounds that mismatch. The source extractor therefore emits
no material fit, no feasible cohesive-parameter set, and no confidence
interval.

The existing `cardboard_calibration_ensemble.py` remains the whole-fixture
native adjudicator. It requires complete hash-bound native histories and
certificates, uses the low trace's origin-free 30 N threshold window when
requested, and keeps the pin work unit and failure locus unavailable. A future
native run can be compared to these source-side descriptors only as an
explicitly bounded analogue after supplying explicitly authored nuisance/grade
transfer intervals. Those intervals must be labeled sensitivity assumptions;
the public pin publication supplies none.
