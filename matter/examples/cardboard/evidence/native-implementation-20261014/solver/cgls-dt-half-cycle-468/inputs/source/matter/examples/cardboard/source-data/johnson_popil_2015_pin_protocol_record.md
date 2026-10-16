# Johnson and Popil (2015) pin-test protocol record

This hash-bound project note records the protocol fields used by the native
fixture authoring profile. It is a metadata extraction, not the article PDF or
raw experimental data.

Primary citation: Shane Johnson and Roman Popil, “Corrugated board bonding
defect visualization and characterization,” *International Journal of
Adhesion and Adhesives* 59 (2015), DOI
[10.1016/j.ijadhadh.2015.02.003](https://doi.org/10.1016/j.ijadhadh.2015.02.003).
The author-posted accepted-manuscript text is publicly readable on the
[ResearchGate publication page](https://www.researchgate.net/publication/272753889_Corrugated_board_bonding_defect_visualization_and_characterization).
Its two PDF download endpoints returned 404 during this review; no original
article PDF bytes are retained here.

In §2.2 (printed pp. 10–11), the authors identify the Emerson 06-00-55
C-flute fixture, an Instron 1122 UTM, 50 × 160 mm board specimens, a
12.5 mm/min platen approach, and an out-of-plane selected-facing test using
alternating pin rows. Pins 1, 3, and 5 are attached to one platen; pins 2, 4,
and 6 to the other. The setup separates the bottom liner from the rest of the
corrugated structure. The manuscript does not report the pin cross-section,
diameter, pitch, specimen centering tolerance, or the 50/160 mm specimen axes
relative to MD/CD. Section 2.1 reports 205 g/m² unbleached kraft liner and
122 g/m² regular NSSC medium for the pilot-corrugator defect study; it does not
supply a matched full raw trace file.

The associated 2016 author presentation is a separate artifact. Its slide 15
is the direct source of the digitized illustrative load-versus-displacement
curves, and slide 16 is the direct source of the digitized summary markers.
The accessible manuscript contains its own representative waveform and
all-defect summary figures, but there is no established point-for-point mapping
from the presentation pixels to those article figures or individual specimens.
The two artifacts share study context and may overlap in tested material; do
not count them as independent datasets or substitute article figure pixels for
the checked-in presentation digitization.

The article's §3.1 narrative reports a pin-strength range of 0.17–0.8 N/mm,
while the slide-16 marker centers digitize to 0.13–0.82 N/mm. The differences
are preserved as a cross-artifact discrepancy, not reconciled through plot
rounding or assumed specimen identity. Error bars are visible in the slide,
but their statistical definition is not assigned in this package; the CSV
uncertainties describe only marker-center pixel reading.

The slide-15 axes are load in N and displacement in mm, so integrating the
visible trace gives N·mm. The article narrative and slide 16 label summary
pin-adhesion work as N-m. That conflict remains unresolved. The slide-16
ordinate is therefore retained as an unqualified axis number and must not be
converted or used as a physical work target.
