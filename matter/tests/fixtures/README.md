# Deformable contact fixtures

`open_knee_left_crossing_cell.txt` and
`open_knee_right_crossing_cell.txt` are the eight SI-position triples of the
two source tetrahedra adjacent to crossing pair 0 in the bilateral Open
Knee(s) NHKNEE1 payloads. They are byte-for-byte copies of the files emitted
by `numilab-human/tools/build_patellofemoral_crossing_fixture.py` on
30 September 2026. The Human receipts pin the Open Knee(s) source files,
compiled payloads, owner tetrahedra and full 18-pair crossing loop. They
exercise the production Metal narrowphase with real source geometry; the
probe uses a synthetic silicone material and no physiological load.

`fem_close_contact_cells.txt` is a synthetic pair of 1 mm tetrahedra whose
facing triangles begin 5 µm apart. Its contact activation and accepted step
check that the strict-intersection rejection does not reject ordinary nearby
surfaces.
