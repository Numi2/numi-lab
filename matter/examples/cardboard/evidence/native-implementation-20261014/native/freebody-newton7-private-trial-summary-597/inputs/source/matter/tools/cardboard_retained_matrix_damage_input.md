# Retained-matrix feasible-set inputs

`cardboard_probe` accepts `--liner-damage-parameters FILE` and
`--medium-damage-parameters FILE` together. The files use the strict
`numi.cardboard.retained-matrix-damage-feasible-set.v1` `key=value` schema
implemented in `cardboard_retained_matrix_damage_input.hpp`. The parser rejects
unknown, repeated, missing, non-finite, or incorrectly sized values. Their exact
file bytes are hashed before parsing and the hash and evidence class are emitted
in the manifest and `damage-overlay-program-pool.json`.

Each file must provide:

- `set_id`, `evidence_class`;
- `hardening_source_id`, `hardening_dataset_id`,
  `hardening_dataset_sha256`, `hardening_mapping`, and six values in
  `plastic_hardening_pa` ordered 11, 22, 33, 23, 13, 12;
- fracture, rate, and densification sub-bindings, each with set ID, SHA-256,
  and evidence class;
- eight values each in `fracture_strength_pa`,
  `fracture_energy_j_per_m2`, and `fracture_rate_time_seconds`. The eight
  entries are normal/shear channels for the four selected material-frame planes
  (material axes 1, 2, 3, then the normalized diagonal plane);
- three values each in `densification_modulus_pa`,
  `densification_onset_strain`, and `densification_rate_time_seconds`, ordered
  along the material axes;
- `damage_cap` (exactly 0 for the undamaged control or 1 for full softening) and
  a positive `smooth_strain_width`.

No constitutive values are supplied as defaults. The source set must retain the
Haj-Ali 2009 liner and NSSC material programs and their authored density. The
overlay clones each regional source material, adds the explicit plastic,
fracture, and compression histories, and remaps only substrate tetrahedra;
finite glue stays on its original material. Geometry-derived crack-band lengths
are calculated from each reference tetrahedron's selected-plane cut area and
volume. Mesh JSON carries the final tetrahedron-to-material map; the sidecar
lists unique program mappings and representative plane areas, lengths, and
thresholds. The mapping hash binds the input-set hashes, frame map, and each
tetrahedron's final material index and four regularization lengths.

These options are candidate authoring and compilation controls, not a calibrated
corrugated-board law. The parser cannot establish that evidence classes identify
the actual paper grade, glue, moisture, rate, or process. Users must keep those
bindings explicit and treat native mechanics results as unqualified until
matched data and physical comparisons are available. The overlay is currently
opt-in and cannot be combined with the separate 13-row Hill residual-scaling
experiment.
