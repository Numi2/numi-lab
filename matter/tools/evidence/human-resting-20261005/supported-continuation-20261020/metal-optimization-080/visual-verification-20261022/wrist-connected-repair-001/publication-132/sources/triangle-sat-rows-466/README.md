# Triangle SAT guard rows 466

This opt-in helper extends the 458 row builder without modifying 458. Call
`build_fixed_bone_outward_plane_rows(..., separating_axis_pairs=None)` with
458's inputs. With no selected identities, it preserves the 458 outward
bone-face plane and returns the same top-level fields: `G`, `lower`,
`guard_metadata`, `C`, `h`, `weights`, `metadata`, and `diagnostics`.

To opt in per typed pair, pass an iterable of `(face_row, body, bone_id,
triangle_id)` tuples as `separating_axis_pairs`. For those pairs, 466 considers
the fixed bone face normal, the skin face normal, and the nine bone-edge ×
skin-edge axes. It normalizes each nondegenerate candidate and keeps signs
whose dot with the verified outward bone-face normal is nonnegative; exact
tangent axes retain both signs. It chooses the axis with the lowest
`max(max_fixed_projection - min_skin_projection + float32_ulp_margin, 0)`;
ties use stable candidate order.

Each selected pair receives all three hard rows
`n · d_i >= max_fixed_projection + margin - n · x_i`. Meeting all three puts
the entire moved skin triangle beyond the fixed triangle's support plane on a
triangle SAT candidate axis. Soft rows remain limited to positive movable
corner deficits. Metadata records axis source/index/sign, outward dot, cost,
projection extrema, and the Float32 margin. These local rows do not certify
whole-mesh clearance; exact audits remain required.

Run focused tests with:

```sh
PYTHONPATH=/path/to/fixed-bone-outward-plane-458:/path/to/triangle-sat-rows-466 python -m unittest -v test_fixed_bone_outward_plane_458 test_triangle_sat_rows_466
```

The saved receipt records nine passing synthetic tests across the unchanged
458 owner and the 466 extension. No solver, mesh audit, pipeline wrapper, or
geometry run was used.


## Pre-fit adapter integration receipt

test_run_sat_fit_466_integration.py exercises the actual 408 clearance-row
builder through the 458 install_objective path with a two-pair synthetic
fixture. It verifies the 408 point-row prefix is unchanged, both pairs select
SAT rows, six guard rows are produced (three per pair), axis metadata reaches
guard and soft-target rows, and nonpositive corner guards are appended after
the existing self and fixed-boundary hard rows. The test does not run either
wrapper main, a fit/LP, or a mesh audit.

The actual 457 face-target and point-target NPZ files were independently
hashed against the 458 CLI pins; the 456 base candidate and audit-159 report
also match the 458 pins. These checks confirm input identity only; the
integration fixture remains synthetic. The reused 458 report field
three_corner_lower_bounds_per_pair contains the total lower-row count
(three times pair count), despite its name.
