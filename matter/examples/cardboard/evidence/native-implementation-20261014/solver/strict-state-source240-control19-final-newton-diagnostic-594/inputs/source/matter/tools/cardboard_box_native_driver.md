# Native FEFCO 0201 box-tool driver

The `numi-matter-cardboard-box-native-driver` executable reuses the probe's
published-source `WorldSource`, finite-paper/glue blank, and native Matter
runtime. It defaults to the `literature2009` material recipe and authored
FEFCO 0201 layout. The blank begins in its authored flat, stress-free
reference geometry. From the initial accepted flat snapshot, the driver first
acquires temporary `w=3` kinematic clamps on exposed boundary nodes in the
manufacturer-tab body. This defines the tab reference frame for relative wall
turns; it does not move bulk nodes into a fold. Later commands grip selected
free-edge boundary nodes while the tetrahedral interior remains a native FEM
unknown.

The score tool has twelve capsule segments aligned with the authored score
lines and a separate support anvil. It lowers the fixed-orientation die pack
to first contact and retracts it. The default punch radius is 0.2 mm, clearance
is 0.05 mm, and each approach/retract leg uses 16 steps. The first-contact
center height is `boardCaliper + punchRadius`; there is no overtravel stroke
beyond contact. These are numerical instrument choices, not FEFCO source
parameters. The tool path records contact response only: the score lines remain
continuous paper geometry, there is no crease crushing or damage law, and this
operation does not create a physically scored blank.

For the four wall actions, the first target is a signed relative turn of the
active wall against the anchored tab; each following wall is measured against
its adjacent parent wall. A rigid rotation of the whole flat blank therefore
cannot satisfy the first wall target. The planner fits accepted panel interiors
and commands only the current free-edge grip nodes. Completed wall boundary
clamps remain active as external supports through the later tube and seam
steps. Flap turns use the accepted state at the start of that flap action as
the reference, rather than the original flat coordinates. The angle ledger
reports accepted geometric fits; it is not a force-angle calibration.

Angle completion also requires an accepted-shape fit. For a wall, the active
wall and its adjacent parent panel are checked independently; the reported
gate uses the worse RMS residual and lower coverage so a larger panel cannot
hide a poor fit on a smaller parent. A flap is checked against its action-start
accepted interior shape under the measured rigid rotation about its current
score axis. The defaults require RMS residual no greater than `0.5 ×
boardCaliper` and coverage of at least 95%. These are configurable engineering
gates (`--box-shape-residual-calipers` and `--box-shape-coverage-min`), not
published corrugated-board tolerances. If the target angle is reached while the
shape gate fails, the plan emits no new grip targets, the top-level result is
`shape_fit_rejected`, remaining actions are recorded as
`not_attempted_after_shape_fit_rejection`, and the executable exits with code
4. A failed native solve remains `failed_native` and exits 2; the missing-law
gate remains distinct and exits 3.

The default `--box-assembly-native` sequence stops at the manufacturer's joint
because the model has no corrugated-board adhesive cohesive law. It records
nearest-node seam/flap gaps as diagnostics and does not treat contact as a bond.
The opt-in `--box-fixture-joint` path is a separate tooling experiment. At the
manufacturer's joint it uses temporary FEM boundary clamps on matching tab and
seam perimeter nodes, moving the tab-edge clamp through 64 native substeps to
a configurable stand-off (default 0.05 mm). At each end-closure action it
similarly grips the free-edge nodes of all four folded flaps at their accepted
positions. These are external idealized clamps. No flap interiors or bulk
nodes are placed directly, and no crease weakening or adhesive law is added.
The program attempts the remaining eight flap folds and two end-clamp actions
sequentially, stopping at the first rejected native step. Any accepted fixture
path is labelled externally held, unbonded, unsealed, and not a fixture-free
physical assembly. No successful physical fold or box assembly is claimed.

The blank is represented by one deformable FEM object. With the finite-glue
`literature2009` recipe, deformable contact/self-contact is enabled by default.
`--without-self-contact` explicitly disables the deformable-contact path for
that one board object and is a numerical control, not an alternative physical
validation. The result JSON records whether that control was used and whether
self-contact was enabled.

For replay, `--metallib FILE` selects an explicit retained native Metal image;
record and verify the image hash with the run inputs. `--fgmres-iterations N`
can raise the Krylov-column budget up to the native ceiling of 1024. The
`literature2009` preset still selects 32 iterations by default. The compiled
generic helper defaults remain unchanged (10 iterations/restart); a larger cap
is not evidence of convergence or physical accuracy.

Outputs include `box-native-steps.csv` (accepted and rejected native attempts
and both environment statuses), `box-score-angles.csv` (the twelve geometric
observables after each attempted step), `box-actions.csv` (all fifteen source
actions, including blocked and not-attempted states), final and per-action
accepted meshes, the compiled-world fingerprint, and an explicit
`physical_validation: false` result. The second environment remains a
stationary reference.

Example invocation after building the native target:

```sh
numi-matter-cardboard-box-native-driver \
  --output /tmp/cardboard-box-native \
  --dt 0.0001 \
  --box-score-steps 32 \
  --box-angle-increment-deg 0.25
```

To run the external-fixture path, add `--box-fixture-joint`; adjust the
explicit stand-off only for a documented tooling study, for example
`--box-fixture-gap-mm 0.05 --box-fixture-steps 64`. For a replay, add the
retained image explicitly with `--metallib /path/to/retained.metallib`.

Historical native002 record, from an earlier probe/controller revision: the
first full-blank execution accepted 15 tool-approach steps. Its first contact
step was rejected by the nonlinear certificate (status 10, residual
1.31136e-6 at tolerance 1e-6); all fifteen assembly actions were retained as
unattempted after that failure. Keep this as a failed historical result, not as
evidence of current-controller success or physical box behavior. Physical fold
success and comparison with a physical box remain unqualified.
