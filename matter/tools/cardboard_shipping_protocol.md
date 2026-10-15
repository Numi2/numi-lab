# Shipping-size 300 × 200 × 200 mm box design and protocol

`cardboard_shipping_protocol.hpp` adds coupon-derived geometry and checkpoint-linked
assembly/crush protocol records alongside the retained mini-box fixture. The
target is a FEFCO 0201 with 300 mm internal length, 200 mm internal width and
200 mm internal height. Axes are X=length, Y=upright height, Z=width. The
manufacturer lap is 25 mm, slot kerf is 1 mm, and extrusion subdivision is 16
per score-to-score band. These are explicit design choices, not FEFCO
requirements or measured manufacturing tolerances.

## Score allowance gates the dieline

`deriveShippingBoxDesign` refuses a missing, rejected, nonfinite, unbound, or
wrong-recipe allowance coupon. The coupon records the measured signed
difference `score-center span − inside clear dimension` separately for X, Y,
and Z, along with accepted run/coupon IDs and a lowercase SHA-256 digest. The
helper adds those measured offsets to the internal target to derive panel
length, wall height and panel width. A guessed zero allowance, board-caliper
substitution, or coupon from another recipe cannot be passed off as a valid
measurement. Native runs require that accepted coupon or explicit X/Y/Z
engineering allowances, and keep the latter unqualified in their outputs.
`--shipping-preflight-only` is the one exception: it uses a labelled
zero-offset lower-bound geometry and does not step physics.

The partial-pitch cross-section option preserves the selected 7.9 mm
`literature2009` flute pitch. At zero allowance the perimeter would be
25 + 2(300 + 200) = 1025 mm, or 129.7468 pitches. The old whole-pitch section
path rejected that length; its shipping opt-in now generates periodic flute
samples through 129 full periods and clips the last period and any intersected
finite glue footprints at the exact perimeter. The whole-pitch default and
mini fixture code path remain unchanged. The CPU check constructs this flat
geometry only as a synthetic zero-offset unit fixture. It is not the missing
physical coupon and is not exported as an expected/folded shipping mesh.

The section uses the already declared mixed-source `literature2009` geometry:
7.9 mm pitch; 4.210 mm caliper; 0.277 mm liner on each side; 0.191 mm medium;
0.800 × 0.100 mm lower glue bridge; and 0.600 × 0.030 mm upper bridge. It is a
sinusoidal fallback, not the paper's exact arc profile. The constituent data
remain a mixed-source proxy; the coupon must use that declared recipe or the
helper refuses it. A successful topology build is a flat-blank geometry
result, not a crease, fold, glue-strength, or finished-box qualification.

## Score tool and closure declarations

The shipping score-tool defaults are a 1 mm radius punch, a 6 mm open channel,
10 mm support rails, and a 0.25-caliper approach clearance. Candidate depth
ratios are 0.1, 0.2, 0.3, and 0.4 times caliper (0.421–1.684 mm for 4.210 mm
caliper); each native invocation selects exactly one ratio with
`--shipping-score-depth-ratio`, defaulting to 0.25. The 0.1–0.4 depth study
must use separate invocations from independently constructed flat blanks.
The probe no longer applies these depths cumulatively to one blank. Its
50 mm/s default punch speed is an explicit unqualified engineering assumption;
`--shipping-score-speed-mm-s` overrides it. The accepted step count for each
indent and withdrawal is derived from travel, requested speed, and the chosen
timestep, with the existing per-action step cap enforced. Timestep changes are
therefore distinct from changing the prescribed speed. For each of the 12
score axes, the native source places one rounded punch capsule over two finite
BOX support rails. The rails share one kinematic support body, are aligned with
that axis, and leave the authored 6 mm gap centered below the crease. The
miniature keeps its previous broad single-box anvil. This contact stroke does
not establish a calibrated crease or scored hinge: the crease damage law and
same-stock process remain unqualified.

The shipping controller includes executable scoring, wall-fold, closure,
released-settle, platen-compression, gravity-transfer, and finite-mass drop
stages. A stage only advances from an accepted native transaction. A native
run may use a same-recipe accepted allowance coupon or explicit X/Y/Z
engineering allowances; the latter stays unqualified and is not presented as
a coupon measurement. Exact recipe-grade calibration, simulation completion,
and physical box validation are separate verdicts.

The first shipping closure plan is all glued with hot-melt adhesive (HMA),
without tape. It declares a 25 mm manufacturer lap across the measured wall
height and eight actual 100 × 100 mm major/minor flap overlap patches (four at
each end when allowance is zero). Each patch receives an explicit centered
10 mm-wide strip extending across the overlap. The paired major flaps meet at a
butt seam; that line receives no fictitious bond. Strip width and the choice
of HMA are engineering defaults. The exact adhesive grade, bead, cure,
strength, and cohesive law are not calibrated for this stock. Corrugator starch
bridges and carton-closure HMA remain separate interfaces.

The shipping probe can also accept `--starch-interface-parameters FILE`. The
file must identify itself as an `unvalidated_engineering_assumption` and give
all mixed-mode cohesive parameters explicitly. The probe duplicates glue-side
nodes at conforming paper/glue faces, preserves source tetrahedron frames, and
registers dormant native mixed-mode mortar patches with identity barycentric
maps and the declared gap tolerance. The input schema requires the named
`activation_event=initial_accepted_state`; obsolete integer control-step
activation is rejected. Before any scoring load, the probe accepts the
assembled reference in a zero-displacement transaction and submits the
activation command on the next preDynamics transaction. The interface latches
that already accepted geometry and remains dormant if the command-bearing
transaction rejects. These values are not a grade-calibrated starch law.
Internal corrugator starch is distinct from closure HMA. This path does not
supply closure HMA paired faces or an HMA cohesive material. Without an
explicit HMA parameter file, adhesive stages stop at the first required HMA
operation. Supplying `--shipping-hma-parameters FILE` enables the declared lap
and flap operations with that explicit law, including an analogue-transfer or
unvalidated engineering law. That permits an unqualified simulation to
proceed; it does not calibrate the adhesive for this stock or turn starch into
seam closure.

## Assembly release and slow compression/drop sequence

The protocol declaration requires every stage to use the latest accepted
checkpoint. It lists wall erection, manufacturer-lap activation, minor/major
flaps, HMA-set hold, fixture release, and final settle/checkpoint. The probe
offers `--shipping-through-stage scoring|walls|manufacturer-lap|flaps|complete`.
`scoring` performs one selected-depth approach and withdrawal across the
12-axis tool pack; this is a diagnostic contact pass and does not create a
calibrated crease hinge. The selected depth, speed, timestep, and compiled
source fingerprints are bound by a separate score-command SHA in the manifest,
checkpoints, and result. A depth-sensitivity study reruns from fresh flat
blanks at each requested ratio. `walls` continues through the four controlled
wall turns. The lap and flap choices proceed when an explicit HMA parameter
file is supplied; without it, execution stops at the first adhesive operation. Accepted
same-build snapshots and JSON bindings are written under `checkpoints/` after
the flat initial state, score pass, each completed action, and any
held-boundary settle. Held-boundary settling is not a released assembly state.

The default CPU release gates use 1% internal-dimension error per axis (3 mm
for X and 2 mm for Y and Z), a separate 1 mm closure-gap/shape residual,
0.1 mm/s maximum nodal speed, and 0.001 normalized force imbalance. These are
declared engineering gates, not published carton tolerances. Simulation
assembly completion requires accepted actions/checkpoint, release and settle,
the geometric/dimension gates, accepted lap activation, and all eight actual
flap-overlap patches. It permits declared analogue or unvalidated cohesive
laws, while reporting their evidence class separately. Source-grade
calibration additionally requires matched-stock crease, corrugator-starch and
closure-HMA data. Physical shipping-box validation additionally requires a
physical box test. Passing the simulation gate alone establishes neither of
those evidence axes.

The release controller has no permanent FEM node constraints: the final
release scan rejects every active `restAndFixed.w > 0.5`, including authored
fixed tags as well as temporary grips. With the box still held by its boundary
grips, the native support body is moved under the accepted bottom surface. Its
upper face is set strictly below the measured bottom by the smallest FP32
rounding clearance accepted inside the authored contact-slop range. Gravity
remains `(0, -9.80665, 0)` throughout assembly and settling; the shipping axes
are x=inside length, y=upright height, z=inside width. The settled gate permits
reaction samples only from the declared floor body; score punches, score rails,
the platen and impact body must be contact-free. Paired rigid-scene checkpoint
metadata records total and floor-only reaction counts, and the Matter checkpoint
retains the corresponding accepted contact history.
The `.nmcb` snapshot serializes and branch-hash-checks Matter's accepted rigid
and deformable contact-history arenas. MetalWorld's separate impulse warm start
is disabled by the shipping runner; this does not clear or replace those Matter
histories.

The native compression sequence runs two distinct speed branches from the same
accepted support-ready assembly checkpoint. The nominal pass uses 12.5 mm/min;
the second pass uses 6.25 mm/min. Both use accepted 0.25 mm increments and a
140 mm ceiling (70% of the 200 mm target inside height): 560 planned
increments, with nominal durations of 672 s and 1,344 s respectively. Neither
speed comparison changes the timestep as a refinement variable. The run
records mean reaction force from native rigid-proxy impulse divided by the
accepted step duration, complete accepted force–stroke samples, trapezoidal
platen work, FEM nodal kinetic energy, and kinetic/work ratio. The first rising
local peak detector separately requires two accepted declining samples and at
least a 1% drop. The detector latches `Fpeak`, then each branch continues to
the 140 mm endpoint so the postpeak response is retained. A distinct
loading-rate sensitivity gate requires peak force within 2%, full-stroke work
within 2%, and maximum pointwise force difference normalized by the nominal
peak within 5%. The normalized RMS curve difference is also recorded.
The first accepted contact-crossing increment is counted as the first 0.25 mm
of stroke, with a zero-force origin for trapezoidal work; the remaining
schedule ends at the same 140 mm total ceiling.
This comparison is not timestep refinement or physical box qualification. If
any gate fails, drop masses are not instantiated. When all pass, the nominal
12.5 mm/min `Fpeak` sets impact masses to `ratio × Fpeak / g` for ratios 0.5,
1.0, and 1.5 at `g=9.80665 m/s²`.

Numerical capacity admission uses a shared-middle mesh and timestep triplet.
The producer records both adjacent comparisons and requires peak, full-curve,
and work differences each to decrease from coarse-to-middle to
middle-to-fine, in addition to the
finest-pair limits. This is version 2 of the source-bound capacity sidecar.

Each mass/location case restores the same released-assembly FEM snapshot,
resets only the old parked rigid projection, creates a matching
nonarticulated finite-mass rigid body, and records its body properties and
scene state. It accepts a zero-gravity mass/pose initialization, publishes a
branch-specific Matter checkpoint and rigid-scene sidecar, then rebuilds the
gravity-on runner with the same mass/inertia and restores that accepted scene
state before the drop. The seed transaction is excluded from the 250 ms drop
observation window. A branch never reuses a rigid-proxy snapshot projected for
the 1 kg parked body or for another impact mass.
The impactor plan uses 1.2 times the measured X/Z extents of the accepted
outer roof after fixture-free settlement. The measurement is exported with its
exterior-face method, accepted FEM checkpoint SHA, physics fingerprint, and
compiled-world fingerprint. The center and positive quarter-width offset are
computed from those same measured bounds. If the accepted exterior roof cannot
be measured and bound to the checkpoint, the driver keeps the compression
history but does not instantiate any gravity-transfer or drop mass. This is a
geometric footprint check; it does not certify the later collision response.
The impactor remains 50 mm high and starts 100 mm above the measured roof. Six
cases cover the measured roof center and its positive quarter-width offset in Z
for each mass ratio. Each is observed for 250 ms at 0.5 ms per step (500
planned steps). These stages are
executable, but no score, released-assembly, compression, or drop result is
promoted by the protocol declaration. Fpeak and an accepted complete assembly
checkpoint are required before any physical interpretation; the HMA and
corrugator-starch parameters remain uncalibrated for this recipe.

The driver also has a distinct gravity-supported body-weight transfer. For
each `0.5`, `1.0`, and `1.5 × Fpeak/g` mass it restores that same released
assembly snapshot, initializes the finite box-shaped body at the accepted roof
bound with only an FP32 coordinate-rounding allowance, and keeps gravity on.
An explicit transaction-scoped upward force on the rigid body starts at its
weight and decreases with midpoint samples over a 0.5 s ramp. A separate
accepted zero-wrench transaction releases the actuator, followed by 0.25 s of
continued gravity. No FEM node targets imitate a platen. The per-step ledger
records body momentum, native contact impulse and mean force, actuator/gravity
work, rigid and FEM kinetic energy, and gravity-relative potential. The rigid
mass and inertia remain unchanged from accepted initialization through release.
The external-wrench input is distinct from Matter's native contact-wrench
output. This is not the 100 mm impact drop, and the protocol currently records
no measured result or qualification.

Impact refinement is a separate optional series:
`--shipping-impact-dt-refinement` runs 500, 250, and 125 μs steps for every
mass and impact location, restoring the same accepted released assembly for
each branch. It is independent of the body-weight-transfer timestep and of
the nominal/half-speed compression-rate comparison. The gravity-transfer
timestep can be varied with `--shipping-gravity-transfer-dt-us`; use 500, 250,
and 125 μs as separate runs with the same accepted assembly inputs. These
options define executable sensitivity protocols only; no convergence or
physical qualification follows from their presence.

The checked-in JSON is a protocol declaration only. Native physics gates,
source-grade calibration, accepted assembly, compression, drops, and physical
shipping qualification remain false or missing. An explicit analogue law can
support a measured but unqualified assembly simulation. The CPU check builds a
synthetic zero-offset blank and synthetic transaction examples to validate
the conversion, finite geometry, glue patches and checkpoint guards; none of
those synthetic records is physical evidence.

## CPU check and static allocation counts

Run the standalone check with:

```sh
c++ -std=c++23 -O1 -Wall -Wextra -Wpedantic -Werror \
  -Imatter/tools -Imatter/include -Iinclude \
  matter/tools/cardboard_shipping_protocol_check.cpp \
  -o /tmp/cardboard_shipping_protocol_check
/tmp/cardboard_shipping_protocol_check
```

The check prints exact generated node/tetrahedron counts for the synthetic
zero-offset fixture. Before building a shipping blank, the probe writes a
perimeter-scaled, 25%-padded cardinality screen based on that reference count;
the screen is a planning estimate, not exact geometry or admission. After the
mesh compiles it writes the exact known-arena lower bound, and after runtime
initialization it records Matter resident bytes, Metal allocation/working-set
values, and host process footprint when available. It samples again before
each step and stops before advancing when any measured allocation/footprint
exceeds 16 GiB. The reported FEM-only FGMRES figure is a lower bound:
the runtime has two mixed unknowns per FEM node and restart is capped at 16
(shipping runs default to restart 16; `--fgmres-restart` can select 1–16);
the basis and preconditioned basis account for `2*N*(17+16)*16` bytes, plus
80 bytes per `NMFEMNodeStateGPU`. It excludes element history, operator,
contact, checkpoint/snapshot duplication, rigidity, allocator overhead, and
all other runtime arenas. It is not a 24 GB capacity or performance claim.

Shipping assembly also requires the native private full-contact residual
descent path on every physics transaction. Its required FEM, rigid-contact,
and cohesive trial arenas are estimated even when no runtime selector is set;
the report distinguishes these required bytes from arenas actually allocated.
Current runtime admission is bounded to 16,384 FEM nodes, 32,768 tetrahedra,
131,072 deformable-contact slots, 65,536 rigid contact pairs, 16 rigid
proxies, and 4,096 cohesive faces, with separate byte caps. The 300 × 200 ×
200 mm shipping mesh exceeds those limits. The native full-contact path also
has a narrower compiled object/tool scope than the full shipping model. For
this 300 × 200 × 200 mm configuration, the probe exports
`shipping-residual-descent-admission.json` and stops before native
initialization because the required runtime scope and arenas are unavailable.
The probe selects the full-contact residual descent on every shipping physics
transaction when the compiled world is admitted, and rejects conflicting
solver selectors so an environment setting cannot silently fall back to the
ordinary solve. Supported analogue simulations may run with an explicitly
unqualified verdict; execution admission is not solver or physical
qualification. Compile-only and preflight modes export both availability and
qualification separately. No full-box assembly result is implied by a
passing memory estimate or protocol check.
