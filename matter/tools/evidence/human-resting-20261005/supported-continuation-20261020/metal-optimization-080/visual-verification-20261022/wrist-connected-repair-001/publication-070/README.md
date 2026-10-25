# Connected skin/fat reconstruction and mass ownership
Work was performed on SSH Numi, Mac18,5, macOS 27.0. This increment is authoring/ownership work. No new anatomy candidate or physical timestep has been admitted, and no simulation performance or clinical claim is made.

## Geometry
The exact registered 294 source was separated from the coarse mechanics and the transfer operator. Native weighted skinning reproduces the emitted detailed surface within 2.252 micrometres. Forcing the coarse O/I/D columns to the source outer anchors creates 457 nonpositive cells; preserving layer thickness alone does not preserve a valid connected volume.

Three connected outer patches can be retessellated without moving vertices or changing outside faces. Exact whole-coarse-surface intersections fall from 9 to 0, with all 45 pre-existing boundary edges retained. This does not assert a closed complete skin shell.

The corresponding O/I/D cavity topology changes 72 cells and preserves exact oriented side chains, but changes nonplanar cap triangulations and reference material volume. Its row map is not conservative mass ancestry. The target outer surface with the old I/D coordinates still has 126 nonpositive cells.

Two bounded whole-interior fits failed and are retained: 006 ended with 154 nonpositive cells; alternate initialization 007 ended with 400. Neither is promoted. A fixed-surface necessary feasibility check identifies inner node 3656 / outer 1245: all 15 associated tetrahedra share fixed outer faces whose halfspace constraints admit no strictly positive shared-apex placement to floating-point diagnostic precision, even without displacement bounds. The LP dual witness is retained. A connected topology reconstruction is required.

## Physical accounting
A FEM node's one registered frame/attachment owner is distinct from the source bodies whose mass created it. The production preparation and tissue-mass consumers are extended with explicit source contributions, preserving unique node rows, per-node mass closure, original source local points, rigid mass/first/second moments, and COM rebasing. Legacy tissues retain a one-contribution-per-node fallback. This introduces no CPU per-step physics or new force owner.

The source-only native export retains the original 9,644 premerge contribution rows. Compiled native parent-cell masses are authoritative: skin density is exactly Float32 1000.0000610351562 kg/m3, fat 900 kg/m3. The conservative native-parent→37,935-child reference map covers all 28,632 parents with no cross-material overlap; maximum coverage residuals are 7.77e-12 child and 6.13e-12 parent.

Independent full-ledger checking of the producer's 1/16 child-corner/parent-corner source attribution closes all 18,640 child node/material entries. Donor/material cooked→child mass, first and second moments agree within 4.84e-15 relative error. Source→compiled rounding is reported separately: skin +3.82567e-7kg, fat −3.85005e-7kg. No global mass rescale hides these differences.

This successful map applies to the retained 037006 refinement/reference domain, not the new 72-cell cavity topology or a qualified current anatomical geometry.

## Remaining gates
Connected source patch and interior reconstruction, full detailed transfer, all nearby solids, source-material pullback for any changed cap domain, positive exact Float32 cells, complete retained poses and native integrated qualification remain required. Failed geometries are not installed. Native qualification also requires restoring the declared 12 GiB free-space floor; current available space is about 3.4 GiB. The existing native scene and shared dirty checkout remain preserved.

## Validation of this source increment
33 focused Python checks passed across connected authoring, source-frame and source-topology helpers. The source-contribution producer fixture, whole-body mass reference and tissue mass probe passed; the probe includes 64 rigid-motion and 11 negative cases. The lead independently reran all three resulting binaries and pinned their hashes. The complete resting-scene translation unit passed Objective-C++ syntax checking with frozen edited source; this is not a linked native scene or GPU qualification. Exact commands, failures and logs are retained in evidence/mass-owner-tests/.
