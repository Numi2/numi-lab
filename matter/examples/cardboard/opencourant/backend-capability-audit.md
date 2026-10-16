# OpenCourant backend capability audit

**Decision: stop this migration at admission.** The pinned free Linux ARM64
package did not execute even one implicit mechanics cycle: the owner-run
LAW1 and LAW117 controls both parsed and started, then exited at cycle 0 with
`MUMPS required` (exit 2). The PCG card was accepted. That makes the current
packaged implicit path unavailable; it is not evidence of a cohesive-law
failure or a physical assembly result. This meets the stopping rule in
[`admission-plan.json`](admission-plan.json). No solver changes or shell
substitution are proposed.

The evidence boundary is the OpenCourant `latest-20261011` source at
`cb760c206e346eff12161a2268b134ef9d48a4ef`, and package SHA-256
`ded9814e2460767353d591c89695ba1f76251291041943ab0ad73070a6c865f7`.
The source links below are pinned to that commit. Native outcomes in this
note are the 2026-10-16 coordinator-run smoke results; they are reported as
observed, not as independently reproduced here.

## Implicit solver admission

`/IMPL/SOLV/01` is parsed as `ISOLV=1`; the following list-directed record is
read into `IPREC, L_LIM, ITOL, L_TOL`. Thus the tested
`1 0 3 1.e-10` body selects PCG with ITOL 3. The native log also confirmed
that selection. The four values do not need fixed-width padding; the reader
uses list-directed input. See [`freimpl.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/input/freimpl.F#L261-L267).

That parser result does not provide an executable PCG path in the tested
package. The normal implicit dispatch in `resol.F` calls `IMP_SOLV` only in
the `MUMPS5` preprocessor branch; its alternative emits a fatal MUMPS error.
Inside `imp_solv.F`, the no-`MUMPS5` preprocessor alternative is also a fatal
error. The buckling dispatch has a separate fatal branch. These compile-time
guards explain why accepting the PCG card is insufficient: the observed
binary still lacks the required implicit implementation. See
[`resol.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/engine/resol.F#L8767-L8806),
[`imp_solv.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/implicit/imp_solv.F#L1297-L1310),
and the separate buckling branch in
[`resol.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/engine/resol.F#L8807-L8855).

The native output printed source line `8852`, but the source has multiple
MUMPS fatal sites under preprocessor alternatives. Without the exact native
preprocessed line map and branch state, that printed number alone does not
identify which site ran. The independent conclusion is unchanged: both
LAW1 and LAW117 implicit smoke decks terminate before mechanics with the
MUMPS fatal. An earlier no-title/output configuration also crashed at cycle
0; that separate startup failure is not used as mechanics evidence.

## Material, element, and activation findings

LAW112 declares the `SOLID_ORTHOTROPIC` property class, which assigns solid
property code 2. The compatibility table rejects code 2 for property type
14, but permits it for type 6. The corrected fixture therefore uses
`/PROP/TYPE6/1`; this resolves the input incompatibility, not the solver
admission failure. See [`hm_read_mat112.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/starter/source/materials/mat/mat112/hm_read_mat112.F#L490-L500),
[`init_mat_keyword.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/starter/source/materials/mat/init_mat_keyword.F#L208-L219),
and [`check_mat_elem_prop_compatibility.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/starter/source/materials/mat/check_mat_elem_prop_compatibility.F#L192-L218).

There is a real whole-group activation mechanism. `forint.F` skips a group
when its group-off flag is set. `ELOFF` records activated solid elements,
sets their group `OFF` entries, then recomputes the group-off flag from all
elements. This corrects the overly broad interpretation that deactivation
must be irreversible or that a dedicated fully-off group can never re-enter
the element loop. See [`forint.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/forint.F#L338-L347)
and [`eloff.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/eloff.F#L107-L159),
[`eloff.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/eloff.F#L265-L270).

This is not a proof of stress-free cohesive activation. For TYPE43,
`SUSER43` constructs a local frame/area from its geometry call, multiplies
the strain increments by the local element `OFF` value, and passes stored
`EPE`, stress and damage variables to LAW117. LAW117 initializes its
characteristic length from the current area when that variable is zero.
Those are relevant ingredients, but the inspected activation path does not
show an explicit, activation-time reset/rebase of cohesive reference state
or a measured preservation test for irreversible history. Partial activation
inside an otherwise active group also must not be assumed equivalent to
fully-off-group activation. See [`suser43.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/solid/sconnect/suser43.F#L255-L269),
[`suser43.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/solid/sconnect/suser43.F#L365-L384),
[`suser43.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/solid/sconnect/suser43.F#L436-L443),
and [`sigeps117.F90`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/materials/mat/mat117/sigeps117.F90#L130-L145).

For implicit stiffness, the 8-node solid dispatcher warns for LAW112 and
routes the general case through `S8ZKE3`; no LAW117/TYPE43-specific
algorithmic cohesive tangent is established by that dispatch. The separate
`ITY==6` spring branch only recognizes its enumerated spring types and is
not evidence about TYPE43. Consequently, even with a usable implicit binary,
TYPE43/LAW117 tangent behavior would still need a native force/stiffness
control before using it in a physical assembly. See
[`imp_glob_k.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/implicit/imp_glob_k.F#L229-L294)
and [`imp_glob_k.F`](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/implicit/imp_glob_k.F#L3443-L3489).

## Admission disposition

The property-card mismatch was repaired, and PCG input was parsed, but both
implicit material controls stopped before the first mechanics cycle. That is
an indispensable unavailable capability in the required slow-stage configuration.
Do not advance to assembly, claim current-geometry bond activation, or infer
physical calibration from these parser and source checks. Preserve the native
failure and stop this backend migration unless a separately admitted free
ARM64 package provides an executable implicit solver path.
