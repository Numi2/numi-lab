# Human vascular state in the Matter transaction

Matter ABI 26 and package 11 add a persistent passive vascular network to
`WorldSource` and `CompiledWorld`. HumanPack authoring supplies source-bound
anatomical identities and independently sourced SI parameters. Matter owns
lowering, executable state, numerical acceptance and publication.

The new `readHumanPhysiologyNetwork` Apple offline reader consumes
`HumanPack.physiology-native.v1`; `numi-matter-physiologyc INPUT OUTPUT --timestep
SECONDS --environments COUNT` produces the ordinary `.nmatterpack` format. A
caller may compose the resulting `VascularNetworkSource` with existing rigid,
FEM and MPM objects before `compileWorld`. A vascular-only world needs no fake
material or tetrahedron. Unsupported laws, qualification claims, duplicate
JSON keys, duplicate physical-volume owners and invalid endpoints fail closed.
The exact input SHA256, upstream source/authored SHA256 values, semantic strings,
coefficients, initial values, scales and topology enter immutable package
identity. Python performs no physical stepping.

## Equations and authority

Each compartment stores volume V and each species stores amount M. Pressure is
`P = P_external + P_reference + (V - V_reference) / compliance`; concentration
is `M / V`. Each oriented connection stores flow Q with resistance R and
optional inertance L. A backward-Euler root solves:

- `V - V_old + dt * sum(outgoing signed Q) = 0`.
- `L * (Q - Q_old) / dt + R * Q - (P_from - P_to) = 0`.
- `M - M_old + dt * sum(outgoing signed Q * upstream concentration + exchange) = 0`.
- Exchange is `permeability_surface * (c_blood - c_tissue / partition)`;
  the corresponding tissue row receives the opposite amount flux.

The implementation gathers canonical incidence in fixed order without float
atomics. Its analytical Jacobian includes flow direction, changing volume,
concentration and exchange. Unknowns are normalized by authored physical
scales in a new block of the same Newton/FGMRES state as mechanics and fields.
Every seed, norm, Arnoldi/reorthogonalization pass, restart, preconditioner,
correction and final physical certificate covers this block. Vascular rows
have independent declared residual tolerances. Positive volume and nonnegative
amounts constrain the shared line-search step; no post-update clamping injects
mass or species. Nonrepresentable arithmetic fails admission or the transaction.

One private normalized float4.x arena owns accepted vascular state. Candidate,
checkpoint, episode reset, environment rollback, prepared rejection, explicit
snapshot/restore and accepted proof include it. Snapshot archive version 5
retains its bytes; accepted-proof manifest 5 adds `matterVascularState`.
Prepared publication still requires the existing joint Human/Matter protocol.
No additional queue, commit, wait or host integrator exists in the runtime.

## Qualified boundary on physical Apple M4 Pro

The focused native cohort uses Metal API validation and passes 11 compiler,
package, snapshot, physics and transaction checks. Each vascular runtime case
loads a serialized `.nmatterpack` before initialization. Synthetic two-pool
cases cover forward/reverse flow, inertance, finite exchange, a real FEM region
association, 16 steps at 0.01 seconds, two environments, exact snapshot replay,
isolated rejection, isolated episode reset and rejected invalid restoration.

The largest normalized discrepancy from independent FP64 elimination is
2.9931e-7 (gate 8e-5); relative total-volume/species error is at most 8.3157e-8
(gate 3e-5). A three-compartment/three-edge graph with two species, two tissue
reservoirs and three exchanges conserves each species and volume within
6.0723e-8 and replays exactly. Equal-duration hydraulic refinement at 20/10/5 ms
has errors 4.5415e-9, 2.2968e-9 and 1.1552e-9 m3 over 0.16 seconds.

The real prepared Human/Matter test additionally mutates vascular state and
requires a changed Matter proof while Human bytes remain unchanged. Its existing
accept, reject, pending, abort/retry and publication-fence checks include
vascular continuation in their authority comparison. Existing stateful FEM/MPM,
monolithic multiphysics and production rollback checks pass on the changed ABI.
The native input test accepts the source-bound synthetic payload and rejects
12 malformed variants, including escaped duplicate keys. Compiler tests cover
canonical permutations, complete package identities, optional real FEM region
bindings and rehashed malformed cooked layouts.

## Remaining physiology and compatibility gates

The fixture's kidney identifiers exercise anatomical traceability; its values
are synthetic. Fixed tissue reservoirs exchange species with blood, while
optional FEM node weights are an identity association only. Deforming organ
pressure/volume feedback, vessel-wall mechanics, blood-to-body mass partition,
cardiac drive/valves, coronary/portal detail, gas exchange, metabolism, thermal
coupling, subject calibration and physiological-duration validation remain
unqualified. Graph mutation and mechanics feedback reject until implemented.

This package/ABI increment rejects older Matter packages. Prior Human standing,
replay and material receipts retain their original revisions and are historical
for this stack; the vascular checks do not requalify their full source payloads.
