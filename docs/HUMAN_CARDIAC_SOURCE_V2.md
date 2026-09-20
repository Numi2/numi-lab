# Human cardiac source owner, version 2

Matter owns the hydraulic state and solve on its existing accepted-step timeline. Human authors the source graph as `HumanPack.physiology-native.v2`; `readHumanPhysiologyNetwork` lowers it into `WorldSource::vascular`, and the ordinary compiler/package path produces `CompiledWorld::vascular`. There is no separate cardiac integrator or host-controlled phase counter in the native runtime.

The public definitions are in [matter.hpp](../matter/include/numi/matter/matter.hpp) and [shared.h](../matter/include/numi/matter/shared.h). Admission and cooking live in [vascular_impl.cpp](../matter/src/vascular_impl.cpp); residuals, derivatives, working sets and accepted time live in [vascular.metalinc](../matter/src/metal/vascular.metalinc) and [runtime.mm](../matter/src/runtime.mm). [The passive vascular owner document](HUMAN_VASCULAR_TRANSACTION_V1.md) describes the preceding capability and its historical evidence.

## Authored quantities and laws

Every unknown occupies one `float4.x`, storing the physical quantity divided by its authored positive variable scale. Volume/storage, flow and amount rows have independent residual scales and normalized tolerances. Other lanes remain zero.

| Quantity | SI unit |
| --- | --- |
| Absolute volume or signed storage displacement | m³ |
| Pressure | Pa |
| Compliance | m³/Pa |
| Elastance | Pa/m³ |
| Flow | m³/s |
| Resistance | Pa·s/m³ |
| Inertance | Pa·s²/m³ |
| Orifice coefficient `CV` | m³/(s·√Pa) |
| Period | s |
| Activation start/end/duration | Dimensionless cycle fractions |
| Species amount | mol |
| Permeability-surface product | m³/s |

`VascularStorageKind::absoluteVolume` requires positive volume. `storageDisplacement` permits finite signed values: the six source vascular pressure states are represented as **C·P storage**, not measured absolute blood volumes. A graph containing signed storage rejects species, tissue reservoirs and exchanges, because their concentrations require physical positive volumes. The four cardiac chambers retain absolute volumes. Synthetic absolute-volume graphs may combine valves, species transport and tissue exchange.

For storage coordinate `X`, pressure is

```text
P = Pexternal + Preference + E(p) · (X − Xreference)
X − Xold + dt · sum(outgoing signed Q) = 0
```

`VascularPressureLaw::linearCompliance` uses `E=1/C`. The ventricular and atrial laws prescribe `E(p)=Emin+(Emax−Emin)·a(p)/2`, with candidate phase frozen throughout Newton. Their compliance field is zero. The exact source literal `sourcePi=3.14159` is retained.

For ventricular `start=Ts1`, `end=Ts2`:

```text
p <= start:       a = 1 − cos(pi·p/start)
start < p <= end: a = 1 + cos(pi·(p−start)/(end−start))
otherwise:        a = 0
```

For atrial `start=Tpwb`, `duration=Tpww`, the admitted source waveform crosses the cycle boundary:

```text
p <= start+duration−1: a = 1 − cos(2·pi·(p−start+1)/duration)
p <= start:            a = 0
otherwise:             a = 1 − cos(2·pi·(p−start)/duration)
```

The source atrial values are `start=.92`, `duration=.09`. These are cycle fractions, not seconds.

`VascularFlowLaw::resistanceInertance` retains the signed law

```text
L·(Q−Qold)/dt + R·Q − (Pfrom−Pto) = 0
```

`oneWayOrifice` implements the curated source's algebraic valve:

```text
Q = CV·sqrt(max(Pfrom−Pto, 0))
g = (Q/CV)² − (Pfrom−Pto)
F = min(Q·pressureScale/flowScale, g) = 0
```

The native residual uses the last form, with a deterministic closed branch at a tie and its consistent analytic directional derivative. This avoids a singular square-root derivative. The selected valve has no inertance or leaflet state.

A primal working set handles directions that cross `Q=0`. Constrained flow is removed consistently from every incident conservation and species flux row, its affine correction is exactly `deltaQ=−Q`, and the shared update is `Qnew=(1−alpha)·Q`. A newly selected constraint receives a free-variable solve before pressure-based release. Opening steps use bounded Armijo backtracking on the original physical residual. Final certification always evaluates the original min law, conservation equations, admissibility and authored row tolerances. The working mask and trial buffers are private scratch; no accepted quantity is repaired by clamping. The same environment alpha governs rigid/support corrections even when no continuum objects exist.

## Accepted time and transaction compatibility

`NMVascularClockGPU {low, high}` holds unsigned 128-bit binary ticks per environment. The compiler selects quantum exponent `ilogb(float(frameTimestep))−23−16`, covering the supported cadence and microtick subdivisions. Each period must convert exactly to positive integer ticks no larger than `INT64_MAX`; the runtime rejects a per-call Float32 timestep that cannot convert exactly before mutation.

The device advances candidate time once per physical microtick, then computes each chamber's phase with full 128-bit modulo by its period ticks. Derived elastance stays fixed during all Newton/FGMRES work. Healthy commit advances accepted time; failure, prepared rejection and restore recover the prior clock and state. Episode reset restores authored initial state and time zero. Overflow rejects the step instead of wrapping accepted time. `microstep.time.z` is root-relative metadata and is not cardiac phase authority.

`RuntimeStateSnapshot::vascularClock` participates in snapshot, archive, replay, prepared restore, rollback and alias protection. Compatibility versions are **Matter ABI 27, package 12, snapshot archive 6 and accepted-state proof manifest 6**. Proof sources include vascular state `0x201a` and accepted vascular clock `0x201b`. Working masks and elastance/trial scratch are regenerated and protected against external aliasing; they add no serialized physical state.

## Pinned source and circuit mapping

The curated source is *Zero dimensional (lumped parameter) modelling of native human cardiovascular dynamics*, attributed to Yubing Shi, Rod Hose and the Physiome Model Repository contributors. Human pins repository revision `a679cdc2e97429fb5280af8132c119758626c1f2` and archive SHA-256 `91d6b586c0caaa0fd59ec21cef873f1348563af0033210fbadfe84921a52782d`. The [curated exposure](https://models.physiomeproject.org/exposure/c49d416ae3a5132882e6ea7479ba50f5) is distributed under [CC BY 3.0](https://models.physiomeproject.org/exposure/c49d416ae3a5132882e6ea7479ba50f5/ModelMain.cellml/license_citation). Human's `tools/shi_hose_reference/source-lock.json` records imported CellML and saved license hashes; original attribution remains in `third_party/physiome/shi_hose_2009`. Numi changes representation and adds numerical integration and evidence export.

The full closed circuit has ten compartments and ten connections:

```text
LA → mitral → LV → aortic → Sas → Sat → Svn → RA
RA → tricuspid → RV → pulmonary valve → Pas → Pat → Pvn → LA
```

LA/LV/RA/RV map to `ModelHeart`; Sas/Sat/Svn to `ModelSys`; Pas/Pat/Pvn to `ModelPul`. Stable `CellML:shi_hose_2009:…` identifiers preserve that mapping. Four algebraic valve flows and six passive vascular flows join the four absolute chamber volumes and six C·P coordinates. The source graph has no species or tissue exchange. Its simplified valves do not reproduce leaflet motion, regurgitation or measured anatomical valve geometry.

## Reproduction commands and evidence boundary

Set `NUMI_HUMAN_SOURCE`, `NUMI_MATTER_SOURCE` and `NUMI_MATTER_BUILD` to the Human checkout, native checkout and native build directory. Run the native commands on physical Apple silicon with Metal, for example through `ssh macmini`.

```sh
cd "$NUMI_HUMAN_SOURCE"
numi human-cardiac --output Build/shi-hose/closed-loop.native.json

cmake -S "$NUMI_MATTER_SOURCE" -B "$NUMI_MATTER_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$NUMI_MATTER_BUILD" --target numi-matter-physiologyc numi-matter-cardiac-check numi-matter-cardiac-transaction-check

"$NUMI_MATTER_BUILD/matter/numi-matter-physiologyc" \
  "$NUMI_HUMAN_SOURCE/Build/shi-hose/closed-loop.native.json" \
  "$NUMI_HUMAN_SOURCE/Build/shi-hose/closed-loop.nmatterpack" \
  --timestep .002 --environments 2

"$NUMI_MATTER_BUILD/matter/numi-matter-cardiac-check" \
  "$NUMI_HUMAN_SOURCE/Build/shi-hose/closed-loop.native.json" \
  --steps 5000 --dt .002 --trace "$NUMI_HUMAN_SOURCE/Build/shi-hose/ten-cycles.csv"

"$NUMI_MATTER_BUILD/matter/numi-matter-cardiac-transaction-check"
```

`cardiac-check` independently compiles, writes and reloads the ordinary package before native execution; its input argument is the authored JSON. Repeat equal-duration runs at refined timesteps and retain complete logs and traces. The independent FP64 reference is generated from the pinned CellML expressions; Human's `tools/shi_hose_reference/README.md` documents generation and source verification. Authoring options are documented in Human's `Docs/SHI_HOSE_SOURCE_AUTHORING_V2.md`.

Record exact code/source/build identities, failed and successful runs, numerical refinement and acceptance results in Human's `Docs/CARDIAC_SOURCE_REPRODUCTION_20260912.md`. This owner document contains no mutable measurement table or completion claim. Source-model numerical reproduction, accepted-state replay, absolute vascular volume calibration and biological validation are separate evidence categories; standing, walking and whole-Human completion remain outside this source capability.
