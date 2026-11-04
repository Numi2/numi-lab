# Aligned FEM contact: positive-gap IPC and consistent distance

This increment corrects numerical contact evaluation. It is not sustained Human
motion qualification or a performance success.

The native141 witness stopped root24 at an immutable 1 nm clearance. The FEM
continuation capped force and set normal curvature to zero before that boundary.
For aligned positive gaps the implementation now evaluates the original IPC
polynomial in dimensionless coordinates, retaining force and curvature.
The unaligned legacy and MPM continuation paths are unchanged. The strict
nonpositive-gap, CCD, material, mass, residual and rollback gates remain active.
This is a numerical contact algorithm revision; it is not a bitwise-only tuning.

A separate exact-Float32/Decimal70 reconstruction found a 5.7486% difference
between the blocking contact's rounded response gap and its paired-coordinate
VT plane gap. Tiny aligned gaps now subtract clearance before Float32 projection
and use the same nearest-feature geometry for response and Armijo energy.
A private existing metric-row metadata component retains the resolved gap for
energy reporting and completed-geometry reuse. No public interface, physical
mass, source point, force owner or new GPU buffer was introduced.

The barrier's singular positive-gap behavior follows the original
[IPC formulation and technical supplement](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf).
The implementation and comparisons here do not establish physiological or
clinical accuracy.

## Native checks and actual limits

All execution used SSH Numi, Apple M6, the same 15.625 us step and unchanged
iteration budgets. The source geometry is connected-conditioning134, already
retained in Numi2/numilab-human at 534c08d94.

- Direct Float64 energy/force/curvature and domain checks: 119 cases PASS,
  including positive paired gaps lost by premature Float32 projection.
- Full native FGMRES contract: 916,997 checks PASS.
- Native139: 24 accepted steps; next contact sweep stops.
- Native145: 25 accepted steps; next step fails nonlinear certification.
- Native148: 26 accepted steps (0.40625 ms); next step fails contact Armijo,
  code6, skin object15, contact2567089. The diagnostic is current
  energy4.498301874e-15, directional derivative-8.908889251e-16, initial alpha1.
  No threshold was relaxed and rejected physiology did not become an accepted
  physical step.

Native139/145 share exact values in all 11 retained CSV prefixes. Native145/148
share exact prefixes in 10/11 CSVs; the remaining differences are three
blood-accounting diagnostic values at row25, differing by 2.22e-10 ml.
Displayed respiratory and cardiovascular outputs otherwise match in that prefix.
These CSVs do not replace complete fine-FEM force/strain endpoint qualification.

Native148 accepted GPU time was217.6165 s, GPU-only factor1.8668e-6x.
The 25-step common prefix took193.2060 s versus192.0013 s for native145.
This single exploratory pair establishes no speedup. Initialization is separate;
supported equilibrium, sustained motion, native presentation, five-minute
physiology/intervention and >=0.1x remain unqualified.

The prior eta0.005 forcing trial was rechecked: its regional force and strain
gates failed. It is not promoted. No larger timestep experiment was repeated.

## Recovery and storage

The two new source overlays, applied after the published097/102/110/130/136
chain, reconstruct39/39 pinned execution files for each trial. Recovery checks
cover source bytes, not unavailable external anatomy assets or old binaries.
Plans retain exact binary, dependency, configuration, asset and source hashes.
Failed141/145/148 traces and their commands remain retained.

The verified144 hardlink consolidation recovered1,285,292,032 allocated bytes
while preserving identical evidence bytes and named paths. Earlier139 recovered
5,151,608,832 bytes separately; do not count either ledger twice.

The next failure is isolated to contact Armijo. The present change does not
disable that check or admit the unsupported full-coupled residual-Armijo mode.

