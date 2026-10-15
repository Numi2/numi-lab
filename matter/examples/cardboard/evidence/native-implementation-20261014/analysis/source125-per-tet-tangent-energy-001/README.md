# Source125 per-tetrahedron tangent-energy audit

This is a bounded CPU-only decomposition of the selected environment-0, root-1/Newton-48 production tangent capture from run 419. It reads all 1,920 captured row-major 9x9 FP32 tangent matrices, promotes them to FP64 for contractions and eigenvalue diagnostics, and checks the resulting directional actions against the captured production tangent, an independent FP64 material-direction replay, and the retained native assembled element action. It does not run the solver, update source, or launch Metal work.

The main result is that the captured ray is negative for the assembled operator (A=M-dt^2K): the tangent contraction gives (-0.01581514044) J for the stiffness part, and (-0.00084223598) J after adding its mass work. The retained independent assembled report gives (-0.01581523578) J and (-0.00084233132) J, respectively. The per-tet product-rule split attributes the negative stiffness work to the initial-stress/geometric term: (-0.1426308191) J across the mesh, offset by (+0.1268156786) J from the remaining algorithmic material tangent. The groups sum to (-0.0232711893) J for the liner, (+0.0064463473) J for the medium, and (+0.0010097016) J for glue. Independent FP64 directional actions give (-0.0232713023), (+0.0064463634), and (+0.0010097020) J for those groups.

The split follows the source law's Green-strain energy. For (P=FS), where (S=F^{-1}P) is the recovered second-Piola stress,

\[
D_FP[H] = H S + F\,D_FS[H],\qquad C_{geo}[iJ,kL]=\delta_{ik}S[LJ],\qquad C_{material}=C-C_{geo}.
\]

The first term is the initial-stress/geometric contribution. The second retains the constitutive response, including the 15-state implicit projection; it is not claimed to be an independently identified elastic potential. The solver's internal element force is (-V P Dm^{-T}), so (v^TKv=-V(H:C:H)) and the signed contribution to (A=M-dt^2K) is (dt^2V(H:C:H)).

The strongest negative owner is liner tet 1440: total (-0.02231424) J, split into (-0.02323785) J geometric and (+0.00092361) J algorithmic material. Its recovered second-Piola principal stresses are approximately ([-0.380524,-0.017361,+0.002986]) MPa, and its captured accumulated multiplier increases by (1.3793\times10^{-5}). Eight other liner tets are also negative; three glue tets contribute only about (4.66\times10^{-8}) J combined. This identifies a finite-state compressive prestress term along this ray. It does not establish a sign bug, a physical energy violation, or a full coupon result.

The two Haj-Ali-derived 15-state assets are source-specific transfer models. Their authored Green-strain normal stiffness blocks are positive definite (liner eigenvalues 19.10, 2065.69, 4895.95 MPa; medium 17.90, 1601.57, 4554.33 MPa), and their shear coefficients are positive. The assets also use positive MD/CD tensile-fit hardening values but explicitly treat compression and shear hardening as zero assumptions. These small-strain coefficients do not make the finite-state tangent globally positive. On this captured ray, the aggregate material part is positive for liner, medium, and glue; the negative total arises from the current compressive initial-stress term.

The independent source-reference projection evidence is separately bound, not rerun here. The FP64 bytecode interpreter in `cardboard_bilinear_material_oracle.hpp` evaluates the authored residual/Jacobian, solves the same 15-state system, and forms the implicit directional tangent. The rowwise result-005 gate reports all 1,728 liner/medium rows passing at least three contiguous predeclared epsilon-grid points with endpoint projection and observed branch checks plus local relative error at most (10^{-6}). That is pointwise directional evidence; it is not an independent full 9x9 positive-definiteness proof, a native FP32 finite-difference pass, or physical calibration.

Major symmetry is reported rather than assumed. For the full (C=dP/dF), the maximum relative major-asymmetry is (4.47\times10^{-4}) in liner and (5.39\times10^{-3}) in medium; the medium p99 is (1.37\times10^{-3}). The glue matrices are symmetric in this capture. The recovered (S) is symmetric to relative (8.02\times10^{-8}) or better, and the constructed (C_geo) asymmetry is at most (8.02\times10^{-8}). After subtracting (C_{geo}), some symmetric-part minimum eigenvalues of (C_{material}) are slightly negative, but the worst ratios to each matrix's largest absolute eigenvalue are only (2.48\times10^{-8}) (liner), (2.61\times10^{-7}) (medium), and (1.29\times10^{-8}) (glue). These small residual negatives and the measured asymmetry are retained as limitations, not silently rounded away or called proof of monotonicity.

The material-frame rotations are orthogonal (maximum (\|Q^TQ-I\|_F=1.58\times10^{-16})). The captured coordinates satisfy (F_{local}\approx F_{world}Q), (H_{local}\approx H_{world}Q), and (P_{local}\approx P_{world}Q), each within (9.84\times10^{-8}) relative. With the frame fixed for this tangent, this is a reference-coordinate change, not another work contribution.

## Source and input references

- The exact run capture is `solver/two-stage-first-failure-rich-fd-419/operator-capture`, environment 0, Newton 48. The capture header says the tangent cache is a read-only snapshot of the production per-tet (dP_{local}/dF_{local}), with the rate coupling (\delta\dot F=\delta F/dt); the cache was valid for all 1,920 tets.
- The independent operator-FD adapter is `analysis/operator-fd-e2-adapter-001/coupon419-003/environment-0.npz`. Its README describes reconstruction of independent 15-state paper and finite-glue tangent actions and separate mass-minus-stiffness assembly.
- The independent FP64 rowwise derivative report is `analysis/source125-fp64-smooth-direction-sweep-006/result-004/report.json`; result-005 adds the contiguous interval/branch gate. The report says no universal noise floor or native derivative qualification is claimed.
- The material source files are in the frozen source125 snapshot. In `hajali2009_liner_hill_bilinear_fit.nmatter` and the corresponding medium asset: energy is at lines 99–112; implicit normalized second-Piola stress at 136–167; associated Hill flow at 169–240; tensile MD/CD hardening accumulations and yield complementarity at 242–283. Asset comments at lines 1–37 explicitly distinguish tensile-fit values from compression/shear assumptions and transfer choices.
- `matter/src/metal/fem.metalinc:260–276` defines the negative internal-force sign; lines 3060–3078 form the deformation and rate directions; lines 3125–3140 call the algorithmic tangent. `matter/src/metal/fgmres.metalinc:5116–5124` assembles (M-dt^2K).
- `matter/tools/cardboard_bilinear_material_oracle.hpp:205–307,355–400,402–421` contains the independent bytecode evaluation, residual, FP64 projection, and implicit tangent routines.

## Reproduction

```sh
/Users/home/.pyenv/versions/3.10.12/bin/python3 /Users/home/cardboard-implementation-20261014/analysis/source125-per-tet-tangent-energy-001/audit.py
```

The command writes `report.json`, `per-tet.csv`, and stdout capture in this package. It fails closed on missing/duplicate tetrahedron records, invalid tangent-cache entries, topology/index mismatches, nonfinite values, or failed source-volume/frame checks. All result JSON is written with nonfinite values forbidden.

`manifest.json` binds this script and output files to the native capture, run receipt, compiled source manifest, adapter, independent reports, material files, and exact source files. No source or production files were edited.
