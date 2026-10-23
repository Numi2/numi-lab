# LAW112 implicit initial-matrix proposal

Status: review proposal only. It is not applied to the solver source, compiled, or solver-qualified. No physics cards, tolerances, rates, or element formulation are changed by the patch. The patch changes only the initial elastic matrix assembled in `MMATS` for internal material type 112.

## Source binding

- Pinned OpenCourant source commit: `cb760c206e346eff12161a2268b134ef9d48a4ef`.
- `engine/source/elements/solid/solide8z/mmats.F`, SHA-256 `9d90a99513b01b694c2d9b02ad324efd549c8c488a1bbfbba7f16e077e274243`. Existing `IORTH>0` fallback is at lines 146-158 and discards orthotropy by setting `IORTH=0`, then takes generic `PM(21), PM(32)` values.
- `engine/source/materials/mat/mat112/mat112_xia_newton.F`, SHA-256 `c437b2a6a75c71bd4ff9f95aebdf0190e76a1df7c9a53183c904bd3184b0c1b5`. Native LAW112 uses UPARAM(1:3)=E1,E2,E3; UPARAM(4:5)=nu12,nu21; UPARAM(6:9)=A11,A12,A21,A22; UPARAM(10:12)=G12,G23,G31 (lines 125-136). Its zero-plastic trial stress is `sigma_x=A11*eps_x+A12*eps_y`, `sigma_y=A21*eps_x+A22*eps_y`, `sigma_z=E3*eps_z` on the zero/tensile side and shear `G12/G23/G31` (lines 308-321).
- `starter/source/materials/mat/mat112/hm_read_mat112.F`, SHA-256 `dd039d31ab25e87cd383cb947d8708de5f94e73afe4cadd64c1d6faa9656aba2`. It derives `nu12=nu21*E1/E2` and A11:A22 using reciprocal plane-stress equations (lines 324-345).
- `engine/source/elements/solid/solide8z/mstiforth.F`, SHA-256 `edcbc905395aac77de788260f8eb7440bdf938aa7491a06ccae1a1b1b3c7fb22`. The assembler reads the upper triangle of `CC`, mirrors it, and uses `G3` for the three shear moduli (lines 54-79, 114-139).
- Numerical inputs are bound to `matter/examples/cardboard/opencourant/paper-parameters.json` SHA-256 `46d33c7e6e4c796ad3285e269df03b1a0a6442dc70e6cb86de8a38ab5568527b` and the role-specific liner transfer helper `matter/tools/cardboard_opencourant_material.py` SHA-256 `aad441cc5e9e1401873d1d8e4a266251f8d2434153e5f92eb1137c3bf9586ce2`.
- Source URLs: [mmats.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/solid/solide8z/mmats.F), [mat112_xia_newton.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/materials/mat/mat112/mat112_xia_newton.F), [hm_read_mat112.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/starter/source/materials/mat/mat112/hm_read_mat112.F), [mstiforth.F](https://github.com/OpenCourant/OpenCourant/blob/cb760c206e346eff12161a2268b134ef9d48a4ef/engine/source/elements/solid/solide8z/mstiforth.F).

## Proposal and analytical check

The new branch maps `(CC11, CC22, CC33, CC12; G3)` to `(A11, A22, E3, (A12+A21)/2; G12,G23,G31)` and calls the existing `MSTIFORTH`. The half-sum is the symmetric matrix entry. The pinned LAW112 reader derives `nu12=nu21*E1/E2` and calculates A11:A22 with the reciprocal orthotropic plane-stress equations; this makes A12=A21 for reciprocal card inputs. The cardboard medium and liner cards satisfy that relation exactly in FP64. At zero out-of-plane strain, the model's compression-side tangent is also E3 for these cards because `E3C=E3` and `CC=1`.

FP64 calculations against the current source-bound medium and liner records:

| Role | C11 MPa | C22 MPa | C33 MPa | C12 MPa | det(in-plane block) MPa² | MD/CD free-side slopes MPa | min shear MPa |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medium | 4524.845992027823 | 1631.049136661192 | 17.9 | 293.588844599014 | 7,294,051.739149 | 4472 / 1612 | 198.72 |
| Liner | 4843.623442562828 | 2118.007486507055 | 19.1 | 381.241347571270 | 10,113,485.748071 | 4775 / 2088 | 137.28 |

Both in-plane determinants, all three normal diagonals, and all three shear moduli are positive, so the assembled reciprocal orthotropic matrix is SPD. The MD and CD uniaxial free-transverse Schur slopes recover E1 and E2 exactly; C13=C23=0 leaves the through-thickness normal response uncoupled. These checks validate the mapping arithmetic only.

## Limits / required follow-up

- This predicts a better initial operator; it does not establish that the implicit run converges, that Quasi-Newton history is repaired, or that the patched source is valid for every LAW112 model. The existing LAW112 response still evolves nonlinear plasticity and out-of-plane compression as before; this proposal does not add its consistent elastoplastic tangent.
- `MMATS` must be reached with `IORTH>0` and `MTN==112` for the target Type6/Law112 deck. This source-flow assumption still needs confirmation from an instrumented native build.
- Reciprocity is required for symmetric `MSTIFORTH` to match both native in-plane stress cross terms. The source reader derives reciprocal nu values; hand-built UPARAM records with inconsistent A12/A21 would be symmetrized and would not exactly match the nonsymmetric constitutive update.
- The analytical check is not compile or runtime evidence. A later isolated source build should inspect initialized UPARAM values and show the first assembled MMATS matrix against this oracle before a solver test; preserve the existing failed nonlinear histories.
