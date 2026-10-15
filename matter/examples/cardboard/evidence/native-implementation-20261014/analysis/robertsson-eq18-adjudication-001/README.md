# Robertsson Eq. 18 absolute-energy adjudication 001

This external CPU-only package adjudicates the printed compression energy convention against the official paper and the checked-in source implementation. It is not a code patch, native run, experimental observation, or transfer of this grade to kraft/NSSC.

## Verdict

The official Wiley full text prints `-2` in Eq. 18 and defines `H=1` when `I13 >= 1`. Thus `paper_absolute` is the source-text reading. For the article's single-ply softwood grade (`E6=0.365 MPa`, `E7=11.5`), the virgin path `Fe=F=diag(1,1,lambda_z)`, `Fp=I`, fixed damage gives compressed-side energy limit `E6*(1/E7-1) = -0.333260869565217391... MPa`; at the switch H selects the zero-valued tension branch. This contradicts the article's nearby `Psi0 >= 0` premise used in the damage-dissipation argument. The inconsistency is explicit in the source equation; neither an author implementation nor an erratum was available to resolve what the executable did.

`recentered` adds `E6*(1-1/E7)` on the compression branch. It restores a zero energy at the switch and nonnegative energy along this path. This is an engineering/thermodynamic correction and historical comparison, not a source-verified author implementation. The constant leaves within-branch mechanical stress/tangent unchanged but changes `A=Psi0`. Since `A` drives damage, this is not an innocuous energy-zero shift for the damage model.

The admissible benchmark is restricted to the published Eq. 18 and Eq. 20 / Appendix A1 responses for virgin diagonal F13 loading, fixed damage, `Fp=I`, and points away from the branch switch. At the switch, only the exact source-selected H=1 value is reported. These are analytic equation checks, not 27-state damage calibration or author-code reproduction.

## Primary-source scope

- [Official Wiley full text](https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774), DOI `10.1002/pts.2774`; Eq. 18, Eqs. 2, 6–7, 19–20, Appendix A1, Table C1, and Eq. 17 discussion.
- Grade: the article's 0.39 mm, 788 kg/m^3 single-ply softwood paperboard, not corrugated kraft/NSSC.
- The paper reports commercial-code simulations. This package did not acquire an original PDF, author executable/source, or published correction. `robertsson_2023_eq18_source_extract.md` is the retained transcription and is hash-bound in `manifest.json`; publisher HTML is linked but not retained as bytes.

## Reproduce

```sh
python3 -m unittest -v test_benchmark.py
python3 benchmark.py --output report.json
```

Only Python `decimal` at 70-digit precision is used. No GPU, native solver, CMake build, or external library is involved. The tests check branch limits, fixed-branch energy differentiation against Eq. A1/Eq. 20, tangent differentiation, and the damage-conjugate difference.
