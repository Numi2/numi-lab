# Robertsson 2023 absolute-energy audit

This is a deterministic CPU source-equation/software check for the single-ply
softwood-paperboard model in Robertsson et al. (2023). It is not experimental
data, a reconstruction of the authors' executable, or a property transfer to
the corrugated kraft/NSSC recipe.

The source reports thickness `0.39 mm`, density `788 kg/m^3`, and a single-ply
softwood construction. Appendix C reports `E1..E6 = [1690, 292, 28.4, 1333,
78.0, 0.365] MPa` and dimensionless `E7=11.5`; Table C2 gives the twelve
initial yield distances and available hardening coefficients, and Table C3
gives damage coefficients for modes 9–12. The implemented reference setup
also uses the paper's `k=3`, `m=0.7`, `nu12=0.46`, `nu21=0.19`, damage cap
`0.96`, and reference projected area `0.0011 mm^2`. These are parameters for
that different paperboard grade. Table C2 contains reported model parameters,
not raw observations; the article does not provide numeric response tables
for the underlying specimen curves.

The official open-access [Wiley article](https://onlinelibrary.wiley.com/doi/full/10.1002/pts.2774)
sets `Psi=(1-alpha) Psi_0` and `A=-dPsi/dalpha=Psi_0` in Eqs. 2 and 7. It
prints the ZD compression bracket with `-2` in Eq. 18 and defines `H=0` for
`I13<1`; Table C1 gives `E6=0.365 MPa`, `E7=11.5`. At the undeformed state
`F=Fp=I`, `kappa=0`, the energy is zero. On the compressed side, however,

```text
lim(I13 -> 1-) E6 [I13^2 + exp(-E7 (I13^2 - 1))/E7 - 2]
  = E6 (1/E7 - 1)
  = -0.3332608695652174 MPa.
```

The article separately states `Psi_0 >= 0` in its Eq. 17 discussion. The
literal equation therefore conflicts with the stated nonnegative-energy
condition on this branch. At `I13=0.99`, the independent Eq. 18 value is
`-0.3323625426152471 MPa`. With `alpha=0.25`, the full free-energy density is
`-249271.912917 Pa` and its damage conjugate at zero damage is
`A=-332362.550556 Pa`, matching the builder's selected absolute energy.

The comparison convention `recentered` adds the constant
`E6(1-1/E7)=0.3332608695652174 MPa` on the compression branch. The compiled
energy difference at `I13=0.99`, `alpha=0.25` is `249945.664380 Pa` (the
constant multiplied by `1-alpha`). This leaves the deformation derivative
inside the branch unchanged but changes `A`; it is not a source-verified
author implementation. The primary article reports commercial-code
simulations. No author implementation source or alternate executable constant
was identified in the retained primary-source material.

Run the CPU-only equation check with:

```sh
cmake --build build-cardboard --target numi-matter-robertsson-fem-coupling-check -j2
./build-cardboard/numi-matter-robertsson-fem-coupling-check --cpu-only
```

The check compiles both explicit conventions into a CPU `CompiledWorld`, then
independently evaluates their attached source energy-expression graphs at the
reference, compression, and tension states. It does not run the Metal energy
kernel and performs zero native FEM timesteps. A pass establishes equation/builder
consistency for those states only; it does not resolve the paper's internal
thermodynamic inconsistency, identify which convention the authors executed,
reproduce the folding experiment, or validate the current liner, medium, or
starch interface.

The published crease/fold geometry and loading are different from the native
one-tetrahedron prescribed-`F13` FEM integration instrument. That instrument
checks accepted state, reaction, rollback, and reset flow for the 27-state
material adapter; it does not execute the published die, contact, crease, or
fold test. Figure 4 remains an un-digitized plot rather than a numeric held-out
curve in this package.

Primary-source transcription and access limitations are in
[`robertsson_2023_eq18_source_extract.md`](robertsson_2023_eq18_source_extract.md).
The extract SHA-256 is recorded in `catalog.json`; the original article PDF
and publisher figure pixels were not retained.
