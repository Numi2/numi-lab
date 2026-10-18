# Blood MPM constitutive proxy: parameter provenance and scope

This material is a bounded high-shear Newtonian constitutive proxy for the existing MPM material-bytecode owner. It does not create or transport blood by itself. The current simulation's CVSim remains the sole blood-volume and species stock; no receiver activation or vascular-to-MPM transfer is implemented or qualified here.

## Reference values

- Density is 1050.6 kg/m^3, with authored bounds 1047.5–1053.7 kg/m^3. This converts the 37 °C whole-blood relative density 1.0506 (95% interval 1.0475–1.0537) reported from gravimetric measurements of 25 healthy volunteers by Trudnowski and Rico (1974), relative to water at 4 °C (using 1000 kg/m^3 for the reference-water density). It is a population reference, not a subject-specific estimate: [PubMed record](https://pubmed.ncbi.nlm.nih.gov/4826961/), [DOI](https://doi.org/10.1093/clinchem/20.5.615).
- Dynamic viscosity is 0.004 Pa s, with an authored sensitivity range 0.003–0.005 Pa s. A 2018 whole-blood measurement at 37 °C reports control viscosity near 4 mPa s at 1000 s^-1 and a much larger value near 70 mPa s at 0.1 s^-1. Thus the Newtonian value is limited to high shear; low-shear aggregation, yield behavior, cell-scale effects, and clotting are omitted: [primary study](https://www.nature.com/articles/s41598-018-25317-8).
- The volumetric penalty coefficient 2.5 MPa, bounded 1–5 MPa, is a numerical weak-compressibility proxy, not a measured blood bulk modulus. Direct experiments on canine arterial and human venous blood found compressibility near distilled water, so this penalty must not be interpreted as the physical compressibility of blood: [Sacks and Tickner (1968)](https://doi.org/10.3233/BIR-1968-5403).

## Implemented equations and checks

The authored energy is W = K/2 (J-1)^2. The authored dissipation is the Rayleigh potential phi = mu J Ddev:Ddev with D = sym(Fdot F^-1). Existing constitutive bytecode differentiation supplies Pvisc = 2 mu J Ddev F^-T; the power identity tested is Pvisc:Fdot = 2 phi.

For hydrostatic pressure p = K(1-J), the penalty gives rho/rho0 = 1/J = K/(K-p). At p=30 kPa, the tested density rises are about 3.09%, 1.21%, and 0.604% for K=1, 2.5, and 5 MPa respectively. This is the authored penalty model's sensitivity, not evidence that these values reproduce real blood density under pressure.

The compiler's CFL estimate uses h=5 mm, CFL=0.35, the lower density bound, the upper K bound, and a 1/480 s frame. It selects exponent 7 (128 subdivisions) for the rate solver. This is a compile-time bound only; no native timestep stability or performance qualification is claimed.

The CPU fixture evaluates compiled bytecode for simple shear, arbitrary deformation/rate, rigid rotation, static shear, compression sign, viscous power, pressure/density sensitivity, and the CFL exponent. No GPU dispatch, active MPM liquid receiver, donor/receiver transfer, contact, fluid density evolution, conservation across phases, or accepted/rejected-step rollback is covered.
