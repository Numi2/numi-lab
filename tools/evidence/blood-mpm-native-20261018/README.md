# Native high-shear blood MPM probe

This source-pinned increment exercises the existing Matter MPM solver with the current blood high-shear proxy material. It admits eight active 5 mm-cubed particles (total reference volume 1 mL) and one finite, positive-mass dormant reserve slot. The native probe runs eight accepted steps and a separate GPU-injected rejection. The accepted path retained 1.0506 g active mass and finite positive J; the rejection restored particle and material state.

The probe does not debit or receive circulation blood stock. It does not validate a free surface, vessel coupling, whole-blood rheology, low-shear aggregation, clotting, or blood-loss transfer. Kinetic energy fell from 0.00328312 J to 0.000466047 J; the rough ideal high-shear viscous-work estimate over the 1 ms accepted horizon is about 4e-6 J. The much larger observed decrease remains unexplained (including possible MPM transfer/grid damping), so this is not a dissipation or constitutive-accuracy qualification.

The material's 4 mPa s high-shear parameter is an approximation associated with the published high-shear regime described in [Effects of Aggregation on Blood Sedimentation and Conductivity](https://pmc.ncbi.nlm.nih.gov/articles/PMC4457804/). It should not be applied to stagnant pooled blood.

The evidence JSON records source, object, link-time and runtime library, shader, material, exclusive wrapper, and native output hashes. The preflight failure record retains a wrapper-pin error that stopped before dispatch. The native run used the designated exclusive job wrapper.
