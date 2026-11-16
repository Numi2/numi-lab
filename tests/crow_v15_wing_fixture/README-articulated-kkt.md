# Experimental articulated bird and rod Metal reference

`metalrobo.crow.rod_v15_articulated_reference` checks four one-step linearized controls on Metal using an archived 46-DOF v15 mass matrix and hand-point Jacobian, a synthetic anchor and a 27-coordinate rod Hessian. It compares a dense FP32 KKT solve to FP64 references, with reversed rod node orders. Details and limitations are in [evidence commit 17c241d](https://github.com/Numi2/numi-gpu-physics-pigeon-sim/commit/17c241d).

It is a standalone numerical correctness experiment, with a frozen pose and no physical contacts, nonlinear rod, aero, flight, landing or balance. It neither tests nor fixes the live MetalWorld solver. The v15 inertia and feather surrogate are engineering priors.
