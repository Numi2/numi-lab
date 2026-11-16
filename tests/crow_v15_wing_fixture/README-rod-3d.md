# Coupled 3D rod modal research regression

The CTest `metalrobo.crow.rod_coupled_3d_modal_reference` tests 80 fixed 0.125 ms GPU steps of a linearized 3D nine-node centerline with three moving point masses, whose attached coordinates are eliminated exactly. It checks sampled position against the independent FP64 modal reference, energy, 3D momentum and force residual. It is a standalone numerical test, not a nonlinear feather, articulated-body, contact, aerodynamic or MetalWorld implementation. Its native-energy Hessian, model limits, inputs, source, original run and generator are archived in [evidence commit 449b0f8](https://github.com/Numi2/numi-gpu-physics-pigeon-sim/commit/449b0f8).

The 8 kHz output rate undersamples much of the modeled spectrum; exact sampled states do not resolve intermediate feather motion, collisions, visual flutter or rapidly changing loads. No biological or mission qualification follows.
