# Compensated position and stretch experiment

`metalrobo.crow.rod_split_stretch_reference` exercises a research-only native
Metal kernel on 96 near-rest 3D segments translated by 0, 1 and 16 metres.
It compares compensated high/low endpoint differences, squared length, square
root and rest-length subtraction against a separately computed double
Hookean tension reference. It also requires ordinary one-float arithmetic to
expose a case with significant error. Runtime shader compilation disables
fast math; the compensated operations depend on that compiler contract.

This kernel is a numerical fixture, not the production rod integrator. It does
not preserve low coordinate parts through world integration, bending, damping,
collisions, attachments or rendering. It does not qualify force balance,
performance, flight, landing, a full physical feather or a biological model.
The experiment, full inputs and failed world force controls are archived at
https://github.com/Numi2/numi-gpu-physics-pigeon-sim/tree/agent/crow-neural-balance-evidence-20261115/evidence/crow-feather-rachis-20261116/split-gpu-stretch-20261116
