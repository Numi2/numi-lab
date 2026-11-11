# Existing-model gravity controls

The existing floating G1 host/CPU parity/lifecycle/offset probe and fixed-root Franka and surgical PSM CPU-parity probes all passed against the isolated gravity-reformulated standalone ABA shader. Their maximum scaled acceleration errors were 1.99037e-6, 1.41278e-6 and 2.33619e-7 respectively; each probe's original next-position and next-velocity bounds also passed. These controls do not test MetalWorld's other shader paths or qualify the crow's contact dynamics and landing, and the isolated host is not production integration.
