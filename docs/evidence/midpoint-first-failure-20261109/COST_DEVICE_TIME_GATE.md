# Physical-time correction for the cost comparison

The native benchmark reports accepted duration using its parsed double-precision timestep, while the Matter request and device dispatch receive the timestep as binary32. The physical-time metric must use the latter. This is a reporting correction and does not change the integrator, timestep used by the device, or any accuracy gate.

All six declared backward Euler timesteps represent an accepted 320,000-to-10,000-step 10 ms rollout as 0.009999999974752427 s on the device. The retained 640-step midpoint candidate represents it as 0.010000000474974513 s. Their relative difference is approximately 0.050 parts per million; it is much smaller than the predeclared 10% practical-benefit target, but the corrected rate and threshold decision will still be used.

The post-cost audit requires six complete, already accepted matched rollouts, recomputes each rate from its step count, binary32 device timestep and measured wall time, and reapplies the same median and all-pairs criteria. A synthetic six-run fixture checked the correction and confirmed that a rejected rollout cannot pass. No measurement or speedup follows from this audit setup.
