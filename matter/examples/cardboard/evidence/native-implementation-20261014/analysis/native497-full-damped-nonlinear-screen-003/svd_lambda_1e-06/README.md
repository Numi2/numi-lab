# External damped direction (CPU only)

full total step from zero, dense binary64 SVD Tikhonov lambda=1e-6. Evaluated at alpha 0, 0.25, 0.5, and 1 using strict compensated FP64 material projection/stress, anchored to the captured accepted state and center residual. Candidate components are narrowed to FP32 for the production-shaped velocity/displacement update. No native trial or solver certificate is inferred.
