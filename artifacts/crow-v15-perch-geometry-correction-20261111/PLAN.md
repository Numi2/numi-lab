# Correct runtime perch geometry

The GPU host changes the perch half-width from 0.10 m to 0.025 m in build/crow-v15-hip-roll-journey-integration-20261111/c_api.cpp. Previous SAT and CPU collision-oracle checks incorrectly used the original 0.10 m scene geometry. Preserve those outputs as contradicted evidence; repeat clearance and CPU collision checks using the actual runtime geometry, then determine whether any deep toe-perch overlap remains. Compare selected contact samples and audit all prior release claims.
