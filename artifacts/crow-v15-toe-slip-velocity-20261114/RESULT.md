# Toe contact velocity observation: inconclusive timing cross-check

The observation-only run reproduced all logged control states, commands, poses and endpoint contacts from the prior friction observer in lane 0 across 450 intervals. The two lanes were identical in those records. All 252 requested q/v samples were present; native forward-kinematic foot and right-toe positions at the last substep matched the logged endpoint poses within 5.7e-10 m. There were no observer errors.

In lane 0, during intervals [290,300), the post-solve point-velocity calculation for 172 qualifying right foretoe contact rows had median tangential speed 0.0187 m/s; 169/172 exceeded 0.01 m/s while their tangent impulse was at least 95% of the dynamic cone. The rows are highly correlated, not independent experiments. Similar estimates occur around [300,305). This is consistent with slip in the constructed model.

However, the required independent check did not validate the velocity interpretation: across sampled adjacent substeps, finite-difference body COM velocities differed from the computed post-solve velocities by median 0.0352 m/s and 95th percentile 0.0627 m/s. These discrepancies are as large as or larger than the estimated slip. The exact contact versus state epoch and the reason for the position/velocity mismatch remain unresolved; hybrid CCD/event splitting is enabled but its contribution has not been measured. Therefore this trial does not prove actual slip, and it does not justify changing the friction coefficient. The separately observed toe effort saturation also does not establish causality. See `analysis.json`, `analyze.py`, the native kinematics source and raw logs for details.

No grip, landing, flight, or mission qualification is claimed.
