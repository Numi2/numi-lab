# Spatial rebound 003 superseded before launch

The 003 rebound and larger-mesh coordinator were stopped while waiting for the separate control study; neither launched a reference. This was done after finding that the inherited n64 analysis checked finite trajectory samples but did not independently validate every force-bin value, mass and clearance field before using the result. A malformed non-finite bin must not be eligible for a refinement claim. The 003 waiting receipt and logs are retained; their `waiting` status reflects termination rather than a completed run.

The queued 004 rebound uses the unchanged physical parameters, native evaluator, timesteps and refinement gates. Before evaluating a mesh result, its analysis also invokes the existing stricter reference validator for each new n64 output and both retained n128 inputs. That validator checks the full weight distribution, total area, mass, clearance, time samples, finite nonnegative force bins, sampled peak and impulse consistency. It accepted the four historical n64/n128 files; a deliberate NaN force-bin corruption was rejected. The 004 supervisor binds the validator and all transitive analysis files by hash and checks them before and after the native reference runs.

No new trajectory or spatial qualification follows from these instrument checks. The currently running control was not interrupted.
