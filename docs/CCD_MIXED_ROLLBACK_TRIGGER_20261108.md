# Mixed-island rollback trigger preparation

The first resident test initialized the zero-gap mixed rod/rigid scene successfully with one cached manifold. It submitted an effort of -1000 on the floating root to provoke a failed contact step. Both clean main and the candidate accepted, and the row capture showed pre-solve ground normal velocity 0 m/s and final body normal velocity 0 m/s. Cache and state changed during accepted dynamics, so those differences provide no rejection-restoration evidence.

The fixture used the effort API incorrectly. `MetalWorldStepConfig::actuationMode` documents that in effort mode floating-root and unactuated commands resolve to zero; external loads use body wrenches. The observed lack of root acceleration is consistent with this contract, not a demonstrated dynamics defect.

The first replacement attempted to drive a kinematic ground. Both runs exited 3 during world composition with `component scene reset is invalid`: the scene state had been made kinematic but the corresponding body descriptor remained static. `validSceneState` checks that these motion types match. A second version changes the body descriptor too, while retaining zero inverse mass and unit orientation. This version is not qualified until it actually runs and a mixed normal failure occurs.

Raw outputs, source and binary identities for both failed attempts are retained in the mixed containment evidence directory. The zero-gap scene, initial cache count, attempted commands, composition error and all status values are part of the record.
