# Head rolls almost rigidly with the torso

During the 4–6 s development walk, the reconstructed head and torso world roll had correlation above 0.9999 in these closely related simulated lanes; torso roll spanned approximately 0.041 rad while relative Euler roll spanned about 0.00049 rad. The model's existing neck degrees of freedom are yaw and pitch, with no direct neck roll coordinate, so this mechanical limitation is consistent with the observed coupling. Head pitch relative to torso varied around 0.022 rad, and relative yaw around 0.008 rad; see analysis.json for each lane.

These are computed physical-body orientations, not eye gaze, whole-neck anatomy or real-crow motion. A missing roll degree of freedom constrains head stabilization, but the data here do not establish an American-crow range, head-bob cycle or a particular corrective controller. Altering topology would change the body and policy fingerprints and require a separate validation; no cosmetic motion was added.
