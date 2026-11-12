# Reached seated-state transfer: invalid contact continuation

The first batch rolled back at control step 0, substep 13, with MR_STEP_DID_NOT_CONVERGE in contact. It produced no accepted state and cannot establish grip or landing. Reconstructed physical body poses match the translated source replay to at worst 2.5e-7 m in this separate kinematics reconstruction. Several selected toe centers from the seated source replay are already below the top plane of the wider course perch; transferring a contact state without solver warm-start and into a fresh solver can create an invalid initial contact problem. The exact cause of nonconvergence is unproven; this does not establish a general solver defect.

The source-state probe uses high uncalibrated friction, lacks the original controller and warm-start history, and is not an uninterrupted autonomous landing. A separately retained pre-impact continuation is prepared using frames whose physical toe/foot bounding boxes are above the perch top.
