# Complete respiratory-cycle source-state capture regression

Native build 016 ran the same integrated 907 scene and 20-second configuration as run 909 on the SSH M4 Pro Mac mini. The additional source capture records all 86 skin-binding body poses, the accepted diaphragm/rib swept volumes and velocities, and the exact uploaded existing skin mapping records.

All 10,000 coupled physiology, joint-integration and momentum rows and all 320,000 support-impulse rows are byte-for-byte identical to 909. At all eight retained poses, the captured geometry, body and respiratory state hashes, accepted root fingerprint and transaction fingerprint are unchanged. This confirms that the capture increment does not alter this run's simulation state.

Wall time was 201.262173625 seconds for 20 simulated seconds (0.09937 times real time, including launch). Performance tuning is stopped at the user's request. This is a capture regression, not anatomical clearance or a five-minute endurance qualification. The moving skin and pleural defects recorded with run 909 remain open and must be corrected before final acceptance.

The full native recording, geometry captures and traces remain at `/Users/n/numi-human-resting-evidence-20261005/native-source-state-cycle-914`. The retained verification script compares both run directories and hashes the source-map files referenced by every capture.
