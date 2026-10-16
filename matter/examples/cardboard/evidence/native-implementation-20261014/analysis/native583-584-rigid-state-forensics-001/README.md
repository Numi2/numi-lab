# Native 583/584 rigid-state failure forensics

CPU-only field decode of retained transaction snapshots. The changed NMRigidStateGPU entry matches first-use projection of the authored BOX pose; external MRBodyStateGPU bytes were not exported, so rollback remains open.

Run python3 decode_rigid_states.py on the recorded MacMini paths to repeat decoding. The manifest binds report, decoder, source, build and native inputs/outputs. No GPU dispatch was performed.
