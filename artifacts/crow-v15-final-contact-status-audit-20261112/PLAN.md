# Final task contact-status audit

The scene-index correction still yielded only zero sensor readings. Passively copy the same final contact-status buffer that the task-completion kernel reads after the native command buffer is encoded, and report active contacts, required constraints, status, body stride and scene stride alongside the already-recorded per-substep contacts and sensor value. Compare all 300 states and all contact substeps exactly against the previous corrected-sensor run before interpreting. This is an observer of existing buffers, with no solver or controller changes.
