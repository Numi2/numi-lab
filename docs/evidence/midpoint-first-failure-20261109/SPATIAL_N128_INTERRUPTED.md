# Interrupted 128-division reference attempt

The first 0.5 microsecond run was stopped after 1,494.7 seconds when its
free-space supervisor observed less than 3 GiB available. The pressure was
caused by a separate task-created checkout that unexpectedly expanded tracked
example data; that incomplete checkout was removed and approximately 4.5 GiB
of free space was restored. The sampled peak resident memory for the reference
process was 4.73 GiB. It returned from termination rather than completing a
trajectory; its empty output, log, execution metadata and reason are retained.
This is an execution interruption, not a spatial or temporal accuracy result.
A separate uniquely named retry uses the original frozen reference binary.
