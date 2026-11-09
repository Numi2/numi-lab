# Retrospective slider position representability

For each accepted step of the four retained slider prefixes, independently
form the full-step constant-mass midpoint equation using the recorded start
and endpoint velocities: q1 = q0 + h * (v0 + v1) / 2. Use exact decimal
arithmetic over the stored binary inputs. Compare the recorded endpoint,
the closest single binary32 endpoint, and a binary32 high/low representation.
Use the same form as the native position bound, 64 times binary32 epsilon
times max(abs(h * midpoint velocity), 1e-6 m). Record the source-trace hashes,
worst absolute defects and number exceeding this independently reconstructed
bound.

This evaluates each retained step independently from the stored start and
endpoint velocities, with the original trajectory held fixed. It is not the
native stage-increment certificate, an updated contact solve, a test of
Metal floating-point instructions, or evidence of an accepted or replayable
two-way transaction.
