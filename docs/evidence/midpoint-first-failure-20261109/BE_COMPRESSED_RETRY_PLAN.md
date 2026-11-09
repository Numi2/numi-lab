# Compressed backward Euler retry

The previous 0.03125 microsecond attempt stopped on its resource guard after
7.31 ms. It did not complete a rollout or replay; its partial trace remains
archived as evidence. Retry the same frozen program, shader, reference,
physical inputs, thresholds and preregistered finest-first doubling ladder.
The only change is lossless streaming compression of the diagnostic streams.
It does not alter the native binary or turn this diagnostic run into a cost
benchmark. Both compressed and uncompressed byte hashes are retained; failed
or interrupted output remains evidence. Stop at the first failed rung.

For this compressed retry the free-space guard is 3.1 GiB, while the 4 GiB
producer RSS and 24-hour guards remain. The previous 3.2 GiB guard protected
the much larger uncompressed trace; the compressed retry checks space every
two seconds and stops 0.1 GiB above the concurrent spatial reference's 3.0 GiB
guard. Its earlier partial trace compressed to 19.1 MiB plus 1.4 MiB log;
this is a storage forecast, not a guaranteed bound. Preserve all streams if
the guard stops the run. Do not start while the guard cannot be met. If a producer completes, require
the original geometry, local mechanical, independently scaled residual,
force-bin, sampled peak, total and cumulative impulse, replay, restoration,
and identity gates before trying the next timestep. Record tested timesteps
without extrapolating to untested values. The later cost comparison must use
complete no-per-step-snapshot runs on the same law and physical fixture.

The existing complete 1 microsecond failed result was reanalyzed through plain
and compressed input with identical output and failure status. A simulated
producer verified full byte and hash preservation of both streams when the
producer fails; the cost-input loader rejects ambiguous plain and compressed
copies. See compressed-diagnostics-validation-001.json.
