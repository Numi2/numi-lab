# Concurrency amendment for independent n=64 traces

The parallel evaluator passed the preregistered parity checks. To finish the
reference sooner, start its 0.25 us run while the existing single-worker
0.5 us run is still active instead of waiting for the latter to finish.
Maximum use is five low-priority CPU workers across two independent processes;
they do not share model state or output files. At launch the host has more than
5 GiB free disk and the existing reference uses about 254 MiB resident memory;
monitor resource use. No GPU timing benchmark may overlap. CPU wall durations
under contention are execution metadata only, not a performance comparison.

The previously recorded marker still prevents the single-worker supervisor
from launching its duplicate 0.25 us run. Both distinct numerical traces must
complete and pass the unchanged temporal gate before any spatial result is
interpreted. Retain each process outcome and all partial results if one fails.
