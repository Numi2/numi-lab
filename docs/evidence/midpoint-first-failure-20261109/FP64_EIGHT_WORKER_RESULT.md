# Ordered eight-worker reference parity

The independent evaluator with eight workers passed its energy-gradient
self-test and reproduced the complete retained n=2 and n=8 trajectories at
0.25 microseconds and n=32 trajectory at 0.5 microseconds byte for byte. Its
128-division allocation and one-force-evaluation capacity probe completed with
2,146,689 nodes, 12,582,912 tetrahedra, 0.0004 m² total reference area and
0.0088 kg total FEM mass. Sampled process resident memory peaked at 3.16 GiB
during that single evaluation. A full RK4 rollout uses additional state and
must still obey the registered memory and free-space limits.

The short capacity timing overlapped the ongoing four-worker reference, while
the earlier four-worker capacity timing did not. They are not a controlled
worker-speed comparison. No native Matter or GPU performance conclusion
follows. The ongoing four-worker 128-division reference remains unchanged;
the eight-worker parity is an independently retained option for a later run.
